# SPDX-License-Identifier: Apache-2.0
"""End-to-end scheduler-flow tests on CPU (fake core, no vllm/torch).

Covers: hit miss->store->commit->hit; async load protocol (pin,
update_state_after_alloc, batched event, recving ack); lazy vs eager
staging; end-of-session snapshot; preemption cleanup; worker-count
aggregation; deferred-free release.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

from fissionkv.config import FissionKVConfig  # noqa: E402
from fissionkv.scheduler import FissionKVScheduler  # noqa: E402
from fissionkv.wire import FissionKVWorkerMetadata  # noqa: E402


# ------------------------------------------------------------- fake core
@dataclass
class FakeSpec:
    block_size: int = 4


class FullAttentionSpec(FakeSpec):  # name matters: scheduler probes type name
    pass


@dataclass
class FakeGroup:
    kv_cache_spec: Any
    layer_names: list = field(default_factory=lambda: ["layer.0"])


@dataclass
class FakeKVCacheConfig:
    kv_cache_groups: list


@dataclass
class FakeParallel:
    world_size: int = 1
    decode_context_parallel_size: int = 1


@dataclass
class FakeTransfer:
    kv_connector_extra_config: dict


@dataclass
class FakeVllmConfig:
    kv_transfer_config: Any
    parallel_config: Any = field(default_factory=FakeParallel)


@dataclass
class FakeRequest:
    request_id: str
    tokens: int
    hashes: list
    num_computed_tokens: int = 0
    skip_reading_prefix_cache: bool = False

    @property
    def num_tokens(self):
        return self.tokens

    @property
    def block_hashes(self):
        return self.hashes


@dataclass
class FakeNewReq:
    req_id: str
    block_ids: tuple


@dataclass
class FakeCachedReqs:
    req_ids: list = field(default_factory=list)
    resumed_req_ids: set = field(default_factory=set)
    new_block_ids: list = field(default_factory=list)


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list = field(default_factory=list)
    scheduled_cached_reqs: FakeCachedReqs = field(
        default_factory=FakeCachedReqs)
    num_scheduled_tokens: dict = field(default_factory=dict)
    preempted_req_ids: set | None = None


@dataclass
class FakeBlockPool:
    """Allocates fake GPU block ids like the core block pool."""
    next_id: int = 1

    def alloc(self, n):
        ids = list(range(self.next_id, self.next_id + n))
        self.next_id += n
        return ids


@dataclass
class FakeBlocks:
    per_group: list

    def get_block_ids(self):
        return tuple(self.per_group)


@dataclass
class FakeConnectorOutput:
    finished_recving: set | None = None
    kv_connector_worker_meta: Any = None


def make_sched(extra=None, n_blocks=8):
    cfg = {"cpu_bytes_to_use": 1 << 30, "lazy_store": False}
    cfg.update(extra or {})
    vllm = FakeVllmConfig(kv_transfer_config=FakeTransfer(cfg))
    kvcc = FakeKVCacheConfig(kv_cache_groups=[FakeGroup(FullAttentionSpec())])
    sched = FissionKVScheduler(vllm, kvcc)
    sched.set_host_capacity(n_blocks)
    return sched


def req(rid, n_tokens, seed=b"\x00"):
    """hash_i = seed+prefix bytes (deterministic prefix chain)."""
    hashes = [seed + bytes([i]) for i in range((n_tokens + 3) // 4)]
    return FakeRequest(rid, n_tokens, hashes)


def step_store(sched, r, gpu, n_new_blocks):
    """simulate: request scheduled, GPU gets n_new_blocks, meta built."""
    ids = gpu.alloc(n_new_blocks)
    blocks = FakeBlocks([ids])
    r.num_computed_tokens += 4 * n_new_blocks
    out = FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq(r.request_id, (ids,))],
        num_scheduled_tokens={r.request_id: 4 * n_new_blocks})
    meta = sched.build_connector_metadata(out)
    return meta, blocks


def ack_stores(sched, meta, world=1):
    for _ in range(world):
        wm = FissionKVWorkerMetadata(
            completed_store_events={meta.store_event: 1})
        co = FakeConnectorOutput(kv_connector_worker_meta=wm)
        sched.update_connector_output(co)


# ------------------------------------------------------------------ tests
def test_store_then_hit_second_session():
    s = make_sched()
    gpu = FakeBlockPool()
    r1 = req("r1", 16)                     # 4 blocks
    s.update_state_after_alloc(r1, FakeBlocks([gpu.alloc(4)]), 0)
    r1.num_computed_tokens = 16
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r1", ([1, 2, 3, 4],))],
        num_scheduled_tokens={"r1": 16}))
    assert meta.store_event >= 0
    assert meta.store_gpu_blocks == [1, 2, 3, 4]
    ack_stores(s, meta)
    assert s.index.ready == 4

    r2 = req("r2", 16)
    n, is_async = s.get_num_new_matched_tokens(r2, 0)
    assert n == 16 - 4                     # last-token cap: (16-1)//4=3
    assert is_async
    hit = s._pending_hits["r2"]
    assert len(hit.slots) == 3
    s.update_state_after_alloc(r2, FakeBlocks([gpu.alloc(4)]), 12)
    load = s._reqs_to_load["r2"]
    assert load.slots == hit.slots and len(load.gpu_ids) == 3

    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r2", ([5, 6, 7, 8],))],
        num_scheduled_tokens={"r2": 4}))
    assert meta.load_event >= 0
    assert meta.load_gpu_blocks == load.gpu_ids
    s.update_connector_output(FakeConnectorOutput(finished_recving={"r2"}))
    assert "r2" not in s._reqs_to_load
    assert all(s.index.get(h) is not None for h in r2.hashes[:3])


def test_eager_boundary_and_resumed_shadow():
    s = make_sched()
    r = req("r3", 20)
    s.update_state_after_alloc(r, FakeBlocks([[]]), 0)
    # step 1: 2 blocks scheduled, computed crosses one full boundary
    r.num_computed_tokens = 8
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r3", ([10, 11],))],
        num_scheduled_tokens={"r3": 8}))
    # pre-step num_computed is 8 here? we set 8 => done=2 at build (facts:
    # core passes PRE-step value; we emulate build after pre-step state 4):
    assert meta.store_event >= 0
    assert len(meta.store_gpu_blocks) == 2
    # resume with a shorter window resets shadow + cursor
    s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCachedReqs(
            req_ids=["r3"], resumed_req_ids={"r3"},
            new_block_ids=[([10],)])))
    assert s._shadow["r3"] == [10]
    # cursor stays: blocks 0-1 already staged; resume does not re-queue
    s.update_connector_output(FakeConnectorOutput(
        kv_connector_worker_meta=FissionKVWorkerMetadata(
            completed_store_events={meta.store_event: 1})))


def test_lazy_stays_cold_until_pressure():
    s = make_sched({"lazy_store": True, "high_water": 0.9})
    gpu = FakeBlockPool()
    r = req("r4", 32)
    s.update_state_after_alloc(r, FakeBlocks([gpu.alloc(8)]), 0)
    r.num_computed_tokens = 32
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r4", (list(range(1, 9)),))],
        num_scheduled_tokens={"r4": 32}))
    assert meta.store_event < 0             # no pressure -> stay cold
    assert len(s._to_store) == 8

    # finish -> snapshot tail forced even without pressure, holds free
    hold, extra = s.request_finished_all_groups(
        r, (list(range(1, 9)),))
    assert hold is True
    meta = s.build_connector_metadata(FakeSchedulerOutput())
    assert meta.store_event >= 0
    assert len(meta.store_gpu_blocks) == 8
    ack_stores(s, meta)
    assert s.drain_ready_to_free() == {"r4"}   # deferred free released


def test_preemption_drops_bookkeeping():
    s = make_sched()
    r = req("r5", 12)
    s.update_state_after_alloc(r, FakeBlocks([[1, 2, 3]]), 0)
    r.num_computed_tokens = 12
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r5", ([1, 2, 3],))],
        num_scheduled_tokens={"r5": 12}))
    eid = meta.store_event
    assert eid >= 0 and len(meta.store_gpu_blocks) == 3
    s.on_preemptions({"r5"})
    assert "r5" not in s._shadow and "r5" not in s._reqs_to_store
    # in-flight event table kept: flushed copies still commit on ack
    assert eid in s._event_stores
    s.update_connector_output(FakeConnectorOutput(
        kv_connector_worker_meta=FissionKVWorkerMetadata(
            completed_store_events={eid: 1})))
    assert s.index.ready == 3


def test_worker_count_aggregation():
    s = make_sched({"cpu_bytes_to_use": 1 << 30})
    s._expected_worker_count = 2
    r = req("r6", 8)
    s.update_state_after_alloc(r, FakeBlocks([[1, 2]]), 0)
    r.num_computed_tokens = 8
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r6", ([1, 2],))],
        num_scheduled_tokens={"r6": 8}))
    eid = meta.store_event
    s.update_connector_output(FakeConnectorOutput(
        kv_connector_worker_meta=FissionKVWorkerMetadata(
            completed_store_events={eid: 1})))
    assert s.index.ready == 0               # 1 of 2 workers: not committed
    s.update_connector_output(FakeConnectorOutput(
        kv_connector_worker_meta=FissionKVWorkerMetadata(
            completed_store_events={eid: 1})))
    assert s.index.ready == 2


def test_load_pin_survives_request_finish():
    s = make_sched()
    gpu = FakeBlockPool()
    r = req("r7", 8)
    s.update_state_after_alloc(r, FakeBlocks([gpu.alloc(2)]), 0)
    r.num_computed_tokens = 8
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r7", ([1, 2],))],
        num_scheduled_tokens={"r7": 8}))
    ack_stores(s, meta)

    r2 = req("r8", 8)
    n, _ = s.get_num_new_matched_tokens(r2, 0)
    assert n == 4
    s.update_state_after_alloc(r2, FakeBlocks([gpu.alloc(2)]), 4)
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r8", ([3, 4],))],
        num_scheduled_tokens={"r8": 4}))
    assert meta.load_event >= 0
    # request ends while load in flight -> hold, release on ack
    hold, _ = s.request_finished_all_groups(r2, ([3, 4],))
    assert hold
    s.update_connector_output(FakeConnectorOutput(finished_recving={"r8"}))
    assert s.drain_ready_to_free() == {"r8"}


def test_orphan_hit_released():
    s = make_sched()
    r = req("r9", 12)
    s.update_state_after_alloc(r, FakeBlocks([[1, 2, 3]]), 0)
    r.num_computed_tokens = 12
    meta = s.build_connector_metadata(FakeSchedulerOutput(
        scheduled_new_reqs=[FakeNewReq("r9", ([1, 2, 3],))],
        num_scheduled_tokens={"r9": 12}))
    ack_stores(s, meta)

    r2 = req("r10", 12)
    n, _ = s.get_num_new_matched_tokens(r2, 0)
    assert n == 8
    pinned_before = s.index.pinned
    # core allocated ZERO external tokens (edge case): pins must free
    s.update_state_after_alloc(r2, FakeBlocks([[]]), 0)
    assert s.index.pinned == pinned_before - 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
