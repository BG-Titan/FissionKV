# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side tier manager (CPU-authoritative staging policy).

The object model and core call order mirror vLLM's
``SimpleCPUOffloadScheduler`` (vllm/v1/simple_kv_offload/manager.py):
get_num_new_matched_tokens -> (tokens, is_async) pins a CPU hit; the core
allocates GPU blocks and calls update_state_after_alloc, which records the
GPU<->host pairs; build_connector_metadata batches one store event and one
load event per step; completions return through finished_recving (loads)
and kv_connector_worker_meta counts (stores, summed across workers).

FissionKV changes the *staging policy* (docs/strata-kv-mechanism.md §1.3),
not the state machine:

* lazy (default, Strata cold staging): a full block is queued when it
  completes but only copied once host-tier pressure (``high_water``) or
  request finish forces it. Zero GPU work while the host tier has headroom.
* eager: queued blocks drain every step, paced by ``store_batch_blocks``.

Two deliberate deviations from the reference, both strict-safety:
* ``request_finished`` may return ``(True, {})`` (native defer protocol:
  the core keeps the blocks alive until the connector reports the id in
  ``finished_sending``; we inject that from ``drain_ready_to_free`` once
  every transfer touching the request completed).
* finished requests stage their remaining complete blocks first
  (end-of-session snapshot, docs §1.3), so the next session hits from CPU.

v1 tiering scope: single-group configs (plain attention / MLA). Multi-group
hybrid configs run pass-through: restoring only the full-attention group
while SWA/mamba groups miss the skipped tokens would corrupt output.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .block_index import HostBlockIndex
from .config import FissionKVConfig
from .wire import FissionKVMetadata

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


@dataclass
class _ReqLoad:
    gpu_ids: list[int] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)   # pinned until done
    event_id: int | None = None
    finished: bool = False        # request ended while load in flight


@dataclass
class _ReqStore:
    request: Any                  # Request ref (same-process invariant)
    cursor_fa: int = 0            # FA blocks enqueued so far
    events: set[int] = field(default_factory=set)
    tail: list[tuple[Any, int]] | None = None  # snapshot pending issue
    finished: bool = False


@dataclass
class _PendingHit:
    slots: list[int]
    hit_len: int                  # tokens
    num_computed: int             # tokens, at match time


class FissionKVScheduler:
    """Host index + staging policy; owned by FissionKVConnector (scheduler)."""

    def __init__(self, vllm_config, kv_cache_config,
                 kv_cache_events=None, kv_connector=None) -> None:
        self.vllm_config = vllm_config
        kv_cfg = vllm_config.kv_transfer_config
        self.config = FissionKVConfig.from_extra_config(
            kv_cfg.kv_connector_extra_config)
        self.config.validate()

        specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups]
        self.fa_group_idx: int | None = None
        for i, spec in enumerate(specs):
            if type(spec).__name__ == "FullAttentionSpec" or \
                    type(spec).__name__.startswith("MLA"):
                self.fa_group_idx = i
                break
        self.tiering = self.fa_group_idx is not None and len(specs) == 1
        if self.fa_group_idx is None:
            logger.warning("FissionKV: no full-attention group found; pass-through")
        elif len(specs) > 1:
            logger.warning(
                "FissionKV: multi-group hybrid config; tiering disabled in v1 "
                "(host restore must cover every cacheable group)")
        if getattr(vllm_config.parallel_config,
                   "decode_context_parallel_size", 0) > 1:
            raise NotImplementedError(
                "FissionKV: decode_context_parallel is not supported yet")

        self.block_size = specs[self.fa_group_idx or 0].block_size
        # Request.block_hashes granularity == the group block size for
        # single-group configs; >1 only if a finer hash block size is set.
        self.hash_block_size = getattr(kv_cache_config, "hash_block_size",
                                       None) or self.block_size
        if self.hash_block_size > self.block_size:
            raise NotImplementedError(
                "FissionKV: hash granularity finer than block size unsupported")
        self.sub_per_fa = self.block_size // self.hash_block_size

        self.store_mode = self.config.store_mode
        self.index: HostBlockIndex | None = None
        self.num_cpu_blocks = 0
        self._expected_worker_count = \
            vllm_config.parallel_config.world_size

        self._pending_hits: dict[str, _PendingHit] = {}
        self._reqs_to_load: dict[str, _ReqLoad] = {}
        self._reqs_to_store: dict[str, _ReqStore] = {}
        self._shadow: dict[str, list[int]] = {}
        self._to_store: deque[tuple[Any, int, str]] = deque()
        self._staged_gpu: set[int] = set()
        self._event_stores: dict[int, list[tuple[Any, int]]] = {}
        self._event_reqs: dict[int, set[str]] = {}
        self._event_count: dict[int, int] = {}
        self._load_event_reqs: dict[int, set[str]] = {}
        self._next_event_id = 0
        self._ready_to_free: set[str] = set()
        self._holds: set[str] = set()
        self._shutdown = False

        logger.info(
            "FissionKV scheduler: store_mode=%s tiering=%s block_size=%d "
            "sub_per_fa=%d high_water=%.2f", self.store_mode, self.tiering,
            self.block_size, self.sub_per_fa, self.config.high_water)

    # ---------------------------------------------------------------- binds
    def set_host_capacity(self, num_cpu_blocks: int) -> None:
        if self.index is not None or num_cpu_blocks <= 0:
            return
        self.num_cpu_blocks = num_cpu_blocks
        self.index = HostBlockIndex(num_cpu_blocks)
        logger.info("FissionKV host tier online: %d block slots", num_cpu_blocks)

    # ---------------------------------------------------------------- keys
    def _fa_chain(self, request: "Request", n_cap: int | None = None) -> list:
        """One key per complete FA block: its last sub-hash (the unique
        hash covering that block's token range)."""
        hashes = request.block_hashes
        n = len(hashes) // self.sub_per_fa
        if n_cap is not None:
            n = min(n, n_cap)
        out = [hashes[(i + 1) * self.sub_per_fa - 1] for i in range(n)]
        return out

    # ---------------------------------------------------------------- hits
    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if not self.tiering or self.index is None or self._shutdown:
            return 0, False
        req_id = request.request_id
        stale = self._pending_hits.pop(req_id, None)
        if stale is not None:
            self._release_hit(stale)
        if getattr(request, "skip_reading_prefix_cache", False):
            return 0, False
        if num_computed_tokens % self.block_size != 0:
            return 0, False
        gpu_fa = num_computed_tokens // self.block_size
        max_new = request.num_tokens - 1 - num_computed_tokens
        if max_new <= 0:
            return 0, False
        chain = self._fa_chain(request)
        if len(chain) <= gpu_fa:
            return 0, False
        hit_slots = self.index.match_chain(chain[gpu_fa:])
        if not hit_slots:
            return 0, False
        k = min(len(hit_slots), max_new // self.block_size)
        hit_slots = hit_slots[:k]
        if not hit_slots:
            return 0, False
        for slot in hit_slots:
            self.index.pin(slot)
        self._pending_hits[req_id] = _PendingHit(
            hit_slots, k * self.block_size, num_computed_tokens)
        return k * self.block_size, True

    def _release_hit(self, hit: _PendingHit) -> None:
        if self.index is not None:
            for slot in hit.slots:
                self.index.unpin(slot)

    def update_state_after_alloc(self, request: "Request", blocks: Any,
                                 num_external_tokens: int) -> None:
        req_id = request.request_id
        if self.tiering and not self._shutdown:
            state = self._reqs_to_store.get(req_id)
            if state is None:
                self._reqs_to_store[req_id] = _ReqStore(request=request)
            elif state.request is not request:
                state.request = request

        pending = self._pending_hits.pop(req_id, None)
        if num_external_tokens <= 0:
            if pending is not None:
                logger.warning(
                    "FissionKV: no external blocks allocated for %s despite a "
                    "hit; releasing %d pinned slots", req_id, len(pending.slots))
                self._release_hit(pending)
            return
        if pending is None:
            if not getattr(self, "_warned_orphan", False):
                self._warned_orphan = True
                logger.warning(
                    "FissionKV: update_state_after_alloc with %d external "
                    "tokens but no pending hit for %s", num_external_tokens,
                    req_id)
            return
        if pending.hit_len != num_external_tokens:
            self._release_hit(pending)
            logger.warning(
                "FissionKV: accepted hit %d != external %d for %s",
                pending.hit_len, num_external_tokens, req_id)
            return

        n = pending.hit_len // self.block_size
        try:
            group_ids = blocks.get_block_ids()[self.fa_group_idx]
        except (AttributeError, IndexError, KeyError):
            group_ids = list(blocks)
        start = pending.num_computed // self.block_size
        gpu_ids = list(group_ids[start:start + n])
        if len(gpu_ids) != n:
            self._release_hit(pending)
            logger.warning("FissionKV: GPU block window short for %s", req_id)
            return
        # Slots stay pinned through the load; _finish_load releases the pin.
        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = _ReqLoad(gpu_ids=gpu_ids,
                                              slots=pending.slots)

    # ---------------------------------------------------------------- build
    def build_connector_metadata(
            self, scheduler_output: "SchedulerOutput") -> FissionKVMetadata:
        meta = FissionKVMetadata()
        if not self.tiering or self.index is None or self._shutdown:
            return meta

        self._update_shadow(scheduler_output)

        # --- loads: one batched event for requests armed but not issued ---
        for req_id, load in self._reqs_to_load.items():
            if load.event_id is not None:
                continue
            if meta.load_event < 0:
                meta.load_event = self._next_event_id
                self._next_event_id += 1
                self._load_event_reqs[meta.load_event] = set()
            load.event_id = meta.load_event
            self._load_event_reqs[meta.load_event].add(req_id)
            meta.load_gpu_blocks.extend(load.gpu_ids)
            meta.load_host_blocks.extend(load.slots)
            meta.load_event_to_reqs.setdefault(meta.load_event,
                                               set()).add(req_id)

        # --- queue blocks completed by this step (copy deferred by policy) ---
        scheduled = scheduler_output.num_scheduled_tokens or {}
        for req_id, state in self._reqs_to_store.items():
            if state.finished:
                continue
            request = state.request
            done = request.num_computed_tokens // self.block_size
            if self.store_mode == "eager":
                # blocks this step writes count: the worker orders the D2H
                # after the compute-stream event (wait_for_save).
                done = (request.num_computed_tokens
                        + scheduled.get(req_id, 0)) // self.block_size
            n = min(done, len(self._shadow.get(req_id, ())),
                    len(self._fa_chain(request)))
            if n <= state.cursor_fa:
                continue
            ids = self._shadow[req_id]
            chain = self._fa_chain(request, n)
            for pos in range(state.cursor_fa, n):
                h, gpu_id = chain[pos], ids[pos]
                if h in self.index or gpu_id in self._staged_gpu:
                    continue
                self._staged_gpu.add(gpu_id)
                self._to_store.append((h, gpu_id, req_id))
            state.cursor_fa = n

        # --- end-of-session snapshots (forced, ahead of pressure) ---
        batch: list[tuple[Any, int, str]] = []
        for req_id, state in self._reqs_to_store.items():
            if state.tail:
                batch.extend((h, g, req_id) for h, g in state.tail)
                state.tail = None
        batch.extend(self._select_stores())
        self._emit_stores(meta, batch)
        return meta

    def _update_shadow(self, scheduler_output: "SchedulerOutput") -> None:
        """Shadow copy of per-request FA block ids (the manager never reads
        the live block pool; mirrors the reference store tracking)."""
        fa = self.fa_group_idx
        for new_req in scheduler_output.scheduled_new_reqs:
            ids = new_req.block_ids
            self._shadow[new_req.req_id] = (list(ids[fa]) if ids
                                            else [])
        cached = scheduler_output.scheduled_cached_reqs
        if cached is None:
            return
        for i, req_id in enumerate(cached.req_ids):
            new_ids = cached.new_block_ids[i]
            if new_ids is None:
                continue
            fa_ids = list(new_ids[fa]) if new_ids else []
            if req_id in cached.resumed_req_ids:
                self._shadow[req_id] = fa_ids
                state = self._reqs_to_store.get(req_id)
                if state is not None:
                    state.cursor_fa = 0
            else:
                self._shadow.setdefault(req_id, []).extend(fa_ids)

    # ---------------------------------------------------------------- stores
    def _select_stores(self) -> list[tuple[Any, int, str]]:
        if not self._to_store:
            return []
        if self.store_mode == "lazy":
            headroom = self.index.free_slots / max(self.num_cpu_blocks, 1)
            if headroom > 1.0 - self.config.high_water:
                return []           # no pressure yet: stay cold (docs §1.3)
            n = len(self._to_store)  # drain everything under pressure
        else:
            n = self.config.store_batch_blocks
        n = min(len(self._to_store), n)
        return [self._to_store.popleft() for _ in range(n)]

    def _emit_stores(self, meta: FissionKVMetadata,
                     batch: list[tuple[Any, int, str]]) -> None:
        if not batch:
            return
        try:
            slots = self.index.reserve(len(batch))
        except IndexError:
            self._requeue(batch)
            return
        if len(slots) < len(batch):
            self._requeue(batch[len(slots):])
            batch = batch[:len(slots)]
        if meta.store_event < 0:
            meta.store_event = self._next_event_id
            self._next_event_id += 1
            self._event_stores[meta.store_event] = []
            self._event_reqs[meta.store_event] = set()
        event_id = meta.store_event
        pend = self._event_stores[event_id]
        for (h, gpu_id, req_id), slot in zip(batch, slots):
            self.index.begin_store(h, slot)
            self._staged_gpu.discard(gpu_id)
            idx = len(meta.store_gpu_blocks)
            meta.store_gpu_blocks.append(gpu_id)
            meta.store_host_blocks.append(slot)
            meta.store_blocks_to_reqs[idx] = req_id
            pend.append((h, slot))
            self._event_reqs[event_id].add(req_id)
            state = self._reqs_to_store.get(req_id)
            if state is not None:
                state.events.add(event_id)

    def _requeue(self, batch: list[tuple[Any, int, str]]) -> None:
        for item in reversed(batch):
            self._staged_gpu.add(item[1])
            self._to_store.appendleft(item)

    # ---------------------------------------------------------------- output
    def update_connector_output(self, connector_output: Any) -> None:
        for req_id in connector_output.finished_recving or ():
            self._finish_load(req_id)
        worker_meta = getattr(connector_output, "kv_connector_worker_meta",
                              None)
        if worker_meta is None:
            return
        for event_id, count in getattr(worker_meta,
                                       "completed_store_events",
                                       {}).items():
            total = self._event_count.get(event_id, 0) + count
            if total >= self._expected_worker_count:
                self._event_count.pop(event_id, None)
                self._finish_store_event(event_id)
            else:
                self._event_count[event_id] = total

    def _finish_load(self, req_id: str) -> None:
        load = self._reqs_to_load.pop(req_id, None)
        if load is None:
            return
        if self.index is not None:
            for slot in load.slots:
                self.index.unpin(slot)
        if load.finished:
            self._maybe_release(req_id)

    def _finish_store_event(self, event_id: int) -> None:
        pend = self._event_stores.pop(event_id, None)
        reqs = self._event_reqs.pop(event_id, set())
        if pend is None:
            return  # stale ack from before a reset
        for h, slot in pend:
            self.index.commit_store(slot)
        for req_id in reqs:
            state = self._reqs_to_store.get(req_id)
            if state is None:
                continue
            state.events.discard(event_id)
            if state.finished and not state.events and not state.tail:
                self._reqs_to_store.pop(req_id, None)
                self._maybe_release(req_id)

    def _maybe_release(self, req_id: str) -> None:
        """Request had a deferred free; everything touching it is done."""
        if req_id in self._holds and req_id not in self._reqs_to_load \
                and req_id not in self._reqs_to_store:
            self._holds.discard(req_id)
            self._shadow.pop(req_id, None)
            self._ready_to_free.add(req_id)

    def drain_ready_to_free(self) -> set[str]:
        out, self._ready_to_free = self._ready_to_free, set()
        return out

    # ---------------------------------------------------------------- finish
    def request_finished_all_groups(self, request: "Request",
                                    block_ids: tuple[list[int], ...]
                                    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        pending = self._pending_hits.pop(req_id, None)
        if pending is not None:
            self._release_hit(pending)

        hold = False
        load = self._reqs_to_load.get(req_id)
        if load is not None:
            if load.event_id is not None:
                load.finished = True      # in flight: cleanup on its ack
                hold = True
            else:
                if self.index is not None:
                    for slot in load.slots:
                        self.index.unpin(slot)
                self._reqs_to_load.pop(req_id, None)

        state = self._reqs_to_store.get(req_id)
        if state is not None and self.index is not None:
            # queued-but-unissued items join the forced snapshot (lazy mode
            # keeps them queued indefinitely; GPU blocks die with the free)
            queued: list[tuple[Any, int]] = []
            if self._to_store:
                keep = deque()
                for item in self._to_store:
                    if item[2] == req_id:
                        queued.append((item[0], item[1]))
                        self._staged_gpu.discard(item[1])
                    else:
                        keep.append(item)
                self._to_store = keep
            fa_ids = (block_ids[self.fa_group_idx]
                      if len(block_ids) > self.fa_group_idx else [])
            # complete blocks = whole blocks of the finished request
            n_full = min(len(fa_ids), request.num_computed_tokens
                         // self.block_size)
            chain = self._fa_chain(request, n_full)
            tail = list(state.tail or []) + queued
            for pos in range(min(state.cursor_fa, n_full), n_full):
                h, gpu_id = chain[pos], fa_ids[pos]
                if h not in self.index and gpu_id not in self._staged_gpu:
                    tail.append((h, gpu_id))
            if tail or state.events:
                state.finished = True
                state.tail = tail or None
                hold = True
            else:
                self._reqs_to_store.pop(req_id, None)
        elif state is not None:
            self._reqs_to_store.pop(req_id, None)

        if hold:
            self._holds.add(req_id)
            return True, {}
        self._shadow.pop(req_id, None)
        return False, None

    # ---------------------------------------------------------------- misc
    def on_preemptions(self, preempted: set[str]) -> None:
        """The worker synced all in-flight copies (need_flush) before the
        core recycles these blocks; drop bookkeeping, keep in-flight event
        tables so their (completed) acks still commit."""
        for req_id in preempted:
            pending = self._pending_hits.pop(req_id, None)
            if pending is not None:
                self._release_hit(pending)
            load = self._reqs_to_load.pop(req_id, None)
            if load is not None and self.index is not None:
                for slot in load.slots:
                    self.index.unpin(slot)
            self._reqs_to_store.pop(req_id, None)
            self._shadow.pop(req_id, None)
        if self._to_store:
            keep = deque(item for item in self._to_store
                         if item[2] not in preempted)
            self._to_store = keep

    def has_pending_block_frees(self) -> bool:
        return bool(self._holds or self._ready_to_free)

    def has_pending_work(self) -> bool:
        return bool(self._to_store or self._event_stores
                    or any(s.tail for s in self._reqs_to_store.values()))

    def get_block_ids_with_load_errors(self) -> list[int]:
        return []

    def take_worker_handshake(self, num_cpu_blocks: int,
                              grow_hint: int) -> None:
        self.set_host_capacity(num_cpu_blocks)

    def shutdown(self) -> None:
        self._shutdown = True
        self._to_store.clear()
        self._event_stores.clear()
