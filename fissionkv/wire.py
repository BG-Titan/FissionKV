# SPDX-License-Identifier: Apache-2.0
"""Wire metadata exchanged between scheduler and worker each step.

Mirrors vLLM's SimpleCPUOffload metadata shape (one batched list per step
with a shared CUDA event id) plus what FissionKV adds:

* ``scheduled_max_block_id`` - elastic-mode grow hint: the largest GPU block
  id scheduled this step, so the VMM pool maps its chunks BEFORE forward
  writes KV there (attention itself touches the block).
* ``layer_groups`` - when per-layer overlap is on, one load event per group
  of layer names; ``wait_for_layer_load`` waits only on that group's event.

The module imports vLLM's metadata base classes when available and falls
back to plain object bases so the pure-python parts stay unit-testable
without vLLM installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised in the vLLM environment
    from torch.cuda import Event
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata,
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics_kv_connector import (  # noqa: E501
        KVConnectorWorkerMetricsWrapper,
    )
except ImportError:  # standalone unit-test fallback
    Event = Any  # type: ignore[assignment,misc]

    class KVConnectorMetadata:  # type: ignore[no-redef]
        pass

    class KVConnectorWorkerMetadata:  # type: ignore[no-redef]
        def aggregate(self, *args, **kwargs):
            raise NotImplementedError

        def is_empty(self) -> bool:
            return True

    KVConnectorWorkerMetricsWrapper = KVConnectorWorkerMetadata  # type: ignore[misc,assignment]


@dataclass
class FissionKVMetadata(KVConnectorMetadata):
    """One batched step of FissionKV work (built by the scheduler)."""

    # loads: GPU block ids <- host slot ids, issued on first forward pass
    load_event: int = -1
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_host_blocks: list[int] = field(default_factory=list)
    # load event -> req ids, so the worker marks requests done
    load_event_to_reqs: dict[int, set[str]] = field(default_factory=dict)

    # stores: GPU -> host (eager write-through scan or lazy aging)
    store_event: int = -1
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_host_blocks: list[int] = field(default_factory=list)

    # per-request store accounting: store_host_blocks index -> req id
    store_blocks_to_reqs: dict[int, str] = field(default_factory=dict)

    need_flush: bool = False
    # elastic: map VMM chunks covering all ids <= this before forward
    scheduled_max_block_id: int = -1
    # opt-in per-layer overlap: list of layer-name groups; one event each
    layer_groups: list[list[str]] | None = None

    def __bool__(self) -> bool:
        return (
            self.load_event >= 0
            or self.store_event >= 0
            or self.need_flush
            or self.scheduled_max_block_id >= 0
        )


@dataclass
class FissionKVWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> scheduler ack of completed store events.

    Each worker reports ``{event_id: 1}``; ``aggregate`` sums counts
    across workers in one step; the scheduler processes an event only
    when all ``world_size`` workers have reported (reference contract).
    Load completions travel through the base ``finished_recving`` field.
    """

    completed_store_events: dict[int, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.completed_store_events

    def aggregate(self, worker_metadata: "KVConnectorWorkerMetadata") -> None:
        if isinstance(
            worker_metadata,
            (FissionKVWorkerMetadata, KVConnectorWorkerMetricsWrapper),
        ):
            for eid, count in worker_metadata.completed_store_events.items():
                self.completed_store_events[eid] = \
                    self.completed_store_events.get(eid, 0) + count
