# SPDX-License-Identifier: Apache-2.0
"""FissionKVConnector: the vLLM KVConnectorBase_V1 glue.

Registered externally (no in-tree vLLM edits) via
``KVConnectorFactory.register_connector("FissionKV", ...)`` from a
``vllm.general_plugins`` entry point (see fissionkv/plugin.py + pyproject).

Scheduler process role -> FissionKVScheduler (host index + staging policy).
Worker process role   -> FissionKVWorker (pinned arena + copy engine).
Mirrors vllm/distributed/kv_transfer/kv_connector/v1/
simple_cpu_offload_connector.py method-for-method so core flows
(WAITING_FOR_REMOTE_KVS, two-phase frees, preemption flush) behave the same.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from .config import FissionKVConfig
from .wire import FissionKVMetadata

if TYPE_CHECKING:
    import torch

    from .scheduler import FissionKVScheduler
    from .worker import FissionKVWorker

    from vllm.config import VllmConfig, KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: F401
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


class FissionKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Tiered + elastic KV cache connector (Strata-inspired)."""

    @property
    def requires_kv_delivery(self) -> bool:
        # Best-effort cache: a dropped save is a future miss (reference
        # same choice), while we hold frees for in-flight saves anyway.
        return False

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig") -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        self.scheduler_manager: FissionKVScheduler | None = None
        self.worker_handler = None
        # base stores _kv_cache_config, but mirror it locally: our helpers
        # use self.kv_cache_config / self.vllm_config explicitly
        self.kv_cache_config = kv_cache_config
        self.vllm_config = vllm_config
        self._kv_cache_mgr = None
        from . import patch as _patch
        self._elastic = _patch.elastic_enabled()

        if role == KVConnectorRole.SCHEDULER:
            # local import: scheduler side must load without the native
            # module / torch in CPU-only scheduler processes
            from .scheduler import FissionKVScheduler
            self.scheduler_manager = FissionKVScheduler(
                vllm_config, kv_cache_config,
                kv_cache_events=getattr(self, "kv_cache_events", {}),
                kv_connector=self)
            self._init_host_capacity_on_scheduler()
        elif role == KVConnectorRole.WORKER:
            from .worker import FissionKVWorker
            self.worker_handler = FissionKVWorker(
                vllm_config, kv_cache_config, self._config_from(vllm_config))
        # KVConnectorRole.KV_BOTH (in-process engine) gets both:
        if getattr(KVConnectorRole, "KV_BOTH", None) is not None and \
                role == KVConnectorRole.KV_BOTH:
            from .scheduler import FissionKVScheduler
            self.scheduler_manager = FissionKVScheduler(
                vllm_config, kv_cache_config, kv_connector=self)
            self._init_host_capacity_on_scheduler()
            from .worker import FissionKVWorker
            self.worker_handler = FissionKVWorker(
                vllm_config, kv_cache_config, self._config_from(vllm_config))

    @staticmethod
    def _config_from(vllm_config) -> FissionKVConfig:
        cfg = FissionKVConfig.from_extra_config(
            vllm_config.kv_transfer_config.kv_connector_extra_config)
        cfg.validate()
        return cfg

    # ---------------------------------------------------------------- sizing
    def _init_host_capacity_on_scheduler(self) -> None:
        """Host arena size must be known to the scheduler BEFORE the first
        step (index capacity); computed from config exactly as the worker
        will lay it out."""
        sched = self.scheduler_manager
        assert sched is not None
        cfg = sched.config
        if not cfg.tiering_enabled or not sched.tiering:
            return
        fa = sched.fa_group_idx
        group = self.kv_cache_config.kv_cache_groups[fa]
        spec = group.kv_cache_spec
        bytes_per_slot = len(group.layer_names) * spec.page_size_bytes
        if bytes_per_slot <= 0:
            return
        world = self.vllm_config.parallel_config.world_size
        per_rank = cfg.cpu_bytes_to_use // world
        if cfg.cpu_bytes_to_use_per_rank > 0:
            per_rank = cfg.cpu_bytes_to_use_per_rank
        sched.set_host_capacity(int(per_rank // bytes_per_slot))

    # --------------------------------------------------------- worker role
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker_handler is not None:
            assert isinstance(connector_metadata, FissionKVMetadata)
            self.worker_handler.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker_handler is not None:
            self.worker_handler.clear_connector_metadata()

    def handle_preemptions(self, kv_connector_metadata) -> None:
        if self.worker_handler is not None:
            self.worker_handler.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context, **kwargs) -> None:
        if self.worker_handler is not None:
            self.worker_handler.start_load_kv()

    def wait_for_layer_load(self, layer_name: str) -> None:
        # FissionKV loads whole blocks once, before the first forward; no
        # per-layer gating needed. (opt-in per-layer overlap is a
        # documented extension using FissionKVMetadata.layer_groups.)
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata=None, **kwargs) -> None:
        # stores are issued in batch at wait_for_save()
        return

    def wait_for_save(self) -> None:
        if self.worker_handler is not None:
            self.worker_handler.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]
                     ) -> tuple[set[str] | None, set[str] | None]:
        if self.worker_handler is not None:
            _, finished_recving = self.worker_handler.get_finished()
            # Stores report completion through worker metadata events;
            # sending stays None so the core frees only via our
            # request_finished hold + drain path.
            return None, finished_recving or None
        return None, None

    def build_connector_worker_meta(self):
        if self.worker_handler is not None:
            return self.worker_handler.build_connector_worker_meta()
        return None

    # ------------------------------------------------------ scheduler role
    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.get_num_new_matched_tokens(
                request, num_computed_tokens)
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks,
                                 num_external_tokens: int) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.update_state_after_alloc(
                request, blocks, num_external_tokens)

    def bind_kv_cache_manager(self, kv_cache_manager) -> None:
        # base also binds the GPU block pool; we keep the manager to read
        # the elastic block-id watermark (patch.py tracks it on BlockPool)
        super().bind_kv_cache_manager(kv_cache_manager)
        self._kv_cache_mgr = kv_cache_manager

    def build_connector_meta(
            self, scheduler_output) -> KVConnectorMetadata:
        if self.scheduler_manager is not None:
            preempted = getattr(scheduler_output, "preempted_req_ids", None)
            if preempted:
                # worker synced all in-flight copies via handle_preemptions
                # (need_flush below) BEFORE these blocks can be reused;
                # drop scheduler-side bookkeeping for the dead ids.
                self.scheduler_manager.on_preemptions(set(preempted))
            meta = self.scheduler_manager.build_connector_metadata(
                scheduler_output)
            meta.need_flush = bool(preempted)
            if self._elastic:
                pool = getattr(getattr(self, "_kv_cache_mgr", None),
                               "block_pool", None)
                meta.scheduled_max_block_id = getattr(
                    pool, "_fissionkv_max_issued", -1)
            return meta
        return FissionKVMetadata()

    def update_connector_output(self, connector_output) -> None:
        if self.scheduler_manager is None:
            return
        sched = self.scheduler_manager
        sched.update_connector_output(connector_output)
        # delayed frees whose transfers just completed: core consumes
        # finished_sending right after this call (scheduler.py:3448-3461)
        ready = sched.drain_ready_to_free()
        if ready:
            sending = connector_output.finished_sending
            if sending is None:
                connector_output.finished_sending = ready
            else:
                sending.update(ready)

    def request_finished(self, request: "Request", block_ids: list[int]
                         ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, (block_ids,))
        return False, None

    def request_finished_all_groups(
            self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler_manager is not None:
            return self.scheduler_manager.request_finished_all_groups(
                request, block_ids)
        return False, None

    def has_pending_push_work(self) -> bool:
        """Liveness hint: stores queued or in flight should defer engine
        shutdown steps (same role as the reference method)."""
        sched = self.scheduler_manager
        return sched.has_pending_work() if sched is not None else False

    def has_pending_block_frees(self) -> bool:
        """Pending transfers will release deferred-free blocks instead of
        forcing a preemption (core scheduler.py:913 backpressure path)."""
        sched = self.scheduler_manager
        return sched.has_pending_block_frees() if sched is not None else False

    def take_events(self) -> Iterable:
        return []

    def shutdown(self) -> None:
        if self.scheduler_manager is not None:
            self.scheduler_manager.shutdown()
        if self.worker_handler is not None:
            self.worker_handler.shutdown()
