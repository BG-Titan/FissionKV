# SPDX-License-Identifier: Apache-2.0
"""Worker-side FissionKV state: pinned host KV buffers + batch copy engine.

Mirrors vLLM's SimpleCPUOffloadWorker event-watermark pattern (background
copy-manager thread, capture-replay batches, pin-before-store) - that part
is the vLLM-correct way to sequence H2D/D2H against forward passes. The
copy *execution* is FissionKV's native core: one dlopen'd CUDA driver call
carries all descriptors for the step, with Strata run-coalescing so long
id-adjacent ranges become single bulk copies (docs §1.3).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

from .config import FissionKVConfig
from .wire import FissionKVMetadata, FissionKVWorkerMetadata

logger = init_logger(__name__)

try:
    from . import _C
    HAVE_NATIVE = True
except ImportError:  # pragma: no cover
    _C = None
    HAVE_NATIVE = False


@dataclass
class _CopyJob:
    is_store: bool
    event_idx: int
    stream_ptr: int
    event: torch.cuda.Event
    gpu_block_ids: list[int]
    host_block_ids: list[int]


def build_planes(kv_caches: dict) -> list:
    """GPU planes: [num_blocks, block_stride_bytes] byte views.

    vllm hands us STRIDED dtype views over one flat int8 backing;
    .view(uint8).reshape() would COPY them (detaching swaps from the real
    cache), so rebuild the byte plane on the SAME storage.  Shape-generic:
    covers the dense/FA layouts ([blocks, 2|H, page, D]), the MLA layout
    ([blocks, page, 1, kv_lora_rank + rope]) and whatever strided view a
    hybrid plan hands over.
    """
    planes = []
    for t in kv_caches.values():
        num_blocks = t.shape[0]
        bpb = t[0].numel() * t.element_size()       # layer bytes / block
        row_bytes = t.stride(0) * t.element_size()  # block -> block
        off = t.storage_offset() * t.element_size()
        plane = torch.empty(0, dtype=torch.uint8, device=t.device).set_(
            t.untyped_storage(), off, (num_blocks, bpb), (row_bytes, 1))
        planes.append(plane)
    return planes


@dataclass
class PlaneGeometry:
    """Physical byte layout of a set of planes.

    ``layer_stride`` is the physical delta between consecutive layers, which
    is only meaningful when every layer's plane is laid out the same way.
    MLA / hybrid plans can mix sizes, so the exact span is computed per plane
    unless ``uniform_layer_stride`` says the closed form is safe.
    """

    num_blocks: int
    cpu_block_bytes: int        # payload bytes per block, summed over planes
    block_stride: int           # physical bytes between GPU blocks (plane 0)
    layer_stride: int           # physical bytes between layers (plane 1 - plane 0)
    block_strides: list         # per-plane physical block stride
    plane_offsets: list         # plane i data_ptr - plane 0 data_ptr
    uniform_layer_stride: bool

    def span_for(self, max_block_id: int) -> int:
        """Highest byte (relative to plane 0's start) that blocks
        0..max_block_id can touch.  Per-plane exact; with a uniform layout
        this collapses to the closed form the fast path uses."""
        if self.uniform_layer_stride:
            return ((len(self.plane_offsets) - 1) * self.layer_stride
                    + (max_block_id + 1) * self.block_stride)
        return max(off + (max_block_id + 1) * bs
                   for off, bs in zip(self.plane_offsets, self.block_strides))


def plane_geometry(planes: list) -> PlaneGeometry:
    base = planes[0].data_ptr()
    offsets = [t.data_ptr() - base for t in planes]
    strides = [t.stride(0) for t in planes]
    layer_stride = offsets[1] if len(offsets) > 1 else planes[0].shape[1]
    uniform = (all(offsets[i] - offsets[i - 1] == layer_stride
                   for i in range(1, len(offsets)))
               and all(s == strides[0] for s in strides))
    return PlaneGeometry(
        num_blocks=planes[0].shape[0],
        cpu_block_bytes=sum(t.shape[1] for t in planes),
        block_stride=strides[0],
        layer_stride=layer_stride,
        block_strides=strides,
        plane_offsets=offsets,
        uniform_layer_stride=uniform,
    )


class FissionKVWorker:
    """Owns the pinned host arena; executes the step's D2H/H2D batches."""

    def __init__(self, vllm_config, kv_cache_config,
                 fissionkv_config: FissionKVConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.fissionkv_config = fissionkv_config
        self.spec_decode_extra_kv_tokens = 0
        self._finalized = False

        self._copy_queue: list[_CopyJob] = []
        self._copy_lock = threading.Lock()
        self._copy_cv = threading.Condition(self._copy_lock)
        self._stop = threading.Event()
        self._copy_thread: threading.Thread | None = None

        # event id -> (event, [req ids])
        self._load_events: dict[int, tuple[torch.cuda.Event, list[str]]] = {}
        self._store_events: dict[int, tuple[torch.cuda.Event, list[str]]] = {}
        self._completed_store_events: dict[int, int] = {}
        self._next_load_event_idx = 0
        self._next_store_event_idx = 0

        self._gpu_tensors: list[torch.Tensor] = []
        self._cpu_tensors: list[torch.Tensor] = []
        self.kv_cache_events: list[torch.cuda.Event] = []
        self._copy_stream: torch.cuda.Stream | None = None
        self._copy_in_progress: list[torch.cuda.Event] = []
        self._num_cpu_blocks = 0
        self._cpu_block_bytes = 0
        self._metadata = FissionKVMetadata()
        self._last_scheduled_max_block_id = -1
        self._elastic_pool = None          # set in register_kv_caches
        self.geo = None                    # PlaneGeometry, set in register_kv_caches
        self._elastic_mapped_ceil = 0
        self._elastic_last_max = -1
        self._elastic_span_base = 0

        self._native_ctx = None
        self._native_registered = False
        self._device = None

    # ------------------------------------------------------------------
    # resource setup
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self._copy_stream is None
        device = next(iter(kv_caches.values())).device if kv_caches else None
        # copy stream inherits device from the KV caches (a background
        # thread has no ambient current device; Stream(device_index=...)
        # pins it explicitly)
        self._copy_stream = (torch.cuda.Stream() if device is None
                             else torch.cuda.Stream(device=device))
        self._device = device
        if device is None:
            logger.warning("FissionKV: no GPU KV cache to register.")
            return

        # Strided uint8 planes over the SAME storage as the real cache
        # (never .contiguous(): a copy would detach swaps from the cache the
        # attention kernels read).
        self._gpu_tensors = build_planes(kv_caches)
        import os as _os
        if _os.environ.get("FISSION_KV_DEBUG_PLANES"):
            for _i, _t in enumerate(self._gpu_tensors):
                logger.info("FissionKV plane %d ptr=0x%x size=%s stride=%s "
                            "soff=%d storage=%d bytes", _i, _t.data_ptr(),
                            tuple(_t.shape), tuple(_t.stride()),
                            _t.storage_offset(),
                            _t.untyped_storage().nbytes())

        if not self._gpu_tensors:
            logger.warning("FissionKV: empty KV cache set.")
            return
        self.geo = plane_geometry(self._gpu_tensors)
        num_gpu_blocks = self.geo.num_blocks
        self._cpu_block_bytes = self.geo.cpu_block_bytes  # bytes/block
        self._gpu_block_stride = self.geo.block_stride
        self._gpu_layer_stride = self.geo.layer_stride
        if not self.geo.uniform_layer_stride:
            # Mixed-size plan (e.g. an MLA group plus another cache group, or
            # layers with different page sizes).  The dense path's
            # "(L-1)*layer_stride" span is then wrong, so grow uses the
            # per-plane exact span; tiering itself stays single-group (see
            # scheduler.py), so this is a layout-robustness guard.
            logger.warning(
                "FissionKV: KV planes do not share a uniform layout "
                "(layer_stride=%d, offsets=%s, block_strides=%s); elastic grow "
                "falls back to the per-plane exact span.",
                self.geo.layer_stride, self.geo.plane_offsets,
                self.geo.block_strides)

        # elastic mode (patch.py owns the pool): validate registration is
        # inside the VMM range and seed the mapped watermark.
        from . import patch as _patch
        pool = _patch.ELASTIC_POOL
        if pool is not None:
            lo, hi = pool.base, pool.base + pool.va_bytes
            p0 = self._gpu_tensors[0].data_ptr()
            outside = [t.data_ptr() for t in self._gpu_tensors
                       if not lo <= t.data_ptr() < hi]
            if outside:
                import traceback as _tb
                logger.warning(
                    "FissionKV elastic: plane-check failed pid=%d; caller "
                    "stack:\n%s", __import__("os").getpid(),
                    "".join(_tb.format_stack()[-6:-1]))
                raise RuntimeError(
                    "FissionKV: elastic pool active but KV planes live outside "
                    f"it (plane 0 at 0x{p0:x}, {len(outside)} plane(s) outside "
                    f"pool [{lo:x},{hi:x}).")
            self._elastic_pool = pool
            self._elastic_span_base = p0 - lo
            self._elastic_mapped_ceil = pool.mapped_bytes
            adv = self._elastic_span_base + self.geo.span_for(
                num_gpu_blocks - 1)
            logger.info("FissionKV worker: elastic pool attached (mapped %.2f "
                        "GiB of advertised %.1f GiB)",
                        pool.mapped_bytes / (1 << 30), adv / (1 << 30))

        budget = self.fissionkv_config.cpu_bytes_to_use
        n_cpu = budget // self._cpu_block_bytes if self._cpu_block_bytes else 0
        if self.vllm_config.cache_config.enable_prefix_caching:
            # mirror upstream: shrink CPU pool to free GPU blocks for
            # prefix-cache reuse
            shrink = (self.vllm_config.cache_config.num_gpu_blocks_override
                      or num_gpu_blocks) // 8 + self.spec_decode_extra_kv_tokens
            n_cpu = max(0, n_cpu - shrink)
            logger.info(
                "FissionKV: prefix caching enabled, CPU blocks reduced by %d",
                shrink)
        if n_cpu <= 0:
            logger.warning(
                "FissionKV: cpu_bytes_to_use=%d too small for block size %d.",
                budget, self._cpu_block_bytes)
        self._num_cpu_blocks = n_cpu

        # pinned host arena: one big allocation then per-plane strided views
        # (all planes share the arena; plane i block b lives at
        #  base + plane_off[i] + b * plane_stride[i]) - matches the flat
        # host_block_ids * cpu_block_bytes slot layout: a slot = one block
        # across ALL planes, so per-plane views have stride cpu_block_bytes.
        arena_bytes = n_cpu * self._cpu_block_bytes
        self._arena = self._alloc_pinned(arena_bytes)
        self._cpu_tensors = []
        offset = 0
        for t in self._gpu_tensors:
            plane_stride = t.shape[1]
            view = self._arena.narrow(0, offset, n_cpu * plane_stride)
            self._cpu_tensors.append(view.view(n_cpu, plane_stride))
            offset += n_cpu * plane_stride
        logger.info(
            "FissionKV worker: %d CPU blocks (%.2f GiB pinned), "
            "%d bytes/block, native=%s",
            n_cpu, arena_bytes / (1 << 30), self._cpu_block_bytes,
            HAVE_NATIVE)

        if HAVE_NATIVE:
            self._native_ctx = _C.CopyEngine()
            ok = True
            for gpu_t, cpu_t in zip(self._gpu_tensors, self._cpu_tensors):
                ok &= self._native_ctx.init_plane2(
                    gpu_t.data_ptr(), cpu_t.data_ptr(),
                    gpu_t.stride(0), cpu_t.stride(0), gpu_t.shape[1])
            self._native_registered = bool(ok)
            if not self._native_registered:
                logger.warning("FissionKV: native plane init failed; "
                               "using torch gather fallback")

        self._copy_thread = threading.Thread(
            target=self._copy_manager_loop, daemon=True,
            name="fissionkv-copy-manager")
        self._copy_thread.start()

    def _alloc_pinned(self, nbytes: int) -> torch.Tensor:
        if nbytes == 0:
            return torch.empty(0, dtype=torch.uint8)
        try:
            buf = torch.empty(nbytes, dtype=torch.uint8,
                              device="cpu", pin_memory=True)
            if HAVE_NATIVE:
                # confirm the caching allocator really pinned it
                _C.host_register(buf.data_ptr(), nbytes)
            return buf
        except (RuntimeError, OSError) as e:  # fall back: unpinned works too
            logger.warning("FissionKV: pin_memory failed (%s), using pageable.",
                           e)
            return torch.empty(nbytes, dtype=torch.uint8)

    @property
    def num_host_blocks(self) -> int:
        return self._num_cpu_blocks

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def bind_connector_metadata(self, m: FissionKVMetadata) -> None:
        self._metadata = m
        if m.need_flush:
            self._flush()
        if m.scheduled_max_block_id >= 0:
            self._last_scheduled_max_block_id = m.scheduled_max_block_id
            # grow VMM mapping HERE, not in start_load_kv: the V2 runner
            # defers start_load_kv to post_forward unless has_sync_kv_loads,
            # but bind always runs before this step's forward (the attention
            # kernel writes KV in place -> pages must be mapped already).
            self._grow_elastic(m.scheduled_max_block_id)
        if m.load_event >= 0 and m.load_gpu_blocks:
            self._register_load(m)
        if m.store_event >= 0 and m.store_gpu_blocks:
            self._register_store(m)

    def clear_connector_metadata(self) -> None:
        self._metadata = FissionKVMetadata()

    # elastic grow: called by the connector before forward so the attention
    # kernel can touch freshly advertised block ids (patch.py owns the pool).
    def grow_hint(self) -> int:
        return self._last_scheduled_max_block_id

    def _register_load(self, m: FissionKVMetadata) -> None:
        event = torch.cuda.Event()
        reqs: list[str] = []
        for s in m.load_event_to_reqs.get(m.load_event, set()):
            reqs.append(s)
        self._load_events[m.load_event] = (event, reqs)
        with self._copy_lock:
            self._copy_queue.append(_CopyJob(
                is_store=False, event_idx=m.load_event,
                stream_ptr=self._copy_stream.cuda_stream,
                event=event,
                gpu_block_ids=list(m.load_gpu_blocks),
                host_block_ids=list(m.load_host_blocks)))
            self._copy_cv.notify()

    def _register_store(self, m: FissionKVMetadata) -> None:
        event = torch.cuda.Event()
        reqs: list[str] = []
        seen: set[str] = set()
        for idx in m.store_blocks_to_reqs:
            r = m.store_blocks_to_reqs[idx]
            if r not in seen:
                seen.add(r)
                reqs.append(r)
        self._store_events[m.store_event] = (event, reqs)
        with self._copy_lock:
            self._copy_queue.append(_CopyJob(
                is_store=True, event_idx=m.store_event,
                stream_ptr=self._copy_stream.cuda_stream,
                event=event,
                gpu_block_ids=list(m.store_gpu_blocks),
                host_block_ids=list(m.store_host_blocks)))
            self._copy_cv.notify()

    # ------------------------------------------------------------------
    # forward-pass hooks
    # ------------------------------------------------------------------

    def start_load_kv(self) -> None:
        # elastic: map VMM chunks covering this step's block writes before
        # forward (attention writes KV in place; a fault is not recoverable)
        self._grow_elastic(self._metadata.scheduled_max_block_id)
        # loads are queued in bind_connector_metadata; nothing else here

    def _grow_elastic(self, max_block_id: int) -> None:
        if max_block_id < 0 or self._elastic_pool is None:
            return
        if max_block_id <= self._elastic_last_max:
            return
        # Sparse per-plane grow: map ONLY the chunks blocks 0..max_block_id
        # touch in EACH plane.  A contiguous span would map every layer
        # plane up to (L-1)*layer_stride on the FIRST block for
        # per-layer-contiguous caches (MLA/hybrid) - defeating elastic
        # boot.  Ranges pay ~one 2 MiB chunk per plane per growth step;
        # overlapping/duplicate ranges are no-ops.
        g = self.geo
        ranges = [
            (self._elastic_span_base + off, (max_block_id + 1) * stride)
            for off, stride in zip(g.plane_offsets, g.block_strides)
        ]
        newly = self._elastic_pool.ensure_ranges(ranges)
        if newly:
            logger.info("FissionKV elastic: grow sparse max_block=%d "
                        "planes=%d newly=%.1f MiB mapped=%.1f MiB",
                        max_block_id, len(ranges), newly / (1 << 20),
                        self._elastic_pool.mapped_bytes / (1 << 20))
        self._elastic_last_max = max(self._elastic_last_max, max_block_id)
        self._elastic_mapped_ceil = max(
            self._elastic_mapped_ceil, self._elastic_pool.mapped_bytes)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # v1: single per-step load event (docs §4.3); per-layer group
        # overlap is a v1.1 upgrade.
        if self._metadata.load_event >= 0:
            entry = self._load_events.get(self._metadata.load_event)
            if entry is not None:
                torch.cuda.current_stream().wait_event(entry[0])

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        # pin compute completion before the D2H reads the blocks
        if self._metadata.store_event < 0 or not self._metadata.store_gpu_blocks:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.kv_cache_events.append(event)
        with self._copy_lock:
            self._copy_cv.notify()

    # ------------------------------------------------------------------
    # copy manager (background thread)
    # ------------------------------------------------------------------

    def _copy_manager_loop(self) -> None:
        # this thread issues driver copies; bind it to the engine's device
        # (primary context) so driver VA resolution matches the engine.
        try:
            torch.cuda.set_device(self._gpu_tensors[0].device)
        except Exception:
            pass
        while not self._stop.is_set():
            with self._copy_cv:
                while (not self._copy_queue
                       and not self.kv_cache_events
                       and not self._stop.is_set()):
                    self._copy_cv.wait(timeout=0.05)
                if self._stop.is_set():
                    break
                jobs = self._copy_queue
                self._copy_queue = []
                if jobs:
                    # compute pins are consumed once per drained batch
                    compute_events = self.kv_cache_events
                    self.kv_cache_events = []
                else:
                    compute_events = []
            if not jobs:
                continue
            # The copy stream object was created on the worker (main)
            # thread: ExternalStream/`torch.cuda.Stream(stream_id=...)`
            # crashes in background threads on this torch ("Expected
            # stream_.device_type() == CUDA"), so use the stream object
            # directly (mirrors upstream simple_kv_offload, which passes
            # Stream objects into its copy thread).
            stream = self._copy_stream
            dev_ctx = (torch.cuda.device(self._device)
                       if self._device is not None else nullcontext())
            with dev_ctx:
                try:
                    with torch.cuda.stream(stream):
                        for job in jobs:
                            if job.is_store:
                                # GPU-side ordering: the D2H must not read
                                # the blocks before forward wrote them.
                                for ev in compute_events:
                                    stream.wait_event(ev)
                                if not compute_events:
                                    logger.debug(
                                        "FissionKV store without compute event; "
                                        "ordering relaxed")
                            self._run_batch(job)
                        done = torch.cuda.Event()
                        done.record(stream)
                        self._copy_in_progress.append(done)
                except Exception:
                    logger.exception("FissionKV copy manager failed")


    def _run_batch(self, job: _CopyJob) -> None:
        n = len(job.gpu_block_ids)
        if n == 0:
            job.event.record()
            return
        if self._native_registered:
            if self._elastic_pool is not None:
                # WSL driver quirk: driver-API copies of VMM-MAPPED pages
                # fail INVALID_VALUE (rc=1) unless the PRIMARY context is
                # current on the calling thread (plain cudaMalloc pages
                # work either way; the main thread always passes).  Make
                # it current per batch — thread-start binding is not
                # durable enough here.
                torch.zeros(1, device=self._gpu_tensors[0].device)
            # direction: load = H2D (src host slots -> dst gpu blocks);
            # store = D2H (src gpu blocks -> dst host slots). Run
            # coalescing inside the engine merges id-adjacent pairs.
            if job.is_store:
                src, dst, direction = (job.gpu_block_ids,
                                       job.host_block_ids, False)
            else:
                src, dst, direction = (job.host_block_ids,
                                       job.gpu_block_ids, True)
            for plane in range(self._native_ctx.num_planes):
                ops = self._native_ctx.copy_batch(plane, src, dst, direction,
                                                  job.stream_ptr)
                if ops < 0:
                    extra = ""
                    if self._elastic_pool is not None:
                        g = self._gpu_tensors[plane]
                        lo = self._elastic_pool.base
                        mapped = self._elastic_pool.mapped_bytes
                        bad = g.data_ptr() + int(src[0]) * g.stride(0)
                        extra = (f" plane_ptr=0x{bad:x} rel=0x{bad - lo:x} "
                                 f"mapped=0x{mapped:x} "
                                 f"{'IN' if bad - lo < mapped else 'OUT'} "
                                 f"of mapped")
                    # Runtime-API fallback: same DMA engine, but the
                    # runtime keeps the primary context current on this
                    # thread, sidestepping the driver-API quirk where
                    # cuMemcpyAsync on freshly VMM-mapped pages reports
                    # INVALID_VALUE from a thread without an attached ctx.
                    if self._runtime_copy(plane, src, dst, direction,
                                          job.stream_ptr):
                        logger.warning(
                            "FissionKV: driver copy failed on plane %d "
                            "(rc<0); used runtime fallback%s", plane, extra)
                        continue
                    raise RuntimeError(
                        f"FissionKV native copy failed on plane {plane}"
                        + extra)
            job.event.record()
            return

        # torch fallback: one index_select per plane (gather copy)
        dev = self._gpu_tensors[0].device
        gpu_idx = torch.tensor(job.gpu_block_ids, dtype=torch.long,
                               device="cpu").to(dev, non_blocking=True)
        host_idx = torch.tensor(job.host_block_ids, dtype=torch.long,
                                device="cpu").to(dev, non_blocking=True)
        if job.is_store:
            for gpu_t, cpu_t in zip(self._gpu_tensors, self._cpu_tensors):
                cpu_t.index_copy_(0, host_idx,
                                  gpu_t.index_select(0, gpu_idx))
        else:
            for gpu_t, cpu_t in zip(self._gpu_tensors, self._cpu_tensors):
                gpu_t.index_copy_(0, gpu_idx,
                                  cpu_t.index_select(0, host_idx))
        job.event.record()

    def _runtime_copy(self, plane, src, dst, direction, stream_ptr) -> bool:
        try:
            g = self._gpu_tensors[plane]
            h = self._cpu_tensors[plane]
            si = torch.as_tensor(list(src), dtype=torch.long,
                                 device=g.device)
            di = torch.as_tensor(list(dst), dtype=torch.long)
            stream = torch.cuda.streams.ExternalStream(stream_ptr,
                                                       device=g.device)
            with torch.cuda.stream(stream):
                if direction:  # H2D
                    g.index_copy_(0, si, h.index_select(0, di).to(g.device))
                else:  # D2H
                    h.index_copy_(0, di,
                                  g.index_select(0, si).to("cpu"))
            torch.cuda.synchronize(g.device)
            return True
        except Exception:
            logger.exception("FissionKV runtime copy fallback failed")
            return False


    def get_finished(self) -> tuple[set[str], set[str]]:
        """Poll copy events. Stores report through worker metadata counts
        (cached here); loads report request ids via finished_recving."""
        done_send, done_recv = self._poll_finished()
        for eid in done_send:
            self._completed_store_events[eid] = 1
        recving: set[str] = set()
        for reqs in done_recv.values():
            recving.update(reqs)
        return set(), recving

    def _poll_finished(self
                       ) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
        done_send: dict[int, set[str]] = {}
        done_recv: dict[int, set[str]] = {}
        if not self._query_done_in_batch():
            return done_send, done_recv
        for eid in list(self._store_events):
            ev, reqs = self._store_events[eid]
            if ev.query():
                done_send[eid] = set(reqs)
                del self._store_events[eid]
        for eid in list(self._load_events):
            ev, reqs = self._load_events[eid]
            if ev.query():
                done_recv[eid] = set(reqs)
                del self._load_events[eid]
        return done_send, done_recv

    def _query_done_in_batch(self) -> bool:
        if self._copy_in_progress:
            oldest = self._copy_in_progress[0]
            if not oldest.query():
                return False
            while (self._copy_in_progress
                   and self._copy_in_progress[0].query()):
                self._copy_in_progress.pop(0)
        return True

    def build_connector_worker_meta(self) -> FissionKVWorkerMetadata | None:
        # get_finished() runs first each step and fills _completed_store_events
        if not self._completed_store_events:
            return None
        meta = FissionKVWorkerMetadata(
            completed_store_events=self._completed_store_events)
        self._completed_store_events = {}
        return meta

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        while (self._copy_in_progress
               or self._load_events or self._store_events):
            self.get_finished()
            time.sleep(0.001)

    def handle_preemptions(self, kv_connector_metadata) -> None:
        if kv_connector_metadata.need_flush:
            self._flush()

    def shutdown(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._stop.set()
        with self._copy_cv:
            self._copy_cv.notify_all()
        if self._copy_thread is not None:
            self._copy_thread.join(timeout=5)
        if hasattr(self, "_arena") and self._arena is not None:
            if HAVE_NATIVE:
                try:
                    _C.host_unregister(self._arena.data_ptr())
                except Exception:  # pragma: no cover
                    pass
        self._arena = None
        self._native_ctx = None
