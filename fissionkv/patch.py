# SPDX-License-Identifier: Apache-2.0
"""In-tree runtime hooks for elastic (feature B) mode - applied by plugin.

Zero source edits: everything below monkey-patches live vLLM objects when
``FISSION_KV_VA_BYTES`` is set (docs §4, L3 floor-port). Three hooks:

1. ``vllm.v1.worker.utils.allocate_kv_cache``
   The engine's single int8 backing buffer (all layers, one allocation -
   assert len(sizes)==1 in utils.py:592) is allocated from a FissionKV
   :class:`fissionkv.elastic.VmmPool` instead of ``torch.zeros``: a huge VA
   reservation with physical 2 MiB chunks mapped lazily. Strata
   ``kv_reserve`` analogue.

2. ``vllm.v1.core.block_pool.BlockPool.get_new_blocks``
   Tracks the max block id ever issued (``_fissionkv_max_issued``). The
   connector ships it to the worker as ``FissionKVMetadata.scheduled_max_block_id``
   so chunks covering this step's writes are mapped BEFORE forward
   (attention writes KV in-place; a fault is not recoverable).

3. ``vllm.Worker.determine_available_memory``
   ``FISSION_KV_KV_BYTES`` advertises a KV pool LARGER than physical VRAM
   (the point of the floor port): vLLM sizes num_blocks against the
   advertised figure; physical follows the grow-hint.

Safety: ``FISSION_KV_PHYS_INIT_BYTES`` physical is mapped eagerly at pool
creation; keep it above whatever the CUDA-graph capture warmup touches
(vLLM capture writes dummy KV into low block ids). Experimental; tiering
(feature A) runs without any of this.

Growth is monotone (Strata rule): chunks are never unmapped mid-decode;
``VmmPool.trim_to`` is an explicit operator action, not wired here.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os

from . import config as _config
from .elastic import HAVE_NATIVE, VmmPool, wrap_cuda_ptr

logger = logging.getLogger(__name__)

ELASTIC_POOL: VmmPool | None = None
_PATCHED = False



def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def elastic_enabled() -> bool:
    return _config.parse_bytes(_env("FISSION_KV_VA_BYTES", ""),
                               0) > 0


def kv_bytes_advertised() -> int:
    return _config.parse_bytes(_env("FISSION_KV_KV_BYTES", ""), 0)


def install() -> bool:
    """Apply all elastic hooks. Idempotent; returns True if active."""
    global _PATCHED
    if _PATCHED:
        return ELASTIC_POOL is not None or elastic_enabled()
    if not elastic_enabled():
        return False
    if not HAVE_NATIVE:
        raise RuntimeError(
            "FISSION_KV_VA_BYTES set but fissionkv native core (_C) is missing; "
            "cannot create VMM pools. Unset FISSION_KV_VA_BYTES or rebuild csrc.")
    _PATCHED = True
    _patch_block_pool()
    _patch_allocate_kv_cache()
    _patch_determine_available_memory()
    logger.info("FissionKV elastic mode active: VA=%s PHYS_INIT=%s KV_ADVERTISED=%s",
                _env("FISSION_KV_VA_BYTES"),
                _env("FISSION_KV_PHYS_INIT_BYTES"),
                _env("FISSION_KV_KV_BYTES"))
    return True


# ---------------------------------------------------------------- pool
def _make_pool(raw_size: int, device) -> VmmPool:
    global ELASTIC_POOL
    cfg = _config.FissionKVConfig.from_extra_config({})
    va = max(cfg.va_bytes, raw_size)
    # physical headroom at boot: explicit init plus a modest floor; the
    # connector grows the mapping before first touch of each block. Keep
    # the floor small: on GPU-P/vGPU partitions overcommit faults as
    # illegal access, and the real free-VRAM probe can lie.
    floor = min(raw_size, _config.parse_bytes(
        _env("FISSION_KV_ELASTIC_BOOT_FLOOR", ""), 512 << 20))
    phys = max(cfg.phys_init_bytes, floor)
    dev_id = device.index if device.index is not None else 0
    pool = VmmPool(va, phys, dev_id)
    ELASTIC_POOL = pool
    logger.info("FissionKV elastic pool: VA %.1f GiB, physical mapped %.2f GiB "
                "at base 0x%x (chunk %d MiB)",
                pool.va_bytes / (1 << 30), pool.mapped_bytes / (1 << 30),
                pool.base, pool.chunk_bytes >> 20)
    return pool


# ------------------------------------------------- 1. allocate_kv_cache
def _patch_allocate_kv_cache() -> None:
    """Wrap vllm's KV backing allocator to source it from an elastic VMM pool.

    Re-entrant: on repeat calls (e.g. from worker init after full vllm import)
    only the caller-name rebinding is redone.
    """
    import vllm.v1.worker.utils as wutils

    if not getattr(wutils.allocate_kv_cache, "_fissionkv_alloc", False):
        import torch

        orig = wutils.allocate_kv_cache

        @functools.wraps(orig)
        def patched(kv_cache_config, device, layout, kernel_block_sizes=None):
            tensors = getattr(kv_cache_config, "kv_cache_tensors", None)
            if not tensors:
                return orig(kv_cache_config, device, layout, kernel_block_sizes)
            sizes = {t.size for t in tensors}
            if len(sizes) != 1:
                # elastic contract expects the single-backing invariant
                logger.warning(
                    "FissionKV elastic: mixed backing sizes %s; stock allocation",
                    sizes)
                return orig(kv_cache_config, device, layout, kernel_block_sizes)
            raw_size = sizes.pop()
            import traceback as _tb
            logger.info("FissionKV elastic: allocate_kv_cache called from:\n%s",
                        "".join(_tb.format_stack()[-4:-1]))
            pool = _make_pool(raw_size, device)

            real_zeros = torch.zeros

            def zeros_like_hook(*shape, **kwargs):
                # intercept ONLY the engine's KV backing allocation
                if (kwargs.get("dtype") is not None
                        and kwargs["dtype"] == torch.int8
                        and shape == (raw_size,)):
                    dev_id = getattr(device, "index", None) or 0
                    t = wrap_cuda_ptr(pool.base, (raw_size,), "int8", dev_id)
                    logger.info(
                        "FissionKV elastic: wrapped tensor data_ptr=0x%x "
                        "pool.base=0x%x (pid %d)", t.data_ptr(), pool.base,
                        os.getpid())
                    # VMM memory is NOT zeroed at creation like torch.zeros:
                    # vLLM relies on zero-init for the null_block (block 0).
                    # Zero only the physically mapped prefix - writing beyond
                    # it would fault on unmapped VA chunks.
                    t.narrow(0, 0, pool.mapped_bytes).zero_()
                    logger.info("FissionKV elastic: KV backing %d bytes served "
                                "from pool at 0x%x (pid %d)", raw_size,
                                pool.base, os.getpid())
                    return t
                return real_zeros(*shape, **kwargs)

            torch.zeros = zeros_like_hook
            try:
                return orig(kv_cache_config, device, layout,
                            kernel_block_sizes)
            finally:
                torch.zeros = real_zeros

        patched._fissionkv_alloc = True
        wutils.allocate_kv_cache = patched
    patched = wutils.allocate_kv_cache

    # vllm imports the name directly in modules whose call sites matter
    # (attn_utils.py:42/:421, gpu_model_runner.py:229/:6935); rebind those
    # references too, or callers keep the original function object. Import
    # failures are tolerated (module may not exist on layout drift).
    for mod_name in ("vllm.v1.worker.gpu.attn_utils",
                     "vllm.v1.worker.gpu_model_runner"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # pragma: no cover - layout drift
            logger.info("FissionKV elastic: rebind skip %s (%s)", mod_name, exc)
            continue
        if getattr(mod, "allocate_kv_cache", None) is not patched:
            mod.allocate_kv_cache = patched
            logger.info("FissionKV elastic: rebound %s.allocate_kv_cache (pid %d)",
                        mod_name, os.getpid())


# ------------------------------------------------- 2. block id watermark
def _patch_block_pool() -> None:
    from vllm.v1.core.block_pool import BlockPool

    orig = BlockPool.get_new_blocks

    @functools.wraps(orig)
    def patched(self, num_blocks: int):
        ret = orig(self, num_blocks)
        if ret:
            top = max(b.block_id for b in ret)
            if top > getattr(self, "_fissionkv_max_issued", -1):
                self._fissionkv_max_issued = top
        return ret

    BlockPool.get_new_blocks = patched


# -------------------------------------- 3. advertise > physical KV pool
def _patch_determine_available_memory() -> None:
    target = kv_bytes_advertised()
    if target <= 0:
        return
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:  # pragma: no cover
        logger.warning("FissionKV: gpu_worker.Worker not importable; "
                       "KV advertisement patch skipped")
        return

    orig = Worker.determine_available_memory

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        free = orig(self, *args, **kwargs)
        if isinstance(free, (int, float)) and free < target:
            logger.warning(
                "FissionKV elastic: advertising KV memory %.1f GiB over %.1f "
                "GiB physical - mapping grows on demand",
                target / (1 << 30), float(free) / (1 << 30))
            return target
        return free

    Worker.determine_available_memory = patched
