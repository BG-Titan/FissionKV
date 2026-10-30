# SPDX-License-Identifier: Apache-2.0
"""Elastic GPU pool: reserve a huge VA, map physical chunks on demand.

Port of Strata ``VmmRange`` (docs §2.1) over the native core ``fissionkv._C``.
The worker can hand the engine DLPack tensors backed by VMM memory, so
``determine_available_memory`` may advertise a KV pool far larger than
physical VRAM: vLLM starts "small" physically and the mapping grows as
sessions fill blocks - the FissionKV floor-port's allocator half.

Growth rule (Strata): monotone at the watermark; unmapping happens only in
an explicit trim step, never mid-decode (docs §2.5).

``fissionkv._C`` is optional: without it, :data:`HAVE_NATIVE` is False and the
caller falls back to stock vLLM allocation.
"""

from __future__ import annotations

import ctypes
import math

try:
    from . import _C  # type: ignore[attr-defined]
    HAVE_NATIVE = True
except ImportError:  # pragma: no cover
    _C = None
    HAVE_NATIVE = False

_MIB = 1 << 20

# ---------------------------------------------------------------------------
# DLPack wrapping of raw device pointers (kDLCUDA), no libtorch build needed.
# ---------------------------------------------------------------------------

_K_CUDA = 2
_DL_CODE = {"int8": 0, "uint8": 1, "float16": 2, "float32": 2,
            "float64": 2, "bfloat16": 15}


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
        ("ndim", ctypes.c_uint),
        ("dtype_code", ctypes.c_uint8),
        ("dtype_bits", ctypes.c_uint8),
        ("dtype_lanes", ctypes.c_ushort),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


# Keep shape arrays alive for the capsule's lifetime (kept for process life;
# KV tensors are never freed in vLLM anyway).
_keepalive: list = []


def wrap_cuda_ptr(ptr: int, shape: tuple[int, ...], dtype: str,
                  device_id: int):
    """Wrap raw device memory as a torch.Tensor via a hand-built DLPack
    capsule. The caller (worker) must keep the backing pool alive."""
    import torch  # local import: keeps this module torch-free at import
    if dtype not in _DL_CODE:
        raise ValueError(f"unsupported dtype {dtype}")
    itemsize = torch.empty(0, dtype=getattr(torch, dtype)).element_size()
    total = itemsize
    for s in shape:
        total *= s
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    _keepalive.append(shape_arr)
    dl = _DLTensor()
    dl.data = ctypes.c_void_p(ptr)
    dl.device_type = _K_CUDA
    dl.device_id = device_id
    dl.ndim = len(shape)
    dl.dtype_code = _DL_CODE[dtype]
    dl.dtype_bits = itemsize * 8
    dl.dtype_lanes = 1
    dl.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    dl.strides = ctypes.POINTER(ctypes.c_int64)()  # C-contiguous
    dl.byte_offset = 0
    _keepalive.append(dl)
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.byref(dl), ctypes.c_char_p(b"dltensor"), None)
    if not capsule:
        raise MemoryError("PyCapsule_New failed")
    from torch.utils import dlpack as torch_dlpack
    t = torch_dlpack.from_dlpack(capsule)
    return t.reshape(shape) if tuple(t.shape) != tuple(shape) else t


class VmmPool:
    """Chunk-mapped VA range with monotone ``ensure_covered`` growth."""

    def __init__(self, va_bytes: int, phys_init_bytes: int = 0,
                 device_id: int = 0):
        if not HAVE_NATIVE:
            raise RuntimeError(
                "fissionkv native core (_C) unavailable; cannot allocate VMM "
                "memory. Rebuild csrc, or unset FISSION_KV_VA_BYTES to run with "
                "stock allocation.")
        if va_bytes <= 0:
            raise ValueError("va_bytes must be > 0")
        self.device_id = device_id
        self._pool = _C.VmmPool.create(int(va_bytes),
                                       int(phys_init_bytes), device_id)
        st = self._pool.stats()
        self.chunk_bytes = st["chunk"]
        self.va_bytes = st["va_size"]
        self._mapped = st["phys_mapped"]

    @classmethod
    def attach(cls, handle: int, base: int, size: int, phys_bytes: int,
               device_id: int) -> "VmmPool | None":
        if not HAVE_NATIVE:
            return None
        pool = _C.VmmPool.attach(handle, base, size, phys_bytes, device_id)
        if pool is None:
            return None
        obj = cls.__new__(cls)
        obj._pool = pool
        st = pool.stats()
        obj.chunk_bytes = st["chunk"]
        obj.va_bytes = st["va_size"]
        obj.device_id = device_id
        obj._mapped = st["phys_mapped"]
        return obj

    @property
    def base(self) -> int:
        return self._pool.stats()["base"]

    @property
    def mapped_bytes(self) -> int:
        st = self._pool.stats()
        self._mapped = st["phys_mapped"]
        return self._mapped

    def ensure_covered(self, upto_bytes: int) -> int:
        """Map chunks so [0, upto_bytes) is accessible. Returns *newly*
        mapped bytes (0 == no-op, the hot path)."""
        if upto_bytes <= self._mapped:
            return 0
        before = self._mapped
        self._mapped = self._pool.ensure_physical(int(upto_bytes))
        return max(self._mapped - before, 0)

    def ensure_ranges(self, ranges) -> int:
        """Sparse grow: map only chunks overlapping the given
        (offset, length) byte ranges relative to the pool base. Returns
        *newly* mapped bytes."""
        before = self._mapped
        self._mapped = self._pool.ensure_ranges(
            [(int(o), int(n)) for o, n in ranges])
        return max(self._mapped - before, 0)

    def ensure_covered_blocks(self, max_block_id: int,
                              block_stride_bytes: int) -> int:
        return self.ensure_covered((int(max_block_id) + 1)
                                   * block_stride_bytes)

    def trim_to(self, keep_bytes: int, pinned_tail_blocks: int = 0,
                block_bytes: int = 0) -> int:
        """Explicit sweep; never on the decode hot path (docs §2.5)."""
        before = self._mapped
        self._mapped = self._pool.shrink_physical(
            int(keep_bytes), int(pinned_tail_blocks), int(block_bytes))
        return max(before - self._mapped, 0)

    def close(self) -> None:
        # shared_ptr destructor in the native module releases the range;
        # drop our reference so it runs even before interpreter shutdown.
        self._pool = None


def driver_free_bytes(device_id: int = 0) -> int:
    """Physically free device memory (bytes), -1 without the native core."""
    if not HAVE_NATIVE:
        return -1
    return _C.free_physical_bytes(device_id)
