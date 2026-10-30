# SPDX-License-Identifier: Apache-2.0
"""Pure copy-plan construction: run coalescing.

Port of Strata ``src/stream/kv_stream.cu::runs_of`` (docs §1.3 / §2.1):
consecutive block ids whose *both* sides are consecutive coalesce into a
single ``cudaMemcpyBatchAsync`` descriptor. A whole-prefix H2D load after a
fresh session start degenerates into exactly one run per layer, turning the
entire context restore into one DMA record - Strata's observation that a
full-prefix load is "one batch, one ring entry".

Pure python (unit-testable without torch); the C++ core applies the same
rule from ``fissionkv/csrc/copy_engine.cpp``.
"""

from __future__ import annotations


def runs_of(pairs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Coalesce (src, dst) block-id pairs into runs.

    Returns ``(src_start, dst_start, length)`` for maximal runs where both
    sides increment by 1. Input order is preserved; a pair that cannot join
    the previous run starts a new one. Matches Strata semantics: sorted
    unique inputs give the minimal descriptor count.
    """
    runs: list[tuple[int, int, int]] = []
    for src, dst in pairs:
        if runs:
            s0, d0, n = runs[-1]
            if s0 + n == src and d0 + n == dst:
                runs[-1] = (s0, d0, n + 1)
                continue
        runs.append((src, dst, 1))
    return runs


def copy_plan(
    pairs: list[tuple[int, int]],
    src_stride_bytes: int,
    dst_stride_bytes: int,
    src_base_offset_bytes: int = 0,
    dst_base_offset_bytes: int = 0,
    max_descriptor_bytes: int | None = None,
) -> list[tuple[int, int, int]]:
    """Byte-level descriptor plan for one tensor plane.

    For each coalesced run, emit ``(src_byte_off, dst_byte_off, nbytes)``
    relative to the plane bases (src_base/dst_base are added by the caller
    per layer). ``max_descriptor_bytes`` splits runs (memcpy has no hard
    limit, but smaller records keep batches schedulable alongside compute).
    """
    plan: list[tuple[int, int, int]] = []
    for s0, d0, n in runs_of(pairs):
        nbytes = n * src_stride_bytes
        if src_stride_bytes != dst_stride_bytes:
            raise ValueError(
                "FissionKV assumes equal src/dst block stride per plane "
                f"(got {src_stride_bytes} vs {dst_stride_bytes})")
        while nbytes and max_descriptor_bytes and nbytes > max_descriptor_bytes:
            take = max_descriptor_bytes - (max_descriptor_bytes % src_stride_bytes)
            take = take or src_stride_bytes
            k = take // src_stride_bytes
            plan.append((
                src_base_offset_bytes + s0 * src_stride_bytes,
                dst_base_offset_bytes + d0 * dst_stride_bytes,
                k * src_stride_bytes,
            ))
            s0 += k
            d0 += k
            nbytes -= k * src_stride_bytes
        if nbytes:
            plan.append((
                src_base_offset_bytes + s0 * src_stride_bytes,
                dst_base_offset_bytes + d0 * dst_stride_bytes,
                nbytes,
            ))
    return plan


def max_abs_delta(new_ids: list[int], old_ids: list[int]) -> int:
    """Strata ``max_abs_delta``: largest |new-old| over the common prefix -
    used for sanity checks that host chains track GPU block churn."""
    common = min(len(new_ids), len(old_ids))
    delta = 0
    for i in range(common):
        d = new_ids[i] - old_ids[i]
        if abs(d) > abs(delta):
            delta = d
    return delta
