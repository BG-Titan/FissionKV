# SPDX-License-Identifier: Apache-2.0
"""Pure copy-plan coalescing tests (no torch/GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fissionkv.plan import copy_plan, max_abs_delta, runs_of


def test_full_prefix_is_single_run():
    # fresh session restore: consecutive gpu ids <- consecutive host slots
    pairs = [(g, h) for g, h in zip(range(40, 52), range(7, 19))]
    assert runs_of(pairs) == [(40, 7, 12)]


def test_gaps_split_runs():
    pairs = [(0, 5), (1, 6), (9, 6), (10, 7), (11, 8), (12, 2)]
    assert runs_of(pairs) == [(0, 5, 2), (9, 6, 3), (12, 2, 1)]


def test_dst_gap_splits_even_if_src_contiguous():
    pairs = [(0, 0), (1, 1), (2, 7)]
    assert runs_of(pairs) == [(0, 0, 2), (2, 7, 1)]


def test_empty():
    assert runs_of([]) == []


def test_copy_plan_offsets_and_bases():
    plan = copy_plan([(2, 4)], 1024, 1024,
                     src_base_offset_bytes=64, dst_base_offset_bytes=128)
    assert plan == [(2 * 1024 + 64, 4 * 1024 + 128, 1024)]


def test_copy_plan_stride_mismatch_rejected():
    try:
        copy_plan([(0, 0)], 1024, 512)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "stride" in str(e)


def test_copy_plan_descriptor_split():
    # run of 10 blocks, 4096 B each; split at 8192-byte records
    pairs = [(i, i) for i in range(10)]
    plan = copy_plan(pairs, 4096, 4096, max_descriptor_bytes=8192)
    assert len(plan) == 5
    assert all(nbytes == 8192 for _, _, nbytes in plan)
    assert [s for s, _, _ in plan] == [0, 8192, 16384, 24576, 32768]


def test_copy_plan_split_non_divisor_cap():
    # 4096-byte blocks, cap 6000 -> take = 6000 - (6000 % 4096) = 4096,
    # so one 4096-byte block per record (3 blocks -> 3 records)
    pairs = [(i, i) for i in range(3)]
    plan = copy_plan(pairs, 4096, 4096, max_descriptor_bytes=6000)
    assert plan == [(0, 0, 4096), (4096, 4096, 4096), (8192, 8192, 4096)]


def test_max_abs_delta():
    assert max_abs_delta([5, 6, 7], [1, 2, 3]) == 4
    assert max_abs_delta([1, 2], [1, 9]) == -7    # signed, largest |delta|
    assert max_abs_delta([1, 2], [1]) == 0        # min-length prefix only
    assert max_abs_delta([], [1]) == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
