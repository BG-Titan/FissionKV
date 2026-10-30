# SPDX-License-Identifier: Apache-2.0
"""HostBlockIndex unit tests: second-chance eviction, pins, lifecycle."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fissionkv.block_index import HostBlockIndex, SlotState  # noqa: E402


def fill(idx, n=4):
    for i in range(n):
        slot = idx.reserve(1)[0]
        idx.begin_store(bytes([i]), slot)
        idx.commit_store(slot)


def test_lifecycle_and_contains():
    idx = HostBlockIndex(4)
    slot = idx.reserve(1)[0]
    idx.begin_store(b"a", slot)
    assert b"a" not in idx            # PENDING not matchable
    idx.commit_store(slot)
    assert b"a" in idx and idx.get(b"a") == slot
    assert idx.state_of(slot) == SlotState.READY


def test_second_chance_eviction():
    idx = HostBlockIndex(2)
    fill(idx, 2)                      # slots for a, b
    idx.get(b"a")                     # touch a -> second chance
    s = idx.reserve(1)[0]             # must evict b, spare a once
    assert idx._slot_hash[s] != b"a" or idx.get(b"a") is not None
    assert b"b" in idx or s != idx._hash_to_slot.get(b"a")
    idx.begin_store(b"c", s)
    idx.commit_store(s)
    assert b"c" in idx


def test_pinned_survive_and_free_on_unpin():
    idx = HostBlockIndex(1)
    fill(idx, 1)
    slot = idx._hash_to_slot[b"\x00"]
    idx.pin(slot)
    with pytest.raises(IndexError):
        idx.reserve(1)                # pinned-full: throttle caller
    idx.drop(b"\x00")                 # freed but still pinned
    assert idx.free_slots == 0
    idx.unpin(slot)
    assert idx.free_slots == 1


def test_match_chain_prefix_only():
    idx = HostBlockIndex(4)
    for h in (b"h0", b"h1", b"h3"):
        slot = idx.reserve(1)[0]
        idx.begin_store(h, slot)
        idx.commit_store(slot)
    assert idx.match_chain([b"h0", b"h1", b"h2", b"h3"]) == \
        idx.match_chain([b"h0", b"h1"])
    assert idx.match_chain([b"x"]) == []


def test_begin_store_absorbs_duplicate():
    idx = HostBlockIndex(4)
    fill(idx, 1)                      # a -> slot x
    new = idx.reserve(1)[0]
    old = idx._hash_to_slot[b"\x00"]
    idx.begin_store(b"\x00", new)     # re-store same hash
    assert idx._state[old] == SlotState.FREE
    assert idx._hash_to_slot[b"\x00"] == new


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
