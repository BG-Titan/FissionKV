# SPDX-License-Identifier: Apache-2.0
"""Host-side block index: hash -> pinned host slot.

Pure python so it is unit-testable without torch/vLLM/CUDA; used by the
scheduler-side manager. Eviction is clock/second-chance (Strata sweep
analogue, docs §1.2): touched entries survive one pass, matching the
`second_chance` behaviour that keeps hot prefixes from thrashing.

Slots are indices into the worker's pinned KV arena. One slot == one GPU
KV block == one page of ``block_size`` tokens, laid out as the flattened
concatenation of all KV tensors' block stride (see ``slot_bytes``).
"""

from __future__ import annotations

from collections.abc import Sequence


class SlotState:
    FREE = 0
    PENDING = 1   # store in flight; not readable for prefix matches yet
    READY = 2     # host copy valid


class HostBlockIndex:
    """Fixed-capacity map from block hash to host slot with second-chance LRU.

    Hash keys are opaque bytes (vLLM's block hashes). Chain lookups follow
    prefix order: an entry only counts if its whole prefix hit chain does;
    the caller supplies chains oldest-first.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        self._hash_to_slot: dict[bytes, int] = {}
        self._slot_hash: list[bytes | None] = [None] * capacity
        self._state = bytearray(capacity)
        self._second_chance = bytearray(capacity)
        self._pins = [0] * capacity
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        # clock hand + insertion-order ring for candidates
        self._hand = 0
        self._lru: list[int] = []  # slots in READY order, oldest first (heap-free)

    # ---------- lookup ----------

    def __contains__(self, block_hash: bytes) -> bool:
        slot = self._hash_to_slot.get(block_hash)
        return slot is not None and self._state[slot] == SlotState.READY

    def get(self, block_hash: bytes) -> int | None:
        """READY slot for hash, touching it, or None."""
        slot = self._hash_to_slot.get(block_hash)
        if slot is None or self._state[slot] != SlotState.READY:
            return None
        self.touch(slot)
        return slot

    def match_chain(self, chain: Sequence[bytes]) -> list[int]:
        """Slots for the longest READY prefix of ``chain`` (oldest first)."""
        slots: list[int] = []
        for h in chain:
            slot = self._hash_to_slot.get(h)
            if slot is None or self._state[slot] != SlotState.READY:
                break
            slots.append(slot)
        # touch after the walk so one miss does not half-age the chain
        for slot in slots:
            self.touch(slot)
        return slots

    # ---------- allocation ----------

    def reserve(self, n: int) -> list[int]:
        """Pick n slots for new stores, evicting as needed. Pinned slots are
        never evicted; raises ``IndexError`` if capacity is pinned-full
        (caller throttles stores until acks arrive)."""
        out: list[int] = []
        for _ in range(n):
            if self._free:
                slot = self._free.pop()
            else:
                slot = self._evict_one()
            out.append(slot)
        return out

    def _evict_one(self) -> int:
        # Second-chance clock over READY slots; PENDING slots are also
        # eligible (store not acked yet - evicting wastes an in-flight copy
        # but is safe: the store's completion just no-ops the mapping).
        limit = self.capacity * 2
        for _ in range(limit):
            slot = self._hand
            self._hand = (self._hand + 1) % self.capacity
            state = self._state[slot]
            if state == SlotState.FREE:  # race-free bookkeeping keeps this rare
                continue
            if self._pins[slot]:
                continue
            if self._second_chance[slot]:
                self._second_chance[slot] = 0
                continue
            self._unmap_slot(slot)
            return slot
        raise IndexError("fissionkv host index is pinned-full; retry later")

    def _unmap_slot(self, slot: int) -> None:
        h = self._slot_hash[slot]
        if h is not None:
            if self._hash_to_slot.get(h) == slot:
                del self._hash_to_slot[h]
            self._slot_hash[slot] = None
        self._state[slot] = SlotState.FREE
        self._second_chance[slot] = 0
        try:
            self._lru.remove(slot)
        except ValueError:
            pass

    # ---------- mapping lifecycle ----------

    def begin_store(self, block_hash: bytes, slot: int) -> None:
        """Caller reserved ``slot``; record intent before issuing the D2H."""
        old = self._slot_hash[slot]
        if old is not None and self._hash_to_slot.get(old) == slot:
            del self._hash_to_slot[old]
        # Replace/absorb an existing mapping for the target hash (re-store).
        prev = self._hash_to_slot.get(block_hash)
        if prev is not None and prev != slot:
            self._free_slot(prev)
        self._slot_hash[slot] = block_hash
        self._hash_to_slot[block_hash] = slot
        self._state[slot] = SlotState.PENDING
        self._second_chance[slot] = 0

    def commit_store(self, slot: int) -> None:
        if self._state[slot] == SlotState.PENDING:
            self._state[slot] = SlotState.READY
            self._lru.append(slot)

    def abort_store(self, slot: int, block_hash: bytes) -> None:
        """Store could not complete (e.g. GPU block died first)."""
        if (self._state[slot] == SlotState.PENDING
                and self._slot_hash[slot] == block_hash):
            self._free_slot(slot)

    def drop(self, block_hash: bytes) -> None:
        slot = self._hash_to_slot.get(block_hash)
        if slot is not None:
            self._free_slot(slot)

    def _free_slot(self, slot: int) -> None:
        self._unmap_slot(slot)
        if not self._pins[slot]:
            self._free.append(slot)
        else:
            # pinned-but-free: released pins will append it
            self._state[slot] = SlotState.FREE

    # ---------- pins & aging ----------

    def touch(self, slot: int) -> None:
        if self._state[slot] == SlotState.READY:
            self._second_chance[slot] = 1

    def pin(self, slot: int) -> None:
        self._pins[slot] += 1

    def unpin(self, slot: int) -> None:
        assert self._pins[slot] > 0, "unbalanced unpin"
        self._pins[slot] -= 1
        if (self._pins[slot] == 0
                and self._state[slot] == SlotState.FREE):
            self._free.append(slot)

    # ---------- introspection ----------

    def state_of(self, slot: int) -> int:
        return self._state[slot]

    @property
    def size(self) -> int:
        return len(self._hash_to_slot)

    @property
    def ready(self) -> int:
        return sum(1 for s in self._state if s == SlotState.READY)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    @property
    def pinned(self) -> int:
        return sum(1 for c in self._pins if c > 0)
