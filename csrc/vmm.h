// Elastic VA pool over the CUDA driver API. Modeled on Strata
// src/core/vmm.cpp: reserve one VA range, map physical chunks lazily at a
// 2 MiB granularity watermark, unmap on shrink.
#pragma once

#include <cstdint>
#include <vector>

namespace fissionkv {

struct VmmPool {
  uint64_t base = 0;         // device VA base
  uint64_t va_size = 0;      // reserved VA bytes
  uint64_t chunk = 0;        // physical granularity (multiple of 2 MiB)
  uint64_t phys_bytes = 0;   // currently mapped bytes (watermark)
  int device_id = 0;
  // true once the initial mapping was created from THIS process' primary
  // context (i.e. not adopted); adopted pools must not shrink.
  bool managed = false;
  bool adopted = false;      // attached to an existing external pool
  std::vector<uint64_t> handles;  // per chunk; 0 = unmapped
  bool valid = false;

  uint64_t phys_chunks() const {
    return phys_bytes ? phys_bytes / chunk : 0;
  }
};

// Attach to a pre-existing externally-created VMM range (e.g. a pool made
// by another module in this process before torch init). Returns false if
// the handle's location/device doesn't match device_id.
bool vmm_attach(VmmPool& pool, uint64_t handle_u64, uint64_t base,
                uint64_t size, uint64_t phys_bytes, int device_id);

// Ensure the pool maps at least `target_bytes` of physical memory.
// Returns mapped bytes after the call (may exceed target, chunk-aligned).
uint64_t vmm_ensure_physical(VmmPool& pool, uint64_t target_bytes);

// Sparse growth: map every chunk overlapping any [offset, offset+len)
// range (offsets relative to pool base). Already-mapped chunks are
// skipped, so planes laid out at distant offsets (per-layer contiguous
// caches) only pay for bytes actually touched. Returns mapped bytes.
uint64_t vmm_ensure_ranges(
    VmmPool& pool, const std::vector<std::pair<uint64_t, uint64_t>>& ranges);

// Release physical chunks above `target_bytes` (unmap+release). Refuses to
// unmap a tail chunk whose first block is still pinned. Returns mapped
// bytes after the call.
uint64_t vmm_shrink_physical(VmmPool& pool, uint64_t target_bytes,
                             uint64_t pinned_tail_blocks,
                             uint64_t block_bytes);

void vmm_destroy(VmmPool& pool);

// Returns pinned-device-memory bytes for the current device
// (cuMemGetInfo free/total); -1 if unavailable.
int64_t driver_free_physical_bytes(int device_id);

}  // namespace fissionkv
