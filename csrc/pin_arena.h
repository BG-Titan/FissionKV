// Host pinned-memory arena + O_DIRECT twin, modeled on Strata
// src/cuda/pin_arena.cpp: bump allocation over aligned malloc, driver-level
// cuMemHostRegister (keeps refcounts so duplicates are safe), optional
// O_DIRECT backing file that cuMemHostRegister can map directly (PCIe
// mapping flag) so staging copies bypass the page cache.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fissionkv {

struct PinnedArena {
  void* base = nullptr;
  uint64_t size = 0;
  uint64_t used = 0;
  bool registered = false;
  int fd = -1;          // O_DIRECT backing file, -1 if malloc-backed
  std::string path;     // backing file path (kept for cleanup), else tmpdir
  bool valid = false;
};

// mode: "auto" | "file" | "malloc". size_hint bytes requested.
// Returns nullptr on failure (reason written to err).
PinnedArena* arena_create(uint64_t size_hint, const std::string& mode,
                          std::string* err);
// Bump-allocate aligned region; nullptr when exhausted.
void* arena_alloc(PinnedArena* a, uint64_t bytes);
void arena_destroy(PinnedArena* a);

}  // namespace fissionkv
