#include "pin_arena.h"
#include "driver_api.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fissionkv {

namespace {
constexpr uint64_t kAlign = 2ull << 20;  // keep arenas 2MiB-aligned
}

PinnedArena* arena_create(uint64_t size_hint, const std::string& mode,
                          std::string* err) {
  const DriverApi* d = driver();
  if (!d) {
    *err = "cuda driver unavailable";
    return nullptr;
  }
  auto* a = new PinnedArena();
  size_hint = (size_hint + kAlign - 1) / kAlign * kAlign;

  bool try_file = (mode == "auto" || mode == "file");
  if (try_file) {
    const char* env = getenv("FISSION_KV_BACKING_DIR");
    std::string dir = env && *env ? env : "/dev/shm";
    a->path = dir + "/fissionkv_arena_" + std::to_string(getpid()) + ".bin";
    int fd = open(a->path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0) {
      if (posix_fallocate(fd, 0, static_cast<off_t>(size_hint)) == 0 ||
          ftruncate(fd, static_cast<off_t>(size_hint)) == 0) {
        void* p = mmap(nullptr, size_hint, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
        if (p != MAP_FAILED) {
          a->base = p;
          a->fd = fd;
          a->size = size_hint;
        }
      }
      if (!a->base) {
        close(fd);
        unlink(a->path.c_str());
      }
    }
  }
  if (!a->base) {
    if (mode == "file") {
      *err = "file-backed arena failed";
      delete a;
      return nullptr;
    }
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, size_hint) != 0 || !p) {
      *err = "posix_memalign failed";
      delete a;
      return nullptr;
    }
    a->base = p;
    a->size = size_hint;
  }

  // Driver-level pin (refcounted, safe with duplicates). Unpinned arenas
  // remain functional, just slower.
  CUresult r = d->mem_host_register(a->base, a->size, 0);
  a->registered = (r == CUDA_SUCCESS);
  if (!a->registered) {
    std::fprintf(stderr,
                 "[fissionkv] cuMemHostRegister failed (%d); arena left unpinned"
                 " (copies still valid, slower)\n",
                 r);
  }
  a->valid = true;
  return a;
}

void* arena_alloc(PinnedArena* a, uint64_t bytes) {
  if (!a || !a->valid) return nullptr;
  bytes = (bytes + 63) / 64 * 64;
  if (a->used + bytes > a->size) return nullptr;
  void* p = static_cast<char*>(a->base) + a->used;
  a->used += bytes;
  return p;
}

void arena_destroy(PinnedArena* a) {
  if (!a) return;
  const DriverApi* d = driver();
  if (a->registered && d) d->mem_host_unregister(a->base);
  if (a->fd >= 0) {
    munmap(a->base, a->size);
    close(a->fd);
    if (!a->path.empty()) unlink(a->path.c_str());
  } else {
    free(a->base);
  }
  delete a;
}

}  // namespace fissionkv
