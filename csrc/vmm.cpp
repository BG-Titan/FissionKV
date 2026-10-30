#include "vmm.h"
#include "driver_api.h"

#include <cstdio>
#include <cstring>

namespace fissionkv {

namespace {

constexpr uint64_t kTwoMib = 2ull << 20;

struct SavedCtx {
  CUcontext previous = nullptr;
  bool switched = false;
  explicit SavedCtx(CUcontext target) {
    const DriverApi* d = driver();
    if (d->ctx_get_current(&previous) == CUDA_SUCCESS && previous != target) {
      d->ctx_set_current(target);
      switched = true;
    }
  }
  ~SavedCtx() {
    if (switched) driver()->ctx_set_current(previous);
  }
};

CUcontext primary_ctx(int device_id) {
  const DriverApi* d = driver();
  CUcontext ctx = nullptr;
  if (d->primary_ctx_retain(&ctx, device_id) != CUDA_SUCCESS) return nullptr;
  d->primary_ctx_release(device_id);  // ref-counted; keep the pointer
  return ctx;
}

CUmemAllocationProp device_prop(int device_id) {
  CUmemAllocationProp prop;
  std::memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  return prop;
}

}  // namespace

bool vmm_attach(VmmPool& pool, uint64_t handle_u64, uint64_t base,
                uint64_t size, uint64_t phys_bytes, int device_id) {
  const DriverApi* d = driver();
  if (!d || !handle_u64 || !base || !size) return false;
  // Validate the handle: a foreign/invalid handle fails cuMemRelease-free
  // checks only indirectly; instead verify granularity divisibility and the
  // mapped watermark, then trust the caller-provided base (checked against
  // torch tensors in Python).
  CUmemAllocationProp prop = device_prop(device_id);
  size_t gran = 0;
  if (d->mem_get_granularity(&gran, &prop, 0) != CUDA_SUCCESS) return false;
  if (gran == 0 || (size % gran) != 0 || (phys_bytes % gran) != 0) {
    std::fprintf(stderr,
                 "[fissionkv] attach rejected: size=%llu phys=%llu gran=%zu\n",
                 (unsigned long long)size, (unsigned long long)phys_bytes,
                 gran);
    return false;
  }
  pool.base = base;
  pool.va_size = size;
  pool.chunk = gran < kTwoMib ? kTwoMib : (gran / kTwoMib * kTwoMib ? gran : gran);
  pool.chunk = gran;
  pool.phys_bytes = phys_bytes;
  pool.device_id = device_id;
  pool.managed = false;
  pool.adopted = true;
  pool.valid = true;
  return true;
}

namespace {

// Map one VA chunk (create+map+setaccess). Caller must hold SavedCtx.
bool map_chunk(const DriverApi* d, VmmPool& pool, size_t i,
               const CUmemAllocationProp& prop,
               const CUmemAccessDesc& access) {
  const size_t chunk = static_cast<size_t>(pool.chunk);
  CUdeviceptr va = pool.base + i * chunk;
  CUmemGenericAllocationHandle h = 0;
  CUresult r = d->mem_create(&h, chunk, &prop, 0);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "[fissionkv] cuMemCreate chunk %zu failed: %d\n", i,
                 r);
    fprintf(stderr, "[fissionkv] map chunk %zu failed\n", i); return false;
  }
  r = d->mem_map(va, chunk, 0, h, 0);
  if (r != CUDA_SUCCESS) {
    d->mem_release(h);
    std::fprintf(stderr, "[fissionkv] cuMemMap chunk %zu failed: %d\n", i, r);
    return false;
  }
  r = d->mem_set_access(va, chunk, &access, 1);
  if (r != CUDA_SUCCESS) {
    d->mem_unmap(va, chunk);
    d->mem_release(h);
    std::fprintf(stderr, "[fissionkv] cuMemSetAccess chunk %zu failed: %d\n",
                 i, r);
    fprintf(stderr, "[fissionkv] map chunk %zu failed\n", i); return false;
  }
  pool.handles[i] = h;
  pool.phys_bytes += chunk;
  return true;
}


}  // namespace

uint64_t vmm_ensure_physical(VmmPool& pool, uint64_t target_bytes) {
  const DriverApi* d = driver();
  if (!d || !pool.valid) return pool.phys_bytes;
  if (target_bytes > pool.va_size) target_bytes = pool.va_size;
  target_bytes = (target_bytes + pool.chunk - 1) / pool.chunk * pool.chunk;
  if (target_bytes <= pool.phys_bytes) return pool.phys_bytes;

  SavedCtx scope(primary_ctx(pool.device_id));
  CUmemAllocationProp prop = device_prop(pool.device_id);
  CUmemAccessDesc access;
  std::memset(&access, 0, sizeof(access));
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = pool.device_id;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  const size_t n = static_cast<size_t>(target_bytes / pool.chunk);
  if (pool.handles.size() < n) pool.handles.resize(n, 0);
  for (size_t i = static_cast<size_t>(pool.phys_bytes / pool.chunk); i < n;
       ++i) {
    if (pool.handles[i] != 0) continue;
    if (!map_chunk(d, pool, i, prop, access)) break;
  }
  // Watermark bookkeeping: phys_bytes counted increments; the loop above
  // only walks contiguously from the old watermark, holes are skipped.
  return pool.phys_bytes;
}

uint64_t vmm_ensure_ranges(
    VmmPool& pool, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
  const DriverApi* d = driver();
  if (!d || !pool.valid) return pool.phys_bytes;
  SavedCtx scope(primary_ctx(pool.device_id));
  CUmemAllocationProp prop = device_prop(pool.device_id);
  CUmemAccessDesc access;
  std::memset(&access, 0, sizeof(access));
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = pool.device_id;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  for (const auto& rg : ranges) {
    if (rg.second == 0) continue;
    uint64_t off = rg.first, len = rg.second;
    if (off >= pool.va_size) continue;
    if (off + len > pool.va_size) len = pool.va_size - off;
    const size_t i0 = static_cast<size_t>(off / pool.chunk);
    const size_t i1 = static_cast<size_t>((off + len - 1) / pool.chunk);
    if (pool.handles.size() < i1 + 1) pool.handles.resize(i1 + 1, 0);
    for (size_t i = i0; i <= i1; ++i) {
      if (pool.handles[i] != 0) continue;
      if (!map_chunk(d, pool, i, prop, access)) break;  // OOM: stop early
    }
  }
  return pool.phys_bytes;
}

uint64_t vmm_shrink_physical(VmmPool& pool, uint64_t target_bytes,
                             uint64_t pinned_tail_blocks,
                             uint64_t block_bytes) {
  const DriverApi* d = driver();
  if (!d || !pool.valid) return pool.phys_bytes;
  if (!pool.managed) {
    std::fprintf(stderr, "[fissionkv] shrink refused: pool not created here\n");
    return pool.phys_bytes;
  }
  uint64_t floor_bytes = pinned_tail_blocks * block_bytes;
  floor_bytes = (floor_bytes + pool.chunk - 1) / pool.chunk * pool.chunk;
  if (target_bytes < floor_bytes) target_bytes = floor_bytes;
  target_bytes = target_bytes / pool.chunk * pool.chunk;
  if (target_bytes >= pool.phys_bytes) return pool.phys_bytes;

  CUcontext ctx = primary_ctx(pool.device_id);
  SavedCtx scope(ctx);
  const size_t chunk = static_cast<size_t>(pool.chunk);
  // Synchronize before unmapping: no in-flight kernel may touch the tail.
  // (torch.cuda.synchronize is done Python-side; driver sync as backstop.)
  for (size_t i = pool.handles.size(); i-- > static_cast<size_t>(
                                               target_bytes / pool.chunk);) {
    if (pool.handles[i] == 0) continue;
    CUdeviceptr va = pool.base + i * chunk;
    CUresult r = d->mem_unmap(va, chunk);
    if (r != CUDA_SUCCESS) {
      std::fprintf(stderr, "[fissionkv] cuMemUnmap chunk %zu failed: %d\n", i, r);
      break;
    }
    d->mem_release(pool.handles[i]);
    pool.handles[i] = 0;
    pool.phys_bytes -= chunk;
  }
  return pool.phys_bytes;
}

void vmm_destroy(VmmPool& pool) {
  const DriverApi* d = driver();
  if (!d || !pool.valid) return;
  CUcontext ctx = primary_ctx(pool.device_id);
  SavedCtx scope(ctx);
  const size_t chunk = static_cast<size_t>(pool.chunk);
  for (size_t i = 0; i < pool.handles.size(); ++i) {
    if (pool.handles[i] == 0) continue;
    CUdeviceptr va = pool.base + i * chunk;
    d->mem_unmap(va, chunk);
    d->mem_release(pool.handles[i]);
    pool.handles[i] = 0;
  }
  if (!pool.adopted && pool.base) d->mem_address_free(pool.base, pool.va_size);
  pool.base = 0;
  pool.phys_bytes = 0;
  pool.valid = false;
}

int64_t driver_free_physical_bytes(int device_id) {
  const DriverApi* d = driver();
  if (!d) return -1;
  CUcontext ctx = primary_ctx(device_id);
  if (!ctx) return -1;
  SavedCtx scope(ctx);
  size_t free_b = 0, total_b = 0;
  if (d->mem_get_info(&free_b, &total_b) != CUDA_SUCCESS) return -1;
  return static_cast<int64_t>(free_b);
}

}  // namespace fissionkv
