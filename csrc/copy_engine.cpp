#include "copy_engine.h"
#include "driver_api.h"

#include <cstdio>
#include <unistd.h>

namespace fissionkv {

bool CopyEngine::init_plane(uint64_t gpu_base, uint64_t host_base,
                            uint64_t stride) {
  return init_plane2(gpu_base, host_base, stride, stride, stride);
}

bool CopyEngine::init_plane2(uint64_t gpu_base, uint64_t host_base,
                             uint64_t gpu_stride, uint64_t host_stride,
                             uint64_t block_bytes) {
  if (!gpu_base || !host_base || !gpu_stride || !host_stride ||
      !block_bytes || block_bytes > gpu_stride ||
      block_bytes > host_stride)
    return false;
  planes.push_back(
      Plane{gpu_base, host_base, gpu_stride, host_stride, block_bytes});
  return true;
}

int64_t CopyEngine::copy_batch(int plane_idx, const uint64_t* src_ids,
                               const uint64_t* dst_ids, size_t n,
                               bool direction, uint64_t stream_handle) {
  const DriverApi* d = driver();
  if (!d || plane_idx < 0 || plane_idx >= (int)planes.size()) return -1;
  const Plane& p = planes[plane_idx];
  CUstream stream = reinterpret_cast<CUstream>(stream_handle);
  int64_t ops = 0;
  size_t i = 0;
  while (i < n) {
    size_t j = i + 1;
    while (j < n && src_ids[j] == src_ids[j - 1] + 1 &&
           dst_ids[j] == dst_ids[j - 1] + 1) {
      ++j;
    }
    // Merging is only valid when BOTH sides advance by block_bytes (a
    // single memcpy of run*block_bytes is then exact). Padded rows
    // (vllm strided planes) must copy one block at a time.
    const bool mergeable =
        p.gpu_stride == p.block_bytes && p.host_stride == p.block_bytes;
    if (!mergeable) j = i + 1;
    const size_t run = j - i;
    const uint64_t s_off = src_ids[i] * p.gpu_stride;
    const uint64_t d_off = dst_ids[i] * p.host_stride;
    const size_t bytes = run * p.block_bytes;
    CUresult r;
    if (direction) {  // H2D: host src -> device dst
      r = d->memcpy_htod_async(
          static_cast<CUdeviceptr>(p.gpu_base + d_off),
          reinterpret_cast<const void*>(p.host_base + s_off), bytes, stream);
    } else {  // D2H: device src -> host dst
      r = d->memcpy_dtoh_async(
          reinterpret_cast<void*>(p.host_base + d_off),
          static_cast<CUdeviceptr>(p.gpu_base + s_off), bytes, stream);
    }
    if (r != CUDA_SUCCESS) {
      std::fprintf(stderr,
                   "[fissionkv] copy_batch plane=%d run@%zu len=%zu dir=%d "
                   "failed: %d (gpu_base=0x%lx host_base=0x%lx s_off=0x%lx "
                   "d_off=0x%lx bytes=%zu gs=%zu hs=%zu bb=%zu stream=0x%lx pid=%d)\n",
                   plane_idx, i, run, int(direction), r, p.gpu_base,
                   p.host_base, s_off, d_off, bytes, p.gpu_stride,
                   p.host_stride, p.block_bytes, stream_handle, (int)getpid());
      return -1;
    }
    ++ops;
    i = j;
  }
  return ops;
}

}  // namespace fissionkv
