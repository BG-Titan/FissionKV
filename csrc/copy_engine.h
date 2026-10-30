// FissionKV copy engine: run-coalesced batch DMA between GPU block planes and
// host pinned planes, issued as driver memcpyAsync on a caller-provided
// stream (mirrors Strata kv_stream.cu resolve->copy approach, simplified to
// explicit paired id lists).
#pragma once

#include <cstdint>
#include <vector>

namespace fissionkv {

struct CopyEngine {
  struct Plane {
    uint64_t gpu_base = 0;    // device VA of plane (block 0)
    uint64_t host_base = 0;   // pinned host VA of plane (slot 0)
    uint64_t gpu_stride = 0;  // bytes between consecutive GPU blocks
    uint64_t host_stride = 0; // bytes between consecutive host slots
    uint64_t block_bytes = 0; // payload bytes copied per block
  };
  std::vector<Plane> planes;

  // stride form: same stride both sides, payload = stride (legacy/tests).
// full form: independent strides + payload (vllm planes are strided rows).
bool init_plane(uint64_t gpu_base, uint64_t host_base, uint64_t stride);
bool init_plane2(uint64_t gpu_base, uint64_t host_base, uint64_t gpu_stride,
                 uint64_t host_stride, uint64_t block_bytes);
  // direction: true = H2D (host->gpu), false = D2H.
  // src_ids[i]/dst_ids[i] are plane-local indices; consecutive pairs with
  // both sides incrementing by 1 are merged into one memcpy.
  // Returns number of memcpy operations issued, -1 on error.
  int64_t copy_batch(int plane_idx, const uint64_t* src_ids,
                     const uint64_t* dst_ids, size_t n, bool direction,
                     uint64_t stream_handle);
};

}  // namespace fissionkv
