// Minimal CUDA driver API surface used by FissionKV, resolved at runtime via
// dlopen("libcuda.so.1") + dlsym -- same approach as Strata
// (src/core/vmm.cpp), so the extension never links against libcuda and
// imports stay cheap on machines without a driver.
#pragma once

#include <cstddef>
#include <cstdint>

namespace fissionkv {

typedef int CUresult;
typedef uint64_t CUdeviceptr;
typedef void* CUstream;
typedef void* CUcontext;
typedef uint64_t CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
#define CU_MEM_ALLOCATION_TYPE_PINNED 0x01
#define CU_MEM_LOCATION_TYPE_DEVICE 0x01
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE 0x03
#define CU_CTX_SCHED_AUTO 0x00
#define CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE 0x01
#define CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD 0x02

struct CUmemAllocationProp {
  int type;
  int requestedHandleTypes;
  struct {
    int type;
    int id;
  } location;
  void* win32HandleMetaData;
  struct {
    unsigned char compressionType;
    unsigned char gpuDirectRDMACapable;
    unsigned short usage;
    unsigned char reserved[4];
  } allocFlags;
};

struct CUmemAccessDesc {
  struct {
    int type;
    int id;
  } location;
  int flags;
};

// Exact-arity function pointers (typedefs keep dlsym casts legal).
typedef CUresult (*PFN_cuMemAddressReserve)(CUdeviceptr*, size_t, size_t,
                                            CUdeviceptr, unsigned long long);
typedef CUresult (*PFN_cuMemAddressFree)(CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemCreate)(CUmemGenericAllocationHandle*, size_t,
                                    const CUmemAllocationProp*,
                                    unsigned long long);
typedef CUresult (*PFN_cuMemRelease)(CUmemGenericAllocationHandle);
typedef CUresult (*PFN_cuMemMap)(CUdeviceptr, size_t, size_t,
                                 CUmemGenericAllocationHandle,
                                 unsigned long long);
typedef CUresult (*PFN_cuMemUnmap)(CUdeviceptr, size_t);
typedef CUresult (*PFN_cuMemSetAccess)(CUdeviceptr, size_t,
                                       const CUmemAccessDesc*, size_t);
typedef CUresult (*PFN_cuMemGetAllocationGranularity)(size_t*,
                                                      const CUmemAllocationProp*,
                                                      int option);
typedef CUresult (*PFN_cuMemHostRegister)(void*, size_t, unsigned int);
typedef CUresult (*PFN_cuMemHostUnregister)(void*);
typedef CUresult (*PFN_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t,
                                          CUstream);
typedef CUresult (*PFN_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t,
                                          CUstream);
typedef CUresult (*PFN_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t,
                                          CUstream);
typedef CUresult (*PFN_cuCtxGetCurrent)(CUcontext*);
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*PFN_cuInit)(unsigned);
typedef CUresult (*PFN_cuDeviceGet)(int*, int);
typedef CUresult (*PFN_cuDevicePrimaryCtxRetain)(CUcontext*, int);
typedef CUresult (*PFN_cuDevicePrimaryCtxRelease)(int);
typedef CUresult (*PFN_cuMemGetInfo)(size_t*, size_t*);
typedef CUresult (*PFN_cuMemHostGetDevicePointer_v2)(CUdeviceptr*, void*,
                                                     unsigned);
typedef CUresult (*PFN_cuDeviceGetAttribute)(int*, int, int);

struct DriverApi {
  PFN_cuMemAddressReserve mem_address_reserve = nullptr;
  PFN_cuMemAddressFree mem_address_free = nullptr;
  PFN_cuMemCreate mem_create = nullptr;
  PFN_cuMemRelease mem_release = nullptr;
  PFN_cuMemMap mem_map = nullptr;
  PFN_cuMemUnmap mem_unmap = nullptr;
  PFN_cuMemSetAccess mem_set_access = nullptr;
  PFN_cuMemGetAllocationGranularity mem_get_granularity = nullptr;
  PFN_cuMemHostRegister mem_host_register = nullptr;
  PFN_cuMemHostUnregister mem_host_unregister = nullptr;
  PFN_cuMemcpyDtoDAsync memcpy_dtod_async = nullptr;
  PFN_cuMemcpyHtoDAsync memcpy_htod_async = nullptr;
  PFN_cuMemcpyDtoHAsync memcpy_dtoh_async = nullptr;
  PFN_cuCtxGetCurrent ctx_get_current = nullptr;
  PFN_cuCtxSetCurrent ctx_set_current = nullptr;
  PFN_cuInit init = nullptr;
  PFN_cuDeviceGet device_get = nullptr;
  PFN_cuDevicePrimaryCtxRetain primary_ctx_retain = nullptr;
  PFN_cuDevicePrimaryCtxRelease primary_ctx_release = nullptr;
  PFN_cuMemGetInfo mem_get_info = nullptr;
  PFN_cuMemHostGetDevicePointer_v2 host_get_device_pointer = nullptr;
  PFN_cuDeviceGetAttribute device_get_attribute = nullptr;

  bool load();
  const char* last_error() const;

  bool ok() const {
    return mem_address_reserve && mem_create && mem_map && mem_unmap &&
           mem_set_access && memcpy_dtod_async && memcpy_htod_async &&
           memcpy_dtoh_async && ctx_get_current && ctx_set_current;
  }

 private:
  void* handle_ = nullptr;
  char error_[256] = {0};
};

// Process-wide lazily loaded driver. Returns nullptr when unavailable.
const DriverApi* driver();

}  // namespace fissionkv
