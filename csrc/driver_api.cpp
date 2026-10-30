#include "driver_api.h"

#include <dlfcn.h>
#include <cstdio>

#include <mutex>

namespace fissionkv {

static const char* kCandidates[] = {"libcuda.so.1", "libcuda.so"};

bool DriverApi::load() {
  if (handle_) return ok();
  for (const char* name : kCandidates) {
    void* h = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (h) {
      handle_ = h;
      break;
    }
  }
  if (!handle_) {
    snprintf(error_, sizeof(error_), "libcuda.so not loadable");
    return false;
  }
#define BIND(field, sym)                                       \
  do {                                                         \
    void* p = dlsym(handle_, sym);                             \
    if (!p) {                                                  \
      snprintf(error_, sizeof(error_), "missing symbol %s",    \
               sym);                                           \
      return false;                                            \
    }                                                          \
    field = reinterpret_cast<decltype(field)>(p);              \
  } while (0)
  BIND(mem_address_reserve, "cuMemAddressReserve");
  BIND(mem_address_free, "cuMemAddressFree");
  BIND(mem_create, "cuMemCreate");
  BIND(mem_release, "cuMemRelease");
  BIND(mem_map, "cuMemMap");
  BIND(mem_unmap, "cuMemUnmap");
  BIND(mem_set_access, "cuMemSetAccess");
  BIND(mem_get_granularity, "cuMemGetAllocationGranularity");
  BIND(mem_host_register, "cuMemHostRegister");
  BIND(mem_host_unregister, "cuMemHostUnregister");
  BIND(memcpy_dtod_async, "cuMemcpyDtoDAsync_v2");
  BIND(memcpy_htod_async, "cuMemcpyHtoDAsync_v2");
  BIND(memcpy_dtoh_async, "cuMemcpyDtoHAsync_v2");
  BIND(ctx_get_current, "cuCtxGetCurrent");
  BIND(ctx_set_current, "cuCtxSetCurrent");
  BIND(init, "cuInit");
  BIND(device_get, "cuDeviceGet");
  BIND(primary_ctx_retain, "cuDevicePrimaryCtxRetain");
  BIND(primary_ctx_release, "cuDevicePrimaryCtxRelease");
  BIND(mem_get_info, "cuMemGetInfo_v2");
  BIND(host_get_device_pointer, "cuMemHostGetDevicePointer_v2");
  BIND(device_get_attribute, "cuDeviceGetAttribute");
#undef BIND
  return ok();
}

const char* DriverApi::last_error() const { return error_; }

static DriverApi& driver_instance() {
  static DriverApi api;
  static std::once_flag once;
  std::call_once(once, [&] { api.load(); });
  return api;
}

const DriverApi* driver() {
  const DriverApi& api = driver_instance();
  return api.ok() ? &api : nullptr;
}

}  // namespace fissionkv
