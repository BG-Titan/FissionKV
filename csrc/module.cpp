// pybind11 bindings for the FissionKV native core: VMM pool + copy engine.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "copy_engine.h"
#include "driver_api.h"
#include "vmm.h"

namespace py = pybind11;
using namespace fissionkv;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FissionKV native core (VMM pool + run-coalesced copy engine)";

  m.def("driver_available", [] { return driver() != nullptr; },
        "True when libcuda driver API symbols resolved");

  m.def(
      "free_physical_bytes",
      [](int device_id) { return driver_free_physical_bytes(device_id); },
      "cuMemGetInfo free bytes for the device's primary context, -1 if N/A");

  m.def(
      "host_register",
      [](uint64_t ptr, uint64_t size) {
        const DriverApi* d = driver();
        if (!d) return false;
        return d->mem_host_register(reinterpret_cast<void*>(ptr),
                                    static_cast<size_t>(size), 0) ==
               CUDA_SUCCESS;
      },
      py::arg("ptr"), py::arg("size"),
      "cuMemHostRegister (refcounted; safe on duplicate registrations)");
  m.def(
      "host_unregister",
      [](uint64_t ptr) {
        const DriverApi* d = driver();
        if (!d) return false;
        return d->mem_host_unregister(reinterpret_cast<void*>(ptr)) ==
               CUDA_SUCCESS;
      },
      py::arg("ptr"));

  py::class_<CopyEngine>(m, "CopyEngine")
      .def(py::init<>())
      .def("init_plane", &CopyEngine::init_plane, py::arg("gpu_base"),
           py::arg("host_base"), py::arg("stride"))
      .def("init_plane2", &CopyEngine::init_plane2, py::arg("gpu_base"),
           py::arg("host_base"), py::arg("gpu_stride"),
           py::arg("host_stride"), py::arg("block_bytes"))
      .def(
          "copy_batch",
          [](CopyEngine& e, int plane, const std::vector<uint64_t>& src,
             const std::vector<uint64_t>& dst, bool direction,
             uint64_t stream) {
            if (src.size() != dst.size())
              throw std::invalid_argument("src/dst length mismatch");
            return e.copy_batch(plane, src.data(), dst.data(), src.size(),
                                direction, stream);
          },
          py::arg("plane_idx"), py::arg("src_ids"), py::arg("dst_ids"),
          py::arg("direction"), py::arg("stream"),
          "Run-coalesced batch copy; direction true=H2D false=D2H")
      .def_property_readonly("num_planes",
                             [](const CopyEngine& e) {
                               return e.planes.size();
                             });

  py::class_<VmmPool, std::shared_ptr<VmmPool>>(m, "VmmPool")
      .def_static(
          "create",
          [](uint64_t va_bytes, uint64_t phys_bytes, int device_id) {
            const DriverApi* d = driver();
            if (!d) throw std::runtime_error("cuda driver unavailable");
            auto pool = std::make_shared<VmmPool>();
            CUmemAllocationProp prop;
            std::memset(&prop, 0, sizeof(prop));
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device_id;
            size_t gran = 0;
            if (d->mem_get_granularity(&gran, &prop, 0) != CUDA_SUCCESS ||
                gran == 0)
              throw std::runtime_error("granularity query failed");
            pool->chunk = gran;
            va_bytes = (va_bytes + gran - 1) / gran * gran;
            CUcontext prev = nullptr;
            d->ctx_get_current(&prev);
            CUcontext ctx = nullptr;
            if (d->primary_ctx_retain(&ctx, device_id) == CUDA_SUCCESS) {
              d->primary_ctx_release(device_id);
              d->ctx_set_current(ctx);
            }
            CUdeviceptr base = 0;
            CUresult r = d->mem_address_reserve(&base, va_bytes, 0, 0, 0);
            if (r == CUDA_SUCCESS) {
              pool->base = base;
              pool->va_size = va_bytes;
              pool->device_id = device_id;
              pool->managed = true;
              pool->valid = true;
              vmm_ensure_physical(*pool, phys_bytes);
              if (pool->phys_bytes == 0) {
                d->mem_address_free(base, va_bytes);
                pool->valid = false;
                d->ctx_set_current(prev);
                throw std::runtime_error("initial physical map failed");
              }
            }
            d->ctx_set_current(prev);
            if (!pool->valid) throw std::runtime_error("VA reserve failed");
            return pool;
          },
          py::arg("va_bytes"), py::arg("phys_bytes"), py::arg("device_id"))
      .def_static(
          "attach",
          [](uint64_t handle, uint64_t base, uint64_t size,
             uint64_t phys_bytes, int device_id) {
            auto pool = std::make_shared<VmmPool>();
            if (!vmm_attach(*pool, handle, base, size, phys_bytes,
                            device_id))
              return std::shared_ptr<VmmPool>();
            return pool;
          },
          py::arg("handle"), py::arg("base"), py::arg("size"),
          py::arg("phys_bytes"), py::arg("device_id"))
      .def("ensure_ranges",
           [](VmmPool& p, std::vector<std::pair<uint64_t, uint64_t>> ranges) {
             return vmm_ensure_ranges(p, ranges);
           },
           py::arg("ranges"))
      .def("ensure_physical",
           [](VmmPool& p, uint64_t target) {
             return vmm_ensure_physical(p, target);
           })
      .def("shrink_physical",
           [](VmmPool& p, uint64_t target, uint64_t pinned_tail_blocks,
              uint64_t block_bytes) {
             return vmm_shrink_physical(p, target, pinned_tail_blocks,
                                        block_bytes);
           })
      .def("stats", [](const VmmPool& p) {
        py::dict out;
        out["base"] = p.base;
        out["va_size"] = p.va_size;
        out["phys_mapped"] = p.phys_bytes;
        out["chunk"] = p.chunk;
        out["device_id"] = p.device_id;
        out["adopted"] = p.adopted;
        return out;
      });
}
