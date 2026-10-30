#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FissionKV L2 native smoke: VMM pool + run-coalesced copy parity.

Needs a GPU and torch; no vLLM.  See docs/testing-guide.md §3.
Exit code 0 == all checks passed.
"""
import sys


def main() -> int:
    import torch

    import fissionkv._C as C

    assert C.driver_available(), "libcuda symbols not resolved"
    print(f"driver_available OK, free_physical_bytes(dev0)="
          f"{C.free_physical_bytes(0)}")

    # --- VMM elastic pool: grow / shrink -------------------------------
    pool = C.VmmPool.create(va_bytes=8 << 30, phys_bytes=1 << 30,
                            device_id=0)
    s = pool.stats()
    assert s["phys_mapped"] >= 1 << 30, s
    pool.ensure_physical(3 << 30)
    assert pool.stats()["phys_mapped"] >= 3 << 30, pool.stats()
    pool.shrink_physical(1 << 30, 0, 1 << 20)
    s = pool.stats()
    print(f"vmm OK: va={s['va_size'] >> 30}GiB "
          f"phys_mapped={s['phys_mapped'] >> 20}MiB chunk={s['chunk']}")

    # --- copy parity + run coalescing -----------------------------------
    blk, n = 1 << 20, 16
    gpu = torch.randint(0, 256, (n, blk), dtype=torch.uint8, device="cuda")
    host = torch.empty_like(gpu, device="cpu").pin_memory()
    eng = C.CopyEngine()
    assert eng.init_plane(gpu.data_ptr(), host.data_ptr(), blk)
    stream = torch.cuda.current_stream().cuda_stream
    ids = list(range(3, 11))
    ops = eng.copy_batch(0, ids, ids, False, stream)        # D2H
    torch.cuda.synchronize()
    assert ops == 1, f"run coalescing broken: ops={ops}"
    assert torch.equal(gpu[3:11], host[3:11]), "D2H mismatch"
    gpu[3:11] = 0
    ops = eng.copy_batch(0, ids, ids, True, stream)         # H2D
    torch.cuda.synchronize()
    assert ops == 1 and torch.equal(gpu[3:11], host[3:11]), "H2D mismatch"
    ops = eng.copy_batch(0, [0, 2, 5], [0, 2, 5], False, stream)
    torch.cuda.synchronize()
    assert ops == 3, f"non-contiguous should emit 3 ops, got {ops}"
    print("copy parity OK (coalesced 1-op runs; discrete 3-op split)")

    # --- host_register on a pinned arena (informational, refcounted) ----
    h = torch.empty(1 << 20, dtype=torch.uint8).pin_memory()
    r1 = C.host_register(h.data_ptr(), h.numel())
    r2 = C.host_unregister(h.data_ptr())
    print(f"host_register={r1} host_unregister={r2} (informational)")
    print("NATIVE SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
