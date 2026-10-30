# SPDX-License-Identifier: Apache-2.0
"""FissionKV: Strata-inspired tiered + elastic KV cache for vLLM.

External module (pip package + vLLM general-plugin) implementing:

* A-tier: pinned host KV arena managed by a C++ core (dlopen'd CUDA driver,
  no cuda-python dependency), run-coalesced batch D2H/H2D, LRU + second-chance
  host block index, async load via ``WAITING_FOR_REMOTE_KVS``.
* B-elastic: VMM-backed GPU pool - reserve huge VA, map physical 2 MiB chunks
  on demand (Strata ``kv_reserve``), plus a device copy-ring staging pool
  (Strata ``kv_alloc``) so the engine's VRAM floor drops below one full
  context window.

Layout:
  fissionkv/config.py      env + extra-config parsing
  fissionkv/wire.py        KVConnectorMetadata / worker metadata
  fissionkv/plan.py        pure copy-plan run coalescing (mirrors Strata runs_of)
  fissionkv/block_index.py pure host block index (LRU + second chance)
  (pinned host arena lives in worker.py: torch pin_memory + native
   host_register; see csrc/pin_arena.cpp for the registration core)
  fissionkv/elastic.py     VMM pool python wrapper (adopt + grow hint)
  fissionkv/_C*.so         native core (csrc/*.cpp, dlopen libcuda; optional)
  fissionkv/worker.py      worker-side scheduler state + copy orchestration
  fissionkv/scheduler.py   scheduler-side manager (hits, stores, metadata)
  fissionkv/connector.py   KVConnectorBase_V1 glue (registered as "FissionKV")
  fissionkv/plugin.py      vllm.general_plugins entry point
  fissionkv/patch.py       startup patches for elastic mode (no vLLM edits)
"""

import logging
import os

FISSIONKV_CONNECTOR_NAME = "FissionKV"

# FISSION_KV_LOG=INFO|DEBUG turns on fissionkv.* logging without touching vLLM's.
_LEVEL = (os.environ.get("FISSION_KV_LOG") or
            os.environ.get("FISSION_KV_LOG") or "").upper()
if _LEVEL in ("DEBUG", "INFO", "WARNING"):
    _root = logging.getLogger("fissionkv")
    _root.setLevel(_LEVEL)
    if not any(isinstance(h, logging.StreamHandler) for h in _root.handlers):
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter(
            "fissionkv %(levelname)s %(name)s: %(message)s"))
        _root.addHandler(_h)
        _root.propagate = False

__all__ = ["FISSION_KV_CONNECTOR_NAME"]
