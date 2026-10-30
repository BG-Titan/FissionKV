# FissionKV — Strata-inspired tiered + elastic KV cache for vLLM

External module (pip package + `vllm.general_plugins` plugin + C++ core).
No vLLM source edits: vLLM adaptation happens through the public
`KVConnectorBase_V1` connector API, the plugin entry point, and (elastic
mode only) three small runtime monkey-patches in
[`fissionkv/patch.py`](fissionkv/patch.py).

Full mechanism analysis (Strata hot/cold window, VMM pool, copy engine,
coupling with attention backends, and the FissionKV mapping) is in the root
[`docs/moe-design.md`](../docs/moe-design.md); the dense stack's longer
write-ups (`strata-kv-mechanism.md`, `testing-guide.md`) live in the sibling
`fuse-dense/fusionkv/docs/` and apply unchanged to everything not listed below.

## fork 差异（相对稠密版 `fusionkv`）

本包是 `fuse-dense/fusionkv` 的**分叉**（整包拷入，不 import 稠密包）：
包名 `fissionkv`、连接器名 `FissionKV`、环境变量前缀 `FISSION_KV_*`、
入口 `fissionkv.plugin:load`（另有 `register()` 别名）。在此之上：

* **MLA 层可入分层**：`MLAAttentionSpec` 的 cache
  `[num_blocks, 1, page, kv_lora_rank + rope]`（单个 latent 物理头）被接受，
  latent + rope 作为一个整体块载荷搬运（不做 latent/rope 拆分）。
  平面构造只看 `shape[0]` 与字节步幅，因此形状本来就无关 ——
  这次把这件事**测住了**：`tests/test_worker_planes_mla.py` 用 storage view
  造出带 padding 的 MLA 形状张量，验证平面字节与视图逐字节一致。
* **spec 按组识别**：`scheduler.py` 逐组读 `g.kv_cache_spec`，把
  `MLAAttentionSpec` 也认成"可分层的注意力组"（稠密版只认 `FullAttentionSpec`），
  于是**单组 MLA 计划可以分层**。存储侧（平面 + 拷贝）本来就与组数无关，
  混合形状的平面在 CPU 测试里已覆盖；但 **调度侧的分层策略 v0 仍只在单组计划
  开启**：多组时显式警告并整体 pass-through（host 恢复必须同时覆盖所有可缓存组
  才有意义，见下）。mamba/SWA 这类非 `AttentionSpec` 组依旧 pass-through。
* **平面构造抽成 `worker.build_planes(kv_caches)`**（纯函数，CPU 可测），
  原 `register_kv_caches` 调用它；零拷贝 storage-view 语义不变。
* 保留稠密版的两个关键修复：**bind_connector_metadata 时的弹性增长**
  （grow-at-bind，前向之前把物理映射跟上 block-id 水位）与
  **boot-floor 环境变量**（`FISSION_KV_PHYS_INIT_BYTES` 兜住 CUDA graph 捕获
  期间的 KV 物理页）。

## Two features

**A — host tier (always safe, default mode).**
A pinned host RAM KV tier managed like Strata's: run-coalesced batch
D2H/H2D through a C++ copy engine, LRU + second-chance host block index,
async prefix restore via vLLM's `WAITING_FOR_REMOTE_KVS`. Sessions whose
KV left the GPU still resume with prompt-recompute skipped; GPU VRAM no
longer bounds how much prefix history survives.

**B — elastic pool (experimental, opt-in).**
Strata's `VmmRange`: reserve a huge virtual address range with
`cuMemAddressReserve`, map physical 2 MiB chunks on demand
(`cuMemMap`). vLLM advertises a KV pool larger than physical VRAM; the
physical mapping follows a per-step block-id grow-hint shipped in the
connector metadata and is mapped *before* forward (attention writes KV in
place). This is the "small VRAM floor" port: the engine boots while
holding only a slice of the KV pool in DRAM.

## Install

```bash
pip install ./fissionkv          # compiles csrc/*.cpp -> fissionkv._C (g++ + pybind11)
```

The native core `dlopen()`s `libcuda` at runtime — no CUDA toolkit needed
to build. On systems without Python dev headers set `FISSION_KV_PY_INC`;
`FISSION_KV_SKIP_BUILD=1` installs pure-python (tiering falls back to torch
gather copies; elastic unavailable).

## Enable (A-tier example)

```bash
VLLM_PLUGINS=fissionkv python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --kv-transfer-config '{"kv_connector":"FissionKV","kv_role":"kv_both",
                           "kv_connector_extra_config":
                           {"cpu_bytes_to_use":"32GiB"}}'
```

(`VLLM_PLUGINS` is optional — general plugins auto-load unless an
allowlist is set.) No `--kv-transfer-config`? The plugin still registers
the connector name, so it can also be selected via `kv_connector_module_path`
in any existing tooling.

## Config knobs

extra_config key / env var (env wins over missing key, key wins over default):

| key | env | default | meaning |
|---|---|---|---|
| `cpu_bytes_to_use` | `FISSION_KV_CPU_BYTES` | 0 (=off) | pinned host arena budget |
| `cpu_bytes_to_use_per_rank` | `FISSION_KV_CPU_BYTES_PER_RANK` | 0 | skip world-size split |
| `lazy_store` | `FISSION_KV_LAZY_STORE` | false | stage only under host pressure |
| `high_water` | `FISSION_KV_HIGH_WATER` | 0.9 | lazy drain threshold |
| `store_batch_blocks` | `FISSION_KV_STORE_BATCH_BLOCKS` | 512 | blocks per step cap |
| `offload_after_secs` | `FISSION_KV_OFFLOAD_AFTER_SECS` | 0 | age before lazy stage |
| `store_scan_step` | `FISSION_KV_STORE_SCAN_STEP` | 1024 | GPU scan step (eager) |
| — | `FISSION_KV_VA_BYTES` | 0 (=off) | **elastic**: VA reservation |
| — | `FISSION_KV_PHYS_INIT_BYTES` | 0 | elastic physical at boot (keep >= graph-capture footprint, e.g. 2 GiB) |
| — | `FISSION_KV_KV_BYTES` | 0 | elastic advertised KV budget (> physical!) |

Byte values accept `K/M/G/T` (binary) and `KiB/MiB/GiB/TiB`.

## Elastic mode

```bash
export VLLM_PLUGINS=fissionkv
export FISSION_KV_VA_BYTES=160GiB        # VA for the KV backing
export FISSION_KV_PHYS_INIT_BYTES=4GiB   # physical at boot (capture headroom)
export FISSION_KV_KV_BYTES=96GiB         # advertise this KV pool (>> physical)
vllm serve /path/to/model --gpu-memory-utilization 0.25
```

Engine sizes `num_blocks` against `FISSION_KV_KV_BYTES`; only ~4 GiB is
physically resident until sessions push the block-id watermark up.
Growth is monotone (Strata rule — never unmap mid-decode); an explicit
operator-side trim exists in the API (`VmmPool.trim_to`) but is not wired
to an automatic policy in v1. Requires the native core and the FissionKV
connector enabled (grow-hints travel in connector metadata).

## Architecture

| file | role |
|---|---|
| `csrc/vmm.cpp` | VMM pool: VA reserve + 2 MiB chunk map/unmap (Strata `vmm.cpp`) |
| `csrc/copy_engine.cpp` | run-coalesced H2D/D2H plane copies, event poll |
| `csrc/pin_arena.cpp` | pinned arena helpers (`cudaHostRegister` path) |
| `csrc/driver_api.cpp` | `dlopen("libcuda.so.1")` symbol table |
| `fissionkv/plan.py` | pure-python twin of `runs_of` coalescing (tested) |
| `fissionkv/block_index.py` | LRU + second chance + pins, prefix-chain aware |
| `fissionkv/scheduler.py` | hits/loads/stores/deferred-free protocol |
| `fissionkv/worker.py` | arena, copy thread, elastic grow, `build_planes` (dense + MLA shapes) |
| `fissionkv/connector.py` | KVConnectorBase_V1 glue (registered "FissionKV") |
| `fissionkv/patch.py` | elastic monkey-patches (allocate/pool watermark/advertise) |
| `fissionkv/plugin.py` | `vllm.general_plugins` entry |

Scheduler-side block accounting mirrors vLLM's own
`simple_cpu_offload` manager contract (two-phase frees via
`request_finished` hold + injected `finished_sending`, worker-count
aggregated store acks, `need_flush` on preemption), so HMA, chunked
prefill and async loads interact exactly the way core expects.

## Tests

Three tiers (CPU unit → GPU native smoke → engine e2e); the dense fork's
guide with acceptance criteria and troubleshooting is
`fuse-dense/fusionkv/docs/testing-guide.md`.

```bash
python -m pytest tests -q           # CPU only: planes (dense + MLA shapes),
                                    # block index, plan coalescing, scheduler
python -m pytest ../fissionkv/tests -q   # run from repo root also fine
python scripts/native_smoke.py      # GPU + torch: VMM pool + copy parity
./scripts/gpu_e2e.sh /path/to/model tier     # real-engine checks (GPU host)
./scripts/gpu_e2e.sh /path/to/model both
```

## Status / limits

* A-tier: unit-tested against the core scheduler contract; run
  `scripts/gpu_e2e.sh` on a GPU host for the live check.
* A **single-group** plan tiers, whether the group is full-attention or MLA
  (see “fork 差异”). Multi-group hybrid plans — including MLA +
  full-attention — still pass through with a warning: restore has to cover
  every cacheable group before tiering is meaningful.
* MLA tiers move latent+rope as one block payload — no latent-only
  sub-pinning yet (that needs a spec-aware copy plan).
* Elastic: experimental; keep `FISSION_KV_PHYS_INIT_BYTES` above the CUDA-graph
  capture KV footprint (2 GiB default in `patch._make_pool`).
* Decode-context-parallel: rejected explicitly (not supported).
