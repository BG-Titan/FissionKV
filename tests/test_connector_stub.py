# SPDX-License-Identifier: Apache-2.0
"""Import + shape tests that need no torch/vllm (uses stub_vllm)."""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

try:                                # do not poison sys.modules when vllm is real
    import vllm  # noqa: F401
    pytest.skip("stub shapes only meaningful without a real vllm install",
                allow_module_level=True)
except ImportError:
    pass

import stub_vllm  # noqa: E402


class FullAttentionSpec:      # name matters: scheduler type-matches it
    block_size = 16
    page_size_bytes = 1024


class FakeSpec:
    block_size = 16
    page_size_bytes = 512


def _mk(kvcfg, role, extra=None):
    os.environ.pop("FISSION_KV_CPU_BYTES", None)
    cfg = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(
            kv_role="kv_both", kv_connector_extra_config=extra or {}),
        parallel_config=types.SimpleNamespace(world_size=1),
        cache_config=types.SimpleNamespace(block_size=16))
    import fissionkv.connector as c
    return c.FissionKVConnector(cfg, role, kvcfg)


def _kvcfg(*specs, hash_block_size=None):
    groups = [types.SimpleNamespace(block_size=s.block_size,
                                    layer_names=[f"l{i}" for i in range(8)],
                                    kv_cache_spec=s)
              for s in specs]
    return types.SimpleNamespace(kv_cache_groups=groups,
                                 hash_block_size=hash_block_size)


def test_scheduler_role_imports_without_torch():
    Role = stub_vllm.install()
    conn = _mk(_kvcfg(FullAttentionSpec()), Role.SCHEDULER,
               {"cpu_bytes_to_use": "64MB"})
    sm = conn.scheduler_manager
    assert sm is not None and sm.tiering
    assert sm.fa_group_idx == 0 and sm.block_size == 16
    # host index sized from page_size_bytes * layers
    assert sm.num_cpu_blocks >= 1
    assert conn.worker_handler is None


def test_multi_group_disables_tiering():
    Role = stub_vllm.install()
    conn = _mk(_kvcfg(FullAttentionSpec(), FakeSpec()), Role.SCHEDULER,
               {"cpu_bytes_to_use": "64MB"})
    assert not conn.scheduler_manager.tiering


def test_hash_block_size_multiple():
    Role = stub_vllm.install()
    conn = _mk(_kvcfg(FullAttentionSpec(), hash_block_size=4),
               Role.SCHEDULER, {"cpu_bytes_to_use": "64MB"})
    assert conn.scheduler_manager.sub_per_fa == 4


def test_finer_hash_than_block_unsupported():
    Role = stub_vllm.install()
    with pytest.raises(NotImplementedError):
        _mk(_kvcfg(FullAttentionSpec(), hash_block_size=32),
            Role.SCHEDULER, {"cpu_bytes_to_use": "64MB"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
