# SPDX-License-Identifier: Apache-2.0
"""Plugin entry behavior without vllm/torch installed."""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import stub_vllm  # noqa: E402

try:                                # the stub must never poison a real vllm
    import vllm  # noqa: F401
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


def test_register_against_real_factory():
    """With vllm installed, registration must land in the real factory.

    Runs first on purpose: it touches only the (lazy) connector registry, and
    it is the check that actually matters in a vllm environment.
    """
    if not HAVE_VLLM:
        pytest.skip("needs a real vllm install")
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    import fissionkv.plugin as plugin
    plugin.register_connector()
    loader = KVConnectorFactory._registry.get("FissionKV")
    assert callable(loader), "the real factory stores a lazy loader"
    cls = loader()                    # resolves fissionkv.connector
    assert cls.__name__ == "FissionKVConnector"
    assert cls.__module__ == "fissionkv.connector"
    plugin.register_connector()       # idempotent, no ValueError
    assert KVConnectorFactory._registry.get("FissionKV") is loader


def test_register_and_load(tmp_path, monkeypatch):
    if HAVE_VLLM:
        pytest.skip("stub shapes only meaningful without vllm")
    stub_vllm.install()
    # minimal factory module stub
    reg = {}

    class F:
        _registry = reg

        @classmethod
        def register_connector(cls, name, module_path, class_name):
            reg[name] = (module_path, class_name)

    fac = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.factory")
    fac.KVConnectorFactory = F
    sys.modules[fac.__name__] = fac

    os.environ.pop("FISSION_KV_VA_BYTES", None)
    import fissionkv.plugin as plugin
    plugin._registered = False
    plugin._installed = False
    plugin.load()
    assert reg == {
        "FissionKV": ("fissionkv.connector", "FissionKVConnector"),
        "FissionKV": ("fissionkv.connector", "FissionKVConnector"),
    }
    plugin.load()                      # idempotent
    assert len(reg) == 1
    # without FISSION_KV_VA_BYTES the elastic patch module is not even imported
    assert "fissionkv.patch" not in sys.modules or \
        sys.modules["fissionkv.patch"].ELASTIC_POOL is None


def test_patch_inactive_without_env(monkeypatch):
    os.environ.pop("FISSION_KV_VA_BYTES", None)
    from fissionkv import patch
    assert patch.elastic_enabled() is False
    assert patch.install() is False
    assert patch.ELASTIC_POOL is None


def test_patch_requires_native(monkeypatch):
    monkeypatch.setenv("FISSION_KV_VA_BYTES", "8GiB")
    from fissionkv import patch
    if patch.HAVE_NATIVE:
        pytest.skip("native present; failure path not reachable")
    with pytest.raises(RuntimeError):
        patch._PATCHED = False
        patch.install()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
