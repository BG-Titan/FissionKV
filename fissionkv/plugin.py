# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin entry point for FissionKV (MoE KV stack).

pyproject registers ``fissionkv = "fissionkv.plugin"`` under the
``vllm.general_plugins`` entry-point group. vLLM calls
``load_general_plugins()`` early in every process (API server, engine
core, workers), so this runs before any KVTransferConfig is resolved.

Two jobs:
1. register the FissionKV connector with KVConnectorFactory so
   ``--kv-transfer-config '{"kv_connector":"FissionKV",...}'`` resolves
2. apply elastic-mode patches (fissionkv.patch) when FISSION_KV_VA_BYTES
   (FISSION_KV_VA_BYTES) is set.

Enable by adding "fissionkv" to VLLM_PLUGINS if plugin auto-loading is
restricted: ``VLLM_PLUGINS=fissionkv``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_registered = False
_installed = False


def register_connector() -> None:
    global _registered
    if _registered:
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
    except ImportError:  # vllm not importable in this process
        return
    if "FissionKV" in getattr(KVConnectorFactory, "_registry", {}):
        _registered = True
        return
    KVConnectorFactory.register_connector(
        "FissionKV", "fissionkv.connector", "FissionKVConnector")
    _registered = True
    logger.info("FissionKV: registered KV connector 'FissionKV' "
                "(fissionkv.connector:FissionKVConnector)")


def load() -> None:
    """General-plugin entry function (setuptools ``load`` convention)."""
    global _installed
    register_connector()
    if _installed:
        return
    _installed = True
    from .patch import _env
    if os.environ.get("FISSION_KV_VA_BYTES", "").strip():
        # lazy: patch touches vllm internals; import vllm here is fine,
        # plugins load after vllm import
        from . import patch
        try:
            patch.install()
        except Exception:
            logger.exception(
                "FissionKV elastic mode failed to initialise; "
                "unset FISSION_KV_VA_BYTES to fall back to stock allocation")
            raise


def register() -> None:
    """Alias of :func:`load` for bootstraps that import ``<pkg>.plugin:register``.

    The dense plugins (fusew4 / fusionattn) expose ``register``; the fork keeps
    vLLM's ``load`` convention for the entry point, so both names work.
    """
    load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load()
