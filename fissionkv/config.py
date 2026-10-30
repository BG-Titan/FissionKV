# SPDX-License-Identifier: Apache-2.0
"""FissionKV configuration.

Two sources, like Strata: environment defaults (``FISSION_KV_*``) overridable by
the per-instance ``kv_transfer_config.extra_config`` dict handed to the
connector. Bytes accept plain ints or strings such as ``"16GiB"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

_GIB = 1 << 30
_MIB = 1 << 20


def parse_bytes(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"invalid byte value: {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper().replace("_", "").replace(" ", "")
    if not text:
        return default
    for suffix, mult in (("KIB", 1024), ("MIB", _MIB), ("GIB", _GIB),
                         ("TIB", 1 << 40), ("KB", 1024), ("MB", _MIB),
                         ("GB", _GIB), ("TB", 1 << 40), ("K", 1024),
                         ("M", _MIB), ("G", _GIB), ("T", 1 << 40),
                         ("B", 1)):
        if text.endswith(suffix):
            num = text[: -len(suffix)] or "0"
            return int(float(num) * mult)
    return int(float(text))


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")




@dataclass
class FissionKVConfig:
    """Runtime knobs for one FissionKV instance.

    Tiering (feature A) is on whenever the connector is enabled with a
    positive host budget. Elasticity (feature B) is opt-in via
    ``FISSION_KV_VA_BYTES`` (see fissionkv/patch.py - it patches the engine at
    startup and needs no connector at all).
    """

    # ---- host tier (A) ----
    cpu_bytes_to_use: int = 0            # pinned host arena budget
    cpu_bytes_to_use_per_rank: int = 0   # overrides world-size split if >0
    offload_after_secs: float = 0.0      # lazy-store delay (0 => eager scan)
    lazy_store: bool = False             # True: stage only under pressure
    high_water: float = 0.9              # lazy: drain when free slots < 1-this
    store_batch_blocks: int = 512        # blocks staged per step (eager cap)
    per_layer_overlap: bool = False      # per-layer-group load events (opt-in)
    store_scan_step: int = 1024          # max GPU blocks scanned per step

    # ---- elastic pool (B) - read by fissionkv.patch, echoed here for docs ----
    va_bytes: int = 0                    # FISSION_KV_VA_BYTES: reserve-this-much VA
    phys_init_bytes: int = 0             # FISSION_KV_PHYS_INIT_BYTES mapped at boot
    ring_bytes: int = 0                  # FISSION_KV_RING_BYTES copy staging pool

    # ---- internals, fixed defaults mirroring Strata ----
    chunk_mib: int = 2                   # VMM physical chunk granularity
    arena_alignment: int = 2 * _MIB

    extra: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_extra_config(cls, extra: dict | None) -> "FissionKVConfig":
        env = dict(os.environ)
        cfg = cls(extra=dict(extra or {}))
        cfg.cpu_bytes_to_use = parse_bytes(
            _pick(extra, "cpu_bytes_to_use", env.get("FISSION_KV_CPU_BYTES")), 0)
        cfg.lazy_store = parse_bool(
            _pick(extra, "lazy_store", env.get("FISSION_KV_LAZY_STORE")), False)
        cfg.cpu_bytes_to_use_per_rank = parse_bytes(
            _pick(extra, "cpu_bytes_to_use_per_rank",
                  env.get("FISSION_KV_CPU_BYTES_PER_RANK")), 0)
        cfg.high_water = float(
            _pick(extra, "high_water",
                  env.get("FISSION_KV_HIGH_WATER", "0.9")))
        cfg.store_batch_blocks = int(
            _pick(extra, "store_batch_blocks",
                  env.get("FISSION_KV_STORE_BATCH_BLOCKS", "512")))
        cfg.offload_after_secs = float(
            _pick(extra, "offload_after_secs",
                  env.get("FISSION_KV_OFFLOAD_AFTER_SECS", "0")) or 0)
        cfg.per_layer_overlap = parse_bool(
            _pick(extra, "per_layer_overlap",
                  env.get("FISSION_KV_PER_LAYER_OVERLAP")), False)
        cfg.store_scan_step = int(
            _pick(extra, "store_scan_step",
                  env.get("FISSION_KV_STORE_SCAN_STEP", "1024")))
        cfg.va_bytes = parse_bytes(
            env.get("FISSION_KV_VA_BYTES", ""), 0)
        cfg.phys_init_bytes = parse_bytes(
            env.get("FISSION_KV_PHYS_INIT_BYTES", ""), 0)
        cfg.ring_bytes = parse_bytes(env.get("FISSION_KV_RING_BYTES", ""), 0)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.cpu_bytes_to_use < 0:
            raise ValueError("cpu_bytes_to_use must be >= 0")
        if not 0.0 < self.high_water <= 1.0:
            raise ValueError("high_water must be in (0, 1]")
        if self.store_batch_blocks <= 0:
            raise ValueError("store_batch_blocks must be > 0")
        if self.offload_after_secs < 0:
            raise ValueError("offload_after_secs must be >= 0")
        if self.va_bytes and self.chunk_mib * _MIB > self.va_bytes:
            raise ValueError("FISSION_KV_VA_BYTES smaller than one chunk")

    @property
    def store_mode(self) -> str:
        return "lazy" if self.lazy_store else "eager"

    @property
    def tiering_enabled(self) -> bool:
        return self.cpu_bytes_to_use > 0

    @property
    def elastic_enabled(self) -> bool:
        return self.va_bytes > 0


def _pick(extra: dict | None, key: str, env_value: object) -> object:
    if extra and key in extra:
        return extra[key]
    return env_value


def describe(cfg: FissionKVConfig) -> str:
    parts = []
    for f in fields(cfg):
        if f.name == "extra":
            continue
        v = getattr(cfg, f.name)
        if v not in (0, 0.0, False, "", None):
            parts.append(f"{f.name}={v}")
    return "FissionKV(" + ", ".join(parts) + ")"
