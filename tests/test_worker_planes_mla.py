# SPDX-License-Identifier: Apache-2.0
"""Plane geometry tests on CPU — dense and MLA KV-cache shapes.

The fork's whole premise is that the tiering plane is a *storage view* of
whatever vLLM hands ``register_kv_caches``, never a copy.  These tests build
both layouts the way vLLM does (strided dtype views over one flat backing,
including the padding a hybrid plan introduces) and assert the byte planes
land exactly on the source bytes.

No GPU, no vllm engine: only torch CPU + the shape helpers in worker.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parents[1]))

try:                                  # worker.py needs vllm.logger only
    import vllm.logger  # noqa: F401
except ImportError:                   # pragma: no cover
    sys.path.insert(0, str(Path(__file__).parent))
    import stub_vllm

    stub_vllm.install()

from fissionkv.worker import build_planes, plane_geometry  # noqa: E402

FP16 = torch.float16


def _view(backing: torch.Tensor, byte_off: int, shape, strides, dtype=FP16):
    """Strided dtype view into a shared uint8 backing, like vLLM's allocator."""
    esz = torch.tensor([], dtype=dtype).element_size()
    assert byte_off % esz == 0
    return torch.empty(0, dtype=dtype).set_(
        backing.untyped_storage(), byte_off // esz, shape, strides)


def _dense_plan(num_layers=4, num_blocks=8, page=4, heads=2, dim=64,
                pad_elems=512):
    """Dense/FA layout [blocks, 2, page, heads, dim] over one backing.

    ``pad_elems`` inflates the physical block stride above the payload, which
    is what a hybrid plan does when another group needs a bigger page.
    """
    payload = 2 * page * heads * dim
    blk = payload + pad_elems
    layer_bytes = num_blocks * blk * 2
    backing = torch.zeros(num_layers * layer_bytes, dtype=torch.uint8)
    caches = {}
    for i in range(num_layers):
        t = _view(backing, i * layer_bytes, (num_blocks, 2, page, heads, dim),
                  (blk, page * heads * dim, heads * dim, dim, 1))
        caches[f"model.layers.{i}.self_attn.attn"] = t
    return caches, dict(payload_elems=payload, blk_elems=blk,
                        layer_bytes=layer_bytes)


def _mla_plan(num_layers=4, num_blocks=6, page=32, kv_lora_rank=512,
              rope_dim=64, pad_elems=0):
    """MLA absorbed layout as vLLM's MLAAttentionSpec allocates it:
    ``[blocks, 1, page, kv_lora_rank + rope]`` — one latent KV head, the rope
    portion stored alongside the latent (single physical head group)."""
    dl = kv_lora_rank + rope_dim
    payload = 1 * page * dl
    blk = payload + pad_elems
    layer_bytes = num_blocks * blk * 2
    backing = torch.zeros(num_layers * layer_bytes, dtype=torch.uint8)
    caches = {}
    for i in range(num_layers):
        # head axis sits at index 1 (size 1); its stride is then irrelevant
        t = _view(backing, i * layer_bytes, (num_blocks, 1, page, dl),
                  (blk, payload, dl, 1))
        caches[f"model.layers.{i}.self_attn.attn"] = t
    return caches, dict(dl=dl, payload_elems=payload, blk_elems=blk,
                        layer_bytes=layer_bytes)


def _plane_bytes(plane: torch.Tensor, block: int) -> bytes:
    return plane[block].contiguous().numpy().tobytes()


# --------------------------------------------------------------- dense/FA
def test_dense_planes_are_storage_views():
    caches, g = _dense_plan()
    planes = build_planes(caches)
    assert len(planes) == 4
    for layer_name, t in zip(caches, planes):
        assert t.shape == (8, g["payload_elems"] * 2)      # bytes / block
        assert t.stride() == (g["blk_elems"] * 2, 1)       # padded stride kept
        assert t.dtype == torch.uint8

    # zero-copy: writing the dtype view must be visible in the byte plane
    name, t = next(iter(caches.items()))
    t.fill_(0.25)
    plane = build_planes({name: t})[0]
    assert _plane_bytes(plane, 3) == t[3].contiguous().view(torch.uint8).numpy().tobytes()
    # ...and the other direction: a plane write is visible in the cache view
    plane2 = build_planes(caches)[0]
    plane2.zero_()
    assert next(iter(caches.values())).abs().sum().item() == 0.0


def test_dense_geometry_uniform_layout():
    caches, g = _dense_plan()
    geo = plane_geometry(build_planes(caches))
    assert geo.num_blocks == 8
    assert geo.block_stride == g["blk_elems"] * 2
    assert geo.layer_stride == g["layer_bytes"]
    assert geo.uniform_layer_stride
    assert geo.cpu_block_bytes == 4 * g["payload_elems"] * 2
    # closed form and per-plane span agree
    assert geo.span_for(7) == 3 * geo.layer_stride + 8 * geo.block_stride
    assert geo.span_for(0) == 3 * geo.layer_stride + 1 * geo.block_stride


# ------------------------------------------------------------------- MLA
def test_mla_planes_match_mla_attention_spec_bytes():
    caches, g = _mla_plan()
    planes = build_planes(caches)
    # MLAAttentionSpec.page_size_bytes = block_size * num_kv_heads(=1) *
    # (kv_lora_rank + qk_rope_head_dim) * dtype_size
    page_bytes = 32 * 1 * (512 + 64) * 2
    assert g["payload_elems"] * 2 == page_bytes
    for t in planes:
        assert t.shape == (6, page_bytes)
        assert t.stride() == (page_bytes, 1)
    geo = plane_geometry(planes)
    assert geo.cpu_block_bytes == 4 * page_bytes     # all layers tier together
    assert geo.uniform_layer_stride


def test_mla_plane_roundtrip_block_scoped():
    """A tiered block's bytes must be recoverable per (layer, block) — the
    exact contract the copy engine's descriptors rely on."""
    caches, _ = _mla_plan(num_layers=3, num_blocks=5)
    names = list(caches)
    for i, name in enumerate(names):         # distinct pattern per layer
        caches[name].fill_(float(i + 1))
    planes = build_planes(caches)
    for i, (plane, name) in enumerate(zip(planes, names)):
        for b in range(5):
            raw = _plane_bytes(plane, b)
            vals = torch.frombuffer(bytearray(raw), dtype=FP16)
            assert torch.all(vals == float(i + 1))
            # rope tail is part of the same block payload (nothing dropped)
            assert vals.numel() == 32 * 576


def test_mla_padded_plan_still_byte_exact():
    """MLA group sharing a plan with a bigger page -> padded block stride."""
    caches, g = _mla_plan(pad_elems=1024)
    planes = build_planes(caches)
    for t in planes:
        assert t.stride(0) == (g["payload_elems"] + 1024) * 2
    name, t = next(iter(caches.items()))
    t.fill_(0.5)
    plane = build_planes({name: t})[0]
    assert _plane_bytes(plane, 1) == t[1].contiguous().view(torch.uint8).numpy().tobytes()


# -------------------------------------------------- mixed (hybrid) layout
def test_mixed_plan_uses_exact_per_plane_span():
    """Mixed-size plan: two MLA layers plus one differently-strided dense
    layer.  The dense closed form (L-1)*layer_stride would be wrong here."""
    mla, gm = _mla_plan(num_layers=2, num_blocks=4)
    dense, gd = _dense_plan(num_layers=1, num_blocks=4)
    planes = build_planes(mla) + build_planes(dense)
    geo = plane_geometry(planes)
    assert len(planes) == 3
    assert not geo.uniform_layer_stride       # separate storages, mixed sizes

    base = planes[0].data_ptr()
    brute = max(t.data_ptr() - base + t.shape[0] * t.stride(0) for t in planes)
    assert geo.span_for(3) == brute
    naive = 2 * geo.layer_stride + 4 * geo.block_stride
    assert naive != brute                     # documents why grow must be exact
    # span is monotone in the block id
    assert geo.span_for(0) < geo.span_for(1) <= geo.span_for(3)


def test_single_plane_geometry():
    caches, g = _mla_plan(num_layers=1)
    geo = plane_geometry(build_planes(caches))
    assert geo.layer_stride == g["payload_elems"] * 2   # fallback = payload
    assert geo.uniform_layer_stride
    assert geo.span_for(0) == g["payload_elems"] * 2


def test_planes_never_copy_a_strided_cache():
    """Regression: `.view(uint8).reshape()` on the dtype view would silently
    COPY, detaching the plane from the cache the kernels write."""
    caches, _ = _dense_plan(num_layers=1, pad_elems=512)
    t = next(iter(caches.values()))
    plane = build_planes(caches)[0]
    assert plane.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
    assert plane.data_ptr() == t.data_ptr()
    assert not t.is_contiguous()               # the case the bug needed
