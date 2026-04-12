"""Unit tests for AttentionLab core infra. Run with:
    python3 -m pytest custom_nodes/ComfyUI-AttentionLab/tests -q
from the ComfyUI root (CPU mode)."""
from types import SimpleNamespace

import pytest
import torch

import comfy.ops
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from attention_lab.core import (enumerate_blocks, model_key, parse_layer_spec,
                                sigma_to_bin, slice_batch)
from attention_lab.core.viz import render_heatmap, tile_grid


# SDXL-base UNet config (matches comfy/supported_models.py:SDXL)
SDXL_CFG = dict(
    image_size=32, in_channels=4, out_channels=4, model_channels=320,
    num_res_blocks=2, channel_mult=[1, 2, 4], num_head_channels=64,
    transformer_depth=[0, 0, 2, 2, 10, 10],
    transformer_depth_output=[0, 0, 0, 2, 2, 2, 10, 10, 10],
    transformer_depth_middle=10,
    context_dim=2048, use_linear_in_transformer=True, use_temporal_attention=False,
    adm_in_channels=2816, use_spatial_transformer=True, legacy=False,
    num_classes="sequential", dtype=torch.float32, device="meta",
    operations=comfy.ops.manual_cast,
)

SD15_CFG = dict(
    image_size=32, in_channels=4, out_channels=4, model_channels=320,
    num_res_blocks=2, channel_mult=[1, 2, 4, 4], num_heads=8,
    transformer_depth=[1, 1, 1, 1, 1, 1, 0, 0],
    transformer_depth_output=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    transformer_depth_middle=1,
    context_dim=768, use_spatial_transformer=True, legacy=False,
    use_temporal_attention=False, dtype=torch.float32, device="meta",
    operations=comfy.ops.manual_cast,
)


def make_patcher(cfg):
    unet = UNetModel(**cfg)
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=unet))


@pytest.fixture(scope="module")
def sdxl():
    return make_patcher(SDXL_CFG)


@pytest.fixture(scope="module")
def sd15():
    return make_patcher(SD15_CFG)


def test_enumerate_sdxl(sdxl):
    blocks = enumerate_blocks(sdxl)
    n_cross = sum(1 for b in blocks if b.has_cross)
    # spec acceptance: ≥70 attn sites on SDXL
    assert len(blocks) >= 70, f"only {len(blocks)} sites"
    assert n_cross == len(blocks) // 2
    # verify n_heads/dim match config: head_channels=64, ch_mult gives 320/640/1280
    dims = {b.dim for b in blocks}
    assert dims <= {320, 640, 1280}, dims
    for b in blocks:
        assert b.dim_head == 64
        assert b.n_heads == b.dim // 64
    assert "UNetModel:" in model_key(sdxl)


def test_enumerate_sd15(sd15):
    blocks = enumerate_blocks(sd15)
    assert len(blocks) == 32  # 16 transformer blocks × {attn1, attn2}
    assert all(b.n_heads == 8 for b in blocks)


def test_layer_spec(sdxl):
    blocks = enumerate_blocks(sdxl)
    assert len(parse_layer_spec("all", blocks)) == len(blocks)
    cross = parse_layer_spec("cross", blocks)
    assert all(bid[3] == "attn2" for bid in cross)
    selfs = parse_layer_spec("self", blocks)
    assert all(bid[3] == "attn1" for bid in selfs)
    mid = parse_layer_spec("mid", blocks)
    assert all(bid[0] == "middle" for bid in mid)
    out04 = parse_layer_spec("out.0-4", blocks)
    assert all(bid[0] == "output" and 0 <= bid[1] <= 4 for bid in out04)
    union = parse_layer_spec("out.4,out.5,mid", blocks)
    assert {bid[0] for bid in union} == {"output", "middle"}
    with pytest.raises(ValueError):
        parse_layer_spec("nonsense", blocks)
    with pytest.raises(ValueError):
        parse_layer_spec("in.99", blocks)


def test_layer_spec_sub_filter(sdxl):
    blocks = enumerate_blocks(sdxl)
    sel = parse_layer_spec("mid attn2", blocks)
    assert all(bid[0] == "middle" and bid[3] == "attn2" for bid in sel)


def test_sigma_to_bin():
    sched = torch.linspace(14.0, 0.0, 21)  # 20 steps + terminal 0
    extra = {"sample_sigmas": sched, "sigmas": sched[:1]}
    assert sigma_to_bin(extra, 8) == 0
    extra["sigmas"] = sched[10:11]
    assert sigma_to_bin(extra, 8) == 4
    extra["sigmas"] = sched[19:20]
    assert sigma_to_bin(extra, 8) == 7
    assert sigma_to_bin({"sigmas": None, "sample_sigmas": None}, 8) == 0


def test_slice_batch():
    x = torch.zeros((4, 5, 6))
    sl, ch = slice_batch(x, {"cond_or_uncond": [1, 0]}, "cond")
    assert ch == 2 and x[sl].shape[0] == 2 and sl == slice(2, 4)
    sl, _ = slice_batch(x, {"cond_or_uncond": [1, 0]}, "uncond")
    assert sl == slice(0, 2)
    sl, _ = slice_batch(x, {"cond_or_uncond": [0]}, "uncond")
    assert x[sl].shape[0] == 0
    sl, _ = slice_batch(x, {}, "cond")
    assert x[sl].shape[0] == 4


def test_render_heatmap_shape():
    mat = torch.rand(10, 8)
    img = render_heatmap(mat, [f"r{i}" for i in range(10)])
    assert img.ndim == 4 and img.shape[0] == 1 and img.shape[-1] == 3


def test_tile_grid():
    tiles = [torch.rand(32, 32, 3) for _ in range(5)]
    img = tile_grid(tiles, [f"t{i}" for i in range(5)])
    assert img.ndim == 4 and img.shape[-1] == 3
