"""§5 — Head-level ablation / isolation."""
from __future__ import annotations

import logging
import random
import re

import torch

import comfy.samplers
import comfy.utils
import nodes as comfy_nodes

from ..core import (enumerate_blocks, index_blocks, install_attn_replace,
                    model_key, parse_int_spec, parse_layer_spec, sigma_to_bin,
                    slice_batch, tile_grid)

log = logging.getLogger("AttentionLab")


def _parse_heads(spec: str, n_heads: int) -> list[int]:
    m = re.fullmatch(r"rand:(\d+)", spec.strip())
    if m:
        return random.sample(range(n_heads), min(int(m.group(1)), n_heads))
    return parse_int_spec(spec, n_heads)


def _new_mask(blocks, key: str, target: str, init: float) -> dict:
    want = ("attn1", "attn2") if target == "both" else (target,)
    mask = {b.id: torch.full((b.n_heads,), float(init))
            for b in blocks if b.sub in want}
    return {"mask": mask, "model_key": key}


class HeadMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "target": (("attn1", "attn2", "both"), {"default": "attn2"}),
            "init": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("HEAD_MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/heads"

    def run(self, model, target, init):
        blocks = enumerate_blocks(model)
        return (_new_mask(blocks, model_key(model, blocks), target, init),)


class HeadMaskEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("HEAD_MASK",),
            "model": ("MODEL",),
            "layers": ("STRING", {"default": "all"}),
            "heads": ("STRING", {"default": "all"}),
            "value": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "op": (("set", "mul"), {"default": "set"}),
        }}

    RETURN_TYPES = ("HEAD_MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/heads"

    def run(self, mask, model, layers, heads, value, op):
        blocks = enumerate_blocks(model)
        sel = set(parse_layer_spec(layers, blocks))
        new = {bid: v.clone() for bid, v in mask["mask"].items()}
        for bid, v in new.items():
            if bid not in sel:
                continue
            idx = _parse_heads(heads, v.shape[0])
            if op == "set":
                v[idx] = value
            else:
                v[idx] *= value
        return ({"mask": new, "model_key": mask["model_key"]},)


class HeadMaskSolo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("HEAD_MASK",),
            "model": ("MODEL",),
            "layer": ("STRING", {"default": "mid attn2"}),
            "head": ("INT", {"default": 0, "min": 0, "max": 256}),
        }}

    RETURN_TYPES = ("HEAD_MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/heads"

    def run(self, mask, model, layer, head):
        blocks = enumerate_blocks(model)
        sel = set(parse_layer_spec(layer, blocks))
        new = {bid: v.clone() for bid, v in mask["mask"].items()}
        for bid in sel:
            if bid in new:
                v = new[bid]
                v[:] = 0.0
                if 0 <= head < v.shape[0]:
                    v[head] = 1.0
        return ({"mask": new, "model_key": mask["model_key"]},)


def _apply_mask_hook(mask_vec: torch.Tensor, mode: str, t_start: float,
                     t_end: float, apply_to: str):
    def fn(q, k, v, extra, prev):
        out = prev(q, k, v, extra)  # [B, HW, heads*dh]
        tb = sigma_to_bin(extra, 64) / 63.0 if extra.get("sample_sigmas") is not None else 0.5
        if not (t_start <= tb <= t_end):
            return out
        heads = extra["n_heads"]
        dh = extra["dim_head"]
        sl, _ = slice_batch(out, extra, apply_to)
        if out[sl].shape[0] == 0:
            return out
        if sl != slice(None):
            out = out.clone()
        B, HW, _ = out[sl].shape
        h = out[sl].reshape(B, HW, heads, dh)
        mvec = mask_vec.to(h.device, h.dtype).view(1, 1, heads, 1)
        if mode in ("zero", "scale"):
            h = h * mvec
        else:
            keep = mvec >= 0.5
            other = h.masked_fill(~keep, 0)
            n_keep = keep.sum().clamp(min=1)
            if mode == "mean_replace":
                repl = other.sum(dim=2, keepdim=True) / n_keep
            else:  # noise_replace
                std = other.float().std(dim=2, keepdim=True).to(h.dtype)
                repl = torch.randn_like(h) * std
            h = torch.where(keep, h, repl)
        out[sl] = h.reshape(B, HW, heads * dh)
        return out
    return fn


class ApplyHeadMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mask": ("HEAD_MASK",),
            "mode": (("zero", "mean_replace", "noise_replace", "scale"),
                     {"default": "zero"}),
            "apply_to": (("cond", "uncond", "both"), {"default": "both"}),
            "t_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "t_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/heads"

    def run(self, model, mask, mode, apply_to, t_start, t_end):
        key = model_key(model)
        if mask["model_key"] != key:
            raise ValueError(
                f"AttentionLab: head mask was built for {mask['model_key']!r} "
                f"but this model is {key!r}.")
        m = model.clone()
        blocks = index_blocks(enumerate_blocks(model))
        for bid, vec in mask["mask"].items():
            if torch.all(vec == 1.0) and mode in ("zero", "scale"):
                continue  # identity for this block
            info = blocks[bid]
            install_attn_replace(
                m, info.sub, info.patch_key,
                _apply_mask_hook(vec, mode, t_start, t_end, apply_to))
        return (m,)


class HeadSweep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "layer": ("STRING", {"default": "mid attn2"}),
            "sweep": (("ablate_each", "solo_each"), {"default": "ablate_each"}),
            "mode": (("zero", "mean_replace", "noise_replace", "scale"),
                     {"default": "mean_replace"}),
            "cols": ("INT", {"default": 0, "min": 0, "max": 16}),
            "include_baseline": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("grid", "head_labels")
    FUNCTION = "run"
    CATEGORY = "AttentionLab/heads"

    def run(self, model, positive, negative, latent, vae, seed, steps, cfg,
            sampler_name, scheduler, layer, sweep, mode, cols, include_baseline):
        blocks = enumerate_blocks(model)
        sel = parse_layer_spec(layer, blocks)
        bid = sel[0]
        info = index_blocks(blocks)[bid]
        n_heads = info.n_heads
        labels: list[str] = []
        tiles: list[torch.Tensor] = []

        def render(m, label):
            (lat,) = comfy_nodes.common_ksampler(
                m, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent, denoise=1.0)
            img = vae.decode(lat["samples"])
            if img.ndim == 5:
                img = img.reshape(-1, *img.shape[-3:])
            tiles.append(img[0])
            labels.append(label)

        if include_baseline:
            log.info("[AttentionLab] HeadSweep baseline")
            render(model, "baseline")

        pbar = comfy.utils.ProgressBar(n_heads)
        key = model_key(model, blocks)
        for h in range(n_heads):
            log.info("[AttentionLab] HeadSweep %s head %d/%d (%s)",
                     info.friendly, h + 1, n_heads, sweep)
            base = _new_mask(blocks, key, info.sub, 1.0)
            vec = base["mask"][bid]
            if sweep == "solo_each":
                vec[:] = 0.0
                vec[h] = 1.0
            else:
                vec[h] = 0.0
            (patched,) = ApplyHeadMask().run(model, base, mode, "both", 0.0, 1.0)
            render(patched, f"h{h}")
            pbar.update(1)

        return (tile_grid(tiles, labels, cols=cols), ",".join(labels))


NODE_CLASS_MAPPINGS = {
    "AL_HeadMask": HeadMask,
    "AL_HeadMaskEdit": HeadMaskEdit,
    "AL_HeadMaskSolo": HeadMaskSolo,
    "AL_ApplyHeadMask": ApplyHeadMask,
    "AL_HeadSweep": HeadSweep,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AL_HeadMask": "🔬 Head Mask",
    "AL_HeadMaskEdit": "🔬 Head Mask Edit",
    "AL_HeadMaskSolo": "🔬 Head Mask Solo",
    "AL_ApplyHeadMask": "🔬 Apply Head Mask",
    "AL_HeadSweep": "🔬 Head Sweep",
}
