"""§1 — Per-layer × per-timestep injection matrix."""
from __future__ import annotations

import torch

from ..core import (block_id_from_extra, enumerate_blocks, model_key,
                    parse_layer_spec, render_heatmap, sigma_to_bin, slice_batch)

PRESETS = ("flat", "linear_in", "linear_out", "ease_in_out", "style",
           "composition", "strong_middle")
TARGETS = ("attn2_out", "attn2_kv", "cond_embed")
# SPEC-DRIFT: ipadapter_kv descoped — IPAdapter not present in this checkout
# and there is no clean seam to detect concatenated image K/V (see DECISIONS.md).


def _preset_row(preset: str, t_bins: int, value: float, b) -> torch.Tensor:
    t = torch.linspace(0, 1, t_bins)
    if preset == "flat":
        row = torch.full((t_bins,), 1.0)
    elif preset == "linear_in":
        row = t
    elif preset == "linear_out":
        row = 1.0 - t
    elif preset == "ease_in_out":
        row = 0.5 - 0.5 * torch.cos(torch.pi * t)
    elif preset == "strong_middle":
        row = (-4 * (t - 0.5) ** 2 + 1).clamp(0)
    elif preset == "style":
        # IP-Adapter "style" weighting: emphasise high-dim (low-res) blocks.
        row = torch.full((t_bins,), 1.0 if b.dim >= 1280 else 0.2)
    elif preset == "composition":
        row = torch.full((t_bins,), 1.0 if b.stage in ("input", "middle") else 0.3)
    else:
        row = torch.full((t_bins,), 1.0)
    return row * value


def _build_matrix(model, t_bins: int, preset: str, value: float) -> dict:
    blocks = enumerate_blocks(model)
    mat = torch.stack([_preset_row(preset, t_bins, value, b) for b in blocks])
    return {"matrix": mat, "block_ids": [b.id for b in blocks],
            "t_bins": t_bins, "model_key": model_key(model, blocks)}


class InjectionMatrix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "t_bins": ("INT", {"default": 16, "min": 1, "max": 256}),
            "preset": (PRESETS, {"default": "flat"}),
            "value": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("INJECTION_MATRIX",)
    RETURN_NAMES = ("matrix",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/matrix"

    def run(self, model, t_bins, preset, value):
        return (_build_matrix(model, t_bins, preset, value),)


class InjectionMatrixEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "matrix": ("INJECTION_MATRIX",),
            "model": ("MODEL",),
            "layers": ("STRING", {"default": "all"}),
            "t_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "t_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "op": (("set", "add", "mul"), {"default": "set"}),
            "value": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("INJECTION_MATRIX",)
    RETURN_NAMES = ("matrix",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/matrix"

    def run(self, matrix, model, layers, t_start, t_end, op, value):
        blocks = enumerate_blocks(model)
        sel = set(parse_layer_spec(layers, blocks))
        ids = matrix["block_ids"]
        T = matrix["t_bins"]
        c0, c1 = int(t_start * T), max(int(t_start * T) + 1, int(round(t_end * T)))
        mat = matrix["matrix"].clone()
        rows = [i for i, bid in enumerate(ids) if bid in sel]
        sub = mat[rows, c0:c1]
        if op == "set":
            sub[:] = value
        elif op == "add":
            sub += value
        else:
            sub *= value
        mat[rows, c0:c1] = sub
        return ({**matrix, "matrix": mat},)


class InjectionMatrixFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "model": ("MODEL",),
            "t_bins": ("INT", {"default": 16, "min": 1, "max": 256}),
            "min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            "max": ("FLOAT", {"default": 1.5, "min": -10.0, "max": 10.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("INJECTION_MATRIX",)
    RETURN_NAMES = ("matrix",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/matrix"

    def run(self, image, model, t_bins, min, max):
        blocks = enumerate_blocks(model)
        L = len(blocks)
        gray = image[0].mean(dim=-1, keepdim=True).permute(2, 0, 1)[None]
        mat = torch.nn.functional.interpolate(gray, size=(L, t_bins), mode="bilinear",
                                              align_corners=False)[0, 0]
        mat = min + mat * (max - min)
        return ({"matrix": mat, "block_ids": [b.id for b in blocks],
                 "t_bins": t_bins, "model_key": model_key(model, blocks)},)


class InjectionMatrixPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "matrix": ("INJECTION_MATRIX",),
            "colormap": (("viridis", "magma", "gray"), {"default": "viridis"}),
            "annotate": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/matrix"

    def run(self, matrix, colormap, annotate):
        from ..core import friendly
        labels = [friendly(bid) for bid in matrix["block_ids"]] if annotate else None
        return (render_heatmap(matrix["matrix"], labels, "t", colormap),)


class ApplyInjectionMatrix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "matrix": ("INJECTION_MATRIX",),
            "target": (TARGETS, {"default": "attn2_out"}),
            "apply_to": (("cond", "uncond", "both"), {"default": "cond"}),
            "cond_index": ("INT", {"default": 0, "min": 0, "max": 64}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/matrix"

    def run(self, model, matrix, target, apply_to, cond_index):
        key = model_key(model)
        if matrix["model_key"] != key:
            raise ValueError(
                f"AttentionLab: matrix was built for {matrix['model_key']!r} "
                f"but this model is {key!r}. Rebuild the matrix from this model.")
        m = model.clone()
        ids = matrix["block_ids"]
        row_of = {bid: i for i, bid in enumerate(ids)}
        T = matrix["t_bins"]
        mat = matrix["matrix"]

        if target == "cond_embed":
            self._apply_cond_embed(m, mat, T, cond_index)
            return (m,)

        sub = "attn2"

        def hook_out(n, extra):
            bid = block_id_from_extra(extra, sub)
            r = row_of.get(bid)
            if r is None:
                return n
            w = float(mat[r, sigma_to_bin(extra, T)])
            if w == 1.0:
                return n
            sl, _ = slice_batch(n, extra, apply_to)
            n = n.clone()
            n[sl] = n[sl] * w
            return n

        def hook_kv(q, ctx, val, extra):
            bid = block_id_from_extra(extra, sub)
            r = row_of.get(bid)
            if r is None:
                return q, ctx, val
            w = float(mat[r, sigma_to_bin(extra, T)])
            if w == 1.0:
                return q, ctx, val
            sl, _ = slice_batch(ctx, extra, apply_to)
            ctx = ctx.clone()
            ctx[sl] = ctx[sl] * w
            if val is not None:
                val = val.clone()
                val[sl] = val[sl] * w
            return q, ctx, val

        if target == "attn2_out":
            m.set_model_attn2_output_patch(hook_out)
        else:  # attn2_kv
            m.set_model_attn2_patch(hook_kv)
        return (m,)

    @staticmethod
    def _apply_cond_embed(m, mat, T, cond_index):  # noqa: ARG004 (reserved for v0.2)
        per_t = mat.mean(dim=0)  # collapse layer axis

        def wrapper(apply_model, args):
            c = {**args["c"]}
            cross = c.get("c_crossattn")
            if cross is not None:
                # Use a unet wrapper to intercept timestep — sample_sigmas lives
                # in transformer_options which c also carries.
                to = c.get("transformer_options", {})
                tb = sigma_to_bin({"sigmas": args["timestep"],
                                   "sample_sigmas": to.get("sample_sigmas")}, T)
                w = float(per_t[tb])
                sl, _ = slice_batch(cross, {"cond_or_uncond": args.get("cond_or_uncond", [])}, "cond")
                if cross[sl].shape[0]:
                    cross = cross.clone()
                    cross[sl] *= w
                    c["c_crossattn"] = cross
            return apply_model(args["input"], args["timestep"], **c)

        m.set_model_unet_function_wrapper(wrapper)


NODE_CLASS_MAPPINGS = {
    "AL_InjectionMatrix": InjectionMatrix,
    "AL_InjectionMatrixEdit": InjectionMatrixEdit,
    "AL_InjectionMatrixFromImage": InjectionMatrixFromImage,
    "AL_InjectionMatrixPreview": InjectionMatrixPreview,
    "AL_ApplyInjectionMatrix": ApplyInjectionMatrix,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AL_InjectionMatrix": "🔬 Injection Matrix",
    "AL_InjectionMatrixEdit": "🔬 Injection Matrix Edit",
    "AL_InjectionMatrixFromImage": "🔬 Injection Matrix (from Image)",
    "AL_InjectionMatrixPreview": "🔬 Injection Matrix Preview",
    "AL_ApplyInjectionMatrix": "🔬 Apply Injection Matrix",
}
