"""§4 — Cross-attention map extraction & visualisation (v0.1: extract +
visualize only; edit/inject ship in v0.2)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..core import (MutableCache, colormap, enumerate_blocks,
                    install_attn_replace, model_key, parse_int_spec,
                    parse_layer_spec, sigma_to_bin, slice_batch, tile_grid)


def attention_probs(q, k, heads):
    """Return softmax(QK^T/√d) shaped ``[B, heads, HW, T]`` for cross-attn."""
    b, hw, _ = q.shape
    t = k.shape[1]
    dh = q.shape[-1] // heads
    q = q.reshape(b, hw, heads, dh).permute(0, 2, 1, 3)
    k = k.reshape(b, t, heads, dh).permute(0, 2, 3, 1)
    scale = dh ** -0.5
    sim = (q.float() @ k.float()) * scale
    return sim.softmax(dim=-1)


def reshape_spatial(p):
    """``[..., HW]`` → ``[..., H, W]`` if ``HW`` is square, else ``[..., HW, 1]``."""
    HW = p.shape[-1]
    side = int(round(math.sqrt(HW)))
    if side * side == HW:
        return p.reshape(*p.shape[:-1], side, side)
    return p.unsqueeze(-1)


def aggregate_maps(maps: MutableCache, layer_agg: str = "mean",
                   t_bin: int | None = None) -> torch.Tensor:
    """Reduce a recorded ATTN_MAPS cache to ``[n_tok, H, W]`` by upsampling
    every entry to the largest spatial size and mean/max-reducing."""
    entries = [(tb, t) for per_t in maps.data.values()
               for tb, t in per_t.items() if t_bin is None or tb == t_bin]
    if not entries:
        raise RuntimeError("AttentionLab: ATTN_MAPS cache is empty.")
    Hs = max(t.shape[-2] for _, t in entries)
    Ws = max(t.shape[-1] for _, t in entries)
    n_tok = entries[0][1].shape[0]
    acc = torch.zeros((n_tok, Hs, Ws))
    for _, t in entries:
        t = t.float().mean(dim=1) if t.ndim == 4 else t.float()
        t = F.interpolate(t.unsqueeze(0), size=(Hs, Ws), mode="bilinear",
                          align_corners=False)[0]
        acc = torch.maximum(acc, t) if layer_agg == "max" else acc + t
    return acc / (1 if layer_agg == "max" else len(entries))


def render_map(m: torch.Tensor, hw, cmap="viridis", overlay=None) -> torch.Tensor:
    m = m / (m.max() + 1e-8)
    m = F.interpolate(m[None, None], size=hw, mode="bilinear",
                      align_corners=False)[0, 0]
    rgb = colormap(m, cmap)
    if overlay is not None:
        rgb = 0.45 * overlay[0, :hw[0], :hw[1], :3] + 0.55 * rgb
    return rgb.clamp(0, 1)


def tokenize_words(clip, prompt: str):
    """Return ``(word_to_positions, labels)`` mapping each whitespace-separated
    word in ``prompt`` to the token positions it occupies in the encoded
    conditioning sequence.

    # SPEC-DRIFT: ComfyUI's ``return_word_ids`` increments per *weight segment*
    # (split on the embedding marker), not per English word — every plain word
    # in an unweighted prompt gets word_id 1. We instead match each word's
    # token-id run inside the full sequence.
    """
    full = clip.tokenize(prompt)
    key = next(iter(full))
    seq: list[int] = [t[0] for batch in full[key] for t in batch]

    words = prompt.split()
    word_pos: dict[int, list[int]] = {}
    labels: list[str] = []
    cursor = 0
    for wi, word in enumerate(words, 1):
        tkw = clip.tokenize(word)[key]
        ids = [t[0] for t in tkw[0]]
        # strip start/end/pad specials by intersecting with the real-token span
        ids = [i for i in ids if i not in (seq[0], seq[-1])] or ids[1:2]
        n = len(ids)
        found = None
        for i in range(cursor, len(seq) - n + 1):
            if seq[i:i + n] == ids:
                found = list(range(i, i + n))
                cursor = i + n
                break
        if found:
            word_pos[wi] = found
            labels.append(word)
    return word_pos, labels


class AttnMapExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "layers": ("STRING", {"default": "cross"}),
                "tokens": ("STRING", {"default": "all"}),
                "t_bins": ("INT", {"default": 8, "min": 1, "max": 256}),
                "keep_heads": ("BOOLEAN", {"default": False}),
                "apply_to": (("cond", "uncond", "both"), {"default": "cond"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "ATTN_MAPS", "STRING")
    RETURN_NAMES = ("model", "maps", "token_labels")
    FUNCTION = "run"
    CATEGORY = "AttentionLab/attnmaps"

    def run(self, model, layers, tokens, t_bins, keep_heads, apply_to,
            clip=None, prompt=""):
        blocks = enumerate_blocks(model)
        sel = [bid for bid in parse_layer_spec(layers, blocks) if bid[3] == "attn2"]
        if not sel:
            raise ValueError("AttentionLab: AttnMapExtract needs cross-attn "
                             "layers; spec matched only self-attn sites.")
        m = model.clone()
        token_idx, labels = self._select_tokens(tokens, clip, prompt)
        cache = MutableCache(model_key(model, blocks),
                             meta={"t_bins": t_bins, "keep_heads": keep_heads,
                                   "token_idx": token_idx, "labels": labels})

        for bid in sel:
            install_attn_replace(m, "attn2", bid[:3],
                                 self._make_hook(cache, bid, t_bins, keep_heads,
                                                 apply_to, token_idx))
        return (m, cache, ",".join(labels) if labels else "")

    @staticmethod
    def _select_tokens(spec, clip, prompt):
        spec = spec.strip()
        if spec in ("", "all", "*"):
            return None, []
        try:
            return parse_int_spec(spec, 1 << 30), spec.split(",")
        except ValueError:
            pass  # not numeric → treat as word list
        if clip is None or not prompt:
            raise ValueError("AttentionLab: word-based token selection needs "
                             "`clip` and `prompt` inputs.")
        word_pos, labels = tokenize_words(clip, prompt)
        wanted = {w.strip() for w in spec.split(",")}
        idx: list[int] = []
        out_labels: list[str] = []
        for i, lab in enumerate(labels, 1):
            if lab in wanted:
                idx.extend(word_pos[i])
                out_labels.append(lab)
        if not idx:
            raise ValueError(f"AttentionLab: none of {wanted} found in prompt words.")
        return idx, out_labels

    @staticmethod
    def _make_hook(cache, bid, t_bins, keep_heads, apply_to, token_idx):
        def fn(q, k, v, extra, prev):
            heads = extra["n_heads"]
            sl, _ = slice_batch(q, extra, apply_to)
            if q[sl].shape[0] > 0:
                probs = attention_probs(q[sl], k[sl], heads)  # [b, h, HW, T]
                if token_idx is not None:
                    idx = [i for i in token_idx if i < probs.shape[-1]]
                    probs = probs[..., idx]
                p = probs.mean(dim=0)  # [h, HW, n_tok]
                if not keep_heads:
                    p = p.mean(dim=0, keepdim=True)
                p = reshape_spatial(p.permute(2, 0, 1))  # [n_tok, h, H, W]
                tb = sigma_to_bin(extra, t_bins)
                bucket = cache.data.setdefault(bid, {})
                cur = bucket.get(tb)
                p = p.detach().to("cpu", torch.float16)
                bucket[tb] = p if cur is None else (cur + p) / 2
            return prev(q, k, v, extra)
        return fn


class AttnMapVisualize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "maps": ("ATTN_MAPS",),
                "token": ("STRING", {"default": "0"}),
                "layer_agg": (("mean", "max"), {"default": "mean"}),
                "t_agg": ("STRING", {"default": "mean"}),
                "grid_all_tokens": ("BOOLEAN", {"default": False}),
                "colormap": (("viridis", "magma", "gray"), {"default": "viridis"}),
            },
            "optional": {"overlay": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/attnmaps"

    def run(self, maps, token, layer_agg, t_agg, grid_all_tokens, colormap,
            overlay=None):
        t_bin = None if t_agg == "mean" else int(t_agg)
        agg = aggregate_maps(maps, layer_agg, t_bin)  # [n_tok, H, W]
        labels = maps.meta.get("labels") or [str(i) for i in range(agg.shape[0])]
        hw = (overlay.shape[1], overlay.shape[2]) if overlay is not None else (256, 256)
        if grid_all_tokens:
            tiles = [render_map(agg[i], hw, colormap, overlay) for i in range(agg.shape[0])]
            return (tile_grid(tiles, labels),)
        idx = labels.index(token) if token in labels else int(token)
        return (render_map(agg[idx], hw, colormap, overlay)[None],)


NODE_CLASS_MAPPINGS = {
    "AL_AttnMapExtract": AttnMapExtract,
    "AL_AttnMapVisualize": AttnMapVisualize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AL_AttnMapExtract": "🔬 AttnMap Extract",
    "AL_AttnMapVisualize": "🔬 AttnMap Visualize",
}
