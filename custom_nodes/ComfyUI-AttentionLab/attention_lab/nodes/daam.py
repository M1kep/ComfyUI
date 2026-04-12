"""§6 — DAAM convenience wrapper (zero-config per-word attention heatmaps)."""
from __future__ import annotations

import torch

from ..core import (MutableCache, enumerate_blocks, install_attn_replace,
                    model_key, sigma_to_bin, slice_batch, tile_grid)
from .attnmaps import (aggregate_maps, attention_probs, render_map,
                       reshape_spatial, tokenize_words)


class DAAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": "a cat on a table"}),
                "layer_agg": (("mean", "max"), {"default": "mean"}),
                "t_agg": (("mean",), {"default": "mean"}),
                "overlay_decoded": ("BOOLEAN", {"default": True}),
            },
            "optional": {"vae": ("VAE",)},
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "ATTN_MAPS")
    RETURN_NAMES = ("model", "heatmaps", "maps")
    FUNCTION = "run"
    CATEGORY = "AttentionLab/daam"

    # SPEC-DRIFT: ComfyUI's lazy evaluation cannot produce both the patched
    # MODEL (consumed by the sampler) and a heatmap IMAGE (requires the cache
    # to already be filled by that same sampler) from a single node without an
    # ordering trigger. We therefore split DAAM into a recorder + a renderer
    # connected through an internal LATENT trigger, exactly the §0.4 pattern.
    def run(self, model, clip, prompt, layer_agg, t_agg, overlay_decoded, vae=None):
        del t_agg, vae  # spec inputs; reserved for v0.2
        blocks = enumerate_blocks(model)
        m = model.clone()
        word_pos, labels = tokenize_words(clip, prompt)
        cache = MutableCache(model_key(model, blocks), meta={
            "t_bins": 4, "labels": labels, "word_pos": word_pos,
            "layer_agg": layer_agg, "overlay_decoded": overlay_decoded,
        })

        # Precompute the BPE→word merge as an index tensor so the hook is cheap.
        positions = sorted({p for ps in word_pos.values() for p in ps})
        col_of = {c: i for i, c in enumerate(positions)}
        gather = torch.full((len(positions),), -1, dtype=torch.long)
        for wi, w in enumerate(word_pos):
            for c in word_pos[w]:
                gather[col_of[c]] = wi

        for b in blocks:
            if b.sub == "attn2":
                install_attn_replace(m, "attn2", b.patch_key,
                                     self._hook(cache, b.id, positions, gather,
                                                len(labels)))
        return (m, torch.zeros((1, 64, 64, 3)), cache)

    @staticmethod
    def _hook(cache, bid, positions, gather, n_words):
        def fn(q, k, v, extra, prev):
            sl, _ = slice_batch(q, extra, "cond")
            if q[sl].shape[0] > 0 and positions:
                probs = attention_probs(q[sl], k[sl], extra["n_heads"])
                cols = [p for p in positions if p < probs.shape[-1]]
                if cols:
                    p = probs[..., cols].mean(dim=(0, 1))  # [HW, n_pos]
                    g = gather[:len(cols)].to(p.device)
                    merged = torch.zeros((p.shape[0], n_words),
                                         device=p.device, dtype=p.dtype)
                    merged.scatter_add_(1, g.expand_as(p), p)
                    merged = reshape_spatial(merged.permute(1, 0))  # [n_words,H,W]
                    tb = sigma_to_bin(extra, cache.meta["t_bins"])
                    bucket = cache.data.setdefault(bid, {})
                    cur = bucket.get(tb)
                    m_cpu = merged.detach().to("cpu", torch.float16)
                    bucket[tb] = m_cpu if cur is None else (cur + m_cpu) / 2
            return prev(q, k, v, extra)
        return fn


class DAAMRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "maps": ("ATTN_MAPS",),
                "latent_trigger": ("LATENT",),
            },
            "optional": {"overlay": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmaps",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/daam"

    def run(self, maps, latent_trigger, overlay=None):
        if not maps.is_filled():
            raise RuntimeError(
                "AttentionLab: DAAM cache empty. Wire the KSampler LATENT "
                "into `latent_trigger` so this runs after sampling.")
        labels = maps.meta["labels"]
        agg = aggregate_maps(maps, maps.meta.get("layer_agg", "mean"))
        ov = overlay if maps.meta.get("overlay_decoded", True) else None
        hw = (overlay.shape[1], overlay.shape[2]) if overlay is not None else (256, 256)
        tiles = [render_map(agg[i], hw, "viridis", ov) for i in range(len(labels))]
        return (tile_grid(tiles, labels),)


NODE_CLASS_MAPPINGS = {
    "AL_DAAM": DAAM,
    "AL_DAAMRender": DAAMRender,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AL_DAAM": "🔬 DAAM",
    "AL_DAAMRender": "🔬 DAAM Render",
}
