from __future__ import annotations

import logging

from ..core import enumerate_blocks, info_table

log = logging.getLogger("AttentionLab")


class ModelBlockInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("info", "n_sites", "n_cross")
    FUNCTION = "run"
    CATEGORY = "AttentionLab/util"
    OUTPUT_NODE = True

    def run(self, model):
        blocks = enumerate_blocks(model)
        n_cross = sum(1 for b in blocks if b.has_cross)
        table = info_table(blocks)
        log.info("[AttentionLab] %d attention sites (%d cross)\n%s",
                 len(blocks), n_cross, table)
        return {"ui": {"text": [table]}, "result": (table, len(blocks), n_cross)}


_BARRIER_HELP = (
    "AttentionLab: cache is empty. The recorder node returns an empty cache "
    "that is filled DURING sampling — wire the KSampler's LATENT output into "
    "this barrier's `latent_trigger` so it executes after sampling.")


class CacheBarrier:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cache": ("ACTIVATION_CACHE",),
                             "latent_trigger": ("LATENT",)}}

    RETURN_TYPES = ("ACTIVATION_CACHE",)
    RETURN_NAMES = ("cache",)
    FUNCTION = "run"
    CATEGORY = "AttentionLab/util"

    def run(self, cache, latent_trigger):
        if not cache.is_filled():
            raise RuntimeError(_BARRIER_HELP)
        return (cache,)


class AttnMapBarrier(CacheBarrier):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cache": ("ATTN_MAPS",),
                             "latent_trigger": ("LATENT",)}}

    RETURN_TYPES = ("ATTN_MAPS",)
    RETURN_NAMES = ("maps",)


NODE_CLASS_MAPPINGS = {
    "AL_ModelBlockInfo": ModelBlockInfo,
    "AL_CacheBarrier": CacheBarrier,
    "AL_AttnMapBarrier": AttnMapBarrier,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AL_ModelBlockInfo": "🔬 Model Block Info",
    "AL_CacheBarrier": "🔬 Cache Barrier",
    "AL_AttnMapBarrier": "🔬 AttnMap Barrier",
}
