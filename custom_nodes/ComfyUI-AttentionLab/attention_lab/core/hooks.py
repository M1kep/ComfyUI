"""Helpers shared by hook callbacks: timestep binning, cond/uncond batch
slicing, and per-block ``patches_replace`` chaining."""
from __future__ import annotations

import torch

from comfy.ldm.modules.attention import optimized_attention

from .blocks import BlockId


def block_id_from_extra(extra: dict, sub: str) -> BlockId:
    blk = extra.get("block", ("?", 0))
    return (blk[0], blk[1], extra.get("block_index", 0), sub)


def sigma_to_bin(extra: dict, n_bins: int) -> int:
    """Map the current sigma (``extra['sigmas']``) to a bin in ``[0, n_bins)``
    using the full schedule (``extra['sample_sigmas']``). Bin 0 = highest noise."""
    if n_bins <= 1:
        return 0
    sched = extra.get("sample_sigmas")
    cur = extra.get("sigmas")
    if sched is None or cur is None:
        return 0
    cur0 = cur[0] if cur.ndim else cur
    # last entry of sched is the terminal sigma (often 0); steps are the rest
    steps = sched[:-1] if sched.shape[0] > 1 else sched
    idx = int(torch.argmin((steps - cur0).abs()))
    n_steps = max(1, steps.shape[0])
    return min(n_bins - 1, idx * n_bins // n_steps)


def slice_batch(x: torch.Tensor, extra: dict, apply_to: str) -> tuple[slice, int]:
    """Return ``(rows, chunk)`` selecting the cond/uncond/both rows of ``x``
    according to ``extra['cond_or_uncond']`` (0=cond, 1=uncond)."""
    cou = extra.get("cond_or_uncond")
    if apply_to == "both" or not cou:
        return slice(None), x.shape[0]
    chunk = x.shape[0] // max(1, len(cou))
    want = 0 if apply_to == "cond" else 1
    if want not in cou:
        return slice(0, 0), chunk
    i = cou.index(want)
    return slice(i * chunk, (i + 1) * chunk), chunk


def default_attn(q, k, v, extra):
    return optimized_attention(q, k, v, heads=extra["n_heads"],
                               attn_precision=extra.get("attn_precision"))


def install_attn_replace(m, sub: str, patch_key: tuple, fn):
    """Register ``fn(q,k,v,extra,prev)`` as a ``patches_replace[sub][patch_key]``
    callback. If a callback already exists for that key it is passed as
    ``prev`` so this pack composes with itself; otherwise ``prev`` runs the
    stock optimized attention.

    # SPEC-DRIFT: ComfyUI's ``set_model_attn*_replace`` *overwrites* per block
    # rather than chaining (model_patcher.py:59), so we wrap manually here.
    """
    to = m.model_options.get("transformer_options", {})
    prev = to.get("patches_replace", {}).get(sub, {}).get(patch_key, default_attn)

    def wrapper(q, k, v, extra, _prev=prev, _fn=fn):
        return _fn(q, k, v, extra, _prev)

    setter = m.set_model_attn1_replace if sub == "attn1" else m.set_model_attn2_replace
    setter(wrapper, *patch_key)
