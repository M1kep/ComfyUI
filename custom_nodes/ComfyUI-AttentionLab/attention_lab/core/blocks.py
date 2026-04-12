"""Block addressing for SD1.5 / SDXL UNets.

A BlockId is ``(stage, block_idx, transformer_depth, sub)`` where the first
three elements exactly match ComfyUI's ``patches_replace`` block key
``(block_name, number, transformer_index)`` and ``sub`` is ``"attn1"`` or
``"attn2"``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

BlockId = tuple  # (stage:str, idx:int, depth:int, sub:str)


def friendly(bid: BlockId) -> str:
    s, i, d, sub = bid
    short = {"input": "in", "middle": "mid", "output": "out"}.get(s, s)
    return f"{short}.{i}{f'/{d}' if d else ''}.{sub}"


@dataclass(frozen=True)
class BlockInfo:
    id: BlockId
    n_heads: int
    dim_head: int
    has_cross: bool
    is_joint: bool = False

    @property
    def stage(self) -> str:
        return self.id[0]

    @property
    def sub(self) -> str:
        return self.id[3]

    @property
    def dim(self) -> int:
        return self.n_heads * self.dim_head

    @property
    def patch_key(self) -> tuple:
        """The 3-tuple ComfyUI uses to address ``patches_replace`` entries."""
        return self.id[:3]

    @property
    def friendly(self) -> str:
        return friendly(self.id)


def _walk_unet_transformers(dm) -> Iterable[tuple[str, int, int, object]]:
    """Yield ``(stage, block_idx, depth, BasicTransformerBlock)`` matching the
    addressing ComfyUI's ``UNetModel.forward`` assigns at runtime."""
    groups = [("input", list(getattr(dm, "input_blocks", []) or []))]
    mid = getattr(dm, "middle_block", None)
    if mid is not None:
        groups.append(("middle", [mid]))
    groups.append(("output", list(getattr(dm, "output_blocks", []) or [])))

    for stage, modules in groups:
        for idx, ts in enumerate(modules):
            for layer in ts:
                tblocks = getattr(layer, "transformer_blocks", None)
                if tblocks is None:
                    continue
                for depth, tb in enumerate(tblocks):
                    yield stage, idx, depth, tb


def enumerate_blocks(model) -> list[BlockInfo]:
    """Enumerate every attention site in the UNet attached to ``model``
    (a ``ModelPatcher``). SD1.5 / SDXL only for now."""
    dm = model.model.diffusion_model
    out: list[BlockInfo] = []
    for stage, idx, depth, tb in _walk_unet_transformers(dm):
        out.append(BlockInfo(id=(stage, idx, depth, "attn1"),
                             n_heads=tb.n_heads, dim_head=tb.d_head, has_cross=False))
        if tb.attn2 is not None:
            out.append(BlockInfo(id=(stage, idx, depth, "attn2"),
                                 n_heads=tb.n_heads, dim_head=tb.d_head, has_cross=True))
    if not out:
        raise RuntimeError(
            "AttentionLab: could not find any transformer blocks. "
            f"Model type {type(dm).__name__} is not supported (SD1.5/SDXL only in v0.1).")
    return out


def model_key(model, blocks: list[BlockInfo] | None = None) -> str:
    n = len(blocks if blocks is not None else enumerate_blocks(model))
    return f"{type(model.model.diffusion_model).__name__}:{n}"


def index_blocks(blocks: list[BlockInfo]) -> dict[BlockId, BlockInfo]:
    return {b.id: b for b in blocks}


def info_table(blocks: list[BlockInfo]) -> str:
    lines = [f"{'#':>3}  {'id':<18} {'heads':>5} {'d_head':>6} {'dim':>5}  cross"]
    for i, b in enumerate(blocks):
        lines.append(
            f"{i:>3}  {b.friendly:<18} {b.n_heads:>5} {b.dim_head:>6} {b.dim:>5}  "
            f"{'✓' if b.has_cross else '·'}")
    return "\n".join(lines)
