"""Dependency-free heatmap rendering (no matplotlib)."""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageDraw


_VIRIDIS = np.array([
    [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
    [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
], dtype=np.float32)

_MAGMA = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
    [245, 125, 21], [252, 193, 65], [252, 253, 191],
], dtype=np.float32)

_MAPS = {"viridis": _VIRIDIS, "magma": _MAGMA, "gray": None}


def colormap(x: torch.Tensor, name: str = "viridis") -> torch.Tensor:
    """Map a ``[...]``-shaped float tensor in [0,1] to ``[..., 3]`` RGB in [0,1]."""
    x = x.clamp(0, 1).float().cpu()
    if name == "gray" or name not in _MAPS:
        return x.unsqueeze(-1).expand(*x.shape, 3)
    anchors = torch.from_numpy(_MAPS[name] / 255.0)
    pos = x * (anchors.shape[0] - 1)
    lo = pos.floor().long().clamp(0, anchors.shape[0] - 2)
    frac = (pos - lo.float()).unsqueeze(-1)
    return anchors[lo] * (1 - frac) + anchors[lo + 1] * frac


def render_heatmap(mat: torch.Tensor, row_labels=None, col_label="t",
                   cmap="viridis", cell=14, label_w=110) -> torch.Tensor:
    """Render a 2-D matrix as a labelled heatmap. Returns a ComfyUI IMAGE
    tensor ``[1,H,W,3]``."""
    mat = mat.detach().float().cpu()
    L, T = mat.shape
    lo, hi = float(mat.min()), float(mat.max())
    norm = (mat - lo) / (hi - lo + 1e-8)
    rgb = (colormap(norm, cmap).numpy() * 255).astype("uint8")

    img = Image.new("RGB", (label_w + T * cell, L * cell + 18), (20, 20, 20))
    img.paste(Image.fromarray(rgb).resize((T * cell, L * cell), Image.NEAREST),
              (label_w, 0))
    d = ImageDraw.Draw(img)
    if row_labels:
        for i, lab in enumerate(row_labels):
            d.text((4, i * cell + 1), str(lab)[:16], fill=(200, 200, 200))
    d.text((4, L * cell + 3),
           f"{col_label}: 0→{T - 1}   range [{lo:.2f}, {hi:.2f}]",
           fill=(200, 200, 200))
    return pil_to_image(img)


def tile_grid(images: list[torch.Tensor], labels: list[str] | None = None,
              cols: int = 0, label_h: int = 16) -> torch.Tensor:
    """Tile ``[H,W,3]`` tensors into a single labelled grid IMAGE."""
    if not images:
        return torch.zeros((1, 8, 8, 3))
    H, W = images[0].shape[:2]
    n = len(images)
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    out = Image.new("RGB", (cols * W, rows * (H + label_h)), (0, 0, 0))
    d = ImageDraw.Draw(out)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * W, r * (H + label_h)
        arr = (im.nan_to_num(0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        out.paste(Image.fromarray(arr), (x, y))
        if labels:
            d.text((x + 2, y + H + 1), str(labels[i])[:24], fill=(230, 230, 230))
    return pil_to_image(out)


def pil_to_image(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB"))
                            .astype(np.float32) / 255.0)[None]
