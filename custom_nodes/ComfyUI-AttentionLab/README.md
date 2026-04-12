# ComfyUI-AttentionLab

Interpretability & surgical manipulation of diffusion-model internals
(cross-attention, attention heads) as composable ComfyUI nodes.

**v0.1** ships:

| Group | Nodes |
| --- | --- |
| `util/` | Model Block Info · Cache Barrier · AttnMap Barrier |
| `matrix/` | Injection Matrix · …Edit · …(from Image) · …Preview · Apply Injection Matrix |
| `heads/` | Head Mask · …Edit · …Solo · Apply Head Mask · Head Sweep |
| `attnmaps/` | AttnMap Extract · AttnMap Visualize |
| `daam/` | DAAM · DAAM Render |

SD1.5 / SDXL only. Flux/SD3 are out of scope for this release.

## Quick start

1. Drop **🔬 Model Block Info** after your checkpoint loader to see every
   addressable attention site.
2. For per-layer×timestep conditioning control, chain
   `Injection Matrix → Injection Matrix Edit → Apply Injection Matrix → KSampler`
   and wire `Injection Matrix Preview` off the side.
3. For a one-click per-head atlas, use **🔬 Head Sweep** (it samples
   `n_heads + 1` times internally).
4. For per-word attention heatmaps:

```
DAAM ──model──▶ KSampler ──latent──▶ VAEDecode ──image──▶ DAAM Render ──▶ Preview
   └──maps────────────────────────────────────────────────▶┘ overlay ◀──┘
```

## Layer-spec grammar

```
all                  → every site
cross | self         → attn2-only / attn1-only
in.* | mid | out.*   → stage filter
out.0-5              → range within stage
out.4,out.7,mid      → comma-separated union
mid attn2            → stage + sub filter
```

## Socket types

`INJECTION_MATRIX`, `HEAD_MASK`, `ATTN_MAPS`, `ACTIVATION_CACHE` are custom
types introduced by this pack. `ATTN_MAPS` / `ACTIVATION_CACHE` are
*mutable* — they are returned empty and filled during sampling, so any
consumer must be sequenced after the sampler via a barrier node (or
`DAAM Render`'s `latent_trigger`). See `docs/DECISIONS.md` for the
execution-ordering rationale.

## Tests

```
cd custom_nodes/ComfyUI-AttentionLab/tests
python3 -m pytest -q
```
