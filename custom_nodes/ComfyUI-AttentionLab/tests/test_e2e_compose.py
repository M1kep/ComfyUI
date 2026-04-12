"""Verify hooks compose and identity is a no-op by comparing *latents*
(the random-weight VAE saturates so image-space comparison is useless)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "custom_nodes" / "ComfyUI-AttentionLab"))

from comfy import cli_args  # noqa: E402
cli_args.args.cpu = True
cli_args.args.force_fp32 = True

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import nodes as comfy_nodes  # noqa: E402

from attention_lab.nodes.matrix import InjectionMatrix, ApplyInjectionMatrix  # noqa: E402
from attention_lab.nodes.heads import HeadMask, ApplyHeadMask  # noqa: E402
from attention_lab.nodes.daam import DAAM  # noqa: E402

CKPT = str(ROOT / "models" / "checkpoints" / "sd15_dummy.safetensors")


def main():
    out = comfy.sd.load_checkpoint_guess_config(CKPT)
    model, clip = out[0], out[1]
    # Bypass CLIP (random-weight encoder is numerically unstable); feed a
    # deterministic finite conditioning tensor of the right shape.
    torch.manual_seed(0)
    cond = torch.randn((1, 77, 768)) * 0.1
    pos = [[cond, {}]]
    neg = [[torch.zeros_like(cond), {}]]
    latent = {"samples": torch.zeros((1, 4, 16, 16))}

    def sample(m):
        (lat,) = comfy_nodes.common_ksampler(
            m, 7, 4, 5.0, "euler", "normal", pos, neg, latent, denoise=1.0)
        return lat["samples"]

    base = sample(model)
    assert torch.isfinite(base).all(), "baseline latent contains NaN/Inf"
    print(f"baseline latent mean={base.mean():.4f} std={base.std():.4f}")

    # identity matrix (W=1) + identity head mask stacked
    (mat,) = InjectionMatrix().run(model, 4, "flat", 1.0)
    (m1,) = ApplyInjectionMatrix().run(model, mat, "attn2_out", "cond", 0)
    (mask,) = HeadMask().run(model, "attn2", 1.0)
    (m2,) = ApplyHeadMask().run(m1, mask, "scale", "both", 0.0, 1.0)
    ident = sample(m2)
    d_id = (base - ident).abs().max().item()
    print(f"identity-stack vs baseline: max|Δ|={d_id:.6g}")
    assert d_id == 0.0, f"identity stack should be an exact no-op (got {d_id})"

    # W=0 across all cross-attn must diverge from baseline
    (mat0,) = InjectionMatrix().run(model, 4, "flat", 0.0)
    (m3,) = ApplyInjectionMatrix().run(model, mat0, "attn2_out", "cond", 0)
    zero = sample(m3)
    d0 = (base - zero).abs().mean().item()
    print(f"W=0 vs baseline:           mean|Δ|={d0:.6g}")
    assert d0 > 1e-6, f"zeroed cross-attn should diverge (got {d0})"

    # head mask zero on mid attn2 must diverge
    (mask2,) = HeadMask().run(model, "attn2", 1.0)
    for bid in mask2["mask"]:
        if bid[0] == "middle":
            mask2["mask"][bid][:] = 0.0
    (m4,) = ApplyHeadMask().run(model, mask2, "zero", "both", 0.0, 1.0)
    hm = sample(m4)
    d_hm = (base - hm).abs().mean().item()
    print(f"mid-heads=0 vs baseline:   mean|Δ|={d_hm:.6g}")
    assert d_hm > 1e-6, f"zeroed mid heads should diverge (got {d_hm})"

    # DAAM stacked on top of W=0 — verify chaining: DAAM's _replace must call
    # through to ApplyInjectionMatrix's output_patch (output should still equal
    # `zero`, since DAAM only records).
    (m5, _, cache) = DAAM().run(m3, clip, "a cat", "mean", "mean", False)
    daam_out = sample(m5)
    d_daam = (zero - daam_out).abs().mean().item()
    print(f"DAAM∘(W=0) vs (W=0):       mean|Δ|={d_daam:.6f}")
    assert d_daam < 1e-5, "DAAM recorder should not perturb the forward pass"
    assert cache.is_filled(), "DAAM cache should be filled after sampling"

    print("\ncompose/identity tests passed.")


if __name__ == "__main__":
    main()
