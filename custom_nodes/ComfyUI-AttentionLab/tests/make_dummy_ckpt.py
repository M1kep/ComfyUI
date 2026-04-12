"""Generate a random-weight SD1.5-shaped checkpoint for offline E2E testing.
Images will be noise; the point is to exercise the full hook pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from comfy import cli_args  # noqa: E402
cli_args.args.cpu = True

import torch  # noqa: E402
import safetensors.torch  # noqa: E402
import comfy.ldm.models.autoencoder  # noqa: E402
import comfy.sd1_clip  # noqa: E402


SD15_UNET = dict(
    image_size=32, in_channels=4, out_channels=4, model_channels=320,
    num_res_blocks=2, channel_mult=[1, 2, 4, 4], num_head_channels=64,
    transformer_depth=[1, 1, 1, 1, 1, 1, 0, 0],
    transformer_depth_output=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    transformer_depth_middle=1,
    context_dim=768, use_spatial_transformer=True, legacy=False,
    use_temporal_attention=False, dtype=torch.float16,
)


@torch.no_grad()
def init_finite(module):
    """ComfyUI builds modules with ``disable_weight_init`` ops which leave
    parameters as uninitialised memory (often NaN). Give every tensor a small
    deterministic value so a deep fp16 forward stays finite."""
    g = torch.Generator().manual_seed(0)
    for name, p in module.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.fill_(1.0)
        elif name.endswith("bias"):
            p.zero_()
        else:
            p.copy_(torch.empty_like(p).normal_(0, 0.01, generator=g))
    for _, b in module.named_buffers():
        if not torch.isfinite(b).all():
            b.zero_()


def main(out_path: Path):
    from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
    unet = UNetModel(**SD15_UNET)
    init_finite(unet)
    sd = {f"model.diffusion_model.{k}": v.to(torch.float16).contiguous()
          for k, v in unet.state_dict().items()}

    vae = comfy.ldm.models.autoencoder.AutoencoderKL(
        ddconfig={"double_z": True, "z_channels": 4, "resolution": 256,
                  "in_channels": 3, "out_ch": 3, "ch": 128,
                  "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
                  "attn_resolutions": [], "dropout": 0.0},
        embed_dim=4)
    init_finite(vae)
    sd.update({f"first_stage_model.{k}": v.to(torch.float16).contiguous()
               for k, v in vae.state_dict().items()})

    clip = comfy.sd1_clip.SD1ClipModel()
    init_finite(clip)
    for k, v in clip.state_dict().items():
        k = k.replace("clip_l.", "")
        sd[f"cond_stage_model.{k}"] = v.to(torch.float16).contiguous()

    safetensors.torch.save_file(sd, str(out_path))
    print(f"wrote {out_path} ({sum(v.numel() for v in sd.values()) * 2 / 1e6:.0f} MB)")


if __name__ == "__main__":
    out = ROOT / "models" / "checkpoints" / "sd15_dummy.safetensors"
    main(out)
