import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # ComfyUI/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "custom_nodes" / "ComfyUI-AttentionLab"))

from comfy import cli_args  # noqa: E402
cli_args.args.cpu = True
