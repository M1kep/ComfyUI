"""E2E: post workflows to a running ComfyUI server and assert they complete.
Uses tiny latents + few steps so CPU is bearable. Images are noise (random
checkpoint) — this verifies the hook plumbing, not visual quality."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
CKPT = "sd15_dummy.safetensors"
SIZE = 128
STEPS = 4


def post_prompt(graph):
    body = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["prompt_id"]


def wait(prompt_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{BASE}/history/{prompt_id}", timeout=30) as r:
            h = json.load(r)
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise RuntimeError(f"workflow failed: {json.dumps(msgs, indent=2)}")
        time.sleep(1)
    raise TimeoutError(f"workflow {prompt_id} did not finish in {timeout}s")


def base_graph():
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": "a cat on a table"}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["1", 1], "text": "blurry"}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
    }


def test_block_info():
    g = base_graph()
    g["10"] = {"class_type": "AL_ModelBlockInfo", "inputs": {"model": ["1", 0]}}
    out = wait(post_prompt(g))
    res = out["outputs"]["10"]
    assert "text" in res and "in.1" in res["text"][0], res
    print("OK block_info: table has", res["text"][0].count("\n"), "rows")


def test_injection_matrix(target):
    g = base_graph()
    g["10"] = {"class_type": "AL_InjectionMatrix",
               "inputs": {"model": ["1", 0], "t_bins": 8, "preset": "flat", "value": 1.0}}
    g["11"] = {"class_type": "AL_InjectionMatrixEdit",
               "inputs": {"matrix": ["10", 0], "model": ["1", 0], "layers": "cross",
                          "t_start": 0.0, "t_end": 1.0, "op": "set", "value": 0.0}}
    g["12"] = {"class_type": "AL_InjectionMatrixPreview",
               "inputs": {"matrix": ["11", 0], "colormap": "viridis", "annotate": True}}
    g["13"] = {"class_type": "SaveImage",
               "inputs": {"images": ["12", 0], "filename_prefix": "AL_e2e_preview"}}
    g["20"] = {"class_type": "AL_ApplyInjectionMatrix",
               "inputs": {"model": ["1", 0], "matrix": ["11", 0], "target": target,
                          "apply_to": "cond", "cond_index": 0}}
    g["21"] = {"class_type": "KSampler",
               "inputs": {"model": ["20", 0], "seed": 1, "steps": STEPS, "cfg": 5.0,
                          "sampler_name": "euler", "scheduler": "normal",
                          "positive": ["2", 0], "negative": ["3", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}}
    g["23"] = {"class_type": "SaveImage",
               "inputs": {"images": ["22", 0], "filename_prefix": "AL_e2e_matrix"}}
    out = wait(post_prompt(g))
    assert "23" in out["outputs"], out["outputs"].keys()
    print(f"OK injection_matrix[{target}]: saved", out["outputs"]["23"]["images"][0]["filename"])


def test_head_mask():
    g = base_graph()
    g["10"] = {"class_type": "AL_HeadMask",
               "inputs": {"model": ["1", 0], "target": "attn2", "init": 1.0}}
    g["11"] = {"class_type": "AL_HeadMaskEdit",
               "inputs": {"mask": ["10", 0], "model": ["1", 0], "layers": "mid",
                          "heads": "0-3", "value": 0.0, "op": "set"}}
    g["12"] = {"class_type": "AL_ApplyHeadMask",
               "inputs": {"model": ["1", 0], "mask": ["11", 0], "mode": "mean_replace",
                          "apply_to": "both", "t_start": 0.0, "t_end": 1.0}}
    g["21"] = {"class_type": "KSampler",
               "inputs": {"model": ["12", 0], "seed": 1, "steps": STEPS, "cfg": 5.0,
                          "sampler_name": "euler", "scheduler": "normal",
                          "positive": ["2", 0], "negative": ["3", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}}
    g["23"] = {"class_type": "SaveImage",
               "inputs": {"images": ["22", 0], "filename_prefix": "AL_e2e_heads"}}
    out = wait(post_prompt(g))
    assert "23" in out["outputs"]
    print("OK head_mask: saved", out["outputs"]["23"]["images"][0]["filename"])


def test_daam():
    g = base_graph()
    g["10"] = {"class_type": "AL_DAAM",
               "inputs": {"model": ["1", 0], "clip": ["1", 1],
                          "prompt": "a cat on a table",
                          "layer_agg": "mean", "t_agg": "mean", "overlay_decoded": True}}
    g["21"] = {"class_type": "KSampler",
               "inputs": {"model": ["10", 0], "seed": 1, "steps": STEPS, "cfg": 5.0,
                          "sampler_name": "euler", "scheduler": "normal",
                          "positive": ["2", 0], "negative": ["3", 0],
                          "latent_image": ["4", 0], "denoise": 1.0}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["1", 2]}}
    g["30"] = {"class_type": "AL_DAAMRender",
               "inputs": {"maps": ["10", 2], "latent_trigger": ["21", 0], "overlay": ["22", 0]}}
    g["31"] = {"class_type": "SaveImage",
               "inputs": {"images": ["30", 0], "filename_prefix": "AL_e2e_daam"}}
    out = wait(post_prompt(g))
    assert "31" in out["outputs"]
    print("OK daam: saved", out["outputs"]["31"]["images"][0]["filename"])


def test_head_sweep():
    g = base_graph()
    g["4"]["inputs"]["width"] = g["4"]["inputs"]["height"] = 64
    g["10"] = {"class_type": "AL_HeadSweep",
               "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                          "latent": ["4", 0], "vae": ["1", 2],
                          "seed": 1, "steps": 2, "cfg": 4.0,
                          "sampler_name": "euler", "scheduler": "normal",
                          "layer": "mid attn2", "sweep": "ablate_each",
                          "mode": "mean_replace", "cols": 0, "include_baseline": True}}
    g["11"] = {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "AL_e2e_sweep"}}
    out = wait(post_prompt(g), timeout=1200)
    assert "11" in out["outputs"]
    print("OK head_sweep: saved", out["outputs"]["11"]["images"][0]["filename"])


def test_barrier_error():
    """Unfilled cache through a barrier should raise, not silently no-op."""
    g = base_graph()
    g["10"] = {"class_type": "AL_DAAM",
               "inputs": {"model": ["1", 0], "clip": ["1", 1],
                          "prompt": "x", "layer_agg": "mean", "t_agg": "mean",
                          "overlay_decoded": False}}
    g["30"] = {"class_type": "AL_DAAMRender",
               "inputs": {"maps": ["10", 2], "latent_trigger": ["4", 0]}}
    g["31"] = {"class_type": "SaveImage",
               "inputs": {"images": ["30", 0], "filename_prefix": "AL_e2e_err"}}
    try:
        wait(post_prompt(g))
    except RuntimeError as e:
        assert "empty" in str(e).lower(), e
        print("OK barrier_error: empty-cache raised as expected")
        return
    raise AssertionError("expected empty-cache error")


if __name__ == "__main__":
    tests = [test_block_info,
             lambda: test_injection_matrix("attn2_out"),
             lambda: test_injection_matrix("attn2_kv"),
             lambda: test_injection_matrix("cond_embed"),
             test_head_mask,
             test_daam,
             test_barrier_error,
             test_head_sweep]
    for t in tests:
        t()
    print(f"\n{len(tests)} E2E tests passed.")
