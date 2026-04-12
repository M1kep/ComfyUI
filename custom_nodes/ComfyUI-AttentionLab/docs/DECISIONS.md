# Design Decisions

Verified against ComfyUI commit `4b1444f`.

## §11.1 — Patch chaining semantics

`set_model_attn{1,2}_replace` **overwrites**: the patch is stored as
`transformer_options["patches_replace"][name][block_key]` (a dict assignment,
`model_patcher.py:59`). A second call on the same `(stage, idx, depth)` key
silently replaces the first.

`set_model_attn{1,2}_patch` and `set_model_attn{1,2}_output_patch` **chain**:
stored as a list and appended (`model_patcher.py:462`); every callback runs
in registration order (`attention.py:889/916/935/956`).

**Consequence:** AttentionLab uses the chained `*_output_patch` / `*_patch`
hooks wherever possible (`ApplyInjectionMatrix`). Where per-head or
pre-`to_out` access is required (`ApplyHeadMask`, `AttnMapExtract`, `DAAM`)
we use `*_replace` and wrap any pre-existing callback for that block via
`core.hooks.install_attn_replace`, which reads the prior entry from
`model_options` before overwriting and passes it to our hook as `prev`.

## §11.2 — Timestep identity inside hooks

`extra_options["sigmas"]` is the *current* sigma (set per call at
`samplers.py:315`). `extra_options["sample_sigmas"]` is the full schedule
(set once at `samplers.py:973`). There is **no** integer step index.
`core.hooks.sigma_to_bin` finds the nearest schedule entry to the current
sigma and maps to a bin.

## §11.3 — Cond/uncond layout

Source of truth: `samplers.py:375` builds `conds = [cond, uncond_]`, so
**index 0 = cond, 1 = uncond**. `transformer_options["cond_or_uncond"]`
(`samplers.py:313`) is the list of those indices in *batch order*, which is
not guaranteed (depends on grouping in `calc_cond_batch`). Batch chunk size
is `B // len(cond_or_uncond)`. `core.hooks.slice_batch` implements this.

## §11.4 — `CONDITIONING` shape

`[[embedding_tensor, {"pooled_output": ..., ...}], ...]` — confirmed at
`comfy/sd.py:305-315`. (Only relevant from v0.2's `ApplySteering`.)

## §11.5 — IP-Adapter K/V seam

`ComfyUI_IPAdapter_plus` is **not** present in this checkout and ComfyUI
core exposes no marker for concatenated image tokens. The `ipadapter_kv`
target is descoped to `(v0.4+)` per the spec's allowance; `attn2_out` /
`attn2_kv` are documented as the substitute.

## Other drift from spec

* **`DAAM` single-node output**: a node cannot emit both the patched `MODEL`
  (input to the sampler) and a heatmap `IMAGE` (depends on the cache filled
  *by* that sampler) without an ordering edge. `DAAM` therefore emits
  `(model, placeholder_image, maps)` and a second `DAAMRender` node takes
  `(maps, latent_trigger[, overlay])` → `IMAGE`. This is exactly the §0.4
  barrier pattern, just folded into the convenience wrapper.
* **Replace-callback signature**: spec hinted `(q, k, v, extra_options)`;
  confirmed (`attention.py:908/948`). q/k/v arrive *post*-`to_q/k/v`, return
  goes through `to_out`.
