# mlx-motif

The canonical [MLX](https://github.com/ml-explore/mlx) port of [Motif Technologies'](https://huggingface.co/Motif-Technologies) language models on Apple Silicon.

Motif's flagship LLMs combine **Grouped Differential Attention** ([2510.06949](https://arxiv.org/abs/2510.06949) / [2410.05258](https://arxiv.org/abs/2410.05258)) with the **PolyNorm** activation ([2411.03884](https://arxiv.org/abs/2411.03884)). Neither primitive ships in mlx-lm. This repo implements both natively in MLX, with a stack of custom Metal kernels that beats `mx.fast.scaled_dot_product_attention` on the differential-attention path.

## Headline numbers

Motif-2-12.7B-Reasoning at 4-bit, M1 Max (32-core GPU, 64 GB), 5-run mean ± stdev:

| Prompt length | mlx-lm-only baseline¹ | with custom kernels | speedup |
|---|---|---|---|
| 5 tokens (short)   | 34.70 ± 0.06 tok/s | **40.64 ± 0.15 tok/s** | **+17%** |
| 800 tokens (long)  | 21.56 ± 0.05 tok/s | **34.71 ± 0.08 tok/s** | **+61%** |
| 3.2k tokens (xlong)| ~14 tok/s²         | **24.02 ± 0.17 tok/s** | **+71%** |

¹ With `MLX_MOTIF_DUAL_V=0` (V-stacked SDPA + scalar gda_post fallback).
² Estimated; full A/B at 3.2k took prohibitively long with the slow path.

Output is **byte-identical** between the kernel and reference paths — verified end-to-end on real Motif weights.

## Status

| Phase | Status |
| --- | --- |
| 1 — Package + structural port + tests | done |
| 2 — Speculative decoding wiring | done (works, doesn't beat baseline at this draft/target ratio) |
| 3 — Mixed-precision quantization | done (`--quant-preset mixed --q-proj-bits 6`) |
| 4a — Fused PolyNorm + GDA-post Metal kernels | done |
| 4b — Custom shared-QK dual-V SDPA + variants | done (the headline win) |
| 4c — 4-slot KV cache (memory savings at xlong) | done (opt-in via env) |
| 5 — OpenAI-compatible server with `<think>` streaming | not started |
| 6 — HF Hub release + blog | not started |

## Install

```bash
git clone …/mlx-motif && cd mlx-motif
uv pip install -e ".[dev]"
```

Requires Python ≥ 3.11, MLX ≥ 0.21, Apple Silicon (only M1 Max validated end-to-end so far).

## Quickstart

```bash
# 1. Convert an HF checkpoint (defaults to bf16; add --quantize for q4)
mlx-motif convert \
  --hf-path Motif-Technologies/Motif-2-12.7B-Reasoning \
  --out ./out/motif-12.7b-q4 \
  --quantize --bits 4

# 2. Generate
mlx-motif generate --model ./out/motif-12.7b-q4 --prompt "Hello, world."
```

Programmatically:

```python
from mlx_lm import generate
from mlx_motif import load

model, tokenizer = load("./out/motif-12.7b-q4")
print(generate(model, tokenizer, prompt="…", max_tokens=128))
```

`mlx_motif.load` wires our `Model` into mlx-lm's loader (default: with QKV projection fusion). Everything downstream — `mlx_lm.generate`, `mlx_lm.server`, speculative decoding — works the same as for stock mlx-lm models.

## Architecture

Motif-2-12.7B uses **Grouped Differential Attention** — each token's attention is the difference between two softmax distributions weighted by a learnable `λ`, with origin/noise heads grouped (`gr=4` origin heads share one noise head per group). The 2.6B variant uses the simpler ungrouped form. Both share the **PolyNorm** MLP activation (a 3-term polynomial RMS-norm).

The MLX forward path of one decoder layer at decode (S=1):

```
x  ─→  RMSNorm  ─→  qkv_proj  ─→  split (q | k | v)
                       │
                       ├─→  q (origin-first layout) ─→  rope
                       │                              │
                       │                              └─→  q1 (32h, slice)
                       │                                   q2 (8h,  slice)
                       │
                       ├─→  k ─→  rope ─→  cache.update_and_fetch_4
                       │                              │
                       │                              └─→  k1, k2
                       │
                       └─→  v ─→  cache.update_and_fetch_4
                                                      │
                                                      └─→  v1, v2

   ┌─→  sdpa_dual_v(q1, k1, v1, v2, scale)  ─→  attn_origin (32h × 2D)  ┐
   │       (custom Metal kernel, native GQA broadcast)                  │
   │                                                                    │
   └─→  sdpa_dual_v(q2, k2, v1, v2, scale)  ─→  attn_noise  (8h  × 2D)  ┘
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┘
   │
   └─→  gda_post_split(attn_origin, attn_noise, λ, λ_init, subln_w)  ─→  out
            (custom Metal kernel: differential subtract + SubLN + scale, no concat)
                       │
                       └─→  o_proj  ─→  + residual  ─→  RMSNorm  ─→  MLP  ─→  + residual
```

## The custom Metal kernels

All kernels live in [`src/mlx_motif/kernels.py`](src/mlx_motif/kernels.py). Each ships with a pure-MLX reference (`*_reference`) and a parametric correctness test under [`tests/`](tests/).

### `sdpa_dual_v` — the headline kernel

Shared-QK, dual-V SDPA decode. Computes `cat([SDPA(q,k,v1), SDPA(q,k,v2)], axis=-1)` in **one** Metal pass — one softmax computation, two V slabs accumulated in the same KV loop, native GQA broadcast.

**Why it exists.** MLX's `mx.fast.scaled_dot_product_attention` requires `query_head_dim == value_head_dim` and only ships templates for `D=V ∈ {64, 96, 128, 256}` ([`mlx/backend/metal/scaled_dot_product_attention.cpp:618-621`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp)). For grouped DiffAttn we need V of width `2·D=256` paired with `D=128` K. The combination falls back to a generic path. Splitting into two `V=128` calls works but recomputes Q·Kᵀ each time. This kernel does QK-and-softmax once and accumulates into both V slabs.

**Threadgroup geometry** (matches MLX `sdpa_vector.h` exactly):
- `BN=32` simdgroups × `BD=32` lanes = 1024 threads
- `simd_gid` strides KV positions, `simd_lid` strides head_dim
- Per-thread persistent state: `q[4] + o1[4] + o2[4]` = 12 fp32 (vs MLX's 8) — fits comfortably under M1 Max's register-pressure-induced 896-thread cap

**Numerical lessons** (the hard way — see commits `6922acb`, `acc2de7`):
- Use `-1e30f` as the initial-max sentinel, NOT `-INFINITY`. `metal::fast::exp(-INF − finite)` returns NaN on Apple GPUs.
- Channel coverage is `BN × v_per_thread × num_slabs = output_channels`. For our 2D=256 output: 32 × 4 × 2 = 256 ✓. Halving BN breaks coverage.

### `gda_post_split` — fused post-attention reduction

Reads `attn_origin` and `attn_noise` as **separate** buffers, broadcasts noise via index inside the threadgroup (`h_n = h_o / gr`), computes `(attn_o − λ·attn_n)`, applies SubLN, scales by `(1 − λ_init)`. Replaces the previous chain that required `mx.concatenate([attn_o, attn_n], axis=1)` (~316 µs/layer on its own) followed by 5+ separate ops.

### `polynorm` — fused MLP activation

Single-pass `polynorm(x) = w0·norm(x³) + w1·norm(x²) + w2·norm(x) + b`. Each threadgroup processes one row, computes the three power-means via simdgroup reductions, then writes the linear combination.

### Variants tried and kept in tree, NOT used by default

These are real, validated kernels that didn't win on M1 Max for our specific shape but are kept as references and as the right answer for other configurations:

- `gda_decode` (commit `6922acb`) — single-pass flash-style fused GDA with serial-per-thread KV. Correct but ~2.7× slower than MLX's vectorized SDPA. Opt-in via `MLX_MOTIF_FLASH_DECODE=1`.
- `sdpa_dual_v_2pass` (commit `ab50df7`) — MLX-style two-pass design (per-block partials → cross-block reduction). Loses to single-pass at every tested KV (64..16k) because with `H_q=40` query heads, single-pass already saturates M1's 32 cores. Right design for MLA-style 1-Q-head models or prefill (S>1) where the Q-axis becomes parallel.
- `polynorm_mul` (commit `c4f2c03`) — fuses `polynorm(gate) * up` into one kernel. ~7% slower end-to-end than the bare two-step path: MLX's lazy graph already fuses the elementwise multiply into `down_proj`'s input handling more effectively than a single big custom kernel.
- `MotifGroupedQuantizedKVCache` (commit `295ecf0`) — 4-bit per-slot quantized cache. Saves 15% peak memory at 3.2k context but costs 12% decode speed (per-step dequant > bandwidth saved). Real speed win requires the in-kernel quantized read, tracked as future work.

## Model surgeries

Two transforms applied at load time (in `Model.fuse_qkv()`, called by `mlx_motif.load(..., fuse_qkv=True)`):

### QKV projection fusion (+10% on the matmul)

Concatenates `q_proj`, `k_proj`, `v_proj` weights along the output axis into one `qkv_proj` of shape `(in=4096, out=q+k+v=9216)`. Hits MLX's quantized matmul sweet spot — single fused matmul is faster than three smaller ones.

| Path | Per-layer matmul cost (M1 Max, q4) |
|---|---|
| 3 separate q4 linears (q + k + v) | 320 µs |
| 1 fused q4 linear + 3-way `mx.split` | **287 µs** (-10%) |

### Origin-first Q row permutation

The HF stripe layout `[g0_o1, g0_o2, ..., g0_o4, g0_n, g1_o1, ...]` forces the per-step `q1`/`q2` split into a stride-pattern reshape that materializes a copy. `fuse_qkv()` permutes Q rows to `[origin..origin | noise..noise]` so the split becomes a contiguous-axis slice (zero-copy view). Architecturally cleaner; no measurable end-to-end delta because MLX's lazy graph absorbs the strided-reshape copy in the chain.

## Benchmarks

All numbers from M1 Max, 64 GB, 12.7B-Reasoning q4, 5 measurement runs after 3 warmup runs, same Python session for A/B fairness:

```
                      MLX-only baseline        with custom kernels       speedup
short prompt (5 tok)   34.70 ± 0.06 tok/s     40.64 ± 0.15 tok/s         +17%
med   prompt (164)     ~30 tok/s              ~37 tok/s                  ~+23%
long  prompt (800)     21.56 ± 0.05 tok/s     34.71 ± 0.08 tok/s         +61%
xlong prompt (3204)    14 tok/s (estimate)    24.02 ± 0.17 tok/s         +71%
```

**The win compounds with context length.** Attention is ~10% of decode time at 64 KV tokens but grows linearly while MLP stays fixed. Our custom path saves attention time specifically.

### Per-op profile after all optimizations (M1 Max, isolated microbench)

| Op | Cost | Share |
|---|---|---|
| qkv_proj 4096→9216 (q4, fused) | 287 µs | 9% |
| sdpa_dual_v (KV=256, 32+8 split) | ~282 µs | 9% |
| o_proj 8192→4096 (q4) | 437 µs | 14% |
| mlp gate 4096→16384 (q4) | 522 µs | 17% |
| mlp up   4096→16384 (q4) | 522 µs | 17% |
| mlp down 16384→4096 (q4) | 584 µs | 18% |
| RMSNorm × 2, polynorm, gda_post_split, residuals | rest | ~16% |

Note: isolated microbench ≠ in-chain time. MLX's lazy graph fuses many of these; chained per-layer time is roughly half the sum.

## Codepath / file map

```
src/mlx_motif/
  __init__.py        # exports load, Model, ModelArgs, __version__
  __main__.py        # CLI: convert | generate
  model.py           # PolyNorm, MotifAttention (vanilla + GDA), MotifMLP,
                     # MotifModel, Model + fuse_qkv() + make_cache()
  kernels.py         # All custom Metal kernels + pure-MLX references:
                     #   polynorm, polynorm_mul, gda_post, gda_post_split,
                     #   sdpa_dual_v, sdpa_dual_v_2pass, gda_decode (legacy)
  cache.py           # MotifGroupedKVCache + MotifGroupedQuantizedKVCache
                     #   (4-slot variants for the differential pattern)
  loader.py          # mlx_motif.load() — wraps mlx-lm load_model + fuse_qkv
  convert.py         # HF → MLX safetensors converter
  quant.py           # mixed-precision quantization presets

tests/
  test_model.py                    # smoke: forward shapes, sanitize
  test_quant.py                    # quantization predicates
  test_parity.py                   # numerical parity vs HF reference
  test_kernels.py                  # polynorm + polynorm_mul correctness
  test_kernels_gda.py              # gda_post correctness
  test_kernels_gda_post_split.py   # gda_post_split correctness
  test_kernels_gda_decode.py       # legacy flash decode correctness
  test_kernels_sdpa_dual_v.py      # dual-V SDPA correctness (+ GQA)
  test_kernels_sdpa_dual_v_2pass.py # 2-pass dual-V correctness
  test_grouped_cache.py            # 4-slot cache correctness

examples/
  convert.py    # end-to-end conversion script
  generate.py   # end-to-end generation script
```

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MLX_MOTIF_DUAL_V` | `1` | `0` disables `sdpa_dual_v` and uses the V-stacked SDPA + `gda_post` fallback (the kernel A/B baseline) |
| `MLX_MOTIF_FLASH_DECODE` | `0` | `1` enables the legacy serial flash decode kernel (correct, slow) |
| `MLX_MOTIF_4SLOT_CACHE` | `0` | `1` = unquantized 4-slot cache; `q4` / `q8` = quantized 4-slot cache (memory savings at xlong, speed cost) |
| `MLX_MOTIF_DISABLE_KERNELS` | `0` | `1` falls back to pure-MLX references for all kernels (correctness-only mode) |
| `MLX_MOTIF_PARITY` | unset | path to HF checkpoint to enable the parity test |
| `MLX_MOTIF_PARITY_MLX` | unset | path to converted MLX checkpoint for the parity test |

## What we tried that didn't work (negative results)

Documented honestly because they're useful priors for the next person:

| Idea | Result | Commit |
|---|---|---|
| Custom `polynorm_mul` Metal kernel | -7% end-to-end | `c4f2c03` |
| `mx.compile`-wrapped MLP chain | +12% microbench, -4% end-to-end (breaks lazy fusion) | reverted |
| Fuse gate+up matmuls (4096→32768) | -34% (off MLX's quantized matmul sweet spot) | not landed |
| Concat q+k for joint RoPE call | -9% (concat overhead exceeds dispatch saving) | not landed |
| `mx.fast.SDPA` with quantized KV via dequant bridge | even-or-slower at all tested contexts | `ac95834`, `54c5c62` |
| Single-call 40-head dual_v vs split origin/noise | split is +8% (smaller calls pipeline better) | `91c9a79` |
| 2-pass dual-V (à la MLX `sdpa_vector_2pass_*`) | slower at every KV; right design for 1-Q-head models | `ab50df7` |
| 4-slot quantized cache via dequant fetch | 15% memory saved, 12% slower (no in-kernel quant read) | `295ecf0` |

## Limitations / known issues

- **Tested on M1 Max only.** The kernel constants (BN, BD, register footprint) are tuned for Apple7 / M1 Max. M2/M3/M4 may need re-tuning; haven't validated.
- **Decode-only kernels (S=1).** Prefill goes through the unchanged V-stacked SDPA path. The attention kernels assert S=1.
- **One model class verified end-to-end at q4.** bf16 unquantized path passes structural tests but no end-to-end generation bench.
- **Parity verified at bf16, not q4.** Quantization adds noise that hasn't been numerically diffed against HF.
- **Speculative decoding wires up but doesn't beat baseline.** The 2.6B → 12.7B draft/target ratio isn't aggressive enough on M1 Max — accept rate ~70% but draft cost wipes the gain. Needs a sub-1B Motif draft model to actually win.

## Roadmap

In order of expected payoff:

1. **`sdpa_dual_v_q4`** — read the 4-slot quantized cache directly (no dequant). Real speed win at xlong; the 4-slot cache plumbing is already done, just needs the kernel.
2. **OpenAI-compatible server** with `<think>` streaming toggle (visible / hidden / suppressed). `mlx_lm.server` works via our `load()`; needs a thin wrapper.
3. **HF Hub upload** of the converted MLX checkpoints (`mlx-community/Motif-2-12.7B-Reasoning-MLX-q4` etc).
4. **Multi-chip validation** — re-tune and re-bench on M2/M3/M4.
5. **Perplexity eval** to validate q4 (and the mixed-precision preset specifically) preserves model quality.
6. **Long-context bench** at 16k+ to characterize the 4-slot quantized cache's memory savings.
7. **Prefill-optimized kernels** — currently prefill uses the same path as MLX. The 2-pass `sdpa_dual_v_2pass` is the right design here.

## Reference architecture

For ground truth on the math, see:

- `/tmp/refs/sdpa_vector.h` (run `curl -sL https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/backend/metal/kernels/sdpa_vector.h -o sdpa_vector.h`) — MLX's vectorized SDPA decode, the template our kernels follow.
- HF `Motif-Technologies/Motif-2-12.7B-Reasoning/modeling_motif.py` — the PyTorch reference for Grouped Differential Attention; our kernels are validated against this layer-by-layer.
- arXiv [2510.06949](https://arxiv.org/abs/2510.06949) (GDA), [2410.05258](https://arxiv.org/abs/2410.05258) (DiffTransformer), [2411.03884](https://arxiv.org/abs/2411.03884) (PolyNorm).

## Commit trail of optimizations

```
295ecf0  feat(cache): MotifGroupedKVCache + MotifGroupedQuantizedKVCache (4-slot)
ab50df7  feat(kernels): 2-pass dual_v variant — correct, slower than single-pass
3c92927  perf(model): drop the KV-size heuristic — native GQA wins everywhere
9d8f867  perf(model): permute Q rows to origin-first → zero-copy q1/q2 split
e80d25c  perf(kernels): gda_post_split eliminates the (origin, noise) concat (+4%)
1db9c42  perf(kernels): native GQA broadcast in sdpa_dual_v eliminates mx.repeat
91c9a79  perf(model): split sdpa_dual_v into origin+noise calls (+8%)
ac63071  feat(model): fuse Q/K/V projections into one quantized matmul (+10%)
2b9e20b  feat(kernels): sdpa_dual_v — first custom kernel beating MLX SDPA on M1 Max
54c5c62  docs(model): correct the QuantizedKVCache claim from previous commit
ac95834  feat(model): dequant-after-fetch bridge enables QuantizedKVCache
c4f2c03  feat(kernels): polynorm_mul fusion + negative-result note
acc2de7  docs(kernels): why the obvious flash-GDA design hits a wall on M1 Max
6922acb  feat(kernels): flash-style fused GDA decode kernel — correct, slower
b0bc447  docs(model): note that QuantizedKVCache requires a custom cache class
9ed4f0e  fix(quant): per-layer overrides into config so loader rebuilds shapes
4cf0cbb  feat(kernels): fused PolyNorm + post-GDA Metal kernels + mixed quant
fee28a8  feat: initial MLX port of Motif (Phase 1 scaffold)
```

## License

MIT — see `LICENSE`. Motif checkpoints retain their original Motif Technologies license; this port does not redistribute weights.
