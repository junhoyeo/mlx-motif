# mlx-motif

![Hero image for mlx-motif](./.github/assets/hero.png)

The canonical [MLX](https://github.com/ml-explore/mlx) port of [Motif Technologies'](https://huggingface.co/Motif-Technologies) language models on Apple Silicon.

Motif's flagship LLMs combine **Grouped Differential Attention** ([2510.06949](https://arxiv.org/abs/2510.06949) / [2410.05258](https://arxiv.org/abs/2410.05258)) with the **PolyNorm** activation ([2411.03884](https://arxiv.org/abs/2411.03884)). Neither primitive ships in mlx-lm. This repo implements both natively in MLX, with a stack of custom Metal kernels that beats `mx.fast.scaled_dot_product_attention` on the differential-attention path — plus a full native Swift/macOS port (chat app, servers, CLIs) that runs the same kernels through MLX-Swift.

**Contents:**
[Headline numbers](#headline-numbers) ·
[Status](#status) ·
[Install](#install) ·
[Quickstart](#quickstart) ·
[Server](#openai-compatible-server) ·
[Swift app](#swift--native-macos-app) ·
[Architecture](#architecture) ·
[Metal kernels](#the-custom-metal-kernels) ·
[Prefill fast path](#prefill--vanilla-fast-path) ·
[Speculative decoding](#speculative-decoding) ·
[Development](#development--tests) ·
[Env vars](#environment-variables) ·
[Negative results](#what-we-tried-that-didnt-work-negative-results) ·
[Limitations & roadmap](#limitations--roadmap)

## Headline numbers

Motif-2-12.7B-Reasoning at 4-bit, M1 Max (32-core GPU, 64 GB), measured July 2026 on current `main` (`scripts/bench_decode_e2e.py`, 5 runs after 3 warmups, `max_tokens=64`, median tok/s):

| Prompt length | pure-MLX reference¹ | **4-slot fp16 + kernels²** | q4 packed cache³ | speedup (kernels vs ref) |
|---|---|---|---|---|
| 5 tokens    | 19.0 | **40.9** | 31.1 | **2.2×** |
| 164 tokens  | 17.4 | **40.0** | 31.6 | **2.3×** |
| 800 tokens  | 13.7 | **38.2** | 29.6 | **2.8×** |
| 3204 tokens | 2.8  | **30.7** | 24.8 | **10.9×** |

¹ `MLX_MOTIF_DISABLE_KERNELS=1` — every custom kernel replaced by its pure-MLX reference.
² The **default configuration** (4-slot fp16 cache + `sdpa_dual_v` + `gda_post_split` + PolyNorm); equivalently `MLX_MOTIF_4SLOT_CACHE=1`.
³ `MLX_MOTIF_4SLOT_CACHE=q4` — same kernels reading the packed q4 cache directly; trades throughput at these lengths for cache memory (its per-call attention win needs KV ≥ ~1k and compounds beyond these prompt sizes).

**The win compounds with context length.** Attention is a small share of decode time at short context but grows linearly while MLP stays fixed; the custom path saves attention time specifically. Two provenance notes: the earlier certified sweeps in [`docs/benchmarks/`](docs/benchmarks/README.md) (e.g. 24.0 tok/s at p3204 on the same hardware) predate the July 2026 dispatch/cache-fetch rework — the current numbers above are ~28% faster at long context. And the old table's "mlx-lm-only baseline" (`MLX_MOTIF_DUAL_V=0`, then a stacked-SDPA fallback) is **no longer reproducible**: that fallback was replaced by the per-slab fast path, so today's honest baseline is the all-reference configuration shown here. The 4-slot fp16 cache became the **default** off the back of this sweep (the previous stock-cache default measured ~28 tok/s at p5, leaving ~30% on the table); opt out with `MLX_MOTIF_4SLOT_CACHE=0`.

On the grouped-differential **Motif-2-12.7B-Reasoning q4** checkpoint, greedy output is **byte-identical** between the in-tree kernel and reference (`MLX_MOTIF_DISABLE_KERNELS=1`) decode paths — verified end-to-end on real Motif weights (prompt `"Once upon a time"`, 32 tokens; also checked at 48/96 tokens). This parity is **checkpoint-specific**, not universal: on the smaller **2.6B** vanilla-attention checkpoint the two paths diverge within a couple of greedy tokens (default `"in a small village, there was…"` vs kernels-off `"there was a little girl…"`), because the custom kernels and the stock reference path accumulate their float reductions in different orders and the low-order-bit differences can flip the argmax on that architecture. Both comparisons use the default fp16 KV cache; the lossy q4/q8 4-slot cache is expected to differ and is out of scope for this claim. Numerical parity against the HuggingFace PyTorch reference is **not** independently verified; the 12.7B-q4 quality is checked via perplexity (`scripts/perplexity.py`) instead — see [`docs/benchmarks/`](docs/benchmarks/README.md#127b-reasoning-q4-perplexity).

## Status

Shipped and tested on `main`:

- HF→MLX converter with mixed-precision quantization presets (`uniform | mixed | mlp_lowbit`)
- Custom Metal kernels: fused PolyNorm, GDA-post, shared-QK dual-V SDPA (`sdpa_dual_v`), and a quantized-input variant (`sdpa_dual_v_q4`) that reads the packed q4/q8 KV cache directly — all with runtime-length dispatch (no per-token Metal recompiles) and full-capacity contiguous cache fetches (no per-step copy of the live KV region)
- 4-slot KV cache (fp16 / q4 / q8) with single-dispatch batched quantized writes
- Fast-path prefill and vanilla attention via per-slab native-GQA SDPA (no head-repeat materialization)
- Speculative decoding with real batched verification (persistent draft KV cache, one `[1, K+1]` target forward per cycle) — lossless for greedy *and* sampled decoding (rejection sampling)
- OpenAI-compatible server with `<think>` streaming modes and prompt-based tool calling, plus a bounded tool-**execution** loop (`run_tool_loop` + CLI demo with AST-whitelisted builtin tools)
- Native Swift/macOS stack: SwiftUI chat app with cross-turn KV-cache reuse (EOS-reconciled), `MotifKit` runtime layer, `MotifKitMLX` overlay porting the decoder + Metal kernels to MLX-Swift, and native CLIs (`MotifNativeGenerate`, `MotifNativeEvaluate`, `MotifNativeServe` with OpenAI `tools` parity, `MotifDecodeBench`)
- Both shipped checkpoints validated end-to-end at q4: 12.7B-Reasoning (kernels-on == kernels-off byte-identical, perplexity 12.365 recorded) and 2.6B
- Self-hosted Apple-Silicon CI lane (build + GPU kernel tests on a registered M-series runner)

What's *not* done lives in one place: [Limitations & roadmap](#limitations--roadmap).

## Install

```bash
git clone …/mlx-motif && cd mlx-motif
uv pip install -e ".[dev]"   # or: uv sync --extra dev  (uses the committed uv.lock)
```

Requires Python ≥ 3.11, MLX ≥ 0.21, Apple Silicon (only M1 Max validated end-to-end so far).

## Quickstart

Pre-converted 4-bit checkpoints are on the Hub — pass the ID directly and skip conversion:

- [`junhoyeo/Motif-2-12.7B-Reasoning-MLX-q4`](https://huggingface.co/junhoyeo/Motif-2-12.7B-Reasoning-MLX-q4)
- [`junhoyeo/Motif-2.6B-MLX-q4`](https://huggingface.co/junhoyeo/Motif-2.6B-MLX-q4)

```bash
# Generate straight from the Hub (downloads on first use)
mlx-motif generate --model junhoyeo/Motif-2-12.7B-Reasoning-MLX-q4 --prompt "Hello, world."

# Serve (OpenAI-compatible HTTP)
mlx-motif serve --model junhoyeo/Motif-2-12.7B-Reasoning-MLX-q4 --port 8080
```

Or convert an HF checkpoint yourself (e.g. a different quant preset):

```bash
mlx-motif convert \
  --hf-path Motif-Technologies/Motif-2-12.7B-Reasoning \
  --out ./out/motif-12.7b-q4 \
  --quantize --bits 4
mlx-motif generate --model ./out/motif-12.7b-q4 --prompt "Hello, world."
```

Programmatically:

```python
from mlx_lm import generate
from mlx_motif import load

model, tokenizer = load("./out/motif-12.7b-q4")
print(generate(model, tokenizer, prompt="…", max_tokens=128))
```

`mlx_motif.load` wires our `Model` into mlx-lm's loader (default: with QKV projection fusion). Everything downstream — `mlx_lm.generate`, `mlx_lm.server`, speculative decoding — works the same as for stock mlx-lm models. Runnable end-to-end scripts live in [`examples/`](examples/) (`examples/convert.py`, `examples/generate.py`).

### Conversion presets

`mlx-motif convert` defaults to bf16; `--quantize` enables quantization. The `--quant-preset` flag selects the precision policy (see [`quant.py`](src/mlx_motif/quant.py)):

```bash
# Uniform q4 (default preset when --quantize is set)
mlx-motif convert --hf-path … --out ./out/u-q4 --quantize --bits 4 --group-size 64

# Mixed: keep q_proj higher-precision (--q-proj-bits), everything else at --bits
mlx-motif convert --hf-path … --out ./out/mixed --quantize --quant-preset mixed --q-proj-bits 6 --bits 4

# mlp_lowbit: push MLP projections to --mlp-bits/--mlp-group-size, rest at --bits/--group-size
#   (memory ↓25%, speed −31%, PPL +10% — a documented trade-off, opt-in only)
mlx-motif convert --hf-path … --out ./out/lowbit --quantize --quant-preset mlp_lowbit --mlp-bits 3 --mlp-group-size 32
```

## OpenAI-compatible server

`mlx-motif serve` exposes `POST /v1/chat/completions` (SSE streaming when `stream: true`) and `GET /v1/models`, wrapping `mlx_lm.generate.stream_generate`:

```bash
mlx-motif serve \
  --model ./out/motif-12.7b-q4 \
  --host 127.0.0.1 --port 8080 \
  --model-id motif \
  --think-mode visible   # visible | hidden | captured
```

**`--think-mode`** controls Motif's `<think>…</think>` reasoning trace (overridable per-request via an `extra_body={"think_mode": …}` field):

| Mode | Behavior |
|---|---|
| `visible` (default) | reasoning trace streams as-is |
| `hidden` | tokens between `<think>` and `</think>` are dropped from the output |
| `captured` | trace dropped from the visible stream but returned separately (e.g. a `reasoning` field) |

Motif's reasoning chat template pre-opens the think block (`<|assistant|><think>`), so the filter is primed to start *inside* a think block when the prompt ends that way — there's no opening tag in the stream, only the closing `</think>`.

### Tool calling (prompt-based)

The server accepts the OpenAI `tools` / `tool_choice` fields, implemented as **prompt-injected** tool calling — Motif's chat template has no native tool tokens (see [`tool_calls.py`](src/mlx_motif/tool_calls.py)). When `tools` are present:

1. A deterministic system preamble is prepended describing each tool and instructing the model to emit a single `{"tool_call": {"name", "arguments"}}` JSON object.
2. The output is buffered (not streamed delta-by-delta, since the small model tends to loop the object), and the **first** valid balanced-JSON tool call is extracted and returned as an OpenAI `tool_calls` chunk. If no tool call is emitted, the buffered text is flushed as normal content.

`tool_choice: "none"` suppresses tools; `auto`/`required` behave like the default prompt. This is a pragmatic compatibility shim, not model-native function calling — treat it accordingly. For the **server**, tool execution is the client's job, matching OpenAI semantics; the native Swift server (`MotifNativeServe`) accepts the same `tools` field, injects a byte-identical preamble (locked by a cross-language golden fixture), and returns OpenAI-shaped `tool_calls` with `finish_reason: "tool_calls"`.

For local use there is also a bounded tool-**execution** loop: `run_tool_loop(...)` in [`tool_calls.py`](src/mlx_motif/tool_calls.py) generates → parses the tool call → invokes a caller-supplied executor → appends the result as a `tool` turn → regenerates, capped at `max_rounds`, with executor exceptions surfaced as tool error messages rather than crashes. The CLI demo (`mlx-motif tools-demo`) ships two safe builtins — current time and an AST-whitelisted arithmetic evaluator (no `eval`/`exec`, numeric literals and operators only, capped exponents).

## Swift / native macOS app

A native macOS stack lives under [`swift/`](swift/):

- **`MotifChatApp`** — SwiftUI chat app with markdown rendering, conversation history, context-budget trimming, Liquid Glass chrome (macOS 26), and **cross-turn KV-cache reuse**: the backend and its KV cache persist across turns, re-prefilling only the token suffix that changed (reuse is reconciled after EOS-terminated turns so cache offsets always match the recorded tokens).
- **`MotifKit`** — runtime layer with an OpenAI-compatible bridge and `<think>` filtering that mirrors the Python server.
- **`MotifKitMLX`** (gated by `MOTIFKIT_ENABLE_MLX=1`) — the Motif decoder and custom Metal kernels ported to MLX-Swift, including QKV fusion (originals released after fusing), native-GQA kernel dispatch, runtime-length kernels with full-capacity cache fetches, generation-loop cancellation, and GPU cache-limit configuration.
- Native CLIs: `MotifNativeGenerate` (incl. `--speculative`), `MotifNativeEvaluate` (perplexity via fused log-softmax), `MotifNativeServe` (OpenAI-compatible, incl. `tools`), `MotifDecodeBench`.

```bash
# Default lightweight build (talks to `mlx-motif serve` over /v1)
swift test --package-path swift
swift build --package-path swift --target MotifChatApp

# Native MLX overlay (Swift 6.1+ recommended; see swift/README.md for toolchain pins)
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifChatApp
```

The Swift MLX path reads the same `MLX_MOTIF_*` feature flags as the Python path.

## Architecture

Motif-2-12.7B uses **Grouped Differential Attention** — each token's attention is the difference between two softmax distributions weighted by a learnable `λ` (kept in fp32 in both language ports), with origin/noise heads grouped (`gr=4` origin heads share one noise head per group). The 2.6B variant uses the simpler ungrouped form. Both share the **PolyNorm** MLP activation (a 3-term polynomial RMS-norm).

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

   ┌─→  sdpa_dual_v(q1, k1, v1, v2, scale, kv_len)  ─→  attn_origin (32h × 2D)  ┐
   │       (custom Metal kernel, native GQA broadcast, runtime KV length)       │
   │                                                                            │
   └─→  sdpa_dual_v(q2, k2, v1, v2, scale, kv_len)  ─→  attn_noise  (8h  × 2D)  ┘
                                                                                │
   ┌────────────────────────────────────────────────────────────────────────────┘
   │
   └─→  gda_post_split(attn_origin, attn_noise, λ, λ_init, subln_w)  ─→  out
            (custom Metal kernel: differential subtract + SubLN + scale, no concat)
                       │
                       └─→  o_proj  ─→  + residual  ─→  RMSNorm  ─→  MLP  ─→  + residual
```

The cache hands the kernels its **full step-padded capacity buffers** plus the live length (`kv_len = cache.offset`); the kernels bound their KV loop at `kv_len` and stride rows by the buffer capacity. Prefill (S > 1) takes a separate fast path — see [Prefill & vanilla fast path](#prefill--vanilla-fast-path).

## The custom Metal kernels

All production kernels live in the [`src/mlx_motif/kernels/`](src/mlx_motif/kernels/) package, split by domain (`attention.py`, `gda.py`, `mlp.py`, with shared helpers in `_common.py`). Each production kernel ships with a pure-MLX reference (`*_reference`) and a parametric correctness test under [`tests/`](tests/). The Swift ports of the same kernels live in [`swift/Sources/MotifKitMLX/MotifMetalKernels.swift`](swift/Sources/MotifKitMLX/MotifMetalKernels.swift) with their own Metal-vs-reference equivalence suites.

### Dispatch & cache-fetch discipline

Two properties of the decode hot path (landed July 2026) matter as much as the kernel math:

- **Runtime KV length, static templates.** The kernels take the live KV length as a runtime `int32` input, not a Metal template constant. Template parameters are only the genuinely-static ones (dtype, head dim, GQA factor, bits, group size). This matters because `mx.fast.metal_kernel` JIT-compiles one Metal variant per unique template tuple — and decode grows the KV length by 1 every token. With the length baked into the template, **every generated token paid a fresh ~50–100 ms Metal compile**; with runtime length, the same call is ~1.7 ms (kernel microbench, M1 Max, mlx 0.31.2: fresh-KV-length call 53–99 ms before vs 1.7 ms median after, statistically identical to a zero-compile control).
- **Full-capacity contiguous fetches.** `mx.fast.metal_kernel` defaults to `ensure_row_contiguous=True`, and an exact-length slice `[..., :offset, :]` of a step-padded cache buffer is *not* row-contiguous — so every kernel launch silently copied the entire live KV region (V slabs twice, since origin and noise calls are separate). The caches therefore return their full row-contiguous capacity buffers, and consumers bound reads by `cache.offset` (kernels via `kv_len`; non-kernel paths by slicing back).

The Swift port implements the same contract end-to-end: its kernels take the runtime KV length and its caches return full-capacity buffers, with the `kv_len < capacity` path locked by poisoned-padding tests through the public wrappers.

### `sdpa_dual_v` — the headline kernel

Shared-QK, dual-V SDPA decode. Computes `cat([SDPA(q,k,v1), SDPA(q,k,v2)], axis=-1)` in **one** Metal pass — one softmax computation, two V slabs accumulated in the same KV loop, native GQA broadcast (callers pass unrepeated K/V; the kernel maps `kv_head = q_head / GQA_FACTOR`).

**Why it exists.** MLX's `mx.fast.scaled_dot_product_attention` requires `query_head_dim == value_head_dim` and only ships templates for `D=V ∈ {64, 96, 128, 256}` ([`mlx/backend/metal/scaled_dot_product_attention.cpp:618-621`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scaled_dot_product_attention.cpp)). For grouped DiffAttn we need V of width `2·D=256` paired with `D=128` K. The combination falls back to a generic path. Splitting into two `V=128` calls works but recomputes Q·Kᵀ each time. This kernel does QK-and-softmax once and accumulates into both V slabs.

**Threadgroup geometry** (matches MLX `sdpa_vector.h` exactly):
- `BN=32` simdgroups × `BD=32` lanes = 1024 threads
- `simd_gid` strides KV positions, `simd_lid` strides head_dim
- Per-thread persistent state: `q[4] + o1[4] + o2[4]` = 12 fp32 (vs MLX's 8) — fits comfortably under M1 Max's register-pressure-induced 896-thread cap

**Numerical lessons** (the hard way — see commits `6922acb`, `acc2de7`):
- Use `-1e30f` as the initial-max sentinel, NOT `-INFINITY`. `metal::fast::exp(-INF − finite)` returns NaN on Apple GPUs.
- Channel coverage is `BN × v_per_thread × num_slabs = output_channels`. For our 2D=256 output: 32 × 4 × 2 = 256 ✓. Halving BN breaks coverage.

### `sdpa_dual_v_q4` — quantized-input variant

Same algorithm and threadgroup layout as `sdpa_dual_v`, but `K`, `V1`, `V2` are read straight out of HBM as MLX quantized triples `(data: uint32, scales: T, biases: T)` and dequantized in registers. Auto-selected at decode time when the cache is `MotifGroupedQuantizedKVCache` (gate via `MLX_MOTIF_QUANT_SDPA`).

**Why it exists.** The naive path is `mx.dequantize(K)` + `mx.dequantize(V1)` + `mx.dequantize(V2)` + `sdpa_dual_v`. The dequants run every decode step, materialize three full-cache-size fp16 tensors, and never get cached. By inlining the dequant into the attention kernel we skip 3 ops per layer per token AND keep the K/V reads in their packed 4-bit/8-bit form (≈3.6× lower HBM traffic).

**Bench (M1 Max, 12.7B decode shape, B=1, H_q=40, H_kv=8, D=128, group_size=64; min of 10 trials × 64 batched calls):**

| KV    | dequant→sdpa_dual_v | `sdpa_dual_v_q4` (4b) | (8b)   | q4 vs deq | q8 vs deq |
|-------|---------------------|------------------------|--------|-----------|-----------|
| 1024  | 0.118 ms            | 0.086 ms               | 0.080 ms | **0.73×** | **0.68×** |
| 4096  | 0.320 ms            | 0.227 ms               | 0.220 ms | **0.71×** | **0.62×** |
| 8192  | 0.637 ms            | 0.415 ms               | 0.413 ms | **0.65×** | **0.58×** |
| 16384 | 1.223 ms            | 0.794 ms               | 0.769 ms | **0.65×** | **0.57×** |

(At KV ≤ 800 the dispatch overhead from 9 input buffers makes it borderline-or-slower than the dequant path; only the long-context regime wins.)

**Key implementation tricks** (see the `_SDPA_DUAL_V_Q4_SRC` source):
- **MLX qdot trick** for the QK side: precompute `q_pre[j] = scale * q[j] / 2^(shift_base + j*BITS)` once per lane, then in the kv loop the K nibble at position `j` — appearing in the packed word as `nibble * 2^(shift_base + j*BITS)` after a mask-without-shift — multiplied by `q_pre[j]` gives the correctly-weighted partial dot. Saves one shift per channel per kv step (the K side is shift-free in the inner loop). Borrowed from MLX's `qdot` helper in `mlx/backend/metal/kernels/quantized.h`.
- **Per-lane K mask LUT** computed once outside the loop: 4 `uint32` masks → in-place mask + cast-to-float + multiply-and-add, no shifts.
- **Enforced shape contract**: a lane's channels must all live in the single packed word it loads (`qk_per_thread ≤ group_size` and `qk_per_thread ≤ EL_PER_INT`, i.e. `D/32 ≤ 32/bits`). This is asserted in the Python wrapper and guarded in the Swift dispatcher (out-of-contract shapes fall back to the reference path) — shapes like D=256/q8 would otherwise shift a `uint32` by ≥ 32 bits (UB in Metal) and return silent garbage. Ships with `D=128, group_size ∈ {32, 64, 128}, bits ∈ {4, 8}`, which satisfy it.
- **Same -1e30f sentinel** as `sdpa_dual_v` — `metal::fast::exp(-INF − finite) = NaN` on Apple GPUs.
- **Bit-extract probe** (`tests/test_dequant_probe.py`) locks down the 4/8-bit unpack against `mx.dequantize` standalone, so a future correctness regression in the full kernel can be triaged in isolation.

The quantized cache this kernel reads saves ~15% peak memory at 3.2k context and originally cost 12% decode speed via dequant-bridge fetch — with the kernel reading it directly it's net-positive on both axes (per-call 27-43% faster than dequant→fp16 at KV ∈ [1024, 16384]; e2e on real Motif: +18% at 3k prompt vs the same cache's dequant path). Its write path batches all four slots into a **single `mx.quantize` dispatch per token** (head-axis concat, bit-identical to per-slot quantization).

### `gda_post_split` — fused post-attention reduction

Reads `attn_origin` and `attn_noise` as **separate** buffers, broadcasts noise via index inside the threadgroup (`h_n = h_o / gr`), computes `(attn_o − λ·attn_n)`, applies SubLN, scales by `(1 − λ_init)`. Replaces the previous chain that required `mx.concatenate([attn_o, attn_n], axis=1)` (~316 µs/layer on its own) followed by 5+ separate ops.

### `polynorm` — fused MLP activation

Single-pass `polynorm(x) = w0·norm(x³) + w1·norm(x²) + w2·norm(x) + b`. Each threadgroup processes one row, computes the three power-means via simdgroup reductions, then writes the linear combination.

### Load-time model surgeries

Two weight transforms applied by `Model.fuse_qkv()` (called by `mlx_motif.load(..., fuse_qkv=True)`; the Swift port applies the same fusion and **releases the original q/k/v projections afterwards** so attention-projection memory isn't doubled):

**QKV projection fusion (+10% on the matmul).** Concatenates `q_proj`, `k_proj`, `v_proj` weights along the output axis into one `qkv_proj` of shape `(in=4096, out=q+k+v=9216)`. Hits MLX's quantized matmul sweet spot — a single fused matmul is faster than three smaller ones:

| Path | Per-layer matmul cost (M1 Max, q4) |
|---|---|
| 3 separate q4 linears (q + k + v) | 320 µs |
| 1 fused q4 linear + 3-way `mx.split` | **287 µs** (-10%) |

On the Swift side, fused-vs-unfused logit equivalence (including the q4 path) is gated by `MotifQKVFusionParityTests`. Speed evidence: the synthetic decode micro-benchmark (`MotifDecodeBench`, q4 gs=64, B=1, S=1) shows ~12-20% lower median ms/step at the 12.7B per-layer shape with fusion ON — a synthetic-weights per-decode-step measurement on one machine, **not** an end-to-end tok/s number. Corroborated by an earlier real-checkpoint sweep (1.05×@p500 / 1.20×@p3000 on a converted 12.7B q4); end-to-end re-validation on the target checkpoint is still recommended before release.

**Origin-first Q row permutation.** The HF stripe layout `[g0_o1, g0_o2, ..., g0_o4, g0_n, g1_o1, ...]` forces the per-step `q1`/`q2` split into a stride-pattern reshape that materializes a copy. `fuse_qkv()` permutes Q rows to `[origin..origin | noise..noise]` so the split becomes a contiguous-axis slice (zero-copy view). Architecturally cleaner; no measurable end-to-end delta because MLX's lazy graph absorbs the strided-reshape copy in the chain.

### Per-op decode profile

Where a decode step's time goes after all of the above (M1 Max, 12.7B-q4, isolated microbench):

| Op | Cost | Share |
|---|---|---|
| qkv_proj 4096→9216 (q4, fused) | 287 µs | 9% |
| sdpa_dual_v (KV=256, 32+8 split) | ~282 µs | 9% |
| o_proj 8192→4096 (q4) | 437 µs | 14% |
| mlp gate 4096→16384 (q4) | 522 µs | 17% |
| mlp up   4096→16384 (q4) | 522 µs | 17% |
| mlp down 16384→4096 (q4) | 584 µs | 18% |
| RMSNorm × 2, polynorm, gda_post_split, residuals | rest | ~16% |

Note: isolated microbench ≠ in-chain time. MLX's lazy graph fuses many of these; chained per-layer time is roughly half the sum. The MLP block dominates (~52%), which is why further attention-side kernel work has diminishing end-to-end returns — and why the MLP fusion attempts below were worth trying (and worth documenting when they lost).

### Variants we tried and removed

Several kernels and design alternatives were built, measured, and ultimately did not ship — single-pass dual-V beat 2-pass at every tested KV, `polynorm_mul` lost to the lazy-graph chain fusion, `qmv_dual_q4` lost because `x` was already in L1, and `gda_decode` flash hit the M1 Max register-pressure cap. See [the negative-results table](#what-we-tried-that-didnt-work-negative-results) — useful priors before re-trying any of them on different hardware.

## Prefill & vanilla fast path

The custom Metal kernels above are decode-only (S=1). Prefill (S > 1) and the ungrouped 2.6B ("vanilla") path used to fall back to a stacked construction — gr-fold head repeats, three head-axis concats, and a `q_dim=128 / v_dim=256` SDPA that lands in MLX's slow generic template. Both now use **per-slab SDPA**: four `mx.fast.scaled_dot_product_attention` calls with `D=V=head_dim` (hitting MLX's fast templates) and native GQA broadcast (no materialized head repeats), combined by `gda_post_split` / channel concat.

Measured (M1 Max, fp16, layer-level microbench, n=20; 12.7B grouped shapes):

| Scenario | Peak memory | Time |
|---|---|---|
| Grouped prefill, kv=2560 / chunk=512 | 927 → 457 MB | +18.7% faster |
| Grouped prefill, kv=8704 / chunk=512 | 6921 → 1233 MB (**5.6×**) | −1.7% (within noise) |
| Grouped prefill, square 2048 | 919 → 445 MB | +15.5% faster |
| Vanilla decode, kv=4096 | — | **+74%** faster |
| Vanilla decode, kv=16384 | — | **+126%** faster |

End-to-end on the real 2.6B q4 checkpoint, greedy output is token-identical to the previous implementation (3.5k-token prompt, paired runs). The vanilla path stores its V cache **slab-ordered** at projection time to make the split zero-copy — a layout note that matters only if you read the raw vanilla cache.

## Speculative decoding

`MotifSpeculativeEngine` (Swift, `MotifNativeGenerate --speculative`) implements real speculative decoding: a persistent draft KV cache proposes K tokens per cycle, the target verifies them in a **single batched `[1, K+1]` forward** against its own persistent cache, the longest matching prefix is accepted, and both caches `trim()` past the first mismatch. Greedy output is exactly the target's greedy continuation (gated by an exact-equivalence test against plain decode).

Sampled decoding (temperature > 0) is supported **losslessly** via standard rejection sampling (Leviathan/Chen): a draft token `x` is accepted with probability `min(1, p(x)/q(x))`; on rejection the token is resampled from the normalized residual `max(0, p − q)`, so the output distribution equals plain target sampling exactly. Self-draft acceptance ≈ 1 (38/38 in the gate test, since `p == q`) and seeded runs are deterministic.

Measured (July 2026, M1 Max, release build, greedy, 128 tokens, real checkpoints):

- **Self-draft gate** (2.6B → 2.6B, debug build): 46 tokens in **10 target forwards** (37/40 accepted) — proves the batched-verification mechanism.
- **Real pairing** (2.6B draft → 12.7B target): 128 tokens in **43 target forwards** (3.0× forward reduction, 85/168 drafts accepted ≈ 51%) — but generation-only wall clock was **9.4 s ≈ 13.6 tok/s vs ~30 tok/s plain** (plain isolated as a 128-token run minus a 1-token run, both ~2× measured). The draft's own sequential decode plus batched-verify overhead costs ~2.2× more than the saved target forwards.

So the forward-count win is real and the wall-clock loss is now measured, not predicted: at this draft/target ratio speculative decoding does not pay on M1 Max. A small (sub-1B) Motif draft model remains the missing piece.

## Development & tests

```bash
uv pip install -e ".[dev]"   # pytest, pytest-cov, ruff, torch
pytest                       # kernel correctness, cache, attention equivalence, tool-calls, think-filter
ruff check .                 # lint
```

Most tests run anywhere (the kernel tests compare against pure-MLX references; the attention-equivalence suite locks the fast paths against verbatim reproductions of the previous implementations). The numerical **parity** test against the HF PyTorch reference is opt-in: set `MLX_MOTIF_PARITY=<hf-checkpoint>` and `MLX_MOTIF_PARITY_MLX=<converted-mlx-checkpoint>` before running `pytest`. `scripts/eval_smoke.py` is the one-command end-to-end harness (generation + perplexity against a converted checkpoint).

Swift tests run via `swift test --package-path swift`. The MLX overlay's GPU suites (Metal-vs-reference kernel equivalence, QKV fusion parity, cache-reuse gates against a real checkpoint) need `MOTIFKIT_ENABLE_MLX=1 MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1` plus the mlx metallib installed into the xctest bundle (`scripts/build_mlx_swift_metallib.sh`).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MLX_MOTIF_DUAL_V` | `1` | `0` disables `sdpa_dual_v`, using the per-slab SDPA fallback (the kernel A/B baseline) |
| `MLX_MOTIF_4SLOT_CACHE` | `1` | fp16 4-slot cache is the default (measured-fastest decode config); `0` = stock single-slot cache; `q4` / `q8` = quantized 4-slot cache (see [the q4 kernel section](#sdpa_dual_v_q4--quantized-input-variant) for measured wins). Unsupported `k_ratio > 1` configs fall back to the stock cache by default and only error when set explicitly. |
| `MLX_MOTIF_QUANT_SDPA` | `1` | `0` disables `sdpa_dual_v_q4`, falling back to dequant-bridge fetch + `sdpa_dual_v` even on a quantized cache |
| `MLX_MOTIF_DISABLE_KERNELS` | `0` | `1` = pure-MLX reference paths for all kernels (correctness-only mode) |
| `MLX_MOTIF_FUSE_QKV` (Swift) | `1` | `0`/`off` disables QKV-projection fusion (see [Load-time model surgeries](#load-time-model-surgeries) for the equivalence gate and speed evidence) |
| `MLX_MOTIF_PARITY` | unset | path to HF checkpoint to enable the parity test |
| `MLX_MOTIF_PARITY_MLX` | unset | path to converted MLX checkpoint for the parity test |

## What we tried that didn't work (negative results)

Every experiment — code snippet, bench numbers, root cause, and "when this might win on different hardware" — lives in [`docs/experiments/`](docs/experiments/). The short list:

| Idea | Status |
|---|---|
| `polynorm_mul` fused kernel | -7% e2e |
| 2-pass dual-V SDPA | slower at every KV |
| `qmv_dual_q4` fused gate+up GEMV | -9% to -77% |
| `gda_decode` single-pass flash | ~2.7× slower (M1 Max register cap) |
| `mlp_lowbit` q3/gs=32 MLP preset | **kept** — memory ↓25%, speed -31%, PPL +10% |
| 4-slot quantized cache via dequant bridge | **superseded** by `sdpa_dual_v_q4` |
| `mx.compile` MLP, fuse gate+up matmuls, concat q+k RoPE, single-call 40-head dual_v, composable q4 chain | combined writeup in `docs/experiments/` |

Negative-result code itself is **not in the codebase** — only the writeups, exact deleted-symbol provenance, and commit pointers. The docs keep the load-bearing snippets inline; full removed implementations are recoverable from git history with the file/symbol references in each experiment page.

## Limitations & roadmap

Everything known-incomplete, in one list — caveats first, then planned work. (Each item appears only here.)

**Caveats to know:**

- **Tested on M1 Max only.** The kernel constants (BN, BD, register footprint) are tuned for Apple7 / M1 Max. M2/M3/M4 may need re-tuning; not validated.
- **Custom Metal kernels are decode-only (S=1).** Prefill uses the per-slab fast-path SDPA (MLX's stock fast templates with native GQA) — a large improvement over the old stacked fallback, but not a bespoke prefill kernel.
- **bf16 has no end-to-end bench, and HF parity is verified at bf16, not q4.** Both shipped checkpoints are validated end-to-end at q4 (the 12.7B additionally kernels-on == kernels-off byte-identical with recorded perplexity), but quantization noise hasn't been numerically diffed against the HF PyTorch reference.
- **q4 KV-cache reuse is not byte-identical to a from-scratch prefill.** The `[1,1]` decode and `[1,n]` prefill kernels round differently in quantized arithmetic, so reused turns can diverge from a hypothetical fresh re-prefill in low-order bits. This is a property of batched-vs-incremental quantized kernels (present in every inference stack with prefix caching), not a bug; the achievable invariant — reuse == fresh *batched* reconstruction, bit-exact — is enforced by tests, including after EOS-terminated turns.
- **2.6B e2e peak memory rose ~130 MB** (3.76 → 3.89 GB at a 3.5k prompt) with the per-slab rework. Attributed and kept as a documented trade-off: reverting would save 119 MB at 3.5k but cost +369 MB at 6k and +755 MB at 8k context (bisection and crossover table in `docs/experiments/`).
- **The vanilla V cache is slab-ordered** (`[a-heads | b-heads]`) — the ordering is stamped into the cache's `meta_state`, so restoring a slab-ordered state into code expecting HF order fails loudly, and an inverse-permutation helper recovers HF order when needed.

**Roadmap, in order of expected payoff:**

1. **Multi-chip validation** — re-tune and re-bench the Metal kernels on M2/M3/M4 (several negative results are worth re-running there too).
2. **Long-context bench** at 16k+ to characterize the 4-slot quantized cache's memory savings end-to-end.
3. **Sub-1B draft model** to turn speculative decoding's forward-count win into a wall-clock win (lossless sampling is already in; the draft economics are the blocker).

## Further reading

For where each module/test/script lives and what it does, see [`docs/codebase-tour.md`](docs/codebase-tour.md); deeper design notes, experiment writeups, certified benchmark captures, CI layout, and the Swift roadmap live under [`docs/`](docs/), and the Swift package has its own [`swift/README.md`](swift/README.md).

Ground truth on the math: MLX's [`sdpa_vector.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h) (the template our kernels follow); HF `Motif-Technologies/Motif-2-12.7B-Reasoning/modeling_motif.py` (the PyTorch reference our kernels are validated against layer-by-layer); arXiv [2510.06949](https://arxiv.org/abs/2510.06949) (GDA), [2410.05258](https://arxiv.org/abs/2410.05258) (DiffTransformer), [2411.03884](https://arxiv.org/abs/2411.03884) (PolyNorm).

## License

MIT — see `LICENSE`. Motif checkpoints retain their original Motif Technologies license; this port does not redistribute weights.
