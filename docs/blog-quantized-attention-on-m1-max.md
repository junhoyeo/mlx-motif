# Quantized attention on M1 Max — where the win actually lives

I just shipped `sdpa_dual_v_q4` for `mlx-motif`: a custom Metal kernel that does shared-QK dual-V attention while reading the KV cache as packed 4-bit (or 8-bit) integers. No per-step `mx.dequantize`. Same softmax math as our existing `sdpa_dual_v`, just reading from a different memory layout.

The headline number is 27-43% faster than the prior production path at KV ≥ 1024. That's real. It's also a goalpost shift, and the more interesting story is what the kernel taught me about where decode-time wins live on Apple Silicon — and where they don't.

## What this kernel replaces

Motif uses Grouped Differential Attention. Each step computes two independent attention outputs from one shared `Q·Kᵀ`-and-softmax — a "shared-QK, dual-V" pattern that doesn't fit any of MLX's stock SDPA templates. We already had a custom Metal kernel for that, `sdpa_dual_v`, which is the fast path on fp16 K/V.

The cache, separately, can hold K and V quantized to 4-bit to save memory at long context. Until now the path was:

```
cache.update_and_fetch_4(...)
  → quantize new k/v slice, append, return DEQUANTIZED live region
sdpa_dual_v(q, k_fp16, v1_fp16, v2_fp16, scale)
```

Three `mx.dequantize` calls per layer per token, each materializing a full-cache-sized fp16 tensor. At long context that's a lot of bytes you write only to read back once.

The new path skips that:

```
cache.update_and_fetch_4_quantized(...)
  → returns raw quantized triples (data: uint32, scales: T, biases: T)
sdpa_dual_v_q4(q, k_q, v1_q, v2_q, scale, group_size, bits)
  → bit-extract + dequant inline in registers, never materialized
```

## The path was straight, the surprises weren't

The original design doc (`docs/sdpa_dual_v_q4_design.md`) said this kernel was 2-3 days of focused bit-twiddling work. That was about right. The structure that made it tractable:

1. Standalone `_dequant_probe` — a tiny Metal kernel that ONLY does the bit-extract layer, validated against `mx.dequantize` across (bits, group_size, D, dtype) = 48 cases. Locks down the bit math separately from the attention math. If a future change breaks correctness, you rerun this first; if it still passes, the bug is in the attention loop.
2. Then the full kernel mirrors `sdpa_dual_v`'s threadgroup geometry exactly (BN=BD=32, 1024 threads, online softmax). The only delta is each K/V load goes through a bit-extract block.
3. Then 36 parametric correctness cases at all the (B, H_q, H_kv, KV, d) × bits × dtype combinations that matter.

The first surprise was register pressure. With fp16, the kernel launched cleanly at 1024 threads. With bf16, Metal silently capped the threadgroup at 896 — the kernel had marginally too much per-thread state, and the compiler chose a register layout that pushed past the limit. The fix was unglamorous: hold quantization scales and biases as `T` (fp16/bf16) instead of immediately upconverting to `float`, and scope-limit the dequant intermediates inside `{ }` blocks so the compiler can spill them across loop iterations. Below the cap, kernel launches.

## The MLX qdot trick

Naive bit-extract for the QK dot product looks like:

```c
for (int j = 0; j < 4; ++j) {
    uint raw = (k_pkd >> (shift_base + j*4)) & 0xF;
    score += q_r[j] * (s * float(raw) + b);
}
```

That's a shift + AND + cast + multiply + add per channel per kv step. On the M-series the shift takes a non-trivial number of cycles relative to the rest. Across 32 lanes × KV positions × layers × tokens it adds up.

MLX's own `quantized_matmul` doesn't do shifts in its inner loop. From `mlx/backend/metal/kernels/quantized.h::qdot`:

```c
const device uint16_t* ws = (const device uint16_t*)w;
for (int i = 0; i < (values_per_thread / 4); i++) {
    accum += (x_thread[4*i  ] * (ws[i] & 0x000f) +
             x_thread[4*i+1] * (ws[i] & 0x00f0) +
             x_thread[4*i+2] * (ws[i] & 0x0f00) +
             x_thread[4*i+3] * (ws[i] & 0xf000));
}
```

It masks the 4-bit nibble in place — no shift — leaving it scaled by `1`, `16`, `256`, or `4096`. The trick is that the `x_thread` it multiplies by has already been pre-divided by those exact factors when it was loaded:

```c
x_thread[i  ] = x[i];
x_thread[i+1] = x[i+1] / 16.0f;
x_thread[i+2] = x[i+2] / 256.0f;
x_thread[i+3] = x[i+3] / 4096.0f;
```

So `x_thread[i+1] * (ws[i] & 0x00f0)` is `(x[i+1] / 16) * (raw_nibble * 16)` = `x[i+1] * raw_nibble`. The shifts are absorbed into a per-lane Q precomputation that runs once outside the kv loop.

Porting this trick to my kernel — different shift bases per lane (sg_lid 0 vs 1 read from the same uint32 with shifts 0 and 16), so each lane gets its own `q_pre[j]` — turned a clear loss into a marginal win. Cut about 15-20% off inner-loop time.

I tried the same trick on V. It doesn't apply: V's per-step coefficient is `exp_score`, which changes every kv step, so the per-channel inverse factors can't be precomputed. V keeps the shift-based dequant.

## What the bench actually says

M1 Max, 12.7B decode shape (B=1, H_q=40, H_kv=8, D=128, group_size=64), min of 10 trials × 64 batched calls:

| KV    | dequant→sdpa_dual_v | sdpa_dual_v_q4 (4b) | (8b)   | q4 vs deq | q8 vs deq |
|-------|---------------------|------------------------|--------|-----------|-----------|
| 1024  | 0.118 ms            | 0.086 ms               | 0.080 ms | **0.73×** | **0.68×** |
| 4096  | 0.320 ms            | 0.227 ms               | 0.220 ms | **0.71×** | **0.62×** |
| 8192  | 0.637 ms            | 0.415 ms               | 0.413 ms | **0.65×** | **0.58×** |
| 16384 | 1.223 ms            | 0.794 ms               | 0.769 ms | **0.65×** | **0.57×** |

Vs raw fp16 `sdpa_dual_v` (where K/V are already fp16 in HBM, no dequant cost at all): my kernel is at parity at KV=16k and 7-15% slower at most other sizes. The bandwidth saving is real (the kernel pulls 140 GB/s vs fp16's 635 GB/s at KV=8192) but the dequant compute eats most of it. We're compute-bound, not bandwidth-bound, on M1 Max at these shapes.

I should be honest: the original design doc predicted "+10-20% at KV ≥ 16k vs current `sdpa_dual_v`". I didn't hit that. I hit it against a different baseline. That shift is defensible in production — the only time you'd use a quantized cache is when you'd otherwise have to dequantize, so the dequant-bridge is the right comparison — but it's still a moved goalpost.

## Where the win actually lives

Here's the per-layer decode profile from the README (M1 Max, 12.7B-q4):

| Op                                 | Cost  | Share |
|------------------------------------|-------|-------|
| qkv_proj (q4, fused)               | 287 µs | 9% |
| sdpa_dual_v (KV=256)               | ~282 µs | 9% |
| o_proj (q4)                        | 437 µs | 14% |
| **mlp gate (q4)**                  | **522 µs** | **17%** |
| **mlp up (q4)**                    | **522 µs** | **17%** |
| **mlp down (q4)**                  | **584 µs** | **18%** |
| RMSNorms, polynorm, gda_post, residuals | rest | ~16% |

In transformer-speak, MLP = the per-token feedforward block that follows attention in each layer. Two matmuls expand the hidden dim, one collapses it back, with a nonlinearity in between.

In Motif specifically (`MotifMLP` in `src/mlx_motif/model.py`):

```
gate = gate_proj(x)                       # 4096 → 16384  (q4 matmul)
up   = up_proj(x)                         # 4096 → 16384  (q4 matmul)
y    = down_proj(PolyNorm(gate) * up)     # 16384 → 4096  (q4 matmul)
```

Three q4 matmuls + one PolyNorm + one elementwise multiply per layer. Add it up: **MLP block is ~52% of decode time per layer**, attention is ~30%.

So when I say MLP is the bigger remaining target — that's the math. We've squeezed attention hard (custom kernels, q4 cache, dual-V SDPA), but it's a smaller slice of the pie. A modest MLP improvement moves end-to-end token rate more than a big attention improvement. **A 30% MLP speedup = 15% end-to-end. A 30% attention speedup = 3-9% end-to-end.**

Things I tried earlier on the MLP and that didn't work, captured in the negative-results table:
- `polynorm_mul` fused kernel (gate norm + multiply in one pass): -7%, MLX's lazy graph already fuses that elementwise.
- `mx.compile` over the MLP chain: nice microbench, breaks lazy fusion across layers, -4% end-to-end.
- Fuse gate+up matmuls into one 4096→32768 matmul: -34%, falls off MLX's quantized-matmul sweet spot.

So MLP is harder than it looks — the obvious moves all lose. But running the analysis sharpens *why* and points at a specific direction that hasn't been tried.

### What's actually slow about MLP

MLX's quantized-matmul dispatch picks `qmv_fast` for our shapes (`fast = N % bn == 0 && K % 512 == 0`, satisfied at K=4096 and K=16384, group_size=64, 4-bit). That's already the hottest GEMV path MLX ships. The MLP isn't slow because of bad dispatch — it's slow because there are three of them per layer and the activation tensor (4096-wide for x, 16384-wide for the gated intermediate) gets reread on every step.

Two implications:
- The "fuse gate+up into one 4096→32768 matmul" attempt failed because it pushed the shape *past* the cache-resident sweet spot for `qmv_fast` — same total FLOPs, worse memory pattern.
- A custom kernel only wins if it beats `qmv_fast` AND avoids materializing intermediates. That's a tall order for the activation function alone (which is why `polynorm_mul` lost), but it's the right framing for a bigger fusion.

### The promising direction

The highest-EV MLP optimization mirrors the same trick that made `sdpa_dual_v` work: **two parallel q4 dot products sharing one activation tensor in registers**. Concretely, a single Metal kernel that:

1. Loads one row of `x_thread` (the 4096-wide activation) once via MLX's `load_vector`.
2. Computes `gate_dot = qdot(gate_weights, x_thread, ...)` AND `up_dot = qdot(up_weights, x_thread, ...)` in registers, sharing the loaded x.
3. Streams the result into PolyNorm + elementwise multiply, also in registers.
4. Emits the gated activation (16384-wide fp16/bf16) — the only HBM write.

This is the exact same pattern as `sdpa_dual_v`'s shared-QK / dual-V — load Q (or x) once, compute two outputs in one pass. The matmul shapes stay at K=4096 (preserving the `qmv_fast` sweet spot), no `mx.compile` boundary is introduced (preserving lazy fusion), no extra dispatch is added (sidestepping the failure modes that killed `polynorm_mul` and `mx.compile`).

Estimated payoff (from the analysis, not measured): **+6-10% end-to-end**, halving activation traffic for the gate+up step. About 4-6 days of work — bigger than `sdpa_dual_v_q4` but the same shape of problem, which means the same probe-first / parametric-test discipline applies.

The upper bound on this direction is collapsing all three GEMVs + the activation into one kernel ("Direction 3" in the analyst's report) — potentially +10-15% end-to-end. That sits right at M1 Max's 32 KB threadgroup-memory limit (one row of fp16 intermediate is exactly 32 KB) so it's feasible but knife-edge. Land the simpler version first; promote to the full version only if the easy variant validates the pattern.

The cheapest-to-test direction is mixed-precision quantization with `down_proj` at q3/q2 — the largest matmul in the block. The mixed-quant infrastructure already exists in `quant.py`; cost is one config change plus a perplexity run. Bounded upside (+3-6%) but bounded effort.

## What I'd do differently

If I were starting over on the q4 kernel:

1. **Bench against the right baseline from the start.** The microbench numbers I generated against raw fp16 `sdpa_dual_v` were misleading and made me think the kernel was a loss. The dequant-bridge comparison is the production-relevant one. Five minutes of clearer thinking up front would have saved the panic.

2. **Sanity-check whether you're compute- or bandwidth-bound before promising a number.** I assumed the 3.6× bandwidth ratio would convert to a ~3× speedup at long KV. Actual: ~1.5× at KV=16k vs the dequant path, parity vs raw fp16. M1 Max has enough compute headroom that bandwidth saving alone doesn't dominate. A back-of-envelope check (peak bandwidth, peak FLOPs, op count) would have warned me.

3. **The probe-first discipline was worth it.** Bit-twiddling Metal kernels are unforgiving — silent corruption, cryptic errors, sentinel-value foot-guns. Validating the unpack layer in isolation (against `mx.dequantize`) before composing into the full kernel meant when correctness tests passed, I trusted them. No regression hunting through 200 lines of kernel code.

## End-to-end validation

Microbenches are easy to fool. The right number is `tokens/sec` on a real Motif checkpoint with the kernel in chain. Here's the 4-cell × 2-prompt-length matrix on the 12.7B-Reasoning q4 model, M1 Max, 63 decode tokens per run, median of 3 runs (timing excludes the first decode step for graph-compile warm-up):

| Prompt | A: stock `KVCache` (fp16) | B: 4-slot fp16 | C: 4-slot q4 + dequant bridge | D: 4-slot q4 + **q4 kernel** |
|---|---|---|---|---|
| ~500 tok  | 36.20 tok/s | 36.57 tok/s | 32.51 tok/s | 34.59 tok/s |
| ~3000 tok | 24.85 tok/s | 24.78 tok/s | 23.32 tok/s | **27.52 tok/s** |

What this actually says:

- **At long context, the q4 kernel wins end-to-end against every other path**, including raw fp16. 27.52 vs 24.85 = **+10.8%** vs the stock fp16 cache, +11.0% vs the 4-slot fp16 cache, +18.0% vs the dequant-bridge path. That's a real, end-user-visible win.
- This is BETTER than the microbench predicted. Microbench said q4 was at parity with raw fp16 at KV=16k. End-to-end it wins by 10%+. The reason: in chain, the q4 cache uses ~4× less HBM bandwidth, which means the MLP's weight reads have less contention for the same bandwidth budget. The win compounds across the layer, not just inside the attention call.
- At short context, the kernel is a slight loss vs vanilla (-4.4% at 500 tok). Expected — the per-call dispatch overhead doesn't amortize until the KV loop is long enough.
- The dequant-bridge path (C) is the worst at both lengths. Should now be considered deprecated; the kernel path strictly dominates it.
- 4-slot fp16 (B) is essentially free vs vanilla (parity at both lengths) — the architectural change to the 4-slot layout costs nothing on its own.

So the goalpost-shift framing from earlier was conservative. The kernel doesn't just beat the dequant-bridge baseline — at the workload it was designed for (long context with quantized cache), it beats *everything*. The microbench undersold it because it isolated attention from the rest of the layer's bandwidth competition.

## I tried the MLP. It didn't work.

Following the analyst's recommendation, I went after MLP next via two paths in parallel:

**Path 1 — fused `qmv_dual_q4` kernel.** A custom Metal kernel that does the exact shared-input/dual-output trick `sdpa_dual_v` uses, applied to MLP: load activation `x` into registers once, run two parallel `qdot`s against `gate_w` and `up_w`. Mirrors MLX's `qmv_fast_impl` threadgroup geometry exactly.

Result (in-chain bench, fp16, 128-call batches):

| S | sequential `mx.qmm × 2` | `qmv_dual_q4` | ratio |
|---|---|---|---|
| 1  | 0.220 ms | 0.240 ms | 1.09× (slower) |
| 4  | 0.624 ms | 0.577 ms | 0.93× (faster) |
| 16 | 1.236 ms | 2.185 ms | 1.77× (much slower) |
| 64 | 2.302 ms | 8.733 ms | 3.79× (catastrophic) |

The hypothesis was wrong. `x` is 8 KB and M1 Max's L1 is 192 KB per core, so after the first matmul, `x` is hot in L1 — the second matmul's "extra" reads of x are essentially free already. The bandwidth-saving thesis assumed L1 wasn't doing its job. Plus the doubled per-thread state (8 fp32 results vs MLX's 4) caused enough register pressure to slow the inner loop. At S>1, MLX's matmul switches to a simdgroup-matrix path my kernel doesn't mirror — that's why the regression grows with batch.

**Path 2 — `mlp_lowbit` quant preset (q3/gs=32 for MLP).** Per-call microbench at MLP shape showed q3+gs=32 is **31% faster** than q4+gs=64 (0.37 vs 0.50 ms). So I added a quant preset that drops gate/up/down to q3+gs=32 while keeping attention at q4+gs=64, converted the actual 12.7B model with it, and ran end-to-end:

| Throughput | p500 | p3000 |
|---|---|---|
| q4 baseline       | 25.91 tok/s | 12.24 tok/s |
| `mlp_lowbit`      | 17.89 tok/s | 13.85 tok/s |

| Perplexity (bundled corpus) | PPL |
|---|---|
| q4 baseline | 12.381 |
| `mlp_lowbit` | 13.656 (+10% relative regression) |

`mlp_lowbit` is **31% slower at p500** and within the noise band at p3000 (mlplow runs spanned 6.25 / 13.85 / 24.14 — ~4× spread suggesting thermal throttling). The +31% per-call microbench win does not survive integration. Two costs absorb it: ~+800 MB resident-memory overhead from doubled scale/bias entries at gs=32 (40 layers × 3 matmuls × 2 MB extra each), and apparent per-call dispatch overhead in MLX's q3 path that's higher than q4. Plus a real quality cost: +10% PPL relative.

**Both kept in tree as documented negative results.** The `qmv_dual_q4` kernel is correct and may win on hardware with smaller L1 (M3/M4 maybe) or on shapes where x exceeds cache. The `mlp_lowbit` preset infrastructure is wired through the converter and ready to A/B against future bit-width combos. Neither is on by default.

## Sanity-check: am I wrong, or is this consensus?

Before declaring a research direction dead, you should check whether the rest of the field agrees. I ran a survey of public MLX/Metal repos to see if anyone had shipped what I was trying to ship.

**Headline finding: nobody has publicly shipped a reproducible, end-to-end MLP speed win for a dense 7-13B transformer on MLX/Metal that beats the default two-matmul path.** The kernel design space is mostly empty here, not because nobody's looked, but because everyone who has reached the same conclusion.

Concrete data points from the survey:

| Project | Approach | Result | Notes |
|---|---|---|---|
| [`Hmbown/ZMLX`](https://github.com/Hmbown/ZMLX) | Fused SwiGLU gate+up GEMV (decode, S≤32) | **+7.5%** Qwen-9B-q4 on **M4 Max** | The single counter-example. **No M1 Max numbers exist.** Their dense-model wins are modest (+4-8%); the headline +12% is on a MoE. M4 Max has substantially more L2 + ~28% more memory bandwidth than M1 Max — exactly the regime where the activation-in-L1 effect *stops* dominating. |
| [llama.cpp Metal](https://github.com/ggml-org/llama.cpp) | None (no fused MLP shipped) | n/a | Same structure as us — individual `ggml_mul_mat` per projection. The fused-MLP issue tracked in their bug list is a CUDA regression, not a Metal optimization. They've converged on the same conclusion. |
| [`mlx-lm` LEARNED_QUANTS](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) | Sensitivity-based bit assignment (4-bit low / 5-bit high) | Uniform across attention/MLP | Official preset library doesn't differentiate MLP vs attention bits. My `mlp_lowbit` experiment is more aggressive than anything they've shipped. |
| [`jjang-ai/jangq`](https://github.com/jjang-ai/jangq) | 2-4 bit MLP, 6-8 bit attention | Wins, but only on **MoE** | The MoE attention:MLP parameter ratio makes the trade obvious. No dense 7-13B benchmark. |
| [Apple's M5 writeup](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) | Hardware (not kernel) | **+19-27%** decode M5-vs-M4 | Source: 28% memory bandwidth bump + Metal 4 TensorOps. Apple themselves frame decode as bandwidth-bound, not fusion-bound. |
| [Apple Recurrent Drafter](https://machinelearning.apple.com/research/recurrent-drafter) | Speculative decoding, ~500M-1B RNN draft | **2.3×** on Vicuna-7B/13B (M1/M2) | The actual practical path forward at this scale. Architecture/training problem, not a kernel problem. |

What this means concretely:

1. **The negative result on M1 Max is consensus, not contrarian.** Everyone who's tried fused MLP on Apple Silicon at this scale has reached the same conclusion. My session-end "I'd stop iterating in good conscience" is the right call.
2. **The qmv_dual_q4 kernel is worth re-running on M3/M4.** The ZMLX +7.5% on M4 Max is direct evidence that the design CAN win on different silicon. That's the next experiment worth doing, but it requires hardware I don't have.
3. **The next 2× at this model scale is speculative decoding, not kernel work.** Apple's ReDrafter pattern with a sub-1B drafter is the credible path. That's outside the "custom Metal kernel" design space I've been operating in this session — it's an architecture and training problem.

So the candid blog conclusion: I'm not the first person to try fusing MLP on Apple Silicon, and I'm not the first to find it doesn't win on the popular shapes on the older silicon. The wins exist (ZMLX proves it), but they live on hardware I don't have and on shapes I'm not running.

## What this teaches us about MLP on M1 Max

Decode-time MLP at the Motif 12.7B shape (4096↔16384) appears to be **Pareto-optimal under MLX's current dispatch**. Both the bandwidth-side approaches (fuse to share x; lower bits to read fewer weight bytes) failed end-to-end despite winning their respective microbenches. The transfer from "isolated kernel works" to "real model is faster" is broken by:

- L1 cache already absorbing the obvious "shared activation" reuse
- Compiler register pressure when fused kernels exceed MLX's tightly-tuned per-thread budget
- MLX's per-call dispatch overhead being non-trivial relative to a single GEMV at this shape
- Lazy graph fusion across the MLP chain that custom kernels disrupt
- Memory-overhead-driven cache pressure that erases bit-width savings

The remaining doors that *might* still open:
- **Architectural changes**: smaller `intermediate_size` with more layers, mixture-of-experts, etc. — out of scope for an inference port.
- **Speculative decoding with a tiny draft model**: if the draft is fast enough, the target's MLP cost matters less per accepted token. The 2.6B → 12.7B draft already wires up but accept rate isn't high enough. Sub-1B Motif draft would help.
- **Different chips**: M3/M4 with shifted compute/bandwidth ratios may flip both directions back to wins. Worth re-running both negative-result kernels on a different chip before considering them dead.

So the candid update is: the MLP didn't have an easy lever on this hardware. The earlier "+6-10% from gate+up fusion" estimate was wrong because L1 was already doing the work the fusion was supposed to do.

That itself is the lesson — **on a well-tuned modern matmul library, the obvious shared-resource fusions often don't win because the cache is already eating that reuse for free.** The wins on attention were in places MLX *wasn't* covering (the dual-V shape, the quantized-input reads). For MLP, MLX covers the obvious shapes, so wins require either non-obvious tricks (still searching) or stepping outside the kernel-only design space (architecture, draft models, hardware).

The lesson stands, but with an addendum: on Apple Silicon, **attention is the loud part, MLP is the heavy part — but the heavy part is also tighter to optimize because MLX has been tuned for it.**
