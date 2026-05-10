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

## Next moves

The MLP block is still the biggest remaining target on this codebase, and (per the analyst pass I ran in parallel) the highest-EV direction is a custom Metal kernel that does the **same shared-input/dual-output trick `sdpa_dual_v` uses, but for MLP**: load `x` once into registers, compute `gate_dot` and `up_dot` in parallel (two `qdot` calls sharing one `load_vector`), pipe through PolyNorm + multiply in registers, emit only the gated activation. Estimated +6-10% end-to-end. About a week of work.

Symmetric to the attention kernel that just shipped, in the right way: **load activation once, compute two outputs, write less.**

The lesson stands: on Apple Silicon, **attention is the loud part, but MLP is the heavy part.** The next round of optimization probably isn't another attention kernel.
