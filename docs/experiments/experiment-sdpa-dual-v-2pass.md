# `sdpa_dual_v_2pass` — two-pass dual-V SDPA decode

**Status:** Removed from the codebase. Was committed in `ab50df7`.

## Hypothesis

The shipping `sdpa_dual_v` single-pass kernel uses `BN=32` simdgroups, each walking `KV/32` positions serially. At `KV=16384` that's 512 positions per simdgroup. As KV grows, each simdgroup's serial walk gets longer while the number of simdgroups stays fixed at 32 — the GPU's threadgroup scheduler runs out of work to interleave.

MLX's stock vectorized SDPA solves this exact problem with `sdpa_vector_2pass_1` / `_2pass_2`: split the KV axis into `BLOCKS` per-block partials (computed massively in parallel), then a cross-block reduce kernel merges them. The hypothesis: porting this design to dual-V should help at long KV.

## Snippet

Two kernels — pass 1 emits per-block `(max, sum, partial_o1[D], partial_o2[D])` to global memory, pass 2 does the cross-block reduce:

```c
// Pass 1: one simdgroup per (B, kv_head, block) — strided KV slice.
for (uint p = block_idx; p < KV_SEQ; p += uint(BLOCKS)) {
    // ... QK score ...
    U exp_score = metal::fast::exp(score - new_max);
    // ... online softmax update ...
    for (int j = 0; j < v_per_thread; ++j) {
        o1[j] = o1[j] * fac + exp_score * U(v1_p[j]);
        o2[j] = o2[j] * fac + exp_score * U(v2_p[j]);
    }
}
// Per-block partials written to global.

// Pass 2: BN=32 simdgroups, lane l reads block l's stats.
U lane_max = U(maxs_p[sg_lid]);
U gmax = simd_max(lane_max);
// ... cross-simdgroup transpose-merge for both V slabs ...
```

The 2-pass call site auto-routed by KV length (long KV → 2-pass, short KV → single-pass), via an `sdpa_dual_v_auto` wrapper that no longer exists.

## Bench

Single-call (B=1, H_q=40, H_kv=8, D=128):

| KV | single-pass | 2-pass | ratio |
|---|---|---|---|
| 1k | 0.10 ms | 0.18 ms | 1.8× slower |
| 4k | 0.32 ms | 0.41 ms | 1.3× slower |
| 8k | 0.61 ms | 0.69 ms | 1.13× slower |
| 16k | 1.19 ms | 1.22 ms | 1.025× slower |

2-pass never won at any tested KV. At 16k they were within noise, but the 2-pass overhead (the global write + the extra dispatch) ate any scheduler-saturation benefit.

## Why it lost

With `H_q=40` query heads and `B=1`, the single-pass kernel already launches `40 × 32 = 1280` simdgroups in parallel. M1 Max has 32 GPU cores × ~8 simdgroups-in-flight = ~256 concurrent simdgroups. We're **already 5× over-subscribed** at the start — the scheduler is saturated regardless of how many we add. The 2-pass trick is the right answer for MLA-style models with 1 effective Q head, not for grouped-multi-head architectures.

The win 2-pass delivers in MLX's case is for *prefill* (where the Q-axis becomes parallelism) or for tiny `H_q` configurations — neither applies to Motif's decode.

## When it might win

- **MLA-style or 1-Q-head models** where single-pass cannot saturate the scheduler.
- **Prefill (`S > 1`)** where the Q-axis adds another parallelism dim and we hit the actual bandwidth bottleneck the 2-pass design solves.
- **Future M-chips with many more cores** where the 32 single-pass simdgroups can no longer saturate; though even then, the per-block fixed cost (`BLOCKS` extra threadgroups + their global writes) needs to be amortized.

## References

- Commit `ab50df7` — implementation + bench
- MLX reference: `mlx/backend/metal/kernels/sdpa_vector.h` `sdpa_vector_2pass_1` and `_2pass_2`
