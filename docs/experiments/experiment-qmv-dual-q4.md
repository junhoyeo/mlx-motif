# `qmv_dual_q4` — fused gate+up q4 GEMV via shared-`x` register reuse

**Status:** Removed from the codebase. Was committed in `54dabb5`.

## Removed surface

- Deleted symbols: `_QMV_DUAL_SRC`, `_make_qmv_dual_kernel`, `qmv_dual_q4_reference`, `qmv_dual_q4` from `src/mlx_motif/kernels/mlp.py`.
- Deleted helper dependency: `Q4_QDOT_HEADER` and `src/mlx_motif/kernels/_metal_helpers.py`.
- Deleted exports: `qmv_dual_q4`, `qmv_dual_q4_reference` from `src/mlx_motif/kernels/__init__.py`.
- Deleted test file: `tests/test_kernels_qmv_dual.py`.
- Full removed code: `git show origin/main:src/mlx_motif/kernels/mlp.py` and search for `QMV_DUAL`.

## Hypothesis

At decode time MotifMLP is `down_proj(polynorm(gate_proj(x)) * up_proj(x))`. The `gate_proj` and `up_proj` matmuls are two separate q4 GEMVs, both reading the **same** activation `x` (4096-wide).

Hypothesis: applying the same "load Q once, accumulate into two slabs" trick that `sdpa_dual_v` uses for V should win here too — load `x` into registers once, run two parallel `qdot` chains for the two weight matrices, and emit both outputs from a single threadgroup.

## Snippet

Threadgroup geometry mirrors MLX `qmv_fast_impl`. The key inner loop:

```c
for (int k = 0; k < IN; k += block_size) {
    // ONE x load drives both qdots — the whole reason this kernel exists.
    U sum = load_vector_q4<T, U, values_per_thread>(x_p, x_thread);

    // Two consecutive single-tensor passes share x_thread but minimise
    // simultaneous live register pressure (vs interleaving rows).
    for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* gw = gate_ws + row * in_vec_size_w;
        U gs_v = U(g_s[row * in_vec_size_g]);
        U gb_v = U(g_b[row * in_vec_size_g]);
        result_g[row] += qdot_q4<U, values_per_thread>(gw, x_thread, gs_v, gb_v, sum);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
        const device uint8_t* uw = up_ws + row * in_vec_size_w;
        U us_v = U(u_s[row * in_vec_size_g]);
        U ub_v = U(u_b[row * in_vec_size_g]);
        result_u[row] += qdot_q4<U, values_per_thread>(uw, x_thread, us_v, ub_v, sum);
    }
    // ... advance pointers ...
}
```

The `load_vector_q4` and `qdot_q4` helpers were 4-bit-only specialisations of MLX's stock helpers from `mlx/backend/metal/kernels/quantized.h` (duplicated rather than `#include`d because `mx.fast.metal_kernel` doesn't expose MLX's internal headers to user kernels).

## Bench

`B=1`, `IN=4096`, `OUT=16384`, fp16, `group_size=64`:

| S | sequential `mx.quantized_matmul ×2` | `qmv_dual_q4` (fused) | ratio |
|---|---|---|---|
| 1 | 0.22 ms | 0.24 ms | 1.09× slower |
| 4 | 0.86 ms | 0.80 ms | 0.93× (7% win) |
| 16 | 3.4 ms | 6.1 ms | 1.80× slower |
| 64 | 13.5 ms | 51.3 ms | 3.80× slower |

The S=4 case was the only point with a measurable win. By S=16 the fused kernel was catastrophically worse.

## Why it lost

Two compounding reasons:

1. **`x` was already cached.** `x` is `4096 × 2 = 8 KB`. M1 Max's L1 is 192 KB per core. After the first matmul (`gate_proj`), `x` is hot in L1; the second matmul's "extra" reads of `x` are essentially free. The bandwidth-saving thesis was wrong: there *was* no `x` bandwidth to save.

2. **Doubled per-thread state hurts the compiler.** The fused kernel keeps `result_g[4] + result_u[4] = 8 fp32` live per thread, vs MLX's `4 fp32`. The compiler likely spills, which hurts per-iteration throughput.

3. **MLX switches to a simdgroup-matrix path at S≥16** that this hand-written GEMV doesn't mirror. At S=16+ our kernel stays in the vector path while MLX gets to use the matrix unit; the gap compounds with batch size.

## When it might win

- **Apple chips with smaller L1** where `x` doesn't stay cached between the two matmuls.
- **Shapes where `IN > L1 capacity`** (~96K fp16 elements) so the first matmul evicts `x` before the second starts.
- **Latency-critical regimes** that benefit from fewer dispatches (one kernel launch instead of two), even at a per-call throughput loss.

## What the actual MLP win looked like

The real MLP win on this hardware came from reducing **weight** bandwidth (the dominant cost), not activation bandwidth: see [experiment-mlp-lowbit.md](experiment-mlp-lowbit.md) — q3 gs=32 for MLP weights cuts ~25% of weight bytes. Per-call it's faster; end-to-end at decode it has a quality regression that makes it a memory-saving rather than speed-saving preset.

## References

- Commit `54dabb5` — implementation + bench
- MLX reference: `mlx/backend/metal/kernels/quantized.h` `qmv_fast_impl`, `load_vector`, `qdot`
