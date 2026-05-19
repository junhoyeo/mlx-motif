# `gda_decode` — single-pass serial-per-thread flash GDA decode

**Status:** Removed from the codebase. Was committed in `6922acb`. The env flag `MLX_MOTIF_FLASH_DECODE` is gone with it.

## Removed surface

- Deleted symbols: `_GDA_DECODE_SRC`, `_make_gda_decode_kernel`, `gda_decode_reference`, `gda_decode` from `src/mlx_motif/kernels/gda.py`.
- Deleted exports: `gda_decode`, `gda_decode_reference` from `src/mlx_motif/kernels/__init__.py`.
- Deleted model dispatch: `AttnPath.SERIAL_FLASH` and `MLX_MOTIF_FLASH_DECODE` routing from `src/mlx_motif/model.py`.
- Deleted test file: `tests/test_kernels_gda_decode.py`; deleted `SERIAL_FLASH` resolver cases from `tests/test_model.py`.
- Full removed code: `git show origin/main:src/mlx_motif/kernels/gda.py` and search for `GDA_DECODE`.

## Hypothesis

The full GDA decode pipeline is several ops: `q1·k1` SDPA → `q2·k2` SDPA → V-channel split → λ-subtract → SubLN → scale. The shipping fast path uses a V-stacked `mx.fast.scaled_dot_product_attention` plus a `gda_post` post-reduction kernel.

Hypothesis: a single fused Metal kernel that does the whole pipeline in registers, with one thread per output channel walking all KV positions serially, would eliminate all intermediate writes and beat the multi-kernel chain.

## Snippet

The serial-per-thread inner loop — one thread owns one output channel, walks all KV positions, and maintains its own online softmax state for each of the two branches:

```c
// Each thread loops over KV positions, computing score + weighted-V
// contribution for ITS output channel.
float m_o = -INFINITY, s_o = 0.0f, acc_o = 0.0f;
float m_n = -INFINITY, s_n = 0.0f, acc_n = 0.0f;

for (uint p = 0; p < KV_SEQ; ++p) {
    // Full D-element dot product per thread, per KV position!
    float so = 0.0f, sn = 0.0f;
    for (uint i = 0; i < D; ++i) {
        so += float(q1_b[i]) * float(k1_b[p * D + i]);
        sn += float(q2_b[i]) * float(k2_b[p * D + i]);
    }
    so *= scale; sn *= scale;
    // ... online softmax updates for both branches ...
}
```

Note the `-INFINITY` sentinel — that's a numerical trap on Apple GPUs (see "Numerical lessons" below).

## Bench

M1 Max, 12.7B q4, decode (single-token), median of 10:

| Path | tok/s | vs baseline |
|---|---|---|
| V-stacked `mx.fast.SDPA` + `gda_post` (default) | 34.5 | baseline |
| `gda_decode` flash kernel | ~12.8 | **2.7× slower** |

## Why it lost

We studied MLX `sdpa_vector.h` carefully and the naive port hits a wall.

**MLX's vectorised SDPA decode geometry:**
- `BN × BD = 32 × 32 = 1024 threads`
- per-thread persistent state: `q[4] + o[4] + (max, sum) ≈ 10 fp32` (~40B)
- per-thread transient state: `k[4] ≈ 4 fp32`
- channel coverage: `BN * v_per_thread = 32 * (V/BD=4) = V (=128 for d=128)` ✓

Differential attention forces **two independent online softmaxes** (q1·k1 and q2·k2 have different softmax denominators) AND **2× the output channels** (V_total = 2D = 256). The cleanest port multiplies per-thread persistent state by ~2:

```
q1[4] + q2[4] + o_origin[8] + o_noise[8] + (m_o, s_o, m_n, s_n) ≈ 28 fp32 / 112B per thread
```

M1 Max's `maxTotalThreadsPerThreadgroup` collapses to **896** at this register pressure — `BN=BD=32 → 1024 threads` no longer launches. Reducing to `BN=16` keeps us under the cap but breaks channel coverage: `16 × 8 = 128 ≠ 256`.

Backing off to serial-per-thread (`gda_decode` above) bypasses the register-pressure problem but pays for it ~3× in throughput because each thread now does ~32× more sequential work than the vectorised path would.

## Numerical lessons (the hard way)

Two pitfalls bit us before we got correctness:

- **`-INFINITY` sentinel is a trap on Apple GPUs.** `metal::fast::exp(-INF − finite)` returns `NaN`. The fix (now used in `sdpa_dual_v`, `sdpa_dual_v_q4`): initialise `max_score` to `-1e30f` instead.
- **Channel coverage formula:** `BN × v_per_thread × num_slabs = output_channels`. For our `2D=256` output, single-pass uses `32 × 4 × 2 = 256` ✓. Halving BN breaks coverage. The `gda_decode` design above sidesteps this by going one-thread-per-channel, but pays the throughput cost in return.

These two findings transferred to the surviving kernels (`sdpa_dual_v`, `sdpa_dual_v_q4`), where they still apply.

## When it might win

- **Future hardware with less register-pressure-induced thread cap collapse** — but you'd then probably want to redesign back to MLX-style vectorised, not keep the serial-per-thread structure.
- **Workloads with very few output channels** where the serial-per-thread design's per-channel cost dominates less. (Not Motif.)

## What we shipped instead

The shipping decode path takes a different tradeoff: keep the standard MLX vectorised SDPA (no register-pressure problem) but call it via V-stacking — `mx.fast.scaled_dot_product_attention` with V of width `2·d`, then `gda_post` does the differential subtract + SubLN + scale in a single fused kernel. Custom kernel work that *did* win on this hardware/shape: `sdpa_dual_v` (single-pass shared-QK dual-V), `sdpa_dual_v_q4` (quantized-input variant), `gda_post_split` (no concat).

## References

- Commits `6922acb` (initial), `acc2de7` (the `-INFINITY` lesson)
- MLX reference: `mlx/backend/metal/kernels/sdpa_vector.h`
