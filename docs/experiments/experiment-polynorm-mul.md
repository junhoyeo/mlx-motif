# `polynorm_mul` — fuse `polynorm(gate) * up` into one kernel

**Status:** Removed from the codebase. Was committed in `c4f2c03`.

## Hypothesis

MotifMLP at decode is `down_proj(polynorm(gate_proj(x)) * up_proj(x))`. The two-step `polynorm(gate) * up` writes a full intermediate-sized tensor (`(B·S, 16384)` at the 12.7B layout, ~32 KB/token at bf16, ~1.3 MB/token across 40 layers) to HBM and reads it back into the elementwise `* up` op.

Hypothesis: a single fused kernel that streams `gate` and `up` once, computes PolyNorm in registers, and writes only the final product would save the intermediate round-trip — exactly the kind of memory-bandwidth win that op fusion is supposed to deliver.

## Snippet

The kernel was a straightforward extension of `polynorm` — same two-pass structure (per-row sum of squares → emit), but pass 2 multiplies the PolyNorm result by `up` before storing:

```c
// Pass 2: emit PolyNorm(gate) * up fused.
for (uint i = tid; i < D; i += tgsize) {
    float v  = float(g_row[i]);
    float v2 = v * v;
    float v3 = v2 * v;
    float pn = w0 * (v3 * rs6) + w1 * (v2 * rs4) + w2 * (v * rs2) + b;
    y_row[i] = T(pn * float(u_row[i]));
}
```

Pure-MLX reference (which is what now ships as the production path):

```python
def polynorm_mul_reference(gate, up, weight, bias, eps):
    return polynorm_reference(gate, weight, bias, eps) * up
```

## Bench

Per-kernel microbench at the MLP shape (`B=1, S=1, D=16384`, bf16):

| Path | Cost |
|---|---|
| `polynorm(gate) * up` (two MLX ops) | ~baseline |
| `polynorm_mul` (fused kernel) | +~12% on its own |

End-to-end Motif 12.7B decode:

| Path | tok/s | vs baseline |
|---|---|---|
| Two-step `polynorm(gate) * up` | baseline | — |
| Fused `polynorm_mul` | -7% | regression |

## Why it lost

MLX's lazy graph already fuses the elementwise multiply into the next op (the `down_proj` GEMV input). What looked like an HBM round-trip to a naive reader is actually being elided by the graph compiler — the intermediate never materializes.

When we replace the two-step with one custom kernel, we **gain** nothing (no real bandwidth saved) and **lose** the chained fusion that the lazy graph was doing for free — `down_proj` now has to consume our kernel's actual HBM write, instead of fusing directly into the chain.

The lesson generalises: don't trust an obvious-looking "fuse two ops" candidate on MLX without verifying the lazy graph isn't already fusing it more effectively than you can.

## When it might win

- A backend without lazy-graph fusion (eager-only execution).
- A pipeline where `down_proj` is replaced with something the graph cannot fuse into (e.g., a sparse op, a custom kernel with no chain-fusion path).
- Shapes where the intermediate is large enough to spill out of the lazy-graph fusion window — at MLP `D=16384` we are evidently still inside it, but it's plausible that `D=131072` would behave differently.

## References

- Commit `c4f2c03` — initial implementation + first bench
- Companion blog post on MLP fusion findings: [docs/blog-quantized-attention-on-m1-max.md](../blog-quantized-attention-on-m1-max.md)
