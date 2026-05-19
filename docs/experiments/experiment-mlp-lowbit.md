# `mlp_lowbit` quantization preset — q3/gs=32 for MLP weights

**Status:** **Kept in tree** as opt-in preset (`apply_quant(model, preset="mlp_lowbit")`). Speed-negative but memory-positive — kept for memory-constrained scenarios.

## Live surface

- Live symbols: `_mlp_lowbit_predicate` and `apply_quant(..., preset="mlp_lowbit")` in `src/mlx_motif/quant.py`.
- CLI surface: `mlx-motif convert --quant-preset mlp_lowbit`.
- Test coverage: quantization predicate/config behavior in `tests/test_quant.py`.
- Full code: `src/mlx_motif/quant.py`; search for `mlp_lowbit`.

## Hypothesis

MotifMLP at decode is bandwidth-bound on weight reads. The three MLP projections (`gate_proj`, `up_proj`, `down_proj`) at q4/gs=64 dominate per-layer cost. Dropping them to **q3 with group_size=32** cuts ~25% of MLP weight bytes — at a bandwidth-bound op, that should be a near-linear speedup.

## Snippet

The preset implementation (still live in `src/mlx_motif/quant.py`):

```python
def _mlp_lowbit_predicate(mlp_bits: int, mlp_group_size: int, default_bits: int, default_group_size: int):
    """Return a predicate that quantizes mlp.{gate,up,down}_proj at mlp_bits/mlp_group_size
    and everything else at default_bits/default_group_size."""

    def predicate(path: str, module: nn.Module) -> tuple[int, int] | None:
        if any(name in path for name in ("gate_proj", "up_proj", "down_proj")):
            return (mlp_group_size, mlp_bits)
        return (default_group_size, default_bits)

    return predicate
```

Used as:

```python
apply_quant(model, preset="mlp_lowbit")  # mlp at q3/gs=32, rest at q4/gs=64
```

## Bench

Per-call MLP microbench (M1 Max, B=1, S=1, IN=4096, OUT=16384):

| Preset | gate_proj | up_proj | down_proj (16384→4096) |
|---|---|---|---|
| q4/gs=64 (default) | baseline | baseline | baseline |
| q3/gs=32 (mlp_lowbit) | +31% faster | +31% faster | +28% faster |

End-to-end Motif 12.7B decode (p500 prompt):

| Preset | tok/s | vs baseline | Perplexity (WikiText-2) |
|---|---|---|---|
| q4/gs=64 (default) | baseline | — | baseline |
| q3/gs=32 (mlp_lowbit) | **-31%** | regression | **+10%** |

Memory:

| Preset | MLP weight bytes |
|---|---|
| q4/gs=64 | ~baseline |
| q3/gs=32 | **~25% smaller** |

## Why it lost (on speed)

Per-call the kernel really *is* faster. The end-to-end regression came from the q3/gs=32 layout's extra **scale/bias overhead**: half the group size means twice as many `(scale, bias)` pairs to load per output channel. At q4/gs=64, scale/bias load was a small fraction of the kernel cost; at q3/gs=32, it became significant enough to absorb the bandwidth saving — and then some.

The compounding factor is MLX's matmul dispatch: stock `mx.quantized_matmul` has templated fast paths for `(bits=4, gs=64)` that don't exist for `(bits=3, gs=32)`. The latter falls into a slower generic path. That's the gap that turns a "per-call 31% faster" into "end-to-end 31% slower."

## Why it's kept

**Memory.** ~25% smaller MLP weights is a real win for users who hit a memory ceiling — running 12.7B on a 32 GB Mac, or running an even larger model where weights wouldn't otherwise fit. The 10% PPL regression is acceptable in many of those scenarios; the user can A/B and decide.

The preset wiring lives in `quant.py`:

```python
elif preset == "mlp_lowbit":
    predicate = _mlp_lowbit_predicate(...)
    preset_meta = {"preset": "mlp_lowbit", ...}
```

And it's exposed via `mlx-motif convert --quant-preset mlp_lowbit`.

## When it might win on speed

- **A future MLX with templated fast paths for `(bits=3, gs=32)`** — the per-call microbench shows the bandwidth win is real; only the dispatch path is in the way.
- **An MLP shape where weight bandwidth dominates even more heavily** than at 12.7B — perhaps a wider MLP with a smaller D.
- **Hardware with less arithmetic capacity** where the extra scale/bias decode is cheaper relative to the bandwidth saved.

## References

- Commit `b6a9c77` — implementation + bench
- Companion blog post: [docs/blog-quantized-attention-on-m1-max.md](../blog-quantized-attention-on-m1-max.md)
