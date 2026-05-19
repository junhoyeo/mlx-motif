# Quantized KV cache via per-step dequant bridge

**Status:** Superseded by `sdpa_dual_v_q4` (in-kernel dequant). The bridge path stays as the fallback when `MLX_MOTIF_QUANT_SDPA=0`.

## Live surface

- Live cache symbols: `MotifGroupedQuantizedKVCache.update_and_fetch_4` (dequant bridge) and `update_and_fetch_4_quantized` (packed q4/q8 path) in `src/mlx_motif/cache.py`.
- Live model dispatch: `AttnPath.QUANT_SDPA` and `MLX_MOTIF_QUANT_SDPA` routing in `src/mlx_motif/model.py`.
- Test coverage: `tests/test_grouped_cache.py`, `tests/test_kernels_sdpa_dual_v_q4.py`, `tests/test_dequant_probe.py`, and quant-cache path tests in `tests/test_model.py`.
- Full code: `src/mlx_motif/cache.py` and `src/mlx_motif/model.py`; search for `MotifGroupedQuantizedKVCache` and `QUANT_SDPA`.

## Hypothesis

A 4-bit-per-slot KV cache cuts KV memory by ~4× — at 3.2k context that's the difference between fitting comfortably and not fitting at all. The naive way to make this work without writing a new attention kernel: per decode step, `mx.dequantize` the cache back to fp16 and feed that into the existing `sdpa_dual_v` kernel ("dequant bridge").

The hope was that on M1 Max the dequant cost would be small relative to the attention compute, so we'd get the memory win at roughly equal speed.

## Snippet

The cache class still ships at `src/mlx_motif/cache.py` as `MotifGroupedQuantizedKVCache`. The bridge path is the `_maybe_dequant_kv` helper invoked when `MLX_MOTIF_QUANT_SDPA=0`:

```python
def _maybe_dequant_kv(k, v, cache):
    """When the cache is quantized and we're routing through an fp16 kernel,
    explicitly dequantize K and V back to fp16. This is the "bridge" path —
    correct but slow because it materialises full-cache-size fp16 tensors
    every decode step."""
    if isinstance(cache, MotifGroupedQuantizedKVCache):
        k = mx.dequantize(*k, group_size=cache.group_size, bits=cache.bits)
        v = mx.dequantize(*v, group_size=cache.group_size, bits=cache.bits)
    return k, v
```

## Bench

3.2k prompt, Motif 12.7B q4 on M1 Max:

| Cache + path | Memory | tok/s | vs vanilla fp16 KVCache |
|---|---|---|---|
| Stock fp16 KVCache | baseline | baseline | — |
| q4 cache + dequant bridge → `sdpa_dual_v` | **-15%** | -12% | regression |

The memory win was real. The throughput cost was not OK — 12% off the headline number to save memory we mostly weren't bottlenecked on.

## Why it lost (as a default)

The dequant call runs every decode step, materialises three full-cache-size fp16 tensors (K, V1, V2), and produces nothing reusable for the next step (it has to re-dequant the full cache each token because of the append). The cost is `O(KV_total)` per step, while attention compute itself is also `O(KV_total)` — so dequant is comparable to a single full attention pass, doubling the per-step cost in the regime where the cache is small enough that attention isn't the bottleneck.

## What replaced it

`sdpa_dual_v_q4` reads the quantized triples `(data: uint32, scales: T, biases: T)` straight out of HBM and dequantizes in registers as part of the attention loop. The K-side uses the MLX "qdot" mask-without-shift trick; the V-side does in-register bit-extract + per-group scale/bias. No standalone dequant op, no full-cache materialisation, ~3.6× less HBM traffic per attention step.

After the swap, the same q4 cache becomes net-positive: **-15% memory + +18% tok/s** at 3k prompt vs the dequant-bridge path, and **+10.8% tok/s** vs vanilla fp16 KVCache.

## Why the dequant-bridge path is still in the codebase

Two reasons:

1. **It's the correctness baseline** for `sdpa_dual_v_q4`. End-to-end byte-identity testing (`MLX_MOTIF_DISABLE_KERNELS=1`) compares the in-kernel dequant against the bridge path's reference output.
2. **A/B kill switch** — `MLX_MOTIF_QUANT_SDPA=0` lets users fall back to the bridge if a `sdpa_dual_v_q4` regression turns up on unfamiliar hardware.

It's not the default and not advertised as a perf path.

## References

- Commits `ac95834`, `54c5c62` — initial bridge implementation + the corrected docs claim
- Commit `295ecf0` — `MotifGroupedQuantizedKVCache` itself
- Commit `e741102` — `sdpa_dual_v_q4` that supersedes the bridge
- Design doc: [docs/sdpa_dual_v_q4_design.md](../sdpa_dual_v_q4_design.md)
