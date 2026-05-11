# Design: `sdpa_dual_v_q4` — quantized-input shared-QK dual-V SDPA

Status: **shipped**. Kernel + tests + bench + model wire-in landed.
- Kernel: `src/mlx_motif/kernels.py::sdpa_dual_v_q4` (+ `_dequant_probe` standalone)
- Cache hook: `MotifGroupedQuantizedKVCache.update_and_fetch_4_quantized` returns triples
- Wire-in: `MotifAttention._forward_grouped` auto-selects when cache is quantized (env-gated by `MLX_MOTIF_QUANT_SDPA`, default on)
- Tests: `tests/test_kernels_sdpa_dual_v_q4.py` (37 cases), `tests/test_dequant_probe.py` (48), end-to-end via `tests/test_model.py::test_quant_cache_path_runs_and_close_to_fp16`
- Bench (M1 Max, KV=8192): **0.65× the dequant→sdpa_dual_v path** (i.e. 35% faster than the prior production path on quantized caches)
- Realised perf is better than the +10-20% projection below because the right baseline turned out to be `dequantize×3 + sdpa_dual_v`, not raw `sdpa_dual_v`. Vs raw fp16 `sdpa_dual_v` we're at parity at KV ≥ 16k and 5-15% slower at shorter context (compute-bound from the bit-extract).

Two implementation deltas from the original sketch below:
1. **MLX qdot trick** for the QK side (mask-without-shift; `q_pre[j] = scale*q[j] / 2^(shift_base+j*BITS)`, then in-place mask of K). Borrowed from `mlx/backend/metal/kernels/quantized.h::qdot`. Cuts about 15-20% off the inner loop in our measurements.
2. Register pressure on bf16 needed trimming — scope-limited intermediates and held scales/biases in `T` not `U` (otherwise the kernel hits M1 Max's 896-thread cap and Metal refuses to launch at 1024).

Below is the original design doc, kept for context.

---

## Goal

A Metal kernel functionally equivalent to `sdpa_dual_v(q, k, v1, v2, scale)` but reading `k`, `v1`, `v2` as **quantized triples** `(data: uint32, scales: bf16, biases: bf16)` instead of dequantized fp16 arrays. Eliminates the per-step `mx.dequantize` cost of `MotifGroupedQuantizedKVCache.update_and_fetch_4` and gets the actual bandwidth saving from packed 4-bit storage.

## Expected payoff

Composable-q4 bench (commit `f630dac`) showed `mx.quantized_matmul` chains topping out at 1.01× vs fp16 `sdpa_dual_v` at KV=8192. A hand-written kernel that fuses QK + softmax + attn@V into one pass should beat that by another ~10-15% at xlong because:

- single dispatch instead of QK + softmax + 2× attn@V dispatches per branch
- no intermediate `softmax(scores)` materialization (held in registers)
- shared softmax denominator across both V slabs (same as `sdpa_dual_v`)

Net realistic ceiling: **+10-20% at KV ≥ 16k** vs the current fp16 path. At shorter KV the dispatch+bit-extract overhead loses.

## Implementation sketch

### Threadgroup geometry (same as `sdpa_dual_v`)

- `BN = 32` simdgroups × `BD = 32` lanes = 1024 threads
- `simd_gid` strides KV positions, `simd_lid` strides head_dim
- One threadgroup per `(B, H_q)` head

### Per-thread state

```
qk_per_thread = D / BD = 4    # 4 channels of Q, K per lane
v_per_thread  = D / BD = 4    # 4 channels per V slab per lane

thread float q[4], k[4]                 # transient
thread float o1[4], o2[4]               # persistent (V accumulators)
thread float max_score, sum_exp_score   # online softmax state
```

### 4-bit unpack inside the kernel

MLX 4-bit format: each `uint32` packs 8 values, value `i` lives in bits `(i % 8) * 4 .. (i % 8) * 4 + 3` of `data[i // 8]`. Group-wise dequant:

```
val_fp = scale[i / group_size] * float(data[i / 8] >> ((i % 8) * 4) & 0xF) + bias[i / group_size]
```

Each lane (with `qk_per_thread = 4` channels) needs at most 1 uint32 read for K plus 1-2 scale/bias loads:

```c
// lane l reads channels [l*4, l*4+3] of K[block_kv, head_kv]
uint base_idx = l * 4;             // 4 channels per lane
uint uint32_idx = base_idx / 8;    // = l/2
uint shift     = (base_idx % 8) * 4;  // 0 or 16
uint group_idx = base_idx / GROUP_SIZE;   // GROUP_SIZE = 64

uint32_t packed = K_data[block_kv * D/8 + uint32_idx];
T scale = K_scales[block_kv * D/GROUP_SIZE + group_idx];
T bias  = K_biases[block_kv * D/GROUP_SIZE + group_idx];

for (int j = 0; j < 4; ++j) {
    uint nibble = (packed >> (shift + j * 4)) & 0xF;
    k[j] = float(scale) * float(nibble) + float(bias);
}
```

Note: lanes 0 and 1 read the same `packed` uint32 (different shift). Metal L1 cache should coalesce these, so the redundant read is cheap.

### Online softmax loop

Identical to `sdpa_dual_v`'s loop. The only change is K and V come from the dequant block above instead of plain `T` loads.

### Cross-simdgroup reduction + dual-V output write

Identical to `sdpa_dual_v`. Output is bf16/fp16, not quantized.

## Skeleton

The skeleton should live in `src/mlx_motif/kernels.py` next to `sdpa_dual_v`. Public API:

```python
def sdpa_dual_v_q4(
    q: mx.array,                                                    # (B, H_q, 1, D), bf16/fp16
    k_q: tuple[mx.array, mx.array, mx.array],                       # (data, scales, biases)
    v1_q: tuple[mx.array, mx.array, mx.array],
    v2_q: tuple[mx.array, mx.array, mx.array],
    scale: float,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:                                                      # (B, H_q, 1, 2*D)
    ...
```

A pure-MLX reference is one line:

```python
def sdpa_dual_v_q4_reference(q, k_q, v1_q, v2_q, scale, group_size, bits):
    k  = mx.dequantize(*k_q,  group_size=group_size, bits=bits)
    v1 = mx.dequantize(*v1_q, group_size=group_size, bits=bits)
    v2 = mx.dequantize(*v2_q, group_size=group_size, bits=bits)
    return sdpa_dual_v_reference(q, k, v1, v2, scale)
```

## Validation plan

Adopt the parametric-test pattern from `tests/test_kernels_sdpa_dual_v.py`:

```python
@pytest.mark.parametrize("B,H_q,H_kv,KV,d", [...])
@pytest.mark.parametrize("bits", [4, 8])
def test_sdpa_dual_v_q4_matches_reference(B, H_q, H_kv, KV, d, bits):
    # Generate fp16 K/V, quantize them, run kernel, compare to dequant-then-fp16 path.
    # Tolerance: 5e-2 atol/rtol — quantization adds noise that's already
    # absorbed by the reference path (which dequantizes the same data).
    ...
```

## Bit-twiddling pitfalls to avoid

1. **`uint32` byte order**: MLX uses little-endian packing — value at position `i % 8 == 0` is in the LSB nibble.
2. **Group boundary alignment**: `group_size = 64` and `D = 128` ⇒ exactly 2 groups per row. Lanes 0..15 use group 0, lanes 16..31 use group 1. Different lanes in the SAME simdgroup read different scale/bias entries — group_idx must be computed per-lane.
3. **Sign of `nibble`**: 4-bit values are unsigned (0..15). MLX's dequant formula does NOT subtract 8 (no zero-point shift); just `scale * data + bias`. Check `mx.dequantize` source if in doubt.
4. **Sentinel for empty simdgroups**: same as `sdpa_dual_v` — use `-1e30f`, not `-INFINITY`. `metal::fast::exp(-INF − finite) = NaN`.
5. **Channel coverage formula** still applies: `BN × v_per_thread × 2_slabs = 2D` ⇒ 32 × 4 × 2 = 256 ✓.

## Wire-in plan

```python
# In MotifAttention._forward_grouped, when cache is MotifGroupedQuantizedKVCache:
if isinstance(cache, MotifGroupedQuantizedKVCache):
    k1_q, k2_q, v1_q, v2_q = cache.update_and_fetch_4_quantized(...)  # NEW: returns triples not dequant
    attn_origin = sdpa_dual_v_q4(q1, k1_q, v1_q, v2_q, scale)
    attn_noise  = sdpa_dual_v_q4(q2, k2_q, v1_q, v2_q, scale)
```

Add `update_and_fetch_4_quantized` (returns triples) alongside the existing `update_and_fetch_4` (returns dequantized arrays) so the model can opt into the quantized-input kernel without breaking the existing path.

## Bench plan

Same setup as `sdpa_dual_v_2pass` bench (commit `ab50df7`):

```
KV ∈ {64, 256, 1024, 4096, 8192, 16384}
fp16 sdpa_dual_v vs new sdpa_dual_v_q4 (4-bit and 8-bit)
```

Report ratios. Then end-to-end with `MLX_MOTIF_4SLOT_CACHE=q4` enabling the q4-aware path.

## When to declare success

A real win means **all three** of:
1. ≥ +10% at KV=8192 vs `sdpa_dual_v`
2. No regression > 5% at KV ≤ 800
3. Output `mx.allclose(rtol=5e-2, atol=5e-2)` to the dequant-reference

If only (1) holds, ship behind a env flag. If (2) regresses heavily, keep the env flag opt-in. If (3) fails, debug — don't ship slow-but-wrong.

## Time estimate

Realistic: **2-3 days of focused work**.
- Day 1: bit-extract + group-aware dequant in isolation, validated against `mx.dequantize`
- Day 2: full kernel + parametric tests + correctness across (fp32, bf16) × shapes × bits
- Day 3: bench, debug perf, wire into model, end-to-end validation
