"""MLP-related Metal kernels: PolyNorm + dequant probe (test helper)."""

from __future__ import annotations

import mlx.core as mx

from ._common import _DISABLE

# --------------------------------------------------------------------------- #
# PolyNorm
# --------------------------------------------------------------------------- #
#
# y = w0 * (x^3 / sqrt(mean(x^6) + eps))
#   + w1 * (x^2 / sqrt(mean(x^4) + eps))
#   + w2 * (x   / sqrt(mean(x^2) + eps))
#   + b
#
# `mean` is over the last axis (channel dim D). One threadgroup per (B*S) row.


_POLYNORM_SRC = r"""
    // Each threadgroup handles one row of length D.
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    const device T* xrow = x + row * D;
    device T*       yrow = y + row * D;

    // Per-thread partial sums of x^2, x^4, x^6 over a strided slice of the row.
    float s2 = 0.0f, s4 = 0.0f, s6 = 0.0f;
    for (uint i = tid; i < D; i += tgsize) {
        float v  = float(xrow[i]);
        float v2 = v * v;
        float v4 = v2 * v2;
        s2 += v2;
        s4 += v4;
        s6 += v4 * v2;
    }

    // Threadgroup reduction via simd_sum + threadgroup buffer.
    threadgroup float tg_s2[32];
    threadgroup float tg_s4[32];
    threadgroup float tg_s6[32];

    float r2 = simd_sum(s2);
    float r4 = simd_sum(s4);
    float r6 = simd_sum(s6);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) {
        tg_s2[sg_id] = r2;
        tg_s4[sg_id] = r4;
        tg_s6[sg_id] = r6;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across simdgroups (handled by the first simdgroup).
    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v2 = (lane < n_sg) ? tg_s2[lane] : 0.0f;
        float v4 = (lane < n_sg) ? tg_s4[lane] : 0.0f;
        float v6 = (lane < n_sg) ? tg_s6[lane] : 0.0f;
        v2 = simd_sum(v2);
        v4 = simd_sum(v4);
        v6 = simd_sum(v6);
        if (lane == 0) {
            tg_s2[0] = v2;
            tg_s4[0] = v4;
            tg_s6[0] = v6;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_d = 1.0f / float(D);
    float eps = float(eps_in[0]);
    float rs2 = metal::rsqrt(tg_s2[0] * inv_d + eps);
    float rs4 = metal::rsqrt(tg_s4[0] * inv_d + eps);
    float rs6 = metal::rsqrt(tg_s6[0] * inv_d + eps);

    float w0 = float(weight[0]);
    float w1 = float(weight[1]);
    float w2 = float(weight[2]);
    float b  = float(bias[0]);

    for (uint i = tid; i < D; i += tgsize) {
        float v  = float(xrow[i]);
        float v2 = v * v;
        float v3 = v2 * v;
        float out = w0 * (v3 * rs6) + w1 * (v2 * rs4) + w2 * (v * rs2) + b;
        yrow[i] = T(out);
    }
"""


def _make_polynorm_kernel():
    return mx.fast.metal_kernel(
        name="motif_polynorm",
        input_names=["x", "weight", "bias", "eps_in"],
        output_names=["y"],
        source=_POLYNORM_SRC,
    )


_polynorm_kernel = None


def polynorm_reference(x: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    """Pure-MLX reference, mathematically identical to the kernel."""

    def _rms(z):
        return z * mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + eps)

    x2 = x * x
    x3 = x2 * x
    return weight[0] * _rms(x3) + weight[1] * _rms(x2) + weight[2] * _rms(x) + bias


def polynorm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-6) -> mx.array:
    """
    Fused PolyNorm. Falls back to the reference if kernels are disabled.

    Shape contract: `x` is (..., D); `weight` is (3,); `bias` is (1,); reduces
    over the last axis.
    """
    if _DISABLE:
        return polynorm_reference(x, weight, bias, eps)

    global _polynorm_kernel
    if _polynorm_kernel is None:
        _polynorm_kernel = _make_polynorm_kernel()

    *lead, D = x.shape
    rows = 1
    for n in lead:
        rows *= n
    if rows == 0:
        return mx.zeros_like(x)

    x_flat = x.reshape(rows, D)
    # Threadgroup size — multiple of simd width (32). 256 is a good default.
    tg = min(256, max(32, ((D + 31) // 32) * 32))
    # MLX `grid` is total thread count (not threadgroup count): `rows` threadgroups
    # of `tg` threads each ⇒ grid = rows * tg.
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    eps_arr = mx.array([eps], dtype=mx.float32)
    out = _polynorm_kernel(
        inputs=[x_flat, weight.astype(x.dtype), bias.astype(x.dtype), eps_arr],
        template=[("T", x.dtype), ("D", D)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape)


# --------------------------------------------------------------------------- #
# Quantized bit-extract — standalone probe used to validate the dequant logic
# in isolation before composing into `sdpa_dual_v_q4`.
# --------------------------------------------------------------------------- #
#
# MLX 4-bit packing: each `uint32` packs `EL_PER_INT = 32 / bits` values; value
# at position `i` lives in bits `(i % EL_PER_INT) * bits .. + bits - 1` of
# `data[i // EL_PER_INT]`. Group-wise dequant: `val = scale[g] * raw + bias[g]`,
# where `raw` is the unsigned int (no zero-point shift in MLX).
#
# This kernel reproduces `mx.dequantize` exactly for any (D, bits, group_size)
# satisfying `D % 32 == 0`, `D % group_size == 0`, `bits ∈ {4, 8}`. We use it
# in tests to lock down the bit-twiddling layer separately from the attention
# math, so when `sdpa_dual_v_q4` ships, an end-to-end correctness failure can
# be triaged against this probe without rerunning the full kernel.

_DEQUANT_PROBE_SRC = r"""
    constexpr uint EL_PER_INT  = 32u / uint(BITS);
    constexpr uint CH_PER_LANE = uint(D) / 32u;
    constexpr uint MASK        = (1u << uint(BITS)) - 1u;

    uint row  = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;

    const device uint32_t* d_row = data   + row * (uint(D) / EL_PER_INT);
    const device T*        s_row = scales + row * (uint(D) / uint(GROUP_SIZE));
    const device T*        b_row = biases + row * (uint(D) / uint(GROUP_SIZE));
    device T*              y_row = y      + row * uint(D);

    uint base = lane * CH_PER_LANE;
    for (uint j = 0; j < CH_PER_LANE; ++j) {
        uint ch        = base + j;
        uint u32_idx   = ch / EL_PER_INT;
        uint shift     = (ch % EL_PER_INT) * uint(BITS);
        uint group_idx = ch / uint(GROUP_SIZE);
        uint raw       = (d_row[u32_idx] >> shift) & MASK;
        y_row[ch] = T(float(s_row[group_idx]) * float(raw) + float(b_row[group_idx]));
    }
"""


def _make_dequant_probe_kernel():
    return mx.fast.metal_kernel(
        name="motif_dequant_probe",
        input_names=["data", "scales", "biases"],
        output_names=["y"],
        source=_DEQUANT_PROBE_SRC,
    )


_dequant_probe_kernel = None


def _dequant_probe(
    data: mx.array,
    scales: mx.array,
    biases: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Metal-kernel dequant. Public only as a test helper.

    Inputs match `mx.quantize` outputs: `data` is `uint32` with last-axis
    `D / (32/bits)`; `scales`/`biases` share a `D / group_size` last axis.
    Output is fp16/bf16/fp32 (matching `scales.dtype`) with last axis `D`.
    """
    global _dequant_probe_kernel
    if _dequant_probe_kernel is None:
        _dequant_probe_kernel = _make_dequant_probe_kernel()

    *lead, n_uint = data.shape
    el_per_int = 32 // bits
    D = n_uint * el_per_int
    assert D % 32 == 0 and D % group_size == 0
    rows = 1
    for n in lead:
        rows *= n
    if rows == 0:
        return mx.zeros((*lead, D), dtype=scales.dtype)

    data_flat = data.reshape(rows, n_uint)
    scales_flat = scales.reshape(rows, D // group_size)
    biases_flat = biases.reshape(rows, D // group_size)

    tg = 32
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)
    out = _dequant_probe_kernel(
        inputs=[data_flat, scales_flat, biases_flat],
        template=[("T", scales.dtype), ("D", D), ("BITS", bits), ("GROUP_SIZE", group_size)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, D)],
        output_dtypes=[scales.dtype],
    )[0]
    return out.reshape(*lead, D)
