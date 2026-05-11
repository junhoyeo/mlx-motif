"""MLP-related Metal kernels: PolyNorm, PolyNorm×up fusion, dequant probe, and
dual-q4 GEMV.
"""

from __future__ import annotations

import mlx.core as mx

from ._common import _DISABLE
from ._metal_helpers import Q4_QDOT_HEADER

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
# Fused PolyNorm × up (MLP gate-and-up fusion)
# --------------------------------------------------------------------------- #
#
# Replaces the two-step `polynorm(gate) * up` in MotifMLP with a single
# kernel that streams `gate` and `up` once, computes PolyNorm in registers,
# and writes only the final product. Saves one intermediate write + read of
# a (B*S, intermediate_size) tensor — at the 12.7B layout that's
# (B*S, 16384) per layer, ~32 KB/token at bf16, 1.3 MB/token across 40 layers.

_POLYNORM_MUL_SRC = r"""
    uint row    = threadgroup_position_in_grid.x;
    uint tid    = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    const device T* g_row = gate + row * D;
    const device T* u_row = up   + row * D;
    device T*       y_row = y    + row * D;

    // Pass 1: per-thread partial sums of x^2, x^4, x^6 over a strided slice.
    float s2 = 0.0f, s4 = 0.0f, s6 = 0.0f;
    for (uint i = tid; i < D; i += tgsize) {
        float v  = float(g_row[i]);
        float v2 = v * v;
        float v4 = v2 * v2;
        s2 += v2;
        s4 += v4;
        s6 += v4 * v2;
    }

    threadgroup float tg_s2[32], tg_s4[32], tg_s6[32];
    float r2 = simd_sum(s2), r4 = simd_sum(s4), r6 = simd_sum(s6);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) {
        tg_s2[sg_id] = r2;
        tg_s4[sg_id] = r4;
        tg_s6[sg_id] = r6;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

    // Pass 2: emit PolyNorm(gate) * up fused.
    for (uint i = tid; i < D; i += tgsize) {
        float v  = float(g_row[i]);
        float v2 = v * v;
        float v3 = v2 * v;
        float pn = w0 * (v3 * rs6) + w1 * (v2 * rs4) + w2 * (v * rs2) + b;
        y_row[i] = T(pn * float(u_row[i]));
    }
"""


def _make_polynorm_mul_kernel():
    return mx.fast.metal_kernel(
        name="motif_polynorm_mul",
        input_names=["gate", "up", "weight", "bias", "eps_in"],
        output_names=["y"],
        source=_POLYNORM_MUL_SRC,
    )


_polynorm_mul_kernel = None


def polynorm_mul_reference(
    gate: mx.array, up: mx.array, weight: mx.array, bias: mx.array, eps: float
) -> mx.array:
    """Pure-MLX equivalent: PolyNorm(gate) * up."""
    return polynorm_reference(gate, weight, bias, eps) * up


def polynorm_mul(
    gate: mx.array, up: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-6
) -> mx.array:
    if _DISABLE:
        return polynorm_mul_reference(gate, up, weight, bias, eps)

    global _polynorm_mul_kernel
    if _polynorm_mul_kernel is None:
        _polynorm_mul_kernel = _make_polynorm_mul_kernel()

    *lead, D = gate.shape
    rows = 1
    for n in lead:
        rows *= n
    if rows == 0:
        return mx.zeros_like(gate)

    g_flat = gate.reshape(rows, D)
    u_flat = up.reshape(rows, D)
    tg = min(256, max(32, ((D + 31) // 32) * 32))
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    eps_arr = mx.array([eps], dtype=mx.float32)
    out = _polynorm_mul_kernel(
        inputs=[g_flat, u_flat, weight.astype(gate.dtype), bias.astype(gate.dtype), eps_arr],
        template=[("T", gate.dtype), ("D", D)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, D)],
        output_dtypes=[gate.dtype],
    )[0]
    return out.reshape(*gate.shape)


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


# --------------------------------------------------------------------------- #
# Dual q4 GEMV (`qmv_dual_q4`) — gate+up MLP fusion
# --------------------------------------------------------------------------- #
# STATUS: NEGATIVE RESULT. Kernel is correct but does NOT beat MLX's
# stock `mx.quantized_matmul × 2` at the decode-time MLP shape on M1 Max.
# Kept in tree as reference + as the right design for cases where the
# bandwidth assumption *does* hold (e.g., other Apple chips with smaller
# L1, or shapes where x doesn't fit in cache).
#
# What this attempted. At decode time MotifMLP is `down_proj(act(gate_proj(x))
# * up_proj(x))`. The gate and up projections are two separate q4 matmuls
# both reading the SAME `x` (4096-wide). The hypothesis: fuse the two so `x`
# is loaded into registers once, not twice — same shape of trick as
# `sdpa_dual_v` (one Q load → two V slabs) applied to MLP.
#
# Threadgroup geometry mirrors MLX `qmv_fast_impl`:
#   2 simdgroups × 32 lanes = 64 threads per threadgroup
#   Each simdgroup computes 4 output rows ⇒ 8 output rows per threadgroup
#   For 4-bit, group_size=64: values_per_thread = 16, block_size = 512
#
# Why it didn't win on M1 Max:
# 1. x is 4096 × 2 bytes = 8 KB. M1 Max L1 is 192 KB per core. After the
#    first matmul, x is hot in L1; the second matmul's "extra" reads of x
#    are essentially free. The bandwidth-saving thesis was wrong: there
#    was no x bandwidth to save.
# 2. The fused kernel doubles per-thread state (result_g[4] + result_u[4]
#    = 8 fp32 vs MLX's 4) and adds 3 extra pointer chains. Compiler likely
#    spills, hurting per-iteration throughput.
# 3. Net measurement (B=1 S=1 IN=4096 OUT=16384, fp16, gs=64): fused 0.24
#    ms vs 2× sequential 0.22 ms — a 10% loss. At S=4 it's a 7% win
#    (ratio 0.93×) but at S≥16 it's catastrophic (1.8-3.8× slower) because
#    MLX's matmul switches to a simdgroup-matrix path my kernel doesn't
#    mirror.
#
# Where this would still be the right design:
# - Apple chips with smaller L1 where x doesn't stay cached
# - Shapes where IN > L1 capacity (~96K fp16 elements)
# - Workloads where we WANT to avoid two dispatches (latency-critical)
#
# The actual MLP win on this hardware/shape comes from reducing weight
# bandwidth (the dominant cost): see the `mlp_lowbit` quantization
# preset in `quant.py` (q3 gs=32 for MLP weights), which gets a real
# +X% end-to-end at decode by cutting 25% of weight bytes.
#
# Inlined helpers below are 4-bit-only specialisations of MLX's
# `load_vector` and `qdot` from mlx/backend/metal/kernels/quantized.h.
# We duplicate rather than `#include` because mx.fast.metal_kernel
# doesn't expose MLX's internal headers to user kernels.

_QMV_DUAL_SRC = r"""
    constexpr int packs_per_thread       = 2;
    constexpr int num_simdgroups         = 2;
    constexpr int results_per_simdgroup  = 4;
    constexpr int pack_factor            = 8;     // bits=4 ⇒ 8 nibbles per uint32
    constexpr int bytes_per_pack         = 4;
    constexpr int values_per_thread      = pack_factor * packs_per_thread;  // 16
    constexpr int simd_size              = 32;
    constexpr int block_size             = values_per_thread * simd_size;   // 512
    constexpr int scale_step_per_thread  = GROUP_SIZE / values_per_thread;  // 4 for gs=64

    typedef float U;

    uint3 tid     = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    const device uint8_t* gate_ws = (const device uint8_t*)gate_data;
    const device uint8_t* up_ws   = (const device uint8_t*)up_data;
    const device T*       g_s     = gate_scales;
    const device T*       g_b     = gate_biases;
    const device T*       u_s     = up_scales;
    const device T*       u_b     = up_biases;

    const int in_vec_size_w = IN * bytes_per_pack / pack_factor;
    const int in_vec_size_g = IN / GROUP_SIZE;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup)
                      + simd_gid * results_per_simdgroup;

    gate_ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    g_s     += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    g_b     += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

    up_ws   += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    u_s     += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    u_b     += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;

    const device T* x_p = x + tid.x * IN + simd_lid * values_per_thread;
    device T* gy_p = gate_y + tid.x * OUT + out_row;
    device T* uy_p = up_y   + tid.x * OUT + out_row;

    thread U x_thread[values_per_thread];
    thread U result_g[results_per_simdgroup] = {0};
    thread U result_u[results_per_simdgroup] = {0};

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

        gate_ws += block_size * bytes_per_pack / pack_factor;
        up_ws   += block_size * bytes_per_pack / pack_factor;
        g_s     += block_size / GROUP_SIZE;
        g_b     += block_size / GROUP_SIZE;
        u_s     += block_size / GROUP_SIZE;
        u_b     += block_size / GROUP_SIZE;
        x_p     += block_size;
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
        result_g[row] = simd_sum(result_g[row]);
        result_u[row] = simd_sum(result_u[row]);
        if (simd_lid == 0) {
            gy_p[row] = static_cast<T>(result_g[row]);
            uy_p[row] = static_cast<T>(result_u[row]);
        }
    }
"""


def _make_qmv_dual_kernel():
    return mx.fast.metal_kernel(
        name="motif_qmv_dual_q4",
        input_names=[
            "x",
            "gate_data",
            "gate_scales",
            "gate_biases",
            "up_data",
            "up_scales",
            "up_biases",
        ],
        output_names=["gate_y", "up_y"],
        source=_QMV_DUAL_SRC,
        header=Q4_QDOT_HEADER,
    )


_qmv_dual_kernel = None


def qmv_dual_q4_reference(
    x: mx.array,
    gate_q: tuple,
    up_q: tuple,
    group_size: int = 64,
    bits: int = 4,
) -> tuple:
    """Pure-MLX reference: two `mx.quantized_matmul` calls."""
    gate_y = mx.quantized_matmul(
        x,
        *gate_q,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    up_y = mx.quantized_matmul(
        x,
        *up_q,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    return gate_y, up_y


def qmv_dual_q4(
    x: mx.array,
    gate_q: tuple,
    up_q: tuple,
    group_size: int = 64,
    bits: int = 4,
) -> tuple:
    """Fused gate+up q4 GEMV (decode-time MLP front-half).

    Args:
        x: activation, shape `(..., IN)`, fp16/bf16/fp32.
        gate_q, up_q: each is a `(data, scales, biases)` triple from
            `mx.quantize` of an `(OUT, IN)` weight matrix. Both must
            share the same OUT/IN dims, group_size, and bit width.
        group_size: must divide IN and equal `qdot`'s scale_step assumption
            (64 in the current kernel hard-coding).
        bits: 4 only for now.

    Returns:
        `(gate_y, up_y)` — each shape `(..., OUT)`, dtype matches `x`.

    Constraints (asserted):
        - bits == 4 (the helper inlines the 4-bit `qdot` only)
        - IN % 512 == 0  (block_size for `qmv_fast`)
        - OUT % 8 == 0   (one threadgroup emits 8 rows)
        - group_size == 64 (kernel hard-codes scale_step_per_thread)
    """
    if _DISABLE:
        return qmv_dual_q4_reference(x, gate_q, up_q, group_size, bits)

    assert bits == 4, "qmv_dual_q4 currently 4-bit only"
    assert group_size == 64, "qmv_dual_q4 currently group_size=64 only"

    gate_data, gate_scales, gate_biases = gate_q
    up_data, up_scales, up_biases = up_q

    OUT, IN_packed = gate_data.shape[-2], gate_data.shape[-1]
    IN = IN_packed * (32 // bits)  # 8 for 4-bit
    assert up_data.shape == gate_data.shape, "gate/up weight shapes must match"
    assert IN % 512 == 0, f"IN={IN} must be divisible by block_size=512"
    assert OUT % 8 == 0, f"OUT={OUT} must be divisible by 8 (one threadgroup tile)"
    assert x.shape[-1] == IN, f"x last dim {x.shape[-1]} != IN {IN}"

    global _qmv_dual_kernel
    if _qmv_dual_kernel is None:
        _qmv_dual_kernel = _make_qmv_dual_kernel()

    *lead, in_dim = x.shape
    rows = 1
    for n in lead:
        rows *= n
    if rows == 0:
        return mx.zeros((*lead, OUT), dtype=x.dtype), mx.zeros((*lead, OUT), dtype=x.dtype)

    x_flat = x.reshape(rows, IN)

    # Threadgroup: 32 lanes × 2 simdgroups; tid.x = batch row, tid.y = output tile.
    tiles = OUT // 8
    threadgroup = (32, 2, 1)
    grid = (32 * rows, 2 * tiles, 1)

    gate_y, up_y = _qmv_dual_kernel(
        inputs=[
            x_flat,
            gate_data,
            gate_scales,
            gate_biases,
            up_data,
            up_scales,
            up_biases,
        ],
        template=[("T", x.dtype), ("IN", IN), ("OUT", OUT), ("GROUP_SIZE", group_size)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, OUT), (rows, OUT)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return gate_y.reshape(*lead, OUT), up_y.reshape(*lead, OUT)
