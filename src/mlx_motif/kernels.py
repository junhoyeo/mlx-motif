"""
Custom Metal kernels for the Motif port.

Each kernel ships with a pure-MLX reference and a small validator. On the
M1/M1 Max chips there is a known `mx.fast.metal_kernel` correctness bug
(ml-explore/mlx#2205); we therefore numerically validate every kernel
against its reference and fall back to MLX ops if the kernel is disabled
via the `MLX_MOTIF_DISABLE_KERNELS` environment variable.

Public API:
    polynorm(x, weight, bias, eps) -> mx.array
    polynorm_reference(x, weight, bias, eps) -> mx.array
    polynorm_mul(gate, up, weight, bias, eps) -> mx.array
    polynorm_mul_reference(gate, up, weight, bias, eps) -> mx.array
    gda_post(merged, subln_weight, lambda_full, lambda_init, q_groups, gr, eps) -> mx.array
    gda_post_reference(...) -> mx.array
    gda_decode(q1, q2, k1, k2, v1, v2, subln_weight,
               lambda_full, lambda_init, gr, eps) -> mx.array
    gda_decode_reference(...) -> mx.array
    sdpa_dual_v(q, k, v1, v2, scale) -> mx.array     # NEW: shared QK, dual V
    sdpa_dual_v_reference(q, k, v1, v2, scale) -> mx.array
"""

from __future__ import annotations

import os

import mlx.core as mx

_DISABLE = os.environ.get("MLX_MOTIF_DISABLE_KERNELS", "0") not in ("0", "", "false", "False")


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


def polynorm_reference(
    x: mx.array, weight: mx.array, bias: mx.array, eps: float
) -> mx.array:
    """Pure-MLX reference, mathematically identical to the kernel."""
    def _rms(z):
        return z * mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + eps)

    x2 = x * x
    x3 = x2 * x
    return weight[0] * _rms(x3) + weight[1] * _rms(x2) + weight[2] * _rms(x) + bias


def polynorm(
    x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-6
) -> mx.array:
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
# Fused post-attention GDA reduction
# --------------------------------------------------------------------------- #
#
# Replaces the 5+ MLX ops that follow the two SDPAs in the Grouped Differential
# Attention forward:
#
#   attn_o      = merged[:, :q_origin]                  # (B, q_origin, S, 2d)
#   attn_n_grp  = merged[:, q_origin:]                  # (B, q_groups, S, 2d)
#   attn_n      = repeat(attn_n_grp, gr, axis=1)        # (B, q_origin, S, 2d)
#   diff        = attn_o - lambda_full * attn_n
#   out         = SubLN(diff) * (1 - lambda_init)
#
# where q_origin = q_groups * gr. Output shape: (B, q_origin, S, 2d). The
# noise heads `merged[:, q_origin:]` are not produced — they're consumed by
# the differential subtract.

_GDA_POST_SRC = r"""
    // Layout: one threadgroup per output row (b, h_o, s). Each row has
    // CHANNELS = 2 * head_dim contiguous channels.
    uint row     = threadgroup_position_in_grid.x;
    uint tid     = thread_position_in_threadgroup.x;
    uint tgsize  = threads_per_threadgroup.x;

    // Decompose row into (b, h_o, s) using contiguous (B, q_origin, S) layout.
    uint hs    = (Q_ORIGIN * S);
    uint b     = row / hs;
    uint hosx  = row - b * hs;
    uint h_o   = hosx / S;
    uint s     = hosx - h_o * S;
    uint h_n   = Q_ORIGIN + (h_o / GR);   // matching noise head

    // merged is (B, Q_HEADS, S, CHANNELS) where Q_HEADS = Q_ORIGIN + Q_GROUPS.
    uint q_heads_total = Q_ORIGIN + Q_GROUPS;
    const device T* row_o = merged + (((b * q_heads_total + h_o) * S) + s) * CHANNELS;
    const device T* row_n = merged + (((b * q_heads_total + h_n) * S) + s) * CHANNELS;
    device T*       row_y = y      + (((b * Q_ORIGIN     + h_o) * S) + s) * CHANNELS;

    float lam   = float(lambda_full[0]);
    float scale = float(scale_in[0]);
    float eps   = float(eps_in[0]);

    // Pass 1: compute the per-row sum of squares of the differential.
    float ssq = 0.0f;
    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d = float(row_o[i]) - lam * float(row_n[i]);
        ssq += d * d;
    }

    threadgroup float tg_partial[32];
    float r = simd_sum(ssq);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) tg_partial[sg_id] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v = (lane < n_sg) ? tg_partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) tg_partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = metal::rsqrt(tg_partial[0] / float(CHANNELS) + eps);

    // Pass 2: emit the SubLN-normalised, scaled output.
    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d  = float(row_o[i]) - lam * float(row_n[i]);
        float sw = float(subln_w[i]);
        row_y[i] = T(d * rms_inv * sw * scale);
    }
"""


def _make_gda_post_kernel():
    return mx.fast.metal_kernel(
        name="motif_gda_post",
        input_names=["merged", "subln_w", "lambda_full", "scale_in", "eps_in"],
        output_names=["y"],
        source=_GDA_POST_SRC,
    )


_gda_post_kernel = None


def gda_post_reference(
    merged: mx.array,
    subln_weight: mx.array,
    lambda_full: mx.array,
    lambda_init: float,
    q_groups: int,
    gr: int,
    eps: float,
) -> mx.array:
    """Pure-MLX equivalent of the fused post-attention GDA reduction."""
    q_origin = q_groups * gr
    attn_o = merged[:, :q_origin]
    attn_n_grp = merged[:, q_origin:]
    attn_n = mx.repeat(attn_n_grp, gr, axis=1)
    diff = attn_o - lambda_full * attn_n
    rms_inv = mx.rsqrt(mx.mean(diff * diff, axis=-1, keepdims=True) + eps)
    return diff * rms_inv * subln_weight * (1.0 - lambda_init)


# --------------------------------------------------------------------------- #
# Fused split-input GDA post-reduction (avoids the noise-into-merged concat)
# --------------------------------------------------------------------------- #

_GDA_POST_SPLIT_SRC = r"""
    // Same as gda_post but reads attn_o (B, q_origin, S, 2d) and
    // attn_n (B, q_groups, S, 2d) as separate buffers — saves the
    // (B, q_origin+q_groups, S, 2d) concat allocation upstream.
    uint row     = threadgroup_position_in_grid.x;
    uint tid     = thread_position_in_threadgroup.x;
    uint tgsize  = threads_per_threadgroup.x;

    uint hs    = (Q_ORIGIN * S);
    uint b     = row / hs;
    uint hosx  = row - b * hs;
    uint h_o   = hosx / S;
    uint s     = hosx - h_o * S;
    uint h_n   = h_o / GR;       // noise head index (0..q_groups-1)

    const device T* row_o = attn_o + (((b * Q_ORIGIN + h_o) * S) + s) * CHANNELS;
    const device T* row_n = attn_n + (((b * Q_GROUPS + h_n) * S) + s) * CHANNELS;
    device T*       row_y = y      + (((b * Q_ORIGIN + h_o) * S) + s) * CHANNELS;

    float lam   = float(lambda_full[0]);
    float scale = float(scale_in[0]);
    float eps   = float(eps_in[0]);

    float ssq = 0.0f;
    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d = float(row_o[i]) - lam * float(row_n[i]);
        ssq += d * d;
    }

    threadgroup float tg_partial[32];
    float r = simd_sum(ssq);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) tg_partial[sg_id] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v = (lane < n_sg) ? tg_partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) tg_partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = metal::rsqrt(tg_partial[0] / float(CHANNELS) + eps);

    for (uint i = tid; i < CHANNELS; i += tgsize) {
        float d  = float(row_o[i]) - lam * float(row_n[i]);
        float sw = float(subln_w[i]);
        row_y[i] = T(d * rms_inv * sw * scale);
    }
"""


def _make_gda_post_split_kernel():
    return mx.fast.metal_kernel(
        name="motif_gda_post_split",
        input_names=["attn_o", "attn_n", "subln_w", "lambda_full", "scale_in", "eps_in"],
        output_names=["y"],
        source=_GDA_POST_SPLIT_SRC,
    )


_gda_post_split_kernel = None


def gda_post_split_reference(
    attn_o: mx.array,
    attn_n: mx.array,
    subln_weight: mx.array,
    lambda_full: mx.array,
    lambda_init: float,
    gr: int,
    eps: float,
) -> mx.array:
    """Pure-MLX reference: matches gda_post_reference but takes split inputs."""
    n_broadcast = mx.repeat(attn_n, gr, axis=1)
    diff = attn_o - lambda_full * n_broadcast
    rms_inv = mx.rsqrt(mx.mean(diff * diff, axis=-1, keepdims=True) + eps)
    return diff * rms_inv * subln_weight * (1.0 - lambda_init)


def gda_post_split(
    attn_o: mx.array,
    attn_n: mx.array,
    subln_weight: mx.array,
    lambda_full: mx.array,
    lambda_init: float,
    gr: int,
    eps: float = 1e-5,
) -> mx.array:
    """
    Same fused reduction as `gda_post` but reads `attn_o` (B, q_origin, S, 2d)
    and `attn_n` (B, q_groups, S, 2d) as separate inputs — no upstream
    `mx.concatenate` needed.
    """
    if _DISABLE:
        return gda_post_split_reference(
            attn_o, attn_n, subln_weight, lambda_full, lambda_init, gr, eps
        )

    global _gda_post_split_kernel
    if _gda_post_split_kernel is None:
        _gda_post_split_kernel = _make_gda_post_split_kernel()

    B, q_origin, S, channels = attn_o.shape
    _, q_groups, _, _ = attn_n.shape
    assert q_origin == q_groups * gr, (
        f"q_origin={q_origin} != q_groups({q_groups}) * gr({gr})"
    )

    rows = B * q_origin * S
    tg = min(256, max(32, ((channels + 31) // 32) * 32))
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale = mx.array([1.0 - lambda_init], dtype=mx.float32)
    eps_arr = mx.array([eps], dtype=mx.float32)

    out = _gda_post_split_kernel(
        inputs=[
            attn_o, attn_n,
            subln_weight.astype(attn_o.dtype),
            lambda_full.astype(mx.float32),
            scale, eps_arr,
        ],
        template=[
            ("T", attn_o.dtype),
            ("Q_ORIGIN", q_origin),
            ("Q_GROUPS", q_groups),
            ("GR", gr),
            ("S", S),
            ("CHANNELS", channels),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, q_origin, S, channels)],
        output_dtypes=[attn_o.dtype],
    )[0]
    return out


def gda_post(
    merged: mx.array,
    subln_weight: mx.array,
    lambda_full: mx.array,
    lambda_init: float,
    q_groups: int,
    gr: int,
    eps: float = 1e-5,
) -> mx.array:
    """
    Fused post-SDPA GDA reduction. `merged` is the concatenated output of the
    two-V SDPA call, shape (B, q_groups*gr + q_groups, S, 2*head_dim).
    Returns the SubLN-normalised differential output (B, q_groups*gr, S, 2*head_dim).
    """
    if _DISABLE:
        return gda_post_reference(
            merged, subln_weight, lambda_full, lambda_init, q_groups, gr, eps
        )

    global _gda_post_kernel
    if _gda_post_kernel is None:
        _gda_post_kernel = _make_gda_post_kernel()

    B, q_heads, S, channels = merged.shape
    q_origin = q_groups * gr
    assert q_heads == q_origin + q_groups, (
        f"q_heads={q_heads} expected {q_origin}+{q_groups}"
    )

    rows = B * q_origin * S
    tg = min(256, max(32, ((channels + 31) // 32) * 32))
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale = mx.array([1.0 - lambda_init], dtype=mx.float32)
    eps_arr = mx.array([eps], dtype=mx.float32)

    out = _gda_post_kernel(
        inputs=[
            merged,
            subln_weight.astype(merged.dtype),
            lambda_full.astype(mx.float32),
            scale,
            eps_arr,
        ],
        template=[
            ("T", merged.dtype),
            ("Q_ORIGIN", q_origin),
            ("Q_GROUPS", q_groups),
            ("GR", gr),
            ("S", S),
            ("CHANNELS", channels),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, q_origin, S, channels)],
        output_dtypes=[merged.dtype],
    )[0]
    return out


# --------------------------------------------------------------------------- #
# Shared-QK, dual-V SDPA decode — 2-pass variant (`sdpa_dual_v_2pass`)
# --------------------------------------------------------------------------- #
#
# Modeled after MLX's `sdpa_vector_2pass_1` / `_2pass_2` (long-context
# decode). Two kernels:
#
#   PASS 1: One simdgroup (32 threads) per (B, H_q, block). Each block
#           processes a strided slice of KV positions and writes
#           per-block partials (max, sum, partial_o1[D], partial_o2[D])
#           to global memory. Massive parallelism via the `blocks` axis.
#
#   PASS 2: One threadgroup of 32×32 = 1024 threads per (B, H_q).
#           Reduces the `blocks` partials, computes per-channel weighted
#           sums for both V slabs, and writes the final 2·D output.
#
# Wins over single-pass at very long KV (16k+):
#   - Single-pass uses BN=32 simdgroups; each walks KV/32 positions
#     serially. At KV=16k that's 512 positions/simdgroup.
#   - 2-pass uses `blocks` * 32 simdgroups (~32× more threadgroups), so
#     each simdgroup walks fewer positions and the GPU's threadgroup
#     scheduler stays saturated.
#
# At short KV (< ~1k) the single-pass wins because the per-block fixed
# overhead dominates. The wrapper auto-dispatches based on KV length.

_DUAL_V_PASS1_SRC = r"""
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread  = D / BD;

    typedef float U;

    // Grid: (kv_heads, batch, blocks). Threadgroup: 1 simdgroup of 32 threads.
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint batch_idx   = threadgroup_position_in_grid.y;
    uint block_idx   = threadgroup_position_in_grid.z;
    uint sg_lid      = thread_index_in_simdgroup;
    uint num_kv_heads = threadgroups_per_grid.x;

    // Each kv_head fans out to GQA_FACTOR query heads. We process all of
    // them in this threadgroup so we don't reload K/V from global per Q head.
    // For Q-head loop, this thread emits partials for each Q head one at a time.
    //
    // Layout of inputs/outputs:
    //   q:          (batch, q_heads,  1,    D)
    //   k:          (batch, kv_heads, KV,   D)
    //   v1, v2:     (batch, kv_heads, KV,   D)
    //   partial_o1: (batch, q_heads,  blocks, D)
    //   partial_o2: (batch, q_heads,  blocks, D)
    //   maxs:       (batch, q_heads,  blocks)
    //   sums:       (batch, q_heads,  blocks)
    uint num_q_heads = num_kv_heads * uint(GQA_FACTOR);

    const device T* k_base  = k  + ((batch_idx * num_kv_heads + kv_head_idx) * KV_SEQ * D)
                                  + block_idx * D + sg_lid * qk_per_thread;
    const device T* v1_base = v1 + ((batch_idx * num_kv_heads + kv_head_idx) * KV_SEQ * D)
                                  + block_idx * D + sg_lid * v_per_thread;
    const device T* v2_base = v2 + ((batch_idx * num_kv_heads + kv_head_idx) * KV_SEQ * D)
                                  + block_idx * D + sg_lid * v_per_thread;

    float scale = float(scale_in[0]);

    // Each q-head in this kv-group reuses the same K/V loop but maintains
    // its own softmax + accumulators.
    for (uint sub = 0; sub < uint(GQA_FACTOR); ++sub) {
        uint q_head_idx = kv_head_idx * uint(GQA_FACTOR) + sub;

        const device T* q_p = q + ((batch_idx * num_q_heads + q_head_idx) * D)
                                + sg_lid * qk_per_thread;

        thread U q_r[qk_per_thread];
        for (int j = 0; j < qk_per_thread; ++j) q_r[j] = scale * U(q_p[j]);

        thread U o1[v_per_thread]; thread U o2[v_per_thread];
        for (int j = 0; j < v_per_thread; ++j) { o1[j] = 0; o2[j] = 0; }

        U max_score = U(-1e30f);
        U sum_exp_score = U(0);

        const device T* k_p  = k_base;
        const device T* v1_p = v1_base;
        const device T* v2_p = v2_base;

        // Stride by `BLOCKS * D` along the KV axis.
        int kv_stride = int(BLOCKS) * D;
        for (uint p = block_idx; p < KV_SEQ; p += uint(BLOCKS)) {
            thread U k_r[qk_per_thread];
            for (int j = 0; j < qk_per_thread; ++j) k_r[j] = U(k_p[j]);
            U score = 0;
            for (int j = 0; j < qk_per_thread; ++j) score += q_r[j] * k_r[j];
            score = simd_sum(score);

            U new_max = max(max_score, score);
            U fac = metal::fast::exp(max_score - new_max);
            U exp_score = metal::fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * fac + exp_score;

            for (int j = 0; j < v_per_thread; ++j) {
                o1[j] = o1[j] * fac + exp_score * U(v1_p[j]);
                o2[j] = o2[j] * fac + exp_score * U(v2_p[j]);
            }
            k_p += kv_stride; v1_p += kv_stride; v2_p += kv_stride;
        }

        // Write per-block partials. Each thread owns v_per_thread channels
        // of o1 and o2 at offset `sg_lid * v_per_thread`.
        device T* p1 = partial_o1 + ((batch_idx * num_q_heads + q_head_idx) * uint(BLOCKS) + block_idx) * D
                                  + sg_lid * v_per_thread;
        device T* p2 = partial_o2 + ((batch_idx * num_q_heads + q_head_idx) * uint(BLOCKS) + block_idx) * D
                                  + sg_lid * v_per_thread;
        for (int j = 0; j < v_per_thread; ++j) {
            p1[j] = T(o1[j]);
            p2[j] = T(o2[j]);
        }

        if (sg_lid == 0) {
            uint stat_off = (batch_idx * num_q_heads + q_head_idx) * uint(BLOCKS) + block_idx;
            maxs[stat_off] = float(max_score);
            sums[stat_off] = float(sum_exp_score);
        }
    }
"""


_DUAL_V_PASS2_SRC = r"""
    // Mirrors MLX `sdpa_vector_2pass_2` (sdpa_vector.h:320-394). Assumes
    // BLOCKS == BN (= 32). Each simdgroup g handles block g — reads its
    // block's per-thread channel slice, scales by block g's local factor.
    // Cross-simdgroup transpose then merges blocks per output channel.
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;

    typedef float U;

    uint head_idx = threadgroup_position_in_grid.x;
    uint sg_gid   = simdgroup_index_in_threadgroup;
    uint sg_lid   = thread_index_in_simdgroup;

    // Pointer offsets:
    //   p*_base: skip to (this head, block sg_gid, lane's channel slice)
    //   y_base:  skip to (this head, simdgroup's output channel slice)
    const device T* p1_base = partial_o1 + head_idx * uint(BLOCKS) * D
                                         + sg_gid * D
                                         + sg_lid * elem_per_thread;
    const device T* p2_base = partial_o2 + head_idx * uint(BLOCKS) * D
                                         + sg_gid * D
                                         + sg_lid * elem_per_thread;
    const device float* maxs_p = maxs + head_idx * uint(BLOCKS);
    const device float* sums_p = sums + head_idx * uint(BLOCKS);
    device T*           y_base = y + head_idx * (2 * D)
                                   + sg_gid * elem_per_thread;

    // Cross-simdgroup max/sum reduction. Lane l reads block l's stats.
    U lane_max = U(maxs_p[sg_lid]);
    U gmax = simd_max(lane_max);
    U lane_fac = metal::fast::exp(lane_max - gmax);
    U gsum = simd_sum(U(sums_p[sg_lid]) * lane_fac);

    // Per-simdgroup factor (this simdgroup handles block sg_gid).
    U my_fac = metal::fast::exp(U(maxs_p[sg_gid]) - gmax);

    // Read this lane's channel slice of block sg_gid's partials, scaled.
    thread U o1[elem_per_thread]; thread U o2[elem_per_thread];
    for (int j = 0; j < elem_per_thread; ++j) {
        o1[j] = my_fac * U(p1_base[j]);
        o2[j] = my_fac * U(p2_base[j]);
    }

    // Transpose-merge across simdgroups for both V slabs.
    threadgroup U xfer[BN * BD];
    for (int i = 0; i < elem_per_thread; ++i) {
        xfer[sg_lid * BD + sg_gid] = o1[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o1[i] = simd_sum(xfer[sg_gid * BD + sg_lid]);
        o1[i] = (gsum == 0) ? o1[i] : (o1[i] / gsum);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        xfer[sg_lid * BD + sg_gid] = o2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o2[i] = simd_sum(xfer[sg_gid * BD + sg_lid]);
        o2[i] = (gsum == 0) ? o2[i] : (o2[i] / gsum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 of each simdgroup writes its v_per_thread channels for both slabs.
    if (sg_lid == 0) {
        for (int i = 0; i < elem_per_thread; ++i) {
            y_base[i] = T(o1[i]);
            y_base[D + i] = T(o2[i]);
        }
    }
"""


def _make_dual_v_pass1_kernel():
    return mx.fast.metal_kernel(
        name="motif_dual_v_pass1",
        input_names=["q", "k", "v1", "v2", "scale_in"],
        output_names=["partial_o1", "partial_o2", "maxs", "sums"],
        source=_DUAL_V_PASS1_SRC,
    )


def _make_dual_v_pass2_kernel():
    return mx.fast.metal_kernel(
        name="motif_dual_v_pass2",
        input_names=["partial_o1", "partial_o2", "maxs", "sums"],
        output_names=["y"],
        source=_DUAL_V_PASS2_SRC,
    )


_dual_v_pass1_kernel = None
_dual_v_pass2_kernel = None
_DUAL_V_2PASS_BLOCKS = 32  # must equal BN in pass 2 (32)


def sdpa_dual_v_2pass(
    q: mx.array, k: mx.array, v1: mx.array, v2: mx.array, scale: float
) -> mx.array:
    """2-pass variant of `sdpa_dual_v`. Same semantics, optimized for long KV."""
    if _DISABLE:
        H_q = q.shape[1]; H_kv = k.shape[1]
        gqa = H_q // H_kv
        if gqa > 1:
            k = mx.repeat(k, gqa, axis=1)
            v1 = mx.repeat(v1, gqa, axis=1)
            v2 = mx.repeat(v2, gqa, axis=1)
        return sdpa_dual_v_reference(q, k, v1, v2, scale)

    global _dual_v_pass1_kernel, _dual_v_pass2_kernel
    if _dual_v_pass1_kernel is None:
        _dual_v_pass1_kernel = _make_dual_v_pass1_kernel()
        _dual_v_pass2_kernel = _make_dual_v_pass2_kernel()

    B, H_q, S_q, D = q.shape
    _, H_kv, KV, _ = k.shape
    assert S_q == 1, "decode-only"
    assert D % 32 == 0
    assert H_q % H_kv == 0
    gqa_factor = H_q // H_kv

    BLOCKS = _DUAL_V_2PASS_BLOCKS
    scale_arr = mx.array([scale], dtype=mx.float32)

    # Pass 1 grid: (kv_heads, batch, blocks). Threadgroup: 32 threads (1 simdgroup).
    pass1_grid = (H_kv * 32, B, BLOCKS)
    pass1_threadgroup = (32, 1, 1)

    # Output shapes for pass 1.
    p1_shape = (B, H_q, BLOCKS, D)
    stat_shape = (B, H_q, BLOCKS)

    p1, p2, maxs, sums = _dual_v_pass1_kernel(
        inputs=[q, k, v1, v2, scale_arr],
        template=[
            ("T", q.dtype), ("D", D), ("KV_SEQ", KV),
            ("GQA_FACTOR", gqa_factor), ("BLOCKS", BLOCKS),
        ],
        grid=pass1_grid,
        threadgroup=pass1_threadgroup,
        output_shapes=[p1_shape, p1_shape, stat_shape, stat_shape],
        output_dtypes=[q.dtype, q.dtype, mx.float32, mx.float32],
    )

    # Pass 2 grid: (B*H_q, 1, 1). Threadgroup: 1024.
    pass2_grid = (B * H_q * 1024, 1, 1)
    pass2_threadgroup = (1024, 1, 1)

    out = _dual_v_pass2_kernel(
        inputs=[p1, p2, maxs, sums],
        template=[("T", q.dtype), ("D", D), ("BLOCKS", BLOCKS)],
        grid=pass2_grid,
        threadgroup=pass2_threadgroup,
        output_shapes=[(B, H_q, 1, 2 * D)],
        output_dtypes=[q.dtype],
    )[0]
    return out


# --------------------------------------------------------------------------- #
# Shared-QK, dual-V SDPA decode (`sdpa_dual_v`)
# --------------------------------------------------------------------------- #
#
# Custom Metal kernel that performs ONE softmax(Q·Kᵀ) computation and applies
# the resulting attention weights to TWO independent V tensors (V1, V2) in a
# single pass. Output: cat([attn·V1, attn·V2], axis=-1) — shape (B, H, 1, 2·D).
#
# Why this exists:
#   `mx.fast.scaled_dot_product_attention` requires `query_head_dim ==
#   value_head_dim` and only ships templates for D=V ∈ {64, 96, 128, 256}
#   (see `mlx/backend/metal/scaled_dot_product_attention.cpp:618-621`). For
#   the GDA grouped variant we need V of width 2·D=256 paired with D=128
#   K — that combination falls back to the slow generic SDPA path.
#
#   Calling MLX SDPA twice with V=128 (once per V slab) avoids the fallback
#   but recomputes Q·Kᵀ + softmax (~50% of the per-call work) twice. This
#   kernel computes the QK-and-softmax exactly once and accumulates into TWO
#   V slabs in the same KV loop, so each KV step does:
#     - 1 partial-dot for QK score (same as MLX)
#     - 2 weighted-V updates (vs MLX's 1)
#   That's roughly 1.5× MLX's per-step compute but produces 2× the output —
#   net ~25% saving over the two-call approach.
#
# Threadgroup layout matches MLX's `sdpa_vector`: BN=32 simdgroups × BD=32
# lanes = 1024 threads. Per-thread persistent state stays at 10 fp32
# (q[4] + o1[4] + o2[4] - actually 12, but transient k[4] is reused) so
# we don't blow the register-pressure cap that the 4-V flash variant hit.
#
# Channel coverage: each simdgroup g produces v_per_thread=D/BD=4 channels
# at output offsets `[g*4, g*4+4)` (V1 slab) and `[D + g*4, D + g*4+4)`
# (V2 slab). 32 simdgroups × 4 channels × 2 slabs = 2D ✓.

_SDPA_DUAL_V_SRC = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread  = D / BD;

    typedef float U;

    uint head_idx = threadgroup_position_in_grid.x;     // batch*Q_HEADS linear index
    uint sg_gid   = simdgroup_index_in_threadgroup;
    uint sg_lid   = thread_index_in_simdgroup;

    // GQA broadcast: each `GQA_FACTOR` consecutive Q heads share one (K, V*) head.
    // For GQA_FACTOR=1 this is the standard one-to-one mapping.
    uint kv_head_idx = head_idx / uint(GQA_FACTOR);

    const device T* q_p  = q  + head_idx    * D                       + sg_lid * qk_per_thread;
    const device T* k_p  = k  + kv_head_idx * (KV_SEQ * D) + sg_gid * D + sg_lid * qk_per_thread;
    const device T* v1_p = v1 + kv_head_idx * (KV_SEQ * D) + sg_gid * D + sg_lid * v_per_thread;
    const device T* v2_p = v2 + kv_head_idx * (KV_SEQ * D) + sg_gid * D + sg_lid * v_per_thread;
    device T*       y_p  = y  + head_idx    * (2 * D);

    float scale = float(scale_in[0]);

    // Per-lane Q (scaled once).
    thread U q_r[qk_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) {
        q_r[j] = scale * U(q_p[j]);
    }

    // Per-lane V accumulators — two slabs, v_per_thread channels each.
    thread U o1[v_per_thread];
    thread U o2[v_per_thread];
    for (int j = 0; j < v_per_thread; ++j) { o1[j] = 0; o2[j] = 0; }

    // -1e30 instead of -INF so fast::exp(-INF - finite) doesn't return NaN
    // on M-series (lesson from commits 6922acb / acc2de7).
    U max_score = U(-1e30f);
    U sum_exp_score = U(0);

    int kv_stride = BN * D;
    for (int p = sg_gid; p < KV_SEQ; p += BN) {
        // Load this lane's K slice.
        thread U k_r[qk_per_thread];
        for (int j = 0; j < qk_per_thread; ++j) k_r[j] = U(k_p[j]);

        // QK dot, then simd_sum across the 32-lane simdgroup.
        U score = 0;
        for (int j = 0; j < qk_per_thread; ++j) score += q_r[j] * k_r[j];
        score = simd_sum(score);

        // Online softmax update.
        U new_max = max(max_score, score);
        U fac = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * fac + exp_score;

        // Update BOTH V accumulators with the same softmax weight.
        for (int j = 0; j < v_per_thread; ++j) {
            o1[j] = o1[j] * fac + exp_score * U(v1_p[j]);
            o2[j] = o2[j] * fac + exp_score * U(v2_p[j]);
        }

        k_p  += kv_stride;
        v1_p += kv_stride;
        v2_p += kv_stride;
    }

    // Cross-simdgroup max/sum reduce — exact MLX SDPA pattern.
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    threadgroup U outputs[BN * BD];

    if (sg_lid == 0) {
        max_scores[sg_gid] = max_score;
        sum_exp_scores[sg_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U lane_max = max_scores[sg_lid];
    U new_max = simd_max(lane_max);
    U lane_factor = metal::fast::exp(lane_max - new_max);
    U lane_sum = simd_sum(sum_exp_scores[sg_lid] * lane_factor);

    // Reduce both V slabs across simdgroups.
    for (int i = 0; i < v_per_thread; ++i) {
        // V1 slab
        outputs[sg_lid * BD + sg_gid] = o1[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o1[i] = simd_sum(outputs[sg_gid * BD + sg_lid] * lane_factor);
        o1[i] = (lane_sum == 0) ? o1[i] : (o1[i] / lane_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V2 slab
        outputs[sg_lid * BD + sg_gid] = o2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o2[i] = simd_sum(outputs[sg_gid * BD + sg_lid] * lane_factor);
        o2[i] = (lane_sum == 0) ? o2[i] : (o2[i] / lane_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 of each simdgroup writes its v_per_thread channels of each slab.
    if (sg_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            y_p[sg_gid * v_per_thread + i] = T(o1[i]);
            y_p[D + sg_gid * v_per_thread + i] = T(o2[i]);
        }
    }
"""


def _make_sdpa_dual_v_kernel():
    return mx.fast.metal_kernel(
        name="motif_sdpa_dual_v",
        input_names=["q", "k", "v1", "v2", "scale_in"],
        output_names=["y"],
        source=_SDPA_DUAL_V_SRC,
    )


_sdpa_dual_v_kernel = None


def sdpa_dual_v_reference(
    q: mx.array, k: mx.array, v1: mx.array, v2: mx.array, scale: float
) -> mx.array:
    """Pure-MLX equivalent: cat(SDPA(q,k,v1), SDPA(q,k,v2), axis=-1)."""
    a1 = mx.fast.scaled_dot_product_attention(q, k, v1, scale=scale)
    a2 = mx.fast.scaled_dot_product_attention(q, k, v2, scale=scale)
    return mx.concatenate([a1, a2], axis=-1)


def sdpa_dual_v(
    q: mx.array, k: mx.array, v1: mx.array, v2: mx.array, scale: float
) -> mx.array:
    """
    Shared-QK, dual-V SDPA with native GQA broadcast.

    `q` shape (B, H_q, 1, D); `k, v1, v2` shape (B, H_kv, K, D).
    Each gqa_factor=H_q/H_kv consecutive Q heads share one (K, V*) head.
    Returns (B, H_q, 1, 2·D) = cat([SDPA(q,k,v1), SDPA(q,k,v2)], axis=-1).

    Decode-only (S=1). For GQA_FACTOR=1 this is plain dual-V SDPA.
    """
    if _DISABLE:
        # Reference path: explicit repeat then 2 SDPAs.
        H_q = q.shape[1]
        H_kv = k.shape[1]
        gqa = H_q // H_kv
        if gqa > 1:
            k = mx.repeat(k, gqa, axis=1)
            v1 = mx.repeat(v1, gqa, axis=1)
            v2 = mx.repeat(v2, gqa, axis=1)
        return sdpa_dual_v_reference(q, k, v1, v2, scale)

    global _sdpa_dual_v_kernel
    if _sdpa_dual_v_kernel is None:
        _sdpa_dual_v_kernel = _make_sdpa_dual_v_kernel()

    B, H_q, S_q, D = q.shape
    _, H_kv, KV, _ = k.shape
    assert S_q == 1, "sdpa_dual_v is decode-only"
    assert D % 32 == 0, f"D={D} must be a multiple of 32"
    assert H_q % H_kv == 0, f"H_q={H_q} must be a multiple of H_kv={H_kv}"
    gqa_factor = H_q // H_kv
    assert v1.shape == v2.shape == (B, H_kv, KV, D), "v1,v2 must match (B, H_kv, K, D)"
    assert k.shape[-1] == D, "K head_dim must match Q"

    rows = B * H_q
    tg = 32 * 32  # BN * BD = 1024
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale_arr = mx.array([scale], dtype=mx.float32)
    out = _sdpa_dual_v_kernel(
        inputs=[q, k, v1, v2, scale_arr],
        template=[("T", q.dtype), ("D", D), ("KV_SEQ", KV), ("GQA_FACTOR", gqa_factor)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, H_q, 1, 2 * D)],
        output_dtypes=[q.dtype],
    )[0]
    return out


# --------------------------------------------------------------------------- #
# Flash-style fused GDA decode (S=1)  -- legacy slow version, kept for ref
# --------------------------------------------------------------------------- #
#
# Replaces the entire grouped-attention pipeline at decode time:
#
#   q_f, k_f, v1_f, v2_f construction
#       -> two SDPA calls (or one with V-stacking)
#       -> channel split + λ-subtract + SubLN + scale
#
# with a single Metal kernel. Handles S=1 (the common decode case) only.
#
# Inputs are the *post-RoPE, post-cache* per-branch tensors:
#   q1: (B, q_origin, 1, d)   origin Q (e.g., 32 heads for the 12.7B)
#   q2: (B, q_groups, 1, d)   noise Q  (e.g., 8)
#   k1: (B, q_groups, K, d)   origin K (already broadcast to q_groups; for
#                              kv_repeat=1, k_groups == q_groups)
#   k2: (B, q_groups, K, d)   noise K
#   v1: (B, q_groups, K, d)   origin V (head_dim slice — the first d channels
#                              of the doubled V)
#   v2: (B, q_groups, K, d)   noise V (the second d channels)
#
# Output: (B, q_origin, 1, 2*d) — ready for the existing reshape + o_proj.
#
# ----------------------------------------------------------------------------
# WHY THE NAIVE PORT OF MLX `sdpa_vector` DOESN'T BEAT MLX (yet):
# ----------------------------------------------------------------------------
# Studied `mlx/backend/metal/kernels/sdpa_vector.h` (vanilla SDPA decode):
#
#   threadgroup geometry: BN × BD = 32 × 32 = 1024 threads
#   per-thread persistent state: q[4] + o[4] + (max, sum)  ≈ 10 fp32 (~40B)
#   per-thread transient state:  k[4]                        ≈ 4 fp32
#   channel coverage: BN * v_per_thread = 32 * (V/BD=4) = V (=128 for d=128)
#
# Differential attention forces TWO independent online softmaxes (q1·k1 and
# q2·k2 produce DIFFERENT softmax denominators) and produces 2D output
# channels. The cleanest port multiplies per-thread persistent state by ~2:
#
#   q1[4] + q2[4] + o_origin[v_pt] + o_noise[v_pt] + (m_o, s_o, m_n, s_n)
#
# With BN=BD=32 and V_total=2D=256, v_per_thread = V_total/BD = 8, giving
# 4+4+8+8+4 = 28 fp32 (~112B) per thread — about 2.8× MLX vanilla. M1 Max's
# `maxTotalThreadsPerThreadgroup` collapses to 896 at this register
# pressure, so 1024 threads won't launch (verified empirically).
#
# Reducing to BN=16 keeps us under the cap but BREAKS channel coverage:
# `BN * v_per_thread = 16 * 8 = 128 ≠ 256`, so the cross-simdgroup
# transpose would only emit half the output channels.
#
# Three viable paths for a future iteration, in increasing complexity:
#
# 1. Two-pass kernel à la MLX `sdpa_vector_2pass_1` / `_2pass_2`. First
#    pass writes per-block (max, sum, partial output) for both branches to
#    global memory; second pass merges across blocks and emits the
#    differential output. Register pressure drops because partials live
#    in global, not registers. Adds one global write + one extra dispatch.
#
# 2. Two-kernel split: vanilla MLX SDPA shape but called twice (one per
#    branch), each writing 2D-wide output to scratch; a tiny third kernel
#    does the differential subtract + SubLN. Equivalent to current path
#    in op count but kernel selection avoids the V-cat materialisation.
#
# 3. Custom register-pressure-aware design that reduces persistent state to
#    fit BN=BD=32. Possibilities: spill output accumulators to threadgroup
#    memory (32KB ceiling on M1 — tight but feasible), or use fp16 V
#    accumulators (precision concern, especially with later SubLN).
#
# Until one of those lands, we ship the slow-but-correct serial kernel
# below behind `MLX_MOTIF_FLASH_DECODE=1` so the call site, the
# correctness reference, and the perf baseline all stay in tree. The
# default decode path is the V-stacked SDPA + `gda_post` fusion, which
# already gives +10.7% over the Phase 1 baseline (34.5 → 38.2 tok/s).
#
# ----------------------------------------------------------------------------
# DECODE-TIME PROFILE (M1 Max, 12.7B-q4, single-token isolated benchmarks):
# ----------------------------------------------------------------------------
#   q_proj 4096 -> 5120 (q4)        328 µs   (10%)
#   k_proj 4096 -> 2048 (q4)        288 µs   ( 9%)
#   v_proj 4096 -> 2048 (q4)        288 µs   ( 9%)
#   o_proj 8192 -> 4096 (q4)        437 µs   (13%)
#   mlp gate 4096 -> 16384 (q4)     522 µs   (16%)
#   mlp up   4096 -> 16384 (q4)     522 µs   (16%)
#   mlp down 16384 -> 4096 (q4)     584 µs   (18%)
#   SDPA q40 k16 v16(2d)            316 µs   (10%)
#
# Attention is *only 10%* of decode time per layer. The MLP path (gate +
# up + down) is 50%. Even a perfect flash GDA kernel would save at most
# 10% per token; the realistic ceiling for further attention work is
# small. The bigger remaining targets are MLP fusion and quantized KV
# (which would also need the custom cache class noted in model.py).
#
# Confirms: the current default decode path is essentially optimal for
# the attention component. Any future flash-GDA work should focus on
# *prefill* (long S) where attention compute is much larger.
# ----------------------------------------------------------------------------


_GDA_DECODE_SRC = r"""
    // One threadgroup per (b, h_o); each thread owns one output channel of 2d.
    uint row    = threadgroup_position_in_grid.x;
    uint cidx   = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;

    uint b   = row / Q_ORIGIN;
    uint h_o = row - b * Q_ORIGIN;
    uint group = h_o / GR;

    // Per-thread output channel: first d channels are branch-1, last d are branch-2.
    bool is_branch_b = cidx >= D;
    uint v_ch = is_branch_b ? (cidx - D) : cidx;

    // Pointers — all batched, fixed-stride.
    const device T* q1_b = q1 + ((b * Q_ORIGIN + h_o) * D);
    const device T* q2_b = q2 + ((b * Q_GROUPS + group) * D);
    const device T* k1_b = k1 + (((b * Q_GROUPS + group) * KV_SEQ) * D);
    const device T* k2_b = k2 + (((b * Q_GROUPS + group) * KV_SEQ) * D);
    const device T* v1_b = v1 + (((b * Q_GROUPS + group) * KV_SEQ) * D);
    const device T* v2_b = v2 + (((b * Q_GROUPS + group) * KV_SEQ) * D);
    device T*       y_b  = y  + (((b * Q_ORIGIN + h_o)  * 1)      * (2 * D));

    float scale = float(scale_in[0]);

    // Each thread loops over KV positions, computing score + weighted-V
    // contribution for ITS output channel. Per channel we maintain online
    // softmax state (max, sumexp, accum) for the channel's branch only.
    float m_o = -INFINITY, s_o = 0.0f, acc_o = 0.0f;
    float m_n = -INFINITY, s_n = 0.0f, acc_n = 0.0f;

    for (uint p = 0; p < KV_SEQ; ++p) {
        // Score for the origin branch: q1 . k1[p]  (full d-element dot product)
        float so = 0.0f, sn = 0.0f;
        for (uint i = 0; i < D; ++i) {
            so += float(q1_b[i]) * float(k1_b[p * D + i]);
            sn += float(q2_b[i]) * float(k2_b[p * D + i]);
        }
        so *= scale;
        sn *= scale;

        // Online softmax update for origin branch.
        if (so > m_o) {
            float r = metal::exp(m_o - so);
            s_o = s_o * r + 1.0f;
            acc_o = acc_o * r + (is_branch_b ? float(v2_b[p * D + v_ch])
                                              : float(v1_b[p * D + v_ch]));
            m_o = so;
        } else {
            float w = metal::exp(so - m_o);
            s_o += w;
            acc_o += w * (is_branch_b ? float(v2_b[p * D + v_ch])
                                       : float(v1_b[p * D + v_ch]));
        }
        // Online softmax update for noise branch.
        if (sn > m_n) {
            float r = metal::exp(m_n - sn);
            s_n = s_n * r + 1.0f;
            acc_n = acc_n * r + (is_branch_b ? float(v2_b[p * D + v_ch])
                                              : float(v1_b[p * D + v_ch]));
            m_n = sn;
        } else {
            float w = metal::exp(sn - m_n);
            s_n += w;
            acc_n += w * (is_branch_b ? float(v2_b[p * D + v_ch])
                                       : float(v1_b[p * D + v_ch]));
        }
    }

    float val_o = acc_o / s_o;
    float val_n = acc_n / s_n;
    float diff  = val_o - float(lambda_full[0]) * val_n;

    // Threadgroup reduction over the 2·D channels for SubLN's per-row RMS.
    threadgroup float tg[32];
    threadgroup float diff_buf[2 * D_MAX];   // staging for per-thread diff
    diff_buf[cidx] = diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ssq = diff * diff;
    float r = simd_sum(ssq);
    uint sg_id = simdgroup_index_in_threadgroup;
    uint lane  = thread_index_in_simdgroup;
    if (lane == 0) tg[sg_id] = r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0) {
        uint n_sg = simdgroups_per_threadgroup;
        float v = (lane < n_sg) ? tg[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) tg[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = metal::rsqrt(tg[0] / float(2 * D) + float(eps_in[0]));
    float scale_out = float(scale_out_in[0]);   // (1 - lambda_init)
    float sw = float(subln_w[cidx]);

    y_b[cidx] = T(diff * rms_inv * sw * scale_out);
"""


def _make_gda_decode_kernel():
    return mx.fast.metal_kernel(
        name="motif_gda_decode",
        input_names=[
            "q1", "q2", "k1", "k2", "v1", "v2",
            "subln_w", "lambda_full", "scale_in", "scale_out_in", "eps_in",
        ],
        output_names=["y"],
        source=_GDA_DECODE_SRC,
    )


_gda_decode_kernel = None


def gda_decode_reference(
    q1: mx.array, q2: mx.array,
    k1: mx.array, k2: mx.array,
    v1: mx.array, v2: mx.array,
    subln_weight: mx.array,
    lambda_full: mx.array,
    lambda_init: float,
    gr: int,
    scale: float,
    eps: float = 1e-5,
) -> mx.array:
    """Pure MLX equivalent of `gda_decode`. S = 1 only."""
    # Concatenate V channels: each branch's V becomes a 2·d-wide slab.
    v_origin = mx.concatenate([v1, v2], axis=-1)  # (B, q_groups, K, 2d)
    v_noise = v_origin                              # noise branch shares the same V

    # attn_o per origin head: SDPA(q1[h_o], k1[group], v_origin[group])
    # We compute per-group then broadcast back across `gr` origin heads.
    a_o = mx.fast.scaled_dot_product_attention(
        mx.repeat(mx.reshape(q1, q1.shape), 1, axis=1) if False else q1,
        mx.repeat(k1, gr, axis=1),
        mx.repeat(v_origin, gr, axis=1),
        scale=scale,
        mask=None,
    )  # (B, q_origin, 1, 2d)
    a_n_group = mx.fast.scaled_dot_product_attention(
        q2, k2, v_noise, scale=scale, mask=None
    )  # (B, q_groups, 1, 2d)
    a_n = mx.repeat(a_n_group, gr, axis=1)  # broadcast across origin heads in group

    diff = a_o - lambda_full * a_n
    rms_inv = mx.rsqrt(mx.mean(diff * diff, axis=-1, keepdims=True) + eps)
    return diff * rms_inv * subln_weight * (1.0 - lambda_init)


def gda_decode(
    q1: mx.array, q2: mx.array,
    k1: mx.array, k2: mx.array,
    v1: mx.array, v2: mx.array,
    subln_weight: mx.array,
    lambda_full: mx.array,
    lambda_init: float,
    gr: int,
    scale: float,
    eps: float = 1e-5,
) -> mx.array:
    if _DISABLE:
        return gda_decode_reference(
            q1, q2, k1, k2, v1, v2, subln_weight, lambda_full, lambda_init, gr, scale, eps
        )

    global _gda_decode_kernel
    if _gda_decode_kernel is None:
        _gda_decode_kernel = _make_gda_decode_kernel()

    B, q_origin, S_q, d = q1.shape
    _, q_groups, kv_seq, _ = k1.shape
    assert S_q == 1, "gda_decode only supports S=1"
    assert q_origin == q_groups * gr, f"q_origin={q_origin} != q_groups({q_groups}) * gr({gr})"

    channels = 2 * d
    rows = B * q_origin
    tg = channels  # one thread per output channel
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale_in = mx.array([scale], dtype=mx.float32)
    scale_out = mx.array([1.0 - lambda_init], dtype=mx.float32)
    eps_arr = mx.array([eps], dtype=mx.float32)

    out = _gda_decode_kernel(
        inputs=[
            q1, q2, k1, k2, v1, v2,
            subln_weight.astype(q1.dtype),
            lambda_full.astype(mx.float32),
            scale_in, scale_out, eps_arr,
        ],
        template=[
            ("T", q1.dtype),
            ("Q_ORIGIN", q_origin),
            ("Q_GROUPS", q_groups),
            ("GR", gr),
            ("KV_SEQ", kv_seq),
            ("D", d),
            ("D_MAX", d),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, q_origin, 1, channels)],
        output_dtypes=[q1.dtype],
    )[0]
    return out


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
    data: mx.array, scales: mx.array, biases: mx.array,
    group_size: int = 64, bits: int = 4,
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
# Quantized-input shared-QK dual-V SDPA (`sdpa_dual_v_q4`)
# --------------------------------------------------------------------------- #
#
# Same algorithm and threadgroup layout as `sdpa_dual_v`, but K, V1, V2 are
# read as MLX quantized triples `(data: uint32, scales: T, biases: T)` and
# dequantized in registers. Two wins over the dequant-then-fp16 path:
#   1. Bandwidth: 4-bit packs ~4× more channels per HBM byte. For D=128,
#      group_size=64, fp16 scale/bias: per-step bytes drop from 768 (fp16
#      K+V1+V2) to 216 (4-bit), ≈3.6×. At long context where attention is
#      bandwidth-bound, that's the dominant savings.
#   2. Dispatch: avoids the per-step `mx.dequantize` of K and 2×V (4 ops
#      per layer per token replaced with 0).
#
# Threadgroup geometry (mirrors `sdpa_dual_v`):
#   BN=BD=32, 1024 threads per threadgroup, one threadgroup per (B, H_q).
#   Per-lane: 4 channels of K & each V slab; channel coverage 32×4×2 = 2D ✓.
#
# Quantization layout assumptions baked into this kernel:
#   - bits ∈ {4, 8} ⇒ EL_PER_INT ∈ {8, 4}
#   - qk_per_thread = D/BD = 4 channels per lane per tensor
#   - For all qk_per_thread channels of one lane to share one scale/bias and
#     one packed uint32: `qk_per_thread <= group_size && qk_per_thread <=
#     EL_PER_INT`. Both hold for the (D=128, group_size ∈ {32,64,128}, bits ∈
#     {4,8}) combos this kernel targets.
#
# All five pitfalls from `docs/sdpa_dual_v_q4_design.md` apply:
#   sentinel=-1e30f, no zero-point shift, group_idx is per-lane, channel
#   coverage formula, little-endian uint32 packing.

_SDPA_DUAL_V_Q4_SRC = r"""
    constexpr int  BN            = 32;
    constexpr int  BD            = 32;
    constexpr int  qk_per_thread = D / BD;          // 4 for D=128
    constexpr int  v_per_thread  = D / BD;          // 4
    constexpr int  EL_PER_INT    = 32 / BITS;
    constexpr uint MASK          = (1u << BITS) - 1u;
    constexpr int  U32_PER_ROW   = D / EL_PER_INT;
    constexpr int  SCB_PER_ROW   = D / GROUP_SIZE;

    typedef float U;

    uint head_idx = threadgroup_position_in_grid.x;     // batch*Q_HEADS linear index
    uint sg_gid   = simdgroup_index_in_threadgroup;
    uint sg_lid   = thread_index_in_simdgroup;
    uint kv_head_idx = head_idx / uint(GQA_FACTOR);

    // Per-lane channel slice — same uint32_idx, shift, group_idx for K, V1, V2.
    uint base         = uint(sg_lid) * uint(qk_per_thread);
    uint u32_idx_base = base / uint(EL_PER_INT);
    uint shift_base   = (base % uint(EL_PER_INT)) * uint(BITS);
    uint group_idx    = base / uint(GROUP_SIZE);

    // Per-tensor row offset for this kv_head (computed once).
    uint head_row_u32 = kv_head_idx * uint(KV_SEQ) * uint(U32_PER_ROW);
    uint head_row_scb = kv_head_idx * uint(KV_SEQ) * uint(SCB_PER_ROW);

    // Q (fp16/bf16): same indexing as sdpa_dual_v.
    const device T* q_p = q + head_idx * uint(D) + sg_lid * uint(qk_per_thread);

    float scale = float(scale_in[0]);

    // Per-lane Q with the MLX qdot trick: precompute q_pre[j] = scale * q[j] /
    // 2^(shift_base + j*BITS). Then in the kv loop, K's nibble at position j —
    // which appears in the packed word as `nibble * 2^(shift_base + j*BITS)`
    // when masked WITHOUT shifting — multiplied by q_pre[j] gives the
    // correctly-weighted partial dot. Saves one shift per channel per step.
    thread U q_pre[qk_per_thread];
    U q_sum = U(0);
    for (int j = 0; j < qk_per_thread; ++j) {
        U qj = scale * U(q_p[j]);
        q_sum += qj;
        // Inverse of the bit-position factor 2^(shift_base + j*BITS).
        U inv_factor = U(1.0f) / U(1u << (shift_base + uint(j * BITS)));
        q_pre[j] = qj * inv_factor;
    }

    // Per-lane V accumulators — two slabs.
    thread U o1[v_per_thread];
    thread U o2[v_per_thread];
    for (int j = 0; j < v_per_thread; ++j) { o1[j] = 0; o2[j] = 0; }

    // Per-lane masks for the in-place K mask-without-shift trick.
    thread uint k_masks[qk_per_thread];
    for (int j = 0; j < qk_per_thread; ++j) {
        k_masks[j] = MASK << (shift_base + uint(j * BITS));
    }

    // -1e30 sentinel (avoid -INF / NaN trap in fast::exp).
    U max_score = U(-1e30f);
    U sum_exp_score = U(0);

    for (int p = sg_gid; p < KV_SEQ; p += BN) {
        uint row_u32 = head_row_u32 + uint(p) * uint(U32_PER_ROW) + u32_idx_base;
        uint row_scb = head_row_scb + uint(p) * uint(SCB_PER_ROW) + group_idx;

        // ---- K: in-place mask + qdot trick -----------------------------
        // dot(q, dequant(k)) = scale_k * sum_j(q[j] * raw[j]) + bias_k * sum_j(q[j])
        //                    = scale_k * sum_j(q_pre[j] * (k_pkd & k_masks[j])) + bias_k * q_sum
        uint32_t k_pkd = k_data[row_u32];
        U partial = 0;
        for (int j = 0; j < qk_per_thread; ++j) {
            partial += q_pre[j] * U(k_pkd & k_masks[j]);
        }
        U score = U(k_scales[row_scb]) * partial + U(k_biases[row_scb]) * q_sum;
        score = simd_sum(score);

        // ---- Online softmax update -------------------------------------
        U new_max = max(max_score, score);
        U fac = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * fac + exp_score;

        // ---- V1: bit-extract + accumulate ------------------------------
        {
            uint32_t v_pkd = v1_data[row_u32];
            U s = U(v1_scales[row_scb]);
            U b = U(v1_biases[row_scb]);
            for (int j = 0; j < v_per_thread; ++j) {
                uint raw = (v_pkd >> (shift_base + uint(j * BITS))) & MASK;
                o1[j] = o1[j] * fac + exp_score * (s * U(raw) + b);
            }
        }

        // ---- V2: bit-extract + accumulate ------------------------------
        {
            uint32_t v_pkd = v2_data[row_u32];
            U s = U(v2_scales[row_scb]);
            U b = U(v2_biases[row_scb]);
            for (int j = 0; j < v_per_thread; ++j) {
                uint raw = (v_pkd >> (shift_base + uint(j * BITS))) & MASK;
                o2[j] = o2[j] * fac + exp_score * (s * U(raw) + b);
            }
        }
    }

    device T* y_p = y + head_idx * uint(2 * D);

    // Cross-simdgroup max/sum reduce — exact MLX SDPA pattern.
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    threadgroup U outputs[BN * BD];

    if (sg_lid == 0) {
        max_scores[sg_gid] = max_score;
        sum_exp_scores[sg_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U lane_max = max_scores[sg_lid];
    U new_max = simd_max(lane_max);
    U lane_factor = metal::fast::exp(lane_max - new_max);
    U lane_sum = simd_sum(sum_exp_scores[sg_lid] * lane_factor);

    for (int i = 0; i < v_per_thread; ++i) {
        outputs[sg_lid * BD + sg_gid] = o1[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o1[i] = simd_sum(outputs[sg_gid * BD + sg_lid] * lane_factor);
        o1[i] = (lane_sum == 0) ? o1[i] : (o1[i] / lane_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        outputs[sg_lid * BD + sg_gid] = o2[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o2[i] = simd_sum(outputs[sg_gid * BD + sg_lid] * lane_factor);
        o2[i] = (lane_sum == 0) ? o2[i] : (o2[i] / lane_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            y_p[sg_gid * v_per_thread + i] = T(o1[i]);
            y_p[D + sg_gid * v_per_thread + i] = T(o2[i]);
        }
    }
"""


def _make_sdpa_dual_v_q4_kernel():
    return mx.fast.metal_kernel(
        name="motif_sdpa_dual_v_q4",
        input_names=[
            "q",
            "k_data", "k_scales", "k_biases",
            "v1_data", "v1_scales", "v1_biases",
            "v2_data", "v2_scales", "v2_biases",
            "scale_in",
        ],
        output_names=["y"],
        source=_SDPA_DUAL_V_Q4_SRC,
    )


_sdpa_dual_v_q4_kernel = None


def sdpa_dual_v_q4_reference(
    q: mx.array,
    k_q: tuple,
    v1_q: tuple,
    v2_q: tuple,
    scale: float,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Pure-MLX reference: dequantize K, V1, V2 and run `sdpa_dual_v_reference`.

    Tolerance for kernel-vs-reference comparisons must absorb the small
    roundoff from the kernel's fp32 accumulator vs the reference's fp16
    accumulator. The quantization noise itself is shared (both paths
    consume the exact same packed bits), so it does NOT enter the diff.
    """
    k = mx.dequantize(*k_q, group_size=group_size, bits=bits)
    v1 = mx.dequantize(*v1_q, group_size=group_size, bits=bits)
    v2 = mx.dequantize(*v2_q, group_size=group_size, bits=bits)
    H_q = q.shape[1]; H_kv = k.shape[1]
    gqa = H_q // H_kv
    if gqa > 1:
        k = mx.repeat(k, gqa, axis=1)
        v1 = mx.repeat(v1, gqa, axis=1)
        v2 = mx.repeat(v2, gqa, axis=1)
    return sdpa_dual_v_reference(q, k, v1, v2, scale)


def sdpa_dual_v_q4(
    q: mx.array,
    k_q: tuple,
    v1_q: tuple,
    v2_q: tuple,
    scale: float,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Quantized-input shared-QK, dual-V SDPA decode kernel.

    Args:
        q: (B, H_q, 1, D), bf16/fp16/fp32.
        k_q: `(data, scales, biases)` triple from `mx.quantize` of a
             (B, H_kv, KV, D) tensor.
        v1_q, v2_q: same triple format as `k_q`, two independent V tensors.
        scale: softmax scaling factor (1/sqrt(d_head)).
        group_size: quantization group size — must divide D.
        bits: 4 or 8.

    Returns:
        (B, H_q, 1, 2*D) — `cat([attn(q,k,v1), attn(q,k,v2)], axis=-1)`.
    """
    if _DISABLE:
        return sdpa_dual_v_q4_reference(q, k_q, v1_q, v2_q, scale, group_size, bits)

    global _sdpa_dual_v_q4_kernel
    if _sdpa_dual_v_q4_kernel is None:
        _sdpa_dual_v_q4_kernel = _make_sdpa_dual_v_q4_kernel()

    k_data, k_scales, k_biases = k_q
    v1_data, v1_scales, v1_biases = v1_q
    v2_data, v2_scales, v2_biases = v2_q

    B, H_q, S_q, D = q.shape
    _, H_kv, KV, n_uint = k_data.shape
    el_per_int = 32 // bits
    assert n_uint * el_per_int == D, (
        f"k_data last axis = {n_uint} ⇒ packed D = {n_uint * el_per_int}, expected {D}"
    )
    assert S_q == 1, "sdpa_dual_v_q4 is decode-only"
    assert D % 32 == 0
    assert D % group_size == 0
    assert H_q % H_kv == 0
    assert bits in (4, 8)
    gqa_factor = H_q // H_kv

    rows = B * H_q
    tg = 32 * 32
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale_arr = mx.array([scale], dtype=mx.float32)
    out = _sdpa_dual_v_q4_kernel(
        inputs=[
            q,
            k_data, k_scales, k_biases,
            v1_data, v1_scales, v1_biases,
            v2_data, v2_scales, v2_biases,
            scale_arr,
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("KV_SEQ", KV),
            ("GQA_FACTOR", gqa_factor),
            ("BITS", bits),
            ("GROUP_SIZE", group_size),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, H_q, 1, 2 * D)],
        output_dtypes=[q.dtype],
    )[0]
    return out


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

_QMV_DUAL_HEADER = r"""
template <typename T, typename U, int values_per_thread>
inline U load_vector_q4(const device T* x, thread U* x_thread) {
    U sum = 0;
    for (int i = 0; i < values_per_thread; i += 4) {
        sum += float(x[i]) + float(x[i+1]) + float(x[i+2]) + float(x[i+3]);
        x_thread[i  ] = float(x[i]);
        x_thread[i+1] = float(x[i+1]) / 16.0f;
        x_thread[i+2] = float(x[i+2]) / 256.0f;
        x_thread[i+3] = float(x[i+3]) / 4096.0f;
    }
    return sum;
}

template <typename U, int values_per_thread>
inline U qdot_q4(const device uint8_t* w,
                 const thread U* x_thread,
                 U scale, U bias, U sum) {
    U accum = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
        accum += (x_thread[4*i  ] * float(ws[i] & 0x000fu) +
                  x_thread[4*i+1] * float(ws[i] & 0x00f0u) +
                  x_thread[4*i+2] * float(ws[i] & 0x0f00u) +
                  x_thread[4*i+3] * float(ws[i] & 0xf000u));
    }
    return scale * accum + sum * bias;
}
"""


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
            "gate_data", "gate_scales", "gate_biases",
            "up_data", "up_scales", "up_biases",
        ],
        output_names=["gate_y", "up_y"],
        source=_QMV_DUAL_SRC,
        header=_QMV_DUAL_HEADER,
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
        x, *gate_q, transpose=True, group_size=group_size, bits=bits,
    )
    up_y = mx.quantized_matmul(
        x, *up_q, transpose=True, group_size=group_size, bits=bits,
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
            gate_data, gate_scales, gate_biases,
            up_data, up_scales, up_biases,
        ],
        template=[("T", x.dtype), ("IN", IN), ("OUT", OUT), ("GROUP_SIZE", group_size)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, OUT), (rows, OUT)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return gate_y.reshape(*lead, OUT), up_y.reshape(*lead, OUT)
