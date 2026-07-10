"""GDA (Grouped Differential Attention) post-reduction and decode kernels."""

from __future__ import annotations

import mlx.core as mx

from ._common import _DISABLE, _scalar_f32

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

    // Pass 1: compute the per-row sum of squares of the differential, caching
    // each thread's differential(s) in registers so pass 2 does not re-read
    // row_o/row_n from device memory and recompute (mirrors sdpa_dual_v's
    // in-register o1/o2 accumulation). TG == threads_per_threadgroup.x, so
    // each thread owns PER_THREAD channels at stride TG.
    constexpr uint PER_THREAD = (uint(CHANNELS) + uint(TG) - 1u) / uint(TG);
    float d_reg[PER_THREAD];
    float ssq = 0.0f;
    for (uint j = 0; j < PER_THREAD; ++j) {
        uint i = tid + j * uint(TG);
        float d = (i < CHANNELS) ? (float(row_o[i]) - lam * float(row_n[i])) : 0.0f;
        d_reg[j] = d;
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

    // Pass 2: emit the SubLN-normalised, scaled output from the cached d.
    for (uint j = 0; j < PER_THREAD; ++j) {
        uint i = tid + j * uint(TG);
        if (i < CHANNELS) {
            float sw = float(subln_w[i]);
            row_y[i] = T(d_reg[j] * rms_inv * sw * scale);
        }
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

    // Cache the per-thread differential(s) in registers during pass 1 so pass 2
    // reuses them instead of re-reading row_o/row_n and recomputing. TG ==
    // threads_per_threadgroup.x; each thread owns PER_THREAD channels at stride TG.
    constexpr uint PER_THREAD = (uint(CHANNELS) + uint(TG) - 1u) / uint(TG);
    float d_reg[PER_THREAD];
    float ssq = 0.0f;
    for (uint j = 0; j < PER_THREAD; ++j) {
        uint i = tid + j * uint(TG);
        float d = (i < CHANNELS) ? (float(row_o[i]) - lam * float(row_n[i])) : 0.0f;
        d_reg[j] = d;
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

    for (uint j = 0; j < PER_THREAD; ++j) {
        uint i = tid + j * uint(TG);
        if (i < CHANNELS) {
            float sw = float(subln_w[i]);
            row_y[i] = T(d_reg[j] * rms_inv * sw * scale);
        }
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
    assert q_origin == q_groups * gr, f"q_origin={q_origin} != q_groups({q_groups}) * gr({gr})"

    rows = B * q_origin * S
    tg = min(256, max(32, ((channels + 31) // 32) * 32))
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale = _scalar_f32(1.0 - lambda_init)
    eps_arr = _scalar_f32(eps)

    out = _gda_post_split_kernel(
        inputs=[
            attn_o,
            attn_n,
            subln_weight.astype(attn_o.dtype),
            lambda_full.astype(mx.float32),
            scale,
            eps_arr,
        ],
        template=[
            ("T", attn_o.dtype),
            ("Q_ORIGIN", q_origin),
            ("Q_GROUPS", q_groups),
            ("GR", gr),
            ("S", S),
            ("CHANNELS", channels),
            ("TG", tg),
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
        return gda_post_reference(merged, subln_weight, lambda_full, lambda_init, q_groups, gr, eps)

    global _gda_post_kernel
    if _gda_post_kernel is None:
        _gda_post_kernel = _make_gda_post_kernel()

    B, q_heads, S, channels = merged.shape
    q_origin = q_groups * gr
    assert q_heads == q_origin + q_groups, f"q_heads={q_heads} expected {q_origin}+{q_groups}"

    rows = B * q_origin * S
    tg = min(256, max(32, ((channels + 31) // 32) * 32))
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale = _scalar_f32(1.0 - lambda_init)
    eps_arr = _scalar_f32(eps)

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
            ("TG", tg),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, q_origin, S, channels)],
        output_dtypes=[merged.dtype],
    )[0]
    return out
