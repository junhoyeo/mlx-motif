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
    gda_post(merged, subln_weight, lambda_full, lambda_init, q_groups, gr, eps) -> mx.array
    gda_post_reference(...) -> mx.array
    gda_decode(q1, q2, k1, k2, v1, v2, subln_weight,
               lambda_full, lambda_init, gr, eps) -> mx.array
    gda_decode_reference(...) -> mx.array
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
# Flash-style fused GDA decode (S=1)
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
