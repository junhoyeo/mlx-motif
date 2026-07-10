"""Attention-related Metal kernels: dual-V SDPA + quantized-input variant."""

from __future__ import annotations

import mlx.core as mx

from ._common import _DISABLE

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


def sdpa_dual_v(q: mx.array, k: mx.array, v1: mx.array, v2: mx.array, scale: float) -> mx.array:
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
            "k_data",
            "k_scales",
            "k_biases",
            "v1_data",
            "v1_scales",
            "v1_biases",
            "v2_data",
            "v2_scales",
            "v2_biases",
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
    H_q = q.shape[1]
    H_kv = k.shape[1]
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
    # Packed-SDPA word contract: each lane loads exactly ONE packed uint32 per
    # tensor per step (attention.py: `k_pkd = k_data[row_u32]`) and unpacks
    # qk_per_thread = D/32 channels from it. That is only valid when all of a
    # lane's channels live in that single word, i.e. qk_per_thread <= EL_PER_INT
    # (equivalently D/32 <= 32/bits). For e.g. D=256/bits=8 the shifts reach 56
    # bits (UB on uint32 in Metal) and half the channels are never read, so the
    # kernel would return silent garbage — fail loudly instead. See the design
    # note at attention.py:246-249. Ships only with D=128, which satisfies this.
    assert D // 32 <= 32 // bits, (
        f"sdpa_dual_v_q4 requires D/32 <= 32/bits (packed word contract "
        f"qk_per_thread <= EL_PER_INT); got D={D}, bits={bits}"
    )
    gqa_factor = H_q // H_kv

    rows = B * H_q
    tg = 32 * 32
    grid = (rows * tg, 1, 1)
    threadgroup = (tg, 1, 1)

    scale_arr = mx.array([scale], dtype=mx.float32)
    out = _sdpa_dual_v_q4_kernel(
        inputs=[
            q,
            k_data,
            k_scales,
            k_biases,
            v1_data,
            v1_scales,
            v1_biases,
            v2_data,
            v2_scales,
            v2_biases,
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
