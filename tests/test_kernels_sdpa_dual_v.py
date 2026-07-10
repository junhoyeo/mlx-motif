"""Numerical validation for the shared-QK dual-V SDPA decode kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import sdpa_dual_v, sdpa_dual_v_reference


@pytest.mark.parametrize(
    "B,H,KV,d",
    [
        (1, 8, 1, 128),  # tiny
        (1, 8, 16, 128),  # short
        (1, 40, 64, 128),  # 12.7B-shape
        (1, 40, 256, 128),  # decoder bench shape
        (1, 8, 17, 128),  # odd KV
        (2, 16, 64, 64),  # smaller d
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sdpa_dual_v_matches_reference(B, H, KV, d, dtype):
    mx.random.seed(0)
    scale = 1.0 / (d**0.5)
    q = (mx.random.normal((B, H, 1, d)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, H, KV, d)) * 0.1).astype(dtype)
    v1 = (mx.random.normal((B, H, KV, d)) * 0.5).astype(dtype)
    v2 = (mx.random.normal((B, H, KV, d)) * 0.5).astype(dtype)

    ref = sdpa_dual_v_reference(q, k, v1, v2, scale)
    got = sdpa_dual_v(q, k, v1, v2, scale)
    mx.eval(got)

    rtol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    diff = float(np.max(np.abs(ref_np - got_np)))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"sdpa_dual_v mismatch B={B} H={H} KV={KV} d={d} dtype={dtype}: max diff={diff:.4e}"
    )


@pytest.mark.parametrize(
    "B,H_q,H_kv,KV,d",
    [
        (1, 32, 8, 64, 128),  # 12.7B GQA: gqa=4
        (1, 16, 8, 32, 128),  # gqa=2
        (1, 32, 4, 100, 128),  # gqa=8
        (2, 32, 8, 17, 128),  # batched
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sdpa_dual_v_gqa_matches_reference(B, H_q, H_kv, KV, d, dtype):
    """GQA broadcast: K/V have fewer heads than Q; kernel broadcasts internally."""
    mx.random.seed(0)
    scale = 1.0 / (d**0.5)
    q = (mx.random.normal((B, H_q, 1, d)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, H_kv, KV, d)) * 0.1).astype(dtype)
    v1 = (mx.random.normal((B, H_kv, KV, d)) * 0.5).astype(dtype)
    v2 = (mx.random.normal((B, H_kv, KV, d)) * 0.5).astype(dtype)

    # Reference: explicit repeat then standard dual-V
    gqa = H_q // H_kv
    k_r = mx.repeat(k, gqa, axis=1)
    v1_r = mx.repeat(v1, gqa, axis=1)
    v2_r = mx.repeat(v2, gqa, axis=1)
    ref = sdpa_dual_v_reference(q, k_r, v1_r, v2_r, scale)
    got = sdpa_dual_v(q, k, v1, v2, scale)
    mx.eval(got)

    rtol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    diff = float(np.max(np.abs(ref_np - got_np)))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"GQA mismatch B={B} H_q={H_q} H_kv={H_kv} KV={KV} d={d} dtype={dtype}: max diff={diff:.4e}"
    )


@pytest.mark.parametrize(
    "B,H_q,H_kv,cap,kv_len,d",
    [
        (1, 8, 8, 256, 1, 128),  # decode step 1 into a step-padded buffer
        (1, 8, 8, 256, 17, 128),  # odd live length
        (1, 32, 8, 256, 100, 128),  # 12.7B GQA origin call shape
        (1, 40, 8, 512, 385, 128),  # capacity > one step, unaligned length
        (2, 16, 8, 256, 200, 64),  # batched, smaller d
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sdpa_dual_v_runtime_kv_len_matches_sliced_reference(B, H_q, H_kv, cap, kv_len, d, dtype):
    """Full-capacity buffers + runtime kv_len must equal the reference on the
    exact-length live slice (padding rows past kv_len are never read).

    This is the 4-slot-cache decode contract: the cache hands the kernel its
    full step-padded row-contiguous buffers and kv_len = cache.offset.
    """
    mx.random.seed(7)
    scale = 1.0 / (d**0.5)
    q = (mx.random.normal((B, H_q, 1, d)) * 0.1).astype(dtype)
    # Poison the padding region with large values: if the kernel ever reads
    # past kv_len, the output diverges loudly.
    k = (mx.random.normal((B, H_kv, cap, d)) * 0.1).astype(dtype)
    v1 = (mx.random.normal((B, H_kv, cap, d)) * 0.5).astype(dtype)
    v2 = (mx.random.normal((B, H_kv, cap, d)) * 0.5).astype(dtype)
    poison = mx.ones((B, H_kv, cap - kv_len, d), dtype=dtype) * 1e4
    k = mx.concatenate([k[..., :kv_len, :], poison], axis=2)
    v1 = mx.concatenate([v1[..., :kv_len, :], poison], axis=2)
    v2 = mx.concatenate([v2[..., :kv_len, :], poison], axis=2)
    mx.eval(k, v1, v2)

    gqa = H_q // H_kv
    k_r = mx.repeat(k[..., :kv_len, :], gqa, axis=1)
    v1_r = mx.repeat(v1[..., :kv_len, :], gqa, axis=1)
    v2_r = mx.repeat(v2[..., :kv_len, :], gqa, axis=1)
    ref = sdpa_dual_v_reference(q, k_r, v1_r, v2_r, scale)
    got = sdpa_dual_v(q, k, v1, v2, scale, kv_len=kv_len)
    mx.eval(got)

    rtol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    diff = float(np.max(np.abs(ref_np - got_np)))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"kv_len mismatch B={B} H_q={H_q} H_kv={H_kv} cap={cap} kv_len={kv_len} "
        f"d={d} dtype={dtype}: max diff={diff:.4e}"
    )


def test_sdpa_dual_v_growing_kv_len_no_respecialization():
    """Growing kv_len over a fixed-capacity buffer must stay numerically
    correct at every length (the loop bound is a runtime input, so no new
    kernel specialization is created per length)."""
    mx.random.seed(11)
    B, H_q, H_kv, cap, d = 1, 32, 8, 256, 128
    scale = 1.0 / (d**0.5)
    q = (mx.random.normal((B, H_q, 1, d)) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((B, H_kv, cap, d)) * 0.1).astype(mx.bfloat16)
    v1 = (mx.random.normal((B, H_kv, cap, d)) * 0.5).astype(mx.bfloat16)
    v2 = (mx.random.normal((B, H_kv, cap, d)) * 0.5).astype(mx.bfloat16)

    gqa = H_q // H_kv
    for kv_len in (1, 2, 31, 32, 33, 100, 255, 256):
        k_r = mx.repeat(k[..., :kv_len, :], gqa, axis=1)
        v1_r = mx.repeat(v1[..., :kv_len, :], gqa, axis=1)
        v2_r = mx.repeat(v2[..., :kv_len, :], gqa, axis=1)
        ref = sdpa_dual_v_reference(q, k_r, v1_r, v2_r, scale)
        got = sdpa_dual_v(q, k, v1, v2, scale, kv_len=kv_len)
        ref_np = np.array(ref.astype(mx.float32))
        got_np = np.array(got.astype(mx.float32))
        diff = float(np.max(np.abs(ref_np - got_np)))
        assert np.allclose(ref_np, got_np, rtol=5e-2, atol=5e-2), (
            f"kv_len={kv_len}: max diff={diff:.4e}"
        )
