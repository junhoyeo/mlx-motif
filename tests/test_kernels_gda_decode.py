"""Numerical validation for the fused flash-style GDA decode kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import gda_decode, gda_decode_reference


@pytest.mark.parametrize(
    "B,q_groups,gr,kv_seq,d",
    [
        (1, 2, 4, 4, 16),  # tiny smoke
        (1, 2, 4, 32, 16),  # moderate kv
        (1, 8, 4, 1, 128),  # 12.7B-shaped, single token of context
        (1, 8, 4, 64, 128),  # 12.7B with realistic kv
        (2, 8, 4, 17, 128),  # batched, odd kv
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_gda_decode_matches_reference(B, q_groups, gr, kv_seq, d, dtype):
    mx.random.seed(0)
    q_origin = q_groups * gr
    scale = 1.0 / (d**0.5)
    lambda_full = mx.array([0.4321], dtype=mx.float32)
    lambda_init = 0.65
    eps = 1e-5

    q1 = (mx.random.normal((B, q_origin, 1, d)) * 0.1).astype(dtype)
    q2 = (mx.random.normal((B, q_groups, 1, d)) * 0.1).astype(dtype)
    k1 = (mx.random.normal((B, q_groups, kv_seq, d)) * 0.1).astype(dtype)
    k2 = (mx.random.normal((B, q_groups, kv_seq, d)) * 0.1).astype(dtype)
    v1 = (mx.random.normal((B, q_groups, kv_seq, d)) * 0.5).astype(dtype)
    v2 = (mx.random.normal((B, q_groups, kv_seq, d)) * 0.5).astype(dtype)
    subln_w = mx.random.normal((2 * d,)).astype(dtype)

    ref = gda_decode_reference(
        q1, q2, k1, k2, v1, v2, subln_w, lambda_full, lambda_init, gr, scale, eps
    )
    got = gda_decode(q1, q2, k1, k2, v1, v2, subln_w, lambda_full, lambda_init, gr, scale, eps)

    rtol = {mx.float32: 5e-3, mx.bfloat16: 1e-1}[dtype]
    atol = {mx.float32: 5e-3, mx.bfloat16: 1e-1}[dtype]
    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    diff = np.max(np.abs(ref_np - got_np))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"gda_decode mismatch B={B} q_groups={q_groups} gr={gr} kv={kv_seq} d={d} dtype={dtype}: "
        f"max diff={diff:.4e}"
    )
