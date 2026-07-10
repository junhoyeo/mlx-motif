"""Numerical validation for the fused post-SDPA GDA reduction kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import gda_post, gda_post_reference


@pytest.mark.parametrize(
    "B,q_groups,gr,S,d",
    [
        (1, 2, 4, 4, 16),  # tiny smoke
        (1, 8, 4, 1, 128),  # decode shape (12.7B layout)
        (2, 8, 4, 7, 128),  # batched + multi-step
        (1, 8, 4, 32, 128),  # short prefill (12.7B)
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_gda_post_matches_reference(B, q_groups, gr, S, d, dtype):
    mx.random.seed(0)
    q_origin = q_groups * gr
    q_heads = q_origin + q_groups
    channels = 2 * d

    merged = mx.random.normal((B, q_heads, S, channels)).astype(dtype)
    subln_w = mx.random.normal((channels,)).astype(dtype)
    lambda_full = mx.array([0.4321], dtype=mx.float32)
    lambda_init = 0.65
    eps = 1e-5

    ref = gda_post_reference(merged, subln_w, lambda_full, lambda_init, q_groups, gr, eps)
    got = gda_post(merged, subln_w, lambda_full, lambda_init, q_groups, gr, eps)

    rtol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]

    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"gda_post mismatch B={B} q_groups={q_groups} gr={gr} S={S} d={d} dtype={dtype}: "
        f"max diff={np.max(np.abs(ref_np - got_np)):.4e}"
    )


# Exercise the in-register differential cache (finding 24) across the two edge
# regimes not hit by the production head_dim=128 (channels=256, tg=256) shape:
#   d=192 -> channels=384 > tg=256 -> PER_THREAD=2 (multi-iteration register array)
#   d=20  -> channels=40,  tg=64  -> tail threads masked (i >= CHANNELS)
@pytest.mark.parametrize("d", [192, 20, 17])
def test_gda_post_register_cache_edge_channels(d):
    mx.random.seed(1)
    B, q_groups, gr, S = 1, 2, 4, 3
    q_origin = q_groups * gr
    q_heads = q_origin + q_groups
    channels = 2 * d

    merged = mx.random.normal((B, q_heads, S, channels)).astype(mx.float32)
    subln_w = mx.random.normal((channels,)).astype(mx.float32)
    lambda_full = mx.array([0.4321], dtype=mx.float32)
    lambda_init = 0.65
    eps = 1e-5

    ref = gda_post_reference(merged, subln_w, lambda_full, lambda_init, q_groups, gr, eps)
    got = gda_post(merged, subln_w, lambda_full, lambda_init, q_groups, gr, eps)
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)),
        rtol=1e-4, atol=1e-4,
    )
