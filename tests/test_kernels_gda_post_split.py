"""Validate gda_post_split (no upstream concat needed)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import gda_post_split, gda_post_split_reference


@pytest.mark.parametrize(
    "B,q_groups,gr,S,d",
    [
        (1, 8, 4, 1, 128),  # 12.7B decode
        (1, 8, 4, 32, 128),  # short prefill
        (2, 4, 4, 1, 128),  # batched
        (1, 2, 4, 1, 64),  # smaller d
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_gda_post_split_matches_reference(B, q_groups, gr, S, d, dtype):
    mx.random.seed(0)
    q_origin = q_groups * gr
    channels = 2 * d

    attn_o = mx.random.normal((B, q_origin, S, channels)).astype(dtype)
    attn_n = mx.random.normal((B, q_groups, S, channels)).astype(dtype)
    subln_w = mx.random.normal((channels,)).astype(dtype)
    lambda_full = mx.array([0.4321], dtype=mx.float32)
    lambda_init = 0.65
    eps = 1e-5

    ref = gda_post_split_reference(attn_o, attn_n, subln_w, lambda_full, lambda_init, gr, eps)
    got = gda_post_split(attn_o, attn_n, subln_w, lambda_full, lambda_init, gr, eps)

    rtol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    diff = float(np.max(np.abs(ref_np - got_np)))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"gda_post_split mismatch B={B} q_groups={q_groups} gr={gr} S={S} d={d} dtype={dtype}: "
        f"max diff={diff:.4e}"
    )
