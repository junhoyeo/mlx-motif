"""Numerical validation for fused Metal kernels."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import (
    polynorm, polynorm_reference,
    polynorm_mul, polynorm_mul_reference,
)


@pytest.mark.parametrize("shape", [(2, 4, 16), (1, 1, 64), (3, 7, 256), (1, 1, 4096)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_polynorm_matches_reference(shape, dtype):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(dtype)
    w = mx.array([0.4, 0.3, 0.3], dtype=dtype)
    b = mx.array([0.05], dtype=dtype)
    eps = 1e-6

    ref = polynorm_reference(x, w, b, eps)
    got = polynorm(x, w, b, eps)

    # Bf16/fp16 have looser tolerances than fp32 due to accumulation noise.
    rtol = {mx.float32: 1e-4, mx.float16: 5e-2, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.float16: 5e-2, mx.bfloat16: 5e-2}[dtype]
    assert np.allclose(np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32)),
                       rtol=rtol, atol=atol), (
        f"polynorm mismatch shape={shape} dtype={dtype}: "
        f"max diff={np.max(np.abs(np.array(ref.astype(mx.float32)) - np.array(got.astype(mx.float32)))):.4e}"
    )


@pytest.mark.parametrize("shape", [(2, 4, 16), (1, 1, 64), (3, 7, 256), (1, 1, 16384)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_polynorm_mul_matches_reference(shape, dtype):
    mx.random.seed(0)
    gate = mx.random.normal(shape).astype(dtype)
    up = mx.random.normal(shape).astype(dtype)
    w = mx.array([0.4, 0.3, 0.3], dtype=dtype)
    b = mx.array([0.05], dtype=dtype)
    eps = 1e-6

    ref = polynorm_mul_reference(gate, up, w, b, eps)
    got = polynorm_mul(gate, up, w, b, eps)

    rtol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-4, mx.bfloat16: 5e-2}[dtype]
    assert np.allclose(np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32)),
                       rtol=rtol, atol=atol), (
        f"polynorm_mul mismatch shape={shape} dtype={dtype}: "
        f"max diff={np.max(np.abs(np.array(ref.astype(mx.float32)) - np.array(got.astype(mx.float32)))):.4e}"
    )
