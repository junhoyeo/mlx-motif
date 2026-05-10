"""Numerical validation for the shared-QK dual-V SDPA decode kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import sdpa_dual_v, sdpa_dual_v_reference


@pytest.mark.parametrize(
    "B,H,KV,d",
    [
        (1, 8, 1, 128),     # tiny
        (1, 8, 16, 128),    # short
        (1, 40, 64, 128),   # 12.7B-shape
        (1, 40, 256, 128),  # decoder bench shape
        (1, 8, 17, 128),    # odd KV
        (2, 16, 64, 64),    # smaller d
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sdpa_dual_v_matches_reference(B, H, KV, d, dtype):
    mx.random.seed(0)
    scale = 1.0 / (d ** 0.5)
    q  = (mx.random.normal((B, H, 1, d)) * 0.1).astype(dtype)
    k  = (mx.random.normal((B, H, KV, d)) * 0.1).astype(dtype)
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
        f"sdpa_dual_v mismatch B={B} H={H} KV={KV} d={d} dtype={dtype}: "
        f"max diff={diff:.4e}"
    )
