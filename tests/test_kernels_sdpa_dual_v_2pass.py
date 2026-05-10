"""Numerical validation for the 2-pass dual-V SDPA decode kernel."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.kernels import sdpa_dual_v_2pass, sdpa_dual_v_reference


@pytest.mark.parametrize(
    "B,H_q,H_kv,KV,d",
    [
        (1, 8, 8, 64, 128),  # gqa=1, short (=BLOCKS*2)
        (1, 32, 8, 64, 128),  # 12.7B GQA gqa=4
        (1, 32, 8, 256, 128),  # medium KV
        (1, 32, 8, 1024, 128),  # long KV (where 2-pass should shine)
        (1, 32, 8, 4096, 128),  # very long KV
        (2, 16, 8, 128, 128),  # batched
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_sdpa_dual_v_2pass_matches_reference(B, H_q, H_kv, KV, d, dtype):
    mx.random.seed(0)
    scale = 1.0 / (d**0.5)
    q = (mx.random.normal((B, H_q, 1, d)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, H_kv, KV, d)) * 0.1).astype(dtype)
    v1 = (mx.random.normal((B, H_kv, KV, d)) * 0.5).astype(dtype)
    v2 = (mx.random.normal((B, H_kv, KV, d)) * 0.5).astype(dtype)

    gqa = H_q // H_kv
    k_r = mx.repeat(k, gqa, axis=1) if gqa > 1 else k
    v1_r = mx.repeat(v1, gqa, axis=1) if gqa > 1 else v1
    v2_r = mx.repeat(v2, gqa, axis=1) if gqa > 1 else v2
    ref = sdpa_dual_v_reference(q, k_r, v1_r, v2_r, scale)

    got = sdpa_dual_v_2pass(q, k, v1, v2, scale)
    mx.eval(got)

    rtol = {mx.float32: 1e-3, mx.bfloat16: 5e-2}[dtype]
    atol = {mx.float32: 1e-3, mx.bfloat16: 5e-2}[dtype]
    ref_np = np.array(ref.astype(mx.float32))
    got_np = np.array(got.astype(mx.float32))
    diff = float(np.max(np.abs(ref_np - got_np)))
    assert np.allclose(ref_np, got_np, rtol=rtol, atol=atol), (
        f"sdpa_dual_v_2pass mismatch B={B} H_q={H_q} H_kv={H_kv} KV={KV} d={d} dtype={dtype}: "
        f"max diff={diff:.4e}"
    )
