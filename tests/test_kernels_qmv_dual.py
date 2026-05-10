"""Correctness tests for `qmv_dual_q4` (fused gate+up q4 GEMV).

Compared against `qmv_dual_q4_reference`, which is two `mx.quantized_matmul`
calls. The fused kernel does the same arithmetic in the same precision —
the only diff is the order of operations, so the tolerance is tight.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_motif.kernels import qmv_dual_q4, qmv_dual_q4_reference


_SHAPES = [
    # (B, S, IN, OUT) — must satisfy IN % 512 == 0 and OUT % 8 == 0
    (1, 1, 4096, 16384),  # production MLP shape (Motif 12.7B gate/up)
    (1, 1, 4096,    16),  # tiny OUT
    (1, 1,  512,  4096),  # tiny IN
    (1, 1, 2048,  8192),  # half-size MLP
    (1, 4, 4096, 16384),  # multi-token (prefill)
    (2, 1, 4096, 16384),  # batch
]


@pytest.mark.parametrize("B,S,IN,OUT", _SHAPES)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_qmv_dual_matches_reference(B, S, IN, OUT, dtype):
    mx.random.seed(0)
    x = mx.random.normal((B, S, IN)).astype(dtype)
    # Small init magnitude so accumulation noise stays bounded.
    gate_w = (mx.random.normal((OUT, IN)) * 0.02).astype(dtype)
    up_w = (mx.random.normal((OUT, IN)) * 0.02).astype(dtype)

    gate_q = mx.quantize(gate_w, group_size=64, bits=4)
    up_q = mx.quantize(up_w, group_size=64, bits=4)

    g_k, u_k = qmv_dual_q4(x, gate_q, up_q, group_size=64, bits=4)
    g_r, u_r = qmv_dual_q4_reference(x, gate_q, up_q, group_size=64, bits=4)

    assert g_k.shape == g_r.shape == (B, S, OUT)
    assert u_k.shape == u_r.shape == (B, S, OUT)
    assert g_k.dtype == g_r.dtype == dtype

    # Tolerance: kernel and reference both pull the same packed bits
    # through the same scale*nibble+bias arithmetic. Diff is fp32-vs-fp16
    # accumulator drift across thousands of FMAs.
    atol = 5e-2 if dtype == mx.float16 else (5e-2 if dtype == mx.bfloat16 else 1e-3)
    rtol = 5e-2 if dtype == mx.float16 else (5e-2 if dtype == mx.bfloat16 else 1e-3)
    g_diff = float(mx.max(mx.abs(g_k - g_r)))
    u_diff = float(mx.max(mx.abs(u_k - u_r)))
    assert mx.allclose(g_k, g_r, atol=atol, rtol=rtol).item(), (
        f"gate max diff = {g_diff:.3e} (tol={atol})"
    )
    assert mx.allclose(u_k, u_r, atol=atol, rtol=rtol).item(), (
        f"up max diff = {u_diff:.3e} (tol={atol})"
    )


def test_qmv_dual_independent_outputs():
    """Distinct gate and up weights ⇒ distinct outputs."""
    mx.random.seed(0)
    B, S, IN, OUT = 1, 1, 4096, 16384
    x = mx.random.normal((B, S, IN)).astype(mx.float16)
    gate_q = mx.quantize((mx.random.normal((OUT, IN)) * 0.02).astype(mx.float16),
                         group_size=64, bits=4)
    up_q = mx.quantize((mx.random.normal((OUT, IN)) * 0.02).astype(mx.float16),
                       group_size=64, bits=4)
    g, u = qmv_dual_q4(x, gate_q, up_q, group_size=64, bits=4)
    # Distinct weights, distinct outputs.
    assert not mx.allclose(g, u, atol=1e-3).item()


def test_qmv_dual_same_weights_same_outputs():
    """Same weights for gate and up ⇒ identical outputs."""
    mx.random.seed(0)
    B, S, IN, OUT = 1, 1, 4096, 16384
    x = mx.random.normal((B, S, IN)).astype(mx.float16)
    w_q = mx.quantize((mx.random.normal((OUT, IN)) * 0.02).astype(mx.float16),
                      group_size=64, bits=4)
    g, u = qmv_dual_q4(x, w_q, w_q, group_size=64, bits=4)
    assert mx.allclose(g, u).item()
