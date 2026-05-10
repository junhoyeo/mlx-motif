"""Validate the standalone Metal dequant probe against `mx.dequantize`.

This is the bit-extract layer that `sdpa_dual_v_q4` will inline. Locking it
down in isolation means a future correctness failure in the full kernel is
easier to triage: rerun this test, and if it still passes, the bug is in
the attention math, not the bit-twiddling.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_motif.kernels import _dequant_probe


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_dequant_probe_matches_mx_dequantize(bits, group_size, D, dtype):
    if D % group_size != 0:
        pytest.skip(f"D={D} not a multiple of group_size={group_size}")

    mx.random.seed(0)
    x = mx.random.normal((2, 5, 7, D)).astype(dtype)
    data, scales, biases = mx.quantize(x, group_size=group_size, bits=bits)

    probe_out = _dequant_probe(data, scales, biases, group_size=group_size, bits=bits)
    ref_out = mx.dequantize(data, scales, biases, group_size=group_size, bits=bits)

    assert probe_out.shape == ref_out.shape == x.shape, (probe_out.shape, ref_out.shape, x.shape)
    assert probe_out.dtype == ref_out.dtype == dtype
    # Probe and mx.dequantize should be bit-for-bit equivalent (same arithmetic
    # in the same precision).
    assert mx.allclose(probe_out, ref_out, atol=1e-5, rtol=1e-5).item(), (
        f"max diff = {float(mx.max(mx.abs(probe_out - ref_out))):.3e}"
    )
