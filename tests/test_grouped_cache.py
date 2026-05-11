"""Smoke tests for the 4-slot grouped KV caches."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.cache import MotifGroupedKVCache, MotifGroupedQuantizedKVCache


def _rand(shape, dtype=mx.bfloat16):
    return (mx.random.normal(shape) * 0.1).astype(dtype)


def test_unquantized_cache_roundtrip():
    mx.random.seed(0)
    B, H, D = 1, 8, 128
    cache = MotifGroupedKVCache()
    # Step 1: 5-token prompt
    k1, k2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    v1, v2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    out = cache.update_and_fetch_4(k1, k2, v1, v2)
    assert all(t.shape == (B, H, 5, D) for t in out)
    assert cache.offset == 5

    # Step 2: 1 more token
    k1b, k2b = _rand((B, H, 1, D)), _rand((B, H, 1, D))
    v1b, v2b = _rand((B, H, 1, D)), _rand((B, H, 1, D))
    out = cache.update_and_fetch_4(k1b, k2b, v1b, v2b)
    assert all(t.shape == (B, H, 6, D) for t in out)

    # Tail of fetched k1 must equal the just-pushed k1b.
    np.testing.assert_allclose(
        np.array(out[0][..., -1, :].astype(mx.float32)),
        np.array(k1b[..., 0, :].astype(mx.float32)),
        rtol=1e-5,
    )


def test_unquantized_cache_grows_past_step_boundary():
    cache = MotifGroupedKVCache()
    cache.step = 4  # tiny step to force grow
    B, H, D = 1, 4, 32
    for _ in range(10):
        k1, k2 = _rand((B, H, 1, D)), _rand((B, H, 1, D))
        v1, v2 = _rand((B, H, 1, D)), _rand((B, H, 1, D))
        cache.update_and_fetch_4(k1, k2, v1, v2)
    assert cache.offset == 10


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_cache_roundtrip(bits):
    mx.random.seed(0)
    B, H, D = 1, 8, 128
    cache = MotifGroupedQuantizedKVCache(group_size=64, bits=bits)
    k1, k2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    v1, v2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    out = cache.update_and_fetch_4(k1, k2, v1, v2)
    assert all(t.shape == (B, H, 5, D) for t in out)
    assert cache.offset == 5

    # Quantization noise should be small at bits=8, larger at bits=4.
    tol = {4: 0.5, 8: 0.05}[bits]
    np.testing.assert_allclose(
        np.array(out[0].astype(mx.float32)),
        np.array(k1.astype(mx.float32)),
        atol=tol,
        rtol=tol,
    )


def test_quantized_cache_to_quantized_from_unquantized():
    mx.random.seed(0)
    B, H, D = 1, 8, 128
    plain = MotifGroupedKVCache()
    k1, k2 = _rand((B, H, 7, D)), _rand((B, H, 7, D))
    v1, v2 = _rand((B, H, 7, D)), _rand((B, H, 7, D))
    plain.update_and_fetch_4(k1, k2, v1, v2)

    q = plain.to_quantized(group_size=64, bits=8)
    assert q.offset == 7
    # Add one more step to confirm growth + fetch still produce sensible data.
    k1b, _ = _rand((B, H, 1, D)), _rand((B, H, 1, D))
    out = q.update_and_fetch_4(k1b, k1b, k1b, k1b)
    assert all(t.shape == (B, H, 8, D) for t in out)
