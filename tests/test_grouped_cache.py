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


def _np(a):
    """Compare-friendly numpy view (uint32 stays integer; floats -> fp32)."""
    if a.dtype == mx.uint32:
        return np.array(a)
    return np.array(a.astype(mx.float32))


def test_unquantized_cache_state_trims_to_offset_and_roundtrips():
    """(VALIDATION 4, fp16 sibling) — .state trims to the live [:offset] region
    and survives a save/restore round-trip through a fresh cache."""
    mx.random.seed(0)
    B, H, D = 1, 4, 128
    cache = MotifGroupedKVCache()
    # 3 tokens: capacity rounds up to step=256, so offset (3) < capacity.
    for _ in range(3):
        t = _rand((B, H, 1, D))
        cache.update_and_fetch_4(t, t, t, t)
    assert cache.offset == 3

    saved = cache.state
    for arr in saved:
        assert arr.shape[2] == cache.offset  # trimmed, not step-padded

    restored = MotifGroupedKVCache()
    restored.meta_state = cache.meta_state
    restored.state = saved
    assert restored.offset == 3
    for a, b in zip(saved, restored.state, strict=True):
        np.testing.assert_array_equal(_np(a), _np(b))

    # Restored cache keeps working: growth past the restored offset.
    tb = _rand((B, H, 1, D))
    out = restored.update_and_fetch_4(tb, tb, tb, tb)
    assert all(x.shape == (B, H, 4, D) for x in out)
    assert restored.offset == 4


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_cache_state_trims_to_offset_and_roundtrips(bits):
    """(VALIDATION 4) — the quantized .state getter serializes only the live
    [:offset] region of each (data, scales, biases) triple (previously it
    returned the full step-padded capacity) and round-trips exactly."""
    mx.random.seed(0)
    B, H, D = 1, 4, 128
    cache = MotifGroupedQuantizedKVCache(group_size=64, bits=bits)
    for _ in range(3):
        t = _rand((B, H, 1, D))
        cache.update_and_fetch_4(t, t, t, t)
    assert cache.offset == 3

    saved = cache.state
    # Each slot is a (data, scales, biases) triple; all trimmed to offset.
    for slot in saved:
        for arr in slot:
            assert arr.shape[2] == cache.offset

    restored = MotifGroupedQuantizedKVCache()
    restored.meta_state = cache.meta_state
    restored.state = saved
    assert restored.offset == 3
    assert (restored.group_size, restored.bits) == (64, bits)
    for orig_slot, new_slot in zip(saved, restored.state, strict=True):
        for a, b in zip(orig_slot, new_slot, strict=True):
            assert a.shape == b.shape
            np.testing.assert_array_equal(_np(a), _np(b))

    # Restored cache keeps working after a non-step-aligned restore.
    tb = _rand((B, H, 1, D))
    out = restored.update_and_fetch_4(tb, tb, tb, tb)
    assert all(x.shape == (B, H, 4, D) for x in out)
    assert restored.offset == 4


@pytest.mark.parametrize("offset", [0, 5])
def test_rope_before_split_equals_rope_after_split(offset):
    """(VALIDATION 2) — RoPE acts independently per head, so roping the full k
    once before the head-axis split into k1/k2 is bit-identical to roping each
    slice separately (the optimization applied in _forward_grouped)."""
    from mlx_lm.models.rope_utils import initialize_rope

    mx.random.seed(0)
    B, S, d = 1, 4, 128
    k_groups, kr = 2, 1  # num_kv_heads = k_groups * (kr + 1) = 4
    num_kv_heads = k_groups * (kr + 1)
    rope = initialize_rope(
        d, base=10000.0, traditional=False, scaling_config=None, max_position_embeddings=4096
    )

    k = mx.random.normal((B, num_kv_heads, S, d))

    # New path: rope the full k, then split on the head axis.
    k_roped = rope(k, offset=offset)
    kp = k_roped.reshape(B, k_groups, kr + 1, S, d)
    k1_after = kp[:, :, :kr, :, :].reshape(B, k_groups * kr, S, d)
    k2_after = kp[:, :, kr:, :, :].reshape(B, k_groups, S, d)

    # Old path: split first, then rope each slice.
    kp0 = k.reshape(B, k_groups, kr + 1, S, d)
    k1_new = kp0[:, :, :kr, :, :].reshape(B, k_groups * kr, S, d)
    k2_new = kp0[:, :, kr:, :, :].reshape(B, k_groups, S, d)
    k1_before = rope(k1_new, offset=offset)
    k2_before = rope(k2_new, offset=offset)

    np.testing.assert_array_equal(np.array(k1_after), np.array(k1_before))
    np.testing.assert_array_equal(np.array(k2_after), np.array(k2_before))
