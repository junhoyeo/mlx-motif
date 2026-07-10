"""Smoke tests for the 4-slot grouped KV caches."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_motif.cache import MotifGroupedKVCache, MotifGroupedQuantizedKVCache


def _rand(shape, dtype=mx.bfloat16):
    return (mx.random.normal(shape) * 0.1).astype(dtype)


def test_make_mask_matches_mlx_lm_contract():
    """`make_mask` must accept the mlx-lm call signature
    `cache.make_mask(N, return_array=..., window_size=...)` and forward the
    cache offset. Regression guard: the previous import pulled the base-module
    `create_attention_mask` (no `offset` kwarg), so any invocation raised
    TypeError. The bug was dormant only because the model passes the whole
    cache list to `create_attention_mask`, never routing through `make_mask`.
    """
    cache = MotifGroupedKVCache()
    cache.offset = 3

    # N == 1 short-circuits to no mask.
    assert cache.make_mask(1, return_array=False, window_size=None) is None

    # N > 1 without return_array is the string "causal" fast path.
    assert cache.make_mask(4, return_array=False, window_size=None) == "causal"

    # return_array builds an explicit (N, offset + N) causal mask array.
    mask = cache.make_mask(4, return_array=True, window_size=None)
    assert mask is not None
    assert mask.shape == (4, cache.offset + 4)

    # Quantized variant shares the same base implementation.
    qcache = MotifGroupedQuantizedKVCache(group_size=64, bits=8)
    qcache.offset = 2
    assert qcache.make_mask(1, return_array=False, window_size=None) is None
    assert qcache.make_mask(5, return_array=False, window_size=None) == "causal"


@pytest.mark.parametrize(
    "cache_factory",
    [
        MotifGroupedKVCache,
        lambda: MotifGroupedQuantizedKVCache(group_size=64, bits=8),
    ],
    ids=["fp16", "quantized"],
)
def test_mismatched_slot_heads_fail_fast(cache_factory):
    """The 4-slot caches allocate every slot with k1's head count, so a
    k_ratio > 1 write (k1 has more heads than k2/v1/v2) must fail loudly at
    write time rather than raising an opaque shape error or silently
    corrupting. Regression guard for the k_ratio > 1 latent bug.
    """
    cache = cache_factory()
    B, D = 1, 128
    k1 = _rand((B, 4, 1, D))  # k_groups * k_ratio heads (k_ratio = 2)
    k2 = _rand((B, 2, 1, D))  # k_groups heads
    v1 = _rand((B, 2, 1, D))
    v2 = _rand((B, 2, 1, D))
    with pytest.raises(ValueError, match="share a head count"):
        cache.update_and_fetch_4(k1, k2, v1, v2)


def test_unquantized_cache_roundtrip():
    mx.random.seed(0)
    B, H, D = 1, 8, 128
    cache = MotifGroupedKVCache()
    # Step 1: 5-token prompt.
    # Contract: update_and_fetch_4 returns the FULL step-padded capacity
    # buffers (row-contiguous, no ensure_row_contiguous copy downstream);
    # the live region is [..., :cache.offset, :].
    k1, k2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    v1, v2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    out = cache.update_and_fetch_4(k1, k2, v1, v2)
    assert all(t.shape == (B, H, cache.step, D) for t in out)
    assert cache.offset == 5
    # The exact-length live region is exposed via `state`.
    assert all(t.shape == (B, H, 5, D) for t in cache.state)

    # Step 2: 1 more token
    k1b, k2b = _rand((B, H, 1, D)), _rand((B, H, 1, D))
    v1b, v2b = _rand((B, H, 1, D)), _rand((B, H, 1, D))
    out = cache.update_and_fetch_4(k1b, k2b, v1b, v2b)
    assert cache.offset == 6
    assert all(t.shape == (B, H, cache.step, D) for t in out)

    # Last live row of fetched k1 must equal the just-pushed k1b.
    np.testing.assert_allclose(
        np.array(out[0][..., cache.offset - 1, :].astype(mx.float32)),
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


def _quantize_per_slot_reference(cache_cls, group_size, bits, steps):
    """Reference cache whose write path quantizes each slot separately.

    Reproduces the pre-batching `_update_4` structure (4 separate mx.quantize
    calls) so we can assert the batched implementation is bit-identical.
    """
    cache = cache_cls(group_size=group_size, bits=bits)

    def _reference_update(k1, k2, v1, v2):
        B, H, S, D = k1.shape
        prev = cache.offset
        cache._grow(B, H, S, D, k1.dtype)
        for fresh, slot in [
            (k1, cache.k1),
            (k2, cache.k2),
            (v1, cache.v1),
            (v2, cache.v2),
        ]:
            q_data, q_scales, q_biases = mx.quantize(fresh, group_size=group_size, bits=bits)
            slot[0][..., prev : prev + S, :] = q_data
            slot[1][..., prev : prev + S, :] = q_scales
            slot[2][..., prev : prev + S, :] = q_biases
        cache.offset += S

    fetched = None
    for k1, k2, v1, v2 in steps:
        _reference_update(k1, k2, v1, v2)
        o = cache.offset
        fetched = tuple(
            (s[0][..., :o, :], s[1][..., :o, :], s[2][..., :o, :])
            for s in (cache.k1, cache.k2, cache.v1, cache.v2)
        )
    return fetched


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_batched_write_matches_per_slot(bits):
    """The batched _update_4 (1 quantize) must store bit-identical packed
    triples to the old per-slot path (4 quantize), and the quantized fetch
    must return identical values before/after the batching change."""
    mx.random.seed(1)
    B, H, D = 1, 8, 128
    group_size = 64

    # A 5-token prefill followed by three single-token decode steps.
    steps = [
        (_rand((B, H, 5, D)), _rand((B, H, 5, D)), _rand((B, H, 5, D)), _rand((B, H, 5, D))),
        (_rand((B, H, 1, D)), _rand((B, H, 1, D)), _rand((B, H, 1, D)), _rand((B, H, 1, D))),
        (_rand((B, H, 1, D)), _rand((B, H, 1, D)), _rand((B, H, 1, D)), _rand((B, H, 1, D))),
        (_rand((B, H, 1, D)), _rand((B, H, 1, D)), _rand((B, H, 1, D)), _rand((B, H, 1, D))),
    ]

    ref = _quantize_per_slot_reference(MotifGroupedQuantizedKVCache, group_size, bits, steps)

    batched = MotifGroupedQuantizedKVCache(group_size=group_size, bits=bits)
    live = None
    for k1, k2, v1, v2 in steps:
        live = batched.update_and_fetch_4_quantized(k1, k2, v1, v2)

    # Packed triples (data/scales/biases) must be bit-identical per slot.
    # update_and_fetch_4_quantized returns FULL step-padded capacity triples;
    # the reference holds exact-length live slices, so compare the live region.
    o = batched.offset
    for slot_ref, slot_live in zip(ref, live, strict=True):
        for comp_ref, comp_live in zip(slot_ref, slot_live, strict=True):
            live_region = comp_live[..., :o, :]
            assert comp_ref.shape == live_region.shape
            assert mx.array_equal(comp_ref, live_region)

    # And the dequantized fetch contract must also match exactly.
    deq = MotifGroupedQuantizedKVCache(group_size=group_size, bits=bits)
    deq_out = None
    for k1, k2, v1, v2 in steps:
        deq_out = deq.update_and_fetch_4(k1, k2, v1, v2)
    for slot_ref, deq_slot in zip(ref, deq_out, strict=True):
        expected = mx.dequantize(
            slot_ref[0], slot_ref[1], slot_ref[2], group_size=group_size, bits=bits
        )
        assert mx.array_equal(expected, deq_slot)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_fetch_returns_full_capacity_triples(bits):
    """update_and_fetch_4_quantized returns FULL step-padded capacity triples;
    the live region is [..., :offset, :] of each component."""
    mx.random.seed(0)
    B, H, D = 1, 8, 128
    cache = MotifGroupedQuantizedKVCache(group_size=64, bits=bits)
    k1, k2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    v1, v2 = _rand((B, H, 5, D)), _rand((B, H, 5, D))
    out = cache.update_and_fetch_4_quantized(k1, k2, v1, v2)
    assert cache.offset == 5
    for triple in out:
        assert len(triple) == 3
        for component in triple:
            assert component.shape[2] == cache.step

    # Live region of the returned k1 triple dequantizes back to ~k1.
    deq = mx.dequantize(
        out[0][0][..., :5, :],
        out[0][1][..., :5, :],
        out[0][2][..., :5, :],
        group_size=64,
        bits=bits,
    )
    tol = {4: 0.5, 8: 0.05}[bits]
    np.testing.assert_allclose(
        np.array(deq.astype(mx.float32)),
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
