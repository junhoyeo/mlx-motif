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
    assert restored.offset == 4
    # update_and_fetch_4 returns FULL step-padded capacity buffers; the
    # exact-length live region is what .state exposes.
    assert all(x.shape[2] >= 4 for x in out)
    assert all(t.shape == (B, H, 4, D) for t in restored.state)


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
