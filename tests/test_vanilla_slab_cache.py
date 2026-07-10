"""Tests for the vanilla (2.6B, ungrouped) DiffAttn V-cache slab ordering.

The vanilla forward stores V in *slab* head order
``[va_0..va_{Hk-1}, vb_0..vb_{Hk-1}]`` instead of the projection's *paired*
order ``[v0_a, v0_b, v1_a, v1_b, ...]``. `MotifVanillaKVCache` marks that
ordering in ``meta_state`` so it cannot silently leak, and exposes an
``hf_ordered_values()`` accessor plus documented inverse-permutation helpers.

These tests cover:
  * inverse-permutation round-trip ``inverse(perm(x)) == x``,
  * the permutation matches the model's reshape/transpose reorder,
  * ``hf_ordered_values()`` restores HF paired order,
  * ``meta_state`` state round-trip works,
  * a mismatched / unmarked marker fails loudly,
  * ``make_cache`` hands the vanilla path a ``MotifVanillaKVCache`` and the
    grouped-non-4slot path a stock ``KVCache``.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlx_motif.cache import (
    MotifVanillaKVCache,
    vanilla_v_paired_to_slab_perm,
    vanilla_v_slab_to_paired_perm,
)
from mlx_motif.model import Model, ModelArgs


def _vanilla_args(**overrides) -> ModelArgs:
    base = dict(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        hidden_act="poly_norm",
    )
    base.update(overrides)
    return ModelArgs(**base)


def _grouped_args(**overrides) -> ModelArgs:
    base = dict(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=10,
        num_key_value_heads=4,
        num_noise_heads=2,
        k_ratio=1,
        vocab_size=128,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        hidden_act="poly_norm",
    )
    base.update(overrides)
    return ModelArgs(**base)


# --------------------------------------------------------------------------- #
# Permutation helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4, 8])
def test_perm_inverse_round_trips(n_kv_heads):
    """inverse(perm(x)) == x on the head axis for any V tensor."""
    p = vanilla_v_paired_to_slab_perm(n_kv_heads)
    inv = vanilla_v_slab_to_paired_perm(n_kv_heads)

    # Head-axis permutation composed with its inverse is the identity.
    composed = p[inv]
    assert composed.tolist() == list(range(2 * n_kv_heads))
    composed_other = inv[p]
    assert composed_other.tolist() == list(range(2 * n_kv_heads))

    # And on a concrete tensor: paired -> slab -> paired recovers the input.
    x = mx.random.normal((1, 2 * n_kv_heads, 3, 8))
    slab = x[:, p]
    back = slab[:, inv]
    assert mx.allclose(back, x).item()


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_perm_matches_model_reorder(n_kv_heads):
    """The paired->slab permutation equals the model's reshape/transpose.

    The vanilla forward reorders V with
    ``reshape(B, S, Hk, 2, d).transpose(0, 3, 2, 1, 4).reshape(B, 2*Hk, S, d)``.
    Applied to a paired-order head tensor that is exactly the head-axis
    permutation produced by ``vanilla_v_paired_to_slab_perm``.
    """
    B, S, d, Hk = 2, 3, 8, n_kv_heads
    # Paired-order head tensor: head block index == 2*h + s.
    paired = mx.random.normal((B, 2 * Hk, S, d))

    # Model-style reorder from paired head order into slab head order.
    slab_ref = paired.reshape(B, Hk, 2, S, d).transpose(0, 2, 1, 3, 4).reshape(B, 2 * Hk, S, d)
    slab_perm = paired[:, vanilla_v_paired_to_slab_perm(Hk)]
    assert mx.allclose(slab_ref, slab_perm).item()


# --------------------------------------------------------------------------- #
# hf_ordered_values accessor
# --------------------------------------------------------------------------- #


def test_hf_ordered_values_restores_paired_order():
    """`hf_ordered_values()` un-does the slab ordering back to HF paired order."""
    B, Hk, S, d = 1, 3, 5, 8
    cache = MotifVanillaKVCache()

    # Build a known paired-order V, reorder to slab (as the model does), store it.
    paired = mx.arange(B * 2 * Hk * S * d, dtype=mx.float32).reshape(B, 2 * Hk, S, d)
    slab = paired[:, vanilla_v_paired_to_slab_perm(Hk)]
    k = mx.zeros((B, 2 * Hk, S, d))
    cache.update_and_fetch(k, slab)

    restored = cache.hf_ordered_values()
    assert restored.shape == (B, 2 * Hk, S, d)
    assert mx.allclose(restored, paired).item()


def test_hf_ordered_values_empty_cache_is_none():
    assert MotifVanillaKVCache().hf_ordered_values() is None


# --------------------------------------------------------------------------- #
# meta_state marker: round-trip + loud failure
# --------------------------------------------------------------------------- #


def test_meta_state_round_trip():
    """state + meta_state round-trip reconstructs an equivalent cache."""
    B, H, S, d = 1, 4, 6, 8
    cache = MotifVanillaKVCache()
    k = mx.random.normal((B, H, S, d))
    v = mx.random.normal((B, H, S, d))
    cache.update_and_fetch(k, v)

    state = cache.state
    meta = cache.meta_state
    assert meta == ("slab",)

    restored = MotifVanillaKVCache.from_state(state, meta)
    assert restored.offset == cache.offset
    assert mx.allclose(restored.keys[..., : restored.offset, :], k).item()
    assert mx.allclose(restored.values[..., : restored.offset, :], v).item()


def test_from_state_rejects_mismatched_marker():
    """A state marked with a different V head order fails loudly."""
    B, H, S, d = 1, 4, 6, 8
    cache = MotifVanillaKVCache()
    cache.update_and_fetch(mx.random.normal((B, H, S, d)), mx.random.normal((B, H, S, d)))
    state = cache.state

    with pytest.raises(ValueError, match="V head-order marker mismatch"):
        MotifVanillaKVCache.from_state(state, ("paired",))
    with pytest.raises(ValueError, match="V head-order marker mismatch"):
        MotifVanillaKVCache.from_state(state, ("hf",))


def test_from_state_rejects_unmarked_state():
    """An unmarked (stock KVCache) state has unknown V ordering -> loud fail."""
    B, H, S, d = 1, 4, 6, 8
    cache = MotifVanillaKVCache()
    cache.update_and_fetch(mx.random.normal((B, H, S, d)), mx.random.normal((B, H, S, d)))
    state = cache.state

    with pytest.raises(ValueError, match="no V head-order marker"):
        MotifVanillaKVCache.from_state(state, ())
    with pytest.raises(ValueError, match="no V head-order marker"):
        MotifVanillaKVCache.from_state(state, "")


# --------------------------------------------------------------------------- #
# make_cache wiring
# --------------------------------------------------------------------------- #


def test_make_cache_vanilla_uses_marked_cache():
    model = Model(_vanilla_args())
    caches = model.make_cache()
    assert len(caches) == model.args.num_hidden_layers
    assert all(isinstance(c, MotifVanillaKVCache) for c in caches)


def test_make_cache_grouped_default_uses_4slot_cache(monkeypatch):
    # The 4-slot cache is the DEFAULT for grouped models (measured-fastest
    # decode configuration); the vanilla-marked subclass never appears here.
    monkeypatch.delenv("MLX_MOTIF_4SLOT_CACHE", raising=False)
    model = Model(_grouped_args())
    caches = model.make_cache()
    assert all(type(c).__name__ == "MotifGroupedKVCache" for c in caches)
    assert all(not isinstance(c, MotifVanillaKVCache) for c in caches)


def test_make_cache_grouped_opt_out_uses_stock_cache(monkeypatch):
    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "0")
    model = Model(_grouped_args())
    caches = model.make_cache()
    # Stock KVCache (not the vanilla-marked subclass) on the opted-out path.
    assert all(isinstance(c, KVCache) for c in caches)
    assert all(not isinstance(c, MotifVanillaKVCache) for c in caches)


def test_vanilla_forward_runs_with_marked_cache():
    """End-to-end: decode through the marked cache produces finite logits and a
    populated V slab (behaviour parity with the stock cache path)."""
    model = Model(_vanilla_args())
    caches = model.make_cache()
    tokens = mx.array([[1, 2, 3, 4]])
    logits = model(tokens, cache=caches)
    mx.eval(logits)
    assert logits.shape == (1, 4, model.args.vocab_size)
    assert bool(mx.all(mx.isfinite(logits)).item())
    # V cache populated and HF-view accessor works post-forward.
    hf_v = caches[0].hf_ordered_values()
    assert hf_v is not None
    assert hf_v.shape[2] == caches[0].offset
