"""
Smoke tests for the MLX Motif port.

These are *structural* tests: they construct the model from a config, push
random input through it, and assert the output shapes line up. They do **not**
verify numerical equivalence to the HF reference — that's done separately
in `tests/test_parity.py` once a real checkpoint is available.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_motif.model import Model, ModelArgs, MotifAttention, PolyNorm


def _vanilla_args(**overrides) -> ModelArgs:
    base = dict(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        head_dim=16,
        attn_rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        hidden_act="poly_norm",
    )
    base.update(overrides)
    return ModelArgs(**base)


def _grouped_args(**overrides) -> ModelArgs:
    """Tiny GDA config that mirrors the 12.7B head topology (40/16/8 -> 5/2/1)."""
    base = dict(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=10,   # = 5 * 2 noise heads
        num_key_value_heads=4,    # = 2 * (1 + k_ratio)  with k_ratio=1
        num_noise_heads=2,
        k_ratio=1,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        head_dim=16,
        attn_rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        hidden_act="poly_norm",
    )
    base.update(overrides)
    return ModelArgs(**base)


def test_polynorm_shape_invariance():
    p = PolyNorm()
    x = mx.random.normal((2, 3, 8))
    y = p(x)
    assert y.shape == x.shape


def test_vanilla_diff_attention_forward():
    args = _vanilla_args()
    assert not args.is_grouped
    layer = MotifAttention(args, layer_idx=1)
    x = mx.random.normal((1, 4, args.hidden_size))
    y = layer(x)
    assert y.shape == (1, 4, args.hidden_size)


def test_grouped_diff_attention_forward():
    args = _grouped_args()
    assert args.is_grouped
    layer = MotifAttention(args, layer_idx=1)
    x = mx.random.normal((1, 4, args.hidden_size))
    y = layer(x)
    assert y.shape == (1, 4, args.hidden_size)


@pytest.mark.parametrize("variant", ["vanilla", "grouped"])
def test_full_model_forward(variant):
    args = _vanilla_args() if variant == "vanilla" else _grouped_args()
    model = Model(args)
    inputs = mx.array([[1, 2, 3, 4]])
    logits = model(inputs)
    assert logits.shape == (1, 4, args.vocab_size)


def test_sanitize_drops_rotary_buffers():
    args = _vanilla_args()
    model = Model(args)
    raw = {
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((4,)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((args.hidden_size, args.hidden_size)),
    }
    cleaned = model.sanitize(raw)
    assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in cleaned
    assert "model.layers.0.self_attn.q_proj.weight" in cleaned
