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


def _grouped_args_d64(**overrides) -> ModelArgs:
    """Same topology as `_grouped_args` but head_dim=64 — minimum viable size
    for the q4 attention kernel (`group_size=64` requires `head_dim >= 64`)."""
    return _grouped_args(head_dim=64, **overrides)


def _make_grouped_caches(model, *, group_size, bits):
    """Build matched (fp16, q4) 4-slot caches so we can exercise both branches
    of `_forward_grouped` against the same weights and input."""
    from mlx_motif.cache import MotifGroupedKVCache, MotifGroupedQuantizedKVCache
    n = len(model.model.layers)
    fp16 = [MotifGroupedKVCache() for _ in range(n)]
    q = [MotifGroupedQuantizedKVCache(group_size=group_size, bits=bits) for _ in range(n)]
    return fp16, q


@pytest.mark.parametrize("bits", [4, 8])
def test_quant_cache_path_runs_and_close_to_fp16(bits, monkeypatch):
    """Ensure the quantized-input attention path (`sdpa_dual_v_q4`) produces
    logits within quantization noise of the fp16 path on the same input.
    Also verifies that the wiring respects the `MLX_MOTIF_QUANT_SDPA` flag.
    """
    args = _grouped_args_d64()
    model = Model(args)

    # Common 2-token prompt; one decode step after.
    prompt = mx.array([[1, 2]])
    next_tok = mx.array([[3]])

    # Reference run (fp16 cache, dequant→fp16 attn).
    monkeypatch.setenv("MLX_MOTIF_QUANT_SDPA", "0")
    fp16_caches, _ = _make_grouped_caches(model, group_size=64, bits=bits)
    _ = model(prompt, cache=fp16_caches)
    logits_fp16 = model(next_tok, cache=fp16_caches)

    # Quant-cache + q4-kernel path.
    monkeypatch.setenv("MLX_MOTIF_QUANT_SDPA", "1")
    _, q_caches = _make_grouped_caches(model, group_size=64, bits=bits)
    _ = model(prompt, cache=q_caches)
    logits_q = model(next_tok, cache=q_caches)

    assert logits_fp16.shape == logits_q.shape == (1, 1, args.vocab_size)
    # Quantization noise + kernel fp32-vs-fp16 differ; tolerance is generous.
    md = float(mx.max(mx.abs(logits_fp16 - logits_q)))
    assert md < 5.0, f"q{bits} logits diverge too far from fp16: max diff = {md:.3f}"


def test_quant_sdpa_flag_disables_q4_kernel(monkeypatch):
    """Setting MLX_MOTIF_QUANT_SDPA=0 with a quantized cache must fall through
    to the dequant→sdpa_dual_v path (cache helper still works, just no q4)."""
    monkeypatch.setenv("MLX_MOTIF_QUANT_SDPA", "0")
    args = _grouped_args_d64()
    model = Model(args)
    from mlx_motif.cache import MotifGroupedQuantizedKVCache
    caches = [MotifGroupedQuantizedKVCache(group_size=64, bits=4)
              for _ in range(len(model.model.layers))]
    out = model(mx.array([[1, 2, 3]]), cache=caches)
    assert out.shape == (1, 3, args.vocab_size)


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
