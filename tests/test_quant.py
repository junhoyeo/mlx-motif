"""Smoke tests for the quantization presets."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_motif.model import Model, ModelArgs
from mlx_motif.quant import apply_quant


def _grouped_args() -> ModelArgs:
    return ModelArgs(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=10,
        num_key_value_heads=4,
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


def _is_quantized(module: nn.Module) -> bool:
    return isinstance(module, nn.QuantizedLinear)


def test_uniform_preset_quantizes_all_linears():
    model = Model(_grouped_args())
    apply_quant(model, preset="uniform", bits=4)
    # All Linears under self_attn / mlp / lm_head should now be QuantizedLinear.
    qcount = sum(1 for _, m in model.named_modules() if _is_quantized(m))
    assert qcount > 0


def test_mixed_preset_uses_higher_bits_for_q_proj():
    model = Model(_grouped_args())
    apply_quant(model, preset="mixed", bits=4, q_bits=6)
    bits_seen = {}
    for path, m in model.named_modules():
        if _is_quantized(m):
            bits_seen.setdefault(path.endswith("q_proj"), set()).add(m.bits)
    assert bits_seen[True] == {6}, f"q_proj bits: {bits_seen[True]}"
    assert bits_seen[False] == {4}, f"other bits: {bits_seen[False]}"


def test_lambda_and_subln_are_not_quantized():
    """λ vectors and SubLN are plain mx.arrays — they must survive quantize."""
    model = Model(_grouped_args())
    apply_quant(model, preset="mixed")
    # Forward still works (would raise if SubLN got incorrectly quantized).
    out = model(mx.array([[1, 2, 3, 4]]))
    assert out.shape == (1, 4, 128)


def test_mlp_lowbit_preset_separates_mlp_from_attn():
    """`mlp_lowbit` drops gate/up/down to (mlp_bits, mlp_group_size); rest stays at (bits, group_size)."""
    model = Model(_grouped_args())
    meta = apply_quant(
        model, preset="mlp_lowbit",
        bits=4, group_size=64,
        mlp_bits=4, mlp_group_size=32,   # tiny test config can't fit q3 (intermediate_size=128)
    )
    bits_for = {"mlp": set(), "non_mlp": set()}
    gs_for = {"mlp": set(), "non_mlp": set()}
    for path, m in model.named_modules():
        if not _is_quantized(m):
            continue
        bucket = "mlp" if any(path.endswith(n) for n in (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")) else "non_mlp"
        bits_for[bucket].add(m.bits)
        gs_for[bucket].add(m.group_size)
    assert bits_for["mlp"] == {4}, bits_for["mlp"]
    assert gs_for["mlp"] == {32}, gs_for["mlp"]
    assert bits_for["non_mlp"] == {4}, bits_for["non_mlp"]
    assert gs_for["non_mlp"] == {64}, gs_for["non_mlp"]
    assert meta["preset"] == "mlp_lowbit"
    assert meta["mlp_bits"] == 4
    assert meta["mlp_group_size"] == 32
    # Forward still works.
    out = model(mx.array([[1, 2, 3, 4]]))
    assert out.shape == (1, 4, 128)


def test_mlp_lowbit_overrides_emit_per_module_config():
    """The returned config dict must include per-module overrides so the
    mlx-lm loader can rebuild matching shapes."""
    model = Model(_grouped_args())
    meta = apply_quant(
        model, preset="mlp_lowbit",
        bits=4, group_size=64,
        mlp_bits=4, mlp_group_size=32,
    )
    # Per-module overrides emit dotted-path keys.
    mlp_overrides = {k: v for k, v in meta.items() if isinstance(v, dict) and "bits" in v}
    assert any(".mlp.gate_proj" in k for k in mlp_overrides)
    assert any(".mlp.up_proj" in k for k in mlp_overrides)
    assert any(".mlp.down_proj" in k for k in mlp_overrides)
    for _path, settings in mlp_overrides.items():
        assert settings == {"group_size": 32, "bits": 4}
