"""Regression tests for the converted-checkpoint config emission.

mlx-swift-lm's `BaseConfiguration` decodes every key under `quantization`
other than `group_size`/`bits` as a per-layer override dict. A scalar like
`preset: "uniform"` there makes the native Swift loader fail with a
`typeMismatch` ("The data couldn't be read because it isn't in the correct
format"). These tests lock the `quantization` block to scalars only.
"""

from mlx_motif.convert import _apply_quantization_config

# Values mlx-swift-lm's BaseConfiguration accepts directly under `quantization`.
_SCALAR_KEYS = {"group_size", "bits"}


def test_quantization_block_is_scalar_only_uniform():
    cfg = _apply_quantization_config({}, 64, 4, {"preset": "uniform"})
    assert cfg["quantization"] == {"group_size": 64, "bits": 4}
    # preset metadata is preserved, just relocated
    assert cfg["quantization_config"]["preset"] == "uniform"


def test_quantization_block_has_no_string_values_across_presets():
    presets = [
        {"preset": "uniform"},
        {"preset": "mixed", "q_bits": 6},
        {"preset": "mlp_lowbit", "mlp_bits": 3, "mlp_group_size": 32},
    ]
    for meta in presets:
        cfg = _apply_quantization_config({}, 64, 4, meta)
        q = cfg["quantization"]
        # only the two scalar keys, and both are ints (never strings/dicts)
        assert set(q) == _SCALAR_KEYS, f"unexpected keys for {meta}: {set(q)}"
        for v in q.values():
            assert isinstance(v, int), f"non-int in quantization for {meta}: {v!r}"


def test_quantization_config_carries_full_metadata():
    cfg = _apply_quantization_config({}, 32, 3, {"preset": "mlp_lowbit", "mlp_bits": 3})
    assert cfg["quantization_config"] == {
        "group_size": 32,
        "bits": 3,
        "preset": "mlp_lowbit",
        "mlp_bits": 3,
    }


def test_existing_config_keys_are_preserved():
    base = {"model_type": "motif", "hidden_size": 2048}
    cfg = _apply_quantization_config(dict(base), 64, 4, {"preset": "uniform"})
    assert cfg["model_type"] == "motif"
    assert cfg["hidden_size"] == 2048
