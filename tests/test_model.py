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

from mlx_motif.model import (
    AttnPath,
    Model,
    ModelArgs,
    MotifAttention,
    PolyNorm,
    _resolve_attention_path,
)


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
        num_attention_heads=10,  # = 5 * 2 noise heads
        num_key_value_heads=4,  # = 2 * (1 + k_ratio)  with k_ratio=1
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_attention_heads": 0}, "num_attention_heads must be greater than zero"),
        ({"num_key_value_heads": 0}, "num_key_value_heads must be greater than zero"),
        ({"num_noise_heads": 0}, "num_noise_heads must be greater than zero"),
        ({"k_ratio": -1}, "k_ratio must be greater than or equal to 1"),
        (
            {"num_key_value_heads": 1, "k_ratio": 1},
            r"num_key_value_heads must be at least k_ratio \+ 1",
        ),
        ({"num_hidden_layers": 0}, "num_hidden_layers must be greater than zero"),
        ({"num_hidden_layers": -3}, "num_hidden_layers must be greater than zero"),
        ({"hidden_size": 0}, "hidden_size must be greater than zero"),
        ({"intermediate_size": 0}, "intermediate_size must be greater than zero"),
        ({"vocab_size": 0}, "vocab_size must be greater than zero"),
        ({"max_position_embeddings": 0}, "max_position_embeddings must be greater than zero"),
        (
            {"hidden_size": 1, "head_dim": None},
            "derived head_dim must be greater than zero",
        ),
    ],
)
def test_model_args_rejects_unsafe_structural_values(overrides, message):
    """Untrusted config values must fail before model shape arithmetic runs."""
    with pytest.raises(ValueError, match=message):
        _grouped_args(**overrides)


@pytest.mark.parametrize(
    ("factory", "overrides", "message"),
    [
        (_vanilla_args, {"num_attention_heads": 3}, "num_attention_heads must be even"),
        (_vanilla_args, {"num_key_value_heads": 3}, "num_key_value_heads must be even"),
        (
            _vanilla_args,
            {"num_attention_heads": 6, "num_key_value_heads": 4},
            "num_attention_heads must be divisible by num_key_value_heads",
        ),
        (
            _vanilla_args,
            {"head_dim": 8},
            "head_dim \\* num_attention_heads must equal hidden_size",
        ),
        (
            _grouped_args,
            {"num_attention_heads": 9},
            "num_attention_heads must be divisible by num_noise_heads",
        ),
        (
            _grouped_args,
            {"num_key_value_heads": 5},
            r"num_key_value_heads must be divisible by k_ratio \+ 1",
        ),
        (
            _grouped_args,
            {"hidden_size": 65, "head_dim": None},
            "hidden_size must be divisible by num_attention_heads",
        ),
        (
            _grouped_args,
            {"k_ratio": 3, "num_key_value_heads": 8},
            "grouped attention ratio must be divisible by k_ratio",
        ),
    ],
)
def test_model_args_rejects_incompatible_head_topology(factory, overrides, message):
    """Head counts that cannot be reshaped or repeated must fail actionably."""
    with pytest.raises(ValueError, match=message):
        factory(**overrides)


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


def _grouped_args_kr2(**overrides) -> ModelArgs:
    """Grouped config with k_ratio=2 (num_key_value_heads = k_noise * 3).

    Exercises the unsupported 4-slot combination: k1 carries k_groups*k_ratio
    heads while k2/v1/v2 carry k_groups heads.
    """
    base = dict(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=10,
        num_key_value_heads=6,  # = 2 noise heads * (1 + k_ratio=2)
        num_noise_heads=2,
        k_ratio=2,
        vocab_size=128,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        hidden_act="poly_norm",
    )
    base.update(overrides)
    return ModelArgs(**base)


def test_make_cache_rejects_4slot_with_k_ratio_gt_1(monkeypatch):
    """`make_cache` must fail fast when the 4-slot cache is requested for a
    k_ratio > 1 grouped model: every slot is allocated with k1's head count,
    which differs from k2/v1/v2 when k_ratio > 1. Without the guard the
    mismatch surfaces as an opaque shape error mid-forward at cache-write time.
    """
    model = Model(_grouped_args_kr2())

    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "1")
    with pytest.raises(ValueError, match="k_ratio"):
        model.make_cache()

    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "q4")
    with pytest.raises(ValueError, match="k_ratio"):
        model.make_cache()

    # With the 4-slot cache disabled the k_ratio > 1 model falls back to the
    # stock single-slot KVCache without error.
    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "0")
    caches = model.make_cache()
    assert len(caches) == model.args.num_hidden_layers
    assert type(caches[0]).__name__ == "KVCache"

    # DEFAULT-on (env unset) must NOT raise for k_ratio > 1: only an explicit
    # request fails loudly; the default quietly falls back to the stock cache
    # so unsupported configs still work out of the box.
    monkeypatch.delenv("MLX_MOTIF_4SLOT_CACHE", raising=False)
    caches = model.make_cache()
    assert type(caches[0]).__name__ == "KVCache"


def test_make_cache_allows_4slot_with_k_ratio_1(monkeypatch):
    """The default k_ratio == 1 grouped model builds the 4-slot cache both
    explicitly and by default (env unset)."""
    model = Model(_grouped_args())
    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "1")
    caches = model.make_cache()
    assert type(caches[0]).__name__ == "MotifGroupedKVCache"

    monkeypatch.delenv("MLX_MOTIF_4SLOT_CACHE", raising=False)
    caches = model.make_cache()
    assert type(caches[0]).__name__ == "MotifGroupedKVCache"


def test_make_cache_rejects_quantized_4slot_with_nondivisible_head_dim(monkeypatch):
    """The quantized 4-slot cache packs head_dim into fixed group_size=64
    quantization groups; head_dim=80 would silently truncate the packed
    scale/bias allocation and then fail opaquely inside mx.quantize on the
    first cache write. q4/q8 is always an explicit request, so `make_cache`
    must fail fast at construction instead of mid-forward.
    """
    model = Model(_grouped_args(head_dim=80))
    for env in ("q4", "q8"):
        monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", env)
        with pytest.raises(ValueError, match="head_dim"):
            model.make_cache()

    # The unquantized 4-slot cache has no group-size constraint: both the
    # explicit fp request and the default-on case still work for head_dim=80.
    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "1")
    assert type(model.make_cache()[0]).__name__ == "MotifGroupedKVCache"
    monkeypatch.delenv("MLX_MOTIF_4SLOT_CACHE", raising=False)
    assert type(model.make_cache()[0]).__name__ == "MotifGroupedKVCache"


def test_make_cache_allows_quantized_4slot_with_divisible_head_dim(monkeypatch):
    """head_dim divisible by the group size builds the quantized cache."""
    model = Model(_grouped_args(head_dim=64))
    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "q4")
    caches = model.make_cache()
    assert type(caches[0]).__name__ == "MotifGroupedQuantizedKVCache"
    assert caches[0].bits == 4

    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", "q8")
    caches = model.make_cache()
    assert type(caches[0]).__name__ == "MotifGroupedQuantizedKVCache"
    assert caches[0].bits == 8


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

    caches = [
        MotifGroupedQuantizedKVCache(group_size=64, bits=4) for _ in range(len(model.model.layers))
    ]
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


# --------------------------------------------------------------------------- #
# Unit tests for _resolve_attention_path
# --------------------------------------------------------------------------- #


def _stub_env(**kv):
    """Return a dict-like stub that acts as an env mapping for _resolve_attention_path."""
    return kv


def _fp16_4slot_cache():
    from mlx_motif.cache import MotifGroupedKVCache

    return MotifGroupedKVCache()


def _quant_4slot_cache(bits=4):
    from mlx_motif.cache import MotifGroupedQuantizedKVCache

    return MotifGroupedQuantizedKVCache(group_size=64, bits=bits)


class TestResolveAttentionPath:
    """Unit tests for _resolve_attention_path — one case per AttnPath value."""

    # QUANT_SDPA: quantized cache + default QUANT_SDPA flag.
    def test_quant_sdpa_with_quantized_cache(self):
        env = _stub_env(MLX_MOTIF_QUANT_SDPA="1")
        path = _resolve_attention_path(
            cache=_quant_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.QUANT_SDPA

    def test_quant_sdpa_default_flag_is_on(self):
        # MLX_MOTIF_QUANT_SDPA defaults to "1" when key absent from env.
        env = _stub_env()
        path = _resolve_attention_path(
            cache=_quant_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.QUANT_SDPA

    def test_quant_sdpa_disabled_falls_through_to_dual_v(self):
        # MLX_MOTIF_QUANT_SDPA=0 with quant cache -> DUAL_V (not QUANT_SDPA).
        env = _stub_env(MLX_MOTIF_QUANT_SDPA="0")
        path = _resolve_attention_path(
            cache=_quant_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.DUAL_V

    # DUAL_V: fp16 4-slot cache + default DUAL_V flag.
    def test_dual_v_with_fp16_4slot_cache(self):
        env = _stub_env(MLX_MOTIF_DUAL_V="1")
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.DUAL_V

    def test_dual_v_default_flag_is_on(self):
        env = _stub_env()
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.DUAL_V

    # FALLBACK: various decode-ineligible conditions.
    def test_fallback_on_prefill_multi_token(self):
        env = _stub_env()
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=4,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_fallback_when_kv_repeat_gt_1(self):
        env = _stub_env()
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=2,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_fallback_when_kr_gt_1(self):
        env = _stub_env()
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=2,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_fallback_when_fused_rope(self):
        env = _stub_env()
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=True,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_quant_sdpa_blocked_by_fused_rope(self):
        """QUANT_SDPA consumes post-RoPE q/k from the standard rope() call
        — fused-rope routes around that, so this kernel must fall through."""
        env = _stub_env(MLX_MOTIF_QUANT_SDPA="1")
        path = _resolve_attention_path(
            cache=_quant_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=True,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_dual_v_blocked_by_fused_rope(self):
        """DUAL_V — same reason as QUANT_SDPA above."""
        env = _stub_env()  # MLX_MOTIF_DUAL_V defaults to "1"
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=True,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_fallback_when_dual_v_disabled(self):
        env = _stub_env(MLX_MOTIF_DUAL_V="0")
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    def test_fallback_with_none_cache(self):
        # No cache at all (prefill without caching) — decode-eligible check
        # requires isinstance(cache, MotifGroupedQuantizedKVCache) to be False,
        # so we reach DUAL_V for single-token… but here S=4 forces FALLBACK.
        env = _stub_env()
        path = _resolve_attention_path(
            cache=None,
            S=4,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.FALLBACK

    # Env-flag edge cases: all falsy string variants for QUANT_SDPA.
    @pytest.mark.parametrize("val", ["0", "", "false", "False"])
    def test_quant_sdpa_falsy_strings(self, val):
        env = _stub_env(MLX_MOTIF_QUANT_SDPA=val)
        path = _resolve_attention_path(
            cache=_quant_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        # Falsy QUANT_SDPA with fp16-eligible conditions -> DUAL_V, not QUANT_SDPA.
        assert path is AttnPath.DUAL_V

    @pytest.mark.parametrize("val", ["0", "", "false", "False"])
    def test_dual_v_falsy_strings(self, val):
        env = _stub_env(MLX_MOTIF_DUAL_V=val)
        path = _resolve_attention_path(
            cache=_fp16_4slot_cache(),
            S=1,
            kv_repeat=1,
            kr=1,
            fused_rope=False,
            env=env,
        )
        assert path is AttnPath.FALLBACK
