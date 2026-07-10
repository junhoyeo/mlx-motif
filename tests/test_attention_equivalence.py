"""
Old-vs-new logits equivalence for the per-slab SDPA fast-path rework.

The prefill FALLBACK (grouped) and the vanilla forward were rewritten from a
single stacked SDPA over repeat+concat-materialized K/V (q_dim=d, v_dim=2d —
MLX's slow generic SDPA path) to per-slab `mx.fast.scaled_dot_product_attention`
calls with query_head_dim == value_head_dim == d and native GQA broadcast.

These tests pin the rework to the OLD implementation, reproduced verbatim
below as reference functions that consume the same module weights:

* randomized configs, grouped AND vanilla (including n_rep>1 / kv_repeat>1
  GQA-broadcast variants);
* chunked prefill with chunk sizes > 1 and nonzero cache offsets;
* single-token decode steps through the reworked FALLBACK branch;
* integrated chunked-prefill-vs-full-forward consistency (causal mask and
  cache-offset handling through `create_attention_mask`).
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlx_motif.kernels import gda_post
from mlx_motif.model import Model, ModelArgs, MotifAttention, _repeat

# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


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
    """Tiny GDA config mirroring the 12.7B head topology (40/16/8 -> 10/4/2)."""
    base = dict(
        model_type="motif",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=10,  # = 5 * 2 noise heads -> gr=4
        num_key_value_heads=4,  # = 2 * (1 + k_ratio) with k_ratio=1
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


def _causal_mask(S: int, offset: int, dtype=mx.float32):
    """Additive causal mask for a chunk of S queries at `offset` into the KV."""
    if S == 1:
        return None
    rinds = mx.arange(offset + S)
    linds = mx.arange(offset, offset + S)
    keep = linds[:, None] >= rinds[None]
    return mx.where(keep, mx.array(0.0, dtype), mx.array(-1e9, dtype))


# --------------------------------------------------------------------------- #
# OLD implementations (verbatim from pre-rework model.py), driven by the
# same layer weights so old-vs-new differences isolate the attention rewrite.
# --------------------------------------------------------------------------- #


def _old_forward_vanilla(layer: MotifAttention, x, mask, cache):
    B, S, _ = x.shape
    d = layer.head_dim
    H = layer.num_heads
    Hk = layer.num_kv_heads

    q = layer.q_proj(x).reshape(B, S, 2 * H, d).transpose(0, 2, 1, 3)
    k = layer.k_proj(x).reshape(B, S, 2 * Hk, d).transpose(0, 2, 1, 3)
    v = layer.v_proj(x).reshape(B, S, 2 * Hk, d).transpose(0, 2, 1, 3)

    offset = cache.offset if cache is not None else 0
    q = layer.rope(q, offset=offset)
    k = layer.rope(k, offset=offset)

    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    kv_seq = k.shape[2]

    v = v.reshape(B, Hk, 2, kv_seq, d).transpose(0, 1, 3, 2, 4).reshape(B, Hk, kv_seq, 2 * d)
    v = _repeat(v, layer.n_rep, axis=1)

    q_ = q.reshape(B, H, 2, S, d)
    k_ = k.reshape(B, Hk, 2, kv_seq, d)
    q1, q2 = q_[:, :, 0], q_[:, :, 1]
    k1, k2 = (
        _repeat(k_[:, :, 0], layer.n_rep, axis=1),
        _repeat(k_[:, :, 1], layer.n_rep, axis=1),
    )

    out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=layer.scale, mask=mask)
    out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=layer.scale, mask=mask)

    lam = layer._lambda_full(out1.dtype)
    out = out1 - lam * out2
    out = layer.subln(out)
    out = out * (1.0 - layer.lambda_init)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H * 2 * d)
    return layer.o_proj(out)


def _old_forward_grouped_fallback(layer: MotifAttention, x, mask, cache):
    """Pre-rework FALLBACK branch of `_forward_grouped` (plain KVCache path)."""
    B, S, _ = x.shape
    d = layer.head_dim

    q = layer.q_proj(x).reshape(B, S, layer.q_heads, d).transpose(0, 2, 1, 3)
    k = layer.k_proj(x).reshape(B, S, layer.num_kv_heads, d).transpose(0, 2, 1, 3)
    v_dim = 2 * layer.k_noise_heads
    v = layer.v_proj(x).reshape(B, S, v_dim, d).transpose(0, 2, 1, 3)

    offset = cache.offset if cache is not None else 0
    q = layer.rope(q, offset=offset)
    k = layer.rope(k, offset=offset)
    if cache is not None:
        k, v = cache.update_and_fetch(k, v)
    kv_seq = k.shape[2]

    gr = layer.grouped_ratio
    q_groups = layer.q_heads // (gr + 1)
    kr = layer.k_ratio
    k_groups = layer.num_kv_heads // (kr + 1)

    q_ = q.reshape(B, q_groups, gr + 1, S, d)
    q1 = q_[:, :, :gr, :, :].reshape(B, q_groups * gr, S, d)
    q2 = q_[:, :, gr:, :, :].reshape(B, q_groups, S, d)
    k_ = k.reshape(B, k_groups, kr + 1, kv_seq, d)
    k1 = k_[:, :, :kr, :, :].reshape(B, k_groups * kr, kv_seq, d)
    k2 = k_[:, :, kr:, :, :].reshape(B, k_groups, kv_seq, d)
    v_ = v.reshape(B, layer.k_noise_heads, 2, kv_seq, d)
    v1 = v_[:, :, 0, :, :]
    v2 = v_[:, :, 1, :, :]

    q_f = mx.concatenate([q1, q2], axis=1)
    if layer.kv_repeat > 1:
        k1 = _repeat(k1, layer.kv_repeat, axis=1)
        k2 = _repeat(k2, layer.kv_repeat, axis=1)
        v1 = _repeat(v1, layer.kv_repeat, axis=1)
        v2 = _repeat(v2, layer.kv_repeat, axis=1)

    lam = layer._lambda_full(mx.float32).reshape(1)

    if kr == 1:
        k_f = mx.concatenate([_repeat(k1, gr, axis=1), k2], axis=1)
    else:
        k_f = mx.concatenate([k1, k2], axis=1)
    v1_f = mx.concatenate([_repeat(v1, gr, axis=1), v1], axis=1)
    v2_f = mx.concatenate([_repeat(v2, gr, axis=1), v2], axis=1)
    v_cat = mx.concatenate([v1_f, v2_f], axis=-1)
    attn_cat = mx.fast.scaled_dot_product_attention(q_f, k_f, v_cat, scale=layer.scale, mask=mask)
    out = gda_post(
        attn_cat,
        layer.subln.weight,
        lam,
        layer.lambda_init,
        q_groups,
        gr,
        eps=layer.args.attn_rms_norm_eps,
    )

    out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
    return layer.o_proj(out)


# --------------------------------------------------------------------------- #
# Layer-level old-vs-new equivalence, chunked prefill + decode
# --------------------------------------------------------------------------- #

# Chunk schedule: multi-token prefill chunks with nonzero offsets, then decode.
_CHUNKS = [5, 4, 3, 1, 1]
_ATOL = 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        # n_rep=1 (matches the shipped 2.6B: q heads == kv heads)
        dict(num_attention_heads=4, num_key_value_heads=4),
        # n_rep=2 GQA broadcast variant
        dict(num_attention_heads=4, num_key_value_heads=2),
        # larger randomized-ish shape
        dict(hidden_size=128, num_attention_heads=8, num_key_value_heads=8, head_dim=16),
    ],
)
def test_vanilla_old_new_equivalence_chunked(kwargs):
    mx.random.seed(7)
    args = _vanilla_args(**kwargs)
    layer = MotifAttention(args, layer_idx=1)
    # Nonzero lambda parameters so the differential term is exercised.
    layer.lambda_q1 = mx.random.normal((layer.head_dim,)) * 0.1
    layer.lambda_k1 = mx.random.normal((layer.head_dim,)) * 0.1
    layer.lambda_q2 = mx.random.normal((layer.head_dim,)) * 0.1
    layer.lambda_k2 = mx.random.normal((layer.head_dim,)) * 0.1

    old_cache, new_cache = KVCache(), KVCache()
    offset = 0
    for S in _CHUNKS:
        x = mx.random.normal((1, S, args.hidden_size))
        mask = _causal_mask(S, offset)
        y_old = _old_forward_vanilla(layer, x, mask, old_cache)
        y_new = layer(x, mask=mask, cache=new_cache)
        md = float(mx.max(mx.abs(y_old - y_new)))
        assert md < _ATOL, f"chunk S={S} offset={offset}: max diff {md:.2e}"
        offset += S


@pytest.mark.parametrize(
    "kwargs",
    [
        # 12.7B-like topology: gr=4, kv_repeat=1, kr=1
        dict(),
        # kv_repeat=2 GQA-broadcast variant: k_noise_heads=1, num_noise=2
        dict(num_key_value_heads=2),
        # more noise groups, head_dim=32
        dict(
            hidden_size=128,
            num_attention_heads=12,  # 4 noise heads -> gr=2
            num_key_value_heads=8,
            num_noise_heads=4,
            head_dim=32,
        ),
    ],
)
def test_grouped_fallback_old_new_equivalence_chunked(kwargs, monkeypatch):
    # Force FALLBACK on the S=1 decode chunks too, so the reworked branch is
    # exercised at every offset (DUAL_V would otherwise take over at S=1).
    monkeypatch.setenv("MLX_MOTIF_DUAL_V", "0")
    monkeypatch.setenv("MLX_MOTIF_QUANT_SDPA", "0")

    mx.random.seed(11)
    args = _grouped_args(**kwargs)
    layer = MotifAttention(args, layer_idx=1)
    layer.lambda_q1 = mx.random.normal((layer.head_dim,)) * 0.1
    layer.lambda_k1 = mx.random.normal((layer.head_dim,)) * 0.1
    layer.lambda_q2 = mx.random.normal((layer.head_dim,)) * 0.1
    layer.lambda_k2 = mx.random.normal((layer.head_dim,)) * 0.1

    old_cache, new_cache = KVCache(), KVCache()
    offset = 0
    for S in _CHUNKS:
        x = mx.random.normal((1, S, args.hidden_size))
        mask = _causal_mask(S, offset)
        y_old = _old_forward_grouped_fallback(layer, x, mask, old_cache)
        y_new = layer(x, mask=mask, cache=new_cache)
        md = float(mx.max(mx.abs(y_old - y_new)))
        assert md < _ATOL, f"chunk S={S} offset={offset}: max diff {md:.2e}"
        offset += S


# --------------------------------------------------------------------------- #
# Integrated model-level consistency: chunked prefill (real mask pipeline,
# nonzero cache offsets) must match the full-sequence forward.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ["vanilla", "grouped"])
@pytest.mark.parametrize("four_slot", ["0", "1"])
def test_model_chunked_prefill_matches_full_forward(variant, four_slot, monkeypatch):
    if variant == "vanilla" and four_slot == "1":
        pytest.skip("4-slot cache is grouped-only")
    monkeypatch.setenv("MLX_MOTIF_4SLOT_CACHE", four_slot)

    mx.random.seed(3)
    args = _vanilla_args() if variant == "vanilla" else _grouped_args()
    model = Model(args)

    tokens = mx.array([[3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46, 26]])
    full = model(tokens)  # mask auto-created ("causal")

    caches = model.make_cache()
    outs = []
    pos = 0
    for S in [5, 4, 3]:
        chunk = tokens[:, pos : pos + S]
        outs.append(model(chunk, cache=caches))
        pos += S
    chunked = mx.concatenate(outs, axis=1)

    md = float(mx.max(mx.abs(full - chunked)))
    assert md < 5e-4, f"chunked-vs-full logits diverge: max diff {md:.2e}"
