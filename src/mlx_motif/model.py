"""
MLX implementation of the Motif family.

Two configurations are supported via a single attention module:

* **Motif-2.6B** — vanilla Differential Attention (Microsoft DiffTransformer
  variant, arXiv:2410.05258). `num_noise_heads` absent in config.
* **Motif-2-12.7B** — Grouped Differential Attention (GDA, arXiv:2510.06949).
  `num_noise_heads` present.

Both share PolyNorm activation (arXiv:2411.03884) and per-head SubLN over
`2 * head_dim` channels.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope


def _maybe_dequant_kv(k, v, cache):
    """Decode-time bridge for QuantizedKVCache.

    The differential pattern slices K and V along the head axis AFTER cache
    fetch (`k[:, :, 0]`, `k[:, :, 1]`, etc.). When the cache is quantized,
    `update_and_fetch` returns triples (data, scales, biases) which can't be
    sliced. We trade attention-time quantized-SDPA (already incompatible with
    our split anyway) for cache-time *memory* savings: K/V live in HBM at
    `kv_bits`, but get dequantized into registers per step before the split.
    Net win: 2–4× cache memory at long context, no decode-speed regression.
    """
    if hasattr(cache, "bits"):
        k = mx.dequantize(*k, group_size=cache.group_size, bits=cache.bits)
        v = mx.dequantize(*v, group_size=cache.group_size, bits=cache.bits)
    return k, v


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    head_dim: Optional[int] = None
    num_noise_heads: Optional[int] = None
    k_ratio: int = 1
    attn_rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    rope_scaling: Optional[dict] = None
    hidden_act: str = "poly_norm"
    use_bias: bool = False
    expanded: bool = False
    sliding_window: Optional[int] = None
    use_sliding_window: bool = False
    max_window_layers: Optional[int] = None
    fused_rope: bool = False
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    @property
    def is_grouped(self) -> bool:
        """True for GDA (12.7B), False for plain DiffAttention (2.6B)."""
        return self.num_noise_heads is not None


# --------------------------------------------------------------------------- #
# Activations / norms
# --------------------------------------------------------------------------- #


class PolyNorm(nn.Module):
    """
    Trainable polynomial activation from arXiv:2411.03884.

        y = w0 * norm(x^3) + w1 * norm(x^2) + w2 * norm(x) + b

    where `norm(z) = z / sqrt(mean(z^2) + eps)`. Uses a fused Metal kernel
    when available; falls back to plain MLX ops via `MLX_MOTIF_DISABLE_KERNELS=1`.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((3,)) / 3.0
        self.bias = mx.zeros((1,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        from mlx_motif.kernels import polynorm

        return polynorm(x, self.weight, self.bias, self.eps)


def get_activation(name: str) -> nn.Module:
    if name == "poly_norm":
        return PolyNorm()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported hidden_act: {name}")


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #


class MotifMLP(nn.Module):
    """Gated MLP with a poly-norm activation.

    Plain `down_proj(act(gate) * up)` form is the fastest end-to-end path
    on M-series GPUs. MLX's lazy graph optimizer fuses the elementwise
    chain across many decoder layers in ways that an `mx.compile` of a
    single MLP block actively *prevents* (-4% end-to-end despite +12%
    in the isolated MLP microbench).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        h, i = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=args.use_bias)
        self.up_proj = nn.Linear(h, i, bias=args.use_bias)
        self.down_proj = nn.Linear(i, h, bias=args.use_bias)
        self.act_fn = get_activation(args.hidden_act)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #


def _repeat(x: mx.array, n_rep: int, axis: int = 1) -> mx.array:
    """Expand-then-reshape repeat along `axis`. Equivalent to repeat_interleave."""
    if n_rep == 1:
        return x
    return mx.repeat(x, repeats=n_rep, axis=axis)


class MotifAttention(nn.Module):
    """
    Differential / Grouped-Differential Attention.

    Layout (12.7B):
        num_heads       = 40 (q heads, after factoring grouped_ratio)
        num_noise_heads = 8
        grouped_ratio   = (num_heads - num_noise_heads) // num_noise_heads = 4
        q_proj  : H -> q_heads      * head_dim    = 40 * 128
        k_proj  : H -> num_kv_heads * head_dim    = 16 * 128
        v_proj  : H -> 2 * k_noise  * head_dim    = 16 * 128   (noise_heads × 2 split)
        o_proj  : 2 * grouped_ratio * num_noise_heads * head_dim -> H

    Layout (2.6B, vanilla DiffAttn):
        head_dim        = hidden_size // num_attention_heads (orig 16, halved -> 8 effective heads)
        q_proj  : H -> 2 * num_heads     * head_dim
        k_proj  : H -> 2 * num_kv_heads  * head_dim
        v_proj  : H -> num_kv_heads      * 2 * head_dim
        o_proj  : H -> H
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.head_dim = args.head_dim or (args.hidden_size // args.num_attention_heads)
        self.scale = self.head_dim ** -0.5
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))

        if args.is_grouped:
            self._init_grouped(args)
        else:
            self._init_vanilla(args)

        # λ parameters — kept fp32 for numerical stability of exp/sub
        self.lambda_q1 = mx.zeros((self.head_dim,), dtype=mx.float32)
        self.lambda_k1 = mx.zeros((self.head_dim,), dtype=mx.float32)
        self.lambda_q2 = mx.zeros((self.head_dim,), dtype=mx.float32)
        self.lambda_k2 = mx.zeros((self.head_dim,), dtype=mx.float32)

        # SubLN over `2 * head_dim` (concatenated per-head output of two SDPAs)
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=args.attn_rms_norm_eps)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    # -- variant-specific initialisers -------------------------------------- #

    def _init_grouped(self, args: ModelArgs) -> None:
        self.num_noise_heads = args.num_noise_heads
        self.num_kv_heads = args.num_key_value_heads
        self.grouped_ratio = (args.num_attention_heads - self.num_noise_heads) // self.num_noise_heads
        self.q_heads = (self.grouped_ratio + 1) * self.num_noise_heads
        self.k_ratio = args.k_ratio
        self.k_noise_heads = self.num_kv_heads // (self.k_ratio + 1)
        self.kv_repeat = self.num_noise_heads // self.k_noise_heads

        h = args.hidden_size
        d = self.head_dim
        self.q_proj = nn.Linear(h, self.q_heads * d, bias=args.use_bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * d, bias=args.use_bias)
        self.v_proj = nn.Linear(h, 2 * self.k_noise_heads * d, bias=args.use_bias)
        self.o_proj = nn.Linear(
            2 * self.grouped_ratio * self.num_noise_heads * d, h, bias=args.use_bias
        )
        # Sentinel for fused QKV (set by fuse_qkv() at load time).
        self.qkv_proj = None
        self._q_dim = self.q_heads * d
        self._k_dim = self.num_kv_heads * d
        self._v_dim = 2 * self.k_noise_heads * d

    def _init_vanilla(self, args: ModelArgs) -> None:
        # 2.6B halves heads — see modeling_motif.py:352 in the 2.6B reference.
        h = args.hidden_size
        self.num_heads = args.num_attention_heads // 2
        self.num_kv_heads = args.num_key_value_heads // 2
        self.n_rep = self.num_heads // self.num_kv_heads
        d = self.head_dim
        self.q_proj = nn.Linear(h, h, bias=args.use_bias)
        self.k_proj = nn.Linear(h, h // self.n_rep, bias=args.use_bias)
        self.v_proj = nn.Linear(h, h // self.n_rep, bias=args.use_bias)
        self.o_proj = nn.Linear(h, h, bias=args.use_bias)

    # -- common helpers ----------------------------------------------------- #

    def _lambda_full(self, dtype: mx.Dtype) -> mx.array:
        l1 = mx.exp(mx.sum(self.lambda_q1 * self.lambda_k1, axis=-1).astype(mx.float32))
        l2 = mx.exp(mx.sum(self.lambda_q2 * self.lambda_k2, axis=-1).astype(mx.float32))
        return (l1 - l2 + self.lambda_init).astype(dtype)

    # -- forward dispatch --------------------------------------------------- #

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        if self.args.is_grouped:
            return self._forward_grouped(x, mask, cache)
        return self._forward_vanilla(x, mask, cache)

    # -- vanilla differential attention (2.6B) ----------------------------- #

    def _forward_vanilla(self, x, mask, cache):
        # Mathematically equivalent to the HF eager path: by linearity of @V,
        #     (softmax(QaKa^T) − λ·softmax(QbKb^T)) · V
        # equals
        #     SDPA(Qa, Ka, V) − λ · SDPA(Qb, Kb, V).
        # V is stored in the cache at the same head granularity as K (so
        # `n_kv_heads = num_key_value_heads`), then reshaped into 2·d wide
        # heads for the attention matmul.
        B, S, _ = x.shape
        d = self.head_dim
        H = self.num_heads      # halved (effective DiffAttn heads)
        Hk = self.num_kv_heads  # halved (effective DiffAttn KV heads)

        q = self.q_proj(x).reshape(B, S, 2 * H, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, 2 * Hk, d).transpose(0, 2, 1, 3)
        # Store V at d-per-head so the standard KVCache (which assumes K/V have
        # matching shapes) accepts it. Restore 2d-per-head after fetch.
        v = self.v_proj(x).reshape(B, S, 2 * Hk, d).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            k, v = _maybe_dequant_kv(k, v, cache)
        kv_seq = k.shape[2]

        # Recover (B, Hk, S, 2d) — see `_init_vanilla` shape derivation.
        v = v.reshape(B, Hk, 2, kv_seq, d).transpose(0, 1, 3, 2, 4).reshape(B, Hk, kv_seq, 2 * d)
        v = _repeat(v, self.n_rep, axis=1)

        # Stripe split for differential attention: heads pair adjacent in flat layout.
        q_ = q.reshape(B, H, 2, S, d)
        k_ = k.reshape(B, Hk, 2, kv_seq, d)
        q1, q2 = q_[:, :, 0], q_[:, :, 1]
        k1, k2 = _repeat(k_[:, :, 0], self.n_rep, axis=1), _repeat(k_[:, :, 1], self.n_rep, axis=1)

        out1 = mx.fast.scaled_dot_product_attention(q1, k1, v, scale=self.scale, mask=mask)
        out2 = mx.fast.scaled_dot_product_attention(q2, k2, v, scale=self.scale, mask=mask)

        lam = self._lambda_full(out1.dtype)
        out = out1 - lam * out2
        out = self.subln(out)
        out = out * (1.0 - self.lambda_init)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * 2 * d)
        return self.o_proj(out)

    # -- grouped differential attention (12.7B) ---------------------------- #

    def _forward_grouped(self, x, mask, cache):
        B, S, _ = x.shape
        d = self.head_dim

        # If qkv_proj has been fused at load time, use a single matmul + 3-way
        # split (saves ~10% on the q/k/v projection cost — hits MLX's
        # quantized matmul sweet spot at output_dim=q+k+v=9216 vs 3 separate
        # 5120/2048/2048 calls; see commit history for microbench).
        if self.qkv_proj is not None:
            qkv = self.qkv_proj(x)
            q_flat = qkv[..., : self._q_dim]
            k_flat = qkv[..., self._q_dim : self._q_dim + self._k_dim]
            v_flat = qkv[..., self._q_dim + self._k_dim :]
        else:
            q_flat = self.q_proj(x)
            k_flat = self.k_proj(x)
            v_flat = self.v_proj(x)

        q = q_flat.reshape(B, S, self.q_heads, d).transpose(0, 2, 1, 3)
        k = k_flat.reshape(B, S, self.num_kv_heads, d).transpose(0, 2, 1, 3)
        v_dim = 2 * self.k_noise_heads
        v = v_flat.reshape(B, S, v_dim, d).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            k, v = _maybe_dequant_kv(k, v, cache)

        kv_seq = k.shape[2]

        # Split q on the head axis: stripe layout [gr origin | 1 noise] per group.
        gr = self.grouped_ratio
        q_groups = self.q_heads // (gr + 1)
        q_ = q.reshape(B, q_groups, gr + 1, S, d)
        q1 = q_[:, :, :gr, :, :].reshape(B, q_groups * gr, S, d)
        q2 = q_[:, :, gr:, :, :].reshape(B, q_groups, S, d)

        # Split k similarly with k_ratio
        kr = self.k_ratio
        k_groups = self.num_kv_heads // (kr + 1)
        k_ = k.reshape(B, k_groups, kr + 1, kv_seq, d)
        k1 = k_[:, :, :kr, :, :].reshape(B, k_groups * kr, kv_seq, d)
        k2 = k_[:, :, kr:, :, :].reshape(B, k_groups, kv_seq, d)

        # Split v with grouped_ratio=1 (always two groups of equal size = k_noise_heads each)
        v_ = v.reshape(B, self.k_noise_heads, 2, kv_seq, d)
        v1 = v_[:, :, 0, :, :]
        v2 = v_[:, :, 1, :, :]

        # Concat along the head axis to feed two SDPAs that share a queries-and-keys layout.
        q_f = mx.concatenate([q1, q2], axis=1)

        if self.kv_repeat > 1:
            k1 = _repeat(k1, self.kv_repeat, axis=1)
            k2 = _repeat(k2, self.kv_repeat, axis=1)
            v1 = _repeat(v1, self.kv_repeat, axis=1)
            v2 = _repeat(v2, self.kv_repeat, axis=1)

        # Available flash kernels:
        #   gda_decode        : original serial-per-thread (correct, slow)
        #   sdpa_dual_v       : shared-QK dual-V SDPA (default at decode)
        #   gda_post_split    : split-input post-reduction (avoids concat)
        # See kernels.py docstrings for design notes.
        from mlx_motif.kernels import gda_decode, gda_post, gda_post_split, sdpa_dual_v

        lam = self._lambda_full(mx.float32).reshape(1)
        use_serial_flash = (
            os.environ.get("MLX_MOTIF_FLASH_DECODE", "0") not in ("0", "", "false", "False")
        )
        use_dual_v = (
            os.environ.get("MLX_MOTIF_DUAL_V", "1") not in ("0", "", "false", "False")
            and S == 1
            and self.kv_repeat == 1
            and kr == 1
            and not self.args.fused_rope
        )

        if use_serial_flash and S == 1 and self.kv_repeat == 1 and kr == 1:
            out = gda_decode(
                q1, q2, k1, k2, v1, v2,
                self.subln.weight, lam, self.lambda_init,
                gr, self.scale, eps=self.args.attn_rms_norm_eps,
            )
        elif use_dual_v:
            # Custom kernel: shared QK, dual V with native GQA broadcast.
            # Origin call: 32 Q heads with 8 KV heads (gqa=4) — kernel
            # broadcasts internally via `kv_head_idx = head_idx / GQA_FACTOR`,
            # so we avoid materializing the repeated 32-head K/V tensors.
            # Noise call: 8 Q heads, 8 KV heads (gqa=1).
            attn_origin = sdpa_dual_v(q1, k1, v1, v2, self.scale)  # GQA=gr
            attn_noise  = sdpa_dual_v(q2, k2, v1, v2, self.scale)  # GQA=1
            # Skip the (B, q_origin+q_groups, S, 2d) concat — gda_post_split
            # reads attn_o and attn_n separately and broadcasts noise heads
            # via index inside the kernel. Saves a 316us-class allocation.
            out = gda_post_split(
                attn_origin, attn_noise, self.subln.weight, lam, self.lambda_init,
                gr, eps=self.args.attn_rms_norm_eps,
            )
        else:
            if kr == 1:
                k_f = mx.concatenate([_repeat(k1, gr, axis=1), k2], axis=1)
            else:
                k_f = mx.concatenate([k1, k2], axis=1)
            v1_f = mx.concatenate([_repeat(v1, gr, axis=1), v1], axis=1)
            v2_f = mx.concatenate([_repeat(v2, gr, axis=1), v2], axis=1)
            v_cat = mx.concatenate([v1_f, v2_f], axis=-1)
            attn_cat = mx.fast.scaled_dot_product_attention(
                q_f, k_f, v_cat, scale=self.scale, mask=mask
            )
            out = gda_post(
                attn_cat, self.subln.weight, lam, self.lambda_init,
                q_groups, gr, eps=self.args.attn_rms_norm_eps,
            )

        # (B, q_groups*gr, S, 2d) -> (B, S, q_groups*gr*2d)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Decoder layer / model
# --------------------------------------------------------------------------- #


class MotifDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = MotifAttention(args, layer_idx)
        self.mlp = MotifMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MotifModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [MotifDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if mask is None:
            mask = create_attention_mask(h, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """
    Top-level CausalLM wrapper. Mirrors mlx-lm conventions so the model is
    consumable by `mlx_lm.generate` / `mlx_lm.server`.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MotifModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache=None,
    ) -> mx.array:
        h = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self) -> int:
        return self.args.head_dim or (self.args.hidden_size // self.args.num_attention_heads)

    @property
    def n_kv_heads(self) -> int:
        # KV cache stores K and V at the *physical* head granularity, which
        # matches `num_key_value_heads` for both vanilla and grouped variants
        # (vanilla halves it internally but stores 2× heads of width d).
        return self.args.num_key_value_heads

    def fuse_qkv(self) -> None:
        """Fuse `q_proj`, `k_proj`, `v_proj` of every grouped-attention layer
        into a single `qkv_proj` linear (concatenated along the output axis).

        Cuts ~10% off the per-layer projection cost by hitting MLX's
        quantized matmul sweet spot (single 4096→9216 op vs three smaller
        ops). Call after `load_weights`.
        """
        if not self.args.is_grouped:
            return
        for layer in self.model.layers:
            attn = layer.self_attn
            q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
            quantized = isinstance(q, nn.QuantizedLinear)
            in_dim = q.weight.shape[1] * (32 // q.bits if quantized else 1)
            out_dim = (q.weight.shape[0] if quantized else q.weight.shape[0]) + \
                      (k.weight.shape[0] if quantized else k.weight.shape[0]) + \
                      (v.weight.shape[0] if quantized else v.weight.shape[0])
            if quantized:
                fused = nn.QuantizedLinear(
                    in_dim, out_dim, bias=False,
                    group_size=q.group_size, bits=q.bits, mode=q.mode,
                )
                fused.weight = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
                fused.scales = mx.concatenate([q.scales, k.scales, v.scales], axis=0)
                fused.biases = mx.concatenate([q.biases, k.biases, v.biases], axis=0)
            else:
                fused = nn.Linear(in_dim, out_dim, bias=False)
                fused.weight = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
            attn.qkv_proj = fused
            # Drop the originals so they don't double the parameter count
            # in `model.parameters()`.
            attn.q_proj = nn.Identity()
            attn.k_proj = nn.Identity()
            attn.v_proj = nn.Identity()

    def sanitize(self, weights: dict) -> dict:
        """Strip any unsupported buffers (rotary inv_freq) and pass through."""
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k and "rope.inv_freq" not in k
        }
