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

import enum
import math
import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope


def _maybe_dequant_kv(k, v, cache):
    """Decode-time bridge for QuantizedKVCache.

    The differential pattern slices K and V along the head axis AFTER cache
    fetch (`k[:, :, 0]`, `k[:, :, 1]`, etc.). When the cache is quantized,
    `update_and_fetch` returns triples (data, scales, biases) which can't be
    sliced. This fallback materializes dequantized K/V for fp16/bf16 attention
    kernels; the quantized-input `sdpa_dual_v_q4` path bypasses it when possible.
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
    head_dim: int | None = None
    num_noise_heads: int | None = None
    k_ratio: int = 1
    attn_rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    rope_scaling: dict | None = None
    hidden_act: str = "poly_norm"
    use_bias: bool = False
    expanded: bool = False
    sliding_window: int | None = None
    use_sliding_window: bool = False
    max_window_layers: int | None = None
    fused_rope: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None

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


# --------------------------------------------------------------------------- #
# Attention-path resolution
# --------------------------------------------------------------------------- #


class AttnPath(enum.Enum):
    """Which kernel + cache-fetch strategy to use in _forward_grouped.

    QUANT_SDPA   — sdpa_dual_v_q4: K/V stay quantized through the kernel.
    DUAL_V       — sdpa_dual_v: fp16 K/V, shared-QK dual-V kernel.
    FALLBACK     — per-slab fast SDPA + gda_post_split (prefill / GQA repeat
                   path); legacy stacked SDPA + gda_post for k_ratio > 1.
    """

    QUANT_SDPA = "sdpa_dual_v_q4"
    DUAL_V = "sdpa_dual_v"
    FALLBACK = "stacked_sdpa"


def _resolve_attention_path(
    cache,
    S: int,
    kv_repeat: int,
    kr: int,
    fused_rope: bool,
    env=os.environ,
) -> AttnPath:
    """Choose the kernel + cache-fetch strategy for one forward step.

    Pure function of its arguments — no global side-effects. Pass a plain
    dict (or any object with a `.get(name, default)` method) for `env` to
    control the feature flags in tests.

    Decision order (first match wins):

    1. QUANT_SDPA — cache is ``MotifGroupedQuantizedKVCache`` AND
       ``MLX_MOTIF_QUANT_SDPA != 0`` AND single-token decode with no GQA
       repeat, ``k_ratio == 1``, and ``not fused_rope`` (kernel consumes
       post-RoPE q/k from the standard rope() call).
    2. DUAL_V    — ``MLX_MOTIF_DUAL_V != 0`` AND same decode conditions
       AND ``not fused_rope`` (same reason as QUANT_SDPA).
    3. FALLBACK  — everything else (prefill, GQA repeat, k_ratio > 1, …).
    """
    from mlx_motif.cache import MotifGroupedQuantizedKVCache

    _is_falsy = ("0", "", "false", "False")

    # Single-token decode shape preconditions shared by both custom paths.
    decode_shape_ok = S == 1 and kv_repeat == 1 and kr == 1

    if (
        decode_shape_ok
        and not fused_rope
        and isinstance(cache, MotifGroupedQuantizedKVCache)
        and env.get("MLX_MOTIF_QUANT_SDPA", "1") not in _is_falsy
    ):
        return AttnPath.QUANT_SDPA

    if decode_shape_ok and not fused_rope and env.get("MLX_MOTIF_DUAL_V", "1") not in _is_falsy:
        return AttnPath.DUAL_V

    return AttnPath.FALLBACK


class _LambdaCache:
    """Holds memoized ``lambda_full`` arrays outside the nn.Module parameter tree.

    Stored as a plain (non-array/dict/list/tuple) attribute so mlx's
    ``Module.__setattr__`` keeps it off the module's parameter dict — otherwise
    the cached arrays would be picked up by ``parameters()`` / ``update()`` /
    quantization and corrupt weight round-tripping.
    """

    __slots__ = ("store",)

    def __init__(self):
        self.store: dict = {}


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
        self.scale = self.head_dim**-0.5
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

        # Memoized lambda_full per output dtype. The four lambda vectors are
        # frozen at inference, so lambda_full is a constant recomputed per layer
        # per token on the decode hot loop — cache it on first use. Held on a
        # plain object (see _LambdaCache) so it stays out of the parameter tree.
        # Correctness note: this assumes weights are set once before the first
        # forward (the load path does load_weights()/fuse_qkv() up front, neither
        # of which mutates lambda_q/k). Reload the module to invalidate.
        self._lam_cache = _LambdaCache()

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
        self.grouped_ratio = (
            args.num_attention_heads - self.num_noise_heads
        ) // self.num_noise_heads
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
        self.q_proj = nn.Linear(h, h, bias=args.use_bias)
        self.k_proj = nn.Linear(h, h // self.n_rep, bias=args.use_bias)
        self.v_proj = nn.Linear(h, h // self.n_rep, bias=args.use_bias)
        self.o_proj = nn.Linear(h, h, bias=args.use_bias)

    # -- common helpers ----------------------------------------------------- #

    def _lambda_full(self, dtype: mx.Dtype) -> mx.array:
        cached = self._lam_cache.store.get(dtype)
        if cached is not None:
            return cached
        l1 = mx.exp(mx.sum(self.lambda_q1 * self.lambda_k1, axis=-1).astype(mx.float32))
        l2 = mx.exp(mx.sum(self.lambda_q2 * self.lambda_k2, axis=-1).astype(mx.float32))
        val = (l1 - l2 + self.lambda_init).astype(dtype)
        self._lam_cache.store[dtype] = val
        return val

    # -- forward dispatch --------------------------------------------------- #

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
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
        # INVARIANT: V is stored in the cache at the same head granularity as K
        # (so `n_kv_heads = num_key_value_heads`) but in *slab* head order
        # [va_0..va_{Hk-1}, vb_0..vb_{Hk-1}] instead of the projection's
        # paired order [v0_a, v0_b, v1_a, v1_b, ..]. That makes the two
        # d-wide value slabs contiguous head-axis slices after fetch — the
        # old paired layout needed a reshape→transpose→reshape that
        # materialized a copy of the ENTIRE cached V every forward step.
        #
        # This slab ordering is a TRAP for any code reading the raw V cache
        # expecting HF head order. `Model.make_cache` therefore hands the
        # vanilla path a `MotifVanillaKVCache`, which carries the ordering
        # marker in `meta_state` (loud failure on mismatched restore) and
        # exposes `hf_ordered_values()` for consumers that need HF order. The
        # reorder itself is `vanilla_v_paired_to_slab_perm` applied on the head
        # axis, expressed here as a fused reshape/transpose on the new tokens.
        B, S, _ = x.shape
        d = self.head_dim
        H = self.num_heads  # halved (effective DiffAttn heads)
        Hk = self.num_kv_heads  # halved (effective DiffAttn KV heads)

        q = self.q_proj(x).reshape(B, S, 2 * H, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, 2 * Hk, d).transpose(0, 2, 1, 3)
        # Rearrange V into slab order at projection time — O(S·d) on the new
        # tokens only (S=1 at decode), instead of O(context) after fetch.
        v = self.v_proj(x).reshape(B, S, Hk, 2, d).transpose(0, 3, 2, 1, 4).reshape(B, 2 * Hk, S, d)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
            k, v = _maybe_dequant_kv(k, v, cache)
        kv_seq = k.shape[2]

        # Value slabs: contiguous head-axis slices, (B, Hk, kv_seq, d) each.
        va, vb = v[:, :Hk], v[:, Hk:]

        # Stripe split for differential attention: heads pair adjacent in flat layout.
        q_ = q.reshape(B, H, 2, S, d)
        k_ = k.reshape(B, Hk, 2, kv_seq, d)
        q1, q2 = q_[:, :, 0], q_[:, :, 1]
        k1, k2 = k_[:, :, 0], k_[:, :, 1]

        # Per-slab SDPA with query_head_dim == value_head_dim == d hits MLX's
        # fast attention templates — q width d against a 2d-wide V is exactly
        # the combination that falls back to the slow generic SDPA (see
        # kernels/attention.py). K/V are passed at Hk heads: `mx.fast`
        # SDPA's native GQA broadcast maps q head i -> kv head i // n_rep,
        # identical to the explicit `_repeat` (repeat_interleave) it replaces.
        #
        # Memory note: these four fast-template calls are O(S) in prefill
        # memory (no S×S score buffer). The old generic v=2d path was O(S²);
        # it is ~119 MB cheaper e2e at a 3.5k prompt but +369 MB at 6k and
        # +755 MB at 8k, and it loses the decode-speed win — so the fast path
        # is kept despite a measured +131 MB e2e peak at 3.5k. Full attribution
        # + crossover table: docs/experiments/experiment-vanilla-prefill-peak-memory.md
        o1a = mx.fast.scaled_dot_product_attention(q1, k1, va, scale=self.scale, mask=mask)
        o1b = mx.fast.scaled_dot_product_attention(q1, k1, vb, scale=self.scale, mask=mask)
        o2a = mx.fast.scaled_dot_product_attention(q2, k2, va, scale=self.scale, mask=mask)
        o2b = mx.fast.scaled_dot_product_attention(q2, k2, vb, scale=self.scale, mask=mask)

        lam = self._lambda_full(o1a.dtype)
        # (out1 - λ·out2) computed slab-wise, then one small channel concat.
        out = mx.concatenate([o1a - lam * o2a, o1b - lam * o2b], axis=-1)
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
        # quantized matmul sweet spot at output_dim=q+k+v=9216).
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
        # If fuse_qkv() permuted Q rows to origin-first, the q tensor already
        # has [origin-heads | noise-heads] layout — q1 / q2 below become
        # contiguous slices (zero-copy) instead of stride-pattern reshapes.
        self._q_origin_first = getattr(self, "_qkv_origin_first", False)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        # RoPE acts independently per head, so roping the full k here — before
        # the head-axis split into k1/k2 below — is identical to roping each
        # slice separately, but costs one kernel dispatch instead of two.
        k = self.rope(k, offset=offset)

        from mlx_motif.cache import MotifGroupedKVCache, MotifGroupedQuantizedKVCache

        gr = self.grouped_ratio
        q_groups = self.q_heads // (gr + 1)
        kr = self.k_ratio
        k_groups = self.num_kv_heads // (kr + 1)
        is_4slot = isinstance(cache, (MotifGroupedKVCache, MotifGroupedQuantizedKVCache))

        # Single decision point: resolve which kernel + cache-fetch to use.
        # Both the cache-fetch branch below and the kernel-selection branch
        # further down key off the same `path` value — no condition is
        # re-evaluated independently.
        path = _resolve_attention_path(
            cache,
            S,
            self.kv_repeat,
            kr,
            self.args.fused_rope,
        )

        # ------------------------------------------------------------------ #
        # Cache fetch — shape depends on path
        # ------------------------------------------------------------------ #

        k1_q = k2_q = v1_q = v2_q = None  # quantized triples; QUANT_SDPA only

        if is_4slot:
            # k is already RoPE'd above (before the split); slice it into the
            # 4-slot layout for the cache write.
            k_pre = k.reshape(B, k_groups, kr + 1, S, d)
            k1_new = k_pre[:, :, :kr, :, :].reshape(B, k_groups * kr, S, d)
            k2_new = k_pre[:, :, kr:, :, :].reshape(B, k_groups, S, d)
            v_pre = v.reshape(B, self.k_noise_heads, 2, S, d)
            v1_new = v_pre[:, :, 0, :, :]
            v2_new = v_pre[:, :, 1, :, :]
            if path is AttnPath.QUANT_SDPA:
                # Return raw quantized triples; sdpa_dual_v_q4 reads packed
                # bits directly — no per-step mx.dequantize. The triples are
                # FULL capacity buffers (row-contiguous); the kernel bounds
                # reads by the runtime kv_len = cache.offset passed below.
                k1_q, k2_q, v1_q, v2_q = cache.update_and_fetch_4_quantized(
                    k1_new,
                    k2_new,
                    v1_new,
                    v2_new,
                )
                k1 = k2 = v1 = v2 = None  # unused on this path
            else:
                k1, k2, v1, v2 = cache.update_and_fetch_4(k1_new, k2_new, v1_new, v2_new)
                if path is not AttnPath.DUAL_V:
                    # FALLBACK consumes exact-length tensors (generic SDPA has
                    # no runtime length bound, and the mask only covers the
                    # live region) — slice the step-padded capacity buffers.
                    # MotifGroupedQuantizedKVCache's dequant bridge already
                    # returns exact-length slices, so this is a no-op view
                    # for that cache type.
                    o = cache.offset
                    k1 = k1[..., :o, :]
                    k2 = k2[..., :o, :]
                    v1 = v1[..., :o, :]
                    v2 = v2[..., :o, :]
            kv_seq = cache.offset
        else:
            # k already RoPE'd above.
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
                k, v = _maybe_dequant_kv(k, v, cache)
            kv_seq = k.shape[2]

        # ------------------------------------------------------------------ #
        # Q / K / V splits
        # ------------------------------------------------------------------ #

        # Split q on the head axis. Two layouts:
        #   - origin-first (after fuse_qkv): q is [origin | noise], plain slice
        #   - HF stripe layout: q is [g0_o..g0_o, g0_n, g1_o..g1_o, g1_n, ..],
        #     which forces a memory-copy reshape.
        if self._q_origin_first:
            q1 = q[:, : q_groups * gr, :, :]
            q2 = q[:, q_groups * gr :, :, :]
        else:
            q_ = q.reshape(B, q_groups, gr + 1, S, d)
            q1 = q_[:, :, :gr, :, :].reshape(B, q_groups * gr, S, d)
            q2 = q_[:, :, gr:, :, :].reshape(B, q_groups, S, d)

        if not is_4slot:
            # Split k similarly with k_ratio
            k_ = k.reshape(B, k_groups, kr + 1, kv_seq, d)
            k1 = k_[:, :, :kr, :, :].reshape(B, k_groups * kr, kv_seq, d)
            k2 = k_[:, :, kr:, :, :].reshape(B, k_groups, kv_seq, d)
            # Split v with grouped_ratio=1 (always two equal halves of k_noise_heads each)
            v_ = v.reshape(B, self.k_noise_heads, 2, kv_seq, d)
            v1 = v_[:, :, 0, :, :]
            v2 = v_[:, :, 1, :, :]

        # The q_f concat and the kv_repeat expansion below are only consumed by
        # the FALLBACK path, so they live inside that branch — the decode
        # kernel paths (QUANT_SDPA / DUAL_V) build no dead nodes and their
        # dataflow (q1/q2 fed directly to the custom kernels) is explicit.
        # kv_repeat > 1 is also unreachable on the custom-kernel paths:
        # _resolve_attention_path requires kv_repeat == 1 for both.

        # ------------------------------------------------------------------ #
        # Kernel selection — keyed off the same `path` computed above
        # ------------------------------------------------------------------ #

        # Available decode-path kernels:
        #   sdpa_dual_v       : shared-QK dual-V SDPA (default at decode)
        #   sdpa_dual_v_q4    : same, reads quantized cache directly
        #   gda_post_split    : split-input post-reduction (avoids concat)
        # See kernels/ docstrings for design notes.
        from mlx_motif.kernels import (
            gda_post,
            gda_post_split,
            sdpa_dual_v,
            sdpa_dual_v_q4,
        )

        lam = self._lambda_full(mx.float32).reshape(1)

        if path is AttnPath.QUANT_SDPA:
            # Quantized-input fast path: K, V1, V2 stay packed all the way
            # into the attention kernel — no per-step `mx.dequantize`. The
            # kernel does mask-without-shift for QK (MLX qdot trick) plus
            # in-register dequant for V. See `sdpa_dual_v_q4` docstring.
            attn_origin = sdpa_dual_v_q4(
                q1,
                k1_q,
                v1_q,
                v2_q,
                self.scale,
                group_size=cache.group_size,
                bits=cache.bits,
                kv_len=kv_seq,
            )
            attn_noise = sdpa_dual_v_q4(
                q2,
                k2_q,
                v1_q,
                v2_q,
                self.scale,
                group_size=cache.group_size,
                bits=cache.bits,
                kv_len=kv_seq,
            )
            out = gda_post_split(
                attn_origin,
                attn_noise,
                self.subln.weight,
                lam,
                self.lambda_init,
                gr,
                eps=self.args.attn_rms_norm_eps,
            )
        elif path is AttnPath.DUAL_V:
            # Custom kernel: shared QK, dual V SDPA with native GQA.
            # Origin call: GQA=gr (kernel broadcasts k1/v1/v2 internally).
            # Noise call:  GQA=1.
            # kv_len bounds the KV loop at runtime: with the 4-slot cache,
            # k1/k2/v1/v2 are full step-padded capacity buffers and
            # kv_seq = cache.offset < capacity; on the non-4-slot path the
            # tensors are exactly kv_seq long, so this is equivalent.
            attn_origin = sdpa_dual_v(q1, k1, v1, v2, self.scale, kv_len=kv_seq)
            attn_noise = sdpa_dual_v(q2, k2, v1, v2, self.scale, kv_len=kv_seq)
            out = gda_post_split(
                attn_origin,
                attn_noise,
                self.subln.weight,
                lam,
                self.lambda_init,
                gr,
                eps=self.args.attn_rms_norm_eps,
            )
        elif kr == 1:  # AttnPath.FALLBACK, standard k_ratio (all shipped configs)
            # Per-slab SDPA prefill path. Four calls with query_head_dim ==
            # value_head_dim == d hit MLX's fast attention templates; the old
            # stacked construction (gr-fold `_repeat` of k1/v1/v2 over the
            # full kv length + 3 head-axis concats + one SDPA with q_dim=d,
            # v_dim=2d) took the documented slow generic SDPA path (see
            # kernels/attention.py) and re-materialized those copies for
            # every chunk of a chunked prefill.
            #
            # Head mapping is unchanged: `mx.fast` SDPA's native GQA
            # broadcast maps q head i -> kv head i // (Hq // Hkv), exactly
            # the repeat_interleave layout `_repeat` produced (and the
            # composition of the kv_repeat and gr repeats, when
            # kv_repeat > 1). The same `mask` is applied per-head in each
            # call, so causal semantics for chunked prefill with a nonzero
            # cache offset are preserved exactly.
            attn_origin = mx.concatenate(
                [
                    mx.fast.scaled_dot_product_attention(q1, k1, v1, scale=self.scale, mask=mask),
                    mx.fast.scaled_dot_product_attention(q1, k1, v2, scale=self.scale, mask=mask),
                ],
                axis=-1,
            )
            attn_noise = mx.concatenate(
                [
                    mx.fast.scaled_dot_product_attention(q2, k2, v1, scale=self.scale, mask=mask),
                    mx.fast.scaled_dot_product_attention(q2, k2, v2, scale=self.scale, mask=mask),
                ],
                axis=-1,
            )
            out = gda_post_split(
                attn_origin,
                attn_noise,
                self.subln.weight,
                lam,
                self.lambda_init,
                gr,
                eps=self.args.attn_rms_norm_eps,
            )
        else:  # AttnPath.FALLBACK, exotic k_ratio > 1: legacy stacked SDPA
            if self.kv_repeat > 1:
                k1 = _repeat(k1, self.kv_repeat, axis=1)
                k2 = _repeat(k2, self.kv_repeat, axis=1)
                v1 = _repeat(v1, self.kv_repeat, axis=1)
                v2 = _repeat(v2, self.kv_repeat, axis=1)
            # Concat q on the head axis to feed a single SDPA over both halves.
            # This branch is only reachable for kr > 1 (kr == 1 takes the
            # per-slab fast path above), so no gr-fold k1 repeat is needed.
            q_f = mx.concatenate([q1, q2], axis=1)
            k_f = mx.concatenate([k1, k2], axis=1)
            v1_f = mx.concatenate([_repeat(v1, gr, axis=1), v1], axis=1)
            v2_f = mx.concatenate([_repeat(v2, gr, axis=1), v2], axis=1)
            v_cat = mx.concatenate([v1_f, v2_f], axis=-1)
            attn_cat = mx.fast.scaled_dot_product_attention(
                q_f, k_f, v_cat, scale=self.scale, mask=mask
            )
            out = gda_post(
                attn_cat,
                self.subln.weight,
                lam,
                self.lambda_init,
                q_groups,
                gr,
                eps=self.args.attn_rms_norm_eps,
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
        mask: mx.array | None = None,
        cache=None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if mask is None:
            mask = create_attention_mask(h, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache, strict=True):
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
        mask: mx.array | None = None,
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

    def make_cache(self):
        """Override mlx-lm's default cache construction.

        For grouped-attention layers, return a `MotifGroupedKVCache` (4 slots)
        if `MLX_MOTIF_4SLOT_CACHE=1`. The 4-slot path lets the per-step head
        split happen BEFORE caching, so each layer's K and V live as
        independent k1/k2/v1/v2 tensors instead of being re-sliced post-fetch.

        With `MLX_MOTIF_4SLOT_CACHE=q4` the slots are quantized to 4-bit
        (group_size=64); `=q8` for 8-bit. Quantized variants feed
        `sdpa_dual_v_q4` by default; set `MLX_MOTIF_QUANT_SDPA=0` to use the
        slower dequant-bridge fallback for A/B or debugging.
        """
        from mlx_lm.models.cache import KVCache

        from mlx_motif.cache import (
            MotifGroupedKVCache,
            MotifGroupedQuantizedKVCache,
            MotifVanillaKVCache,
        )

        env = os.environ.get("MLX_MOTIF_4SLOT_CACHE", "0").lower()
        use_4slot = env not in ("0", "", "false", "off")
        # Fail-fast: the 4-slot grouped caches allocate all four slots (k1/k2/
        # v1/v2) with a single head count derived from k1, but with k_ratio > 1
        # the origin slot k1 has k_groups*k_ratio heads while k2/v1/v2 have
        # k_groups heads — the mismatched slot writes would fail (or silently
        # corrupt) mid-forward at cache-write time. No shipped config uses
        # k_ratio > 1, so guard the unsupported combination loudly at cache
        # construction instead. (Swift mirror: MotifMLXModel.newCache.)
        if use_4slot and self.args.is_grouped and self.args.k_ratio != 1:
            raise ValueError(
                "MLX_MOTIF_4SLOT_CACHE is not supported with k_ratio > 1 "
                f"(got k_ratio={self.args.k_ratio}): the 4-slot grouped KV cache "
                "allocates every slot with k1's head count, which differs from "
                "k2/v1/v2 when k_ratio > 1. Unset MLX_MOTIF_4SLOT_CACHE for this "
                "model configuration."
            )
        caches = []
        for _layer in self.layers:
            if use_4slot and self.args.is_grouped:
                if env in ("q4", "4"):
                    caches.append(MotifGroupedQuantizedKVCache(group_size=64, bits=4))
                elif env in ("q8", "8"):
                    caches.append(MotifGroupedQuantizedKVCache(group_size=64, bits=8))
                else:
                    caches.append(MotifGroupedKVCache())
            elif not self.args.is_grouped:
                # Vanilla (2.6B) path stores V in *slab* head order (see
                # _forward_vanilla and MotifVanillaKVCache). Use the marked
                # cache so the non-standard V ordering is self-describing and
                # cannot silently leak to a consumer expecting HF head order.
                # Runtime behaviour is identical to KVCache (it inherits
                # update_and_fetch); only meta_state / accessors differ.
                caches.append(MotifVanillaKVCache())
            else:
                # Grouped model without the 4-slot cache: V is stored in its
                # projected order (no slab reorder happens on this path), so
                # the stock KVCache head ordering is correct here.
                caches.append(KVCache())
        return caches

    @staticmethod
    def _origin_first_perm(q_groups: int, gr: int, d: int) -> mx.array:
        """Build the row permutation that maps the HF head-stripe layout
        `[g0_o1, g0_o2, .., g0_o<gr>, g0_n, g1_o1, ..]` (per d-element block)
        to the origin-first layout `[g0_o1..g7_o4, g0_n..g7_n]`.

        Returns an int32 index tensor of shape `(q_heads * d,)` suitable for
        `weight = weight[perm]` row-permutation.
        """
        # New head order: [g0_o1, g0_o2, .., g_<groups-1>_o<gr>, g0_n, .., g_<groups-1>_n]
        new_to_old_head = []
        # origin heads: gr per group, q_groups groups
        for g in range(q_groups):
            for p in range(gr):
                new_to_old_head.append(g * (gr + 1) + p)
        # noise heads: 1 per group, after origin
        for g in range(q_groups):
            new_to_old_head.append(g * (gr + 1) + gr)
        # Expand to row indices (each head spans d rows of the weight matrix).
        rows = []
        for h in new_to_old_head:
            rows.extend(range(h * d, (h + 1) * d))
        return mx.array(rows, dtype=mx.int32)

    def fuse_qkv(self) -> None:
        """Fuse `q_proj`, `k_proj`, `v_proj` of every grouped-attention layer
        into a single `qkv_proj` linear (concatenated along the output axis),
        AND permute Q rows from HF's head-stripe layout to origin-first so
        the per-forward q1/q2 split becomes zero-copy views.

        Two wins stacked:
          1. ~10% on the qkv matmul (sweet-spot output dim)
          2. ~316 µs/layer saved by eliminating the Q split materialisation

        Skips fusion if the three projections were quantized at different
        bit widths (e.g., the `mixed` preset that keeps q_proj at 6-bit
        and k/v at 4-bit). Concatenating different-width packed weights
        is not supported; the per-layer Q origin-first permutation also
        becomes irrelevant since q_proj stays its own QuantizedLinear.
        """
        if not self.args.is_grouped:
            return
        for layer in self.model.layers:
            attn = layer.self_attn
            q, k, v = attn.q_proj, attn.k_proj, attn.v_proj
            quantized = isinstance(q, nn.QuantizedLinear)
            if quantized and not (
                q.bits == k.bits == v.bits and q.group_size == k.group_size == v.group_size
            ):
                # Mixed-precision quant: skip fusion. Forward path falls
                # back to three separate QuantizedLinear calls, which is
                # ~10% slower at this op but preserves the per-layer bit
                # budget the convert step chose.
                continue
            in_dim = q.weight.shape[1] * (32 // q.bits if quantized else 1)

            # Permute Q rows so origin heads come first.
            d = attn.head_dim
            q_groups = attn.num_noise_heads
            gr = attn.grouped_ratio
            perm = self._origin_first_perm(q_groups, gr, d)
            q_w = q.weight[perm]
            if quantized:
                q_s = q.scales[perm]
                q_b = q.biases[perm]

            out_dim = q.weight.shape[0] + k.weight.shape[0] + v.weight.shape[0]
            if quantized:
                fused = nn.QuantizedLinear(
                    in_dim,
                    out_dim,
                    bias=False,
                    group_size=q.group_size,
                    bits=q.bits,
                    mode=q.mode,
                )
                fused.weight = mx.concatenate([q_w, k.weight, v.weight], axis=0)
                fused.scales = mx.concatenate([q_s, k.scales, v.scales], axis=0)
                fused.biases = mx.concatenate([q_b, k.biases, v.biases], axis=0)
            else:
                fused = nn.Linear(in_dim, out_dim, bias=False)
                fused.weight = mx.concatenate([q_w, k.weight, v.weight], axis=0)
            attn.qkv_proj = fused
            attn.q_proj = nn.Identity()
            attn.k_proj = nn.Identity()
            attn.v_proj = nn.Identity()
            attn._qkv_origin_first = True

    def sanitize(self, weights: dict) -> dict:
        """Strip any unsupported buffers (rotary inv_freq) and pass through."""
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k and "rope.inv_freq" not in k
        }
