"""
Custom KV cache for Motif's grouped differential attention.

The grouped attention path needs k1/k2/v1/v2 split per-step. mlx-lm's stock
`QuantizedKVCache` quantizes K and V together (single tensor each), which
forces the differential head split to dequantize the full cache and re-slice
post-fetch — defeating the bandwidth win.

`MotifGroupedKVCache` and `MotifGroupedQuantizedKVCache` keep the four
slots separate end-to-end so each branch can be fed straight to the
attention kernel without intermediate concat. Quantized slots can be consumed
directly by `sdpa_dual_v_q4`; the dequant bridge remains as a fallback.

Both classes implement the mlx-lm `_BaseCache` contract (`offset`,
`update_and_fetch`, `state`, `meta_state`, `is_trimmable`, `trim`,
`make_mask`) plus a 4-slot `update_and_fetch_4(k1, k2, v1, v2)` helper
that the model uses when it detects the cache type.

Fetch contract: the hot-path fetches (`MotifGroupedKVCache.update_and_fetch_4`
and `update_and_fetch_4_quantized`) return the FULL row-contiguous capacity
buffers — axis-2 length is the step-padded capacity, NOT the live length.
Consumers must bound reads by `cache.offset` (the decode kernels take it as a
runtime `kv_len` input; other consumers slice `[..., :offset, :]`). This keeps
the buffers contiguous so no per-step `ensure_row_contiguous` copy of the live
KV region is inserted before custom-kernel launches. The quantized dequant
bridge (`MotifGroupedQuantizedKVCache.update_and_fetch_4`) still returns
exact-length dequantized slices.
"""

from __future__ import annotations

import mlx.core as mx

# NOTE: import the cache-module `create_attention_mask`, whose signature is
# `(N, offset, return_array, window_size)` — NOT the base-module one
# `(h, cache=None, window_size=None, return_array=False)`. `make_mask` below is
# invoked by mlx-lm as `cache.make_mask(N, return_array=..., window_size=...)`
# and forwards `offset=self.offset`, which only the cache-module function
# accepts. Importing from `mlx_lm.models.base` here raises TypeError on any
# invocation (see cache-module `_BaseCache.make_mask`, whose body this mirrors).
from mlx_lm.models.cache import _BaseCache, create_attention_mask


class MotifGroupedKVCacheBase(_BaseCache):
    """Shared base for the 4-slot grouped KV caches.

    Holds offset bookkeeping and all methods whose implementation is identical
    between the fp16 and quantized variants:
        - step constant (overridable per instance for tests)
        - update_and_fetch (NotImplementedError shim)
        - is_trimmable / trim / make_mask
        - state.setter (unpacks 4-tuple into k1/k2/v1/v2)
        - _grow skeleton (delegates storage differences to subclass hooks)

    Subclass contract (must set in __init__ before any call):
        self.k1 = self.k2 = self.v1 = self.v2 = None
        self.offset = 0

    Subclass hooks for _grow:
        _slot_capacity(slot) -> int   index-2 length of an allocated slot
        _init_storage(B, H, n, D, dtype) -> slot   fresh zeroed slot of length n
        _trim_slot(slot, prev) -> slot              slice slot to [:prev] along axis-2
        _expand_slot(slot, fresh) -> slot           concatenate slot + fresh along axis-2
    """

    step = 256

    # ------------------------------------------------------------------
    # mlx-lm _BaseCache contract: single-slot shim
    # ------------------------------------------------------------------

    def update_and_fetch(self, keys, values):
        # Should not be reached for the grouped attention path; kept so that
        # generic mlx-lm code paths (e.g., spec decoding cache trim) still work.
        raise NotImplementedError(
            "MotifGroupedKVCacheBase subclasses require the model to call "
            "update_and_fetch_4 directly — the 1-slot path is intentionally unsupported."
        )

    # ------------------------------------------------------------------
    # mlx-lm _BaseCache contract: trimmability
    # ------------------------------------------------------------------

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    # ------------------------------------------------------------------
    # state setter (getter differs between subclasses)
    # ------------------------------------------------------------------

    @property
    def state(self):
        raise NotImplementedError

    @state.setter
    def state(self, v):
        self.k1, self.k2, self.v1, self.v2 = v

    # ------------------------------------------------------------------
    # _grow skeleton — storage differences delegated to subclass hooks
    # ------------------------------------------------------------------

    def _grow(self, B: int, H: int, S: int, D: int, dtype) -> None:
        prev = self.offset
        if self.k1 is None or (prev + S) > self._slot_capacity(self.k1):
            n_steps = (self.step + S - 1) // self.step
            new_len = n_steps * self.step
            if self.k1 is not None:
                if prev % self.step != 0:
                    self.k1 = self._trim_slot(self.k1, prev)
                    self.k2 = self._trim_slot(self.k2, prev)
                    self.v1 = self._trim_slot(self.v1, prev)
                    self.v2 = self._trim_slot(self.v2, prev)
                fresh = self._init_storage(B, H, new_len, D, dtype)
                self.k1 = self._expand_slot(self.k1, fresh)
                self.k2 = self._expand_slot(self.k2, fresh)
                self.v1 = self._expand_slot(self.v1, fresh)
                self.v2 = self._expand_slot(self.v2, fresh)
            else:
                self.k1 = self._init_storage(B, H, new_len, D, dtype)
                self.k2 = self._init_storage(B, H, new_len, D, dtype)
                self.v1 = self._init_storage(B, H, new_len, D, dtype)
                self.v2 = self._init_storage(B, H, new_len, D, dtype)

    # ------------------------------------------------------------------
    # Subclass hooks (abstract)
    # ------------------------------------------------------------------

    def _slot_capacity(self, slot) -> int:
        raise NotImplementedError

    def _init_storage(self, B: int, H: int, n: int, D: int, dtype):
        raise NotImplementedError

    def _trim_slot(self, slot, prev: int):
        raise NotImplementedError

    def _expand_slot(self, slot, fresh):
        raise NotImplementedError

    @staticmethod
    def _assert_uniform_heads(k1, k2, v1, v2) -> None:
        """All four slots share one allocation head count (derived from k1).

        With k_ratio > 1 the model produces k1 with ``k_groups*k_ratio`` heads
        but k2/v1/v2 with ``k_groups`` heads; storing them in equally-shaped
        slots would fail (or silently corrupt) at write time. Guard loudly here
        so the unsupported combination is unmistakable even if a caller
        constructs the cache directly (bypassing ``Model.make_cache``).
        """
        h1 = k1.shape[1]
        if not (k2.shape[1] == v1.shape[1] == v2.shape[1] == h1):
            raise ValueError(
                "4-slot grouped KV cache requires k1/k2/v1/v2 to share a head "
                f"count; got k1={h1}, k2={k2.shape[1]}, v1={v1.shape[1]}, "
                f"v2={v2.shape[1]}. This happens with k_ratio > 1, which the "
                "4-slot cache does not support."
            )


class MotifGroupedKVCache(MotifGroupedKVCacheBase):
    """4-slot unquantized KV cache for grouped DiffAttn (k1, k2, v1, v2).

    Behaves like a fp16/bf16 KVCache but stores the four buffers separately,
    which lets the forward path skip the post-fetch head-axis split.
    """

    def __init__(self):
        self.k1 = None
        self.k2 = None
        self.v1 = None
        self.v2 = None
        self.offset = 0

    # ------------------------------------------------------------------
    # Storage hooks (fp16/bf16 tensors)
    # ------------------------------------------------------------------

    def _slot_capacity(self, slot) -> int:
        return slot.shape[2]

    def _init_storage(self, B: int, H: int, n: int, D: int, dtype):
        return mx.zeros((B, H, n, D), dtype)

    def _trim_slot(self, slot, prev: int):
        return slot[..., :prev, :]

    def _expand_slot(self, slot, fresh):
        return mx.concatenate([slot, fresh], axis=2)

    # ------------------------------------------------------------------
    # 4-slot update/fetch
    # ------------------------------------------------------------------

    def update_and_fetch_4(self, k1, k2, v1, v2):
        """Append the 4 incoming slices and return the FULL capacity buffers.

        The returned buffers are row-contiguous and step-padded: their axis-2
        length is the allocated capacity (a multiple of `step`), not the live
        length. Consumers must bound reads by `self.offset` — either by
        passing `kv_len=self.offset` to a runtime-length kernel
        (`sdpa_dual_v`) or by slicing `[..., :self.offset, :]`.

        Returning full buffers (instead of exact-length `[..., :o, :]`
        slices) keeps the tensors row-contiguous so `mx.fast.metal_kernel`'s
        `ensure_row_contiguous` does not copy the whole live KV region on
        every decode step.
        """
        self._assert_uniform_heads(k1, k2, v1, v2)
        B, H, S, D = k1.shape
        prev = self.offset
        self._grow(B, H, S, D, k1.dtype)
        self.k1[..., prev : prev + S, :] = k1
        self.k2[..., prev : prev + S, :] = k2
        self.v1[..., prev : prev + S, :] = v1
        self.v2[..., prev : prev + S, :] = v2
        self.offset += S
        return self.k1, self.k2, self.v1, self.v2

    # ------------------------------------------------------------------
    # mlx-lm _BaseCache contract: state / meta_state
    # ------------------------------------------------------------------

    @property
    def state(self):
        if self.offset == 0 or self.k1 is None:
            return self.k1, self.k2, self.v1, self.v2
        o = self.offset
        return (
            self.k1[..., :o, :],
            self.k2[..., :o, :],
            self.v1[..., :o, :],
            self.v2[..., :o, :],
        )

    @state.setter
    def state(self, v):
        self.k1, self.k2, self.v1, self.v2 = v

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_quantized(self, group_size: int = 64, bits: int = 8) -> MotifGroupedQuantizedKVCache:
        """Switch to the quantized variant — keeps the 4 slots, quantizes them."""
        new = MotifGroupedQuantizedKVCache(group_size=group_size, bits=bits)
        if self.offset > 0:
            o = self.offset
            new.k1 = mx.quantize(self.k1[..., :o, :], group_size=group_size, bits=bits)
            new.k2 = mx.quantize(self.k2[..., :o, :], group_size=group_size, bits=bits)
            new.v1 = mx.quantize(self.v1[..., :o, :], group_size=group_size, bits=bits)
            new.v2 = mx.quantize(self.v2[..., :o, :], group_size=group_size, bits=bits)
            new.offset = self.offset
        return new


class MotifGroupedQuantizedKVCache(MotifGroupedKVCacheBase):
    """4-slot quantized KV cache. Each of (k1, k2, v1, v2) lives in HBM as
    a quantized triple `(data: uint32, scales: dtype, biases: dtype)`.

    Bandwidth win at long context — packed 4-bit reads ¼ the bytes of fp16.
    `update_and_fetch_4_quantized` returns the packed triples for
    `sdpa_dual_v_q4`; `update_and_fetch_4` keeps the slower dequantized bridge
    for fp16-input attention fallbacks.
    """

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.k1 = self.k2 = self.v1 = self.v2 = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    # ------------------------------------------------------------------
    # Storage hooks (quantized triples)
    # ------------------------------------------------------------------

    def _slot_capacity(self, slot) -> int:
        return slot[0].shape[2]

    def _init_storage(self, B: int, H: int, n: int, D: int, dtype):
        el_per_int = 8 * mx.uint32.size // self.bits
        return (
            mx.zeros((B, H, n, D // el_per_int), dtype=mx.uint32),
            mx.zeros((B, H, n, D // self.group_size), dtype=dtype),
            mx.zeros((B, H, n, D // self.group_size), dtype=dtype),
        )

    def _trim_slot(self, slot, prev: int):
        return tuple(x[..., :prev, :] for x in slot)

    def _expand_slot(self, slot, fresh):
        return tuple(mx.concatenate([slot[i], fresh[i]], axis=-2) for i in range(3))

    # ------------------------------------------------------------------
    # 4-slot update/fetch
    # ------------------------------------------------------------------

    def update_and_fetch_4(self, k1, k2, v1, v2):
        """Quantize the 4 incoming slices and append; return dequantized
        slices of the live cache region (so the fp16/bf16 attention kernel
        consumes unchanged inputs)."""
        self._update_4(k1, k2, v1, v2)
        o = self.offset

        # Dequantize the live region for downstream attention kernel.
        def _deq(slot):
            return mx.dequantize(
                slot[0][..., :o, :],
                slot[1][..., :o, :],
                slot[2][..., :o, :],
                group_size=self.group_size,
                bits=self.bits,
            )

        return _deq(self.k1), _deq(self.k2), _deq(self.v1), _deq(self.v2)

    def update_and_fetch_4_quantized(self, k1, k2, v1, v2):
        """Quantize the 4 incoming slices and append; return each slot's FULL
        capacity buffers as raw quantized triples `(data, scales, biases)`.

        This is the bandwidth-saving path: the consumer (a quant-input
        attention kernel like `sdpa_dual_v_q4`) reads packed 4/8-bit
        memory directly without paying the per-step `mx.dequantize` cost.

        The triples are row-contiguous and step-padded (axis-2 length is the
        allocated capacity, not the live length); consumers must bound reads
        by `self.offset` (pass `kv_len=self.offset` to `sdpa_dual_v_q4`).
        Exact-length `[..., :o, :]` slices would be non-contiguous views and
        force `ensure_row_contiguous` to copy all 9 live-region buffers on
        every decode step.
        """
        self._update_4(k1, k2, v1, v2)
        return self.k1, self.k2, self.v1, self.v2

    def _update_4(self, k1, k2, v1, v2):
        self._assert_uniform_heads(k1, k2, v1, v2)
        B, H, S, D = k1.shape
        prev = self.offset
        self._grow(B, H, S, D, k1.dtype)

        # Batch the four same-shape (B, H, S, D) slices along the head axis and
        # quantize once, collapsing 4 mx.quantize dispatches into 1 concat + 1
        # quantize on the per-token decode hot path. mx.quantize groups along
        # the last axis (D), so a head-axis (axis 1) concatenation never mixes
        # groups — the packed triple is bit-identical per head to quantizing
        # each slot separately (verified in tests/test_grouped_cache.py).
        fresh = mx.concatenate([k1, k2, v1, v2], axis=1)
        q_data, q_scales, q_biases = mx.quantize(fresh, group_size=self.group_size, bits=self.bits)

        # Split the packed triple back into per-slot head partitions and write
        # each into its own backing triple. The four slots stay physically
        # separate so the fetch path returns contiguous per-slot views
        # unchanged — feeding head-offset slices of one unified buffer to
        # `sdpa_dual_v_q4` (ensure_row_contiguous=True) would instead force a
        # copy of the whole growing live region every token.
        for i, slot in enumerate((self.k1, self.k2, self.v1, self.v2)):
            h0, h1 = i * H, (i + 1) * H
            slot[0][..., prev : prev + S, :] = q_data[:, h0:h1, :, :]
            slot[1][..., prev : prev + S, :] = q_scales[:, h0:h1, :, :]
            slot[2][..., prev : prev + S, :] = q_biases[:, h0:h1, :, :]

        self.offset += S

    # ------------------------------------------------------------------
    # mlx-lm _BaseCache contract: state / meta_state
    # ------------------------------------------------------------------

    @property
    def state(self):
        return self.k1, self.k2, self.v1, self.v2

    @state.setter
    def state(self, v):
        self.k1, self.k2, self.v1, self.v2 = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)
