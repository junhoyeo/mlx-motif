"""
Custom KV cache for Motif's grouped differential attention.

The grouped attention path needs k1/k2/v1/v2 split per-step. mlx-lm's stock
`QuantizedKVCache` quantizes K and V together (single tensor each), which
forces the differential head split to dequantize the full cache and re-slice
post-fetch — defeating the bandwidth win.

`MotifGroupedKVCache` and `MotifGroupedQuantizedKVCache` keep the four
slots separate end-to-end so each branch can be fed straight to the
attention kernel without intermediate concat or dequant.

Both classes implement the mlx-lm `_BaseCache` contract (`offset`,
`update_and_fetch`, `state`, `meta_state`, `is_trimmable`, `trim`,
`make_mask`) plus a 4-slot `update_and_fetch_4(k1, k2, v1, v2)` helper
that the model uses when it detects the cache type.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import _BaseCache


class MotifGroupedKVCache(_BaseCache):
    """4-slot unquantized KV cache for grouped DiffAttn (k1, k2, v1, v2).

    Behaves like a fp16/bf16 KVCache but stores the four buffers separately,
    which lets the forward path skip the post-fetch head-axis split.
    """

    step = 256

    def __init__(self):
        self.k1 = None
        self.k2 = None
        self.v1 = None
        self.v2 = None
        self.offset = 0

    def _grow(self, B: int, H: int, S: int, D: int, dtype) -> None:
        prev = self.offset
        if self.k1 is None or (prev + S) > self.k1.shape[2]:
            n_steps = (self.step + S - 1) // self.step
            new_len = n_steps * self.step
            shape = (B, H, new_len, D)
            new_k1, new_k2 = mx.zeros(shape, dtype), mx.zeros(shape, dtype)
            new_v1, new_v2 = mx.zeros(shape, dtype), mx.zeros(shape, dtype)
            if self.k1 is not None:
                if prev % self.step != 0:
                    self.k1 = self.k1[..., :prev, :]
                    self.k2 = self.k2[..., :prev, :]
                    self.v1 = self.v1[..., :prev, :]
                    self.v2 = self.v2[..., :prev, :]
                self.k1 = mx.concatenate([self.k1, new_k1], axis=2)
                self.k2 = mx.concatenate([self.k2, new_k2], axis=2)
                self.v1 = mx.concatenate([self.v1, new_v1], axis=2)
                self.v2 = mx.concatenate([self.v2, new_v2], axis=2)
            else:
                self.k1, self.k2, self.v1, self.v2 = new_k1, new_k2, new_v1, new_v2

    def update_and_fetch_4(self, k1, k2, v1, v2):
        B, H, S, D = k1.shape
        prev = self.offset
        self._grow(B, H, S, D, k1.dtype)
        self.k1[..., prev : prev + S, :] = k1
        self.k2[..., prev : prev + S, :] = k2
        self.v1[..., prev : prev + S, :] = v1
        self.v2[..., prev : prev + S, :] = v2
        self.offset += S
        o = self.offset
        return self.k1[..., :o, :], self.k2[..., :o, :], self.v1[..., :o, :], self.v2[..., :o, :]

    # Compatibility shim: a single-slot interface that mlx-lm calls when it
    # doesn't know about update_and_fetch_4 (returns combined K/V tensors).
    def update_and_fetch(self, keys, values):
        # Should not be reached for the grouped attention path; kept so that
        # generic mlx-lm code paths (e.g., spec decoding cache trim) still work.
        raise NotImplementedError(
            "MotifGroupedKVCache requires the model to call update_and_fetch_4 "
            "directly — the 1-slot path is intentionally unsupported."
        )

    @property
    def state(self):
        if self.offset == 0 or self.k1 is None:
            return self.k1, self.k2, self.v1, self.v2
        o = self.offset
        return (
            self.k1[..., :o, :], self.k2[..., :o, :],
            self.v1[..., :o, :], self.v2[..., :o, :],
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

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def to_quantized(self, group_size: int = 64, bits: int = 8) -> "MotifGroupedQuantizedKVCache":
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


class MotifGroupedQuantizedKVCache(_BaseCache):
    """4-slot quantized KV cache. Each of (k1, k2, v1, v2) lives in HBM as
    a quantized triple `(data: uint32, scales: dtype, biases: dtype)`.

    Bandwidth win at long context — packed 4-bit reads ¼ the bytes of fp16.
    The fetch dequantizes back to fp16/bf16 (cost paid once per layer per
    step) so existing fp16-input attention kernels stay unchanged. A future
    quantization-native sdpa_dual_v_q4 would skip the dequant entirely.
    """

    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.k1 = self.k2 = self.v1 = self.v2 = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def _init_quant(self, B: int, H: int, total_len: int, D: int, dtype) -> tuple:
        el_per_int = 8 * mx.uint32.size // self.bits
        return (
            mx.zeros((B, H, total_len, D // el_per_int), dtype=mx.uint32),
            mx.zeros((B, H, total_len, D // self.group_size), dtype=dtype),
            mx.zeros((B, H, total_len, D // self.group_size), dtype=dtype),
        )

    def _grow(self, B: int, H: int, S: int, D: int, dtype) -> None:
        prev = self.offset
        if self.k1 is None or (prev + S) > self.k1[0].shape[2]:
            n_steps = (self.step + S - 1) // self.step
            new_len = n_steps * self.step
            if self.k1 is not None:
                # trim to the live region then concat fresh space
                if prev % self.step != 0:
                    def _trim(t):
                        return tuple(x[..., :prev, :] for x in t)
                    self.k1 = _trim(self.k1); self.k2 = _trim(self.k2)
                    self.v1 = _trim(self.v1); self.v2 = _trim(self.v2)
                fresh = self._init_quant(B, H, new_len, D, dtype)
                def _expand(t, f):
                    return tuple(mx.concatenate([t[i], f[i]], axis=-2) for i in range(3))
                self.k1 = _expand(self.k1, fresh)
                self.k2 = _expand(self.k2, fresh)
                self.v1 = _expand(self.v1, fresh)
                self.v2 = _expand(self.v2, fresh)
            else:
                self.k1 = self._init_quant(B, H, new_len, D, dtype)
                self.k2 = self._init_quant(B, H, new_len, D, dtype)
                self.v1 = self._init_quant(B, H, new_len, D, dtype)
                self.v2 = self._init_quant(B, H, new_len, D, dtype)

    def update_and_fetch_4(self, k1, k2, v1, v2):
        """Quantize the 4 incoming slices and append; return dequantized
        slices of the live cache region (so the fp16/bf16 attention kernel
        consumes unchanged inputs)."""
        B, H, S, D = k1.shape
        prev = self.offset
        self._grow(B, H, S, D, k1.dtype)

        for fresh, slot in [(k1, self.k1), (k2, self.k2), (v1, self.v1), (v2, self.v2)]:
            q_data, q_scales, q_biases = mx.quantize(
                fresh, group_size=self.group_size, bits=self.bits
            )
            slot[0][..., prev : prev + S, :] = q_data
            slot[1][..., prev : prev + S, :] = q_scales
            slot[2][..., prev : prev + S, :] = q_biases

        self.offset += S
        o = self.offset
        # Dequantize the live region for downstream attention kernel.
        def _deq(slot):
            return mx.dequantize(
                slot[0][..., :o, :], slot[1][..., :o, :], slot[2][..., :o, :],
                group_size=self.group_size, bits=self.bits,
            )
        return _deq(self.k1), _deq(self.k2), _deq(self.v1), _deq(self.v2)

    def update_and_fetch(self, keys, values):
        raise NotImplementedError(
            "MotifGroupedQuantizedKVCache requires update_and_fetch_4."
        )

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

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)
