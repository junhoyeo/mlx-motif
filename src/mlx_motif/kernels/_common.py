"""Shared imports and the disable-flag for all kernel submodules."""

from __future__ import annotations

import os
from functools import cache

import mlx.core as mx

_DISABLE = os.environ.get("MLX_MOTIF_DISABLE_KERNELS", "0") not in ("0", "", "false", "False")


@cache
def _scalar_f32(value: float) -> mx.array:
    """Return a memoized 1-element fp32 array for a constant scalar kernel input.

    Kernel wrappers pass per-model constants (eps, per-layer 1-lambda scale) as
    ``mx.array([value], dtype=mx.float32)`` on every invocation. These values are
    frozen at inference, so allocating a fresh host array per layer per decode
    token is pure allocation/dispatch overhead. Keyed on the Python float, the
    cache returns the same immutable array — safe to reuse because kernels only
    read it, never mutate it.
    """
    return mx.array([value], dtype=mx.float32)
