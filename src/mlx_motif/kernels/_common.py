"""Shared imports and the disable-flag for all kernel submodules."""

from __future__ import annotations

import os

import mlx.core as mx

_DISABLE = os.environ.get("MLX_MOTIF_DISABLE_KERNELS", "0") not in ("0", "", "false", "False")
