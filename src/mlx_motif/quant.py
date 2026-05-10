"""
Quantization presets for the Motif port.

`mlx.nn.quantize` accepts a `class_predicate(path, module)` callback that
returns either a bool (use default settings) or a dict (custom `group_size` /
`bits`). We expose two presets:

- `uniform`  — every Linear gets the same `bits` (the standard `nn.quantize`
  behaviour, exposed here for symmetry).
- `mixed`    — keeps the noise-generating `q_proj` weights at higher precision
  (default 6-bit) while quantizing everything else to `bits` (default 4-bit).
  λ vectors and SubLN weights are plain `mx.array` attributes (not Linear)
  and are therefore skipped automatically.

Usage:

    from mlx_motif.quant import apply_quant
    apply_quant(model, preset="mixed", bits=4, q_bits=6)
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.nn as nn

PresetName = str


def _uniform_predicate(bits: int, group_size: int) -> Callable:
    def predicate(path: str, module: nn.Module):
        if isinstance(module, nn.Linear):
            return {"group_size": group_size, "bits": bits}
        return False

    return predicate


def _mixed_predicate(bits: int, q_bits: int, group_size: int) -> Callable:
    """Higher precision for the q_proj noise channels; uniform for the rest."""

    def predicate(path: str, module: nn.Module):
        if not isinstance(module, nn.Linear):
            return False
        if path.endswith("self_attn.q_proj"):
            return {"group_size": group_size, "bits": q_bits}
        return {"group_size": group_size, "bits": bits}

    return predicate


def apply_quant(
    model: nn.Module,
    preset: PresetName = "uniform",
    bits: int = 4,
    group_size: int = 64,
    q_bits: int = 6,
    predicate: Optional[Callable] = None,
) -> dict:
    """Quantize `model` in-place using the chosen preset.

    Returns a small dict describing what was applied — useful for logging
    into the converted checkpoint's `config.json`.
    """
    if predicate is None:
        if preset == "uniform":
            predicate = _uniform_predicate(bits=bits, group_size=group_size)
            meta = {"preset": "uniform", "bits": bits, "group_size": group_size}
        elif preset == "mixed":
            predicate = _mixed_predicate(bits=bits, q_bits=q_bits, group_size=group_size)
            meta = {
                "preset": "mixed",
                "bits": bits,
                "q_bits": q_bits,
                "group_size": group_size,
            }
        else:
            raise ValueError(f"Unknown quant preset: {preset}")
    else:
        meta = {"preset": "custom", "group_size": group_size, "bits": bits}

    nn.quantize(model, class_predicate=predicate)
    return meta
