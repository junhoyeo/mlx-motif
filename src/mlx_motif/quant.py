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

from collections.abc import Callable

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
    predicate: Callable | None = None,
) -> dict:
    """Quantize `model` in-place using the chosen preset.

    Returns a dict shaped for `config["quantization"]` that mlx-lm's loader
    can read back. The top-level `bits`/`group_size` are the *defaults*;
    per-module overrides go in keys named after their dotted path
    (`model.layers.0.self_attn.q_proj`) — that's the contract enforced by
    `mlx_lm.utils.load_model._quantize`.
    """
    if predicate is None:
        if preset == "uniform":
            predicate = _uniform_predicate(bits=bits, group_size=group_size)
            preset_meta = {"preset": "uniform"}
        elif preset == "mixed":
            predicate = _mixed_predicate(bits=bits, q_bits=q_bits, group_size=group_size)
            preset_meta = {"preset": "mixed", "q_bits": q_bits}
        else:
            raise ValueError(f"Unknown quant preset: {preset}")
    else:
        preset_meta = {"preset": "custom"}

    nn.quantize(model, class_predicate=predicate)

    # Walk the model and emit per-module settings for any QuantizedLinear that
    # diverges from the top-level defaults. This is what the mlx-lm loader
    # reads to rebuild the model with matching shapes.
    overrides: dict = {}
    for path, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        if module.bits != bits or module.group_size != group_size:
            overrides[path] = {"group_size": module.group_size, "bits": module.bits}

    return {
        "group_size": group_size,
        "bits": bits,
        **preset_meta,
        **overrides,
    }
