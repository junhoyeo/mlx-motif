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
- `mlp_lowbit` — drops the three MLP projections (`gate_proj`, `up_proj`,
  `down_proj`) to 3-bit / group_size=32; everything else stays at the
  default (4-bit / group_size=64). **Validated as a NEGATIVE result on
  M1 Max** at the Motif 12.7B shape: the +31% per-call microbench win on
  q3+gs=32 vs q4+gs=64 does NOT carry to end-to-end decode (mlp_lowbit
  measured **31% slower at 500-token prompt**, ~parity at 3k tokens), and
  PPL regresses by +1.3 (~+10% relative). Two costs absorb the
  per-call win: extra scale/bias overhead (~+800 MB resident on the
  12.7B model from doubled scale entries at gs=32), and Q3 accumulator
  precision requirements that slow down chained dispatches. **Kept in
  the codebase as opt-in infrastructure** — the preset wiring may still
  win on different hardware (smaller L1, different bandwidth/compute
  ratios) or on other models/shapes.

Usage:

    from mlx_motif.quant import apply_quant
    apply_quant(model, preset="mlp_lowbit")           # mlp at q3/gs=32
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


_MLP_PROJ_NAMES = (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")


def _mlp_lowbit_predicate(
    bits: int, group_size: int, mlp_bits: int, mlp_group_size: int,
) -> Callable:
    """Lower-bit MLP projections; everything else at the (bits, group_size) default."""

    def predicate(path: str, module: nn.Module):
        if not isinstance(module, nn.Linear):
            return False
        if any(path.endswith(name) for name in _MLP_PROJ_NAMES):
            return {"group_size": mlp_group_size, "bits": mlp_bits}
        return {"group_size": group_size, "bits": bits}

    return predicate


def apply_quant(
    model: nn.Module,
    preset: PresetName = "uniform",
    bits: int = 4,
    group_size: int = 64,
    q_bits: int = 6,
    mlp_bits: int = 3,
    mlp_group_size: int = 32,
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
        elif preset == "mlp_lowbit":
            predicate = _mlp_lowbit_predicate(
                bits=bits, group_size=group_size,
                mlp_bits=mlp_bits, mlp_group_size=mlp_group_size,
            )
            preset_meta = {
                "preset": "mlp_lowbit",
                "mlp_bits": mlp_bits, "mlp_group_size": mlp_group_size,
            }
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
