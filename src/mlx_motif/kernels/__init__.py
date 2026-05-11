"""
Custom Metal kernels for the Motif port.

Each kernel ships with a pure-MLX reference (`*_reference`) and a parametric
correctness test under ``tests/``. Disable all kernels and fall back to the
references via ``MLX_MOTIF_DISABLE_KERNELS=1`` (correctness-only mode).

Layout (split by domain — see each module for design notes):

    attention.py — sdpa_dual_v*  (shared-QK dual-V SDPA family)
    gda.py       — gda_post*, gda_decode  (GDA post-attention reduction
                                           + legacy serial flash kernel)
    mlp.py       — polynorm, polynorm_mul, qmv_dual_q4, _dequant_probe

Status summary:

    Production (default-on):
        polynorm                  — fused MLP activation, always
        gda_post                  — fallback prefill / GQA-repeat path
        gda_post_split            — default decode post-attention reduction
        sdpa_dual_v               — default decode attention (fp16 cache)
        sdpa_dual_v_q4            — default decode attention (quant cache;
                                    auto-selected when cache is quantized)

    Opt-in via env (kept for A/B / debugging):
        gda_decode                — MLX_MOTIF_FLASH_DECODE=1 (legacy serial)

    Documented negative results (kept as in-tree references for future
    hardware A/B; NOT wired into MotifAttention/MotifMLP):
        polynorm_mul              — -7% e2e, lazy graph already fuses
        sdpa_dual_v_2pass         — slower than single-pass at our shapes
        qmv_dual_q4               — loses on M1 Max (x already in L1);
                                    may flip on M3/M4 with smaller L1

    Test helpers (private, prefix `_`):
        _dequant_probe            — bit-extract validation against
                                    mx.dequantize (used by tests/)

Each `*_reference` is the pure-MLX equivalent and is also exported so
callers can A/B against the kernel without env vars.
"""

from .attention import (
    sdpa_dual_v,
    sdpa_dual_v_2pass,
    sdpa_dual_v_q4,
    sdpa_dual_v_q4_reference,
    sdpa_dual_v_reference,
)
from .gda import (
    gda_decode,
    gda_decode_reference,
    gda_post,
    gda_post_reference,
    gda_post_split,
    gda_post_split_reference,
)
from .mlp import (
    _dequant_probe,  # noqa: F401  — kept importable for tests/test_dequant_probe.py; intentionally NOT in __all__ (see note below)
    polynorm,
    polynorm_mul,
    polynorm_mul_reference,
    polynorm_reference,
    qmv_dual_q4,
    qmv_dual_q4_reference,
)

__all__ = [
    "polynorm",
    "polynorm_reference",
    "polynorm_mul",
    "polynorm_mul_reference",
    "qmv_dual_q4",
    "qmv_dual_q4_reference",
    "gda_post",
    "gda_post_reference",
    "gda_post_split",
    "gda_post_split_reference",
    "gda_decode",
    "gda_decode_reference",
    "sdpa_dual_v",
    "sdpa_dual_v_reference",
    "sdpa_dual_v_2pass",
    "sdpa_dual_v_q4",
    "sdpa_dual_v_q4_reference",
]
# `_dequant_probe` is intentionally NOT in __all__ — the underscore prefix
# marks it test-internal. Tests import it via explicit name
# (`from mlx_motif.kernels import _dequant_probe`), which works regardless
# of __all__; only `from mlx_motif.kernels import *` is affected.
