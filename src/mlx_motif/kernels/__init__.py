"""
Custom Metal kernels for the Motif port.

Each kernel ships with a pure-MLX reference (`*_reference`) and a parametric
correctness test under ``tests/``. Disable all kernels and fall back to the
references via ``MLX_MOTIF_DISABLE_KERNELS=1`` (correctness-only mode).

Layout (split by domain — see each module for design notes):

    attention.py — sdpa_dual_v, sdpa_dual_v_q4  (shared-QK dual-V SDPA)
    gda.py       — gda_post, gda_post_split     (GDA post-attention reduction)
    mlp.py       — polynorm, _dequant_probe     (test helper)

All kernels here are production paths used at decode. Negative-result
experiments (polynorm_mul, sdpa_dual_v_2pass, qmv_dual_q4, gda_decode) and
their bench results are documented in ``docs/experiments/`` — kept as
calibrated priors for re-trying on different hardware, but removed from
the codebase to keep the build surface small.
"""

from .attention import (
    sdpa_dual_v,
    sdpa_dual_v_q4,
    sdpa_dual_v_q4_reference,
    sdpa_dual_v_reference,
)
from .gda import (
    gda_post,
    gda_post_reference,
    gda_post_split,
    gda_post_split_reference,
)
from .mlp import (
    _dequant_probe,  # noqa: F401  — kept importable for tests/test_dequant_probe.py; intentionally NOT in __all__ (see note below)
    polynorm,
    polynorm_reference,
)

__all__ = [
    "polynorm",
    "polynorm_reference",
    "gda_post",
    "gda_post_reference",
    "gda_post_split",
    "gda_post_split_reference",
    "sdpa_dual_v",
    "sdpa_dual_v_reference",
    "sdpa_dual_v_q4",
    "sdpa_dual_v_q4_reference",
]
# `_dequant_probe` is intentionally NOT in __all__ — the underscore prefix
# marks it test-internal. Tests import it via explicit name
# (`from mlx_motif.kernels import _dequant_probe`), which works regardless
# of __all__; only `from mlx_motif.kernels import *` is affected.
