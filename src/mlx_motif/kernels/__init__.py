"""
Custom Metal kernels for the Motif port.

Each kernel ships with a pure-MLX reference and a small validator. On the
M1/M1 Max chips there is a known `mx.fast.metal_kernel` correctness bug
(ml-explore/mlx#2205); we therefore numerically validate every kernel
against its reference and fall back to MLX ops if the kernel is disabled
via the `MLX_MOTIF_DISABLE_KERNELS` environment variable.

Public API:
    polynorm(x, weight, bias, eps) -> mx.array
    polynorm_reference(x, weight, bias, eps) -> mx.array
    polynorm_mul(gate, up, weight, bias, eps) -> mx.array
    polynorm_mul_reference(gate, up, weight, bias, eps) -> mx.array
    gda_post(merged, subln_weight, lambda_full, lambda_init, q_groups, gr, eps) -> mx.array
    gda_post_reference(...) -> mx.array
    gda_decode(q1, q2, k1, k2, v1, v2, subln_weight,
               lambda_full, lambda_init, gr, eps) -> mx.array
    gda_decode_reference(...) -> mx.array
    sdpa_dual_v(q, k, v1, v2, scale) -> mx.array     # NEW: shared QK, dual V
    sdpa_dual_v_reference(q, k, v1, v2, scale) -> mx.array
"""

from .mlp import (
    polynorm,
    polynorm_reference,
    polynorm_mul,
    polynorm_mul_reference,
    _dequant_probe,
    qmv_dual_q4,
    qmv_dual_q4_reference,
)
from .gda import (
    gda_post,
    gda_post_reference,
    gda_post_split,
    gda_post_split_reference,
    gda_decode,
    gda_decode_reference,
)
from .attention import (
    sdpa_dual_v,
    sdpa_dual_v_reference,
    sdpa_dual_v_2pass,
    sdpa_dual_v_q4,
    sdpa_dual_v_q4_reference,
)

__all__ = [
    "polynorm",
    "polynorm_reference",
    "polynorm_mul",
    "polynorm_mul_reference",
    "_dequant_probe",
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
