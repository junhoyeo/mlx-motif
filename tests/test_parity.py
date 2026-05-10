"""
Numerical parity vs the Hugging Face reference.

Runs the same prompt through both implementations and asserts the next-token
distributions agree within bf16 noise. Skipped by default — opt in with the
env var `MLX_MOTIF_PARITY=<hf_dir>` pointing at a downloaded HF checkpoint.

Example:

    MLX_MOTIF_PARITY=$PWD/.models/Motif-2.6B \
    MLX_MOTIF_PARITY_MLX=$PWD/.models/motif-2.6b-mlx \
    uv run pytest tests/test_parity.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

HF_DIR = os.environ.get("MLX_MOTIF_PARITY")
MLX_DIR = os.environ.get("MLX_MOTIF_PARITY_MLX")

pytestmark = pytest.mark.skipif(
    not (HF_DIR and MLX_DIR),
    reason="set MLX_MOTIF_PARITY (hf dir) and MLX_MOTIF_PARITY_MLX (mlx dir)",
)


def _hf_logits(hf_dir: str, input_ids: list[int]) -> np.ndarray:
    """Run the HF reference model on CPU in fp32 to get a noise floor."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([input_ids], dtype=torch.long))
    return out.logits[0, -1].cpu().numpy()


def _mlx_logits(mlx_dir: str, input_ids: list[int]) -> np.ndarray:
    from mlx_motif import load

    model, _ = load(mlx_dir)
    logits = model(mx.array([input_ids]))
    return np.array(logits[0, -1].astype(mx.float32))


def test_parity_top_token_agrees():
    """The argmax of the final-position logits must match between HF and MLX."""
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    hf = _hf_logits(HF_DIR, input_ids)
    mlx = _mlx_logits(MLX_DIR, input_ids)

    hf_top = int(np.argmax(hf))
    mlx_top = int(np.argmax(mlx))

    print(f"HF  argmax: {hf_top}, top logit: {hf[hf_top]:.4f}")
    print(f"MLX argmax: {mlx_top}, top logit: {mlx[mlx_top]:.4f}")
    print(f"max abs diff: {np.max(np.abs(hf - mlx)):.4f}")
    print(f"hf  top-5 ids: {np.argsort(-hf)[:5].tolist()}")
    print(f"mlx top-5 ids: {np.argsort(-mlx)[:5].tolist()}")

    assert hf_top == mlx_top, (
        f"Top token mismatch: HF={hf_top} vs MLX={mlx_top}. "
        "Likely lambda_init / RoPE / V-reshape divergence."
    )


def test_parity_top5_overlap():
    """Top-5 next-token candidates should overlap by >= 4."""
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    hf = _hf_logits(HF_DIR, input_ids)
    mlx = _mlx_logits(MLX_DIR, input_ids)

    hf_top5 = set(np.argsort(-hf)[:5].tolist())
    mlx_top5 = set(np.argsort(-mlx)[:5].tolist())
    overlap = len(hf_top5 & mlx_top5)
    assert overlap >= 4, f"Top-5 overlap too low: {overlap} ({hf_top5} vs {mlx_top5})"
