"""
Loader that wires the mlx_motif `Model` class into mlx-lm's loading machinery.

mlx-lm's stock `load()` only resolves models under `mlx_lm.models.*`. Until
the upstream registers `motif`, this module supplies a thin `load()` that
injects our class via the `get_model_classes` hook on `mlx_lm.utils.load_model`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mlx.nn as nn
from mlx_lm.utils import TokenizerWrapper, load_tokenizer
from mlx_lm.utils import load_model as _mlx_lm_load_model

from mlx_motif.model import Model, ModelArgs


def _get_motif_classes(config: dict):
    return Model, ModelArgs


def load(path: str | Path) -> Tuple[nn.Module, TokenizerWrapper]:
    """Load a converted MLX Motif checkpoint and its tokenizer."""
    path = Path(path)
    model, _ = _mlx_lm_load_model(path, get_model_classes=_get_motif_classes)
    tokenizer = load_tokenizer(path)
    return model, tokenizer
