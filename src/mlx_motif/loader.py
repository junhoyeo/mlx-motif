"""
Loader that wires the mlx_motif `Model` class into mlx-lm's loading machinery.

mlx-lm's stock `load()` only resolves models under `mlx_lm.models.*`. Until
the upstream registers `motif`, this module supplies a thin `load()` that
injects our class via the `get_model_classes` hook on `mlx_lm.utils.load_model`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.nn as nn
from mlx_lm.utils import TokenizerWrapper, load_tokenizer
from mlx_lm.utils import load_model as _mlx_lm_load_model

from mlx_motif.model import Model, ModelArgs


def _get_motif_classes(config: dict):
    return Model, ModelArgs


def _register_extra_eos_from_generation_config(
    tokenizer: TokenizerWrapper,
    path: Path,
) -> None:
    """HF ``generation_config.json`` may list multiple EOS token ids
    (Motif: ``[<|endoftext|>, <|endofturn|>]``), but ``tokenizer.eos_token_id``
    is a single int. mlx-lm's ``stream_generate`` only stops on ids in
    ``tokenizer._eos_token_ids``, so without this any extra EOS tokens leak
    as text into the stream (verified end-to-end on Motif 12.7B — the
    ``<|endofturn|>`` token was appearing in the SSE content delta).
    """
    cfg_path = path / "generation_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError:
        return
    eos = cfg.get("eos_token_id")
    if eos is None:
        return
    if isinstance(eos, int):
        eos = [eos]
    for tid in eos:
        try:
            tokenizer.add_eos_token(str(int(tid)))
        except (ValueError, TypeError):
            # Bad id — skip rather than failing the whole load.
            pass


def load(
    path: str | Path,
    fuse_qkv: bool = True,
) -> tuple[nn.Module, TokenizerWrapper]:
    """Load a converted MLX Motif checkpoint and its tokenizer.

    Args:
        path: directory containing `model.safetensors*` and tokenizer files
        fuse_qkv: if True (default), call `model.fuse_qkv()` after loading
            so the 3 grouped-attn projections collapse into one QuantizedLinear
            (~+10% on the per-layer projection cost).
    """
    path = Path(path)
    model, _ = _mlx_lm_load_model(path, get_model_classes=_get_motif_classes)
    if fuse_qkv and hasattr(model, "fuse_qkv"):
        model.fuse_qkv()
    tokenizer = load_tokenizer(path)
    _register_extra_eos_from_generation_config(tokenizer, path)
    return model, tokenizer
