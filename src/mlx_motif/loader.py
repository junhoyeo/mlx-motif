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
from huggingface_hub import snapshot_download
from mlx_lm.utils import TokenizerWrapper, load_tokenizer
from mlx_lm.utils import load_model as _mlx_lm_load_model

from mlx_motif.model import Model, ModelArgs

# Mirrors mlx-lm's `_download` allow-list so a Hub repo id pulls exactly the
# files our checkpoints ship (weights, config, tokenizer, chat template).
_HUB_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def _get_motif_classes(config: dict):
    return Model, ModelArgs


def _resolve_model_path(path: str | Path, revision: str | None = None) -> Path:
    """Resolve `path` to a local directory, downloading from the Hub if needed.

    A local directory that exists is used as-is; anything else is treated as a
    Hugging Face repo id and fetched via `snapshot_download` (matching what
    mlx-lm's own `load()` does). Without this, `load("org/model")` would look
    for `./org/model/config.json` on disk and fail.
    """
    p = Path(path)
    if p.exists():
        return p
    return Path(
        snapshot_download(
            str(path),
            revision=revision,
            allow_patterns=_HUB_ALLOW_PATTERNS,
        )
    )


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
    revision: str | None = None,
) -> tuple[nn.Module, TokenizerWrapper]:
    """Load a converted MLX Motif checkpoint and its tokenizer.

    Args:
        path: a local directory containing `model.safetensors*` + tokenizer
            files, OR a Hugging Face repo id (e.g.
            ``"junhoyeo/Motif-2.6B-MLX-q4"``), which is downloaded on first use.
        fuse_qkv: if True (default), call `model.fuse_qkv()` after loading
            so the 3 grouped-attn projections collapse into one QuantizedLinear
            (~+10% on the per-layer projection cost).
        revision: optional Hub revision (branch, tag, or commit) when `path`
            is a repo id.
    """
    path = _resolve_model_path(path, revision=revision)
    model, _ = _mlx_lm_load_model(path, get_model_classes=_get_motif_classes)
    if fuse_qkv and hasattr(model, "fuse_qkv"):
        model.fuse_qkv()
    tokenizer = load_tokenizer(path)
    _register_extra_eos_from_generation_config(tokenizer, path)
    return model, tokenizer
