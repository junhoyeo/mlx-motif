"""
Convert a Hugging Face Motif checkpoint to the MLX layout.

Usage (programmatic):

    from mlx_motif.convert import convert

    convert(
        hf_path="Motif-Technologies/Motif-2.6B",
        out_path="./out/motif-2.6b-mlx",
        dtype="bfloat16",
        quantize=False,
    )

CLI:

    mlx-motif convert --hf-path <repo-or-dir> --out <dir> [--quantize --bits 4]
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm.utils import save_config, save_model
from safetensors import safe_open

from mlx_motif.model import Model, ModelArgs
from mlx_motif.quant import apply_quant

# Files copied verbatim so the converted dir works with `transformers.AutoTokenizer`.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)

_HF_DOWNLOAD_PATTERNS = [
    "*.json",
    "*.txt",
    "*.jinja",
    "*.safetensors",
    "*.safetensors.index.json",
]


def _resolve_hf_path(hf_path: str) -> Path:
    p = Path(hf_path)
    if p.exists():
        return p
    return Path(snapshot_download(hf_path, allow_patterns=_HF_DOWNLOAD_PATTERNS))


def _load_args(hf_dir: Path) -> ModelArgs:
    cfg = json.loads((hf_dir / "config.json").read_text())
    cfg.setdefault("model_type", "motif")
    return ModelArgs.from_dict(cfg)


def _iter_safetensors(hf_dir: Path):
    """Yield `(name, mx.array)` for every weight, walking shards if present."""
    index = hf_dir / "model.safetensors.index.json"
    if index.exists():
        meta = json.loads(index.read_text())
        shards = sorted(set(meta["weight_map"].values()))
    elif (hf_dir / "model.safetensors").exists():
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors found under {hf_dir}")

    for shard in shards:
        with safe_open(str(hf_dir / shard), framework="numpy") as f:
            for k in f.keys():
                yield k, mx.array(f.get_tensor(k))


def _load_hf_weights(hf_dir: Path, target_dtype: mx.Dtype) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for name, arr in _iter_safetensors(hf_dir):
        if "rotary_emb.inv_freq" in name:
            continue
        # Lambda parameters stay fp32 (numerical stability of exp/sub).
        if "lambda_" in name:
            weights[name] = arr.astype(mx.float32)
            continue
        if arr.dtype != target_dtype:
            arr = arr.astype(target_dtype)
        weights[name] = arr
    return weights


def _apply_quantization_config(
    cfg: dict,
    q_group_size: int,
    q_bits: int,
    quant_meta: dict,
) -> dict:
    """Write quantization metadata into a Motif config dict.

    The ``quantization`` block holds ONLY scalar ``group_size``/``bits`` — the
    fields mlx-lm reads. mlx-swift-lm's ``BaseConfiguration`` decodes every
    *other* key under ``quantization`` as a per-layer override dict, so a scalar
    such as ``preset: "uniform"`` (or ``q_bits``/``mlp_bits`` from the mixed /
    mlp_lowbit presets) makes the Swift loader fail with a ``typeMismatch``.
    Preset metadata therefore lives in ``quantization_config`` instead, which our
    own loader reads and mlx-swift-lm ignores.
    """
    cfg["quantization"] = {"group_size": q_group_size, "bits": q_bits}
    cfg["quantization_config"] = {
        "group_size": q_group_size,
        "bits": q_bits,
        **quant_meta,
    }
    return cfg


def convert(
    hf_path: str,
    out_path: str,
    dtype: str = "bfloat16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
    quant_preset: str = "uniform",
    q_proj_bits: int = 6,
    mlp_bits: int = 3,
    mlp_group_size: int = 32,
) -> Path:
    src = _resolve_hf_path(hf_path)
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)

    args = _load_args(src)
    model = Model(args)

    target_dtype = getattr(mx, dtype)
    weights = _load_hf_weights(src, target_dtype)
    model.load_weights(list(weights.items()), strict=False)

    quant_meta: dict = {}
    if quantize:
        quant_meta = apply_quant(
            model,
            preset=quant_preset,
            bits=q_bits,
            group_size=q_group_size,
            q_bits=q_proj_bits,
            mlp_bits=mlp_bits,
            mlp_group_size=mlp_group_size,
        )

    save_model(out, model)

    cfg = json.loads((src / "config.json").read_text())
    cfg["model_type"] = "motif"
    if quantize:
        _apply_quantization_config(cfg, q_group_size, q_bits, quant_meta)
    save_config(cfg, out / "config.json")

    for fname in _TOKENIZER_FILES:
        candidate = src / fname
        if candidate.exists():
            shutil.copy2(candidate, out / fname)

    return out
