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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _CheckpointSource:
    path: Path
    allowed_external_roots: tuple[Path, ...] = ()


def _hf_repo_blobs_root(snapshot_path: Path) -> Path:
    """Return the one external root trusted for an HF cache snapshot."""
    snapshot = snapshot_path.resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError(f"Hugging Face snapshot must be a directory: {snapshot_path}")

    snapshots_dir = snapshot.parent
    repo_cache = snapshots_dir.parent
    if snapshots_dir.name != "snapshots" or not repo_cache.name.startswith("models--"):
        raise ValueError(
            "Hugging Face download did not resolve to a models--*/snapshots/<revision> "
            f"cache directory: {snapshot_path}"
        )

    blobs = (repo_cache / "blobs").resolve(strict=True)
    canonical_repo_cache = repo_cache.resolve(strict=True)
    if not blobs.is_dir() or blobs.parent != canonical_repo_cache:
        raise ValueError(f"Invalid Hugging Face cache blobs directory: {blobs}")
    return blobs


def _resolve_hf_path(hf_path: str) -> _CheckpointSource:
    p = Path(hf_path)
    if p.exists():
        return _CheckpointSource(path=p)

    snapshot = Path(snapshot_download(hf_path, allow_patterns=_HF_DOWNLOAD_PATTERNS))
    return _CheckpointSource(
        path=snapshot,
        allowed_external_roots=(_hf_repo_blobs_root(snapshot),),
    )


def _load_checkpoint_config(
    hf_dir: Path,
    *,
    allowed_external_roots: tuple[Path, ...] = (),
) -> dict:
    config_path = _resolve_checkpoint_file(
        hf_dir,
        "config.json",
        source="Checkpoint config",
        allowed_external_roots=allowed_external_roots,
    )
    cfg = json.loads(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint config must contain a JSON object")
    return cfg


def _load_args(
    hf_dir: Path,
    *,
    allowed_external_roots: tuple[Path, ...] = (),
) -> ModelArgs:
    cfg = _load_checkpoint_config(
        hf_dir,
        allowed_external_roots=allowed_external_roots,
    )
    cfg.setdefault("model_type", "motif")
    return ModelArgs.from_dict(cfg)


def _resolve_checkpoint_file(
    checkpoint_root: Path,
    relative_path: str,
    *,
    source: str,
    allowed_external_roots: tuple[Path, ...] = (),
) -> Path:
    """Resolve an untrusted checkpoint-relative filename without escaping its root."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{source} must name a non-empty relative path; got {relative_path!r}")
    if "\0" in relative_path:
        raise ValueError(f"{source} must not contain a null byte; got {relative_path!r}")

    shard_path = Path(relative_path)
    if shard_path.is_absolute():
        raise ValueError(f"{source} must use a relative path; got {relative_path!r}")
    if ".." in shard_path.parts:
        raise ValueError(f"{source} must not contain '..'; got {relative_path!r}")

    root = checkpoint_root.resolve(strict=True)
    try:
        resolved = (root / shard_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{source} does not exist: {relative_path!r}") from exc

    allowed_roots = (root,) + tuple(
        external_root.resolve(strict=True) for external_root in allowed_external_roots
    )
    if not any(resolved.is_relative_to(allowed_root) for allowed_root in allowed_roots):
        boundary = "allowed checkpoint roots" if allowed_external_roots else "checkpoint root"
        raise ValueError(f"{source} resolves outside {boundary}: {relative_path!r}")
    if not resolved.is_file():
        raise ValueError(f"{source} must resolve to a regular file; got {relative_path!r}")
    return resolved


def _iter_safetensors(
    hf_dir: Path,
    *,
    allowed_external_roots: tuple[Path, ...] = (),
):
    """Yield `(name, mx.array)` for every weight, walking shards if present."""
    checkpoint_root = hf_dir.resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise ValueError(f"Checkpoint root must be a directory: {hf_dir}")

    index = checkpoint_root / "model.safetensors.index.json"
    if index.exists():
        safe_index = _resolve_checkpoint_file(
            checkpoint_root,
            "model.safetensors.index.json",
            source="Safetensors index",
            allowed_external_roots=allowed_external_roots,
        )
        meta = json.loads(safe_index.read_text())
        if not isinstance(meta, dict) or not isinstance(meta.get("weight_map"), dict):
            raise ValueError("Safetensors index must contain a weight_map object")
        if not meta["weight_map"]:
            raise ValueError("Safetensors index weight_map must not be empty")
        shards = sorted(
            {
                _resolve_checkpoint_file(
                    checkpoint_root,
                    shard,
                    source="Safetensors index shard",
                    allowed_external_roots=allowed_external_roots,
                )
                for shard in meta["weight_map"].values()
            }
        )
    elif (checkpoint_root / "model.safetensors").exists():
        shards = [
            _resolve_checkpoint_file(
                checkpoint_root,
                "model.safetensors",
                source="Safetensors model",
                allowed_external_roots=allowed_external_roots,
            )
        ]
    else:
        raise FileNotFoundError(f"No safetensors found under {checkpoint_root}")

    for shard in shards:
        with safe_open(str(shard), framework="numpy") as f:
            for k in f.keys():
                yield k, mx.array(f.get_tensor(k))


def _copy_tokenizer_files(
    checkpoint_root: Path,
    out: Path,
    *,
    allowed_external_roots: tuple[Path, ...] = (),
) -> None:
    """Copy optional tokenizer metadata without following untrusted escapes."""
    root = checkpoint_root.resolve(strict=True)
    for fname in _TOKENIZER_FILES:
        candidate = root / fname
        if not candidate.exists():
            continue
        safe_candidate = _resolve_checkpoint_file(
            root,
            fname,
            source="Checkpoint tokenizer file",
            allowed_external_roots=allowed_external_roots,
        )
        shutil.copy2(safe_candidate, out / fname)


def _load_hf_weights(
    hf_dir: Path,
    target_dtype: mx.Dtype,
    *,
    allowed_external_roots: tuple[Path, ...] = (),
) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for name, arr in _iter_safetensors(
        hf_dir,
        allowed_external_roots=allowed_external_roots,
    ):
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
    source = _resolve_hf_path(hf_path)
    src = source.path
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)

    args = _load_args(
        src,
        allowed_external_roots=source.allowed_external_roots,
    )
    model = Model(args)

    target_dtype = getattr(mx, dtype)
    weights = _load_hf_weights(
        src,
        target_dtype,
        allowed_external_roots=source.allowed_external_roots,
    )
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

    cfg = _load_checkpoint_config(
        src,
        allowed_external_roots=source.allowed_external_roots,
    )
    cfg["model_type"] = "motif"
    if quantize:
        _apply_quantization_config(cfg, q_group_size, q_bits, quant_meta)
    save_config(cfg, out / "config.json")

    _copy_tokenizer_files(
        src,
        out,
        allowed_external_roots=source.allowed_external_roots,
    )

    return out
