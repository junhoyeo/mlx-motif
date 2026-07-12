"""Security regression tests for checkpoint shard path resolution."""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

import mlx_motif.convert as convert_module
from mlx_motif.convert import (
    _copy_tokenizer_files,
    _iter_safetensors,
    _resolve_hf_path,
    convert,
)


def _write_index(checkpoint, shard: str) -> None:
    _write_index_metadata(checkpoint, {"weight_map": {"model.weight": shard}})


def _write_index_metadata(checkpoint, metadata) -> None:
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(metadata))


def test_convert_rejects_invalid_config_before_weight_lookup(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "motif",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 0,
                "num_key_value_heads": 4,
                "vocab_size": 128,
            }
        )
    )

    with pytest.raises(ValueError, match="num_attention_heads must be greater than zero"):
        convert(str(checkpoint), str(tmp_path / "output"))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "must contain a weight_map object"),
        ({"weight_map": []}, "must contain a weight_map object"),
        ({"weight_map": {}}, "weight_map must not be empty"),
        ({"weight_map": {"model.weight": 42}}, "non-empty relative path"),
        ({"weight_map": {"model.weight": ""}}, "non-empty relative path"),
    ],
)
def test_index_rejects_invalid_schema_before_opening_shards(
    tmp_path, monkeypatch, metadata, message
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_index_metadata(checkpoint, metadata)

    def _unexpected_safe_open(*args, **kwargs):
        raise AssertionError("safe_open must not run for an invalid index")

    monkeypatch.setattr(convert_module, "safe_open", _unexpected_safe_open)
    with pytest.raises(ValueError, match=message):
        list(_iter_safetensors(checkpoint))


def test_index_rejects_absolute_shard_path(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    _write_index(checkpoint, str(outside))

    with pytest.raises(ValueError, match="relative path"):
        list(_iter_safetensors(checkpoint))


def test_index_rejects_parent_traversal(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "outside.safetensors").touch()
    _write_index(checkpoint, "../outside.safetensors")

    with pytest.raises(ValueError, match="must not contain '\\.\\.'"):
        list(_iter_safetensors(checkpoint))


def test_index_rejects_symlink_escape(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.touch()
    (checkpoint / "linked.safetensors").symlink_to(outside)
    _write_index(checkpoint, "linked.safetensors")

    with pytest.raises(ValueError, match="outside checkpoint root"):
        list(_iter_safetensors(checkpoint))


def test_tokenizer_copy_rejects_local_symlink_escape(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    checkpoint.mkdir()
    output.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": true}')
    (checkpoint / "tokenizer.json").symlink_to(outside)

    with pytest.raises(ValueError, match="outside checkpoint root"):
        _copy_tokenizer_files(checkpoint, output)

    assert not (output / "tokenizer.json").exists()


def test_index_rejects_non_file_shard(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "shard-dir").mkdir()
    _write_index(checkpoint, "shard-dir")

    with pytest.raises(ValueError, match="regular file"):
        list(_iter_safetensors(checkpoint))


def test_index_allows_nested_shard_within_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    shard_dir = checkpoint / "shards"
    shard_dir.mkdir(parents=True)
    save_file(
        {"model.weight": np.array([1.0, 2.0], dtype=np.float32)}, shard_dir / "part.safetensors"
    )
    _write_index(checkpoint, "shards/part.safetensors")

    weights = list(_iter_safetensors(checkpoint))

    assert [name for name, _ in weights] == ["model.weight"]
    assert weights[0][1].shape == (2,)


def _hf_snapshot_layout(tmp_path):
    repo_cache = tmp_path / "hub" / "models--org--model"
    snapshot = repo_cache / "snapshots" / "abc123"
    blobs = repo_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    return repo_cache, snapshot, blobs


def test_remote_hf_snapshot_allows_same_repo_blob_symlink(tmp_path, monkeypatch):
    _, snapshot, blobs = _hf_snapshot_layout(tmp_path)
    blob = blobs / "deadbeef"
    save_file({"model.weight": np.array([1.0], dtype=np.float32)}, blob)
    (snapshot / "model-00001-of-00001.safetensors").symlink_to(blob)
    index_blob = blobs / "indexbeef"
    index_blob.write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}})
    )
    (snapshot / "model.safetensors.index.json").symlink_to(index_blob)
    tokenizer_blob = blobs / "tokenizerbeef"
    tokenizer_blob.write_text('{"version": "1.0"}')
    (snapshot / "tokenizer.json").symlink_to(tokenizer_blob)
    monkeypatch.setattr(convert_module, "snapshot_download", lambda *args, **kwargs: str(snapshot))

    source = _resolve_hf_path("org/model")
    weights = list(
        _iter_safetensors(
            source.path,
            allowed_external_roots=source.allowed_external_roots,
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    _copy_tokenizer_files(
        source.path,
        output,
        allowed_external_roots=source.allowed_external_roots,
    )

    assert [name for name, _ in weights] == ["model.weight"]
    assert (output / "tokenizer.json").read_text() == '{"version": "1.0"}'


@pytest.mark.parametrize("escape_kind", ["cache-sibling", "cross-repo"])
def test_remote_hf_snapshot_rejects_non_blob_symlink_target(tmp_path, monkeypatch, escape_kind):
    repo_cache, snapshot, _ = _hf_snapshot_layout(tmp_path)
    if escape_kind == "cache-sibling":
        target_dir = repo_cache / "refs"
    else:
        target_dir = tmp_path / "hub" / "models--other--model" / "blobs"
    target_dir.mkdir(parents=True)
    target = target_dir / "escape"
    target.touch()
    (snapshot / "model-00001-of-00001.safetensors").symlink_to(target)
    _write_index(snapshot, "model-00001-of-00001.safetensors")
    monkeypatch.setattr(convert_module, "snapshot_download", lambda *args, **kwargs: str(snapshot))

    source = _resolve_hf_path("org/model")
    with pytest.raises(ValueError, match="outside allowed checkpoint roots"):
        list(
            _iter_safetensors(
                source.path,
                allowed_external_roots=source.allowed_external_roots,
            )
        )
