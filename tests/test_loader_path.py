"""Path resolution for `mlx_motif.load` — local dir vs Hugging Face repo id.

Regression guard: `load()` used to pass its argument straight to
`Path(...)` and mlx-lm's `load_model`, so a Hub repo id like
``"junhoyeo/Motif-2.6B-MLX-q4"`` was looked up as the local path
``./junhoyeo/Motif-2.6B-MLX-q4/config.json`` and failed with
``FileNotFoundError``. `_resolve_model_path` now snapshot-downloads repo ids
while leaving existing local directories untouched. These tests exercise the
resolver directly (no network — `snapshot_download` is monkeypatched).
"""

import json
from pathlib import Path

import pytest

from mlx_motif import loader


def test_existing_local_dir_is_used_as_is(tmp_path, monkeypatch):
    def _boom(*a, **k):  # network must not be touched for a local path
        raise AssertionError("snapshot_download called for an existing local dir")

    monkeypatch.setattr(loader, "snapshot_download", _boom)
    resolved = loader._resolve_model_path(tmp_path)
    assert resolved == Path(tmp_path)


def test_repo_id_is_snapshot_downloaded(tmp_path, monkeypatch):
    calls = {}

    def _fake_download(repo, revision=None, allow_patterns=None):
        calls["repo"] = repo
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        return str(tmp_path)

    monkeypatch.setattr(loader, "snapshot_download", _fake_download)
    resolved = loader._resolve_model_path("junhoyeo/Motif-2.6B-MLX-q4", revision="main")

    assert resolved == Path(tmp_path)
    assert calls["repo"] == "junhoyeo/Motif-2.6B-MLX-q4"
    assert calls["revision"] == "main"
    # The allow-list must cover every file our checkpoints ship.
    patterns = calls["allow_patterns"]
    assert "*.json" in patterns  # config / tokenizer / index / generation_config
    assert "model*.safetensors" in patterns  # sharded or single weights
    assert "*.txt" in patterns  # merges.txt
    assert "*.jinja" in patterns  # chat_template.jinja


def test_allow_patterns_match_a_real_checkpoint_layout():
    """The allow-list must admit each filename a converted checkpoint contains,
    so a Hub download is complete (no missing config/tokenizer/weights)."""
    import fnmatch

    checkpoint_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
    ]
    for name in checkpoint_files:
        assert any(fnmatch.fnmatch(name, pat) for pat in loader._HUB_ALLOW_PATTERNS), (
            f"{name} would not be downloaded by the allow-list"
        )


def test_load_validates_config_before_upstream_weight_loading(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
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

    def _unexpected_upstream_load(*args, **kwargs):
        raise AssertionError("upstream weight loading must not run for an invalid config")

    monkeypatch.setattr(loader, "_mlx_lm_load_model", _unexpected_upstream_load)

    with pytest.raises(ValueError, match="num_attention_heads must be greater than zero"):
        loader.load(tmp_path)
