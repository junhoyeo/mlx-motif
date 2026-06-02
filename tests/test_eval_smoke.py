"""
Unit tests for eval_smoke.py pure-logic helpers.

These tests run WITHOUT a model or MLX installed — they only exercise
the parsing/aggregation helpers and report assembly logic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any  # noqa: F401

# ---------------------------------------------------------------------------
# Loader — mirrors the pattern used in test_benchmark_sweep.py
# ---------------------------------------------------------------------------


def _load_eval_smoke():
    spec = importlib.util.spec_from_file_location("eval_smoke", Path("scripts/eval_smoke.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_smoke"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# aggregate_tok_per_sec
# ---------------------------------------------------------------------------


def test_aggregate_tok_per_sec_empty():
    m = _load_eval_smoke()
    result = m.aggregate_tok_per_sec([])
    assert result == {"mean": None, "min": None, "max": None}


def test_aggregate_tok_per_sec_single():
    m = _load_eval_smoke()
    result = m.aggregate_tok_per_sec([42.0])
    assert result["mean"] == 42.0
    assert result["min"] == 42.0
    assert result["max"] == 42.0


def test_aggregate_tok_per_sec_multiple():
    m = _load_eval_smoke()
    result = m.aggregate_tok_per_sec([10.0, 20.0, 30.0])
    assert result["mean"] == 20.0
    assert result["min"] == 10.0
    assert result["max"] == 30.0


def test_aggregate_tok_per_sec_all_equal():
    m = _load_eval_smoke()
    result = m.aggregate_tok_per_sec([5.5, 5.5, 5.5])
    assert result["mean"] == 5.5
    assert result["min"] == 5.5
    assert result["max"] == 5.5


# ---------------------------------------------------------------------------
# assemble_report
# ---------------------------------------------------------------------------


def _fake_generations() -> list[dict[str, Any]]:
    return [
        {"prompt": "Hello", "output": "world", "tokens": 10, "elapsed_s": 0.5, "tok_per_sec": 20.0},
        {"prompt": "Foo", "output": "bar", "tokens": 5, "elapsed_s": 0.25, "tok_per_sec": None},
        {"prompt": "MLX", "output": "fast", "tokens": 8, "elapsed_s": 0.4, "tok_per_sec": 20.0},
    ]


def _fake_host() -> dict[str, Any]:
    return {"hostname": "test-host", "chip": "M1 Max", "os": "macOS"}


def test_assemble_report_schema_field():
    m = _load_eval_smoke()
    report = m.assemble_report(
        model="/path/to/model",
        generations=_fake_generations(),
        perplexity=12.345,
        peak_memory_gb=2.1,
        host=_fake_host(),
    )
    assert report["schema"] == "eval-smoke-v1"


def test_assemble_report_has_disclaimer():
    m = _load_eval_smoke()
    report = m.assemble_report(
        model="/path/to/model",
        generations=_fake_generations(),
        perplexity=12.345,
        peak_memory_gb=2.1,
        host=_fake_host(),
    )
    assert "disclaimer" in report
    assert len(report["disclaimer"]) > 10
    # Must label it as a local smoke — not certified.
    assert "LOCAL SMOKE" in report["disclaimer"]
    assert "NOT certified" in report["disclaimer"]


def test_assemble_report_timestamp_placeholder():
    m = _load_eval_smoke()
    report = m.assemble_report(
        model="/path/to/model",
        generations=_fake_generations(),
        perplexity=None,
        peak_memory_gb=None,
        host=_fake_host(),
    )
    # Real runner replaces this; unit tests get the placeholder.
    assert report["timestamp"] == "PLACEHOLDER"


def test_assemble_report_tok_per_sec_summary_filters_none():
    m = _load_eval_smoke()
    # One generation has tok_per_sec=None — should be excluded from aggregation.
    report = m.assemble_report(
        model="/tmp/model",
        generations=_fake_generations(),  # [20.0, None, 20.0]
        perplexity=10.0,
        peak_memory_gb=1.0,
        host=_fake_host(),
    )
    summary = report["tok_per_sec_summary"]
    assert summary["mean"] == 20.0
    assert summary["min"] == 20.0
    assert summary["max"] == 20.0


def test_assemble_report_empty_generations():
    m = _load_eval_smoke()
    report = m.assemble_report(
        model="/tmp/model",
        generations=[],
        perplexity=None,
        peak_memory_gb=None,
        host=_fake_host(),
    )
    assert report["generations"] == []
    assert report["tok_per_sec_summary"] == {"mean": None, "min": None, "max": None}


def test_assemble_report_fields_present():
    m = _load_eval_smoke()
    report = m.assemble_report(
        model="/tmp/motif",
        generations=_fake_generations(),
        perplexity=8.5,
        peak_memory_gb=3.2,
        host=_fake_host(),
    )
    for key in (
        "schema",
        "disclaimer",
        "model",
        "timestamp",
        "host",
        "generations",
        "tok_per_sec_summary",
        "perplexity",
        "peak_memory_gb",
    ):
        assert key in report, f"missing key: {key}"
    assert report["model"] == "/tmp/motif"
    assert report["perplexity"] == 8.5
    assert report["peak_memory_gb"] == 3.2


# ---------------------------------------------------------------------------
# CLI skip path — no model dir, no --require-model → exit 0
# ---------------------------------------------------------------------------


def test_cli_skips_gracefully_when_model_missing(tmp_path: Path):
    """The CLI must exit 0 when model dir is absent and --require-model is unset."""
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/eval_smoke.py",
            "--model",
            str(tmp_path / "does-not-exist"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, f"Expected exit 0, got {proc.returncode}:\n{proc.stderr}"
    assert "skipping" in proc.stderr.lower() or "not found" in proc.stderr.lower()


def test_cli_require_model_exits_nonzero_when_missing(tmp_path: Path):
    """With --require-model the CLI must exit non-zero when model dir is absent."""
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/eval_smoke.py",
            "--model",
            str(tmp_path / "does-not-exist"),
            "--require-model",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0, "Expected non-zero exit when model is missing + --require-model"
    assert "ERROR" in proc.stderr or "not found" in proc.stderr.lower()
