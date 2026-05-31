"""Certification validator for benchmark sweep JSONs.

A sweep JSON is "certified" when it meets all of the following minimums:

    n_runs          >= 5      (enough repetitions for meaningful stdev)
    warmup_runs     >= 1      (first-token/compile cost excluded from timed runs)
    max_tokens      >= 64     (enough decode steps so first-step compile cost is diluted)
    prompt_lens     ⊇ {500, 3000, 16000}   (short / mid / long-context coverage)
    backends        ⊇ {"python", "swift"}   (both backends required)
    models          ⊇ {"motif-2.6b-q4", "motif-12.7b-q4"}  (2.6B + 12.7B target checkpoints)

These minimums are derived from:
- The README "Full intended sweep" example (n_runs=5, warmup_runs=1, max_tokens=64,
  prompt-lens 500 3000 16000, both backends, both models).
- The CI workflow_dispatch defaults (n_runs=3, max_tokens=64, warmup_runs=1).
  n_runs=5 is chosen as the certified bar because 5 runs provide a meaningful
  stdev estimate while keeping wall-clock time manageable on local Apple Silicon.

The current checked-in target sweep (benchmark-sweep-target-20260527T103000Z.json)
was generated with n_runs=1 / warmup_runs=0 / max_tokens=8.  It is a
smoke / first-evidence run.  The test below explicitly asserts that it is
NOT certified — this is intentional; it documents the evidence status honestly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Certification minimums
# ---------------------------------------------------------------------------
CERTIFIED_MIN_N_RUNS: int = 5
CERTIFIED_MIN_WARMUP_RUNS: int = 1
CERTIFIED_MIN_MAX_TOKENS: int = 64
CERTIFIED_REQUIRED_PROMPT_LENS: frozenset[int] = frozenset({500, 3000, 16000})
CERTIFIED_REQUIRED_BACKENDS: frozenset[str] = frozenset({"python", "swift"})
CERTIFIED_REQUIRED_MODELS: frozenset[str] = frozenset({"motif-2.6b-q4", "motif-12.7b-q4"})


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------


def validate_certification(sweep: dict[str, Any]) -> list[str]:
    """Return a list of certification violations for *sweep*.

    An empty list means the sweep passes all minimums.  Each entry in the
    returned list is a human-readable sentence describing one violation.

    The function reads the top-level ``config`` block of the sweep JSON and,
    for model coverage, the ``models`` array (or the ``cells`` array as
    a fallback when the ``models`` key is absent).
    """
    violations: list[str] = []
    config = sweep.get("config", {})

    # --- run counts ---
    n_runs = config.get("n_runs")
    if n_runs is None:
        violations.append("config.n_runs is missing")
    elif n_runs < CERTIFIED_MIN_N_RUNS:
        violations.append(f"config.n_runs={n_runs} < required {CERTIFIED_MIN_N_RUNS}")

    warmup_runs = config.get("warmup_runs")
    if warmup_runs is None:
        violations.append("config.warmup_runs is missing")
    elif warmup_runs < CERTIFIED_MIN_WARMUP_RUNS:
        violations.append(
            f"config.warmup_runs={warmup_runs} < required {CERTIFIED_MIN_WARMUP_RUNS}"
        )

    # --- token budget ---
    max_tokens = config.get("max_tokens")
    if max_tokens is None:
        violations.append("config.max_tokens is missing")
    elif max_tokens < CERTIFIED_MIN_MAX_TOKENS:
        violations.append(f"config.max_tokens={max_tokens} < required {CERTIFIED_MIN_MAX_TOKENS}")

    # --- prompt lengths ---
    prompt_lens_raw = config.get("prompt_lens")
    if prompt_lens_raw is None:
        violations.append("config.prompt_lens is missing")
    else:
        present = frozenset(int(p) for p in prompt_lens_raw)
        missing_prompts = CERTIFIED_REQUIRED_PROMPT_LENS - present
        if missing_prompts:
            violations.append(
                f"config.prompt_lens missing required lengths: {sorted(missing_prompts)}"
            )

    # --- backends ---
    backends_raw = config.get("backends")
    if backends_raw is None:
        violations.append("config.backends is missing")
    else:
        present_backends = frozenset(backends_raw)
        missing_backends = CERTIFIED_REQUIRED_BACKENDS - present_backends
        if missing_backends:
            violations.append(
                f"config.backends missing required backends: {sorted(missing_backends)}"
            )

    # --- model coverage ---
    # Prefer the top-level ``models`` array; fall back to ``cells``.
    model_ids: frozenset[str]
    if "models" in sweep and sweep["models"]:
        model_ids = frozenset(m["id"] for m in sweep["models"] if "id" in m)
    else:
        model_ids = frozenset(c["model_id"] for c in sweep.get("cells", []) if "model_id" in c)

    missing_models = CERTIFIED_REQUIRED_MODELS - model_ids
    if missing_models:
        violations.append(f"sweep is missing required models: {sorted(missing_models)}")

    return violations


# ---------------------------------------------------------------------------
# Helpers for building synthetic sweep dicts
# ---------------------------------------------------------------------------


def _make_sweep(
    *,
    n_runs: int = CERTIFIED_MIN_N_RUNS,
    warmup_runs: int = CERTIFIED_MIN_WARMUP_RUNS,
    max_tokens: int = CERTIFIED_MIN_MAX_TOKENS,
    prompt_lens: list[int] | None = None,
    backends: list[str] | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal synthetic sweep dict for testing."""
    if prompt_lens is None:
        prompt_lens = sorted(CERTIFIED_REQUIRED_PROMPT_LENS)
    if backends is None:
        backends = sorted(CERTIFIED_REQUIRED_BACKENDS)
    if models is None:
        models = [{"id": m} for m in sorted(CERTIFIED_REQUIRED_MODELS)]
    return {
        "schema_version": 1,
        "config": {
            "n_runs": n_runs,
            "warmup_runs": warmup_runs,
            "max_tokens": max_tokens,
            "prompt_lens": prompt_lens,
            "backends": backends,
            "dry_run": False,
        },
        "models": models,
        "cells": [],
        "comparisons": [],
    }


# ---------------------------------------------------------------------------
# Unit tests — synthetic PASS dict
# ---------------------------------------------------------------------------


class TestValidateCertificationPass:
    def test_minimal_pass_returns_no_violations(self) -> None:
        sweep = _make_sweep()
        assert validate_certification(sweep) == []

    def test_extra_prompt_lens_still_passes(self) -> None:
        sweep = _make_sweep(prompt_lens=[500, 3000, 16000, 32000])
        assert validate_certification(sweep) == []

    def test_extra_models_still_passes(self) -> None:
        sweep = _make_sweep(
            models=[
                {"id": "motif-2.6b-q4"},
                {"id": "motif-12.7b-q4"},
                {"id": "motif-extra"},
            ]
        )
        assert validate_certification(sweep) == []

    def test_higher_n_runs_passes(self) -> None:
        sweep = _make_sweep(n_runs=10, warmup_runs=2, max_tokens=128)
        assert validate_certification(sweep) == []


# ---------------------------------------------------------------------------
# Unit tests — synthetic FAIL dict (mirrors the current checked-in target sweep)
# ---------------------------------------------------------------------------


class TestValidateCertificationFail:
    """Each test isolates one violation to confirm the validator catches it."""

    def test_n_runs_1_is_violation(self) -> None:
        sweep = _make_sweep(n_runs=1)
        violations = validate_certification(sweep)
        assert any("n_runs=1" in v for v in violations), violations

    def test_warmup_runs_0_is_violation(self) -> None:
        sweep = _make_sweep(warmup_runs=0)
        violations = validate_certification(sweep)
        assert any("warmup_runs=0" in v for v in violations), violations

    def test_max_tokens_8_is_violation(self) -> None:
        sweep = _make_sweep(max_tokens=8)
        violations = validate_certification(sweep)
        assert any("max_tokens=8" in v for v in violations), violations

    def test_missing_long_context_prompt_len_is_violation(self) -> None:
        sweep = _make_sweep(prompt_lens=[500, 3000])  # missing 16000
        violations = validate_certification(sweep)
        assert any("16000" in v for v in violations), violations

    def test_missing_backend_is_violation(self) -> None:
        sweep = _make_sweep(backends=["python"])  # missing swift
        violations = validate_certification(sweep)
        assert any("swift" in v for v in violations), violations

    def test_missing_model_is_violation(self) -> None:
        sweep = _make_sweep(models=[{"id": "motif-2.6b-q4"}])  # missing 12.7b
        violations = validate_certification(sweep)
        assert any("motif-12.7b-q4" in v for v in violations), violations

    def test_target_sweep_like_settings_produce_three_violations(self) -> None:
        """Sweep with the same weak settings as the target (n_runs=1, warmup=0,
        max_tokens=8) has at least three violations."""
        sweep = _make_sweep(n_runs=1, warmup_runs=0, max_tokens=8)
        violations = validate_certification(sweep)
        assert len(violations) >= 3, violations

    def test_empty_config_returns_violations_for_every_field(self) -> None:
        violations = validate_certification({"config": {}, "models": [], "cells": []})
        assert len(violations) >= 5, violations


# ---------------------------------------------------------------------------
# Honest evidence-status assertion for the checked-in target sweep
# ---------------------------------------------------------------------------
# This test documents that the current target sweep is NOT certified.
# It is encoded as a POSITIVE assertion (violations must be non-empty) so that
# CI stays green regardless of whether a certified sweep exists yet.
# When a proper certified sweep is eventually checked in and pointed to here,
# this test should be updated or replaced.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET_SWEEP_PATH = (
    _REPO_ROOT / "docs" / "benchmarks" / "benchmark-sweep-target-20260527T103000Z.json"
)
_CERTIFIED_SWEEP_PATH = (
    _REPO_ROOT / "docs" / "benchmarks" / "benchmark-sweep-certified-20260531T184554Z.json"
)


@pytest.mark.skipif(
    not _TARGET_SWEEP_PATH.exists(),
    reason="target sweep JSON not present in working tree",
)
def test_existing_target_sweep_is_not_certified() -> None:
    """The checked-in target sweep (n_runs=1, max_tokens=8) must report violations.

    This is an intentional, honest documentation of its evidence status:
    it is a smoke / first-evidence run, not a certified repeated-thermal sweep.
    """
    sweep = json.loads(_TARGET_SWEEP_PATH.read_text(encoding="utf-8"))
    violations = validate_certification(sweep)
    assert violations, (
        "Expected the target sweep to report certification violations "
        "(n_runs=1, warmup_runs=0, max_tokens=8) but got none. "
        "If the file was replaced with a certified sweep, update this test."
    )
    # Confirm the specific known weaknesses are flagged
    violation_text = "\n".join(violations)
    assert "n_runs" in violation_text, f"Expected n_runs violation. Got: {violations}"
    assert "warmup_runs" in violation_text, f"Expected warmup_runs violation. Got: {violations}"
    assert "max_tokens" in violation_text, f"Expected max_tokens violation. Got: {violations}"


@pytest.mark.skipif(
    not _CERTIFIED_SWEEP_PATH.exists(),
    reason="certified sweep JSON not present in working tree",
)
def test_checked_in_certified_sweep_passes_certification() -> None:
    """The checked-in certified sweep must satisfy the certification minimums."""
    sweep = json.loads(_CERTIFIED_SWEEP_PATH.read_text(encoding="utf-8"))
    assert sweep["config"].get("dry_run") is False
    assert validate_certification(sweep) == []
    failed_cells = [
        cell["cell_id"] for cell in sweep.get("cells", []) if cell.get("status") != "pass"
    ]
    assert failed_cells == []
