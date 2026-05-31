"""Perf-regression sentinel: Swift-vs-Python q4 throughput floor.

This test loads the most recent benchmark-sweep-target-*.json, extracts all
``swift_vs_python`` comparison rows, and asserts that every row's ``speedup``
has not degraded below (recorded_baseline * TOLERANCE).

Scope of this test
------------------
* This is a GAP-TRACKING sentinel, NOT a parity claim.
* Swift is currently ~0.20–0.86x Python on motif-2.6b-q4 and ~0.25–0.33x on
  motif-12.7b-q4 (q4-direct, prompt-tokens-match rows only).  Closing that gap
  requires hardware-level perf work; this test just makes any *regression* from
  the recorded baseline auditable and deliberate.

Honesty constraints encoded here
---------------------------------
* Floor assertions are only applied to rows where ``prompt_tokens_match`` is
  True (the "clean" comparisons where both backends processed the same actual
  token count).
* For ``prompt_tokens_match == False`` rows (all motif-12.7b-q4 rows in the
  current sweep, because the two tokenisers rendered a different number of
  tokens from the same target-length prompt), we assert:
    (a) those rows are still present in the sweep file, and
    (b) they remain flagged as mismatched (i.e. no silent fix that would make
        us miss the caveat).
  We do NOT treat their speedup as a parity baseline because the token-count
  difference means we are comparing apples to oranges.

Methodology caveat
------------------
All sweep cells use ``--max-tokens 8`` with ``--n-runs 1`` and no warmup.  The
tiny decode budget means the measurement is dominated by prompt-processing
(prefill) latency.  Treat numbers as relative indicators, not steady-state
throughput figures.

Updating the baseline
---------------------
If a re-run legitimately improves or changes Swift performance:
1. Regenerate the sweep JSON via ``scripts/bench_sweep.py``.
2. Update ``RECORDED_BASELINES`` in this file to match the new numbers.
3. Commit both together so the change is intentional and visible in review.
"""

from __future__ import annotations

import json
from pathlib import Path

# Regression tolerance: a new sweep's speedup must be >= baseline * TOLERANCE.
# 0.80 means we allow up to a 20% degradation before the sentinel fires.
TOLERANCE = 0.80

# Recorded speedup values from benchmark-sweep-target-20260527T103000Z.json.
# Keys are (model_id, cache_cell, prompt_target_tokens).
#
# ONLY rows where prompt_tokens_match == True are included here.
# motif-12.7b-q4 is intentionally absent; see module docstring and the
# separate assertions for mismatched rows below.
RECORDED_BASELINES: dict[tuple[str, str, int], float] = {
    # motif-2.6b-q4 / q4_bridge — prompt_tokens_match: True
    ("motif-2.6b-q4", "q4_bridge", 500): 0.26433850511896423,
    ("motif-2.6b-q4", "q4_bridge", 3000): 0.38028784846646707,
    ("motif-2.6b-q4", "q4_bridge", 16000): 0.861545676687921,
    # motif-2.6b-q4 / q4_direct — prompt_tokens_match: True
    ("motif-2.6b-q4", "q4_direct", 500): 0.2002045917406077,
    ("motif-2.6b-q4", "q4_direct", 3000): 0.35365051891967814,
    ("motif-2.6b-q4", "q4_direct", 16000): 0.6869829815392222,
}

# The 12.7b rows that are flagged with prompt_tokens_match == False.
# We verify they exist and remain flagged — but do not floor their speedup.
KNOWN_MISMATCHED_KEYS: set[tuple[str, str, int]] = {
    ("motif-12.7b-q4", "q4_bridge", 500),
    ("motif-12.7b-q4", "q4_bridge", 3000),
    ("motif-12.7b-q4", "q4_bridge", 16000),
    ("motif-12.7b-q4", "q4_direct", 500),
    ("motif-12.7b-q4", "q4_direct", 3000),
    ("motif-12.7b-q4", "q4_direct", 16000),
}


def _latest_sweep() -> tuple[Path, dict]:
    """Return (path, parsed JSON) for the most recent target sweep file."""
    candidates = sorted(Path("docs/benchmarks").glob("benchmark-sweep-target-*.json"))
    assert candidates, (
        "No benchmark-sweep-target-*.json found in docs/benchmarks/. "
        "Checked-in sweep file is required for this test."
    )
    path = candidates[-1]
    return path, json.loads(path.read_text())


def _swift_vs_python_rows(sweep: dict) -> list[dict]:
    return [c for c in sweep.get("comparisons", []) if c.get("comparison") == "swift_vs_python"]


def _cache_cell_from_candidate(row: dict) -> str:
    """Extract the cache_cell name from the candidate_cell id string.

    candidate_cell format: ``<model_id>/<backend>/<cache_cell>/<prompt_bucket>``
    e.g. ``motif-2.6b-q4/swift/q4_direct/p500``
    """
    parts = row["candidate_cell"].split("/")
    assert len(parts) == 4, f"Unexpected candidate_cell format: {row['candidate_cell']}"
    return parts[2]


def _row_key(row: dict) -> tuple[str, str, int]:
    return (
        row["model_id"],
        _cache_cell_from_candidate(row),
        row["prompt_target_tokens"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sweep_file_exists_and_has_comparisons() -> None:
    """Sanity-check that the sweep file is present and structurally valid."""
    path, sweep = _latest_sweep()
    rows = _swift_vs_python_rows(sweep)
    assert rows, f"{path.name}: expected at least one swift_vs_python comparison row"
    assert sweep.get("schema_version") == 1


def test_clean_rows_meet_recorded_floor() -> None:
    """For prompt_tokens_match==True rows, speedup must not fall below recorded * TOLERANCE.

    If this test fails, a checked-in sweep JSON shows Swift performance has
    regressed materially vs the recorded baseline.  Either fix the regression
    or deliberately update RECORDED_BASELINES above with a justification commit.
    """
    _, sweep = _latest_sweep()
    rows = _swift_vs_python_rows(sweep)

    failures: list[str] = []
    checked = 0

    for row in rows:
        if not row.get("prompt_tokens_match", False):
            # Token-count mismatch: skip floor check (see module docstring).
            continue

        key = _row_key(row)
        if key not in RECORDED_BASELINES:
            # New row added by a future sweep — no baseline yet, skip silently.
            continue

        baseline = RECORDED_BASELINES[key]
        floor = baseline * TOLERANCE
        actual = row["speedup"]
        checked += 1

        if actual < floor:
            model_id, cache_cell, prompt_target = key
            failures.append(
                f"  REGRESSION: {model_id}/{cache_cell}/p{prompt_target}: "
                f"speedup={actual:.4f} < floor={floor:.4f} "
                f"(baseline={baseline:.4f}, tolerance={TOLERANCE})"
            )

    assert checked > 0, (
        "No clean (prompt_tokens_match==True) swift_vs_python rows found that "
        "match RECORDED_BASELINES keys.  Sweep schema may have changed."
    )

    assert not failures, (
        "Swift-vs-Python perf regression detected in sweep JSON.\n"
        "Failing rows:\n" + "\n".join(failures) + "\n\n"
        "To accept a genuine improvement/change: update RECORDED_BASELINES in "
        "tests/test_swift_python_perf_regression.py and commit with a rationale."
    )


def test_mismatched_token_rows_are_present_and_still_flagged() -> None:
    """Known prompt_tokens_match==False rows must remain present and flagged.

    This guards against a silent schema change that would drop the mismatch
    flag and let these dubious numbers flow into parity claims.
    """
    _, sweep = _latest_sweep()
    rows = _swift_vs_python_rows(sweep)

    seen_mismatched: set[tuple[str, str, int]] = set()
    for row in rows:
        key = _row_key(row)
        if key in KNOWN_MISMATCHED_KEYS:
            # The row must still be flagged as mismatched.
            assert row.get("prompt_tokens_match") is False, (
                f"Row {key} was previously flagged prompt_tokens_match=False "
                "but is now True.  Verify the tokeniser fix is real, then "
                "move the key from KNOWN_MISMATCHED_KEYS to RECORDED_BASELINES."
            )
            seen_mismatched.add(key)

    missing = KNOWN_MISMATCHED_KEYS - seen_mismatched
    assert not missing, (
        f"Expected mismatch-flagged rows not found in sweep: {missing}. "
        "The sweep file may have been regenerated without these model/cell "
        "combinations.  Update KNOWN_MISMATCHED_KEYS if intentional."
    )


def test_all_swift_vs_python_rows_have_required_fields() -> None:
    """Schema guard: every comparison row must carry speedup and prompt_tokens_match."""
    _, sweep = _latest_sweep()
    rows = _swift_vs_python_rows(sweep)
    assert rows, "No swift_vs_python rows found"

    for row in rows:
        key = _row_key(row)
        assert "speedup" in row, f"Row {key} missing 'speedup' field"
        assert "prompt_tokens_match" in row, f"Row {key} missing 'prompt_tokens_match' field"
        assert isinstance(row["speedup"], float), (
            f"Row {key}: 'speedup' must be a float, got {type(row['speedup'])}"
        )
        assert isinstance(row["prompt_tokens_match"], bool), (
            f"Row {key}: 'prompt_tokens_match' must be a bool"
        )
