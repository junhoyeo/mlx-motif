from __future__ import annotations

import json
from pathlib import Path

REQUIRED_GATES = {
    "swift_build",
    "python_perplexity",
    "swift_perplexity",
    "python_logits",
    "swift_logits",
    "swift_speculative",
    "swift_q4_direct_bench",
    "swift_q4_bridge_bench",
    "python_decode_long_context",
    "swift_decode_long_context",
}


def _latest_report() -> tuple[Path, dict]:
    reports = sorted(Path("docs/benchmarks").glob("swift-python-hard-parity-*.json"))
    assert reports, "expected a checked-in Swift/Python hard-parity JSON report"
    report_path = reports[-1]
    return report_path, json.loads(report_path.read_text())


def test_swift_hard_parity_report_has_all_passing_gates() -> None:
    report_path, report = _latest_report()
    markdown_path = report_path.with_suffix(".md")

    assert markdown_path.exists(), "hard-parity JSON report must have a markdown summary"
    assert set(report["runs"]) >= REQUIRED_GATES

    failures = {
        name: cell
        for name, cell in report["runs"].items()
        if name in REQUIRED_GATES and cell["returncode"] != 0
    }
    assert failures == {}

    markdown = markdown_path.read_text()
    for gate in REQUIRED_GATES:
        assert f"| `{gate}` | PASS |" in markdown


def test_swift_hard_parity_quality_metrics_stay_within_recorded_tolerance() -> None:
    _, report = _latest_report()
    runs = report["runs"]

    python_ppl = runs["python_perplexity"]["json"]["result"]
    swift_ppl = runs["swift_perplexity"]["json"]
    assert python_ppl["tokens"] == swift_ppl["tokens"]
    assert abs(python_ppl["ppl"] - swift_ppl["perplexity"]) / python_ppl["ppl"] < 0.005
    assert abs(python_ppl["nll_per_token"] - swift_ppl["nllPerToken"]) < 0.002

    python_logits = runs["python_logits"]["json"]["result"]
    swift_logits = runs["swift_logits"]["json"]
    assert python_logits["top_k"][0]["token"] == swift_logits["topK"][0]["token"]

    python_top = {entry["token"] for entry in python_logits["top_k"]}
    swift_top = {entry["token"] for entry in swift_logits["topK"]}
    assert len(python_top & swift_top) >= 8

    checksum_delta = abs(python_logits["checksum"] - swift_logits["checksum"])
    assert checksum_delta / abs(python_logits["checksum"]) < 0.03


def test_swift_hard_parity_runtime_evidence_covers_direct_q4_speculative_and_long_context() -> None:
    _, report = _latest_report()
    runs = report["runs"]

    direct = runs["swift_q4_direct_bench"]["json"]["swift_native"]["json"]
    bridge = runs["swift_q4_bridge_bench"]["json"]["swift_native"]["json"]
    assert direct["output"] == bridge["output"]
    assert direct["generatedTokens"] == bridge["generatedTokens"]
    assert direct["generationInfo"]["tokensPerSecond"] >= (
        bridge["generationInfo"]["tokensPerSecond"] * 0.95
    )

    speculative = runs["swift_speculative"]["json"]["metrics"]
    assert speculative["targetTokens"] > 0
    assert speculative["proposedDraftTokens"] > 0
    assert speculative["acceptedDraftTokens"] > 0
    assert speculative["acceptedDraftTokens"] <= speculative["proposedDraftTokens"]

    python_long = runs["python_decode_long_context"]["json"]["results"]
    largest_python_prompt = max(cell["actual_prompt_len"] for cell in python_long.values())
    swift_long = runs["swift_decode_long_context"]["json"]["swift_native"]["json"]
    assert largest_python_prompt >= 1_600
    assert swift_long["promptTokens"] >= 1_600
    assert swift_long["generationInfo"]["tokensPerSecond"] > 0
