"""Shared lightweight parity fixtures for the Swift Motif port.

The fixtures are intentionally tiny and weight-free so Swift and Python tests can
agree on Motif config shape, `<think>` stream filtering, and deterministic
component math before full checkpoint parity exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from mlx_motif.model import ModelArgs
from mlx_motif.server import ThinkFilter

FIXTURE = Path(__file__).parent / "fixtures" / "motif_parity_cases.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_config_fixtures_match_python_model_args_shape():
    fixture = _load_fixture()

    for case in fixture["configs"]:
        args = ModelArgs(**case["python"])
        expectations = case["expectations"]

        assert args.is_grouped is expectations["is_grouped"]
        assert (args.head_dim or args.hidden_size // args.num_attention_heads) == expectations[
            "effective_head_dim"
        ]
        if args.is_grouped:
            assert args.num_noise_heads is not None
            assert (
                args.num_attention_heads // args.num_noise_heads
                == expectations["origin_heads_per_noise_head"]
            )
            assert (
                args.num_key_value_heads // args.num_noise_heads
                == expectations["kv_heads_per_group"]
            )


def test_think_filter_fixtures_match_python_server_behavior():
    fixture = _load_fixture()

    for case in fixture["think_filter_cases"]:
        filter_ = ThinkFilter(case["mode"], start_in_think=case["start_in_think"])
        emitted = "".join(filter_.feed(chunk) for chunk in case["chunks"])

        assert emitted == case["expected_emitted"], case["name"]
        assert filter_.captured == case["expected_captured"], case["name"]


def test_polynorm_component_fixture_matches_reference_formula():
    fixture = _load_fixture()
    [case] = [item for item in fixture["component_checks"] if item["kind"] == "polynorm"]

    eps = case["eps"]
    weight = case["weight"]
    bias = case["bias"][0]
    actual = []
    for row in case["input"]:
        mean_x2 = sum(value**2 for value in row) / len(row)
        mean_x4 = sum(value**4 for value in row) / len(row)
        mean_x6 = sum(value**6 for value in row) / len(row)
        inv_rms_x = 1 / math.sqrt(mean_x2 + eps)
        inv_rms_x2 = 1 / math.sqrt(mean_x4 + eps)
        inv_rms_x3 = 1 / math.sqrt(mean_x6 + eps)
        actual.append(
            [
                weight[0] * (value**3 * inv_rms_x3)
                + weight[1] * (value**2 * inv_rms_x2)
                + weight[2] * (value * inv_rms_x)
                + bias
                for value in row
            ]
        )

    for actual_row, expected_row in zip(actual, case["expected"], strict=True):
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            assert math.isclose(actual_value, expected_value, rel_tol=0, abs_tol=1e-12)
