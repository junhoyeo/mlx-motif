"""Unit tests for the pure prompt-based tool-call parser.

These exercise `mlx_motif.tool_calls.parse_tool_call` and
`build_tools_preamble` directly — no MLX, no model weights, no HTTP server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlx_motif.tool_calls import (
    build_tools_preamble,
    parse_tool_call,
    run_tool_loop,
    safe_arithmetic,
)

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_valid_single_tool_call_object():
    text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}'
    assert parse_tool_call(text, _WEATHER_TOOLS) == {
        "name": "get_weather",
        "arguments": {"city": "Tokyo"},
    }


def test_standard_openai_shape_without_wrapper():
    text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    assert parse_tool_call(text, _WEATHER_TOOLS) == {
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }


def test_repeated_objects_returns_first():
    # The small model loops without a stop condition: the parser must return
    # the FIRST complete tool-call object and ignore the repeats.
    one = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}'
    two = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Osaka"}}}'
    text = f"{one}\n{two}\n{one}"
    assert parse_tool_call(text, _WEATHER_TOOLS) == {
        "name": "get_weather",
        "arguments": {"city": "Tokyo"},
    }


def test_surrounding_prose_is_tolerated():
    text = (
        "Sure, let me check that for you.\n"
        '{"tool_call": {"name": "get_weather", "arguments": {"city": "Berlin"}}}\n'
        "I will get back to you."
    )
    assert parse_tool_call(text, _WEATHER_TOOLS) == {
        "name": "get_weather",
        "arguments": {"city": "Berlin"},
    }


def test_no_tool_call_returns_none():
    assert parse_tool_call("The weather in Tokyo is sunny.", _WEATHER_TOOLS) is None


def test_malformed_json_returns_none():
    text = '{"tool_call": {"name": "get_weather", "arguments": {city: Tokyo}}'
    assert parse_tool_call(text, _WEATHER_TOOLS) is None


def test_non_tool_json_returns_none():
    # Valid JSON, but not a tool-call shape.
    assert parse_tool_call('{"foo": "bar", "n": 1}', _WEATHER_TOOLS) is None


def test_arguments_with_nested_structures_preserved():
    args = {"city": "São Paulo", "opts": {"units": "metric", "days": [1, 2, 3]}}
    text = json.dumps({"tool_call": {"name": "get_weather", "arguments": args}})
    result = parse_tool_call(text, _WEATHER_TOOLS)
    assert result is not None
    assert result["arguments"] == args


def test_arguments_with_braces_in_string_value():
    # Braces inside an argument string value must not confuse brace matching.
    args = {"city": "Tokyo {special}"}
    text = json.dumps({"tool_call": {"name": "get_weather", "arguments": args}})
    assert parse_tool_call(text, _WEATHER_TOOLS) == {
        "name": "get_weather",
        "arguments": args,
    }


def test_unknown_tool_name_is_skipped_when_tools_given():
    text = '{"tool_call": {"name": "send_email", "arguments": {"to": "a@b.c"}}}'
    assert parse_tool_call(text, _WEATHER_TOOLS) is None


def test_unknown_tool_name_accepted_when_no_tools_filter():
    text = '{"tool_call": {"name": "send_email", "arguments": {"to": "a@b.c"}}}'
    assert parse_tool_call(text, None) == {
        "name": "send_email",
        "arguments": {"to": "a@b.c"},
    }


def test_arguments_default_to_empty_dict_when_omitted():
    text = '{"tool_call": {"name": "get_weather"}}'
    assert parse_tool_call(text, _WEATHER_TOOLS) == {
        "name": "get_weather",
        "arguments": {},
    }


def test_non_dict_arguments_rejected():
    text = '{"tool_call": {"name": "get_weather", "arguments": "Tokyo"}}'
    assert parse_tool_call(text, _WEATHER_TOOLS) is None


def test_empty_text_returns_none():
    assert parse_tool_call("", _WEATHER_TOOLS) is None


def test_build_preamble_is_deterministic_and_mentions_tools():
    a = build_tools_preamble(_WEATHER_TOOLS)
    b = build_tools_preamble(_WEATHER_TOOLS)
    assert a == b
    assert "get_weather" in a
    assert "Get the weather for a city" in a
    assert "tool_call" in a


# ---------------------------------------------------------------------------
# Cross-language golden parity: same tools JSON -> byte-identical preamble.
# The fixture's ``expected`` is the authoritative Python output; the Swift port
# (MotifToolCallingTests) asserts against the SAME fixture. This test guards the
# Python side from drifting away from the frozen golden.
# ---------------------------------------------------------------------------

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_preamble_cases.json"


def test_preamble_matches_cross_language_golden_fixture():
    cases = json.loads(_FIXTURE.read_text())["cases"]
    assert cases, "fixture must contain at least one case"
    for case in cases:
        assert build_tools_preamble(case["tools"]) == case["expected"], case["name"]


# ---------------------------------------------------------------------------
# safe_arithmetic — numeric only, no eval/exec.
# ---------------------------------------------------------------------------


def test_safe_arithmetic_basic():
    assert safe_arithmetic("2 * (3 + 4)") == 14
    assert safe_arithmetic("10 / 4") == 2.5
    assert safe_arithmetic("2 ** 10") == 1024
    assert safe_arithmetic("-7 + 3") == -4
    assert safe_arithmetic("17 % 5") == 2
    assert safe_arithmetic("17 // 5") == 3


def test_safe_arithmetic_rejects_names_and_calls():
    for expr in ["__import__('os')", "os.system('x')", "a + 1", "len([1])", "x"]:
        with pytest.raises(ValueError):
            safe_arithmetic(expr)


def test_safe_arithmetic_rejects_oversized_power():
    with pytest.raises(ValueError):
        safe_arithmetic("2 ** 10000000")


def test_safe_arithmetic_division_by_zero_surfaces():
    with pytest.raises(ZeroDivisionError):
        safe_arithmetic("1 / 0")


# ---------------------------------------------------------------------------
# run_tool_loop — generate -> parse -> execute -> append -> regenerate.
# A scripted generate callable stands in for the model; no MLX involved.
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """A generate() callable that returns a fixed list of outputs in order and
    records the exact message list it was handed on each call."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def __call__(self, messages):
        # Snapshot (deep-ish copy of the role/content we care about) so later
        # mutation of the running conversation does not corrupt the record.
        self.calls.append([dict(m) for m in messages])
        if self._outputs:
            return self._outputs.pop(0)
        return "(no more scripted outputs)"


_CALC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


def _calc_executor(name, arguments):
    assert name == "calculator"
    return str(safe_arithmetic(arguments["expression"]))


def test_loop_executes_tool_then_returns_final_answer():
    model = _ScriptedModel(
        [
            '{"tool_call": {"name": "calculator", "arguments": {"expression": "2 + 2"}}}',
            "The answer is 4.",
        ]
    )
    result = run_tool_loop(
        messages=[{"role": "user", "content": "what is 2+2?"}],
        tools=_CALC_TOOLS,
        tool_executor=_calc_executor,
        generate=model,
    )
    assert result.stopped_reason == "completed"
    assert result.final_text == "The answer is 4."
    assert len(result.rounds) == 1
    assert result.rounds[0].name == "calculator"
    assert result.rounds[0].result == "4"
    assert result.rounds[0].is_error is False

    roles = [m["role"] for m in result.messages]
    # system preamble, user, assistant(tool_call), tool(result)
    assert roles == ["system", "user", "assistant", "tool"]
    assert "You have access to the following tools" in result.messages[0]["content"]
    tool_msg = result.messages[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["name"] == "calculator"
    assert tool_msg["content"] == "4"
    # Second generation saw the tool result appended.
    assert model.calls[1][-1]["role"] == "tool"


def test_loop_is_bounded_by_max_rounds_when_model_never_stops():
    # Model that ALWAYS emits a tool call -> without a bound this would spin
    # forever. The loop must stop after max_rounds executions.
    always_call = '{"tool_call": {"name": "calculator", "arguments": {"expression": "1 + 1"}}}'
    model = _ScriptedModel([always_call] * 50)
    result = run_tool_loop(
        messages=[{"role": "user", "content": "loop"}],
        tools=_CALC_TOOLS,
        tool_executor=_calc_executor,
        generate=model,
        max_rounds=3,
    )
    assert result.stopped_reason == "max_rounds"
    assert len(result.rounds) == 3
    # At most max_rounds + 1 generations.
    assert len(model.calls) == 4


def test_loop_surfaces_executor_errors_as_tool_messages_not_crashes():
    model = _ScriptedModel(
        [
            '{"tool_call": {"name": "calculator", "arguments": {"expression": "1 / 0"}}}',
            "I could not compute that.",
        ]
    )

    def boom_executor(name, arguments):
        return str(safe_arithmetic(arguments["expression"]))  # raises ZeroDivisionError

    # No exception should escape the loop.
    result = run_tool_loop(
        messages=[{"role": "user", "content": "divide by zero"}],
        tools=_CALC_TOOLS,
        tool_executor=boom_executor,
        generate=model,
    )
    assert result.stopped_reason == "completed"
    assert result.final_text == "I could not compute that."
    assert len(result.rounds) == 1
    assert result.rounds[0].is_error is True
    assert "Error executing tool 'calculator'" in result.rounds[0].result
    # The error was appended as a tool-role message the model then saw.
    assert result.messages[-1]["role"] == "tool"
    assert "Error executing tool 'calculator'" in result.messages[-1]["content"]


def test_loop_without_tools_is_single_generation():
    model = _ScriptedModel(["just a plain answer"])
    result = run_tool_loop(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        tool_executor=_calc_executor,
        generate=model,
    )
    assert result.stopped_reason == "completed"
    assert result.final_text == "just a plain answer"
    assert result.rounds == []
    assert len(model.calls) == 1
    # No preamble injected when there are no tools.
    assert [m["role"] for m in result.messages] == ["user"]


def test_loop_does_not_mutate_caller_messages():
    original = [{"role": "user", "content": "what is 2+2?"}]
    model = _ScriptedModel(
        [
            '{"tool_call": {"name": "calculator", "arguments": {"expression": "2 + 2"}}}',
            "4",
        ]
    )
    run_tool_loop(
        messages=original,
        tools=_CALC_TOOLS,
        tool_executor=_calc_executor,
        generate=model,
    )
    assert original == [{"role": "user", "content": "what is 2+2?"}]
