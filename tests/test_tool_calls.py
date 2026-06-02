"""Unit tests for the pure prompt-based tool-call parser.

These exercise `mlx_motif.tool_calls.parse_tool_call` and
`build_tools_preamble` directly — no MLX, no model weights, no HTTP server.
"""

from __future__ import annotations

import json

from mlx_motif.tool_calls import build_tools_preamble, parse_tool_call

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
