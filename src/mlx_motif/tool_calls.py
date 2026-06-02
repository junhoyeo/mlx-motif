"""Prompt-based tool/function calling for the Motif MLX server.

IMPORTANT — honest scope: this is **prompt-injected** tool calling, not
model-native function calling. Motif's chat template has no special
tool/function tokens. Instead we:

  1. Build a deterministic system preamble from the OpenAI ``tools`` list that
     describes each tool and instructs the model to emit a single JSON object
     of the form ``{"tool_call": {"name", "arguments"}}`` (the standard OpenAI
     ``{"name", "arguments"}`` shape is also accepted) when a tool is needed.
  2. Parse the model's raw text output, extracting the FIRST complete JSON
     tool-call object. The small model tends to repeat/loop the object and may
     wrap it in prose; the parser is robust to both — it scans for the first
     balanced ``{...}`` that decodes to a recognised tool-call shape and stops.

This module is intentionally pure and importable so it can be unit-tested
without MLX or model weights.
"""

from __future__ import annotations

import json


def build_tools_preamble(tools: list[dict]) -> str:
    """Build a deterministic system preamble describing ``tools``.

    ``tools`` follows the OpenAI schema::

        [{"type": "function",
          "function": {"name", "description", "parameters"}}]

    The preamble lists each tool's name/description/parameters (JSON Schema)
    and instructs the model to emit a single JSON tool-call object when a tool
    is needed, or to answer normally otherwise. Generation is deterministic for
    a given ``tools`` list so the prompt is reproducible.
    """
    lines: list[str] = [
        "You have access to the following tools. When a tool is needed to "
        "answer the user, respond with a SINGLE JSON object on its own and "
        "nothing else, in this exact form:",
        '{"tool_call": {"name": "<tool name>", "arguments": {<json arguments>}}}',
        "Do not wrap the JSON in markdown fences. Emit it exactly once. If no "
        "tool is needed, answer the user normally in plain text.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "")
        if not name:
            continue
        description = fn.get("description", "")
        parameters = fn.get("parameters", {})
        lines.append(f"- {name}: {description}".rstrip())
        # Render the JSON Schema deterministically (sorted keys) so the prompt
        # is stable across requests with the same tool definitions.
        lines.append(f"  parameters schema: {json.dumps(parameters, sort_keys=True)}")
    return "\n".join(lines)


def _iter_json_objects(text: str):
    """Yield ``(obj, start_index)`` for each top-level balanced ``{...}`` in
    ``text`` that decodes as JSON, scanning left to right.

    Brace matching is string-aware (ignores braces inside JSON strings and
    respects backslash escapes) so braces embedded in argument string values do
    not break the scan. Non-decoding candidates are skipped.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        end = -1
        for j in range(i, n):
            c = text[j]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            # No balanced closer for this opener; stop — nothing further can
            # form a complete object that starts at or before here.
            return
        candidate = text[i : end + 1]
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            obj = None
        if obj is not None:
            yield obj, i
        i = end + 1


def _normalise_tool_call(obj: object) -> dict | None:
    """Return ``{"name", "arguments"}`` if ``obj`` is a recognised tool-call
    shape, else ``None``.

    Accepted shapes:
      * ``{"tool_call": {"name", "arguments"}}`` (preamble-instructed form)
      * ``{"name", "arguments"}`` (standard OpenAI function-call form)

    ``arguments`` must be a JSON object (dict). ``name`` must be a non-empty
    string. ``arguments`` defaults to ``{}`` when omitted.
    """
    if not isinstance(obj, dict):
        return None
    inner = obj.get("tool_call", obj)
    if not isinstance(inner, dict):
        return None
    name = inner.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = inner.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def parse_tool_call(text: str, tools: list[dict] | None = None) -> dict | None:
    """Extract the FIRST valid tool call from raw model output ``text``.

    Returns ``{"name": str, "arguments": dict}`` for the first JSON object that
    matches a recognised tool-call shape, or ``None`` when no tool call is
    present (plain prose, malformed JSON, or non-tool JSON).

    Robust to the model repeating the object and to surrounding prose — only
    the first match is returned. When ``tools`` is provided, the call's
    ``name`` must match one of the declared tool names; an unknown name is
    ignored (the scan continues).
    """
    if not text:
        return None

    known_names = None
    if tools:
        known_names = set()
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                known_names.add(fn["name"])

    for obj, _start in _iter_json_objects(text):
        call = _normalise_tool_call(obj)
        if call is None:
            continue
        if known_names is not None and call["name"] not in known_names:
            continue
        return call
    return None
