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

import ast
import json
import operator
from collections.abc import Callable
from dataclasses import dataclass, field


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


# ---------------------------------------------------------------------------
# Tool-EXECUTION loop
# ---------------------------------------------------------------------------
#
# The pieces above *emit* tool calls (parse the model's JSON). This section
# closes the loop: generate -> parse tool_call -> execute -> append the tool
# result back into the conversation -> regenerate, bounded by ``max_rounds``.
#
# Chat-template note (verified against
# ``~/.models/motif-2.6b-mlx-q4`` tokenizer_config ``chat_template``): Motif's
# template has NO special tool/function tokens. Every message is rendered
# generically as ``<|startofturn|><|{role}|>\n\n{content}<|endofturn|>``. So a
# tool-result turn is simply a message with ``role="tool"`` and the result as
# ``content`` — the template renders ``<|tool|>`` and the model reads the
# result as ordinary conversation text. We keep the OpenAI-style ``role="tool"``
# for wire fidelity; the assistant tool-call turn is normalised to a single
# canonical ``{"tool_call": {...}}`` JSON string so the growing context stays
# tidy regardless of how much the small model looped in its raw output.

# A callable that, given the running list of chat messages, returns the model's
# raw text for the next assistant turn. Injecting this (rather than hard-wiring
# ``mlx_lm.generate``) keeps ``run_tool_loop`` pure and unit-testable without
# MLX or model weights. See ``make_mlx_generate_fn`` for the production adapter.
GenerateFn = Callable[[list[dict]], str]

# ``executor(name, arguments) -> str``. May raise; the loop catches the
# exception and surfaces it as a tool *error message* (never a crash).
ToolExecutor = Callable[[str, dict], object]


@dataclass
class ToolRound:
    """One executed tool call within :func:`run_tool_loop`."""

    name: str
    arguments: dict
    result: str
    is_error: bool


@dataclass
class ToolLoopResult:
    """Outcome of :func:`run_tool_loop`.

    ``final_text`` is the model's last non-tool-call answer (or, if the round
    budget was exhausted, the final generation regardless of shape).
    ``messages`` is the fully assembled conversation including the injected
    tools preamble, each assistant tool-call turn, and each tool-result turn.
    ``rounds`` records every executed tool call in order. ``stopped_reason`` is
    ``"completed"`` (model answered without calling a tool) or ``"max_rounds"``
    (the loop hit its bound while the model kept calling tools).
    """

    final_text: str
    messages: list[dict]
    rounds: list[ToolRound] = field(default_factory=list)
    stopped_reason: str = "completed"


def run_tool_loop(
    messages: list[dict],
    tools: list[dict] | None,
    tool_executor: ToolExecutor,
    generate: GenerateFn,
    *,
    max_rounds: int = 5,
    tool_role: str = "tool",
) -> ToolLoopResult:
    """Run generate -> execute -> regenerate until the model answers or the
    round budget is hit.

    Args:
      messages: OpenAI-style chat messages (not mutated; a copy is assembled).
      tools: OpenAI ``tools`` list. When non-empty a deterministic system
        preamble (:func:`build_tools_preamble`) is prepended and tool-call
        parsing is restricted to the declared names. When falsy, the loop
        degenerates to a single ``generate`` with no tool handling.
      tool_executor: ``executor(name, arguments) -> result``. Any exception it
        raises is caught and appended as a tool *error* message so the model can
        recover; it is never propagated out of the loop.
      generate: ``generate(messages) -> str`` producing the next assistant turn.
      max_rounds: maximum number of tool-execution cycles. The loop performs at
        most ``max_rounds + 1`` generations (one final answer attempt after the
        last tool result).
      tool_role: role string for tool-result messages (default ``"tool"``).

    Returns:
      :class:`ToolLoopResult` with the final text, the assembled messages, the
      executed rounds, and why the loop stopped.
    """
    conversation: list[dict] = list(messages)
    if tools:
        conversation = [
            {"role": "system", "content": build_tools_preamble(tools)},
            *conversation,
        ]

    rounds: list[ToolRound] = []
    for _ in range(max(max_rounds, 0)):
        text = generate(conversation)
        call = parse_tool_call(text, tools) if tools else None
        if call is None:
            return ToolLoopResult(
                final_text=text,
                messages=conversation,
                rounds=rounds,
                stopped_reason="completed",
            )

        # Normalise the assistant turn to a single canonical tool-call object so
        # the context does not accumulate the small model's repeated JSON.
        conversation.append(
            {"role": "assistant", "content": json.dumps({"tool_call": call})}
        )

        try:
            raw_result = tool_executor(call["name"], call["arguments"])
            is_error = False
        except Exception as exc:  # noqa: BLE001 — surface, never crash the loop
            raw_result = f"Error executing tool '{call['name']}': {exc}"
            is_error = True
        result_str = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)

        conversation.append(
            {"role": tool_role, "name": call["name"], "content": result_str}
        )
        rounds.append(
            ToolRound(
                name=call["name"],
                arguments=call["arguments"],
                result=result_str,
                is_error=is_error,
            )
        )

    # Round budget exhausted while the model kept calling tools: make one final
    # generation attempt (it may or may not still be a tool call) and stop.
    final_text = generate(conversation)
    return ToolLoopResult(
        final_text=final_text,
        messages=conversation,
        rounds=rounds,
        stopped_reason="max_rounds",
    )


def make_mlx_generate_fn(model, tokenizer, *, max_tokens: int = 256) -> GenerateFn:
    """Adapt an MLX ``model`` + ``tokenizer`` into a :data:`GenerateFn`.

    Renders messages with the tokenizer's chat template (Motif's template
    handles the ``tool`` role generically) and calls ``mlx_lm.generate``. Import
    of ``mlx_lm`` is deferred so this module stays importable without MLX.
    """
    from mlx_lm import generate as _mlx_generate

    def _generate(messages: list[dict]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover - defensive fallback
            prompt = "".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}\n" for m in messages
            )
        return _mlx_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )

    return _generate


# ---------------------------------------------------------------------------
# Safe builtin tools for the CLI demo (NO eval/exec of arbitrary code)
# ---------------------------------------------------------------------------

_ARITH_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ARITH_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Cap exponents so ``2 ** 10_000_000`` cannot be used to hang/OOM the process.
_MAX_POW_EXPONENT = 1000


def safe_arithmetic(expression: str) -> float:
    """Evaluate a numeric arithmetic ``expression`` WITHOUT ``eval``/``exec``.

    Parses the string to an AST and walks only a whitelist of numeric literals
    and binary/unary arithmetic operators (``+ - * / // % **`` and unary
    ``+``/``-``). Names, calls, attributes, comprehensions, and any other node
    raise :class:`ValueError`, so no arbitrary code can execute.

    Raises:
      ValueError: on a syntax error, a disallowed node, or an oversized power.
      ZeroDivisionError: on division/modulo by zero (surfaced to the caller).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid arithmetic expression: {exc}") from exc
    return _eval_arith_node(tree.body)


def _eval_arith_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported literal: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_BINOPS:
        left = _eval_arith_node(node.left)
        right = _eval_arith_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return _ARITH_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_UNARY:
        return _ARITH_UNARY[type(node.op)](_eval_arith_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")
