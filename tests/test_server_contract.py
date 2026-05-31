"""Structural HTTP-contract tests for the OpenAI-compatible Motif server.

These tests pin the *wire contract* that BOTH server implementations are
supposed to honour:

  * the Python reference server (`mlx_motif.server`), which is exercised live
    here with a stubbed generation backend (no model weights, no MLX), and
  * the Swift native server (`swift/Sources/MotifNativeServe/main.swift`),
    which cannot be compiled/run in this CI lane (it lives behind the
    `MotifKitMLX` overlay and requires the Swift toolchain + Metal).

Where the Swift server cannot be exercised, we encode the *expected* behaviour
as the contract and note the corresponding Swift fix. The known divergences
and what part A of this change repaired are tracked in `docs/server-parity.md`.

The Python server is started in-process on an ephemeral port. Generation is
stubbed by monkeypatching `mlx_lm.generate.stream_generate` (which
`mlx_motif.server` imports lazily inside `_make_handler`), so these tests run
on any machine with the dev dependencies installed.
"""

from __future__ import annotations

import http.client
import importlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import HTTPServer

import pytest

from mlx_motif import server as motif_server

# `import mlx_lm.generate` can resolve to the re-exported `generate` *function*
# rather than the submodule (mlx_lm exports `generate` at package level). Load
# the actual module object so monkeypatching `stream_generate` targets the
# symbol that `mlx_motif.server` imports lazily inside `_make_handler`.
mlx_generate = importlib.import_module("mlx_lm.generate")


@dataclass
class _StubResult:
    """Mimics the per-step result object yielded by `stream_generate`.

    Only the attributes the server reads are provided: `text` for the streamed
    segment, plus the token-count / finish fields it pulls off the final step.
    """

    text: str
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


class _StubTokenizer:
    """Minimal tokenizer exposing `apply_chat_template` like a HF tokenizer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Deterministic rendering; the contract tests never inspect the prompt,
        # they only need a string the server can hand to the (stubbed) generator.
        return "".join(f"{m.get('role', 'user')}: {m.get('content', '')}\n" for m in messages)


def _stub_stream_generate(model, tokenizer, prompt, **kwargs):
    """Yield a fixed two-step generation with non-zero usage counts.

    Token counts are intentionally non-trivial so the `usage`-shape assertions
    would fail if the server forwarded zeros instead of the real result.
    """
    yield _StubResult(text="Hello", prompt_tokens=7, generation_tokens=1, finish_reason=None)
    yield _StubResult(text=" world", prompt_tokens=7, generation_tokens=2, finish_reason="stop")


@contextmanager
def _running_server(
    monkeypatch, think_mode: str = "visible", stream_generate=_stub_stream_generate
) -> Iterator[tuple[str, int]]:
    """Start the Python server in-process with a stubbed backend.

    Yields (host, port). Tears the server down on exit.
    """
    monkeypatch.setattr(mlx_generate, "stream_generate", stream_generate)

    handler_cls = motif_server._make_handler(
        model=object(),
        tokenizer=_StubTokenizer(),
        model_id="motif-stub",
        default_think_mode=think_mode,
    )
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    host, port = httpd.server_address[0], httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield host, port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _request(host, port, method, path, body=None, raw_body=None):
    """Perform an HTTP request, returning (status, headers, raw_bytes)."""
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        payload = raw_body
        if payload is None and body is not None:
            payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# GET /v1/models  → 200 {"object":"list","data":[...]}                          #
# --------------------------------------------------------------------------- #
def test_models_endpoint_shape(monkeypatch):
    with _running_server(monkeypatch) as (host, port):
        status, _headers, data = _request(host, port, "GET", "/v1/models")
    assert status == 200
    payload = json.loads(data)
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    assert payload["data"], "models list must be non-empty"
    assert payload["data"][0]["id"] == "motif-stub"
    assert payload["data"][0]["object"] == "model"


# --------------------------------------------------------------------------- #
# GET /unknown  → 404 {"error":{"message":"not found"}}                         #
# --------------------------------------------------------------------------- #
def test_unknown_path_is_404(monkeypatch):
    with _running_server(monkeypatch) as (host, port):
        status, _headers, data = _request(host, port, "GET", "/unknown")
    assert status == 404
    assert json.loads(data) == {"error": {"message": "not found"}}


# --------------------------------------------------------------------------- #
# empty messages  → 400 messages required                                      #
# --------------------------------------------------------------------------- #
def test_empty_messages_is_400(monkeypatch):
    with _running_server(monkeypatch) as (host, port):
        status, _headers, data = _request(
            host, port, "POST", "/v1/chat/completions", body={"messages": []}
        )
    assert status == 400
    assert json.loads(data)["error"]["message"] == "messages required"


# --------------------------------------------------------------------------- #
# bad JSON  → 400 bad json                                                      #
#                                                                              #
# This encodes the shared contract. The Python server already returns 400; the #
# Swift server historically returned 500 (generic catch around JSONDecoder) —  #
# part A of this change maps the decode failure to a 400 `bad json: …` body to #
# match. The Swift overlay is not compiled in this CI lane, so this test only  #
# exercises the Python side; see docs/server-parity.md.                        #
# --------------------------------------------------------------------------- #
def test_bad_json_body_is_400(monkeypatch):
    with _running_server(monkeypatch) as (host, port):
        status, _headers, data = _request(
            host, port, "POST", "/v1/chat/completions", raw_body=b"{not valid json"
        )
    assert status == 400
    message = json.loads(data)["error"]["message"]
    assert message.startswith("bad json")


# --------------------------------------------------------------------------- #
# non-streaming response includes a `usage` key with integer fields            #
# --------------------------------------------------------------------------- #
def test_non_streaming_usage_has_integer_fields(monkeypatch):
    with _running_server(monkeypatch) as (host, port):
        status, headers, data = _request(
            host,
            port,
            "POST",
            "/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
    assert status == 200
    assert headers.get("Content-Type") == "application/json"
    payload = json.loads(data)
    assert payload["object"] == "chat.completion"
    usage = payload["usage"]
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert field in usage, f"usage missing {field}"
        assert isinstance(usage[field], int), f"usage.{field} must be int"
    # The stub reports prompt=7, generation=2 on the final step; the Python
    # server forwards these real counts (Swift currently emits zeros — tracked
    # as a known gap in docs/server-parity.md).
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 9
    # No `role` echoed in a non-streaming `message` beyond assistant content.
    assert payload["choices"][0]["message"]["content"] == "Hello world"


# --------------------------------------------------------------------------- #
# streaming response: chunks are chat.completion.chunk, ends with [DONE]        #
# --------------------------------------------------------------------------- #
def test_streaming_contract(monkeypatch):
    with _running_server(monkeypatch) as (host, port):
        status, headers, data = _request(
            host,
            port,
            "POST",
            "/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert status == 200
    assert headers.get("Content-Type") == "text/event-stream"
    text = data.decode("utf-8")
    # Must terminate with the SSE done sentinel.
    assert text.endswith("data: [DONE]\n\n")

    # Parse out the JSON SSE events (everything except the [DONE] sentinel).
    events = []
    for line in text.split("\n\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))

    assert events, "expected at least one streamed chunk"
    for ev in events:
        assert ev["object"] == "chat.completion.chunk"
        # Delta must never carry a `role` (parity contract).
        assert "role" not in ev["choices"][0]["delta"]

    # The terminal chunk carries finish_reason == "stop".
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    # Reassembled content matches the stub generation.
    streamed = "".join(ev["choices"][0]["delta"].get("content", "") for ev in events)
    assert streamed == "Hello world"


# --------------------------------------------------------------------------- #
# captured think_mode: reasoning rides on the FINAL/stop chunk (Python), the    #
# behaviour the Swift fix in part A now matches.                                #
# --------------------------------------------------------------------------- #
def test_streaming_captured_reasoning_on_final_chunk(monkeypatch):
    def _stub_with_think(model, tokenizer, prompt, **kwargs):
        yield _StubResult(
            text="<think>secret</think>", prompt_tokens=5, generation_tokens=1, finish_reason=None
        )
        yield _StubResult(text="answer", prompt_tokens=5, generation_tokens=2, finish_reason="stop")

    with _running_server(monkeypatch, think_mode="captured", stream_generate=_stub_with_think) as (
        host,
        port,
    ):
        status, _headers, data = _request(
            host,
            port,
            "POST",
            "/v1/chat/completions",
            body={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert status == 200
    text = data.decode("utf-8")
    events = [
        json.loads(seg.strip()[len("data: ") :])
        for seg in text.split("\n\n")
        if seg.strip().startswith("data: ") and seg.strip()[len("data: ") :] != "[DONE]"
    ]
    final = events[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    # Python attaches captured reasoning to the final chunk; the part-A Swift
    # fix aligns the native server to this placement.
    assert final.get("reasoning") == "secret"
    # The think trace is filtered out of the visible stream.
    visible = "".join(ev["choices"][0]["delta"].get("content", "") for ev in events)
    assert visible == "answer"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
