"""
OpenAI-compatible HTTP server for Motif on MLX, with `<think>` streaming support.

A thin wrapper around `mlx_motif.load` + `mlx_lm.generate.stream_generate`
that exposes `/v1/chat/completions` (and `/v1/models`). Adds a Motif-
specific `think_mode` parameter (or env var) controlling reasoning-trace
visibility:

    visible  (default) — `<think>...</think>` content streams as-is
    hidden   — tokens between <think> and </think> are buffered and
               omitted from the streamed output (final `content` field
               will not contain the trace)
    captured — same as `hidden` but the trace is returned in a separate
               non-OpenAI field `reasoning` for clients that opt in

Run:

    mlx-motif serve --model ./out/motif-12.7b-reasoning-q4 --port 8080

Then hit it with the standard OpenAI Python SDK:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="ignored")
    r = client.chat.completions.create(
        model="motif",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        stream=True,
        extra_body={"think_mode": "hidden"},
    )
    for chunk in r:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ThinkFilter:
    """Stream-time filter for `<think>...</think>` reasoning traces.

    Mode `visible`: passthrough.
    Mode `hidden`: drops content between (and including) the tags.
    Mode `captured`: drops from the visible stream, accumulates trace
        in `self.captured` so the server can include it in the final
        non-streaming response (or in a final SSE chunk for streaming).
    """

    def __init__(self, mode: str = "visible", start_in_think: bool = False):
        """``start_in_think=True`` treats the stream as already inside a
        ``<think>`` block from the first token. Required for chat templates
        that end with ``<|assistant|><think>`` (e.g., Motif's default reasoning
        template) — the model's output begins inside the think block with no
        opening tag in the stream, only a closing ``</think>``.
        """
        if mode not in ("visible", "hidden", "captured"):
            raise ValueError(f"unknown think_mode: {mode}")
        self.mode = mode
        self.in_think = start_in_think
        self.buffer = ""  # carries cross-chunk partial-tag text
        self.captured = ""

    def feed(self, text: str) -> str:
        """Feed a streamed text segment, return what should be emitted."""
        if self.mode == "visible":
            return text

        # Append to buffer to handle partial tags split across chunks.
        self.buffer += text
        out = []

        while self.buffer:
            if self.in_think:
                close_idx = self.buffer.find(_THINK_CLOSE)
                if close_idx < 0:
                    # All buffered is still in-think; capture if requested,
                    # then check whether the tail might be a partial close-tag.
                    keep_partial = self._partial_tag_suffix(self.buffer, _THINK_CLOSE)
                    consume = self.buffer[: len(self.buffer) - keep_partial]
                    if self.mode == "captured":
                        self.captured += consume
                    self.buffer = self.buffer[len(self.buffer) - keep_partial :]
                    return "".join(out)
                else:
                    inside = self.buffer[:close_idx]
                    if self.mode == "captured":
                        self.captured += inside
                    self.in_think = False
                    self.buffer = self.buffer[close_idx + len(_THINK_CLOSE) :]
            else:
                open_idx = self.buffer.find(_THINK_OPEN)
                if open_idx < 0:
                    keep_partial = self._partial_tag_suffix(self.buffer, _THINK_OPEN)
                    emit_now = self.buffer[: len(self.buffer) - keep_partial]
                    out.append(emit_now)
                    self.buffer = self.buffer[len(self.buffer) - keep_partial :]
                    return "".join(out)
                else:
                    out.append(self.buffer[:open_idx])
                    self.in_think = True
                    self.buffer = self.buffer[open_idx + len(_THINK_OPEN) :]

        return "".join(out)

    @staticmethod
    def _partial_tag_suffix(buf: str, tag: str) -> int:
        """How many tail chars of `buf` could be the start of `tag`?"""
        for n in range(min(len(tag) - 1, len(buf)), 0, -1):
            if tag.startswith(buf[-n:]):
                return n
        return 0


def _make_messages_text(messages: list[dict], tokenizer) -> str:
    """Render OpenAI-style messages into a chat-template prompt string."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # Fallback: simple role: content concatenation.
    return (
        "".join(f"{m.get('role', 'user')}: {m.get('content', '')}\n" for m in messages)
        + "assistant: "
    )


def _make_handler(model, tokenizer, model_id: str, default_think_mode: str):
    from mlx_lm.generate import stream_generate

    class APIHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

        def _send_json(self, code: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def do_GET(self):
            if self.path == "/v1/models":
                return self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": model_id, "object": "model", "created": int(time.time())}],
                    },
                )
            self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._send_json(404, {"error": {"message": "not found"}})

            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length))
            except Exception as e:
                return self._send_json(400, {"error": {"message": f"bad json: {e}"}})

            messages = body.get("messages") or []
            stream = bool(body.get("stream", False))
            max_tokens = int(body.get("max_tokens") or 256)
            temperature = float(body.get("temperature", 0.0))
            think_mode = body.get("think_mode") or default_think_mode

            if not messages:
                return self._send_json(400, {"error": {"message": "messages required"}})

            prompt = _make_messages_text(messages, tokenizer)
            # If the chat template put the model in <think> mode (e.g., Motif's
            # default reasoning template ends with `<|assistant|><think>\n`),
            # the response stream starts INSIDE the block with no opening tag —
            # only a closing </think>. Pre-set the filter accordingly.
            start_in_think = _THINK_OPEN in prompt.rsplit(_THINK_CLOSE, 1)[-1]
            filt = ThinkFilter(mode=think_mode, start_in_think=start_in_think)
            req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())

            kwargs = {"max_tokens": max_tokens}
            # `temp` is the mlx-lm sampler control; 0 → greedy.
            if temperature > 0:
                from mlx_lm.sample_utils import make_sampler

                kwargs["sampler"] = make_sampler(temp=temperature)

            if stream:
                self._send_sse()
                try:
                    for r in stream_generate(model, tokenizer, prompt, **kwargs):
                        out = filt.feed(r.text)
                        if out:
                            chunk = {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [
                                    {"index": 0, "delta": {"content": out}, "finish_reason": None}
                                ],
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                    # final chunk with optional captured reasoning
                    final = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    if think_mode == "captured" and filt.captured:
                        final["reasoning"] = filt.captured
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                full = ""
                last = None
                for r in stream_generate(model, tokenizer, prompt, **kwargs):
                    full += filt.feed(r.text)
                    last = r
                payload = {
                    "id": req_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": full},
                            "finish_reason": (last.finish_reason if last else "stop"),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": (last.prompt_tokens if last else 0),
                        "completion_tokens": (last.generation_tokens if last else 0),
                        "total_tokens": (
                            (last.prompt_tokens + last.generation_tokens) if last else 0
                        ),
                    },
                }
                if think_mode == "captured" and filt.captured:
                    payload["reasoning"] = filt.captured
                self._send_json(200, payload)

    return APIHandler


def serve(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    model_id: str = "motif",
    think_mode: str = "visible",
) -> None:
    from mlx_motif import load

    print(f"Loading model from {model_path} …", file=sys.stderr)
    model, tokenizer = load(model_path)
    handler_cls = _make_handler(model, tokenizer, model_id, think_mode)
    httpd = HTTPServer((host, port), handler_cls)
    print(
        f"Serving {model_id} (think_mode={think_mode}) on http://{host}:{port}/v1", file=sys.stderr
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to converted MLX checkpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model-id", default="motif", help="Identifier reported in /v1/models")
    p.add_argument(
        "--think-mode",
        default="visible",
        choices=["visible", "hidden", "captured"],
        help="Default reasoning-trace handling (overridable per-request)",
    )
    args = p.parse_args()
    serve(args.model, args.host, args.port, args.model_id, args.think_mode)


if __name__ == "__main__":
    main()
