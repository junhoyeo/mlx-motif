# Server parity: MotifNativeServe (Swift) ↔ Python `mlx_motif.server`

Two OpenAI-compatible HTTP servers ship in this repo:

- **Python** — `src/mlx_motif/server.py` (`mlx-motif serve`), the reference
  implementation, wrapping `mlx_lm.generate.stream_generate`.
- **Swift** — `swift/Sources/MotifNativeServe/main.swift`, a native
  `Network.framework` server over the `MotifKitMLX` runtime.

Both expose the same minimal surface: `POST /v1/chat/completions` and
`GET /v1/models`, with request params `messages`, `stream`, `max_tokens`,
`temperature`, and the Motif-specific `think_mode`
(`visible` | `hidden` | `captured`).

This document records what is aligned, what this change (PR) fixed, and what
remains divergent or unimplemented. It is intentionally conservative: where a
gap is not yet closed it is listed as a gap, not glossed over.

The structural contract is pinned by `tests/test_server_contract.py`, which
runs the **Python** server in-process against a stubbed generation backend (no
weights, no MLX). The Swift server is **not** compiled or exercised in CI — it
lives behind the `MotifKitMLX` overlay and requires the Swift toolchain plus
Metal. For Swift, the contract below encodes the *expected* behaviour and the
Swift-side fixes are maintainer-/CI-gated (reviewed by diff, not by build).

## Endpoint / shape matrix

| Aspect                                   | Python                          | Swift (after this PR)            | Status |
|------------------------------------------|---------------------------------|----------------------------------|--------|
| `GET /v1/models`                         | `{"object":"list","data":[…]}`  | same                             | aligned |
| Unknown path                             | `404 {"error":{"message":"not found"}}` | same                     | aligned |
| Empty `messages`                         | `400 messages required`         | same                             | aligned |
| Streaming chunk `object`                 | `chat.completion.chunk`         | same                             | aligned |
| Streaming terminator                     | `data: [DONE]\n\n`              | same                             | aligned |
| `role` in streaming delta                | absent                          | absent                           | aligned |
| Non-streaming `object`                   | `chat.completion`               | same                             | aligned |

## Bugs fixed in this PR (Swift side)

These were real Swift-vs-Python divergences. The Python side already behaved
correctly; the fixes below bring `main.swift` into line.

| # | Divergence (before)                                              | Fix in this PR                                                                 | Status |
|---|------------------------------------------------------------------|-------------------------------------------------------------------------------|--------|
| 1 | Bad JSON body → Swift returned **500** (generic catch around `JSONDecoder`); Python returns **400** `bad json: …`. | Decode wrapped in a dedicated `do/catch` that maps the decode failure to `400 {"error":{"message":"bad json: …"}}`. Other runtime errors still surface as 500. | fixed |
| 2 | Non-200 reason phrase hardcoded to `"ERROR"` (e.g. `HTTP/1.1 404 ERROR`). | Added `reasonPhrase(for:)` → `400 Bad Request`, `404 Not Found`, `500 Internal Server Error`, `200 OK`; used in both `sendHeader` and `sendJSON`. | fixed |
| 4 | Captured `reasoning` emitted as a **separate** intermediate SSE chunk *before* the stop chunk; Python attaches it to the **final/stop** chunk. | Reasoning is buffered and merged onto the terminal `.completed` chunk (`final["reasoning"] = …`). Verified safe: the Swift SSE client `OpenAICompatibleMotifBackend.emit` reads the top-level `reasoning` field off whatever chunk carries it, so the client-visible event order is unchanged. | fixed (aligned to Python) |

## Documented gap (NOT fixed — would fake data otherwise)

| # | Gap                                                                  | Why not fixed here                                                                                                                                                                                                 | Status |
|---|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| 3 | Non-streaming `usage` is always `{prompt_tokens:0, completion_tokens:0, total_tokens:0}` in Swift; Python forwards the real counts from `stream_generate`'s final result. | The token counts are **not reachable** at the server: `MotifGenerationEvent` (`.text`/`.reasoning`/`.completed`) carries no token stats, and the underlying `generate(...)` `.info` case that holds them is consumed inside `MotifMLXNativeRuntime` and never surfaced. Reporting real numbers requires threading a usage payload through the enum (e.g. `.completed(usage:)`), a cross-module change deferred to a follow-up. A `TODO(parity)` marks the spot in `main.swift`. We emit a structurally valid zero `usage` object rather than fabricate counts. | divergent (documented) |

## Unimplemented params (both servers)

Neither server implements these OpenAI parameters; they are silently ignored.
Listed for honesty, not claimed as parity work:

| Param      | Python | Swift | Notes |
|------------|--------|-------|-------|
| `top_p`    | ignored | ignored | not plumbed to the sampler |
| `stop`     | ignored | ignored | no stop-sequence handling |
| `seed`     | ignored | ignored | no deterministic seeding path |
| `n`        | ignored | ignored | always one choice (`index: 0`) |
| `logprobs` | ignored | ignored | not emitted |

## What this PR does **not** establish

- **Content-level streaming equivalence.** The contract tests assert wire
  *structure* (chunk shape, `[DONE]`, `usage` field types, error codes), not
  that both servers produce identical tokens for a given prompt. Full
  byte-level output parity requires a live model on Apple-Silicon hardware and
  is out of scope here.
- **Compiled verification of the Swift fixes.** `main.swift` changes are
  reviewed by diff against the real runtime types (`MotifGenerationEvent`,
  `MotifChatBackend`, `OpenAICompatibleMotifBackend`) but not built in CI; the
  MLX overlay is maintainer-/CI-gated.
