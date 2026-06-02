# Server parity: MotifNativeServe (Swift) <-> Python `mlx_motif.server`

Two OpenAI-compatible HTTP servers ship in this repo:

- **Python** — `src/mlx_motif/server.py` (`mlx-motif serve`), the reference
  implementation, wrapping `mlx_lm.generate.stream_generate`.
- **Swift** — `swift/Sources/MotifNativeServe/main.swift`, a native
  `Network.framework` server over the `MotifKitMLX` runtime.

Both expose the same minimal surface: `POST /v1/chat/completions` and
`GET /v1/models`, with request params `messages`, `stream`, `max_tokens`,
`temperature`, and the Motif-specific `think_mode`
(`visible` | `hidden` | `captured`). The Python server additionally supports
prompt-based `tools` / `tool_choice` (see "Tool / function calling" below);
the Swift server does not.

This document records what is aligned, what this change (PR) fixed, and what
remains divergent or unimplemented. It is intentionally conservative: where a
gap is not yet closed it is listed as a gap, not glossed over.

The structural contract is pinned by `tests/test_server_contract.py`, which
runs the **Python** server in-process against a stubbed generation backend (no
weights, no MLX). The Swift server is **not** compiled or exercised in the
public CI lane — it lives behind the `MotifKitMLX` overlay and requires the
Swift toolchain plus Metal. For this PR the Swift fixes (including the
cross-module `usage` change) were compiled and tested **locally** with
Swift 6.0.3; see the verification note below.

## Endpoint / shape matrix

| Aspect                                   | Python                          | Swift (after this PR)            | Status |
|------------------------------------------|---------------------------------|----------------------------------|--------|
| `GET /v1/models`                         | `{"object":"list","data":[…]}`  | same                             | aligned |
| Unknown path                             | `404 {"error":{"message":"not found"}}` | same                     | aligned |
| Empty `messages`                         | `400 messages required`         | same                             | aligned |
| Bad JSON body                            | `400 bad json: …`               | same                             | aligned (fixed this PR) |
| Streaming chunk `object`                 | `chat.completion.chunk`         | same                             | aligned |
| Streaming terminator                     | `data: [DONE]\n\n`              | same                             | aligned |
| `role` in streaming delta                | absent                          | absent                           | aligned |
| Non-streaming `object`                   | `chat.completion`               | same                             | aligned |
| Non-streaming `usage` (real counts)      | `prompt`/`completion`/`total` from `stream_generate` final result | same — surfaced from MLX `.info` via `MotifGenerationUsage` | aligned (fixed this PR) |
| Streaming `stream_options.include_usage` | trailing usage chunk with real counts when requested; omitted by default | same | aligned (fixed this PR) |

## Bugs fixed in this PR (Swift side)

These were real Swift-vs-Python divergences. The Python side already behaved
correctly; the fixes below bring `main.swift` into line.

| # | Divergence (before)                                              | Fix in this PR                                                                 | Status |
|---|------------------------------------------------------------------|-------------------------------------------------------------------------------|--------|
| 1 | Bad JSON body -> Swift returned **500** (generic catch around `JSONDecoder`); Python returns **400** `bad json: …`. | Decode wrapped in a dedicated `do/catch` that maps the decode failure to `400 {"error":{"message":"bad json: …"}}`. Other runtime errors still surface as 500. | fixed |
| 2 | Non-200 reason phrase hardcoded to `"ERROR"` (e.g. `HTTP/1.1 404 ERROR`). | Added `reasonPhrase(for:)` -> `400 Bad Request`, `404 Not Found`, `500 Internal Server Error`, `200 OK`; used in both `sendHeader` and `sendJSON`. | fixed |
| 3 | Non-streaming `usage` was always `{prompt_tokens:0, completion_tokens:0, total_tokens:0}` in Swift; Python forwards the real counts from `stream_generate`'s final result. | **Fixed end-to-end (cross-module).** See the "Usage token counts" implementation note below. | fixed (aligned to Python) |
| 4 | Captured `reasoning` emitted as a **separate** intermediate SSE chunk *before* the stop chunk; Python attaches it to the **final/stop** chunk. | Reasoning is buffered and merged onto the terminal `.completed` chunk (`final["reasoning"] = …`). Verified safe: the Swift SSE client `OpenAICompatibleMotifBackend.emit` reads the top-level `reasoning` field off whatever chunk carries it, so the client-visible event order is unchanged. | fixed (aligned to Python) |
| 5 | `stream_options: {"include_usage": true}` was ignored, so streaming clients could not request authoritative token counts. | Both servers now emit one extra `chat.completion.chunk` after the terminal stop chunk and before `[DONE]`, with empty `choices` and a `usage` object sourced from the final generation result. The chunk is omitted when usage is unavailable or not requested. | fixed (aligned to Python) |

## Usage token counts — fixed end-to-end

This was previously a documented gap (Swift emitted a structurally valid zero
`usage` object because the counts were not reachable at the server). It is now
implemented across three modules, without breaking any existing
`MotifGenerationEvent` consumer:

- **`MotifKit/MotifTypes.swift`** — new `MotifGenerationUsage`
  (`promptTokens` / `completionTokens`, with `totalTokens` computed). The
  terminal `MotifGenerationEvent.completed` case now carries an optional
  `MotifGenerationUsage`. Existing `case .completed:` consumers are unaffected
  (they ignore the associated value); backends that cannot report counts (the
  remote OpenAI-compatible SSE client) construct `.completed(usage: nil)`.
- **`MotifKitMLX/MotifMLXNativeRuntime.swift`** — captures MLX's
  `generate(...)` `.info` completion counts (`promptTokenCount` /
  `generationTokenCount`) and surfaces them on the terminal `.completed` event.
- **`MotifNativeServe/main.swift`** — `completeChat` populates the non-streaming
  `usage` object (`prompt_tokens` / `completion_tokens` / `total_tokens`) from
  the surfaced counts, matching the Python `usage` shape exactly. The streaming
  path supports `stream_options: {include_usage: true}` by sending one trailing
  usage chunk with empty `choices`, after the stop chunk and before `[DONE]`.
  All fields are sourced from terminal generation counts; `total_tokens` is
  never derived from an SSE-chunk counter.

The `MotifNativeGenerate` CLI also prints the terminal counts to stderr
(`[usage] prompt_tokens=… generation_tokens=… total_tokens=…`), keeping stdout
reserved for streamed text.

## Tool / function calling — supported (prompt-based, Python only)

The Python server supports OpenAI-style `tools` / `tool_choice` on
`POST /v1/chat/completions`. **This is prompt-injected tool calling, not
model-native function calling.** Motif's chat template has no special
tool/function tokens, so we do not and cannot claim native tool-calling
parity with the OpenAI API.

How it works (`src/mlx_motif/tool_calls.py`):

1. **Inject.** When `tools` are present, a deterministic system preamble is
   built from the tools list (each tool's name / description / JSON-Schema
   parameters) and prepended to `messages`. The preamble instructs the model
   to emit a single JSON object `{"tool_call": {"name", "arguments"}}` (the
   standard OpenAI `{"name", "arguments"}` shape is also accepted) when a tool
   is needed, or to answer in plain text otherwise.
2. **Parse.** `parse_tool_call(text, tools)` scans the raw model output and
   extracts the **first** balanced JSON object matching a recognised tool-call
   shape. This is necessary because the small Motif-2.6B q4 model **loops /
   repeats the object with no stop condition** — the parser takes the first
   complete call and ignores the repeats and any surrounding prose. Malformed
   JSON, non-tool JSON, and (when `tools` is supplied) unknown tool names
   yield no call. When `name` filtering is requested, the call's `name` must
   match a declared tool.
3. **Shape.** A detected call is returned as OpenAI `tool_calls`:
   `[{"id", "type":"function", "function":{"name", "arguments":<stringified
   JSON>}}]` with `finish_reason: "tool_calls"` and `message.content: null`.
   In streaming, one `chat.completion.chunk` carries the call in
   `delta.tool_calls` and the terminal chunk reports `finish_reason:
   "tool_calls"`. `tool_choice: "none"` disables tools; `"auto"` / `"required"`
   behave like the default prompt.

| Aspect                  | Python                                      | Swift | Status |
|-------------------------|---------------------------------------------|-------|--------|
| `tools` request param   | accepted; injected as system preamble       | not implemented | Python-only |
| `tool_choice`           | `none` disables; otherwise prompt-default   | not implemented | Python-only |
| `tool_calls` response   | OpenAI shape, `finish_reason: "tool_calls"` | not implemented | Python-only |

**Honest limitations:**

- Prompt-based, not model-native. There are no tool tokens in the template;
  reliability depends on the model honouring the preamble.
- Because the small model loops, a server-side **stop/parse layer** is
  required — the parser extracts only the first call (it does not impose a
  generation stop sequence, so tokens are still generated up to `max_tokens`).
- No execution of tools and no multi-turn tool-result handling beyond passing
  `tool`-role messages straight through to the prompt; only the request→tool_call
  emission is shaped here.
- Streaming buffers the output when `tools` are present (it cannot stream a
  partial JSON tool call meaningfully), so the tool call arrives in a single
  delta rather than token-by-token.
- Swift `MotifNativeServe` does not implement tools.

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
- **Compiled verification of the Swift fixes in CI.** The MLX overlay is
  maintainer-/CI-gated and is not built in the public CI lane (no toolchain +
  Metal there). The fixes in this PR — including the cross-module `usage`
  change — were compiled and exercised **locally** with Swift 6.0.3: both
  `swift build` (default) and `MOTIFKIT_ENABLE_MLX=1 swift build` succeed and
  `swift test` passes. The Python contract tests run in CI.
