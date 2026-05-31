# SwiftUI native chat smoke checklist

Use this checklist before claiming the macOS app is ready for native-checkpoint
chat.

## Build

```bash
MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifChatApp
```

For a repeatable backend smoke that does not click the UI:

```bash
MOTIFKIT_ENABLE_MLX=1 scripts/smoke_swift_chat_app.py \
  --model .models/motif-2.6b-mlx-q4 \
  --require-model \
  --output docs/benchmarks/swift-app-smoke-local.json
```

The smoke script now starts the real Swift `MotifNativeServe` process, waits for
`GET /v1/models`, sends a non-streaming `/v1/chat/completions` request, and then
terminates the server. The latest checked-in native-server smoke evidence is
[`benchmarks/swift-app-smoke-native-server-20260527T102500Z.json`](benchmarks/swift-app-smoke-native-server-20260527T102500Z.json).

## Manual UI pass

### Preconditions

| Item | Requirement |
|------|-------------|
| Build | `MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifChatApp` (native path requires this flag; without it `nativeMLXNotCompiled` is the expected error) |
| Checkpoint | `~/.models/motif-2.6b-mlx-q4` — an HF→MLX converted Motif checkpoint directory with `config.json`, `tokenizer.json`, and weight shards present |
| Launch | `MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifChatApp` |
| Sidebar | The **NavigationSplitView** shows two sidebar entries: **Chat** and **Runtime**. All runtime settings live under **Runtime**. |

### Checklist

Tick each row on hardware before claiming the UI pass is green.

| # | Precondition | Action | Expected result | ✅ Pass / ❌ Fail | Notes |
|---|--------------|--------|-----------------|-----------------|-------|
| 1 | App not yet running | Launch via the command above | Window appears; sidebar shows **Chat** and **Runtime**; **Chat** panel is active with an empty message list and the **"Ask Motif…"** text field visible | | `ContentView.swift` `NavigationSplitView` entry point |
| 2 | App running; **Runtime** panel open | In the **Backend** section, click the **"Chat path"** picker and select **Native MLX checkpoint** | Picker label updates to **"Native MLX checkpoint"**; a **"Converted MLX model directory"** text field and a **Choose…** button appear below it | | `ContentView.swift:159-165`; `ChatStore.swift` `backendMode` published property |
| 3 | **Native MLX checkpoint** selected | Click **Choose…**, navigate to `~/.models/motif-2.6b-mlx-q4`, and confirm | Directory path fills the text field; **App status** section shows **"Native checkpoint selected"** in Runtime status | | `ContentView.swift:163-165` file importer; `ChatStore.swift:171-176` `selectNativeModelDirectory` |
| 4 | Checkpoint path set; switch to **Chat** panel | Type a short prompt (e.g. "Hello") and press **⌘↩** or click **Send** | **Send** button is replaced by a **Stop** button (⌘.); assistant bubble appears and text streams in token by token; **Runtime status** shows **"Generating…"** then **"Idle"** after completion | | `ChatStore.swift:96-168` `send()`; `ContentView.swift:94-99` conditional Stop/Send |
| 5 | Generation in progress (long prompt) | Click **Stop** (or press **⌘.**) while the assistant is streaming | Streaming stops immediately; the assistant bubble gains a `\n\n[Cancelled]` suffix; **Runtime status** shows **"Cancelled"**; **Stop** button reverts to **Send** | | `ContentView.swift:94-96`; `ChatStore.swift:87-93` `cancel()` and `CancellationError` handler at line 148-152 |
| 6 | **Native MLX checkpoint** selected; directory field is empty | Clear the directory field and send any prompt | Red error text appears below the input: *"Native MLX model directory cannot be empty."*; **Runtime status** shows **"Error"** | | `ChatStore.swift:217` `emptyNativeModelDirectory`; `ContentView.swift:83-86` error surface |
| 7 | **Native MLX checkpoint** selected | Set the directory field to a path that exists but is not a valid checkpoint (e.g. `/tmp`) and send a prompt | Red error text with a descriptive message; **Runtime status** shows **"Error"** (error propagates from `MotifMLXBackend` initialisation) | | `ChatStore.swift:208` `MotifMLXBackend(modelDirectory:)` throws |
| 8 | App built **without** `MOTIFKIT_ENABLE_MLX=1` (`nativeMLXCompiledIn` is false) | Select **Native MLX checkpoint** and send a prompt | Red error text: *"Native MLX chat is not compiled into this app build. Rebuild with MOTIFKIT_ENABLE_MLX=1."*; **Runtime status** shows **"Error"** | | `ChatStore.swift:209-210` `#else` branch throws `nativeMLXNotCompiled` |
| 9 | **OpenAI-compatible endpoint** selected | Set **OpenAI-compatible endpoint** field to a syntactically invalid URL (e.g. `not a url`) and send a prompt | Red error text: *"Endpoint must be a valid URL."*; **Runtime status** shows **"Error"** | | `ChatStore.swift:201-203` `invalidEndpoint`; `ContentView.swift:154` endpoint text field |
| 10 | `mlx-motif serve` running on `http://127.0.0.1:8080` | Select **OpenAI-compatible endpoint**; confirm **OpenAI-compatible endpoint** field is `http://127.0.0.1:8080/v1`; send a prompt | Assistant text streams in; **Runtime status** goes **"Connecting to endpoint…"** → **"Generating…"** → **"Idle"** | | `ChatStore.swift:199-204` `OpenAICompatibleMotifBackend`; Python `mlx-motif serve` fallback |
| 11 | Swift `MotifNativeServe` running on `http://127.0.0.1:8080` | Same as row 10, but started with Swift `MotifNativeServe` instead of the Python server | Same streaming result as row 10 | | Swift native-server fallback; both are valid OpenAI-compatible endpoints |
| 12 | Any think mode; checkpoint loaded; Chat panel active | In **Runtime → Backend**, set **Thinking** picker to **visible**; send a prompt whose response includes `<think>` blocks | `<think>…</think>` tokens stream visibly inside the assistant bubble; no crash | | `ContentView.swift:172-176` Thinking picker; `ChatStore.swift:46` `thinkMode` persisted |
| 13 | **Thinking** picker set to **hidden** | Send a prompt whose response includes `<think>` blocks | Assistant bubble contains only the final answer — `<think>` content is not shown; no crash | | `MotifThinkMode.hidden` suppresses reasoning tokens in the stream |
| 14 | **Thinking** picker set to **captured** | Send a prompt whose response includes `<think>` blocks | Assistant bubble shows the final answer; a **"Captured reasoning"** `DisclosureGroup` appears below the message list in a yellow-tinted box; expanding it reveals the raw reasoning text | | `ContentView.swift:56-65` `capturedReasoning` DisclosureGroup; `ChatStore.swift:141-143` `.reasoning` event handler |
| 15 | Non-default settings applied (e.g. `nativeMLX` mode, custom checkpoint path, `captured` think mode, `maxTokens` ≠ 512) | Quit the app (`⌘Q`) and relaunch | **Runtime** panel shows the same **Chat path**, **Converted MLX model directory**, **Thinking**, **Max tokens**, and **Temperature** values as before quit | | All `@Published` properties in `ChatStore.swift:34-53` write to `UserDefaults` on `didSet`; verified by `DefaultsKey` enum at line 274 |

## Packaging

Create a local `.app` bundle and zip:

```bash
scripts/package_swift_chat_app.sh
scripts/verify_swift_chat_app_package.py dist/MotifChatApp.metadata.json
```

Optional signing/notarization environment:

```bash
CODESIGN_IDENTITY="Developer ID Application: ..." scripts/package_swift_chat_app.sh
NOTARYTOOL_PROFILE=motif-notary scripts/package_swift_chat_app.sh
```

The packaging script ad-hoc signs by default and writes
`MotifChatApp.metadata.json` with the git commit, Swift toolchain, bundle path,
zip path, and SHA-256 checksums. Distribution builds should provide a Developer
ID identity and a notarytool keychain profile.
