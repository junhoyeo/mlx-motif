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
[`benchmarks/swift-app-smoke-native-server-20260531T184436Z.json`](benchmarks/swift-app-smoke-native-server-20260531T184436Z.json).

## Automated UI pass (XCUITest)

Most of the manual checklist below is now covered by a deterministic XCUITest
suite that drives the real SwiftUI app through its accessibility identifiers —
the correct way to automate the app, replacing sleep-based pokes and manual
clicking. It waits on the accessibility tree (`waitForExistence` / predicate
expectations), never on fixed timers.

```bash
scripts/ui_test_swift_chat_app.sh                    # whole suite
scripts/ui_test_swift_chat_app.sh NavigationUITests  # one class
scripts/ui_test_swift_chat_app.sh GenerationUITests/testStopMidStreamAppendsCancelled
```

> **One-time setup:** XCUITest drives the app through macOS automation, so the
> first run must happen in an **interactive login session** (not over SSH / not a
> detached CI/background shell) and macOS will ask to allow the test runner to
> control your Mac. Approve it (System Settings → Privacy & Security →
> Accessibility / Automation). Without the grant the run fails with *"Timed out
> while enabling automation mode."* — that is a permissions prompt going
> unanswered, not a test failure.

Because SwiftPM cannot host a UI-testing bundle, the suite lives in a thin Xcode
project generated from [`../swift/project.yml`](../swift/project.yml) by
XcodeGen (`brew install xcodegen`). The Xcode **app target is named
`MotifChatUIHost`**, deliberately different from the package's `MotifChatApp`
executable product — sharing the name makes `xcodebuild test` resolve the app to
the package's bare executable and fail to read its bundle identifier. `Package.swift` remains the source of truth
for building and unit-testing; the generated `MotifChatApp.xcodeproj` is
gitignored and rebuilt by the script. The app target is the **lightweight
(no-MLX) build**, so the native path resolves to the `nativeMLXNotCompiled`
branch — which is itself one of the automated tests.

**How the streaming rows stay deterministic:** the tests pass launch arguments
the app honours only in test builds:

| Flag / env | Effect | Source |
|------------|--------|--------|
| `-UITestFakeBackend` | `ChatStore.buildBackend()` returns `FakeStreamingMotifBackend` for **both** modes — a canned stream with **no model, no network**, honouring cancellation and think-mode | `ChatStore.swift`, `FakeStreamingMotifBackend.swift` |
| `-UITestDefaultsSuite <name>` / `-UITestResetDefaults` | routes/wipes `UserDefaults` into an isolated suite so runs never read or clobber real preferences | `ChatStore.defaults` |
| `MOTIF_UITEST_FAKE_CHUNKS` / `_DELAY_MS` / `_ANSWER` / `_REASONING` / `_FINISH` | tune the fake stream's pacing and content (e.g. a slow stream so a test can click **Stop** mid-flight) | `FakeStreamingMotifBackend.Config` |

**Coverage map** (row numbers refer to the checklist below):

| Rows | Status | Test class |
|------|--------|------------|
| 1, 2, 15 | automated | `NavigationUITests` |
| 8, 9 | automated | `ValidationUITests` |
| 4, 5 (UI of 10/11) | automated (fake backend) | `GenerationUITests` |
| 12, 13, 14 | automated (fake backend) | `ThinkModeUITests` |
| 3 | manual — depends on the `NSOpenPanel` file picker | — |
| 6, 7 | manual — need a real `MOTIFKIT_ENABLE_MLX=1` build + checkpoint | — |
| 10, 11 (real endpoints) | manual/backend smoke — see `smoke_swift_chat_app.py` | — |

The rows still marked manual are the ones that genuinely need real hardware
paths (a live MLX checkpoint or the file panel); run those by hand from the
table below.

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
| 1 | App not yet running | Launch via the command above | Window appears; sidebar shows **Chat** and **Runtime**; **Chat** panel is active with an empty message list and the **"Ask Motif…"** text field visible | | `ContentView.swift` `NavigationSplitView` entry point; input field a11y id `motif.chat.input` |
| 2 | App running; **Runtime** panel open | In the **Backend** section, click the **"Chat path"** picker and select **Native MLX checkpoint** | Picker label updates to **"Native MLX checkpoint"**; a **"Converted MLX model directory"** text field and a **Choose…** button appear below it | | `ContentView.swift` `Picker("Chat path", …)`; `ChatStore.swift` `backendMode` published property |
| 3 | **Native MLX checkpoint** selected | Click **Choose…**, navigate to `~/.models/motif-2.6b-mlx-q4`, and confirm | Directory path fills the text field; **App status** section shows **"Native checkpoint selected"** in Runtime status | | `ContentView.swift` `.fileImporter`; `ChatStore.swift` `selectNativeModelDirectory(_:)` (sets `runtimeStatus = "Native checkpoint selected"`) |
| 4 | Checkpoint path set; switch to **Chat** panel | Type a short prompt (e.g. "Hello") and press **⌘↩** or click **Send** | **Send** button is replaced by a **Stop** button (⌘.); assistant bubble appears and text streams in token by token; **Runtime status** shows **"Generating…"** then **"Idle"** after completion | | `ChatStore.swift` `send()`; `ContentView.swift` conditional `store.isGenerating` Stop/Send; a11y ids `motif.chat.send`, `motif.chat.stop`, `motif.chat.generating` |
| 5 | Generation in progress (long prompt) | Click **Stop** (or press **⌘.**) while the assistant is streaming | Streaming stops immediately; the assistant bubble gains a `\n\n[Cancelled]` suffix; **Runtime status** shows **"Cancelled"**; **Stop** button reverts to **Send** | | `ChatStore.swift` `cancel()` and the `catch is CancellationError` handler (appends `[Cancelled]`, sets `runtimeStatus = "Cancelled"`) |
| 6 | **Native MLX checkpoint** selected; directory field is empty | Clear the directory field and send any prompt | Red error text *"Native MLX model directory cannot be empty."* appears below the input; **Runtime status** shows **"Error"** | | `ChatStore.swift` `buildBackend()` → `acquireNativeModelDirectoryAccess()` → `resolvedNativeModelDirectoryFromPath()` throws `NativeModelDirectoryError.emptyPath` (message from `NativeModelDirectoryResolver.swift`) |
| 7 | **Native MLX checkpoint** selected | Set the directory field to a path that exists but is not a valid checkpoint (e.g. `/tmp`) and send a prompt | Red error text with a descriptive message; **Runtime status** shows **"Error"** (error propagates from `MotifMLXBackend` initialisation) | | `ChatStore.swift` `buildBackend()` `MotifMLXBackend(modelDirectory:)` throws |
| 8 | App built **without** `MOTIFKIT_ENABLE_MLX=1` | Select **Native MLX checkpoint** and send a prompt | Red error text: *"Native MLX chat is not compiled into this app build. Rebuild with MOTIFKIT_ENABLE_MLX=1."*; **Runtime status** shows **"Error"** | | `ChatStore.swift` `buildBackend()` `#else` branch throws `MotifChatStoreError.nativeMLXNotCompiled` |
| 9 | **OpenAI-compatible endpoint** selected | Set the **OpenAI-compatible endpoint** field to `http://[` and send a prompt | Red error text: *"Endpoint must be a valid URL."*; **Runtime status** shows **"Error"** | | `ChatStore.swift` `MotifChatStoreError.invalidEndpoint`. Note: on macOS 14+ Foundation's lenient URL parser percent-encodes strings like `not a url` into *valid* URLs, so ordinary garbage no longer trips this validation — it fails later in URLSession as "unsupported URL". A malformed IP-literal (`http://[`) still makes `URL(string:)` return nil and exercises the app's own check. |
| 10 | `mlx-motif serve` running on `http://127.0.0.1:8080` | Select **OpenAI-compatible endpoint**; confirm **OpenAI-compatible endpoint** field is `http://127.0.0.1:8080/v1`; send a prompt | Assistant text streams in; **Runtime status** goes **"Connecting to endpoint…"** → **"Generating…"** → **"Idle"** | | `ChatStore.swift` `OpenAICompatibleMotifBackend`; Python `mlx-motif serve` fallback |
| 11 | Swift `MotifNativeServe` running on `http://127.0.0.1:8080` | Same as row 10, but started with Swift `MotifNativeServe` instead of the Python server | Same streaming result as row 10 | | Swift native-server fallback; both are valid OpenAI-compatible endpoints |
| 12 | Any think mode; checkpoint loaded; Chat panel active | In **Runtime → Backend**, set **Thinking** picker to **visible**; send a prompt whose response includes `<think>` blocks | `<think>…</think>` tokens stream visibly inside the assistant bubble; no crash | | `ContentView.swift` `Picker("Thinking", …)`; `ChatStore.swift` `thinkMode` persisted |
| 13 | **Thinking** picker set to **hidden** | Send a prompt whose response includes `<think>` blocks | Assistant bubble contains only the final answer — `<think>` content is not shown; no crash | | `MotifThinkMode.hidden` suppresses reasoning tokens in the stream |
| 14 | **Thinking** picker set to **captured** | Send a prompt whose response includes `<think>` blocks | Assistant bubble shows the final answer; a **"Captured reasoning"** disclosure (`ReasoningDisclosure` / `DisclosureGroup`) appears below the message list; expanding it reveals the raw reasoning text | | `ContentView.swift` `ReasoningDisclosure(text:)` shown when `store.capturedReasoning` is non-empty; `ChatStore.swift` `.reasoning` event appends to `capturedReasoning` |
| 15 | Non-default settings applied (e.g. `nativeMLX` mode, custom checkpoint path, `captured` think mode, `maxTokens` ≠ 512) | Quit the app (`⌘Q`) and relaunch | **Runtime** panel shows the same **Chat path**, **Converted MLX model directory**, **Thinking**, **Max tokens**, and **Temperature** values as before quit | | The `@Published` settings properties in `ChatStore.swift` write to `UserDefaults` on `didSet`, keyed by the `DefaultsKey` enum |

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
zip path, and SHA-256 checksums. The latest checked-in package verification
evidence is [`benchmarks/swift-app-package-20260531T184436Z.json`](benchmarks/swift-app-package-20260531T184436Z.json). Distribution builds should provide a Developer
ID identity and a notarytool keychain profile.
