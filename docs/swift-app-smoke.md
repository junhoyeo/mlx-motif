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

1. Launch:
   ```bash
   MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifChatApp
   ```
2. Open **Runtime**.
3. Select **Native MLX checkpoint**.
4. Use **Choose…** to select a converted checkpoint directory.
5. Send a short prompt.
6. Verify:
   - assistant text streams into the chat
   - **Stop** cancels an in-flight long response and leaves a visible cancelled marker
   - invalid checkpoint directory shows an actionable error
   - `visible`, `hidden`, and `captured` thinking modes do not crash
   - settings persist across app relaunch
7. Switch back to **OpenAI-compatible endpoint**.
8. Run either Python `mlx-motif serve` or Swift `MotifNativeServe` and verify the server-backed path still streams.

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
