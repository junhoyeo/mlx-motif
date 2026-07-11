#!/usr/bin/env bash
#
# Run the LIVE MLX tool round-trip UI test (model -> app -> execute -> model)
# against a real converted checkpoint through the GUI.
#
# This is the opt-in, slow counterpart to ui_test_swift_chat_app.sh: it builds
# the MotifChatMLXHost app WITH the MLX overlay and drives the native backend
# end to end. Requires:
#   - ~/.models/motif-2.6b-mlx-q4 (the default native checkpoint)
#   - an interactive login session (macOS automation permission), like the
#     standard UI suite
#
# The manifest env MOTIFKIT_ENABLE_MLX=1 must be set so Package.swift exposes the
# MotifKitMLX product; this script sets it for you.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_DIR="$REPO_ROOT/swift"
PROJECT="$SWIFT_DIR/MotifChatApp.xcodeproj"
DERIVED="$HOME/Library/Developer/Xcode/DerivedData/MotifChatMLX-fixed"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "error: xcodegen not found. Install with 'brew install xcodegen'." >&2
  exit 1
fi

echo "==> Ensuring MLX Metal library is built"
"$REPO_ROOT/scripts/build_mlx_swift_metallib.sh" >/dev/null

echo "==> Generating $PROJECT from project.yml"
( cd "$SWIFT_DIR" && xcodegen generate --spec project.yml )

echo "==> Running live MLX tool round-trip (MotifChatMLX scheme)"
run() {
  MOTIFKIT_ENABLE_MLX=1 xcodebuild test \
    -project "$PROJECT" \
    -scheme MotifChatMLX \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath "$DERIVED" \
    CODE_SIGNING_ALLOWED=YES
}

if command -v xcbeautify >/dev/null 2>&1; then
  run | tee "$SWIFT_DIR/.build/mlx-ui-test.log" | xcbeautify
else
  run
fi
