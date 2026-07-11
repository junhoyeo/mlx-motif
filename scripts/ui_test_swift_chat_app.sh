#!/usr/bin/env bash
#
# Run the MotifChatApp XCUITest UI-automation suite.
#
# This replaces the sleep-based smoke poke and the manual click-through checklist
# in docs/swift-app-smoke.md with a deterministic, wait-on-condition automation:
# the app is driven through XCUITest against its accessibility identifiers, using
# an injected in-process fake backend (no model, no network) for the streaming
# rows. See swift/MotifChatAppUITests/ for the tests.
#
# The Xcode project is generated from swift/project.yml by XcodeGen; the SwiftPM
# package (Package.swift) remains the source of truth for building/unit-testing.
#
# REQUIREMENTS: run this in an INTERACTIVE login session (not SSH / not a
# detached background shell). The first run triggers a macOS prompt to let the
# test runner control the Mac — approve it (System Settings > Privacy & Security
# > Accessibility). An unanswered prompt surfaces as "Timed out while enabling
# automation mode", which is a permissions issue, not a test failure.
#
# Usage:
#   scripts/ui_test_swift_chat_app.sh                    # run the whole suite
#   scripts/ui_test_swift_chat_app.sh NavigationUITests  # one test class
#   scripts/ui_test_swift_chat_app.sh NavigationUITests/testLaunchShowsSidebarAndInput
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_DIR="$REPO_ROOT/swift"
PROJECT="$SWIFT_DIR/MotifChatApp.xcodeproj"
SCHEME="MotifChatApp"
UITEST_TARGET="MotifChatAppUITests"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "error: xcodegen not found. Install with 'brew install xcodegen'." >&2
  exit 1
fi

echo "==> Generating $PROJECT from project.yml"
( cd "$SWIFT_DIR" && xcodegen generate --spec project.yml )

# Build -only-testing filters from any positional args (class or class/method).
ONLY_TESTING=()
for arg in "$@"; do
  ONLY_TESTING+=("-only-testing:${UITEST_TARGET}/${arg}")
done

echo "==> Running XCUITest suite (${#ONLY_TESTING[@]} filter(s))"
run_xcodebuild() {
  xcodebuild test \
    -project "$PROJECT" \
    -scheme "$SCHEME" \
    -destination 'platform=macOS' \
    -resultBundlePath "$SWIFT_DIR/.build/ui-test-result.xcresult" \
    "${ONLY_TESTING[@]}" \
    CODE_SIGNING_ALLOWED=YES
}

# Pretty-print with xcbeautify when available, but preserve xcodebuild's exit
# status (pipefail) so a test failure fails the script. Tee the raw log so we can
# detect the automation-permission timeout and print an actionable hint.
RAW_LOG="$SWIFT_DIR/.build/ui-test-xcodebuild.log"
status=0
if command -v xcbeautify >/dev/null 2>&1; then
  run_xcodebuild | tee "$RAW_LOG" | xcbeautify || status=$?
else
  run_xcodebuild | tee "$RAW_LOG" || status=$?
fi

if [ "$status" -ne 0 ] && grep -q "enabling automation mode" "$RAW_LOG" 2>/dev/null; then
  cat >&2 <<'HINT'

──────────────────────────────────────────────────────────────────────────────
"Timed out while enabling automation mode" is a macOS PERMISSION prompt going
unanswered — not a test failure. Run this from an interactive login session and
approve the request to control your Mac:
  System Settings > Privacy & Security > Accessibility (and Automation)
Then re-run this script.
──────────────────────────────────────────────────────────────────────────────
HINT
fi
exit "$status"
