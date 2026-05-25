#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

swift test --package-path swift
swift build --package-path swift --target MotifChatApp
