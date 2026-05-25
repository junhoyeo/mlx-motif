#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MOTIFKIT_ENABLE_MLX=1
swift build --package-path swift --target MotifKitMLX
swift build --package-path swift --target MotifNativeGenerate
swift build --package-path swift --target MotifNativeEvaluate
swift build --package-path swift --target MotifNativeServe
swift test --package-path swift
