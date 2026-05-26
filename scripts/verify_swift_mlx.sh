#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MOTIFKIT_ENABLE_MLX=1
swift build --package-path swift --target MotifKitMLX
swift build --package-path swift --target MotifNativeGenerate
swift build --package-path swift --target MotifNativeEvaluate
swift build --package-path swift --target MotifNativeServe
./scripts/build_mlx_swift_metallib.sh
swift test --package-path swift
MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1 swift test --package-path swift --filter MotifMetalKernelsTests
