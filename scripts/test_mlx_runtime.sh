#!/usr/bin/env bash
set -euo pipefail

# Run the MLX-runtime Swift tests that actually execute Metal ops (numerical
# parity fixtures, cache reuse, speculative decoding, kernel wrappers).
#
# These tests are gated behind MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1 so GPU-less /
# metallib-less CI still passes (it type-checks the same code via the ungated
# build). They are NOT broken when they "skip" under a plain `swift test` — they
# are opt-in. This wrapper is the supported way to run them: it builds the
# colocated mlx.metallib the runtime loader needs, then runs the suite with the
# gate + MLX overlay enabled.
#
# Usage:
#   scripts/test_mlx_runtime.sh                       # all MLX-runtime tests
#   scripts/test_mlx_runtime.sh --filter MotifQKVFusionParityTests
#   SWIFT_CONFIGURATION=release scripts/test_mlx_runtime.sh
#
# Any extra args are forwarded to `swift test` (e.g. --filter ...).

cd "$(dirname "$0")/.."

echo "==> Building colocated mlx.metallib"
SWIFT_CONFIGURATION="${SWIFT_CONFIGURATION:-debug}" bash scripts/build_mlx_swift_metallib.sh >/dev/null

echo "==> Running MLX-runtime tests (MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1)"
cd swift
exec env MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1 MOTIFKIT_ENABLE_MLX=1 \
  swift test "$@"
