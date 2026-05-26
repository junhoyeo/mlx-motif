#!/usr/bin/env bash
set -euo pipefail

# Build the default MLX Swift Metal library that the pinned SwiftPM build does
# not currently materialize as a resource bundle. mlx-swift's C++ loader first
# looks for colocated mlx.metallib next to the running binary/test bundle, so we
# install the generated library into SwiftPM debug products after a build.

cd "$(dirname "$0")/.."

configuration="${SWIFT_CONFIGURATION:-debug}"
triple_dir="swift/.build/arm64-apple-macosx/${configuration}"
metal_dir="swift/.build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
include_dir="swift/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels"
out_dir=".build-motif/mlx-swift-metallib/${configuration}"

if [[ ! -d "$metal_dir" ]]; then
  echo "error: mlx-swift generated Metal sources not found at $metal_dir" >&2
  echo "hint: run MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifKitMLX first" >&2
  exit 1
fi

rm -rf "$out_dir"
mkdir -p "$out_dir"

for source in "$metal_dir"/*.metal; do
  name="$(basename "$source" .metal)"
  xcrun -sdk macosx metal -std=metal3.1 \
    -I "$metal_dir" \
    -I "$include_dir" \
    -c "$source" \
    -o "$out_dir/${name}.air"
done

xcrun -sdk macosx metallib "$out_dir"/*.air -o "$out_dir/mlx.metallib"

if [[ -d "$triple_dir" ]]; then
  cp "$out_dir/mlx.metallib" "$triple_dir/mlx.metallib"
  shopt -s nullglob
  for bundle in "$triple_dir"/*.xctest; do
    mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"
    cp "$out_dir/mlx.metallib" "$bundle/Contents/MacOS/mlx.metallib"
    cp "$out_dir/mlx.metallib" "$bundle/Contents/Resources/default.metallib"
  done
  shopt -u nullglob
fi

printf '%s\n' "$out_dir/mlx.metallib"
