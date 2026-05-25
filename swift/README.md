# Swift Motif workspace

This directory is the native Swift entry point for the Motif-on-Mac stack:

```text
MotifChatApp (SwiftUI)
  -> MotifKit (chat/runtime abstractions + OpenAI-compatible bridge)
  -> MotifKitMLX (optional MLX Swift overlay; enabled with MOTIFKIT_ENABLE_MLX=1)
  -> MLX Swift / mlx-swift-lm
  -> Motif custom Metal kernels
```

## Current state

- `MotifKit` builds without external dependencies and provides:
  - shared chat message/generation types
  - `<think>` stream filtering compatible with the Python server behavior
  - an OpenAI-compatible streaming backend for the existing `mlx-motif serve`
  - a placeholder native backend error type for the future in-process MLX path
- `MotifChatApp` is a SwiftUI macOS app scaffold with chat and runtime panels.
- `MotifKitMLX` is scaffolded but disabled by default so the package stays buildable on the repo's current Xcode 16.2 / Swift 6.0.3 toolchain.

## Verify

```bash
swift test --package-path swift
swift build --package-path swift --target MotifChatApp
```

## Run against today's Python backend

In one terminal:

```bash
mlx-motif serve --model ./out/motif-12.7b-q4 --port 8080
```

In another terminal / Xcode session, run `MotifChatApp` and keep the endpoint as:

```text
http://127.0.0.1:8080/v1
```

## Enable MLX overlay work

The current machine has Swift 6.0.3, while latest `mlx-swift-lm` 3.31.3 requires Swift tools 6.1. For this toolchain the optional overlay pins `mlx-swift-lm` 2.30.6 and `mlx-swift` 0.30.6:

```bash
MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifKitMLX
```

Once the repo moves to Xcode 16.3+ / Swift 6.1+, update the pins in `Package.swift` to `mlx-swift-lm` 3.31.3+ and `mlx-swift` 0.31.3+.

## Custom Metal kernel porting

`MotifKitMLX` now includes a Swift-side kernel manifest mirroring the Python custom kernels in `src/mlx_motif/kernels/` and a buildable PolyNorm wrapper scaffold. Custom Metal execution is intentionally disabled by default; native runtime code uses reference MLX Swift ops unless benchmark/parity work opts in with:

```bash
MOTIFKIT_ENABLE_MLX=1 MOTIFKIT_ENABLE_EXPERIMENTAL_METAL_KERNELS=1 swift test --package-path swift --filter MotifKitMLXTests
```

Set `MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1` only on machines where the MLX Swift runtime can load its default metallib; otherwise the tests stay manifest/build-only. Before enabling any kernel in the app path, add golden fixtures from the Python reference and record TTFT/decode benchmark deltas for the matching Motif shapes.
