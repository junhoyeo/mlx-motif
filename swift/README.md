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
- `MotifChatApp` is a SwiftUI macOS app scaffold with chat and runtime panels. Its default app target still uses the OpenAI-compatible endpoint so non-MLX builds stay lightweight.
- `MotifKitMLX` is disabled by default so the package stays buildable on the repo's current Xcode 16.2 / Swift 6.0.3 toolchain. With `MOTIFKIT_ENABLE_MLX=1`, `MotifMLXBackend(modelDirectory:)` now owns the native reference path: build the Motif decoder, load safetensors through MLXLMCommon, apply the tokenizer/chat template via swift-tokenizers, and stream generated events. Grouped four-slot cache and q4/q8 cache bridges are opt-in; direct custom-Metal speed parity remains gated by fixtures and benchmarks.

## Verify

```bash
swift test --package-path swift
swift build --package-path swift --target MotifChatApp
```

Optional MLX overlay checks are useful for porting work, but they are separate from the default app build:

```bash
MOTIFKIT_ENABLE_MLX=1 swift test --package-path swift --filter MotifKitMLXTests
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

This server-backed path remains the default app path. The optional native Swift MLX path can be exercised from the package overlay with a converted checkpoint; do not describe it as Python-performance-equivalent until the custom Metal/q4 cache gates in `docs/swift-full-parity-followup.md` pass.

## Enable MLX overlay work

The current machine has Swift 6.0.3, while latest `mlx-swift-lm` 3.31.3 requires Swift tools 6.1. For this toolchain the optional overlay pins `mlx-swift-lm` 2.30.6 and `mlx-swift` 0.30.6:

```bash
MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifKitMLX
MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifNativeGenerate
MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifNativeEvaluate
MOTIFKIT_ENABLE_MLX=1 swift build --package-path swift --target MotifNativeServe
```

Once the repo moves to Xcode 16.3+ / Swift 6.1+, update the pins in `Package.swift` to `mlx-swift-lm` 3.31.3+ and `mlx-swift` 0.31.3+.

## Custom Metal kernel porting

`MotifKitMLX` now includes a Swift-side kernel manifest mirroring the Python custom kernels in `src/mlx_motif/kernels/` and a buildable PolyNorm wrapper scaffold. Custom Metal execution is intentionally disabled by default; native runtime code uses reference MLX Swift ops unless benchmark/parity work opts in with:

```bash
MOTIFKIT_ENABLE_MLX=1 MOTIFKIT_ENABLE_EXPERIMENTAL_METAL_KERNELS=1 swift test --package-path swift --filter MotifKitMLXTests
```

Set `MOTIFKIT_RUN_MLX_RUNTIME_TESTS=1` only on machines where the MLX Swift runtime can load its default metallib; otherwise the tests stay manifest/build-only. Before enabling any kernel in the app path, add golden fixtures from the Python reference and record TTFT/decode benchmark deltas for the matching Motif shapes.

## Native reference generation CLI

With a converted checkpoint from `mlx-motif convert`, run:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeGenerate \
  --model ./out/motif-12.7b-q4 \
  --prompt "Explain grouped differential attention in one sentence." \
  --max-tokens 64 \
  --temperature 0
```

This path is correctness-first: it uses MLX Swift reference ops by default. Performance parity with Python requires the follow-up custom Metal and quantized-cache gates documented in `docs/swift-full-parity-followup.md`.

## Native eval/bench CLI

`MotifNativeEvaluate` provides local parity-lab entry points for quality and
latency runs:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeEvaluate \
  --model ./out/motif-12.7b-q4 \
  --mode perplexity \
  --text-file ./fixtures/eval.txt \
  --max-tokens 2048

MLX_MOTIF_4SLOT_CACHE=q4 MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeEvaluate \
  --model ./out/motif-12.7b-q4 \
  --mode bench \
  --prompt "Explain grouped differential attention in one sentence." \
  --max-tokens 64 \
  --temperature 0
```

The Python helper wraps the same commands and records subprocess timing:

```bash
MLX_MOTIF_4SLOT_CACHE=q4 MOTIFKIT_ENABLE_MLX=1 scripts/bench_swift_native.py \
  --model ./out/motif-12.7b-q4 \
  --mode bench \
  --cache-mode q4
```

## Native OpenAI-compatible server

With a converted checkpoint, the Swift server target exposes a small
OpenAI-compatible surface for the macOS app and parity testing:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeServe \
  --model ./out/motif-12.7b-q4 \
  --host 127.0.0.1 \
  --port 8080 \
  --think-mode hidden
```

Supported routes are `GET /v1/models` and `POST /v1/chat/completions`, including
SSE streaming when `stream: true`. This is a native parity surface, not yet a
production-hardened replacement for the Python server.

## Runtime cache/kernel feature flags

The Swift MLX path reads Python-compatible flags:

```bash
MLX_MOTIF_4SLOT_CACHE=1   # grouped fp four-slot cache
MLX_MOTIF_4SLOT_CACHE=q4  # grouped q4 packed cache with dequant bridge
MLX_MOTIF_4SLOT_CACHE=q8  # grouped q8 packed cache with dequant bridge
MLX_MOTIF_DUAL_V=0        # disable dual-V wrapper routing
MLX_MOTIF_QUANT_SDPA=0    # keep q4/q8 cache on the dequant bridge path
MLX_MOTIF_DISABLE_KERNELS=1
```
