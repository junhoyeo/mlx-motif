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
  - a placeholder native backend error type for non-MLX builds
- `MotifChatApp` is a SwiftUI macOS app scaffold with chat and runtime panels. Its default app target uses the OpenAI-compatible endpoint so non-MLX builds stay lightweight. When built with `MOTIFKIT_ENABLE_MLX=1`, it also links `MotifKitMLX` and can stream directly from a converted local checkpoint.
- `MotifKitMLX` is disabled by default so the package stays buildable on the repo's current Xcode 16.2 / Swift 6.0.3 toolchain. With `MOTIFKIT_ENABLE_MLX=1`, `MotifMLXBackend(modelDirectory:)` now owns the native reference path: build the Motif decoder, load safetensors through MLXLMCommon, apply the tokenizer/chat template via swift-tokenizers, and stream generated events. Grouped four-slot q4/q8 caches are opt-in, and decode-time q4 uses direct packed custom Metal by default with `MLX_MOTIF_DISABLE_KERNELS=1` as the reference fallback.

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

This server-backed path remains available. The optional native Swift MLX path can be exercised from the package overlay with a converted checkpoint; rerun `scripts/swift_python_hard_parity.py` on the target machine before making fresh performance claims.

To run the SwiftUI chat app directly against a converted checkpoint instead of a `/v1` endpoint:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifChatApp
```

Then choose **Native MLX checkpoint** in the Runtime panel and set the converted checkpoint directory, for example `~/.models/motif-2.6b-mlx-q4` (the app's default, overridable via `MOTIF_MODEL_DIR`).

For smoke and packaging steps, see [`../docs/swift-app-smoke.md`](../docs/swift-app-smoke.md).

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

`MotifKitMLX` now includes direct Swift custom Metal wrappers for PolyNorm, GDA post/split, `sdpa_dual_v`, and packed `sdpa_dual_v_q4`. Metal execution is default-on for supported decode shapes; force reference routing with:

```bash
MLX_MOTIF_DISABLE_KERNELS=1 MOTIFKIT_ENABLE_MLX=1 swift test --package-path swift --filter MotifKitMLXTests
```

The pinned `mlx-swift` package does not materialize `default.metallib` under SwiftPM, so run `scripts/build_mlx_swift_metallib.sh` after MLX builds or use `scripts/verify_swift_mlx.sh`, which performs that step and then runs runtime kernel tests.

## Native reference generation CLI

With a converted checkpoint from `mlx-motif convert`, run:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeGenerate \
  --model ./out/motif-12.7b-q4 \
  --prompt "Explain grouped differential attention in one sentence." \
  --max-tokens 64 \
  --temperature 0
```

Use `--speculative --speculative-draft-model <dir> --speculative-draft-tokens 4 --json` to run the Swift target/draft speculative decoder and emit acceptance metrics. The decoder is greedy-only (`--temperature 0`): the draft proposes K tokens from a persistent draft KV cache and the target verifies each block with a single batched `[1, K+1]` forward against a persistent target KV cache, trimming rejected rows, so accepted blocks advance up to K+1 tokens per target forward (`targetModelSteps` < `targetTokens` in the metrics). Output follows the greedy accept rule (draft token accepted iff it equals the target argmax), i.e. it is the target model's own greedy continuation; wall-clock speedup additionally requires a draft model meaningfully cheaper than the target.

## Native eval/bench CLI

`MotifNativeEvaluate` provides local parity-lab entry points for quality and
latency runs:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeEvaluate \
  --model ./out/motif-12.7b-q4 \
  --mode perplexity \
  --text-file ./fixtures/eval.txt \
  --max-tokens 2048

MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeEvaluate \
  --model ./out/motif-12.7b-q4 \
  --mode logits \
  --text "Explain grouped differential attention." \
  --top-k 10

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
MLX_MOTIF_4SLOT_CACHE=q4  # grouped q4 packed cache + direct packed sdpa_dual_v_q4
MLX_MOTIF_4SLOT_CACHE=q8  # grouped q8 packed cache + direct packed sdpa_dual_v_q4
MLX_MOTIF_DUAL_V=0        # disable dual-V wrapper routing
MLX_MOTIF_QUANT_SDPA=0    # keep q4/q8 cache on the dequant bridge path
MLX_MOTIF_DISABLE_KERNELS=1
```
