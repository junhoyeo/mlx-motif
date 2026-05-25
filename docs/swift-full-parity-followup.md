# Swift full-parity follow-up scope

This follow-up PR is the stacked continuation after PR #7. It keeps the Swift
parity effort in one branch while separating **implemented native Swift runtime
surfaces** from **performance-proven custom-kernel parity**.

## Capability matrix

| Requested capability | Follow-up PR status | Evidence / remaining gate |
| --- | --- | --- |
| Native Swift MLX token generation | Implemented reference path | `MotifMLXBackend(modelDirectory:)` and `MotifMLXNativeRuntime` load a converted checkpoint, apply the tokenizer/chat template, and stream through `MLXLMCommon.generate`. Requires local converted weights/tokenizer. |
| Full checkpoint tensor loading into Swift model weights | Wired through MLXLMCommon | `MotifMLXNativeRuntime.load` decodes `config.json`, builds `MotifMLXModel`, calls `loadWeights`, and applies quantization metadata. Real checkpoint validation remains local because weights are not vendored. |
| Real tokenizer/chat-template execution in Swift | Wired | `MotifMLXChatInputProcessor` uses swift-tokenizers `applyChatTemplate` with Motif EOS/chat-template metadata. |
| Full Motif decoder forward pass parity | Implemented as MLX Swift reference graph | Decoder layers run embeddings, RMSNorm, PolyNorm MLP, vanilla DiffAttn, grouped DiffAttn, final norm, and lm head. Full-logit parity still needs golden real-checkpoint fixtures. |
| Python-equivalent grouped differential attention execution | Implemented reference path | Grouped attention now routes through Swift 4-slot split/update logic and `MotifGDAPostSplit` / `MotifSDPADualV` reference wrappers. |
| Working Swift custom Metal kernels equivalent to Python kernels | Reference wrappers plus gated manifest | `MotifMetalKernels` enumerates `polynorm`, `gda_post`, `gda_post_split`, `sdpa_dual_v`, `sdpa_dual_v_q4`; GDA/dual-V wrappers are callable reference paths. Direct Metal execution remains disabled until Python fixture parity + same-machine benchmarks pass. |
| Quantized KV cache runtime parity | Implemented opt-in q4/q8 bridge | `MLX_MOTIF_4SLOT_CACHE=q4|q8` selects `MotifGroupedQuantizedKVCache`, stores packed tuples, and can fetch dequantized slices for the Swift model. Direct packed in-kernel speed parity remains gated. |
| `sdpa_dual_v_q4` runtime path | Implemented reference bridge, not direct Metal | Quantized cache can expose packed tuples and `MotifSDPADualVQ4.reference` dequantizes them before shared-QK dual-V SDPA. Direct Python-equivalent packed Metal dispatch still needs fixtures/benchmarks. |
| HF→MLX conversion flow | Python flow remains canonical | Existing `mlx-motif convert` remains the conversion command; Swift runtime consumes its output. A Swift-native converter is intentionally not duplicated. |
| Quantization presets | Load metadata honored | Swift load path passes `config.quantization` / per-layer quantization into `loadWeights`; Python remains the source for producing presets. |
| Python server feature parity | Native Swift OpenAI-compatible server added | `MotifNativeServe` exposes `GET /v1/models` and `POST /v1/chat/completions` with streaming SSE and Motif `<think>` event handling. Production hardening/load tests remain a gate. |
| Speculative decoding | Not implemented in Swift native path | Existing Python wiring remains the source of truth; Swift needs an MLXLMCommon-compatible draft-model API before enabling. |
| Perplexity/e2e quality checks | Native CLI added | `MotifNativeEvaluate --mode perplexity|bench` and `scripts/bench_swift_native.py --mode ...` provide local quality/latency harnesses. Checked-in parity results require local weights/hardware. |
| Long-context benchmarks | Harness path added | `scripts/bench_swift_native.py` accepts long prompts/model dirs and cache modes. No checked-in long-context result without model weights/hardware run. |
| Performance parity against Python | Measurement harness added; not claimed | Custom Metal direct dispatch, packed q4 SDPA, and same-machine Python-vs-Swift benchmark evidence are still required before declaring speed parity. |

## Runtime feature flags shared with Python

The Swift MLX runtime reads the same knobs as the Python backend where possible:

```bash
MLX_MOTIF_4SLOT_CACHE=1      # grouped four-slot fp cache
MLX_MOTIF_4SLOT_CACHE=q4     # grouped four-slot q4 packed cache with dequant bridge
MLX_MOTIF_4SLOT_CACHE=q8     # grouped four-slot q8 packed cache with dequant bridge
MLX_MOTIF_DUAL_V=0           # disable dual-V wrapper routing
MLX_MOTIF_QUANT_SDPA=0       # keep q4/q8 cache on the dequant bridge path
MLX_MOTIF_DISABLE_KERNELS=1  # force reference routing
```

## Why not mark performance parity complete?

The repository does not vendor Motif weights, and a PR should not overclaim
runtime/performance parity without golden numeric fixtures and same-machine
benchmarks. This PR closes the Swift-side runtime surface gaps (server, eval,
4-slot cache, q4/q8 cache bridge, dual-V/GDA wrapper routing) while keeping the
remaining direct-Metal and measured-speed gates explicit.

## Verification commands

```bash
scripts/verify_swift.sh
scripts/verify_swift_mlx.sh
uv run ruff check src tests scripts/bench_swift_native.py
uv run ruff format --check src tests scripts/bench_swift_native.py
uv run pytest -q
git diff --check
```

With a converted checkpoint:

```bash
MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeGenerate \
  --model ./out/motif-12.7b-q4 \
  --prompt "Explain grouped differential attention in one sentence." \
  --max-tokens 64 \
  --temperature 0

MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeEvaluate \
  --model ./out/motif-12.7b-q4 \
  --mode perplexity \
  --text-file ./fixtures/eval.txt \
  --max-tokens 2048

MLX_MOTIF_4SLOT_CACHE=q4 MOTIFKIT_ENABLE_MLX=1 scripts/bench_swift_native.py \
  --model ./out/motif-12.7b-q4 \
  --mode bench \
  --prompt "Explain grouped differential attention in one sentence." \
  --max-tokens 64

MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeServe \
  --model ./out/motif-12.7b-q4 \
  --host 127.0.0.1 \
  --port 8080
```
