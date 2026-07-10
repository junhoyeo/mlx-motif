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
| Working Swift custom Metal kernels equivalent to Python kernels | Implemented default-on direct Metal path | `MotifMetalKernels` now marks PolyNorm, GDA post/split, `sdpa_dual_v`, and `sdpa_dual_v_q4` as `metalReady`/default-enabled. `MLX_MOTIF_DISABLE_KERNELS=1` forces reference routing for differential checks. |
| Quantized KV cache runtime parity | Implemented packed q4/q8 runtime path | `MLX_MOTIF_4SLOT_CACHE=q4|q8` selects `MotifGroupedQuantizedKVCache`; decode-time q4 attention passes packed tuples directly to `MotifSDPADualVQ4.apply` instead of materializing the full dequantized cache. |
| `sdpa_dual_v_q4` runtime path | Implemented direct packed Metal dispatch | Swift ports the Python packed q4/q8 Metal source and runtime tests compare direct packed output against the dequant bridge on deterministic decode fixtures. |
| HF→MLX conversion flow | Python flow remains canonical | Existing `mlx-motif convert` remains the conversion command; Swift runtime consumes its output. A Swift-native converter is intentionally not duplicated. |
| Quantization presets | Load metadata honored | Swift load path passes `config.quantization` / per-layer quantization into `loadWeights`; Python remains the source for producing presets. |
| Python server feature parity | Native Swift OpenAI-compatible server added | `MotifNativeServe` exposes `GET /v1/models` and `POST /v1/chat/completions` with streaming SSE and Motif `<think>` event handling. Production hardening/load tests remain a gate. |
| Speculative decoding | Implemented in Swift native path (greedy-only, batched verification) | `MotifMLXNativeRuntime.speculativeGenerate` and `MotifNativeGenerate --speculative --speculative-draft-model ...` run real draft-propose / batched-target-verify decoding: persistent draft and target KV caches, one `[1, K+1]` target forward per K-token draft block, cache `trim` past the first mismatch. Greedy accept rule preserves the target's greedy output; `MotifSpeculativeDecodingTests` asserts token-for-token equality with plain greedy decode and fewer target forwards than tokens. |
| Perplexity/e2e quality checks | Implemented with checked-in evidence harness | `MotifNativeEvaluate --mode perplexity|bench|logits`, `scripts/perplexity.py --json`, and `scripts/swift_python_hard_parity.py` emit raw same-machine JSON/markdown evidence. |
| Long-context benchmarks | Implemented with evidence harness | `scripts/swift_python_hard_parity.py` runs Python decode and Swift q4 decode on generated long-context prompts and stores raw results under `docs/benchmarks/`. |
| Performance parity against Python | Harnessed, not yet proven as a parity claim | The hard-parity report records Python and Swift cells on one host, but the checked-in local long-context result shows Swift slower than Python. Use `scripts/bench_sweep.py` and the manual benchmark workflow to prove or reject parity for each model/hardware matrix before making performance claims. |

## Runtime feature flags shared with Python

The Swift MLX runtime reads the same knobs as the Python backend where possible:

```bash
MLX_MOTIF_4SLOT_CACHE=1      # grouped four-slot fp cache
MLX_MOTIF_4SLOT_CACHE=q4     # grouped four-slot q4 packed cache with direct packed sdpa_dual_v_q4 by default
MLX_MOTIF_4SLOT_CACHE=q8     # grouped four-slot q8 packed cache with direct packed sdpa_dual_v_q4 by default
MLX_MOTIF_DUAL_V=0           # disable dual-V wrapper routing
MLX_MOTIF_QUANT_SDPA=0       # keep q4/q8 cache on the dequant bridge path
MLX_MOTIF_DISABLE_KERNELS=1  # force reference routing
```

## Why performance parity remains gated

This hard-parity follow-up adds the direct Swift Metal implementations and a
same-machine evidence harness, but the current local long-context report does
not prove Swift equals Python throughput. The raw reports under
`docs/benchmarks/` are the source of truth for performance claims; if a machine,
checkpoint, prompt length, or thermal state changes, rerun the sweep before
carrying any parity claim forward.

For full sweeps across model sizes and hardware, use
[`docs/benchmarks/README.md`](benchmarks/README.md) and the manual
`Benchmark sweep` GitHub Actions workflow. Normal PR CI only validates dry-run
plumbing because model weights are not vendored.


Latest checked-in local evidence: [`docs/benchmarks/swift-python-hard-parity-20260526T091532Z.md`](docs/benchmarks/swift-python-hard-parity-20260526T091532Z.md) (raw JSON alongside it).

CI guardrails:

- GitHub Actions runs the default Swift package verification via `scripts/verify_swift.sh`.
- Python pytest validates the checked-in hard-parity report shape and key quality/runtime tolerances so the evidence cannot silently rot when the stack is rebased.
- Runtime MLX/Metal tests remain local-only because hosted runners do not have the local Motif checkpoint used by `scripts/swift_python_hard_parity.py`.

## Verification commands

```bash
scripts/verify_swift.sh
scripts/verify_swift_mlx.sh  # also builds/installs mlx.metallib and runs runtime kernel tests
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

MOTIFKIT_ENABLE_MLX=1 swift run --package-path swift MotifNativeEvaluate \
  --model ./out/motif-12.7b-q4 \
  --mode logits \
  --text "Explain grouped differential attention." \
  --top-k 10

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
