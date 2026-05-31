# Swift Motif performance roadmap

Goal: move `mlx-motif` from Python+MLX to a native macOS stack while preserving the performance wins from the existing custom Metal kernels.

## Architecture

```text
MotifChat.app
  SwiftUI chat, model/runtime controls, diagnostics
    -> MotifKit
       shared chat types, streaming, thinking-filter policy, backend protocol
    -> MotifKitMLX
       native Motif model implementation over MLX Swift / mlx-swift-lm
    -> custom Metal kernels
       polynorm, gda_post_split, sdpa_dual_v, sdpa_dual_v_q4
```

## Researched constraints

- MLX Swift is the first-class Apple Silicon path for Swift apps: <https://github.com/ml-explore/mlx-swift>
- `mlx-swift-lm` provides LLM/VLM model infrastructure and model-type registries: <https://github.com/ml-explore/mlx-swift-lm>
- MLX Chat Example demonstrates a SwiftUI app with MLX model loading, streaming, cancellation, and memory handling: <https://github.com/ml-explore/mlx-swift-examples/tree/main/Applications/MLXChatExample>
- The latest `mlx-swift-lm` 3.31.3 package requires Swift tools 6.1; this repo currently has Xcode 16.2 / Swift 6.0.3. The optional overlay therefore starts from `mlx-swift-lm` 2.30.6 until the toolchain is upgraded.

## Port order

1. **Parity harness**
   - Export small Python golden fixtures for `ThinkFilter`, `PolyNorm`, GDA post, and attention outputs.
   - Add Swift tests that load the fixtures before changing performance paths.

2. **Swift reference model**
   - `MotifModelConfiguration` -> full `config.json` decoder.
   - `PolyNorm` reference in MLX Swift ops.
   - `MotifMLP` and RMSNorm wiring.
   - Vanilla DiffAttn for 2.6B shape.
   - Grouped Differential Attention reference path for 12.7B shape.

3. **Generation integration**
   - Register `model_type: motif` with the `mlx-swift-lm` factory.
   - Load local converted MLX safetensors first.
   - Apply Motif chat template and EOS IDs from `generation_config.json`.
   - Stream tokens into `MotifChatApp` through `MotifChatBackend`.

4. **Performance parity**
   - Port `polynorm` Metal kernel.
   - Keep `gda_post_split` and `sdpa_dual_v` callable through Swift reference wrappers until Python golden fixtures approve direct Metal dispatch.
   - Use `MotifGroupedKVCache` and `MotifGroupedQuantizedKVCache` behind `MLX_MOTIF_4SLOT_CACHE=1|q4|q8`.
   - Keep `sdpa_dual_v_q4` on the packed-cache dequant bridge until the direct packed Metal kernel has Swift fixture coverage.

5. **Benchmarks**
   - Match Python prompts: short, long, xlong.
   - Report TTFT, decode tok/s, peak resident memory, and parity status.
   - Compare fallback MLX Swift reference vs custom kernels.

## UI reference direction

Use OSS chat apps as interaction references, not copied assets:

- Open WebUI: rich markdown, local-first model/server UX, future RAG affordances.
- LibreChat: provider/custom endpoint abstraction and import/export thinking.
- LobeChat/LobeHub: polished AI workspace, artifacts, thinking panels, chat/document modes.
- Chatbot UI: simple familiar sidebar + composer baseline.
- MLXChatExample / iChat / macMLX: native SwiftUI model loading, streaming, thinking rendering, diagnostics.

## Stop conditions for a performant native backend

- Native Swift path can load a converted Motif q4 checkpoint.
- First generated tokens match Python reference for deterministic greedy prompts.
- `sdpa_dual_v` and `gda_post_split` pass fixture tests against Python outputs.
- Long prompt decode speed is within 10% of Python `mlx-motif` custom-kernel path on the same machine, or the delta is explained by a measured MLX Swift API/runtime gap.

## Follow-up PR status after PR #7

The follow-up branch advances the next vertical slice:

- `MotifMLXModel` now conforms to `LLMModel` and runs a correctness-first decoder path through MLX Swift reference ops.
- `MotifMLXNativeRuntime` loads converted MLX checkpoint directories, tokenizer/chat template metadata, EOS IDs, and quantization metadata.
- `MotifMLXBackend(modelDirectory:)` streams native generation events when a real converted model directory is provided.
- `MotifNativeGenerate`, `MotifNativeEvaluate`, `MotifNativeServe`, and `scripts/bench_swift_native.py` provide generation, perplexity, local OpenAI-compatible serving, and benchmark entry points for same-machine parity runs.
- `MLX_MOTIF_4SLOT_CACHE=1|q4|q8` selects grouped fp/q4/q8 cache implementations in Swift. q4/q8 currently use a dequant bridge; direct packed-Metal speed parity is still a measured-performance gate.

Remaining gates for true Python parity are tracked in [`swift-full-parity-followup.md`](swift-full-parity-followup.md): direct custom Metal runtime enablement, packed `sdpa_dual_v_q4`, speculative decoding, checked-in perplexity/long-context evidence, and measured performance parity.
