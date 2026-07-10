# Codebase tour

What lives where. This is the orientation map for contributors — the README front page only points here.

```
src/mlx_motif/
  __init__.py        # exports load, Model, ModelArgs, __version__
  __main__.py        # CLI: convert | generate | serve | tools-demo
  model.py           # PolyNorm, MotifAttention (vanilla + GDA, AttnPath enum,
                     # per-slab fast-path SDPA for prefill/GQA), MotifMLP,
                     # MotifModel, Model + fuse_qkv() + make_cache() (4-slot
                     # fp16 cache is DEFAULT; MLX_MOTIF_4SLOT_CACHE=0 opts out)
  kernels/           # Custom Metal kernels package + pure-MLX references:
    __init__.py      #   re-exports + pointer to docs/experiments/ for removed kernels
    attention.py     #   sdpa_dual_v, sdpa_dual_v_q4 (runtime kv_len — no
                     #   per-token Metal recompile; q4/q8 packed-word shape guard)
    gda.py           #   gda_post, gda_post_split
    mlp.py           #   polynorm, _dequant_probe (test helper)
    _common.py       #   shared Python helpers
  cache.py           # MotifVanillaKVCache (slab-ordered V + meta_state marker
                     # for the 2.6B ungrouped path) + MotifGroupedKVCacheBase +
                     # MotifGroupedKVCache + MotifGroupedQuantizedKVCache (4-slot
                     # variants for the differential pattern; batched _update_4
                     # quantized writes; caches return full-capacity buffers)
  loader.py          # mlx_motif.load() — wraps mlx-lm load_model + fuse_qkv;
                     # registers all EOS ids from generation_config.json;
                     # resolves HF Hub repo ids via snapshot_download
  convert.py         # HF → MLX safetensors converter
  quant.py           # mixed-precision quantization presets
  server.py          # OpenAI-compatible HTTP server + ThinkFilter for
                     # <think> stream handling (visible|hidden|captured)
  tool_calls.py      # prompt-based tool/function calling: build_tools_preamble
                     # + parse_tool_call + run_tool_loop tool-EXECUTION loop
                     # (importable without MLX/weights)

tests/
  test_model.py                    # smoke: forward shapes, sanitize, AttnPath resolution
  test_quant.py                    # quantization predicates
  test_parity.py                   # numerical parity vs HF reference (opt-in via MLX_MOTIF_PARITY*)
  test_convert_config.py           # converted-checkpoint config emission
  test_think_filter.py             # server <think>-stream filter
  test_tool_calls.py               # prompt-based tool-call parser (build_preamble / parse)
  test_server_contract.py          # OpenAI-compatible server HTTP-contract structure
  test_kernels.py                  # polynorm correctness
  test_kernels_gda.py              # gda_post correctness
  test_kernels_gda_post_split.py   # gda_post_split correctness
  test_kernels_sdpa_dual_v.py      # dual-V SDPA correctness (+ GQA)
  test_kernels_sdpa_dual_v_q4.py   # quantized-input dual-V SDPA (+ GQA)
  test_dequant_probe.py            # standalone 4/8-bit unpack probe
  test_grouped_cache.py            # 4-slot cache correctness
  test_vanilla_slab_cache.py       # MotifVanillaKVCache slab-order + meta_state
  test_attention_equivalence.py    # fast-path vs reference attention equivalence
  test_eval_smoke.py               # eval_smoke.py pure-logic helpers (no MLX/model)
  test_benchmark_sweep.py          # bench_sweep.py harness logic
  test_benchmark_certification.py  # certification validator for benchmark sweep JSONs
  test_swift_parity_fixtures.py    # shared lightweight Swift-port parity fixtures
  test_swift_hard_parity_evidence.py    # required Swift<->Python parity evidence gates
  test_swift_python_perf_regression.py  # Swift-vs-Python q4 throughput floor sentinel
  test_swift_chat_app_package_verify.py # SwiftUI app packaging metadata verify

scripts/
  bench_decode_e2e.py       # end-to-end decode benchmark (the headline-numbers harness)
  bench_sweep.py            # multi-config benchmark sweep harness (certifiable JSONs)
  perplexity.py             # PPL eval for quantization sanity-checks (--json)
  eval_smoke.py             # pure-logic eval helpers (no MLX/model)
  bench_swift_native.py     # Swift native runtime benchmark entry point
  swift_python_hard_parity.py  # same-machine Swift-vs-Python parity evidence harness
  verify_swift.sh           # default Swift package verification (CI)
  verify_swift_mlx.sh       # builds/installs mlx.metallib + runs runtime kernel tests
  build_mlx_swift_metallib.sh  # compiles the custom Metal kernels into mlx.metallib
  package_swift_chat_app.sh    # packages the SwiftUI MotifChat.app bundle
  smoke_swift_chat_app.py      # SwiftUI chat-app packaging smoke test
  verify_swift_chat_app_package.py  # chat-app package metadata verifier

swift/
  Package.swift
  Sources/
    MotifKit/          # shared chat types, streaming, ThinkStreamFilter, backend
                       # protocol, MotifToolCalling (prompt-based tool calling),
                       # native + OpenAI-compatible backends, context budget
    MotifKitMLX/       # native Motif model over MLX Swift: MotifMLXModel,
                       # MotifMLXNativeRuntime, MotifMetalKernels (runtime kv_len,
                       # full-capacity fetches), MotifGroupedAttentionReference,
                       # MotifSpeculativeDecoding (batched verify + rejection
                       # sampling for temperature>0), 4-slot cache
    MotifChatApp/      # SwiftUI chat app (ChatStore, ContentView, GlassStyle, ...)
    MotifNativeGenerate/  # CLI: native generation (+ --speculative)
    MotifNativeEvaluate/  # CLI: perplexity | bench | logits
    MotifNativeServe/     # CLI: OpenAI-compatible server (tools parity)
    MotifDecodeBench/     # CLI: decode microbenchmark
  Tests/
    MotifKitTests/     # ThinkStreamFilter, tool calling, config, parity fixtures, ...
    MotifKitMLXTests/  # MLX runtime: speculative decoding, four-slot cache guard,
                       # EOS reconcile, cache reuse, Metal kernels, QKV fusion, ...

examples/
  convert.py    # end-to-end conversion script
  generate.py   # end-to-end generation script

docs/
  blog-quantized-attention-on-m1-max.md  # design-decisions writeup for the q4 attention path
  sdpa_dual_v_q4_design.md               # kernel-level design notes for sdpa_dual_v_q4
  codebase-tour.md                       # this file
  server-parity.md                       # Python/Swift OpenAI-compatible server parity
  ci.md                                  # CI workflow overview
  swift-motif-roadmap.md                 # Swift native-stack port roadmap
  swift-full-parity-followup.md          # Swift↔Python parity capability matrix
  swift-app-smoke.md                     # SwiftUI chat-app smoke-test notes
  benchmarks/                            # checked-in benchmark evidence (see benchmarks/README.md)
  hf-release/                            # Hugging Face release notes/assets
  experiments/                           # negative-result kernels + writeups (see experiments/README.md)
```

For the negative-result kernels (`polynorm_mul`, `sdpa_dual_v_2pass`, `qmv_dual_q4`, `gda_decode`) that used to live in `kernels/` but were removed, see [`docs/experiments/`](experiments/README.md). Each removed kernel has a writeup with the snippet, bench numbers, and "when this might win on different hardware" — recoverable from git history when needed.
