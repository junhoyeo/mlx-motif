# Codebase tour

What lives where. This is the orientation map for contributors — the README front page only points here.

```
src/mlx_motif/
  __init__.py        # exports load, Model, ModelArgs, __version__
  __main__.py        # CLI: convert | generate | serve
  model.py           # PolyNorm, MotifAttention (vanilla + GDA, AttnPath enum),
                     # MotifMLP, MotifModel, Model + fuse_qkv() + make_cache()
  kernels/           # Custom Metal kernels package + pure-MLX references:
    __init__.py      #   re-exports + pointer to docs/experiments/ for removed kernels
    attention.py     #   sdpa_dual_v, sdpa_dual_v_q4
    gda.py           #   gda_post, gda_post_split
    mlp.py           #   polynorm, _dequant_probe (test helper)
    _common.py       #   shared Python helpers
  cache.py           # MotifGroupedKVCacheBase + MotifGroupedKVCache +
                     # MotifGroupedQuantizedKVCache (4-slot variants for
                     # the differential pattern)
  loader.py          # mlx_motif.load() — wraps mlx-lm load_model + fuse_qkv;
                     # registers all EOS ids from generation_config.json
  convert.py         # HF → MLX safetensors converter
  quant.py           # mixed-precision quantization presets
  server.py          # OpenAI-compatible HTTP server + ThinkFilter for
                     # <think> stream handling (visible|hidden|captured)

tests/
  test_model.py                    # smoke: forward shapes, sanitize, AttnPath resolution
  test_quant.py                    # quantization predicates
  test_parity.py                   # numerical parity vs HF reference
  test_think_filter.py             # server <think>-stream filter
  test_kernels.py                  # polynorm correctness
  test_kernels_gda.py              # gda_post correctness
  test_kernels_gda_post_split.py   # gda_post_split correctness
  test_kernels_sdpa_dual_v.py      # dual-V SDPA correctness (+ GQA)
  test_kernels_sdpa_dual_v_q4.py   # quantized-input dual-V SDPA (+ GQA)
  test_dequant_probe.py            # standalone 4/8-bit unpack probe
  test_grouped_cache.py            # 4-slot cache correctness

scripts/
  bench_decode_e2e.py  # end-to-end decode benchmark (the headline-numbers harness)
  perplexity.py        # PPL eval for quantization sanity-checks

examples/
  convert.py    # end-to-end conversion script
  generate.py   # end-to-end generation script

docs/
  blog-quantized-attention-on-m1-max.md  # design-decisions writeup for the q4 attention path
  sdpa_dual_v_q4_design.md               # kernel-level design notes for sdpa_dual_v_q4
  codebase-tour.md                       # this file
  experiments/                           # negative-result kernels + writeups (see experiments/README.md)
```

For the negative-result kernels (`polynorm_mul`, `sdpa_dual_v_2pass`, `qmv_dual_q4`, `gda_decode`) that used to live in `kernels/` but were removed, see [`docs/experiments/`](experiments/README.md). Each removed kernel has a writeup with the snippet, bench numbers, and "when this might win on different hardware" — recoverable from git history when needed.
