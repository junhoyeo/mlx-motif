# Benchmark sweep artifacts

`benchmark-sweep` reports are the source of truth for Swift-vs-Python performance parity claims. A report is only valid for the exact commit, host, model paths, prompt lengths, cache cells, run counts, and thermal conditions recorded in its JSON.

## Local smoke

```bash
MOTIFKIT_ENABLE_MLX=1 uv run python scripts/bench_sweep.py \
  --model motif-2.6b-q4=.models/motif-2.6b-mlx-q4 \
  --backend python \
  --backend swift \
  --cache-cell q4_bridge:q4,quant_sdpa=0,disable_kernels=1 \
  --cache-cell q4_direct:q4,quant_sdpa=1 \
  --prompt-lens 500 1600 \
  --max-tokens 128 \
  --n-runs 1 \
  --warmup-runs 1 \
  --output artifacts/benchmarks/smoke.json \
  --markdown artifacts/benchmarks/smoke.md
```

## Full intended sweep

```bash
MOTIFKIT_ENABLE_MLX=1 uv run python scripts/bench_sweep.py \
  --manifest docs/benchmarks/models.local.example.json \
  --backend python \
  --backend swift \
  --cache-cell baseline:0 \
  --cache-cell four_slot_fp:1 \
  --cache-cell q4_bridge:q4,quant_sdpa=0,disable_kernels=1 \
  --cache-cell q4_direct:q4,quant_sdpa=1 \
  --cache-cell q8_direct:q8,quant_sdpa=1 \
  --prompt-lens 500 3000 16000 \
  --max-tokens 64 \
  --n-runs 5 \
  --warmup-runs 1 \
  --output artifacts/benchmarks/full-sweep.json \
  --markdown artifacts/benchmarks/full-sweep.md
```

## Checked-in target sweep

The current target sweep evidence is
[`benchmark-sweep-target-20260527T103000Z.md`](benchmark-sweep-target-20260527T103000Z.md)
(raw JSON alongside it). It covers 2.6B q4 and 12.7B q4 on this local Apple
M1 Max at p500/p3000/p16000 with `n_runs=1`, `max_tokens=8`, `q4_bridge`, and
`q4_direct`. It is a target-matrix smoke, not a final repeated thermal parity
claim. The report shows Swift remains below Python for most q4-direct cells, so
performance parity stays gated. For the 12.7B reasoning model the Swift and
Python cells render different prompt token counts at the same target bucket
(the Swift template fallback drops the hardcoded reasoning preamble), so those
`swift_vs_python` rows are flagged in the report and must not be read as a clean
head-to-head.

## CI and artifact expectations

Normal PR CI runs a dry sweep only because Motif weights are not vendored. The manual `Benchmark sweep` workflow is for local/self-hosted Apple Silicon runners with checkpoint directories already present or restored from trusted infrastructure.

The workflow uploads:

- `benchmark-sweep-json`
- `benchmark-sweep-md`
- `benchmark-raw-logs`

Do not update parity documentation from a dry-run report. Use a real report with `config.dry_run == false` and compare Swift `q4_direct` cells against Python cells for the same model and prompt length.

The `swift_vs_python` speedup column is not a steady-state throughput ratio: the Python cell discards its first decode step (steady-state decode tok/s) while the Swift cell's tok/s includes the compile-heavy first generated token. At low `--max-tokens` this dominates and makes Swift look artificially slow. Only draw throughput conclusions from a report run with a high `--max-tokens` and `--warmup-runs 1`, and treat the ratio as approximate even then.
