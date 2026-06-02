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


## Checked-in certified sweep

The current certified sweep evidence is
[`benchmark-sweep-certified-20260531T184554Z.md`](benchmark-sweep-certified-20260531T184554Z.md)
(raw JSON and raw logs alongside it). It covers 2.6B q4 and 12.7B q4 on the
local Apple M1 Max at p500/p3000/p16000 with `n_runs=5`, `warmup_runs=1`,
`max_tokens=64`, `q4_bridge`, and `q4_direct`. All 24 cells passed the
certification matrix.

This certified run is evidence, not a parity claim: clean 2.6B q4-direct
Swift-vs-Python speedups remain below 1.0x at every prompt length
(p500: 0.183x;
p3000: 0.311x;
p16000: 0.781x). The
12.7B rows still have prompt-token mismatches, so their ratios remain flagged
as non-clean comparisons.

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

## Checked-in QKV fusion probe

[`benchmark-sweep-qkv-fusion-20260527T112300Z.md`](benchmark-sweep-qkv-fusion-20260527T112300Z.md)
records the first opt-in Swift grouped-attention QKV fusion probe on the 12.7B
q4 checkpoint. On this local Apple M1 Max, `MLX_MOTIF_FUSE_QKV=1` improves the
Swift q4-direct cell by 1.05x at p500 and 1.20x at p3000, but it is still a
single-run probe and does not close the Python-vs-Swift gap from the target
sweep.


## Swift app readiness evidence

- Native backend smoke: [`swift-app-smoke-native-server-20260531T184436Z.json`](swift-app-smoke-native-server-20260531T184436Z.json)
- Package verification: [`swift-app-package-20260531T184436Z.json`](swift-app-package-20260531T184436Z.json)

These artifacts were produced on local Apple Silicon from `main` after PRs #18-#24 were squash-merged. They verify native generation, cancellation, native OpenAI-compatible server fallback, and ad-hoc package integrity. They are not a substitute for the interactive SwiftUI checklist in [`../swift-app-smoke.md`](../swift-app-smoke.md).

## CI and artifact expectations

Normal PR CI runs a dry sweep only because Motif weights are not vendored. The manual `Benchmark sweep` workflow is for local/self-hosted Apple Silicon runners with checkpoint directories already present or restored from trusted infrastructure.

The workflow uploads:

- `benchmark-sweep-json`
- `benchmark-sweep-md`
- `benchmark-raw-logs`

Do not update parity documentation from a dry-run report. Use a real report with `config.dry_run == false` and compare Swift `q4_direct` cells against Python cells for the same model and prompt length.

The `swift_vs_python` speedup column is not a steady-state throughput ratio: the Python cell discards its first decode step (steady-state decode tok/s) while the Swift cell's tok/s includes the compile-heavy first generated token. At low `--max-tokens` this dominates and makes Swift look artificially slow. Only draw throughput conclusions from a report run with a high `--max-tokens` and `--warmup-runs 1`, and treat the ratio as approximate even then.

## End-to-end smoke eval (`eval_smoke.py`)

`scripts/eval_smoke.py` is a one-command harness that loads a converted MLX
checkpoint, runs a short generation pass, computes perplexity on a fixed text
corpus, and writes a grounded JSON report. It is the fastest way to sanity-check
a freshly converted checkpoint locally.

**Important:** every report produced by this script is explicitly labelled as a
*local smoke* (n=1, one machine, one thermal state). The numbers are **not**
certified performance claims and must not be cited as representative throughput
without a full certified sweep (see above).

### Quickstart

```bash
# Uses $MOTIF_MODEL_DIR or ~/.models/motif-2.6b-mlx-q4 by default.
uv run python scripts/eval_smoke.py

# Explicit model path + explicit output location.
uv run python scripts/eval_smoke.py \
  --model ~/.models/motif-2.6b-mlx-q4 \
  --output docs/benchmarks/eval-smoke-$(date +%Y%m%dT%H%M%SZ).json

# Force non-zero exit when the model directory is absent (useful for local gates).
uv run python scripts/eval_smoke.py --require-model
```

### CI behaviour

When the model directory is absent and `--require-model` is **not** set the
script prints a notice and exits 0, so it can be wired into CI as a no-op on
machines without downloaded weights.

### Report schema (`eval-smoke-v1`)

| Field | Type | Notes |
| --- | --- | --- |
| `schema` | string | Always `"eval-smoke-v1"` |
| `disclaimer` | string | Honesty label — present on every report |
| `model` | string | Resolved absolute path of checkpoint |
| `timestamp` | string | ISO-8601 UTC |
| `host` | object | `hostname`, `chip`, `os`, `machine`, `memory_bytes`, `python` |
| `generations` | array | Per-prompt: `prompt`, `output`, `tokens`, `elapsed_s`, `tok_per_sec` |
| `tok_per_sec_summary` | object | `mean`, `min`, `max` across generation runs |
| `perplexity` | number | On the bundled ~2 k-token encyclopedic corpus |
| `peak_memory_gb` | number | MLX peak memory at end of generation pass |

### Unit tests

Pure-logic helpers (`aggregate_tok_per_sec`, `assemble_report`) are covered by
`tests/test_eval_smoke.py`. These tests require no model or MLX installation and
run as part of the normal `pytest` suite.
