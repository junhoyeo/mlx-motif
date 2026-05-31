# Benchmark certification protocol

This document defines what makes a benchmark sweep **certified** — meaning its
results are acceptable as evidence for performance-parity or optimization claims
— and specifies the exact command to produce one.

---

## Status of the current checked-in target sweep

`docs/benchmarks/benchmark-sweep-target-20260527T103000Z.json` is a
**smoke / first-evidence run**, not a certified sweep.

It was generated with `n_runs=1`, `warmup_runs=0`, `max_tokens=8` to
establish the target-matrix baseline quickly on local Apple Silicon.  Those
settings are deliberately weak: a single run provides no stdev estimate,
zero warmup runs mean the first-token/compile cost is included in the timed
measurement, and eight generated tokens are insufficient to dilute that
compile-heavy first step.

Do **not** cite this sweep as evidence of steady-state throughput or
Swift/Python parity.

---

## Certification minimums

| Parameter | Minimum | Rationale |
|---|---|---|
| `n_runs` | **≥ 5** | Five repetitions give a meaningful stdev and make single-outlier distortion visible. The CI `workflow_dispatch` default is 3; certified sweeps raise this bar to 5. |
| `warmup_runs` | **≥ 1** | The first decode step on Apple Silicon triggers Metal shader compilation. One warm-up run ensures the timed runs measure steady-state decode, not compile latency. |
| `max_tokens` | **≥ 64** | At low token counts the compile-heavy first step dominates reported tok/s. Sixty-four tokens dilute it sufficiently for the median to reflect steady-state decode. |
| `prompt_lens` | **{500, 3000, 16000}** | Short (sub-512), mid (3 K), and long-context (16 K) coverage — the same three buckets used in the README "Full intended sweep" example. |
| `backends` | **python + swift** | Both backends are required so the `swift_vs_python` comparison rows are populated. |
| `models` | **motif-2.6b-q4 + motif-12.7b-q4** | Both target checkpoints (2.6 B and 12.7 B q4) must be present. |

A sweep that meets **all** minimums above is certified.  The validator
`tests/test_benchmark_certification.py::validate_certification` encodes these
rules as a pure Python function and can be run offline against any sweep JSON.

---

## Command to produce a certified sweep

Running this requires **Apple Silicon hardware** with the model checkpoints
already present at the paths referenced in the manifest.  This step is
intentionally **not** automated in CI — the weights are not vendored, and
normal PR CI runs a dry-run only.

```bash
MOTIFKIT_ENABLE_MLX=1 uv run python scripts/bench_sweep.py \
  --manifest docs/benchmarks/models.local.example.json \
  --backend python \
  --backend swift \
  --cache-cell q4_bridge:q4,quant_sdpa=0,disable_kernels=1 \
  --cache-cell q4_direct:q4,quant_sdpa=1 \
  --prompt-lens 500 3000 16000 \
  --max-tokens 64 \
  --n-runs 5 \
  --warmup-runs 1 \
  --raw-dir artifacts/benchmarks/raw-certified-$(date -u +%Y%m%dT%H%M%SZ) \
  --output docs/benchmarks/benchmark-sweep-certified-$(date -u +%Y%m%dT%H%M%SZ).json \
  --markdown docs/benchmarks/benchmark-sweep-certified-$(date -u +%Y%m%dT%H%M%SZ).md
```

Substitute `$(date -u +%Y%m%dT%H%M%SZ)` with the actual timestamp when
running non-interactively (e.g. `20260601T120000Z`).

The `--manifest` path must point to a valid JSON file listing model IDs and
their local checkpoint directories.  See
`docs/benchmarks/models.local.example.json` for the expected schema.

---

## Where to commit outputs

After a certified sweep completes:

1. Copy the output JSON and Markdown into `docs/benchmarks/` using the
   timestamped filename (e.g.
   `benchmark-sweep-certified-20260601T120000Z.json`).
2. Copy the `--raw-dir` directory into `docs/benchmarks/` alongside the JSON
   (e.g. `raw-certified-20260601T120000Z/`).
3. Update `docs/benchmarks/README.md` to reference the new certified sweep as
   the current authoritative evidence.
4. Open a PR with both the JSON/Markdown artifacts and the README update, and
   link to the CI dry-run that validated the schema.

Do **not** update parity documentation from a `dry_run: true` report or from
any report where `config.n_runs < 5` or `config.max_tokens < 64`.

---

## Thermal-variance notes template

Copy the following block into a comment on the PR that introduces a certified
sweep:

```
## Thermal conditions

- Machine:         <model, chip, unified memory>
- Ambient temp:    <°C or "not measured">
- Idle temp before run:  <°C or output of `sudo powermetrics -n 1 --samplers smc | grep -i temp` or "not measured">
- Run spacing:     <"back-to-back" | "cooled between runs (≥N min idle)">
- Active workloads during run:  <"none" | brief description>
- Battery / plugged:  <"plugged in" | "battery, X% charge">
- Notes:           <any anomalies, thermal throttling events, etc.>
```

Back-to-back runs on a warm machine will show higher stdev than cooled runs.
Record which was done so readers can judge reproducibility.

---

## Interpreting `swift_vs_python` speedup ratios

The `swift_vs_python` speedup column in the sweep report compares medians
across measurement regions that differ between backends:

- The **Python** cell reports steady-state decode tok/s: the first decode step
  is discarded as a warm-up inside `bench_decode_e2e.py`.
- The **Swift** cell's tok/s covers **every** generated token, including the
  compile-heavy first step.

At low `max_tokens` this asymmetry dominates and makes Swift look artificially
slow.  A certified sweep with `max_tokens ≥ 64` and `warmup_runs ≥ 1` reduces
this artefact but does not eliminate it entirely.  Treat `swift_vs_python`
ratios as approximate; only claim parity when the ratio is ≥ 1.0 across all
prompt lengths for a given model and the stdev columns are both small relative
to the median.
