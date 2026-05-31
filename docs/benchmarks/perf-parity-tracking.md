# Swift-vs-Python perf-parity tracking

> **Status: GAP-TRACKING sentinel — NOT a parity claim.**
> Swift is materially slower than Python on generation throughput as of the
> recorded baseline below.  This document exists to make the gap visible and
> any regression auditable; it does not claim the gap has been closed.

## Recorded baseline (2026-05-27)

Source file: `docs/benchmarks/benchmark-sweep-target-20260527T103000Z.json`  
Host: Apple M1 Max · macOS 26.2 · `--max-tokens 8` · `--n-runs 1` · no warmup  
Sentinel test: [`tests/test_swift_python_perf_regression.py`](../../tests/test_swift_python_perf_regression.py)

### motif-2.6b-q4 — `swift_vs_python` speedup (prompt_tokens_match: ✅)

| cache_cell | prompt target | Swift tok/s | Python tok/s | speedup |
|------------|--------------|------------|-------------|---------|
| q4_bridge | 500 | 19.90 | 75.27 | **0.264x** |
| q4_bridge | 3000 | 18.55 | 48.79 | **0.380x** |
| q4_bridge | 16000 | 12.87 | 14.94 | **0.862x** |
| q4_direct | 500 | 19.52 | 97.51 | **0.200x** |
| q4_direct | 3000 | 18.58 | 52.55 | **0.354x** |
| q4_direct | 16000 | 11.13 | 16.20 | **0.687x** |

These are the **only clean baselines**: both backends tokenised the same prompt
to the same actual token count, so the comparison is head-to-head.

### motif-12.7b-q4 — `swift_vs_python` speedup (prompt_tokens_match: ⚠️ FALSE)

| cache_cell | prompt target | Swift tok/s | Python tok/s | speedup | caveat |
|------------|--------------|------------|-------------|---------|--------|
| q4_bridge | 500 | 8.44 | 17.99 | 0.469x | ⚠️ token mismatch (469 vs 555) |
| q4_bridge | 3000 | 7.98 | 8.55 | 0.933x | ⚠️ token mismatch (2735 vs 2821) |
| q4_bridge | 16000 | 2.87 | 2.23 | 1.290x | ⚠️ token mismatch (14527 vs 14613) |
| q4_direct | 500 | 7.80 | 30.89 | 0.253x | ⚠️ token mismatch (469 vs 555) |
| q4_direct | 3000 | 8.02 | 23.90 | 0.335x | ⚠️ token mismatch (2735 vs 2821) |
| q4_direct | 16000 | 4.27 | 12.96 | 0.329x | ⚠️ token mismatch (14527 vs 14613) |

The Swift and Python backends rendered **different actual token counts** from
the same target-length prompt (different tokeniser behaviour).  Decode
throughput scales with context length, so these speedups are not a clean
head-to-head comparison.  They are recorded for completeness and monitored for
schema continuity, but **must not be cited as parity evidence**.


## Certified sweep update (20260531T184554Z)

Source file: `docs/benchmarks/benchmark-sweep-certified-20260531T184554Z.json`<br>
Host: Apple M1 Max · macOS-26.2-arm64-arm-64bit · `--max-tokens 64` · `--n-runs 5` · `--warmup-runs 1`<br>
Certification validator: [`tests/test_benchmark_certification.py`](../../tests/test_benchmark_certification.py)

### motif-2.6b-q4 — certified clean `swift_vs_python` speedup

| cache_cell | prompt target | Swift tok/s | Python tok/s | speedup |
|------------|--------------|------------|-------------|---------|
| q4_bridge | 500 | 18.22 | 82.00 | **0.222x** |
| q4_bridge | 3000 | 18.04 | 52.19 | **0.346x** |
| q4_bridge | 16000 | 12.37 | 16.03 | **0.772x** |
| q4_direct | 500 | 18.00 | 98.30 | **0.183x** |
| q4_direct | 3000 | 17.89 | 57.45 | **0.311x** |
| q4_direct | 16000 | 12.75 | 16.33 | **0.781x** |

Certified evidence still shows Swift below Python for all clean 2.6B q4
comparisons.  The long-context q4-direct cell is closest (0.781x), but short and
mid contexts remain far from parity.  Keep the regression sentinel active and
continue optimizing the Swift generation loop / q4 direct path before claiming
performance parity.

### motif-12.7b-q4 — certified rows remain token-mismatched

All certified 12.7B `swift_vs_python` rows still have `prompt_tokens_match ==
false` (Swift 469/2735/14527 vs Python 555/2821/14613 prompt tokens). They are
useful trend evidence but still must not be cited as clean parity.

## Caveats

1. **Token-mismatch caveat (12.7b).** Every motif-12.7b-q4 row has
   `prompt_tokens_match == false`.  The speedup numbers for that model are
   confounded by different context lengths.  The sentinel test asserts these
   rows remain flagged rather than asserting a floor on their values.

2. **Low-`max_tokens` methodology caveat.** All sweep cells ran with
   `--max-tokens 8` and `--n-runs 1` (no warmup).  This makes each measurement
   prefill-dominated.  The numbers reflect prompt-processing (KV-cache fill)
   speed, not steady-state autoregressive decode throughput.  Real-world
   generation throughput may differ.

3. **Single-run, no warmup.** With `--n-runs 1` and `--warmup-runs 0`, thermal
   state and Metal shader JIT can affect the result.  Treat individual numbers
   as indicative, not statistically robust.

## Gap-to-beat

Baseline to beat for **clean q4-direct comparisons on motif-2.6b-q4**:

- p500: 0.200x → target ≥ 0.200x (floor at 0.160x after 20% tolerance)
- p3000: 0.354x → target ≥ 0.354x (floor at 0.283x)
- p16000: 0.687x → target ≥ 0.687x (floor at 0.549x)

Closing the gap at short contexts (p500, p3000) requires hardware-level
optimisation work in the Swift MLX kernel path — it is not a tooling issue.
The p16000 result (0.687x) is substantially better because prefill cost
dominates and both backends spend most time in the same Metal kernels.

## Sentinel test behaviour

`tests/test_swift_python_perf_regression.py` will:

- **Pass** as long as every clean (prompt_tokens_match=True) `swift_vs_python`
  row in the most recent checked-in sweep JSON has
  `speedup >= recorded_baseline * 0.80`.
- **Fail with a regression message** if a newly checked-in sweep JSON shows
  Swift throughput dropping more than 20% below the recorded numbers.
- **Fail with a schema message** if the 12.7b mismatch-flagged rows disappear
  or are silently un-flagged.

To update the baseline after a genuine performance improvement: regenerate the
sweep JSON, update `RECORDED_BASELINES` in the test file, and commit both
together with a rationale.
