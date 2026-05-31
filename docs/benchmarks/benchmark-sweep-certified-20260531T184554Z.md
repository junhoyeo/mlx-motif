# Motif benchmark sweep

Generated at: `2026-05-31T20:12:07Z`  
Commit: `4b5132a9aaa6822e842129e01f9746ff5a939540`  
Host: `Apple M1 Max` / `macOS-26.2-arm64-arm-64bit`

## Cells

| Cell | Status | Median tok/s | Prompt tokens | Runs |
| --- | --- | ---: | ---: | ---: |
| `motif-2.6b-q4/python/q4_bridge/p500` | PASS | 82.00 | 462 | 5 |
| `motif-2.6b-q4/python/q4_bridge/p3000` | PASS | 52.19 | 2727 | 5 |
| `motif-2.6b-q4/python/q4_bridge/p16000` | PASS | 16.03 | 14519 | 5 |
| `motif-2.6b-q4/python/q4_direct/p500` | PASS | 98.30 | 462 | 5 |
| `motif-2.6b-q4/python/q4_direct/p3000` | PASS | 57.45 | 2727 | 5 |
| `motif-2.6b-q4/python/q4_direct/p16000` | PASS | 16.33 | 14519 | 5 |
| `motif-2.6b-q4/swift/q4_bridge/p500` | PASS | 18.22 | 462 | 5 |
| `motif-2.6b-q4/swift/q4_bridge/p3000` | PASS | 18.04 | 2727 | 5 |
| `motif-2.6b-q4/swift/q4_bridge/p16000` | PASS | 12.37 | 14519 | 5 |
| `motif-2.6b-q4/swift/q4_direct/p500` | PASS | 18.00 | 462 | 5 |
| `motif-2.6b-q4/swift/q4_direct/p3000` | PASS | 17.89 | 2727 | 5 |
| `motif-2.6b-q4/swift/q4_direct/p16000` | PASS | 12.75 | 14519 | 5 |
| `motif-12.7b-q4/python/q4_bridge/p500` | PASS | 19.03 | 555 | 5 |
| `motif-12.7b-q4/python/q4_bridge/p3000` | PASS | 8.86 | 2821 | 5 |
| `motif-12.7b-q4/python/q4_bridge/p16000` | PASS | 1.67 | 14613 | 5 |
| `motif-12.7b-q4/python/q4_direct/p500` | PASS | 32.61 | 555 | 5 |
| `motif-12.7b-q4/python/q4_direct/p3000` | PASS | 27.30 | 2821 | 5 |
| `motif-12.7b-q4/python/q4_direct/p16000` | PASS | 12.18 | 14613 | 5 |
| `motif-12.7b-q4/swift/q4_bridge/p500` | PASS | 6.85 | 469 | 5 |
| `motif-12.7b-q4/swift/q4_bridge/p3000` | PASS | 7.31 | 2735 | 5 |
| `motif-12.7b-q4/swift/q4_bridge/p16000` | PASS | 2.21 | 14527 | 5 |
| `motif-12.7b-q4/swift/q4_direct/p500` | PASS | 6.68 | 469 | 5 |
| `motif-12.7b-q4/swift/q4_direct/p3000` | PASS | 7.11 | 2735 | 5 |
| `motif-12.7b-q4/swift/q4_direct/p16000` | PASS | 6.97 | 14527 | 5 |

## Comparisons

| Comparison | Candidate | Baseline | Prompt tokens (cand/base) | Speedup |
| --- | --- | --- | ---: | ---: |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p500` | `motif-2.6b-q4/python/q4_bridge/p500` | 462/462 | 1.199x |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p3000` | `motif-2.6b-q4/python/q4_bridge/p3000` | 2727/2727 | 1.101x |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p16000` | `motif-2.6b-q4/python/q4_bridge/p16000` | 14519/14519 | 1.018x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p500` | `motif-2.6b-q4/python/q4_bridge/p500` | 462/462 | 0.222x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p3000` | `motif-2.6b-q4/python/q4_bridge/p3000` | 2727/2727 | 0.346x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p16000` | `motif-2.6b-q4/python/q4_bridge/p16000` | 14519/14519 | 0.772x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p500` | `motif-2.6b-q4/swift/q4_bridge/p500` | 462/462 | 0.988x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p500` | `motif-2.6b-q4/python/q4_direct/p500` | 462/462 | 0.183x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p3000` | `motif-2.6b-q4/swift/q4_bridge/p3000` | 2727/2727 | 0.992x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p3000` | `motif-2.6b-q4/python/q4_direct/p3000` | 2727/2727 | 0.311x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p16000` | `motif-2.6b-q4/swift/q4_bridge/p16000` | 14519/14519 | 1.030x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p16000` | `motif-2.6b-q4/python/q4_direct/p16000` | 14519/14519 | 0.781x |
| direct_vs_bridge | `motif-12.7b-q4/python/q4_direct/p500` | `motif-12.7b-q4/python/q4_bridge/p500` | 555/555 | 1.714x |
| direct_vs_bridge | `motif-12.7b-q4/python/q4_direct/p3000` | `motif-12.7b-q4/python/q4_bridge/p3000` | 2821/2821 | 3.083x |
| direct_vs_bridge | `motif-12.7b-q4/python/q4_direct/p16000` | `motif-12.7b-q4/python/q4_bridge/p16000` | 14613/14613 | 7.295x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_bridge/p500` | `motif-12.7b-q4/python/q4_bridge/p500` | 469/555 ⚠️ | 0.360x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_bridge/p3000` | `motif-12.7b-q4/python/q4_bridge/p3000` | 2735/2821 ⚠️ | 0.826x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_bridge/p16000` | `motif-12.7b-q4/python/q4_bridge/p16000` | 14527/14613 ⚠️ | 1.321x |
| direct_vs_bridge | `motif-12.7b-q4/swift/q4_direct/p500` | `motif-12.7b-q4/swift/q4_bridge/p500` | 469/469 | 0.974x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_direct/p500` | `motif-12.7b-q4/python/q4_direct/p500` | 469/555 ⚠️ | 0.205x |
| direct_vs_bridge | `motif-12.7b-q4/swift/q4_direct/p3000` | `motif-12.7b-q4/swift/q4_bridge/p3000` | 2735/2735 | 0.972x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_direct/p3000` | `motif-12.7b-q4/python/q4_direct/p3000` | 2735/2821 ⚠️ | 0.260x |
| direct_vs_bridge | `motif-12.7b-q4/swift/q4_direct/p16000` | `motif-12.7b-q4/swift/q4_bridge/p16000` | 14527/14527 | 3.158x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_direct/p16000` | `motif-12.7b-q4/python/q4_direct/p16000` | 14527/14613 ⚠️ | 0.572x |

> ⚠️ Rows marked with a warning compare candidate and baseline cells whose actual prompt token counts differ (same target bucket, different rendered prompt). Decode tok/s depends on prompt length, so those speedups are not a clean head-to-head and must not be read as parity evidence.

## Notes

- Normal PR CI should validate this schema and dry-run plumbing only; real model sweeps require local or self-hosted Apple Silicon with cached checkpoints.
- Treat performance parity as unproven unless this report shows Swift candidate cells meeting the Python baseline for the exact model, host, branch, and thermal conditions.
- `swift_vs_python` ratios mix measurement regions: the Python cell reports steady-state decode tok/s (the first decode step is discarded as warm-up), while the Swift cell's tok/s covers every generated token including the compile-heavy first step. At low `max_tokens` the Swift first-step/compile cost dominates, so these ratios understate Swift steady-state throughput and must not be read as a steady-state speedup. Use a high `max_tokens` (and `warmup_runs >= 1`) before drawing any throughput conclusion.
