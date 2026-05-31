# Motif benchmark sweep

Generated at: `2026-05-27T11:09:04Z`  
Commit: `f5713f7dd49169b58ff11a6289512497a624bb12`  
Host: `Apple M1 Max` / `macOS-26.2-arm64-arm-64bit`

## Cells

| Cell | Status | Median tok/s | Prompt tokens | Runs |
| --- | --- | ---: | ---: | ---: |
| `motif-2.6b-q4/python/q4_bridge/p500` | PASS | 75.27 | 462 | 1 |
| `motif-2.6b-q4/python/q4_bridge/p3000` | PASS | 48.79 | 2727 | 1 |
| `motif-2.6b-q4/python/q4_bridge/p16000` | PASS | 14.94 | 14519 | 1 |
| `motif-2.6b-q4/python/q4_direct/p500` | PASS | 97.51 | 462 | 1 |
| `motif-2.6b-q4/python/q4_direct/p3000` | PASS | 52.55 | 2727 | 1 |
| `motif-2.6b-q4/python/q4_direct/p16000` | PASS | 16.20 | 14519 | 1 |
| `motif-2.6b-q4/swift/q4_bridge/p500` | PASS | 19.90 | 462 | 1 |
| `motif-2.6b-q4/swift/q4_bridge/p3000` | PASS | 18.55 | 2727 | 1 |
| `motif-2.6b-q4/swift/q4_bridge/p16000` | PASS | 12.87 | 14519 | 1 |
| `motif-2.6b-q4/swift/q4_direct/p500` | PASS | 19.52 | 462 | 1 |
| `motif-2.6b-q4/swift/q4_direct/p3000` | PASS | 18.58 | 2727 | 1 |
| `motif-2.6b-q4/swift/q4_direct/p16000` | PASS | 11.13 | 14519 | 1 |
| `motif-12.7b-q4/python/q4_bridge/p500` | PASS | 17.99 | 555 | 1 |
| `motif-12.7b-q4/python/q4_bridge/p3000` | PASS | 8.55 | 2821 | 1 |
| `motif-12.7b-q4/python/q4_bridge/p16000` | PASS | 2.23 | 14613 | 1 |
| `motif-12.7b-q4/python/q4_direct/p500` | PASS | 30.89 | 555 | 1 |
| `motif-12.7b-q4/python/q4_direct/p3000` | PASS | 23.90 | 2821 | 1 |
| `motif-12.7b-q4/python/q4_direct/p16000` | PASS | 12.96 | 14613 | 1 |
| `motif-12.7b-q4/swift/q4_bridge/p500` | PASS | 8.44 | 469 | 1 |
| `motif-12.7b-q4/swift/q4_bridge/p3000` | PASS | 7.98 | 2735 | 1 |
| `motif-12.7b-q4/swift/q4_bridge/p16000` | PASS | 2.87 | 14527 | 1 |
| `motif-12.7b-q4/swift/q4_direct/p500` | PASS | 7.80 | 469 | 1 |
| `motif-12.7b-q4/swift/q4_direct/p3000` | PASS | 8.02 | 2735 | 1 |
| `motif-12.7b-q4/swift/q4_direct/p16000` | PASS | 4.27 | 14527 | 1 |

## Comparisons

| Comparison | Candidate | Baseline | Prompt tokens (cand/base) | Speedup |
| --- | --- | --- | ---: | ---: |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p500` | `motif-2.6b-q4/python/q4_bridge/p500` | 462/462 | 1.295x |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p3000` | `motif-2.6b-q4/python/q4_bridge/p3000` | 2727/2727 | 1.077x |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p16000` | `motif-2.6b-q4/python/q4_bridge/p16000` | 14519/14519 | 1.084x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p500` | `motif-2.6b-q4/python/q4_bridge/p500` | 462/462 | 0.264x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p3000` | `motif-2.6b-q4/python/q4_bridge/p3000` | 2727/2727 | 0.380x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p16000` | `motif-2.6b-q4/python/q4_bridge/p16000` | 14519/14519 | 0.862x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p500` | `motif-2.6b-q4/swift/q4_bridge/p500` | 462/462 | 0.981x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p500` | `motif-2.6b-q4/python/q4_direct/p500` | 462/462 | 0.200x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p3000` | `motif-2.6b-q4/swift/q4_bridge/p3000` | 2727/2727 | 1.002x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p3000` | `motif-2.6b-q4/python/q4_direct/p3000` | 2727/2727 | 0.354x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p16000` | `motif-2.6b-q4/swift/q4_bridge/p16000` | 14519/14519 | 0.865x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p16000` | `motif-2.6b-q4/python/q4_direct/p16000` | 14519/14519 | 0.687x |
| direct_vs_bridge | `motif-12.7b-q4/python/q4_direct/p500` | `motif-12.7b-q4/python/q4_bridge/p500` | 555/555 | 1.717x |
| direct_vs_bridge | `motif-12.7b-q4/python/q4_direct/p3000` | `motif-12.7b-q4/python/q4_bridge/p3000` | 2821/2821 | 2.794x |
| direct_vs_bridge | `motif-12.7b-q4/python/q4_direct/p16000` | `motif-12.7b-q4/python/q4_bridge/p16000` | 14613/14613 | 5.822x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_bridge/p500` | `motif-12.7b-q4/python/q4_bridge/p500` | 469/555 ⚠️ | 0.469x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_bridge/p3000` | `motif-12.7b-q4/python/q4_bridge/p3000` | 2735/2821 ⚠️ | 0.933x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_bridge/p16000` | `motif-12.7b-q4/python/q4_bridge/p16000` | 14527/14613 ⚠️ | 1.290x |
| direct_vs_bridge | `motif-12.7b-q4/swift/q4_direct/p500` | `motif-12.7b-q4/swift/q4_bridge/p500` | 469/469 | 0.924x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_direct/p500` | `motif-12.7b-q4/python/q4_direct/p500` | 469/555 ⚠️ | 0.253x |
| direct_vs_bridge | `motif-12.7b-q4/swift/q4_direct/p3000` | `motif-12.7b-q4/swift/q4_bridge/p3000` | 2735/2735 | 1.005x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_direct/p3000` | `motif-12.7b-q4/python/q4_direct/p3000` | 2735/2821 ⚠️ | 0.335x |
| direct_vs_bridge | `motif-12.7b-q4/swift/q4_direct/p16000` | `motif-12.7b-q4/swift/q4_bridge/p16000` | 14527/14527 | 1.486x |
| swift_vs_python | `motif-12.7b-q4/swift/q4_direct/p16000` | `motif-12.7b-q4/python/q4_direct/p16000` | 14527/14613 ⚠️ | 0.329x |

> ⚠️ Rows marked with a warning compare candidate and baseline cells whose actual prompt token counts differ (same target bucket, different rendered prompt). Decode tok/s depends on prompt length, so those speedups are not a clean head-to-head and must not be read as parity evidence.

## Notes

- Normal PR CI should validate this schema and dry-run plumbing only; real model sweeps require local or self-hosted Apple Silicon with cached checkpoints.
- Treat performance parity as unproven unless this report shows Swift candidate cells meeting the Python baseline for the exact model, host, branch, and thermal conditions.
