# Motif benchmark sweep

Generated at: `2026-05-27T10:22:36Z`  
Commit: `ab01393b98bd1cd6e6f9781a0377218de4923e5f`  
Host: `Apple M1 Max` / `macOS-26.2-arm64-arm-64bit`

## Cells

| Cell | Status | Median tok/s | Prompt tokens | Runs |
| --- | --- | ---: | ---: | ---: |
| `motif-2.6b-q4/python/q4_bridge/p500` | PASS | 75.35 | 462 | 1 |
| `motif-2.6b-q4/python/q4_direct/p500` | PASS | 98.94 | 462 | 1 |
| `motif-2.6b-q4/swift/q4_bridge/p500` | PASS | 22.20 | 462 | 1 |
| `motif-2.6b-q4/swift/q4_direct/p500` | PASS | 21.64 | 462 | 1 |

## Comparisons

| Comparison | Candidate | Baseline | Speedup |
| --- | --- | --- | ---: |
| direct_vs_bridge | `motif-2.6b-q4/python/q4_direct/p500` | `motif-2.6b-q4/python/q4_bridge/p500` | 1.313x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_bridge/p500` | `motif-2.6b-q4/python/q4_bridge/p500` | 0.295x |
| direct_vs_bridge | `motif-2.6b-q4/swift/q4_direct/p500` | `motif-2.6b-q4/swift/q4_bridge/p500` | 0.975x |
| swift_vs_python | `motif-2.6b-q4/swift/q4_direct/p500` | `motif-2.6b-q4/python/q4_direct/p500` | 0.219x |

## Notes

- This smoke report used `max_tokens=4`, `n_runs=1`, `warmup_runs=0` — it exercises the sweep plumbing on a real checkpoint and is NOT a valid throughput measurement.
- Normal PR CI should validate this schema and dry-run plumbing only; real model sweeps require local or self-hosted Apple Silicon with cached checkpoints.
- Treat performance parity as unproven unless this report shows Swift candidate cells meeting the Python baseline for the exact model, host, branch, and thermal conditions.
- The `swift_vs_python` ratios above mix measurement regions: the Python cell reports steady-state decode tok/s (first decode step discarded as warm-up) while the Swift cell's tok/s covers every generated token including the compile-heavy first step. At `max_tokens=4` the Swift first-step/compile cost dominates, so these ratios understate Swift steady-state throughput and must not be read as a steady-state speedup.
