# Motif benchmark sweep

Generated at: `2026-05-27T11:23:43Z`  
Commit: `bbc2c613def97e9f8fdebebff0a521729c33743f`  
Host: `Apple M1 Max` / `macOS-26.2-arm64-arm-64bit`

## Cells

| Cell | Status | Median tok/s | Prompt tokens | Runs |
| --- | --- | ---: | ---: | ---: |
| `motif-12.7b-q4/swift/q4_direct/p500` | PASS | 6.85 | 469 | 1 |
| `motif-12.7b-q4/swift/q4_direct/p3000` | PASS | 6.27 | 2735 | 1 |
| `motif-12.7b-q4/swift/q4_direct_fused/p500` | PASS | 7.19 | 469 | 1 |
| `motif-12.7b-q4/swift/q4_direct_fused/p3000` | PASS | 7.54 | 2735 | 1 |

## Comparisons

| Comparison | Candidate | Baseline | Speedup |
| --- | --- | --- | ---: |
| fused_vs_default | `motif-12.7b-q4/swift/q4_direct_fused/p500` | `motif-12.7b-q4/swift/q4_direct/p500` | 1.050x |
| fused_vs_default | `motif-12.7b-q4/swift/q4_direct_fused/p3000` | `motif-12.7b-q4/swift/q4_direct/p3000` | 1.203x |

## Notes

- Normal PR CI should validate this schema and dry-run plumbing only; real model sweeps require local or self-hosted Apple Silicon with cached checkpoints.
- Treat performance parity as unproven unless this report shows Swift candidate cells meeting the Python baseline for the exact model, host, branch, and thermal conditions.
