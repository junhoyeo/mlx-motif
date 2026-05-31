# Swift/Python hard parity evidence

Commit: `e2ab07e18d6ec678748d46cec052bb3defd07ec7`  
Host: `macOS-26.2-arm64-arm-64bit` / `arm64`  
Model: `.models/motif-2.6b-mlx-q4`  
Generated at: `20260526T091532Z`

## Gate status

| Gate | Status | Wall time |
| --- | --- | ---: |
| `swift_build` | PASS | `20.16s` |
| `python_perplexity` | PASS | `3.76s` |
| `swift_perplexity` | PASS | `5.28s` |
| `python_logits` | PASS | `3.18s` |
| `swift_logits` | PASS | `4.29s` |
| `swift_speculative` | PASS | `5.84s` |
| `swift_q4_direct_bench` | PASS | `5.17s` |
| `swift_q4_bridge_bench` | PASS | `5.24s` |
| `python_decode_long_context` | PASS | `5.63s` |
| `swift_decode_long_context` | PASS | `8.07s` |

## Notes

- Direct Swift Metal is expected to be default-on. The `swift_q4_bridge_bench` cell sets `MLX_MOTIF_DISABLE_KERNELS=1` to force the bridge/reference path.
- Same-machine parity is represented by Python and Swift cells in this single report; raw command stdout/stderr is preserved in the adjacent JSON.
- If any cell is `FAIL`, the PR must not claim that evidence gate as passed.

## Key metrics

| Metric | Python | Swift |
| --- | ---: | ---: |
| Perplexity | `2.130710` | `2.133027` |
| NLL/token | `0.756455` | `0.757542` |
| Perplexity tokens/s | `703.52` | `653.69` |
| Logit checksum | `1503463.898` | `1460731.811` |
| Top-1 token | `5310` | `5310` |

## q4 direct-vs-bridge Swift decode

| Path | Generate time | Tokens/s | Output |
| --- | ---: | ---: | --- |
| Direct packed Metal | `0.408105s` | `19.60` | `Grouped differential attention refers to a neural mechanism` |
| Reference/dequant bridge | `0.406982s` | `19.66` | `Grouped differential attention refers to a neural mechanism` |

## Speculative decoding

Accepted `7` / `7` draft tokens across `2` draft runs; generated `8` target tokens in `0.989s`.

## Long-context decode

Python q4 decode median: `69.19` tok/s at `1600` prompt tokens.  
Swift q4 long-context decode: `18.65` tok/s for `2104` prompt tokens and `8` generated tokens.
