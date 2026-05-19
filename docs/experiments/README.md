# Experiments — what we tried and what we kept

This folder is the graveyard + memorial for kernels and tricks we built, measured, and ultimately did not ship as the default. The goal: keep enough detail that the next person (probably future-us) doesn't redo the same experiment, and knows under what conditions a buried idea might still win.

Each entry is structured the same way:

- **Hypothesis** — what we expected and why
- **Snippet** — the actual code that ran (or a pointer to the commit)
- **Bench** — what we measured on M1 Max
- **Why it lost** — the load-bearing reason it didn't make it
- **When it might win** — the hardware/shape/regime where the idea could flip

## Index

### Custom Metal kernels we deleted

| Doc | Idea | Bench result | Removed in |
|---|---|---|---|
| [experiment-polynorm-mul.md](experiment-polynorm-mul.md) | Fuse `polynorm(gate) * up` into one kernel | -7% e2e | this PR |
| [experiment-sdpa-dual-v-2pass.md](experiment-sdpa-dual-v-2pass.md) | 2-pass dual-V SDPA (per-block partials → cross-block reduce) | slower than single-pass at every tested KV (64..16k) | this PR |
| [experiment-qmv-dual-q4.md](experiment-qmv-dual-q4.md) | Fused gate+up q4 GEMV (shared `x` register reuse) | -9% at S=1, -77% at S=16 | this PR |
| [experiment-gda-decode-flash.md](experiment-gda-decode-flash.md) | Single-pass serial-per-thread flash GDA decode | ~2.7× slower than vectorized SDPA path | this PR |

### Experiments still wired up (with documented trade-offs)

| Doc | Idea | Status |
|---|---|---|
| [experiment-mlp-lowbit.md](experiment-mlp-lowbit.md) | q3/gs=32 for MLP weights | Opt-in preset — **memory ↓ ~25%**, speed -31%, PPL +10%. Kept for memory-constrained scenarios. |
| [experiment-quant-kv-dequant-bridge.md](experiment-quant-kv-dequant-bridge.md) | 4-slot quantized cache via per-step dequant fetch | Superseded by `sdpa_dual_v_q4` (in-kernel dequant). Bridge path stays as `MLX_MOTIF_QUANT_SDPA=0` fallback. |

### Ideas explored briefly and not landed

| Doc | Ideas |
|---|---|
| [experiment-not-landed.md](experiment-not-landed.md) | `mx.compile` MLP, fuse gate+up matmuls (4096→32768), concat-qk for joint RoPE, single-call 40-head dual_v, composable q4 chain, mx.fast.SDPA with quantized KV via dequant bridge |

## Why this folder exists

Negative-result kernels are real engineering work — somebody wrote, tested, and benchmarked them. Deleting the code without recording why it lost throws away the most valuable byproduct: a calibrated prior for the next person evaluating the same idea on different hardware or a different shape. The snippets are preserved verbatim so someone re-trying these on M3/M4 doesn't start from scratch.

What this folder is **not**: an active TODO list. Nothing here is something we plan to revive. If an idea here flips on new hardware, the win should ship as a fresh, well-benchmarked PR, not as a resurrection.
