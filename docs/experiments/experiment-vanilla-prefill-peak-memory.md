# Vanilla (2.6B) prefill peak-memory rise after the per-slab SDPA rework

**Status: wontfix, documented.** The per-slab fast-path rework
(commit `77c59ca`, `MotifAttention._forward_vanilla`) raised end-to-end 2.6B q4
peak memory at a ~3.5k-token prompt from **3.760 GB → 3.891 GB** (+131 MB,
deterministic). This page attributes the delta, shows why the obvious revert is
a net loss, and records why nothing cheaper recovers the memory without giving
back the win.

## Hypothesis

Candidate causes for the +131 MB (from the rework diff):

1. The projection-time slab-order V rearrange
   (`reshape→transpose(0,3,2,1,4)→reshape`).
2. The four per-slab `mx.fast.scaled_dot_product_attention` calls holding four
   intermediate outputs vs. the old two generic-SDPA outputs.
3. Mask materialization (ruled out early — prefill mask is the `"causal"`
   *string*, not an array, for both paths).
4. Full-capacity step-padded cache buffers — **not applicable to vanilla**: the
   2.6B config (`num_noise_heads` absent → `is_grouped == False`) uses mlx-lm's
   stock `KVCache`, not the 4-slot grouped cache the kernel rework touched.

## Bench

All numbers on the real `~/.models/motif-2.6b-mlx-q4` checkpoint, MLX 0.31.2,
`mx.get_peak_memory()` around a scripted prefill + greedy decode. Peak is
reached entirely in the prefill phase (`final_peak == prefill_peak` in every
run); decode never lifts the high-water mark. Output tokens are bit-identical
across every variant (`first_tok=[328, 4410, 6439]` at 3481).

### Bisection at the reported 3481-token operating point

Monkeypatching only `_forward_vanilla` (everything else identical):

| variant | e2e peak (GB) | delta vs pre-rework |
|---|---|---|
| pre-rework: paired-order V + 2 generic SDPA (v=2d) | 3.760 | — |
| slab-order V + 2 generic SDPA (v=2d) | 3.772 | +12 MB (V rearrange) |
| **shipped: slab-order V + 4 fast SDPA (v=d)** | **3.891** | **+131 MB** |

So ~12 MB is the slab-order V rearrange and ~119 MB is the generic→per-slab-fast
attention change. The rise is 100 % inside `_forward_vanilla`.

### The revert is O(S²); the shipped path is O(S)

Peak memory vs prompt length, shipped fast path vs the generic-SDPA revert
(`decode=8`, single deterministic run — peak varies < 0.5 MB run-to-run):

| prompt tokens | shipped (4 fast, v=d) | generic revert (2 SDPA, v=2d) | revert − shipped |
|---|---|---|---|
| 1024 | 3.223 | 3.206 | −17 MB |
| 3481 | 3.891 | 3.772 | −119 MB |
| 6144 | 4.498 | 4.867 | **+369 MB** |
| 8192 | 4.951 | 5.706 | **+755 MB** |

The generic path materializes an `[B, H, S, S]` score buffer (O(S²)); the
fast/flash template does not (O(S), the growth is just the KV cache). They cross
between 3.5k and 6k. At 3.5k the revert happens to be 119 MB cheaper because the
score buffer hasn't yet dominated; past the crossover it explodes, and 6k–16k is
squarely inside this model's `max_position_embeddings = 16384`.

Isolated single-op peak at the 3481 prefill shape confirms the fast path's own
footprint is *smaller*, not larger: four fast SDPA + concat = **100 MB** vs two
generic SDPA = **512 MB**. The e2e rise is therefore a scheduler high-water
artifact (cheap, parallelizable fast ops let MLX pipeline more of the DAG
concurrently), **not** a bigger attention allocation — so it does not compound
and does not erode OOM headroom the way the generic O(S²) path does.

## Why it lost (i.e. why we keep the +131 MB)

- **Reverting prefill to generic SDPA** saves 119 MB at 3.5k but regresses peak
  by +369 MB at 6k and +755 MB at 8k, and reintroduces O(S²) prefill memory for
  the whole supported context window. Net loss for any long-context use.
- **Decode throughput** — the actual documented win of the rework — favors the
  fast per-slab path everywhere (e.g. 43 vs 35 tok/s at 3.5k, 23 vs 16 at 6k in
  the same harness; the rework commit measured +5 %..+126 % vanilla decode at
  kv=256..16384). A prefill-only revert would keep decode speed but still carry
  the O(S²) prefill-memory regression above.
- **Stacked-query 2-call variant** (head-stack q1|q2 so one fast SDPA does both
  origin and noise per value slab, halving concurrent flash workspaces to two)
  is *worse* at every length: 3.936 / 4.557 / 5.005 GB at 3481 / 6144 / 8192 vs
  the shipped 3.891 / 4.498 / 4.951. The extra `q_f`/`k_f`/`va2`/`vb2` concats
  cost more than the workspace they save — confirming the four concurrent
  workspaces are not the dominant term and the shipped 4-call form is already
  near-optimal for the O(S) approach.
- **Slab-sequential eval** (force-free slab a before slab b) saved only ~7 MB in
  isolation (93 vs 100 MB) and would insert a per-layer sync into the lazy graph.

No cheap change recovers the 131 MB without either giving back the decode-speed
win or reintroducing O(S²) prefill memory. The +131 MB at 3.5k is the price of a
memory profile that stays O(S) out to the model's full context.

## When it might win

If a deployment is *pinned* to short prompts (≲4k) and memory-starved, gating
the prefill (`S > 1`) path to the generic 2d-wide-V construction — while keeping
the four fast per-slab calls at decode (`S == 1`, where the speed win lives and
where generic scales the same) — reclaims the 119 MB. It must not ship as the
default: it is a strict memory regression at ≥6k.

## Repro

`scripts/` was intentionally not touched; the probe used lives in the
investigation scratch and is trivially reconstructable: load the checkpoint via
`mlx_motif.loader.load`, run `generate_step` on a fixed-length prompt, and read
`mx.reset_peak_memory()` / `mx.get_peak_memory()` around prefill and decode.
Monkeypatch `MotifAttention._forward_vanilla` with the pre-rework body (recover
from `git show 77c59ca^:src/mlx_motif/model.py`) to reproduce the 3.760 baseline.
