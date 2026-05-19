# Ideas explored briefly and not landed

These ideas got far enough to measure but never reached the codebase. Brief writeups so we don't redo them.

## Provenance

These entries have no deleted production symbols in this PR. They are bench
notes for local or reverted prototypes; each section carries its own commit
reference when one exists.

## `mx.compile`-wrapped MLP chain

**Hypothesis.** Wrapping the MLP chain in `mx.compile` should let MLX's graph compiler do its full fusion pass on the chained ops.

**Result.** +12% microbench, **-4% end-to-end**. `mx.compile` broke the lazy graph's chained fusion with subsequent layers — the compiled subgraph became a fusion boundary. Net regression at the decode-loop level.

**Lesson.** `mx.compile` is not always a free win; it can prevent cross-boundary fusion that the lazy graph would have done otherwise.

**Commit ref.** Reverted in development; never landed.

## Fuse gate+up matmuls (single 4096→32768)

**Hypothesis.** Mirror the QKV fusion win (3 q4 linears → 1 fused linear at 4096→9216, -10%) on the MLP side: fuse `gate_proj` and `up_proj` into one matmul at 4096→32768.

**Result.** **-34% per-call**. The fused shape lands off MLX's quantized matmul sweet spot (the kernel templates favor output dims around 8k–16k, not 32k). The single big matmul ran slower than the two separate ones, undoing the dispatch saving.

**Lesson.** "Fewer dispatches is always better" is wrong when the larger shape misses a templated fast path. Always measure both the dispatch saving and the kernel-template fit.

**Commit ref.** Not landed.

## Concat Q+K for joint RoPE call

**Hypothesis.** Concatenate `q` and `k` along the head axis and call `rope()` once on the combined tensor, saving one MLX dispatch.

**Result.** **-9% end-to-end**. The concat itself costs more (memory traffic on a fresh allocation) than the dispatch-saving win.

**Lesson.** Concat is not free; on MLX it usually shows up as a memcpy. Saving dispatch overhead by doing one is only a win when the kernel cost is high enough to amortize the concat.

**Commit ref.** Not landed.

## Single-call 40-head dual_v (vs split into 32 + 8)

**Hypothesis.** Calling `sdpa_dual_v` once with all 40 Q heads is simpler and avoids one kernel dispatch.

**Result.** Split (32-head + 8-head calls) was **+8% over** single-call. Two smaller kernel launches pipeline better on the M1 Max GPU than one larger one — the second launch can begin issuing while the first is still consuming bandwidth.

**Lesson.** "One big call is always more efficient than two smaller calls" is wrong on GPUs with deep launch pipelines. The split also exposed nicer GQA shapes (one call with `gqa=4`, one with `gqa=1`) that mapped cleanly onto the kernel's native GQA broadcast.

**Commit ref.** `91c9a79` (kept the split, documenting the win in the message)

## `mx.fast.SDPA` with quantized KV via dequant bridge

**Hypothesis.** Same as the [quant-KV dequant-bridge experiment](experiment-quant-kv-dequant-bridge.md), but routing through MLX's stock `mx.fast.scaled_dot_product_attention` (with V-stacking) instead of our `sdpa_dual_v`.

**Result.** Even-or-slower at all tested KV. The dequant bridge cost dominates regardless of which attention kernel consumes the result. See the dequant-bridge writeup for the full reasoning.

**Commit refs.** `ac95834`, `54c5c62`.

## Composable q4 chain (3× `mx.quantized_matmul` + softmax + 2× attn@V)

**Hypothesis.** Build the attention path as a chain of MLX-native quantized ops — let MLX's compiler do the fusion work, rather than writing a hand-rolled kernel.

**Result.** Capped at **+1% vs fp16 `sdpa_dual_v`** at KV=8192. The chain is compute-bound at this scale; no amount of dispatch saving via `mx.quantized_matmul` chaining helps when the bottleneck is arithmetic and dequant overhead.

**Lesson.** Confirmed that the hand-written `sdpa_dual_v_q4` kernel was the right path. The composable approach gave us the floor; the kernel work pushed past it.

**Commit ref.** `f630dac` — bench notes.
