"""End-to-end decode throughput bench for the q4 cache + kernel matrix.

Drives a single (cache mode, quant SDPA) cell. Run multiple times with
different env vars to fill out the matrix:

  MLX_MOTIF_4SLOT_CACHE=0  python scripts/bench_decode_e2e.py --model PATH
  MLX_MOTIF_4SLOT_CACHE=1  python scripts/bench_decode_e2e.py --model PATH
  MLX_MOTIF_4SLOT_CACHE=q4 MLX_MOTIF_QUANT_SDPA=0 python scripts/bench_decode_e2e.py --model PATH
  MLX_MOTIF_4SLOT_CACHE=q4 MLX_MOTIF_QUANT_SDPA=1 python scripts/bench_decode_e2e.py --model PATH

Reports decode tokens/sec at two prompt lengths (~500 and ~3000 tokens),
median of N runs. The first decode step is excluded — graph-compile
warm-up dominates it and isn't representative of steady-state throughput.

Output is a single line `BENCH_RESULT:<json>` (plus per-run progress to
stderr) so a wrapper script can collate cells from many invocations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mlx.core as mx

SEED_TEXT = (
    "The development of large language models has accelerated rapidly in recent years, "
    "with new architectures emerging that challenge conventional wisdom about attention mechanisms. "
    "Differential attention, grouped differential attention, and related techniques offer promising "
    "alternatives to standard scaled dot-product attention. "
)


def make_prompt(target_tokens: int, tokenizer) -> str:
    repeats = max(1, target_tokens // 40)
    text = SEED_TEXT * repeats
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        ids = ids[:target_tokens]
        text = tokenizer.decode(ids)
    return text


def run_bench(prompt: str, model, tokenizer, max_tokens: int) -> float:
    """Decode throughput, tokens/sec. Excludes the first step (warm-up)."""
    from mlx_lm.generate import generate_step

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    gen = generate_step(prompt_tokens, model, max_tokens=max_tokens + 1)

    # First yield = prefill + first decode step. Don't time it.
    _, first_lp = next(gen)
    mx.eval(first_lp)

    t0 = time.perf_counter()
    count = 0
    for _ in range(max_tokens - 1):
        _, lp = next(gen)
        mx.eval(lp)
        count += 1
    t1 = time.perf_counter()
    return count / (t1 - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the converted MLX checkpoint")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[500, 3000])
    args = ap.parse_args()

    cache_mode = os.environ.get("MLX_MOTIF_4SLOT_CACHE", "0")
    quant_sdpa = os.environ.get("MLX_MOTIF_QUANT_SDPA", "n/a")
    label = f"4slot={cache_mode} quant_sdpa={quant_sdpa}"

    print(f"[bench] {label} — loading {args.model}", file=sys.stderr, flush=True)
    from mlx_motif import load

    model, tokenizer = load(args.model)
    print("[bench] model loaded", file=sys.stderr, flush=True)

    results = {}
    for target_len in args.prompt_lens:
        key = f"p{target_len}"
        prompt = make_prompt(target_len, tokenizer)
        actual_len = len(tokenizer.encode(prompt))
        print(f"[bench] {key}: prompt tokens = {actual_len}", file=sys.stderr, flush=True)

        tps_list = []
        for run in range(args.n_runs):
            tps = run_bench(prompt, model, tokenizer, args.max_tokens)
            tps_list.append(tps)
            print(f"[bench]   run {run + 1}: {tps:.2f} tok/s", file=sys.stderr, flush=True)

        median = sorted(tps_list)[args.n_runs // 2]
        results[key] = {"runs": tps_list, "median": median, "actual_prompt_len": actual_len}
        print(f"[bench] {key} median: {median:.2f} tok/s", file=sys.stderr, flush=True)

    output = {
        "config": {
            "MLX_MOTIF_4SLOT_CACHE": cache_mode,
            "MLX_MOTIF_QUANT_SDPA": quant_sdpa,
        },
        "results": results,
    }
    print("BENCH_RESULT:" + json.dumps(output))


if __name__ == "__main__":
    main()
