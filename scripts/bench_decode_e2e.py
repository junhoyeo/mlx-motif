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
import statistics
import sys
import time
from pathlib import Path

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


def render_prompt(text: str, tokenizer, use_chat_template: bool) -> str:
    if not use_chat_template:
        return text
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"user: {text}\nassistant: "


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


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the converted MLX checkpoint")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--backend-label", default="python")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--warmup-runs", type=int, default=0)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[500, 3000])
    ap.add_argument("--cache-mode", default=None)
    ap.add_argument("--quant-sdpa", default=None)
    ap.add_argument("--disable-kernels", choices=["0", "1"], default=None)
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--json-output", type=Path)
    args = ap.parse_args()

    if args.cache_mode is not None:
        os.environ["MLX_MOTIF_4SLOT_CACHE"] = args.cache_mode
    if args.quant_sdpa is not None:
        os.environ["MLX_MOTIF_QUANT_SDPA"] = args.quant_sdpa
    if args.disable_kernels is not None:
        os.environ["MLX_MOTIF_DISABLE_KERNELS"] = args.disable_kernels

    cache_mode = os.environ.get("MLX_MOTIF_4SLOT_CACHE", "0")
    quant_sdpa = os.environ.get("MLX_MOTIF_QUANT_SDPA", "n/a")
    disable_kernels = os.environ.get("MLX_MOTIF_DISABLE_KERNELS", "0")
    label = f"4slot={cache_mode} quant_sdpa={quant_sdpa} disable_kernels={disable_kernels}"

    print(f"[bench] {label} — loading {args.model}", file=sys.stderr, flush=True)
    from mlx_motif import load

    model, tokenizer = load(args.model)
    print("[bench] model loaded", file=sys.stderr, flush=True)

    results = {}
    for target_len in args.prompt_lens:
        key = f"p{target_len}"
        prompt_seed = (
            args.prompt_file.read_text() if args.prompt_file else make_prompt(target_len, tokenizer)
        )
        prompt = render_prompt(prompt_seed, tokenizer, args.chat_template)
        actual_len = len(tokenizer.encode(prompt))
        print(f"[bench] {key}: prompt tokens = {actual_len}", file=sys.stderr, flush=True)

        tps_list = []
        for run in range(args.warmup_runs):
            tps = run_bench(prompt, model, tokenizer, args.max_tokens)
            print(f"[bench]   warmup {run + 1}: {tps:.2f} tok/s", file=sys.stderr, flush=True)
        # Peak memory is process-global and monotone; reset after warmup so
        # each cell reports the high-water mark of its own measured runs only.
        mx.reset_peak_memory()
        for run in range(args.n_runs):
            tps = run_bench(prompt, model, tokenizer, args.max_tokens)
            tps_list.append(tps)
            print(f"[bench]   run {run + 1}: {tps:.2f} tok/s", file=sys.stderr, flush=True)
        peak_gb = mx.get_peak_memory() / 1e9

        stats = summarize(tps_list)
        results[key] = {
            "runs": tps_list,
            "median": stats["median"],
            "mean": stats["mean"],
            "stdev": stats["stdev"],
            "min": stats["min"],
            "max": stats["max"],
            "actual_prompt_len": actual_len,
            "peak_memory_gb": round(peak_gb, 3),
        }
        median = stats["median"] or 0
        print(f"[bench] {key} median: {median:.2f} tok/s", file=sys.stderr, flush=True)

    output = {
        "model_id": args.model_id,
        "backend": args.backend_label,
        "config": {
            "MLX_MOTIF_4SLOT_CACHE": cache_mode,
            "MLX_MOTIF_QUANT_SDPA": quant_sdpa,
            "MLX_MOTIF_DISABLE_KERNELS": disable_kernels,
            "max_tokens": args.max_tokens,
            "n_runs": args.n_runs,
            "warmup_runs": args.warmup_runs,
            "chat_template": args.chat_template,
        },
        "results": results,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("BENCH_RESULT:" + json.dumps(output))


if __name__ == "__main__":
    main()
