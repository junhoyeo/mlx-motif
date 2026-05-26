#!/usr/bin/env python3
"""Run same-machine hard-parity evidence for the Swift Motif follow-up PR.

The script intentionally records failures as JSON instead of hiding them. A passing
report must show successful Python and Swift perplexity/logit runs, long-context
benchmarks, q4 direct-vs-bridge Swift timings, and Python-vs-Swift performance cells.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SEED = (
    "Grouped differential attention compares an origin stream against a noise stream, "
    "then normalizes the difference before the output projection. PolyNorm combines "
    "linear, quadratic, and cubic normalized activations. Quantized four-slot caches "
    "keep Motif decode bandwidth low when the packed q4 attention kernel consumes "
    "cache entries directly. "
)


def run(
    cmd: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: int | None = None
) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    parsed: Any | None = None
    stdout = proc.stdout.strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            for line in stdout.splitlines():
                if line.startswith("BENCH_RESULT:"):
                    parsed = json.loads(line.removeprefix("BENCH_RESULT:"))
                    break
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "json": parsed,
    }


def long_prompt(target_words: int) -> str:
    words = []
    seed_words = SEED.split()
    while len(words) < target_words:
        words.extend(seed_words)
    return " ".join(words[:target_words])


def write_markdown(report: dict[str, Any], path: Path) -> None:
    def status(cell: dict[str, Any]) -> str:
        return "PASS" if cell.get("returncode") == 0 else "FAIL"

    rows = []
    for name, cell in report["runs"].items():
        if isinstance(cell, dict) and "returncode" in cell:
            rows.append(f"| `{name}` | {status(cell)} | `{cell['elapsed_seconds']:.2f}s` |")
    metric_summary = ""
    try:
        py_ppl = report["runs"]["python_perplexity"]["json"]["result"]
        sw_ppl = report["runs"]["swift_perplexity"]["json"]
        py_logits = report["runs"]["python_logits"]["json"]["result"]
        sw_logits = report["runs"]["swift_logits"]["json"]
        spec = report["runs"]["swift_speculative"]["json"]["metrics"]
        direct = report["runs"]["swift_q4_direct_bench"]["json"]["swift_native"]["json"]
        bridge = report["runs"]["swift_q4_bridge_bench"]["json"]["swift_native"]["json"]
        py_long = report["runs"]["python_decode_long_context"]["json"]["results"]
        sw_long = report["runs"]["swift_decode_long_context"]["json"]["swift_native"]["json"]
        long_key = max(py_long, key=lambda key: py_long[key].get("actual_prompt_len", 0))
        metric_summary = f"""
## Key metrics

| Metric | Python | Swift |
| --- | ---: | ---: |
| Perplexity | `{py_ppl["ppl"]:.6f}` | `{sw_ppl["perplexity"]:.6f}` |
| NLL/token | `{py_ppl["nll_per_token"]:.6f}` | `{sw_ppl["nllPerToken"]:.6f}` |
| Perplexity tokens/s | `{py_ppl["tps"]:.2f}` | `{sw_ppl["tokensPerSecond"]:.2f}` |
| Logit checksum | `{py_logits["checksum"]:.3f}` | `{sw_logits["checksum"]:.3f}` |
| Top-1 token | `{py_logits["top_k"][0]["token"]}` | `{sw_logits["topK"][0]["token"]}` |

## q4 direct-vs-bridge Swift decode

| Path | Generate time | Tokens/s | Output |
| --- | ---: | ---: | --- |
| Direct packed Metal | `{direct["generationInfo"]["generateTime"]:.6f}s` | `{direct["generationInfo"]["tokensPerSecond"]:.2f}` | `{direct["output"]}` |
| Reference/dequant bridge | `{bridge["generationInfo"]["generateTime"]:.6f}s` | `{bridge["generationInfo"]["tokensPerSecond"]:.2f}` | `{bridge["output"]}` |

## Speculative decoding

Accepted `{spec["acceptedDraftTokens"]}` / `{spec["proposedDraftTokens"]}` draft tokens across `{spec["draftModelRuns"]}` draft runs; generated `{spec["targetTokens"]}` target tokens in `{spec["elapsedSeconds"]:.3f}s`.

## Long-context decode

Python q4 decode median: `{py_long[long_key]["median"]:.2f}` tok/s at `{py_long[long_key]["actual_prompt_len"]}` prompt tokens.<br>
Swift q4 long-context decode: `{sw_long["generationInfo"]["tokensPerSecond"]:.2f}` tok/s for `{sw_long["promptTokens"]}` prompt tokens and `{sw_long["generatedTokens"]}` generated tokens.
"""
    except Exception as error:  # pragma: no cover - report best-effort summary only
        metric_summary = f"\n## Key metrics\n\nMetric extraction failed; inspect the adjacent JSON. Error: `{error}`\n"

    body = f"""# Swift/Python hard parity evidence

Commit: `{report["git_commit"]}`<br>
Host: `{report["host"]["platform"]}` / `{report["host"]["machine"]}`<br>
Model: `{report["model"]}`<br>
Generated at: `{report["generated_at"]}`

## Gate status

| Gate | Status | Wall time |
| --- | --- | ---: |
{chr(10).join(rows)}

## Notes

- Direct Swift Metal is expected to be default-on. The `swift_q4_bridge_bench` cell sets `MLX_MOTIF_DISABLE_KERNELS=1` to force the bridge/reference path.
- Same-machine parity is represented by Python and Swift cells in this single report; raw command stdout/stderr is preserved in the adjacent JSON.
- If any cell is `FAIL`, the PR must not claim that evidence gate as passed.
{metric_summary}
"""
    path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Converted MLX Motif checkpoint directory")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument(
        "--prompt", default="Explain grouped differential attention in one sentence."
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--long-context-words", type=int, default=2400)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = (repo / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    long_text = long_prompt(args.long_context_words)
    env = os.environ.copy()
    env["MOTIFKIT_ENABLE_MLX"] = "1"

    report: dict[str, Any] = {
        "generated_at": stamp,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "model": args.model,
        "runs": {},
    }

    runs = report["runs"]
    eval_file = out_dir / f"hard-parity-eval-{stamp}.txt"
    eval_file.write_text(long_prompt(360), encoding="utf-8")
    prompt_file = out_dir / f"hard-parity-prompt-{stamp}.txt"
    prompt_file.write_text(args.prompt, encoding="utf-8")

    runs["swift_build"] = run(
        ["bash", "scripts/verify_swift_mlx.sh"], cwd=repo, env=env, timeout=args.timeout
    )
    runs["python_perplexity"] = run(
        [
            "uv",
            "run",
            "python",
            "scripts/perplexity.py",
            "--model",
            args.model,
            "--json",
            "--quiet",
            "--text-file",
            str(eval_file),
            "--max-tokens",
            "512",
        ],
        cwd=repo,
        env=env,
        timeout=args.timeout,
    )
    runs["swift_perplexity"] = run(
        [
            "swift",
            "run",
            "--package-path",
            "swift",
            "MotifNativeEvaluate",
            "--model",
            args.model,
            "--mode",
            "perplexity",
            "--text-file",
            str(eval_file),
            "--max-tokens",
            "512",
        ],
        cwd=repo,
        env=env,
        timeout=args.timeout,
    )
    # Python logits use a temporary file because the perplexity utility accepts
    # text either inline defaults or via --text-file.
    runs["python_logits"] = run(
        [
            "uv",
            "run",
            "python",
            "scripts/perplexity.py",
            "--model",
            args.model,
            "--mode",
            "logits",
            "--json",
            "--quiet",
            "--text-file",
            str(prompt_file),
            "--max-tokens",
            "512",
            "--top-k",
            "10",
        ],
        cwd=repo,
        env=env,
        timeout=args.timeout,
    )
    runs["swift_logits"] = run(
        [
            "swift",
            "run",
            "--package-path",
            "swift",
            "MotifNativeEvaluate",
            "--model",
            args.model,
            "--mode",
            "logits",
            "--text",
            args.prompt,
            "--max-tokens",
            "512",
            "--top-k",
            "10",
        ],
        cwd=repo,
        env=env,
        timeout=args.timeout,
    )
    runs["swift_speculative"] = run(
        [
            "swift",
            "run",
            "--package-path",
            "swift",
            "MotifNativeGenerate",
            "--model",
            args.model,
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--speculative",
            "--speculative-draft-model",
            args.model,
            "--speculative-draft-tokens",
            "4",
            "--json",
        ],
        cwd=repo,
        env=env,
        timeout=args.timeout,
    )
    q4_env = {**env, "MLX_MOTIF_4SLOT_CACHE": "q4", "MLX_MOTIF_QUANT_SDPA": "1"}
    bridge_env = {**q4_env, "MLX_MOTIF_DISABLE_KERNELS": "1"}
    runs["swift_q4_direct_bench"] = run(
        [
            "uv",
            "run",
            "python",
            "scripts/bench_swift_native.py",
            "--model",
            args.model,
            "--mode",
            "bench",
            "--cache-mode",
            "q4",
            "--max-tokens",
            str(args.max_tokens),
            "--prompt",
            args.prompt,
        ],
        cwd=repo,
        env=q4_env,
        timeout=args.timeout,
    )
    runs["swift_q4_bridge_bench"] = run(
        [
            "uv",
            "run",
            "python",
            "scripts/bench_swift_native.py",
            "--model",
            args.model,
            "--mode",
            "bench",
            "--cache-mode",
            "q4",
            "--max-tokens",
            str(args.max_tokens),
            "--prompt",
            args.prompt,
        ],
        cwd=repo,
        env=bridge_env,
        timeout=args.timeout,
    )
    runs["python_decode_long_context"] = run(
        [
            "uv",
            "run",
            "python",
            "scripts/bench_decode_e2e.py",
            "--model",
            args.model,
            "--max-tokens",
            str(args.max_tokens),
            "--n-runs",
            "1",
            "--prompt-lens",
            "500",
            str(args.long_context_words),
        ],
        cwd=repo,
        env=q4_env,
        timeout=args.timeout,
    )
    runs["swift_decode_long_context"] = run(
        [
            "uv",
            "run",
            "python",
            "scripts/bench_swift_native.py",
            "--model",
            args.model,
            "--mode",
            "bench",
            "--cache-mode",
            "q4",
            "--max-tokens",
            str(args.max_tokens),
            "--prompt",
            long_text,
        ],
        cwd=repo,
        env=q4_env,
        timeout=args.timeout,
    )

    json_path = out_dir / f"swift-python-hard-parity-{stamp}.json"
    md_path = out_dir / f"swift-python-hard-parity-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))

    failed = [
        name
        for name, cell in runs.items()
        if isinstance(cell, dict) and cell.get("returncode") != 0
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
