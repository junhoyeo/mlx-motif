#!/usr/bin/env python3
"""Run the native Swift Motif generator and optionally compare to Python mlx-motif.

This is intentionally a harness, not a source of truth: pass a converted Motif MLX
checkpoint directory produced by `mlx-motif convert`. The script records wall-clock
TTFT-ish process latency, total time, and generated text for repeatable follow-up
benchmarks on the same machine.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    elapsed = time.perf_counter() - start
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Converted MLX Motif checkpoint directory")
    parser.add_argument(
        "--prompt", default="Explain grouped differential attention in one sentence."
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mode", choices=["generate", "bench", "perplexity"], default="generate")
    parser.add_argument(
        "--cache-mode", default=None, help="Set MLX_MOTIF_4SLOT_CACHE=1|q4|q8 for Swift run"
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--python",
        action="store_true",
        help="Also run Python mlx-motif server/client comparison later; reserved for parity lab use",
    )
    args = parser.parse_args()

    executable = "MotifNativeGenerate" if args.mode == "generate" else "MotifNativeEvaluate"
    swift_cmd = ["swift", "run", "--package-path", "swift", executable, "--model", args.model]
    if args.mode == "generate":
        swift_cmd += [
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
        ]
    elif args.mode == "bench":
        swift_cmd += [
            "--mode",
            "bench",
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
        ]
    else:
        swift_cmd += [
            "--mode",
            "perplexity",
            "--text",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
        ]
    env = os.environ.copy()
    if args.cache_mode is not None:
        env["MLX_MOTIF_4SLOT_CACHE"] = args.cache_mode
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "mode": args.mode,
        "cache_mode": args.cache_mode,
        "swift_native": run(swift_cmd, args.repo, env=env),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["swift_native"]["returncode"] == 0 else result["swift_native"]["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
