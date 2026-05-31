#!/usr/bin/env python3
"""Run reproducible Swift/Python Motif benchmark sweeps.

This orchestrates model x backend x cache/kernel x prompt-length cells and writes
stable JSON/Markdown artifacts. It preserves failed cells by default so partial
self-hosted or local sweeps remain useful.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEED_TEXT = (
    "The development of large language models has accelerated rapidly in recent years, "
    "with new architectures emerging that challenge conventional wisdom about attention mechanisms. "
    "Differential attention, grouped differential attention, quantized key-value caches, "
    "and custom Metal kernels offer promising alternatives to standard scaled dot-product attention. "
)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    id: str
    path: str
    quantization: str | None = None
    params_b: float | None = None


@dataclass(frozen=True)
class CacheCell:
    name: str
    cache_mode: str
    quant_sdpa: str | None = None
    disable_kernels: bool = False

    @property
    def env(self) -> dict[str, str]:
        env = {"MLX_MOTIF_4SLOT_CACHE": self.cache_mode}
        if self.quant_sdpa is not None:
            env["MLX_MOTIF_QUANT_SDPA"] = self.quant_sdpa
        env["MLX_MOTIF_DISABLE_KERNELS"] = "1" if self.disable_kernels else "0"
        return env


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str
    json: Any | None


Runner = Callable[..., CommandResult]


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float | None = None,
) -> CommandResult:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    return CommandResult(
        command=cmd,
        returncode=proc.returncode,
        elapsed_seconds=elapsed,
        stdout=proc.stdout,
        stderr=proc.stderr,
        json=parse_json_output(proc.stdout),
    )


def parse_json_output(stdout: str) -> Any | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        if line.startswith("BENCH_RESULT:"):
            try:
                return json.loads(line.removeprefix("BENCH_RESULT:"))
            except json.JSONDecodeError:
                return None
    return None


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        path = Path(value)
        return ModelSpec(id=path.name or value, path=value)
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model must be NAME=PATH or PATH")
    return ModelSpec(id=name, path=path)


def load_manifest(path: Path) -> list[ModelSpec]:
    payload = json.loads(path.read_text())
    entries = payload.get("models", payload if isinstance(payload, list) else [])
    models = []
    for entry in entries:
        models.append(
            ModelSpec(
                id=entry["id"],
                path=entry["path"],
                quantization=entry.get("quantization"),
                params_b=entry.get("params_b"),
            )
        )
    return models


def parse_cache_cell(value: str) -> CacheCell:
    if ":" not in value:
        raise argparse.ArgumentTypeError("cache cell must be NAME:CACHE[,key=value]")
    name, spec = value.split(":", 1)
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not name or not parts:
        raise argparse.ArgumentTypeError("cache cell must be NAME:CACHE[,key=value]")
    cache_mode = parts[0]
    settings: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"invalid cache-cell setting: {part}")
        key, raw = part.split("=", 1)
        settings[key.strip().replace("-", "_")] = raw.strip()
    disable = settings.get("disable_kernels", "0").lower() in {"1", "true", "yes", "on"}
    if name.endswith("bridge") and "disable_kernels" not in settings:
        disable = True
    return CacheCell(
        name=name,
        cache_mode=cache_mode,
        quant_sdpa=settings.get("quant_sdpa"),
        disable_kernels=disable,
    )


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median_tokens_per_second": None,
            "mean_tokens_per_second": None,
            "stdev_tokens_per_second": None,
            "min_tokens_per_second": None,
            "max_tokens_per_second": None,
        }
    return {
        "median_tokens_per_second": statistics.median(values),
        "mean_tokens_per_second": statistics.mean(values),
        "stdev_tokens_per_second": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_tokens_per_second": min(values),
        "max_tokens_per_second": max(values),
    }


def make_prompt(target_tokens: int) -> str:
    # Tokenizer-specific adjustment happens inside the benchmark runners. This
    # deterministic text keeps all backends on the same semantic prompt seed.
    approx_words = max(1, int(target_tokens * 0.75))
    words: list[str] = []
    seed_words = SEED_TEXT.split()
    while len(words) < approx_words:
        words.extend(seed_words)
    return " ".join(words[:approx_words])


def host_metadata(repo: Path) -> dict[str, Any]:
    def optional(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                cmd, cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    metal_devices = optional(["system_profiler", "SPDisplaysDataType"]) or ""
    chip = optional(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not chip:
        chip = optional(["sysctl", "-n", "hw.model"])
    return {
        "runner": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "os": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "memory_bytes": int(optional(["sysctl", "-n", "hw.memsize"]) or 0),
        "metal_devices": [
            line.strip() for line in metal_devices.splitlines() if "Chipset Model:" in line
        ],
        "python": sys.version,
        "swift": optional(["swift", "--version"]),
        "xcode": optional(["xcodebuild", "-version"]),
    }


def git_metadata(repo: Path) -> dict[str, Any]:
    def checked(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=repo, text=True).strip()

    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
    return {
        "commit": checked(["git", "rev-parse", "HEAD"]),
        "branch": checked(["git", "branch", "--show-current"]),
        "dirty": dirty,
    }


def normalize_swift_run(payload: dict[str, Any], run_index: int, elapsed: float) -> dict[str, Any]:
    info = payload.get("generationInfo") or {}
    return {
        "run_index": run_index,
        "tokens_per_second": info.get("tokensPerSecond"),
        "prompt_tokens_per_second": info.get("promptTokensPerSecond"),
        "prompt_time_seconds": info.get("promptTime"),
        "generate_time_seconds": info.get("generateTime"),
        "elapsed_seconds": payload.get("elapsedSeconds", elapsed),
        "generated_tokens": payload.get("generatedTokens") or info.get("generationTokenCount"),
        "prompt_actual_tokens": payload.get("promptTokens") or info.get("promptTokenCount"),
        "output_characters": payload.get("outputCharacters"),
        "output_hash": hashlib.sha256((payload.get("output") or "").encode()).hexdigest(),
    }


def normalize_python_runs(
    payload: dict[str, Any], prompt_len: int
) -> tuple[list[dict[str, Any]], int | None]:
    key = f"p{prompt_len}"
    cell = (payload.get("results") or {}).get(key) or {}
    actual = cell.get("actual_prompt_len")
    runs = []
    for idx, tps in enumerate(cell.get("runs", [])):
        runs.append(
            {
                "run_index": idx,
                "tokens_per_second": tps,
                "prompt_tokens_per_second": None,
                "prompt_time_seconds": None,
                "generate_time_seconds": None,
                "elapsed_seconds": None,
                "generated_tokens": None,
                "prompt_actual_tokens": actual,
            }
        )
    return runs, actual


def write_raw(raw_dir: Path, cell_id: str, result: CommandResult, repo: Path) -> dict[str, str]:
    safe = cell_id.replace("/", "_")
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / f"{safe}.stdout"
    stderr_path = raw_dir / f"{safe}.stderr"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    def display(path: Path) -> str:
        try:
            return str(path.relative_to(repo))
        except ValueError:
            return str(path)

    return {"stdout": display(stdout_path), "stderr": display(stderr_path)}


def build_cell(
    *,
    model: ModelSpec,
    backend: str,
    cache: CacheCell,
    prompt_len: int,
    max_tokens: int,
    n_runs: int,
    warmup_runs: int,
    repo: Path,
    raw_dir: Path,
    timeout: float | None,
    runner: Runner = run_command,
) -> dict[str, Any]:
    cell_id = f"{model.id}/{backend}/{cache.name}/p{prompt_len}"
    env = os.environ.copy()
    env.update(cache.env)
    if backend == "swift":
        env["MOTIFKIT_ENABLE_MLX"] = env.get("MOTIFKIT_ENABLE_MLX", "1")

    prompt = make_prompt(prompt_len)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    ) as prompt_file:
        prompt_file.write(prompt)
        prompt_path = Path(prompt_file.name)
    try:
        if backend == "python":
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/bench_decode_e2e.py",
                "--model",
                model.path,
                "--model-id",
                model.id,
                "--max-tokens",
                str(max_tokens),
                "--n-runs",
                str(n_runs),
                "--warmup-runs",
                str(warmup_runs),
                "--prompt-lens",
                str(prompt_len),
                "--prompt-file",
                str(prompt_path),
                "--chat-template",
            ]
            result = runner(cmd, cwd=repo, env=env, timeout=timeout)
            runs, actual_prompt_tokens = normalize_python_runs(result.json or {}, prompt_len)
        elif backend == "swift":
            runs = []
            actual_prompt_tokens = None
            all_results: list[CommandResult] = []
            total_repeats = warmup_runs + n_runs
            for index in range(total_repeats):
                cmd = [
                    "swift",
                    "run",
                    "--package-path",
                    "swift",
                    "MotifNativeEvaluate",
                    "--model",
                    model.path,
                    "--mode",
                    "bench",
                    "--prompt-file",
                    str(prompt_path),
                    "--max-tokens",
                    str(max_tokens),
                    "--temperature",
                    "0",
                ]
                result = runner(cmd, cwd=repo, env=env, timeout=timeout)
                all_results.append(result)
                if (
                    index >= warmup_runs
                    and result.returncode == 0
                    and isinstance(result.json, dict)
                ):
                    normalized = normalize_swift_run(
                        result.json, index - warmup_runs, result.elapsed_seconds
                    )
                    actual_prompt_tokens = (
                        normalized.get("prompt_actual_tokens") or actual_prompt_tokens
                    )
                    runs.append(normalized)
            result = merge_command_results(all_results)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    finally:
        prompt_path.unlink(missing_ok=True)

    artifacts = write_raw(raw_dir, cell_id, result, repo)
    tps_values = [
        run["tokens_per_second"]
        for run in runs
        if isinstance(run.get("tokens_per_second"), (int, float))
    ]
    status = "pass" if result.returncode == 0 and len(runs) >= n_runs else "fail"
    return {
        "cell_id": cell_id,
        "status": status,
        "model_id": model.id,
        "backend": backend,
        "cache_cell": cache.name,
        "prompt_target_tokens": prompt_len,
        "prompt_actual_tokens": actual_prompt_tokens,
        "max_tokens": max_tokens,
        "env": cache.env,
        "runs": runs,
        "summary": summarize(tps_values),
        "artifacts": artifacts,
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
        "error": None if status == "pass" else (result.stderr[-4000:] or result.stdout[-4000:]),
    }


def merge_command_results(results: list[CommandResult]) -> CommandResult:
    if not results:
        return CommandResult([], 1, 0, "", "no runs", None)
    return CommandResult(
        command=results[-1].command,
        returncode=next(
            (r.returncode for r in results if r.returncode != 0), results[-1].returncode
        ),
        elapsed_seconds=sum(r.elapsed_seconds for r in results),
        stdout="\n".join(r.stdout for r in results),
        stderr="\n".join(r.stderr for r in results),
        json=results[-1].json,
    )


def build_comparisons(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (cell["model_id"], cell["backend"], cell["cache_cell"], cell["prompt_target_tokens"]): cell
        for cell in cells
        if cell["status"] == "pass"
    }
    comparisons = []
    for cell in cells:
        if cell["status"] != "pass":
            continue
        model_id = cell["model_id"]
        prompt = cell["prompt_target_tokens"]
        for baseline_backend, baseline_cache, label in [
            (cell["backend"], "q4_bridge", "direct_vs_bridge"),
            ("python", cell["cache_cell"], "swift_vs_python"),
        ]:
            if label == "direct_vs_bridge" and not cell["cache_cell"].endswith("direct"):
                continue
            if label == "swift_vs_python" and cell["backend"] != "swift":
                continue
            baseline = by_key.get((model_id, baseline_backend, baseline_cache, prompt))
            if not baseline or baseline is cell:
                continue
            base_tps = baseline["summary"].get("median_tokens_per_second")
            cand_tps = cell["summary"].get("median_tokens_per_second")
            if not base_tps or not cand_tps:
                continue
            comparisons.append(
                {
                    "model_id": model_id,
                    "prompt_target_tokens": prompt,
                    "comparison": label,
                    "baseline_cell": baseline["cell_id"],
                    "candidate_cell": cell["cell_id"],
                    "speedup": cand_tps / base_tps,
                }
            )
    return comparisons


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Motif benchmark sweep",
        "",
        f"Generated at: `{report['generated_at']}`  ",
        f"Commit: `{report['git']['commit']}`  ",
        f"Host: `{report['host'].get('chip') or report['host'].get('machine')}` / `{report['host'].get('os')}`",
        "",
        "## Cells",
        "",
        "| Cell | Status | Median tok/s | Prompt tokens | Runs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        median = cell["summary"].get("median_tokens_per_second")
        median_text = "n/a" if median is None else f"{median:.2f}"
        lines.append(
            f"| `{cell['cell_id']}` | {cell['status'].upper()} | {median_text} | "
            f"{cell.get('prompt_actual_tokens') or 'n/a'} | {len(cell['runs'])} |"
        )
    failed = [cell for cell in report["cells"] if cell["status"] != "pass"]
    if failed:
        lines.extend(
            ["", "## Failed cells", "", "| Cell | Return code | Error |", "| --- | ---: | --- |"]
        )
        for cell in failed:
            error = (cell.get("error") or "").replace("\n", " ")[:240]
            lines.append(f"| `{cell['cell_id']}` | {cell.get('returncode')} | `{error}` |")
    if report.get("comparisons"):
        lines.extend(
            [
                "",
                "## Comparisons",
                "",
                "| Comparison | Candidate | Baseline | Speedup |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for comp in report["comparisons"]:
            lines.append(
                f"| {comp['comparison']} | `{comp['candidate_cell']}` | `{comp['baseline_cell']}` | "
                f"{comp['speedup']:.3f}x |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Normal PR CI should validate this schema and dry-run plumbing only; real model sweeps require local or self-hosted Apple Silicon with cached checkpoints.",
            "- Treat performance parity as unproven unless this report shows Swift candidate cells meeting the Python baseline for the exact model, host, branch, and thermal conditions.",
            "- `swift_vs_python` ratios mix measurement regions: the Python cell reports steady-state decode tok/s (the first decode step is discarded as warm-up), while the Swift cell's tok/s covers every generated token including the compile-heavy first step. At low `max_tokens` the Swift first-step/compile cost dominates, so these ratios understate Swift steady-state throughput and must not be read as a steady-state speedup. Use a high `max_tokens` (and `warmup_runs >= 1`) before drawing any throughput conclusion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run_cells(
    *,
    models: list[ModelSpec],
    backends: list[str],
    caches: list[CacheCell],
    prompt_lens: list[int],
    max_tokens: int,
    n_runs: int,
) -> list[dict[str, Any]]:
    cells = []
    for model in models:
        for backend in backends:
            for cache in caches:
                for prompt_len in prompt_lens:
                    median = 100.0 if backend == "python" else 95.0
                    if cache.name.endswith("direct"):
                        median *= 1.1
                    runs = [
                        {
                            "run_index": i,
                            "tokens_per_second": median + i,
                            "prompt_tokens_per_second": None,
                            "prompt_time_seconds": None,
                            "generate_time_seconds": None,
                            "elapsed_seconds": None,
                            "generated_tokens": max_tokens,
                            "prompt_actual_tokens": prompt_len,
                        }
                        for i in range(n_runs)
                    ]
                    cell_id = f"{model.id}/{backend}/{cache.name}/p{prompt_len}"
                    cells.append(
                        {
                            "cell_id": cell_id,
                            "status": "pass",
                            "model_id": model.id,
                            "backend": backend,
                            "cache_cell": cache.name,
                            "prompt_target_tokens": prompt_len,
                            "prompt_actual_tokens": prompt_len,
                            "max_tokens": max_tokens,
                            "env": cache.env,
                            "runs": runs,
                            "summary": summarize([run["tokens_per_second"] for run in runs]),
                            "artifacts": {},
                            "command": ["dry-run"],
                            "returncode": 0,
                            "elapsed_seconds": 0,
                            "error": None,
                        }
                    )
    return cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model_spec, default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--backend", action="append", choices=["python", "swift"], default=[])
    parser.add_argument("--cache-cell", action="append", type=parse_cache_cell, default=[])
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[500, 3000])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/benchmark-sweep.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("artifacts/benchmarks/benchmark-sweep.md")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/benchmarks/raw"))
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--fail-on-cell-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    models = list(args.model)
    if args.manifest:
        models.extend(load_manifest(args.manifest))
    if not models:
        models = [ModelSpec(id="dry-model", path="/tmp/motif-dry-model")]
        args.dry_run = True
    backends = args.backend or ["python", "swift"]
    caches = args.cache_cell or [
        CacheCell("baseline", "0"),
        CacheCell("q4_direct", "q4", quant_sdpa="1"),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = (repo / args.raw_dir).resolve() if not args.raw_dir.is_absolute() else args.raw_dir

    if args.dry_run:
        cells = dry_run_cells(
            models=models,
            backends=backends,
            caches=caches,
            prompt_lens=args.prompt_lens,
            max_tokens=args.max_tokens,
            n_runs=args.n_runs,
        )
    else:
        cells = []
        for model in models:
            for backend in backends:
                for cache in caches:
                    for prompt_len in args.prompt_lens:
                        cells.append(
                            build_cell(
                                model=model,
                                backend=backend,
                                cache=cache,
                                prompt_len=prompt_len,
                                max_tokens=args.max_tokens,
                                n_runs=args.n_runs,
                                warmup_runs=args.warmup_runs,
                                repo=repo,
                                raw_dir=raw_dir,
                                timeout=args.timeout,
                            )
                        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_metadata(repo),
        "host": host_metadata(repo),
        "config": {
            "prompt_lens": args.prompt_lens,
            "max_tokens": args.max_tokens,
            "n_runs": args.n_runs,
            "warmup_runs": args.warmup_runs,
            "backends": backends,
            "cache_cells": [cache.name for cache in caches],
            "dry_run": args.dry_run,
        },
        "models": [model.__dict__ for model in models],
        "cells": cells,
        "comparisons": build_comparisons(cells),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown)}, indent=2))

    if args.fail_on_cell_error and any(cell["status"] != "pass" for cell in cells):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
