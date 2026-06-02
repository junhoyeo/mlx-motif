"""
End-to-end smoke eval for an MLX Motif checkpoint.

Runs a short generation + perplexity pass and writes a grounded JSON report.
This is a local smoke (n=1, one machine) — NOT certified performance numbers.

Usage:
    python scripts/eval_smoke.py --model ~/.models/motif-2.6b-mlx-q4
    python scripts/eval_smoke.py  # uses MOTIF_MODEL_DIR env or default path

Exit codes:
    0  — success, or model dir absent and --require-model not set
    1  — model dir absent and --require-model set, or runtime error
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Fixed short prompts that exercise different continuations.
_EVAL_PROMPTS = [
    "The unified memory architecture of Apple Silicon means that",
    "Differential attention cancels noise by computing the difference between",
    "Quantizing a language model to four bits reduces memory by",
]

# Default checkpoint location (env override → explicit default path).
_DEFAULT_MODEL_DIR = os.environ.get(
    "MOTIF_MODEL_DIR",
    str(Path.home() / ".models" / "motif-2.6b-mlx-q4"),
)

# Maximum tokens for generation smoke runs (keep it short — this is a smoke).
_GEN_MAX_TOKENS = 48

# Disclaimer required on every report.
_DISCLAIMER = (
    "LOCAL SMOKE ONLY — n=1, single machine, single thermal state. "
    "These numbers are NOT certified performance claims. "
    "Do not cite them as representative throughput without a full certified sweep."
)


# ---------------------------------------------------------------------------
# Pure-logic helpers (import-safe, no MLX dependency)
# ---------------------------------------------------------------------------


def aggregate_tok_per_sec(values: list[float]) -> dict[str, float | None]:
    """Return mean/min/max tok/s from a list of per-run measurements.

    Returns None-valued keys when the list is empty so callers can always
    key into the returned dict without branching.
    """
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def assemble_report(
    *,
    model: str,
    generations: list[dict[str, Any]],
    perplexity: float | None,
    peak_memory_gb: float | None,
    host: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final JSON report dict from measured components.

    Keeps timestamp as a placeholder string so unit tests can inject fake data
    without freezing time; the real runner substitutes a real ISO timestamp.
    """
    tps_values = [g["tok_per_sec"] for g in generations if g.get("tok_per_sec") is not None]
    return {
        "schema": "eval-smoke-v1",
        "disclaimer": _DISCLAIMER,
        "model": model,
        "timestamp": "PLACEHOLDER",
        "host": host,
        "generations": generations,
        "tok_per_sec_summary": aggregate_tok_per_sec(tps_values),
        "perplexity": perplexity,
        "peak_memory_gb": peak_memory_gb,
    }


# ---------------------------------------------------------------------------
# Host / git metadata (mirrors bench_sweep.py style)
# ---------------------------------------------------------------------------


def _optional_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def host_metadata() -> dict[str, Any]:
    chip = _optional_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or _optional_cmd(
        ["sysctl", "-n", "hw.model"]
    )
    mem_bytes_raw = _optional_cmd(["sysctl", "-n", "hw.memsize"])
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "memory_bytes": int(mem_bytes_raw) if mem_bytes_raw and mem_bytes_raw.isdigit() else None,
        "python": sys.version,
    }


# ---------------------------------------------------------------------------
# Generation runner — imports mlx_lm lazily so the module stays importable
# without MLX installed (for CI skip path and unit tests).
# ---------------------------------------------------------------------------


def run_generation(model_obj: Any, tokenizer: Any, prompt: str, max_tokens: int) -> dict[str, Any]:
    """Run one generation and return {prompt, output, tokens, tok_per_sec}."""
    from mlx_lm import generate

    t0 = time.perf_counter()
    output = generate(model_obj, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    elapsed = time.perf_counter() - t0

    # Count decode tokens (best-effort: tokenize the output).
    try:
        if hasattr(tokenizer, "encode"):
            out_ids = tokenizer.encode(output)
        else:
            out_ids = tokenizer(output)["input_ids"]
        n_tokens = len(out_ids)
    except Exception:
        # Fall back to a rough word-count estimate.
        n_tokens = len(output.split())

    tok_per_sec = n_tokens / elapsed if elapsed > 0 else None
    return {
        "prompt": prompt,
        "output": output,
        "tokens": n_tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_per_sec": round(tok_per_sec, 2) if tok_per_sec is not None else None,
    }


def get_peak_memory_gb() -> float | None:
    """Return current MLX peak memory in GB, or None if unavailable."""
    try:
        import mlx.core as mx

        return round(mx.metal.get_peak_memory() / (1024**3), 3)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main eval routine
# ---------------------------------------------------------------------------


def run_eval(model_dir: str, max_tokens: int = _GEN_MAX_TOKENS) -> dict[str, Any]:
    """Load model, run generations + perplexity, return assembled report."""
    # Lazy imports — only reached when we have a real model dir.
    import importlib.util

    from mlx_motif import load

    # Import perplexity logic from the sibling script; avoid duplicating math.

    _scripts_dir = Path(__file__).resolve().parent
    _ppl_spec = importlib.util.spec_from_file_location(
        "_eval_perplexity", _scripts_dir / "perplexity.py"
    )
    assert _ppl_spec and _ppl_spec.loader
    _ppl_mod = importlib.util.module_from_spec(_ppl_spec)
    _ppl_spec.loader.exec_module(_ppl_mod)  # type: ignore[union-attr]
    _perplexity_fn = _ppl_mod.perplexity  # type: ignore[attr-defined]

    print(f"Loading model from {model_dir} …", file=sys.stderr)
    model_obj, tokenizer = load(model_dir)

    # --- Generations ---
    generations: list[dict[str, Any]] = []
    for prompt in _EVAL_PROMPTS:
        print(f"  generate: {prompt[:60]!r} …", file=sys.stderr)
        result = run_generation(model_obj, tokenizer, prompt, max_tokens)
        generations.append(result)

    peak_mem = get_peak_memory_gb()

    # --- Perplexity ---
    print("  perplexity: evaluating fixed text corpus …", file=sys.stderr)
    ppl_result = _perplexity_fn(model_obj, tokenizer, _ppl_mod._DEFAULT_TEXT)
    perplexity_val = round(ppl_result["ppl"], 3)

    report = assemble_report(
        model=str(Path(model_dir).resolve()),
        generations=generations,
        perplexity=perplexity_val,
        peak_memory_gb=peak_mem,
        host=host_metadata(),
    )
    # Stamp real timestamp.
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


# ---------------------------------------------------------------------------
# Human summary printer
# ---------------------------------------------------------------------------


def print_summary(report: dict[str, Any]) -> None:
    model_name = Path(report["model"]).name
    chip = (report.get("host") or {}).get("chip") or "unknown chip"
    tps = report.get("tok_per_sec_summary") or {}
    ppl = report.get("perplexity")
    mem = report.get("peak_memory_gb")

    print()
    print("=" * 60)
    print("  eval-smoke summary (LOCAL SMOKE — n=1, not certified)")
    print("=" * 60)
    print(f"  model     : {model_name}")
    print(f"  chip      : {chip}")
    if tps.get("mean") is not None:
        print(f"  tok/s     : mean={tps['mean']:.1f}  min={tps['min']:.1f}  max={tps['max']:.1f}")
    else:
        print("  tok/s     : n/a")
    if ppl is not None:
        print(f"  perplexity: {ppl:.3f}")
    else:
        print("  perplexity: n/a")
    if mem is not None:
        print(f"  peak mem  : {mem:.3f} GB")
    else:
        print("  peak mem  : n/a")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end smoke eval: load → generate → perplexity → report."
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_DIR,
        help=(
            "Path to converted MLX checkpoint directory. "
            "Defaults to $MOTIF_MODEL_DIR or ~/.models/motif-2.6b-mlx-q4."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the JSON report. "
            "Defaults to docs/benchmarks/eval-smoke-<timestamp>.json "
            "if that directory exists, else /tmp/eval-smoke-<timestamp>.json."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_GEN_MAX_TOKENS,
        help=f"Max decode tokens per generation prompt (default: {_GEN_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Exit non-zero if the model directory does not exist (default: skip gracefully).",
    )
    return parser.parse_args(argv)


def _default_output_path(timestamp: str) -> Path:
    filename = f"eval-smoke-{timestamp}.json"
    candidate = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / filename
    if candidate.parent.is_dir():
        return candidate
    return Path("/tmp") / filename


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_dir = Path(args.model)

    if not model_dir.exists():
        if args.require_model:
            print(
                f"ERROR: model directory not found: {model_dir}\n"
                "Pass --model <dir> or set MOTIF_MODEL_DIR.",
                file=sys.stderr,
            )
            return 1
        print(
            f"eval-smoke: model directory not found ({model_dir}) — skipping (exit 0).\n"
            "Set MOTIF_MODEL_DIR or pass --model <dir> to run a real eval.",
            file=sys.stderr,
        )
        return 0

    try:
        report = run_eval(str(model_dir), max_tokens=args.max_tokens)
    except Exception as exc:
        print(f"ERROR during eval: {exc}", file=sys.stderr)
        raise

    timestamp = report.get("timestamp", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    # Build a compact timestamp for the filename.
    file_ts = timestamp.replace(":", "").replace("-", "")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _default_output_path(file_ts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print_summary(report)
    print(f"Report written to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
