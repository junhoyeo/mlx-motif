from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_bench_sweep():
    spec = importlib.util.spec_from_file_location("bench_sweep", Path("scripts/bench_sweep.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_sweep"] = module
    spec.loader.exec_module(module)
    return module


def test_model_and_cache_cell_parsing() -> None:
    bench_sweep = _load_bench_sweep()

    model = bench_sweep.parse_model_spec("motif-12.7b-q4=.models/motif-12.7b-reasoning-q4")
    assert model.id == "motif-12.7b-q4"
    assert model.path == ".models/motif-12.7b-reasoning-q4"

    cache = bench_sweep.parse_cache_cell("q4_bridge:q4,quant_sdpa=0,disable_kernels=1")
    assert cache.name == "q4_bridge"
    assert cache.env == {
        "MLX_MOTIF_4SLOT_CACHE": "q4",
        "MLX_MOTIF_QUANT_SDPA": "0",
        "MLX_MOTIF_DISABLE_KERNELS": "1",
    }

    direct = bench_sweep.parse_cache_cell("q4_direct:q4,quant_sdpa=1")
    assert direct.disable_kernels is False
    assert direct.env["MLX_MOTIF_DISABLE_KERNELS"] == "0"


def test_dry_run_cli_writes_schema_report(tmp_path: Path) -> None:
    output = tmp_path / "sweep.json"
    markdown = tmp_path / "sweep.md"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/bench_sweep.py",
            "--dry-run",
            "--model",
            "motif-smoke=/tmp/motif-smoke",
            "--backend",
            "python",
            "--backend",
            "swift",
            "--cache-cell",
            "q4_bridge:q4,quant_sdpa=0,disable_kernels=1",
            "--cache-cell",
            "q4_direct:q4,quant_sdpa=1",
            "--prompt-lens",
            "500",
            "--max-tokens",
            "4",
            "--n-runs",
            "2",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["config"]["dry_run"] is True
    assert {cell["backend"] for cell in report["cells"]} == {"python", "swift"}
    assert {cell["cache_cell"] for cell in report["cells"]} == {"q4_bridge", "q4_direct"}
    assert all(cell["status"] == "pass" for cell in report["cells"])
    assert report["comparisons"], "dry-run matrix should include comparison rows"
    assert "Motif benchmark sweep" in markdown.read_text()


def test_failed_cell_is_preserved(tmp_path: Path) -> None:
    bench_sweep = _load_bench_sweep()

    def fake_runner(*_args: Any, **_kwargs: Any):
        return bench_sweep.CommandResult(
            command=["swift", "run"],
            returncode=42,
            elapsed_seconds=0.1,
            stdout="not json",
            stderr="boom",
            json=None,
        )

    cell = bench_sweep.build_cell(
        model=bench_sweep.ModelSpec(id="motif", path="/tmp/missing"),
        backend="swift",
        cache=bench_sweep.CacheCell("q4_direct", "q4", quant_sdpa="1"),
        prompt_len=128,
        max_tokens=4,
        n_runs=1,
        warmup_runs=0,
        repo=Path.cwd(),
        raw_dir=tmp_path,
        timeout=1,
        runner=fake_runner,
    )
    assert cell["status"] == "fail"
    assert cell["returncode"] == 42
    assert "boom" in cell["error"]
    assert Path(cell["artifacts"]["stderr"]).read_text() == "boom"


def test_command_timeout_is_preserved_as_failed_cell() -> None:
    bench_sweep = _load_bench_sweep()
    result = bench_sweep.run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=Path.cwd(),
        env={},
        timeout=0.1,
    )
    assert result.returncode == 124
    assert "timed out" in result.stderr


def _cell(cell_id: str, backend: str, cache: str, target: int, actual: int, tps: float) -> dict:
    return {
        "cell_id": cell_id,
        "model_id": "m",
        "backend": backend,
        "cache_cell": cache,
        "prompt_target_tokens": target,
        "prompt_actual_tokens": actual,
        "status": "pass",
        "summary": {"median_tokens_per_second": tps},
        "runs": [{}],
    }


def test_comparison_flags_prompt_token_mismatch() -> None:
    bench_sweep = _load_bench_sweep()
    cells = [
        _cell("m/python/q4_direct/p500", "python", "q4_direct", 500, 555, 30.0),
        _cell("m/swift/q4_direct/p500", "swift", "q4_direct", 500, 469, 10.0),
    ]
    comps = bench_sweep.build_comparisons(cells)
    swift = next(c for c in comps if c["comparison"] == "swift_vs_python")
    assert swift["candidate_prompt_tokens"] == 469
    assert swift["baseline_prompt_tokens"] == 555
    assert swift["prompt_tokens_match"] is False


def test_comparison_marks_matching_prompt_tokens() -> None:
    bench_sweep = _load_bench_sweep()
    cells = [
        _cell("m/python/q4_direct/p500", "python", "q4_direct", 500, 462, 30.0),
        _cell("m/swift/q4_direct/p500", "swift", "q4_direct", 500, 462, 10.0),
    ]
    comps = bench_sweep.build_comparisons(cells)
    swift = next(c for c in comps if c["comparison"] == "swift_vs_python")
    assert swift["prompt_tokens_match"] is True
