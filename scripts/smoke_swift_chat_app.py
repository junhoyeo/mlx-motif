#!/usr/bin/env python3
"""Smoke the Swift chat app's backend surfaces.

This does not click the SwiftUI controls. It validates the same backend paths
that the app selector uses:

* native checkpoint generation through MotifNativeGenerate
* OpenAI-compatible fallback through the native Swift MotifNativeServe process
* cancellation by terminating an in-flight native generation process

Use --require-model in release/evidence runs so missing local checkpoints fail
instead of skipping.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    input_text: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 30
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data)
    if payload is not None:
        request.add_header("content-type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def terminate_process_group(proc: subprocess.Popen[str], timeout: float = 10) -> tuple[str, str]:
    try:
        os.killpg(proc.pid, 15)
    except ProcessLookupError:
        pass
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        return proc.communicate(timeout=timeout)


def wait_for_native_server(
    proc: subprocess.Popen[str], port: int, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not polled"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=5)
            return {
                "status": "exited",
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        try:
            models = request_json(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return {"status": "ready", "models": models}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = repr(error)
            time.sleep(0.5)
    return {"status": "timeout", "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=os.environ.get("MOTIF_MODEL_DIR", ".models/motif-2.6b-mlx-q4")
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    model = Path(args.model)
    report: dict[str, Any] = {
        "model": str(model),
        "required": args.require_model,
        "checks": {},
    }

    if not model.exists():
        report["checks"]["model"] = {"status": "missing"}
        if args.require_model:
            print(json.dumps(report, indent=2))
            return 2
        print(json.dumps(report, indent=2))
        return 0

    env = os.environ.copy()
    env["MOTIFKIT_ENABLE_MLX"] = "1"

    print("smoke: native generation", file=sys.stderr, flush=True)
    native = run(
        [
            "swift",
            "run",
            "--package-path",
            "swift",
            "MotifNativeGenerate",
            "--model",
            str(model),
            "--prompt",
            "Reply with one short sentence.",
            "--max-tokens",
            str(args.max_tokens),
            "--json",
        ],
        cwd=repo,
        env=env,
        timeout=args.timeout,
    )
    report["checks"]["native_generate"] = native
    if native["returncode"] != 0:
        print(json.dumps(report, indent=2))
        return 1

    print("smoke: cancellation", file=sys.stderr, flush=True)
    cancel_proc = subprocess.Popen(
        [
            "swift",
            "run",
            "--package-path",
            "swift",
            "MotifNativeGenerate",
            "--model",
            str(model),
            "--prompt",
            "Write a long explanation of grouped differential attention.",
            "--max-tokens",
            "256",
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    time.sleep(2)
    stdout, stderr = terminate_process_group(cancel_proc)
    report["checks"]["cancel_generation"] = {
        "returncode": cancel_proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "terminated": cancel_proc.returncode not in (0, None),
    }

    port = open_port()
    print("smoke: native Swift OpenAI-compatible server", file=sys.stderr, flush=True)
    server_proc = subprocess.Popen(
        [
            "swift",
            "run",
            "--package-path",
            "swift",
            "MotifNativeServe",
            "--model",
            str(model),
            "--port",
            str(port),
        ],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        readiness = wait_for_native_server(server_proc, port, timeout=args.timeout)
        report["checks"]["server_ready"] = readiness
        if readiness.get("status") != "ready":
            print(json.dumps(report, indent=2))
            return 1
        completion = request_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Say hello in three words."}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": False,
            },
            timeout=args.timeout,
        )
        report["checks"]["server_fallback"] = {
            "status": "ok",
            "response": completion,
        }
    finally:
        server_stdout, server_stderr = terminate_process_group(server_proc)
        report["checks"]["server_process"] = {
            "returncode": server_proc.returncode,
            "stdout": server_stdout,
            "stderr": server_stderr,
            "terminated": server_proc.returncode not in (0, None),
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
