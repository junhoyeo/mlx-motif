"""CLI entry point for mlx-motif."""

from __future__ import annotations

import argparse
import sys


def _cmd_convert(args: argparse.Namespace) -> int:
    from mlx_motif.convert import convert

    out = convert(
        hf_path=args.hf_path,
        out_path=args.out,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.bits,
        q_group_size=args.group_size,
        quant_preset=args.quant_preset,
        q_proj_bits=args.q_proj_bits,
        mlp_bits=args.mlp_bits,
        mlp_group_size=args.mlp_group_size,
    )
    print(f"Wrote MLX checkpoint to {out}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from mlx_lm import generate

    from mlx_motif import load

    model, tokenizer = load(args.model)
    out = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=True,
    )
    print(out)
    return 0


def _demo_tools() -> list[dict]:
    """Two SAFE builtin tools for the CLI tool-execution demo.

    Neither tool executes arbitrary code: ``get_current_time`` reads the clock
    and ``calculator`` evaluates a numeric expression via an AST whitelist
    (``mlx_motif.tool_calls.safe_arithmetic`` — no ``eval``/``exec``).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current local date and time as an ISO 8601 string.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Evaluate a basic arithmetic expression "
                    "(supports + - * / // % ** and parentheses)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "e.g. '2 * (3 + 4)'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    ]


def _demo_tool_executor(name: str, arguments: dict) -> str:
    """Execute a builtin demo tool. Raises on unknown tool / bad arguments;
    ``run_tool_loop`` catches and surfaces the error as a tool message."""
    from datetime import datetime

    from mlx_motif.tool_calls import safe_arithmetic

    if name == "get_current_time":
        return datetime.now().isoformat(timespec="seconds")
    if name == "calculator":
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise ValueError("calculator requires a string 'expression' argument")
        return str(safe_arithmetic(expression))
    raise ValueError(f"unknown tool: {name}")


def _cmd_tools_demo(args: argparse.Namespace) -> int:
    from mlx_motif import load
    from mlx_motif.tool_calls import make_mlx_generate_fn, run_tool_loop

    model, tokenizer = load(args.model)
    tools = _demo_tools()
    generate = make_mlx_generate_fn(model, tokenizer, max_tokens=args.max_tokens)

    result = run_tool_loop(
        messages=[{"role": "user", "content": args.prompt}],
        tools=tools,
        tool_executor=_demo_tool_executor,
        generate=generate,
        max_rounds=args.max_rounds,
    )

    for i, rnd in enumerate(result.rounds, 1):
        marker = "ERROR" if rnd.is_error else "ok"
        print(f"[tool round {i}] {rnd.name}({rnd.arguments}) -> {rnd.result} ({marker})")
    print(f"[stopped: {result.stopped_reason}]")
    print(result.final_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlx-motif")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("convert", help="Convert an HF Motif checkpoint to MLX")
    pc.add_argument("--hf-path", required=True, help="HF repo id or local directory")
    pc.add_argument("--out", required=True, help="Output directory")
    pc.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    pc.add_argument("--quantize", action="store_true")
    pc.add_argument("--bits", type=int, default=4)
    pc.add_argument("--group-size", type=int, default=64)
    pc.add_argument(
        "--quant-preset",
        default="uniform",
        choices=["uniform", "mixed", "mlp_lowbit"],
        help=(
            "`mixed` keeps q_proj at --q-proj-bits, rest at --bits; "
            "`mlp_lowbit` drops MLP projections to --mlp-bits/--mlp-group-size, "
            "rest at --bits/--group-size"
        ),
    )
    pc.add_argument("--q-proj-bits", type=int, default=6)
    pc.add_argument(
        "--mlp-bits",
        type=int,
        default=3,
        help="bit width for gate_proj/up_proj/down_proj under the mlp_lowbit preset",
    )
    pc.add_argument(
        "--mlp-group-size",
        type=int,
        default=32,
        help="group_size for MLP weights under the mlp_lowbit preset",
    )
    pc.set_defaults(func=_cmd_convert)

    pg = sub.add_parser("generate", help="Run generation against a converted MLX model")
    pg.add_argument("--model", required=True, help="Path to converted MLX checkpoint")
    pg.add_argument("--prompt", required=True)
    pg.add_argument("--max-tokens", type=int, default=128)
    pg.set_defaults(func=_cmd_generate)

    ps = sub.add_parser("serve", help="Run an OpenAI-compatible HTTP server")
    ps.add_argument("--model", required=True, help="Path to converted MLX checkpoint")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--model-id", default="motif")
    ps.add_argument("--think-mode", default="visible", choices=["visible", "hidden", "captured"])
    ps.set_defaults(
        func=lambda a: (
            __import__("mlx_motif.server", fromlist=["serve"]).serve(
                a.model, a.host, a.port, a.model_id, a.think_mode
            )
            or 0
        )
    )

    pt = sub.add_parser(
        "tools-demo",
        help="Run the tool-EXECUTION loop with 2 safe builtin tools (time, calculator)",
    )
    pt.add_argument("--model", required=True, help="Path to converted MLX checkpoint")
    pt.add_argument("--prompt", required=True)
    pt.add_argument("--max-tokens", type=int, default=256)
    pt.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum tool-execution cycles before stopping",
    )
    pt.set_defaults(func=_cmd_tools_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
