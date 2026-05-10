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
    pc.add_argument("--quant-preset", default="uniform", choices=["uniform", "mixed"],
                    help="`mixed` keeps q_proj at --q-proj-bits, rest at --bits")
    pc.add_argument("--q-proj-bits", type=int, default=6)
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
    ps.add_argument("--think-mode", default="visible",
                    choices=["visible", "hidden", "captured"])
    ps.set_defaults(func=lambda a: __import__("mlx_motif.server", fromlist=["serve"])
                    .serve(a.model, a.host, a.port, a.model_id, a.think_mode) or 0)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
