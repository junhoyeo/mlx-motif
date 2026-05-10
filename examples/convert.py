"""
Convert a Motif checkpoint from Hugging Face to MLX.

Examples:

    # Plain bfloat16 conversion
    python examples/convert.py Motif-Technologies/Motif-2.6B ./out/motif-2.6b

    # 4-bit quantized
    python examples/convert.py Motif-Technologies/Motif-2-12.7B-Reasoning \\
        ./out/motif-12.7b-q4 --quantize --bits 4
"""

from __future__ import annotations

import argparse
import sys

from mlx_motif.convert import convert


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("hf_path")
    p.add_argument("out")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    args = p.parse_args()

    out = convert(
        hf_path=args.hf_path,
        out_path=args.out,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.bits,
        q_group_size=args.group_size,
    )
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
