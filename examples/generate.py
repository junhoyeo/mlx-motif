"""
Quick generation example. Assumes you've already converted a checkpoint:

    mlx-motif convert --hf-path Motif-Technologies/Motif-2.6B --out ./out/motif-2.6b
    python examples/generate.py ./out/motif-2.6b "Hello, world."
"""

from __future__ import annotations

import sys

from mlx_lm import generate

from mlx_motif import load


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: generate.py <model-path> <prompt> [max_tokens]", file=sys.stderr)
        return 2
    model_path = sys.argv[1]
    prompt = sys.argv[2]
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    model, tokenizer = load(model_path)
    print(generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
