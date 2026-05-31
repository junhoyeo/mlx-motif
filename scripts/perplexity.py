"""
Perplexity eval for the MLX Motif port.

Computes negative log-likelihood per token on a fixed text corpus and
reports perplexity. Loads the model once, tokenizes, runs forward passes
in chunks, accumulates -log p(actual next token).

Usage:
    python scripts/perplexity.py \\
        --model ./out/motif-12.7b-reasoning-q4 \\
        --text-file ./scripts/eval_text.txt \\
        --chunk 512

The text file should be plain UTF-8. We use a fixed snippet of WikiText-2
style content (constitutional / encyclopedic prose) to keep the bench
deterministic and not dependent on a network download.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_motif import load


def perplexity(
    model: nn.Module,
    tokenizer,
    text: str,
    chunk_size: int = 512,
    max_tokens: int = 8192,
) -> dict:
    """Compute perplexity over `text` by chunked forward + softmax-loss."""
    # Tokenize whole corpus.
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
    if isinstance(ids, mx.array):
        ids = ids.tolist()
    if max_tokens > 0:
        ids = ids[:max_tokens]
    if len(ids) < 2:
        raise ValueError("Need at least 2 tokens to compute next-token loss")

    total_nll = 0.0
    total_tokens = 0
    chunks = max(1, (len(ids) + chunk_size - 1) // chunk_size)

    t0 = time.perf_counter()
    for i in range(chunks):
        start = i * chunk_size
        end = min(len(ids), start + chunk_size + 1)
        chunk = ids[start:end]
        if len(chunk) < 2:
            continue
        x = mx.array([chunk[:-1]])
        y = mx.array([chunk[1:]])

        # Forward
        logits = model(x)  # (1, L, V) — fp32 inside, may be bf16 storage
        # Cross-entropy via log_softmax + gather
        log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        # Gather the actual next-token log-probs
        gather = mx.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        nll = -gather.sum().item()
        total_nll += nll
        total_tokens += y.size

    elapsed = time.perf_counter() - t0
    avg_nll = total_nll / total_tokens
    return {
        "ppl": math.exp(avg_nll),
        "nll_per_token": avg_nll,
        "tokens": total_tokens,
        "chunks": chunks,
        "elapsed_s": elapsed,
        "tps": total_tokens / elapsed,
    }


def logit_snapshot(
    model: nn.Module, tokenizer, text: str, max_tokens: int = 512, top_k: int = 10
) -> dict:
    """Return a deterministic final-position logit checksum/top-k snapshot."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
    if isinstance(ids, mx.array):
        ids = ids.tolist()
    if max_tokens > 0:
        ids = ids[:max_tokens]
    if not ids:
        raise ValueError("Need at least 1 token to compute logits")

    t0 = time.perf_counter()
    logits = model(mx.array([ids])).astype(mx.float32)
    last = logits[0, len(ids) - 1]
    mx.eval(last)
    values = last.tolist()
    top = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[: max(1, top_k)]
    elapsed = time.perf_counter() - t0
    return {
        "prompt_tokens": len(ids),
        "vocabulary_size": len(values),
        "checksum": float(sum(values)),
        "top_k": [{"token": int(token), "logit": float(logit)} for token, logit in top],
        "elapsed_s": elapsed,
    }


# 4 paragraphs of encyclopedic prose (~2k tokens with most tokenizers).
_DEFAULT_TEXT = """
The differential transformer is a class of attention mechanism introduced in 2024 that
computes attention as the difference between two softmax distributions weighted by a
learnable scalar. This formulation effectively cancels noise in attention scores and
encourages sparser, more focused attention patterns. The grouped variant extends this
by sharing noise heads across multiple origin heads in fixed group sizes, reducing the
parameter count of the noise-generating projection while preserving most of the
denoising benefit. PolyNorm is a related innovation: a polynomial activation that
applies RMS normalization to three power-transformed copies of the input, giving the
network a learnable trade-off between linear, quadratic, and cubic feature
representations. Together these primitives form the architectural backbone of the
Motif-2 family of language models released by Motif Technologies in early 2026.

Apple Silicon GPUs present a distinctive performance landscape compared to traditional
discrete GPUs. The unified memory architecture means that the CPU and GPU share the
same physical memory pool, which eliminates the explicit copy step that dominates many
naive transcoder kernels. However, the on-chip register file per execution unit is
markedly smaller than NVIDIA's per-streaming-multiprocessor budget, which constrains
how aggressively a Metal kernel can stage intermediate state in registers before
spilling to threadgroup memory. The MLX framework, developed by Apple's machine learning
research group, provides a Python-first abstraction over Metal that aims to deliver
PyTorch-like ergonomics with native Apple Silicon performance.

Quantization in the context of large language model inference refers to the practice of
representing model weights and, optionally, key-value cache entries using fewer bits
than the original training precision. Four-bit weight quantization with a group size of
sixty-four elements is a popular operating point: it reduces memory consumption by a
factor of roughly four versus bfloat16 while typically preserving model quality within
a few percent on standard evaluation suites. The MLX quantized matrix-multiply kernel
fuses the dequantization step with the matrix-multiply itself, avoiding a separate
materialization of the full-precision weight tensor and thereby reducing memory bandwidth
pressure during decoding.

Speculative decoding accelerates autoregressive generation by using a small draft model
to propose multiple tokens at once, which the target model then verifies in parallel.
The wall-clock speedup depends on the acceptance rate of draft tokens, the per-token
latency ratio between draft and target models, and the dispatch overhead of running
both models in alternation. In practice, the technique tends to deliver one and a half
to two times decoding throughput improvements on commodity hardware when the draft
model is at least an order of magnitude smaller than the target. For models without a
suitably small draft, prompt-lookup decoding offers a draft-free alternative by
searching the recent context for matching n-gram continuations.
""".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to converted MLX checkpoint")
    p.add_argument(
        "--text-file", default=None, help="Optional UTF-8 text file (defaults to bundled snippet)"
    )
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument(
        "--max-tokens", type=int, default=2048, help="Cap on tokens evaluated; 0 = no cap"
    )
    p.add_argument("--mode", choices=["perplexity", "logits"], default="perplexity")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = _DEFAULT_TEXT

    if not args.quiet:
        print(f"Loading model from {args.model} …", file=sys.stderr)
    model, tokenizer = load(args.model)

    if args.mode == "logits":
        if not args.quiet:
            print(
                f"Tokenizing + evaluating final-position top-{args.top_k} logits …", file=sys.stderr
            )
        res = logit_snapshot(model, tokenizer, text, max_tokens=args.max_tokens, top_k=args.top_k)
    else:
        if not args.quiet:
            print(f"Tokenizing + evaluating chunks of {args.chunk} tokens …", file=sys.stderr)
        res = perplexity(model, tokenizer, text, chunk_size=args.chunk, max_tokens=args.max_tokens)

    if args.json:
        print(
            json.dumps(
                {"mode": args.mode, "model": args.model, "result": res}, indent=2, sort_keys=True
            )
        )
    elif args.mode == "logits":
        print(f"prompt tokens: {res['prompt_tokens']}")
        print(f"vocab size:    {res['vocabulary_size']}")
        print(f"checksum:      {res['checksum']:.6f}")
        print("top_k:")
        for item in res["top_k"]:
            print(f"  {item['token']}: {item['logit']:.6f}")
    else:
        print(f"perplexity:   {res['ppl']:.3f}")
        print(f"nll/token:    {res['nll_per_token']:.4f}")
        print(f"tokens:       {res['tokens']}")
        print(f"chunks:       {res['chunks']}")
        print(f"elapsed:      {res['elapsed_s']:.2f}s")
        print(f"tps:          {res['tps']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
