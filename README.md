# mlx-motif

The canonical [MLX](https://github.com/ml-explore/mlx) port of [Motif](https://huggingface.co/Motif-Technologies) language models on Apple Silicon.

Motif's flagship LLMs combine **Differential Attention** ([2410.05258](https://arxiv.org/abs/2410.05258)) — and its grouped variant **GDA** ([2510.06949](https://arxiv.org/abs/2510.06949)) for the 12.7B class — with the **PolyNorm** activation ([2411.03884](https://arxiv.org/abs/2411.03884)). This repo implements those primitives natively in MLX, so the models run efficiently on M-series Macs without going through PyTorch.

## Status

Phase 1 — package scaffolding, structural model port, weight converter. Verified shapes; numerical parity testing in progress.

| Phase | Status |
| --- | --- |
| 1 — Package + structural port + tests | in progress |
| 2 — Speculative decoding (2.6B → 12.7B) | planned |
| 3 — Mixed-precision quantization | planned |
| 4 — Fused GDA + PolyNorm Metal kernels | planned |
| 5 — OpenAI-compatible server with `<think>` streaming | planned |

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Python ≥ 3.11, MLX ≥ 0.21, an Apple Silicon Mac.

## Convert a checkpoint

```bash
# bfloat16
mlx-motif convert \
  --hf-path Motif-Technologies/Motif-2.6B \
  --out ./out/motif-2.6b

# 4-bit quantized
mlx-motif convert \
  --hf-path Motif-Technologies/Motif-2-12.7B-Reasoning \
  --out ./out/motif-12.7b-q4 \
  --quantize --bits 4
```

## Generate

```bash
mlx-motif generate --model ./out/motif-12.7b-q4 --prompt "Hello, world."
```

Or programmatically:

```python
from mlx_lm import generate
from mlx_motif import load

model, tokenizer = load("./out/motif-12.7b-q4")
print(generate(model, tokenizer, prompt="...", max_tokens=128))
```

`mlx_motif.load` wires our `Model` class into mlx-lm's loader; everything downstream (`generate`, `mlx_lm.server`, speculative decoding) works the same as for stock mlx-lm models.

## Architecture notes

- **GDA dispatch** — when `num_noise_heads` is present in the config, the attention module runs the grouped path; otherwise it falls back to vanilla differential attention. Same `Model` class, same converter.
- **λ stability** — the per-layer λ parameters (`lambda_q1/k1/q2/k2`) are kept in fp32; only the final difference is downcast.
- **PolyNorm** — implemented as plain MLX ops today. Fusion target for Phase 4.
- **KV cache** — uses the standard `mlx_lm` cache contract, so speculative decoding via `--draft-model` works for Motif-2.6B → Motif-2-12.7B (same tokenizer, vocab 219520).

## Layout

```
src/mlx_motif/
  model.py      # PolyNorm, MotifAttention (vanilla + GDA), Model
  convert.py    # HF safetensors -> MLX safetensors
  __main__.py   # CLI: convert | generate
tests/          # structural tests
examples/       # convert + generate scripts
```

## License

MIT — see `LICENSE`. Motif checkpoints retain their original Motif Technologies license; this port does not redistribute weights.
