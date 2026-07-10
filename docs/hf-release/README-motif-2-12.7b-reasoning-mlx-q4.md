---
license: apache-2.0
base_model: Motif-Technologies/Motif-2-12.7B-Reasoning
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - quantized
  - 4-bit
  - grouped-differential-attention
  - polynorm
---

# Motif-2-12.7B-Reasoning — MLX 4-bit

4-bit MLX conversion of [Motif-Technologies/Motif-2-12.7B-Reasoning](https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Reasoning) for Apple Silicon, produced by [mlx-motif](https://github.com/junhoyeo/mlx-motif) — the MLX port of Motif's **Grouped Differential Attention** + **PolyNorm** architecture, with custom Metal kernels for the differential-attention decode path.

Neither architecture primitive ships in mlx-lm, so this checkpoint requires `mlx-motif` (it registers the model class into mlx-lm's loader); it will not load with stock `mlx_lm.load`.

## Usage

```bash
git clone https://github.com/junhoyeo/mlx-motif && cd mlx-motif
uv pip install -e .

mlx-motif generate --model <this-repo> --prompt "Hello, world."
mlx-motif serve --model <this-repo> --port 8080   # OpenAI-compatible
```

```python
from mlx_lm import generate
from mlx_motif import load

model, tokenizer = load("<this-repo>")
print(generate(model, tokenizer, prompt="…", max_tokens=128))
```

## Conversion provenance

- Converter: `mlx-motif convert --hf-path Motif-Technologies/Motif-2-12.7B-Reasoning --out … --quantize --bits 4` (group_size 64, uniform preset)
- mlx-motif: [github.com/junhoyeo/mlx-motif](https://github.com/junhoyeo/mlx-motif) @ `e6c401a` (converted with this repo's `convert.py`; validated at this commit)
- mlx version: 0.31.2

## Validation (measured on Apple M1 Max, 64 GB)

- **Perplexity**: 12.365 (nll/token 2.5149, 592 tokens, `scripts/perplexity.py --chunk 512`).
- **Kernel-path parity**: greedy output on this checkpoint is byte-identical between mlx-motif's custom Metal kernels and its pure-MLX reference path (32/48/96-token checks on real weights). Parity against the HF PyTorch reference is verified at bf16, **not** at q4 — 4-bit quality is characterized by the perplexity above instead.
- **Decode throughput** (July 2026, `bench_decode_e2e.py`, median of 5 runs, `max_tokens=64`, default configuration):

| Prompt length | tok/s |
|---|---|
| 5 | 40.9 |
| 164 | 40.0 |
| 800 | 38.2 |
| 3204 | 30.7 |

## Notes & limitations

- Motif's chat template pre-opens a `<think>` reasoning block; `mlx-motif serve --think-mode visible|hidden|captured` controls how the trace is surfaced.
- Kernel constants are tuned on M1 Max; other M-series chips work but have not been performance-validated.
- Quantization: uniform 4-bit, group size 64. Mixed-precision presets are available in the converter if you want a different quality/size point.

## License & attribution

The model weights are derivative of Motif Technologies' release and remain under **Apache 2.0**, © Motif Technologies. The conversion tooling is MIT (mlx-motif). If you use this model, please attribute the original model card.
