---
license: apache-2.0
base_model: Motif-Technologies/Motif-2.6B
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - quantized
  - 4-bit
  - differential-attention
  - polynorm
---

# Motif-2.6B — MLX 4-bit

4-bit MLX conversion of [Motif-Technologies/Motif-2.6B](https://huggingface.co/Motif-Technologies/Motif-2.6B) for Apple Silicon, produced by [mlx-motif](https://github.com/junhoyeo/mlx-motif) — the MLX port of Motif's Differential Attention + **PolyNorm** architecture. This 2.6B variant uses the ungrouped ("vanilla") differential-attention form.

This checkpoint requires `mlx-motif` (it registers the model class into mlx-lm's loader); it will not load with stock `mlx_lm.load`.

## Usage

```bash
git clone https://github.com/junhoyeo/mlx-motif && cd mlx-motif
uv pip install -e .

mlx-motif generate --model <this-repo> --prompt "Hello, world."
mlx-motif serve --model <this-repo> --port 8080   # OpenAI-compatible
```

## Conversion provenance

- Converter: `mlx-motif convert --hf-path Motif-Technologies/Motif-2.6B --out … --quantize --bits 4` (group_size 64, uniform preset)
- mlx-motif: [github.com/junhoyeo/mlx-motif](https://github.com/junhoyeo/mlx-motif) @ `e6c401a` (converted with this repo's `convert.py`; validated at this commit)
- mlx version: 0.31.2

## Validation (measured on Apple M1 Max, 64 GB)

- End-to-end greedy generation verified on real weights (Python CLI, OpenAI server, and the native Swift runtime).
- **Known numerical note**: on this checkpoint, mlx-motif's custom-kernel path and its pure-MLX reference path produce output that diverges within a few greedy tokens — the two paths accumulate float reductions in different orders and the low-order-bit difference can flip the argmax on this architecture. Both outputs are valid samples of the model; this is documented (and expected) behavior, unlike the 12.7B grouped checkpoint where the two paths are byte-identical.
- Parity against the HF PyTorch reference is verified at bf16, **not** at q4.
- A q4 perplexity number for this checkpoint has not yet been recorded (the 12.7B's is 12.365); it will be added on the next benchmark pass.

## License & attribution

The model weights are derivative of Motif Technologies' release and remain under **Apache 2.0**, © Motif Technologies. The conversion tooling is MIT (mlx-motif). If you use this model, please attribute the original model card.
