# HF Hub release checklist — converted MLX checkpoints

Target repos (draft names; final org/name is a maintainer decision):

- `<org>/Motif-2-12.7B-Reasoning-MLX-q4`  (~7 GB)
- `<org>/Motif-2.6B-MLX-q4`               (~2 GB)

`<org>` candidates: `mlx-community` (wider discovery; requires joining the org
and following its naming conventions) or a personal/org account (full control).

## Licensing (verified 2026-07-11)

Both upstream models are **Apache 2.0** and ungated (checked via the HF API:
`license:apache-2.0`, `gated: false`), so redistributing converted weights with
attribution is permitted. Model cards carry `license: apache-2.0` frontmatter
and an attribution section.

## Pre-upload steps (in order)

1. **Re-convert both checkpoints at the release commit** so provenance is
   reproducible (the current local conversions predate the release commit):
   ```bash
   mlx-motif convert --hf-path Motif-Technologies/Motif-2-12.7B-Reasoning \
     --out ./release/motif-2-12.7b-reasoning-mlx-q4 --quantize --bits 4
   mlx-motif convert --hf-path Motif-Technologies/Motif-2.6B \
     --out ./release/motif-2.6b-mlx-q4 --quantize --bits 4
   ```
2. **Validate each fresh conversion** (record outputs; fail the release on any
   regression):
   - `mlx-motif generate` smoke on both (coherent output).
   - `scripts/perplexity.py --chunk 512 --json` on both — the 12.7B must be
     ~12.37 (matching docs/benchmarks); the 2.6B number is recorded for the
     first time here (RELEASE-TODO in its card).
   - Swift native smoke: `MotifNativeGenerate --model <out> --max-tokens 32`.
3. **Fill the `<fill at release>` fields** in both model cards (mlx-motif
   commit, mlx version) and resolve the two `RELEASE-TODO` comments.
4. **Copy the model card** into each output directory as `README.md`.

## Upload

```bash
huggingface-cli login   # or HF_TOKEN env
huggingface-cli upload <org>/Motif-2-12.7B-Reasoning-MLX-q4 \
  ./release/motif-2-12.7b-reasoning-mlx-q4 . --repo-type model
huggingface-cli upload <org>/Motif-2.6B-MLX-q4 \
  ./release/motif-2.6b-mlx-q4 . --repo-type model
```

## Post-upload

- Download-from-Hub smoke on a clean path: `mlx-motif generate --model <org>/…`
  (exercises the hub-download path of `mlx_motif.load`).
- Update the repo README: Status "Open" list drops the HF Hub item; Quickstart
  gains the hub model IDs so users can skip local conversion.
- Cross-link: PR/note on the upstream Motif model cards is a courtesy option.

## Open decisions (maintainer)

- [ ] Which account/org owns the repos?
- [ ] Repo names as drafted above?
- [ ] Also publish a q8 and/or bf16 variant? (converter supports both; adds
      ~4 GB / ~26 GB respectively for the 12.7B)
