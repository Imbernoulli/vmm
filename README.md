# Visualizing Model Merging

This repository turns `proposal.md` into a runnable research artifact for task-vector model-merging visualizations.

See `RESEARCH_REPORT.md` for the current findings and artifact map, and `PAPER.md` for a paper-style writeup of the completed evidence.

The current implementation has several layers:

1. A complete controlled image-classification surrogate on `sklearn` digits.
2. A randomness/alignment surrogate for independently initialized small models.
3. A Qwen/LLM-compatible weight-space probe and a real Qwen2.5-1.5B base-to-instruct path sweep.
4. Cached GSM8K, MMLU, HumanEval NLL, and safety/refusal benchmark slices for the same Qwen interpolation path.
5. A CIFAR100 coarse-label ViT-style patch-transformer merge landscape.
6. An ImageNet-pretrained ViT-B/16 frozen-backbone transfer merge.
7. A real Qwen2.5-0.5B multi-expert merge plane using instruct and coder experts.

## Quick Start

```bash
PYTHONPATH=src python scripts/run_digits_merge.py --output-dir results/digits_merge
```

The run writes:

- `results/digits_merge/grid_metrics.csv`
- `results/digits_merge/lambda_sweep.csv`
- `results/digits_merge/method_metrics.csv`
- `results/digits_merge/layerwise_task_arithmetic.csv`
- `results/digits_merge/regmean_linear_layers.csv`
- `results/digits_merge/regmean_covariances.csv`
- `results/digits_merge/interference.csv`
- `results/digits_merge/figures/*.png`
- `results/digits_merge/report.md`

The digits setup is deliberately small and deterministic. A shared MLP base is trained briefly on all digit classes, then two experts are fine-tuned strongly from the same base: one on digits `0-4`, the other on digits `5-9`. This creates a clean model-merging surrogate where the experts become class-subset specialists, task vectors have known semantics, and dense grid evaluation is cheap.

The checked-in run artifacts under `results/digits_merge` include a 41x41 grid, method metrics, interference metrics, and figures.

## Dashboard

Build the browsable dashboard from the completed artifacts:

```bash
PYTHONPATH=src python scripts/build_dashboard.py --output-dir results/dashboard
```

Open `results/dashboard/index.html` in a browser. It includes overview metrics, a draggable merge-plane explorer over precomputed grids, single-digit pairwise heatmaps, alignment curves, Qwen path sweeps, GSM8K/MMLU/HumanEval/safety slice metrics, and the Qwen multi-expert merge plane.

## Result Summary And Manifest

Regenerate the consolidated metric summary and artifact manifest:

```bash
PYTHONPATH=src python scripts/collect_results.py
```

This writes:

- `results/summary.json`
- `results/summary.md`
- `ARTIFACT_MANIFEST.json`

The summary includes a coverage audit for the original proposal.

## Randomness / Alignment Demo

The proposal also calls out randomness and alignment. This script trains two same-task MLPs from different random initializations, aligns hidden units with the Hungarian algorithm, and compares interpolation before/after alignment:

```bash
PYTHONPATH=src python scripts/run_alignment_barrier.py --output-dir results/alignment_barrier --device cpu
```

The generated result shows the midpoint accuracy improving from `0.944` before alignment to `0.971` after alignment, with the loss barrier dropping from `0.064` to `0.006`.

## Single-Digit Expert Pairwise Study

This is the one-class surrogate: ten experts are fine-tuned from the same base, one per digit class, then all 45 digit pairs are merged and analyzed:

```bash
PYTHONPATH=src python scripts/run_digit_pairwise_experts.py --output-dir results/digit_pairwise_experts --device cpu
```

The completed run is a useful negative result: linear averaging usually preserves both single-digit tasks, and global conflict metrics only weakly predict merge drop. The clearest failure is the `3`/`9` pair.

## CIFAR-10 Vehicle / Animal Merge

Run a natural-image class-group merge landscape:

```bash
PYTHONPATH=src python scripts/run_cifar_merge.py --output-dir results/cifar_merge
```

This trains a small GroupNorm CNN base and two same-base experts for CIFAR-10 vehicles vs animals. The completed run shows a clear naive-average failure: base worst accuracy `0.376`, linear average worst accuracy `0.249`, and validation grid best worst accuracy `0.426`.

## CIFAR100 ViT-Style Merge

Run the transformer-vision phase:

```bash
PYTHONPATH=src python scripts/run_cifar100_vit_merge.py --output-dir results/cifar100_vit_merge
```

This trains a small patch-transformer on CIFAR100 coarse labels, then fine-tunes two same-base experts for living superclasses vs object/vehicle superclasses. It is not a CLIP transfer run, but it exercises the ViT-style part of the proposal with the same merge-plane, method-overlay, lambda-sweep, PCA-geometry, and interference-atlas machinery. The completed run has base worst accuracy `0.189`, linear average worst accuracy `0.076`, and best task-arithmetic worst accuracy `0.197`.

## Pretrained ViT Transfer Merge

Run a pretrained vision-transfer merge with an ImageNet ViT-B/16 frozen feature extractor:

```bash
PYTHONPATH=src python scripts/run_pretrained_vit_transfer_merge.py --output-dir results/pretrained_vit_transfer_merge
```

This trains CIFAR100 coarse-label linear heads for a base model and living/object experts, then evaluates their head task-vector merge plane. It is not full-backbone fine-tuning, but it supplies the proposal's pretrained ViT transfer evidence. The completed run has base worst accuracy `0.740`, linear average worst accuracy `0.763`, and validation-grid best worst accuracy `0.783`.

## Qwen / LLM Probe

The LLM probe script checks same-shape checkpoint compatibility, computes task-vector magnitudes, and reports layer-wise pairwise conflict for compatible experts:

```bash
PYTHONPATH=src python scripts/probe_qwen_deltas.py \
  --base Qwen/Qwen2.5-1.5B \
  --expert instruct=Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir results/qwen_probe
```

For multiple experts, repeat `--expert NAME=MODEL_OR_PATH`. The output includes:

- `delta_summary.csv`
- `delta_summary_by_group.csv`
- `pairwise_conflict.csv`
- `pairwise_conflict_heatmap.png` when at least two experts are supplied

Use this as the first LLM-stage diagnostic before running expensive benchmark sweeps.

A smoke test was run with the same local Qwen3-0.6B safetensors file as both base and expert:

```bash
PYTHONPATH=src python scripts/probe_qwen_deltas.py \
  --base /srv/home/bohanlyu/qixin/MLS-Bench/vendor/data/models/Qwen3-0.6B/model.safetensors \
  --expert same=/srv/home/bohanlyu/qixin/MLS-Bench/vendor/data/models/Qwen3-0.6B/model.safetensors \
  --output-dir results/qwen_probe_smoke \
  --max-tensors 12
```

It produced zero deltas, which validates the loader and output schema.

## Qwen Path Sweep

For a real LLM path experiment, run:

```bash
PYTHONPATH=src python scripts/run_qwen_path_sweep.py --output-dir results/qwen_path_sweep --dtype bfloat16 --max-length 384
```

This uses local Qwen2.5-1.5B base and Qwen2.5-1.5B-Instruct checkpoints by default. It evaluates `base + lambda * (instruct - base)` on a small fixed general/instruction NLL slice.

The completed run found the best average NLL at `lambda=0.75`; instruction response NLL improved from `3.612` at the base model to `1.811` at `lambda=0.75`.

## Qwen GSM8K Benchmark Slice

Run a small cached GSM8K exact-match slice on the Qwen2.5-1.5B base-to-instruct interpolation path:

```bash
PYTHONPATH=src python scripts/run_qwen_gsm8k_slice.py --output-dir results/qwen_gsm8k_slice
```

The completed run evaluates `12` GSM8K test examples for `lambda=0.0,0.75,1.0`. Strict scoring requires the model to emit the GSM8K `#### <number>` format; loose scoring falls back to the last generated number. The best strict score is `1/12` at `lambda=0.75` and `lambda=1.0`; the best loose score is `3/12` at `lambda=0.75`.

## Qwen MMLU Benchmark Slice

Run a small MMLU multiple-choice log-likelihood slice on the same Qwen2.5-1.5B interpolation path:

```bash
PYTHONPATH=src python scripts/run_qwen_mmlu_slice.py --output-dir results/qwen_mmlu_slice
```

The completed run evaluates `24` MMLU test examples for `lambda=0.0,0.75,1.0` by scoring answer letters A-D. The best accuracy is `18/24 = 0.750` at `lambda=0.75`; the base endpoint scores `7/24`, and the instruct endpoint scores `16/24`.

## Qwen HumanEval NLL Slice

Run a small HumanEval code-completion NLL slice on the same Qwen2.5-1.5B interpolation path:

```bash
PYTHONPATH=src python scripts/run_qwen_humaneval_nll_slice.py --output-dir results/qwen_humaneval_nll_slice
```

The completed run evaluates `24` HumanEval tasks by scoring canonical solutions, without executing generated code. The best token-weighted solution NLL is `0.964` at `lambda=1.0`; `lambda=0.75` is close at `0.971`, and the base endpoint is `0.997`.

## Qwen Safety / Refusal Slice

Run a small BeaverTails safety/refusal NLL slice on the same Qwen2.5-1.5B interpolation path:

```bash
PYTHONPATH=src python scripts/run_qwen_safety_refusal_slice.py --output-dir results/qwen_safety_refusal_slice
```

The completed run scores `12` safe prompts against safe dataset responses and `12` unsafe prompts against a fixed refusal target. It stores prompt hashes, not raw prompt text. The best average safety NLL is `2.546` at `lambda=0.75`.

## Qwen Multi-Expert Merge

Run the real two-expert Qwen merge plane:

```bash
PYTHONPATH=src python scripts/run_qwen_multi_expert_merge.py --output-dir results/qwen_multi_expert_merge
```

This evaluates `base + alpha * instruct_delta + beta * coder_delta` for Qwen2.5-0.5B base, Qwen2.5-0.5B-Instruct, and Qwen2.5-Coder-0.5B-Instruct on small general, instruction-response, and code-response NLL slices. The completed run finds the best average NLL at the instruct endpoint (`3.009`), while naive linear averaging is worse (`5.591` average NLL, `9.553` worst NLL). The instruct/coder task-vector cosine is `0.140` with weighted conflict `0.386`.

## Research Scope Covered

The controlled experiment covers the MVP deliverables from the proposal:

- task-vector 2D merge landscape;
- per-task basin overlay;
- task-arithmetic lambda sweep;
- merge-method overlay for average, task arithmetic, SLERP, TIES, DARE, TIES+DARE, Fisher, RegMean, layer-wise task arithmetic, and validation grid search;
- layer-wise interference atlas;
- single-digit pairwise expert correlation study;
- CIFAR-10 vehicle/animal natural-image merge landscape;
- CIFAR100 coarse-label ViT-style merge landscape with PCA task-vector geometry;
- ImageNet-pretrained ViT-B/16 frozen-backbone transfer merge landscape;
- independent-initialization alignment barrier demo;
- Qwen2.5-1.5B base-to-instruct lambda sweep;
- Qwen2.5-1.5B GSM8K exact-match benchmark slice;
- Qwen2.5-1.5B MMLU multiple-choice benchmark slice;
- Qwen2.5-1.5B HumanEval canonical-solution NLL slice;
- Qwen2.5-1.5B BeaverTails safety/refusal NLL slice;
- Qwen2.5-0.5B instruct+coder multi-expert merge plane;
- interactive dashboard for browsing completed artifacts and dragging/querying precomputed merge-plane points;
- short written interpretation in `report.md`.
