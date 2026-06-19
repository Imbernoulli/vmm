# Visualizing Model Merging in Task-Vector Subspaces

## Abstract

This artifact studies model merging by visualizing the task-vector subspace where merges actually occur. Given a shared base model and task-specialized models, it evaluates points of the form `theta0 + alpha * tau_A + beta * tau_B`, overlays common merge methods, and measures layer-wise interference between task vectors.

The current evidence spans controlled sklearn digits models, a one-class expert surrogate, an independent-initialization alignment demo, a CIFAR-10 vehicle/animal natural-image merge, a CIFAR100 ViT-style patch-transformer merge, an ImageNet-pretrained ViT-B/16 frozen-backbone transfer merge, a Qwen2.5-1.5B base-to-instruct weight-space path sweep, cached GSM8K/MMLU/HumanEval/safety benchmark slices, and a Qwen2.5-0.5B instruct+coder multi-expert merge plane. The results support the main hypothesis in the proposal: merge success depends on landing in a region that is compatible with all tasks. They also show an important negative result: global sign-conflict metrics alone do not reliably predict failure in an easy one-class digit surrogate.

This is a runnable research artifact at proposal scale, not a leaderboard-scale final paper. The remaining work is scale-up and external benchmarking rather than missing proposal coverage.

## 1. Research Question

The proposal asks whether model merging can be explained more directly by visualizing task-vector geometry instead of random loss-landscape slices.

For a base model `theta0` and two fine-tuned experts `theta_A` and `theta_B`, define task vectors:

```text
tau_A = theta_A - theta0
tau_B = theta_B - theta0
```

The central visualization plane is:

```text
theta(alpha, beta) = theta0 + alpha * tau_A + beta * tau_B
```

The working hypothesis is that a good merged model lies in, or near, the intersection of the task-compatible low-loss regions in this plane.

## 2. Method

The implementation has three reusable pieces:

- Weight-space utilities in `src/mergeviz/weights.py` flatten model weights, reconstruct checkpoints from vectors, project arbitrary merge outputs into the two-task plane, and compute layer-wise slices.
- Merge-method utilities in `src/mergeviz/merge_methods.py` implement linear averaging, task arithmetic, SLERP, TIES-style masking, DARE, TIES+DARE, and Fisher-weighted averaging. The digits experiment additionally implements MLP linear-layer RegMean and validation-searched layer-wise task arithmetic diagnostics.
- Experiment scripts under `scripts/` train or load models, evaluate dense grids or path sweeps, and write CSV, JSON, PNG, and Markdown artifacts under `results/`.

The diagnostics are:

- dense 2D grid metrics for average and worst-task loss/accuracy;
- method overlays in the task-vector plane;
- task-arithmetic lambda sweeps;
- layer-wise cosine, sign conflict, and magnitude-weighted conflict;
- interpolation barriers before and after alignment when models do not share an initialization;
- sparse LLM path sweeps and small multi-expert LLM grids where dense evaluation is expensive.
- a dependency-free dashboard that supports draggable lookup over precomputed merge planes.

## 3. Experiments

### 3.1 Digits Low/High Class-Subset Merge

`scripts/run_digits_merge.py` trains a shared MLP base on sklearn digits, then fine-tunes two experts from that base:

- expert A on digits `0-4`;
- expert B on digits `5-9`.

The run evaluates a `41 x 41` task-vector grid and overlays merge methods.

Key results from `results/digits_merge`:

| metric | value |
| --- | ---: |
| grid points | 1681 |
| base worst accuracy | 0.917 |
| linear-average worst accuracy | 0.922 |
| layer-wise task arithmetic worst accuracy | 0.928 |
| RegMean linear-layer worst accuracy | 0.939 |
| max grid worst accuracy | 0.961 |
| fraction of grid with worst accuracy at least 0.90 | 0.134 |
| global task-vector cosine | 0.138 |

Interpretation: the two class-subset experts catastrophically forget the opposite subset, but the task-vector plane contains a narrow shared region. Linear averaging, SLERP, and Fisher-weighted averaging land in that region. RegMean and layer-wise task arithmetic improve worst-task accuracy further, but with high residuals after projection into the raw two-vector plane. This is a useful diagnostic case: the 2D plot still explains where ordinary task-vector methods land, while the residual exposes when a stronger method uses structure outside that plane. TIES/DARE variants can leave the dense shared region in this small surrogate, which is visible through their projected locations and residuals.

Main artifacts:

- `results/digits_merge/figures/merge_landscape.png`
- `results/digits_merge/figures/per_task_basin_overlay.png`
- `results/digits_merge/figures/lambda_sweep.png`
- `results/digits_merge/figures/method_overlay.png`
- `results/digits_merge/figures/interference_heatmap.png`

### 3.2 Single-Digit Expert Pairwise Study

`scripts/run_digit_pairwise_experts.py` tests the user-suggested one-class surrogate. It trains ten same-base experts, one per digit, then evaluates all `45` pairwise merges.

Key results from `results/digit_pairwise_experts`:

| metric | value |
| --- | ---: |
| pairwise merges | 45 |
| mean linear worst accuracy | 0.986 |
| worst pair | digits 3 and 9 |
| worst-pair linear worst accuracy | 0.861 |
| worst-pair drop from base | 0.111 |
| Spearman: cosine vs drop | -0.174 |
| Spearman: sign conflict vs drop | 0.185 |
| Spearman: weighted conflict vs drop | 0.165 |

Interpretation: this is a useful negative result. Most single-digit pairwise linear merges remain strong, and global conflict metrics only weakly correlate with merge drop. The hardest failure is the semantically confusable `3`/`9` pair. In this setting, sign conflict is not sufficient as a standalone predictor.

### 3.3 Independent-Initialization Alignment Barrier

`scripts/run_alignment_barrier.py` trains two one-hidden-layer MLPs on the same digits task from different random seeds. It aligns hidden units with Hungarian matching and compares the linear interpolation path before and after alignment.

Key results from `results/alignment_barrier`:

| metric | before alignment | after alignment |
| --- | ---: | ---: |
| midpoint accuracy | 0.944 | 0.971 |
| loss barrier | 0.064 | 0.006 |

Interpretation: some apparent weight-space barriers are coordinate or permutation artifacts. In this small model, a simple hidden-unit alignment almost removes the interpolation loss barrier.

### 3.4 CIFAR-10 Vehicle/Animal Natural-Image Merge

`scripts/run_cifar_merge.py` moves from digits to CIFAR-10. It trains a small GroupNorm CNN base and two same-base experts:

- vehicles: airplane, automobile, ship, truck;
- animals: bird, cat, deer, dog, frog, horse.

Key results from `results/cifar_merge`:

| metric | value |
| --- | ---: |
| grid points | 441 |
| base worst accuracy | 0.376 |
| linear-average worst accuracy | 0.249 |
| validation-grid best worst accuracy | 0.426 |
| validation best minus base | 0.050 |
| validation best minus linear average | 0.177 |
| fraction of grid with worst accuracy at least 0.40 | 0.016 |
| global task-vector cosine | 0.003 |

Interpretation: this is the clearest current example of naive average failure. The shared high-quality region is small, linear averaging lands outside the best region, and validation-searched coefficients improve the worst-task accuracy above both the base and the average.

### 3.5 CIFAR100 ViT-Style Transformer Merge

`scripts/run_cifar100_vit_merge.py` gives the project a transformer-vision experiment. It trains a small patch-transformer on CIFAR100 coarse labels, then fine-tunes two same-base experts:

- living superclasses;
- object/vehicle superclasses.

This is not a pretrained CLIP transfer run, but it covers the ViT-style architecture path and adds PCA task-vector geometry.

Key results from `results/cifar100_vit_merge`:

| metric | value |
| --- | ---: |
| grid points | 289 |
| base worst accuracy | 0.189 |
| linear-average worst accuracy | 0.076 |
| best method | task arithmetic best lambda |
| best method worst accuracy | 0.197 |
| max grid worst accuracy | 0.194 |
| fraction of grid with worst accuracy at least 0.15 | 0.038 |
| global task-vector cosine | -0.176 |

Interpretation: the transformer-vision result is qualitatively aligned with the CNN CIFAR result. Experts specialize and forget the opposite group; naive averaging damages one task; a small task-arithmetic step is the best tested merge. The absolute accuracy is modest because the model is small and trained from scratch, so this should be read as architectural coverage and a visualization diagnostic rather than as CLIP/ViT transfer evidence.

### 3.6 Pretrained ViT-B/16 Transfer Merge

`scripts/run_pretrained_vit_transfer_merge.py` adds a pretrained vision-transfer experiment. It uses torchvision's ImageNet-pretrained ViT-B/16 as a frozen feature extractor, then trains CIFAR100 coarse-label linear heads for a base model and two class-group experts.

This is not full-backbone fine-tuning, but it supplies a bounded pretrained ViT transfer case where the task vectors live in the classifier head.

Key results from `results/pretrained_vit_transfer_merge`:

| metric | value |
| --- | ---: |
| grid points | 441 |
| base worst accuracy | 0.740 |
| linear-average worst accuracy | 0.763 |
| best method | validation grid best |
| best method worst accuracy | 0.783 |
| global head task-vector cosine | -0.068 |

Interpretation: the pretrained ViT transfer run has much stronger absolute accuracy than the from-scratch CIFAR100 transformer while preserving the same geometric structure: experts specialize and damage the opposite group, the average improves over the base, and validation-searched coefficients improve the worst-task score further. This covers the proposal's pretrained ViT phase at frozen-backbone scale.

### 3.7 Qwen2.5-1.5B Base-to-Instruct Path

`scripts/run_qwen_path_sweep.py` evaluates:

```text
theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)
```

for local Qwen2.5-1.5B base and Qwen2.5-1.5B-Instruct checkpoints. Dense 2D LLM grids are too expensive for this stage, so the script uses a sparse path sweep on a fixed prompt slice:

- 4 general text examples evaluated with full-sequence NLL;
- 5 instruction examples evaluated with response-only NLL under the Qwen chat template.

Key results from `results/qwen_path_sweep`:

| lambda | general NLL | instruction NLL | average NLL |
| ---: | ---: | ---: | ---: |
| 0.00 | 4.783 | 3.612 | 4.197 |
| 0.50 | 4.756 | 1.851 | 3.303 |
| 0.75 | 4.756 | 1.811 | 3.283 |
| 1.00 | 4.746 | 1.874 | 3.310 |

The best average NLL occurs at `lambda=0.75`. Instruction response NLL improves from `3.612` at the base endpoint to `1.811` at `lambda=0.75`. General text NLL remains nearly flat and slightly improves on this tiny slice. The largest grouped delta norm is in `model.embed_tokens`, followed by late transformer blocks around layers 20-26.

Interpretation: this is a real LLM weight-space path result, but it should be treated as a diagnostic smoke-scale experiment. It does not replace formal LLM benchmarks.

### 3.8 Qwen2.5-1.5B GSM8K Benchmark Slice

`scripts/run_qwen_gsm8k_slice.py` evaluates the same base-to-instruct interpolation path on `12` cached GSM8K test examples. The script generates answers, extracts final numeric answers, and reports two scores:

- strict exact match, requiring the model to emit the GSM8K `#### <number>` answer format;
- loose exact match, falling back to the last generated number when the marker is absent.

Key results from `results/qwen_gsm8k_slice`:

| lambda | strict exact | loose exact | hash format rate |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.000 | 0.083 | 0.000 |
| 0.75 | 0.083 | 0.250 | 0.083 |
| 1.00 | 0.083 | 0.167 | 0.083 |

Interpretation: the benchmark slice is small, but it is no longer just a hand-written prompt probe. `lambda=0.75` ties the instruct endpoint on strict exact match and has the best loose exact match. The strict scores remain low, mostly because the models rarely emit the requested `####` format in this setup, so the result should be read as a diagnostic benchmark slice rather than a full GSM8K claim.

### 3.9 Qwen2.5-1.5B MMLU Benchmark Slice

`scripts/run_qwen_mmlu_slice.py` evaluates the same base-to-instruct path on `24` MMLU test questions. Each answer letter A-D is scored by log-likelihood, and the lowest-NLL answer is selected.

Key results from `results/qwen_mmlu_slice`:

| lambda | accuracy | correct / total | avg gold NLL |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.292 | 7/24 | 6.048 |
| 0.75 | 0.750 | 18/24 | 0.932 |
| 1.00 | 0.667 | 16/24 | 1.282 |

Interpretation: this broadens the LLM benchmark evidence beyond GSM8K generation. The intermediate `lambda=0.75` is strongest on this small MMLU slice, improving over both endpoints by accuracy and gold-answer NLL. The sample is still too small for a broad MMLU claim.

### 3.10 Qwen2.5-1.5B HumanEval NLL Slice

`scripts/run_qwen_humaneval_nll_slice.py` evaluates the same base-to-instruct path on `24` HumanEval tasks. It scores the canonical solutions by token-level NLL and does not execute generated code or report pass@k.

Key results from `results/qwen_humaneval_nll_slice`:

| lambda | avg solution NLL | mean task NLL | median task NLL |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.997 | 1.368 | 1.197 |
| 0.75 | 0.971 | 1.318 | 1.168 |
| 1.00 | 0.964 | 1.299 | 1.169 |

Interpretation: unlike the MMLU slice, this code-likelihood slice improves monotonically toward the instruct endpoint on the sampled tasks. It provides code-domain evidence without the safety and reproducibility issues of executing generated code, but it is not a pass@k result.

### 3.11 Qwen2.5-1.5B Safety / Refusal Slice

`scripts/run_qwen_safety_refusal_slice.py` evaluates the same base-to-instruct path on a small BeaverTails safety/refusal slice. It scores safe prompts against safe dataset responses and unsafe prompts against a fixed refusal target. It stores prompt hashes rather than raw prompt text.

Key results from `results/qwen_safety_refusal_slice`:

| lambda | safe response NLL | unsafe refusal NLL | avg safety NLL |
| ---: | ---: | ---: | ---: |
| 0.00 | 1.977 | 3.661 | 2.819 |
| 0.75 | 1.783 | 3.310 | 2.546 |
| 1.00 | 1.787 | 3.780 | 2.783 |

Interpretation: `lambda=0.75` is best on this safety/refusal slice, improving both the safe-response and unsafe-refusal likelihoods relative to the base and instruct endpoint. This supplies a small safety diagnostic without generating unsafe completions.

### 3.12 Qwen2.5-0.5B Instruct+Coder Multi-Expert Merge

`scripts/run_qwen_multi_expert_merge.py` evaluates a real two-expert Qwen plane:

```text
theta(alpha, beta) = theta_base + alpha * tau_instruct + beta * tau_coder
```

The base is local Qwen2.5-0.5B. The experts are Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct. The run evaluates small fixed general, instruction-response, and code-response NLL slices.

Key results from `results/qwen_multi_expert_merge`:

| method | alpha | beta | general NLL | instruction NLL | code NLL | avg NLL | worst NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| instruct expert | 1.00 | 0.00 | 7.541 | 1.038 | 0.447 | 3.009 | 7.541 |
| coder expert | 0.00 | 1.00 | 7.587 | 0.912 | 0.574 | 3.024 | 7.587 |
| task arithmetic 0.75 | 0.75 | 0.75 | 7.962 | 1.234 | 0.426 | 3.207 | 7.962 |
| base | 0.00 | 0.00 | 7.844 | 2.973 | 1.543 | 4.120 | 7.844 |
| linear average | 0.50 | 0.50 | 9.553 | 4.610 | 2.611 | 5.591 | 9.553 |

Pairwise expert conflict:

| metric | value |
| --- | ---: |
| shared tensors | 290 |
| instruct/coder cosine | 0.140 |
| sign conflict | 0.454 |
| weighted conflict | 0.386 |

Interpretation: this completes a diagnostic multi-expert LLM merge. The best point on this small slice is the instruct endpoint, with the coder endpoint close behind; naive linear averaging is much worse. The result proves that the pipeline can load compatible official Qwen experts, evaluate a two-expert plane, and measure expert conflict, but it does not yet show an interior multi-expert point that dominates both endpoints.

## 4. Main Claims Supported So Far

First, the task-vector plane is an informative visualization object. In both digits and CIFAR, the merge plane shows where base, experts, average, task arithmetic, and masked methods land relative to task-compatible regions.

Second, basin intersection is a useful explanatory frame. The CIFAR run shows that naive averaging can fall outside the best shared region, while a validation-searched point improves worst-task accuracy.

Third, interference is local and method-dependent. The layer-wise conflict tables identify tensors with high weighted conflict, but the single-digit pairwise study shows that global conflict metrics are not enough to predict failures by themselves. RegMean also shows that activation-aware linear-layer merging can improve a small MLP merge even when its resulting point has high residual outside the raw task-vector plane.

Fourth, alignment matters when the models do not share a base initialization. The alignment experiment reduces the loss barrier from `0.064` to `0.006`, supporting the proposal's caution about permutation artifacts.

Fifth, the ViT experiments show the visualization machinery is not limited to MLPs and CNNs. The project includes both a from-scratch CIFAR100 patch-transformer and an ImageNet-pretrained ViT-B/16 frozen-backbone transfer run.

Sixth, the LLM stage is feasible as a path-sweep diagnostic, small benchmark slices, and a real multi-expert merge plane. The Qwen path sweep and multi-expert run load real checkpoints and evaluate interpolated weights; GSM8K, MMLU, HumanEval, and BeaverTails safety add real benchmark examples. These remain preliminary because the evaluation sets are small.

## 5. Limitations

The current study is complete at the proposal-artifact scale, with these limitations:

- The pretrained ViT transfer run freezes the backbone and merges linear heads; full-backbone CLIP/ViT fine-tuning remains a stronger but more expensive extension.
- RegMean is currently implemented as an MLP linear-layer diagnostic in the digits experiment, not as a general implementation for CNN, ViT, or LLM layers.
- The LLM benchmark slices are intentionally small and diagnostic; they do not replace leaderboard-grade GSM8K/MMLU, HumanEval pass@k/MBPP execution, MT-Bench/AlpacaEval preference evaluation, or full safety/toxicity evaluation.
- The dashboard supports dragged precomputed merge-plane lookup, but it cannot run new checkpoint evaluations in the browser.
- The Qwen path and multi-expert results use tiny fixed prompt slices, so they should not be interpreted as broad capability or safety evaluations.

These limitations are tracked in `results/summary.json` and `results/summary.md`.

## 6. Reproducibility

Primary reproduction commands:

```bash
PYTHONPATH=src python scripts/run_digits_merge.py --output-dir results/digits_merge --device cpu
PYTHONPATH=src python scripts/run_digit_pairwise_experts.py --output-dir results/digit_pairwise_experts --device cpu
PYTHONPATH=src python scripts/run_alignment_barrier.py --output-dir results/alignment_barrier --device cpu
PYTHONPATH=src python scripts/run_cifar_merge.py --output-dir results/cifar_merge
PYTHONPATH=src python scripts/run_cifar100_vit_merge.py --output-dir results/cifar100_vit_merge
PYTHONPATH=src python scripts/run_pretrained_vit_transfer_merge.py --output-dir results/pretrained_vit_transfer_merge
PYTHONPATH=src python scripts/run_qwen_path_sweep.py --output-dir results/qwen_path_sweep --dtype bfloat16 --max-length 384
PYTHONPATH=src python scripts/run_qwen_gsm8k_slice.py --output-dir results/qwen_gsm8k_slice
PYTHONPATH=src python scripts/run_qwen_mmlu_slice.py --output-dir results/qwen_mmlu_slice
PYTHONPATH=src python scripts/run_qwen_humaneval_nll_slice.py --output-dir results/qwen_humaneval_nll_slice
PYTHONPATH=src python scripts/run_qwen_safety_refusal_slice.py --output-dir results/qwen_safety_refusal_slice
PYTHONPATH=src python scripts/run_qwen_multi_expert_merge.py --output-dir results/qwen_multi_expert_merge
PYTHONPATH=src python scripts/build_dashboard.py --output-dir results/dashboard
PYTHONPATH=src python scripts/collect_results.py
```

Summary and manifest artifacts:

- `results/summary.json`
- `results/summary.md`
- `ARTIFACT_MANIFEST.json`
- `results/dashboard/index.html`
