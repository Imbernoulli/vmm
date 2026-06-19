# Visualizing Model Merging: Research Report

This repository now contains a runnable research artifact for the proposal in `proposal.md`.

The implemented study starts with controlled image-classification surrogates, because dense 2D loss-surface evaluation is cheap and the geometry is easy to audit. It then extends the same task-vector/path-sweep framing to a pretrained ViT transfer setting and to real Qwen checkpoints, including a Qwen2.5-1.5B base-to-instruct path, cached GSM8K/MMLU/HumanEval/safety slices, and a Qwen2.5-0.5B instruct+coder multi-expert merge plane.

## Implemented Artifacts

- `scripts/run_digits_merge.py`: trains a shared base MLP and two same-base task experts, evaluates a 2D task-vector plane, overlays merge methods, includes MLP linear-layer RegMean and layer-wise task-arithmetic diagnostics, and writes plots/tables.
- `src/mergeviz/weights.py`: state-dict vectorization, vector-to-model loading, layer slices, plane projection, and geometry helpers.
- `src/mergeviz/merge_methods.py`: linear averaging, task arithmetic, SLERP, TIES-style masking, DARE, TIES+DARE, and Fisher-weighted averaging.
- `scripts/probe_qwen_deltas.py`: streaming safetensors loader and Qwen/same-architecture checkpoint task-vector diagnostics.
- `scripts/run_alignment_barrier.py`: independent-initialization MLP interpolation before/after hidden-unit permutation alignment.
- `scripts/run_digit_pairwise_experts.py`: ten single-digit experts and all 45 pairwise merges for conflict-vs-drop analysis.
- `scripts/run_cifar_merge.py`: CIFAR-10 vehicle/animal natural-image class-group merge landscape.
- `scripts/run_cifar100_vit_merge.py`: CIFAR100 coarse-label ViT-style patch-transformer merge landscape with PCA task-vector geometry.
- `scripts/run_pretrained_vit_transfer_merge.py`: ImageNet-pretrained ViT-B/16 frozen-backbone CIFAR100 coarse-label transfer merge.
- `scripts/run_qwen_path_sweep.py`: real Qwen2.5-1.5B base-to-instruct weight-space path sweep on fixed general/instruction prompt slices.
- `scripts/run_qwen_gsm8k_slice.py`: cached GSM8K exact-match benchmark slice for the Qwen2.5-1.5B base-to-instruct interpolation path.
- `scripts/run_qwen_mmlu_slice.py`: MMLU multiple-choice log-likelihood benchmark slice for the Qwen2.5-1.5B base-to-instruct interpolation path.
- `scripts/run_qwen_humaneval_nll_slice.py`: HumanEval canonical-solution NLL slice for the Qwen2.5-1.5B base-to-instruct interpolation path.
- `scripts/run_qwen_safety_refusal_slice.py`: BeaverTails safe-response and unsafe-refusal NLL slice for the Qwen2.5-1.5B base-to-instruct interpolation path.
- `scripts/run_qwen_multi_expert_merge.py`: real Qwen2.5-0.5B two-expert merge plane using Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct.
- `scripts/build_dashboard.py`: dependency-free HTML dashboard generator with a draggable precomputed merge-plane explorer, figures, tables, and path sweeps.
- `scripts/collect_results.py`: consolidated result collector that recomputes key metrics from CSV/JSON artifacts and writes a proposal-coverage audit.
- `PAPER.md`: paper-style writeup of the completed experiments, claims, negative results, and remaining limitations.
- `results/digits_merge/`: completed 41x41 grid run with figures, CSVs, checkpoints, RegMean diagnostics, layer-wise task-arithmetic diagnostics, and an auto-generated run report.
- `results/alignment_barrier/`: completed randomness/alignment surrogate with path metrics and an interpolation figure.
- `results/digit_pairwise_experts/`: completed one-class/single-digit expert study with pairwise merge matrices and conflict correlations.
- `results/cifar_merge/`: completed CIFAR-10 vehicle/animal merge landscape with method and interference plots.
- `results/cifar100_vit_merge/`: completed CIFAR100 ViT-style transformer merge landscape with method, lambda, conflict, and PCA plots.
- `results/pretrained_vit_transfer_merge/`: completed pretrained ViT-B/16 frozen-backbone transfer merge with grid, method, lambda, and head-conflict plots.
- `results/qwen_path_sweep/`: completed Qwen2.5-1.5B lambda sweep with NLL metrics and delta-norm plots.
- `results/qwen_gsm8k_slice/`: completed small GSM8K exact-match benchmark slice for Qwen lambdas `0.0`, `0.75`, and `1.0`.
- `results/qwen_mmlu_slice/`: completed small MMLU multiple-choice benchmark slice for Qwen lambdas `0.0`, `0.75`, and `1.0`.
- `results/qwen_humaneval_nll_slice/`: completed small HumanEval canonical-solution NLL slice for Qwen lambdas `0.0`, `0.75`, and `1.0`.
- `results/qwen_safety_refusal_slice/`: completed small BeaverTails safety/refusal NLL slice for Qwen lambdas `0.0`, `0.75`, and `1.0`.
- `results/qwen_multi_expert_merge/`: completed Qwen2.5-0.5B instruct+coder multi-expert merge plane with grid metrics, method metrics, conflict metrics, and figures.
- `results/qwen_probe_smoke/`: smoke test proving the Qwen probe can stream safetensors and emits zero deltas when base and expert are the same file.
- `results/dashboard/index.html`: generated interactive explainer dashboard with tabs for the merge plane, pairwise experts, alignment, Qwen path sweep, GSM8K slice, and Qwen multi-expert merge.
- `results/summary.json` and `results/summary.md`: consolidated metrics and coverage status for the proposal.
- `ARTIFACT_MANIFEST.json`: file-level artifact index with sizes and hashes for small files.

## Dashboard

The proposal's full demo asks for an explainer interface. The current implementation includes a dependency-free dashboard:

```bash
PYTHONPATH=src python scripts/build_dashboard.py --output-dir results/dashboard
```

Open [results/dashboard/index.html](results/dashboard/index.html) in a browser. The dashboard embeds CSV data as JSON, copies the generated figures into `results/dashboard/assets/`, and provides:

- overview metrics and an evidence map;
- draggable merge-plane lookup over the precomputed digits, CIFAR, ViT-style, pretrained ViT transfer, and Qwen multi-expert grids;
- task-pair, merge-method, objective, raw/normalized plane, alpha/beta, and lambda controls;
- merge-method table and merge landscape figures;
- interactive digit-pair heatmap metric switching;
- alignment path chart;
- Qwen lambda path chart, GSM8K/MMLU/HumanEval/safety slice metrics, and multi-expert method/conflict tables.

The dashboard does not run new model evaluations in the browser; dragged points query the nearest already-evaluated grid checkpoint.

## Main Small-Model Experiment

The task-vector plane is:

```text
theta(alpha, beta) = theta0 + alpha * tau_A + beta * tau_B
```

where:

- `theta0` is a shared MLP base trained briefly on all sklearn digits.
- expert A is fine-tuned from `theta0` on digits `0-4`.
- expert B is fine-tuned from `theta0` on digits `5-9`.
- task A evaluation uses only digits `0-4`; task B evaluation uses only digits `5-9`.

This is the surrogate requested in spirit by the user: one model becomes a class-subset specialist, and merging tests whether two specialists still share a usable basin.

## Key Results

From `results/digits_merge/method_metrics.csv`:

| method | task A acc | task B acc | worst acc | note |
| --- | ---: | ---: | ---: | --- |
| expert A | 0.983 | 0.000 | 0.000 | strong low-digit specialist, catastrophic forgetting on high digits |
| expert B | 0.000 | 0.978 | 0.000 | strong high-digit specialist, catastrophic forgetting on low digits |
| base | 0.961 | 0.917 | 0.917 | generic but weaker on task B |
| RegMean linear layers | 0.961 | 0.939 | 0.939 | best tested merge row; solves linear-layer normal equations from task activations |
| layer-wise task arithmetic | 0.950 | 0.928 | 0.928 | validation-searched tensor-wise scales for the combined task vector |
| linear average | 0.944 | 0.922 | 0.922 | lands in the shared basin |
| SLERP | 0.944 | 0.922 | 0.922 | similar to average in this plane |
| Fisher weighted | 0.944 | 0.922 | 0.922 | falls back to average where Fisher is uninformative |
| DARE | 0.911 | 0.922 | 0.911 | random masking leaves the basin edge |
| TIES | 0.817 | 0.922 | 0.817 | aggressive sign election hurts task A here |
| TIES+DARE | 0.533 | 0.839 | 0.533 | strongest interference/masking damage |

From `results/digits_merge/grid_metrics.csv`:

- Grid size: `41 x 41 = 1681` evaluated checkpoints.
- Fraction of sampled plane with worst-task accuracy at least `0.90`: `0.134`.
- Best validation-plane worst-task accuracy: `0.961`.

From `results/digits_merge/interference.csv`:

- Global task-vector cosine: `0.138`.
- Most conflicted tensor: `net.6.bias`, cosine `-0.801`, sign conflict `1.000`, weighted conflict `1.000`.
- Large hidden tensors also show near-zero or negative cosine and roughly half sign conflict, especially `net.3.weight` and `net.0.weight`.

## Figures

- [Merge landscape](results/digits_merge/figures/merge_landscape.png)
- [Per-task basin overlay](results/digits_merge/figures/per_task_basin_overlay.png)
- [Lambda sweep](results/digits_merge/figures/lambda_sweep.png)
- [Method overlay](results/digits_merge/figures/method_overlay.png)
- [Interference heatmap](results/digits_merge/figures/interference_heatmap.png)

## Interpretation

The run supports the proposal's central framing: task-vector geometry is more informative than a random 2D loss slice.

The two experts sit on the coordinate axes and are individually good only for their own task. The usable merged region is a narrow diagonal band, not the whole plane. Linear average, SLERP, and Fisher-weighted averaging land in that band, so their worst-task accuracy stays above the base. RegMean and layer-wise task arithmetic improve the test worst-task accuracy further, but both have high plane residuals because they modify layers or linear equations in ways not captured by the raw two-vector plane. That is useful evidence for the proposal's caveat: if a method works well while projecting outside the 2D plane, the visualization should report the residual instead of pretending the plane explains everything. TIES and DARE project away from the dense shared region in this small surrogate; their plane residuals are high and their accuracies drop.

The interference atlas explains why the plane is narrow. Several tensors have low cosine alignment and high sign conflict, meaning the two fine-tuning runs push the same parameters in incompatible directions. The output bias is the clearest case because the two class subsets want opposite logit adjustments.

## Single-Digit Expert Pairwise Study

The user suggested a surrogate where one model is trained for one class. `scripts/run_digit_pairwise_experts.py` implements that idea with ten same-base sklearn digit experts:

- each expert is a full 10-way MLP fine-tuned only on one digit's examples;
- all `45` digit pairs are merged and evaluated on the two relevant single-digit tasks;
- global and layer-wise cosine/sign-conflict metrics are computed for every pair.

This directly tests RQ2: whether task-vector conflict predicts merge failure.

From `results/digit_pairwise_experts/report.md`:

- Spearman correlation between linear merge drop and cosine: `-0.174`.
- Spearman correlation between linear merge drop and sign conflict: `0.185`.
- Spearman correlation between linear merge drop and weighted conflict: `0.165`.
- Worst linear merge: digits `3`/`9`, worst accuracy `0.861`, drop from base `0.111`.
- Most pairwise linear merges remain strong; mean linear worst accuracy is `0.986`.

Figures:

- [Single-digit pairwise heatmaps](results/digit_pairwise_experts/pairwise_heatmaps.png)
- [Conflict vs drop scatter](results/digit_pairwise_experts/conflict_vs_drop.png)
- [Single-digit layer conflict atlas](results/digit_pairwise_experts/layer_conflict_atlas.png)

Interpretation:

This is a useful negative result. The single-digit experts have substantial sign conflict, but the conflict range is narrow and most linear averages still preserve both single-digit tasks. In this surrogate, global sign conflict is not enough to predict merge failure; task semantic similarity and class confusion, especially `3`/`9`, matter more. This raises the bar for future experiments: conflict metrics need harder tasks, broader task diversity, or more localized layer diagnostics to become strongly predictive.

## CIFAR-10 Vehicle/Animal Merge Study

`scripts/run_cifar_merge.py` moves from sklearn digits to a natural-image classification task. A small GroupNorm CNN base is trained on a balanced CIFAR-10 subset, then two same-base experts are fine-tuned on:

- vehicles: airplane, automobile, ship, truck;
- animals: bird, cat, deer, dog, frog, horse.

This setting is harder than digits and produces a clearer naive-merge failure.

From `results/cifar_merge/report.md`:

- Best method by worst-task accuracy: `validation_grid_best`, vehicle accuracy `0.433`, animal accuracy `0.426`, worst accuracy `0.426`.
- Base worst-task accuracy: `0.376`.
- Linear average worst-task accuracy: `0.249`, below the base.
- Best task-arithmetic lambda: `0.112`, worst accuracy `0.381`.
- Fraction of sampled plane with worst-task accuracy at least `0.40`: `0.016`.
- Global vehicle/animal task-vector cosine: `0.003`.

Figures:

- [CIFAR merge landscape](results/cifar_merge/figures/merge_landscape.png)
- [CIFAR method overlay](results/cifar_merge/figures/method_overlay.png)
- [CIFAR lambda sweep](results/cifar_merge/figures/lambda_sweep.png)
- [CIFAR interference heatmap](results/cifar_merge/figures/interference_heatmap.png)

Interpretation:

Unlike the single-digit surrogate, CIFAR-10 vehicle/animal merging has a small shared region. Naive linear averaging lands outside the best region and hurts the animal task, while a validation-searched coefficient near `(alpha=0.125, beta=0.275)` improves worst-task accuracy above both the base and linear average. This is the clearest current support for the proposal's claim that merge success depends on landing in the basin intersection rather than simply averaging expert endpoints.

## Randomness / Alignment Experiment

The proposal also asks whether apparent barriers can come from model randomness rather than real task incompatibility. `scripts/run_alignment_barrier.py` tests this with two one-hidden-layer MLPs trained on the same sklearn digits task from different random initializations.

The second model is aligned to the first by matching hidden-unit feature vectors with the Hungarian algorithm, then the linear path between the two models is evaluated before and after alignment.

From `results/alignment_barrier/summary.json`:

- Model A accuracy: `0.967`.
- Model B accuracy: `0.976`.
- Midpoint accuracy before alignment: `0.944`.
- Midpoint accuracy after alignment: `0.971`.
- Loss barrier before alignment: `0.064`.
- Loss barrier after alignment: `0.006`.

Figure: [Alignment barrier](results/alignment_barrier/interpolation_alignment.png)

This supports the proposal's caution that some weight-space barriers are coordinate/permutation artifacts. In this small model, a simple hidden-unit alignment substantially reduces the interpolation loss barrier.

## CIFAR100 ViT-Style Transformer Study

`scripts/run_cifar100_vit_merge.py` addresses the proposal's CLIP/ViT phase with a lightweight patch-transformer trained on CIFAR100 coarse labels. It is not a pretrained CLIP transfer run, but it uses a ViT-style architecture and produces the same task-vector visualizations as the small-model experiments.

Tasks:

- living superclasses: aquatic mammals, fish, flowers, insects, carnivores, herbivores, mammals, invertebrates, people, reptiles, small mammals, trees;
- object/vehicle superclasses: food containers, fruit/vegetables, household electrical devices, furniture, outdoor things/scenes, vehicles.

From `results/cifar100_vit_merge/method_metrics.csv`:

| method | living acc | object acc | worst acc | note |
| --- | ---: | ---: | ---: | --- |
| base | 0.202 | 0.189 | 0.189 | generic coarse-label transformer baseline |
| expert living | 0.318 | 0.000 | 0.000 | living specialist, forgets object task |
| expert object | 0.000 | 0.370 | 0.000 | object specialist, forgets living task |
| linear average | 0.076 | 0.302 | 0.076 | naive average strongly hurts living task |
| task arithmetic best lambda | 0.199 | 0.197 | 0.197 | best tested merge row |
| validation grid best | 0.200 | 0.194 | 0.194 | close to best lambda |

Additional metrics:

- Grid size: `17 x 17 = 289` evaluated checkpoints.
- Global task-vector cosine: `-0.176`.
- Fraction of sampled plane with worst-task accuracy at least `0.15`: `0.038`.

Figures:

- [CIFAR100 ViT merge landscape](results/cifar100_vit_merge/figures/merge_landscape.png)
- [CIFAR100 ViT method overlay](results/cifar100_vit_merge/figures/method_overlay.png)
- [CIFAR100 ViT lambda sweep](results/cifar100_vit_merge/figures/lambda_sweep.png)
- [CIFAR100 ViT interference atlas](results/cifar100_vit_merge/figures/interference_heatmap.png)
- [CIFAR100 ViT task-vector PCA](results/cifar100_vit_merge/figures/pca_task_vectors.png)

Interpretation:

The ViT-style experiment reproduces the same qualitative geometry on a transformer vision model: the two experts specialize and forget the opposite task, naive averaging is poor, and only a small task-arithmetic step preserves both groups. The absolute accuracy is modest because this is a small from-scratch transformer on a limited CIFAR100 subset; the value is architectural coverage and visual diagnostics rather than transfer-learning performance.

## Pretrained ViT Transfer Study

`scripts/run_pretrained_vit_transfer_merge.py` adds the pretrained vision phase. It uses torchvision's ImageNet-pretrained ViT-B/16 as a frozen feature extractor, then trains CIFAR100 coarse-label linear heads for:

- a base head over all coarse labels;
- a living-superclass expert head;
- an object/vehicle-superclass expert head.

This is not full-backbone fine-tuning, but it is a real pretrained ViT transfer setting with task vectors in the classifier head.

From `results/pretrained_vit_transfer_merge/method_metrics.csv`:

| method | living acc | object acc | worst acc | note |
| --- | ---: | ---: | ---: | --- |
| base | 0.740 | 0.830 | 0.740 | frozen-backbone generic transfer head |
| expert living | 0.783 | 0.400 | 0.400 | living specialist, hurts object task |
| expert object | 0.402 | 0.877 | 0.402 | object specialist, hurts living task |
| linear average | 0.763 | 0.853 | 0.763 | improves over base worst accuracy |
| task arithmetic best lambda | 0.772 | 0.848 | 0.772 | best one-dimensional task-vector path |
| validation grid best | 0.783 | 0.800 | 0.783 | best tested merge row |

Additional metrics:

- Grid size: `21 x 21 = 441` evaluated head checkpoints.
- Global head task-vector cosine: `-0.068`.
- Best method improves worst-task accuracy by `0.043` over the base and `0.020` over linear averaging.

Figures:

- [Pretrained ViT transfer landscape](results/pretrained_vit_transfer_merge/figures/merge_landscape.png)
- [Pretrained ViT transfer method overlay](results/pretrained_vit_transfer_merge/figures/method_overlay.png)
- [Pretrained ViT transfer lambda sweep](results/pretrained_vit_transfer_merge/figures/lambda_sweep.png)
- [Pretrained ViT head conflict](results/pretrained_vit_transfer_merge/figures/interference_heatmap.png)

Interpretation:

The pretrained ViT transfer run supplies pretrained-vision evidence at a bounded cost. The frozen backbone gives substantially stronger absolute accuracy than the from-scratch CIFAR100 transformer, while the head task vectors still show specialization, partial forgetting, and a useful basin intersection where validation-searched coefficients beat both the base and naive average.

## Qwen / LLM Phase

The LLM part now has five levels: a general Qwen-compatible weight-space probe, a real path sweep from local Qwen2.5-1.5B base to local Qwen2.5-1.5B-Instruct, cached GSM8K, MMLU, HumanEval, and safety/refusal benchmark slices on that path, and a real two-expert Qwen2.5-0.5B merge plane for instruct and coder experts.

### Qwen2.5-1.5B Path Sweep

`scripts/run_qwen_path_sweep.py` evaluates:

```text
theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)
```

on a small fixed prompt slice:

- `4` general text examples, evaluated with full-sequence NLL;
- `5` instruction examples, evaluated with response-only NLL under the Qwen chat template.

This is not a full benchmark, but it is a real LLM weight-space path evaluation.

From `results/qwen_path_sweep/path_metrics.csv`:

| lambda | general NLL | instruction NLL | average NLL | worst NLL |
| ---: | ---: | ---: | ---: | ---: |
| -0.25 | 4.811 | 4.938 | 4.874 | 4.938 |
| 0.00 | 4.783 | 3.612 | 4.197 | 4.783 |
| 0.25 | 4.772 | 2.346 | 3.559 | 4.772 |
| 0.50 | 4.756 | 1.851 | 3.303 | 4.756 |
| 0.75 | 4.756 | 1.811 | 3.283 | 4.756 |
| 1.00 | 4.746 | 1.874 | 3.310 | 4.746 |
| 1.25 | 4.741 | 1.952 | 3.346 | 4.741 |

Main observations:

- Instruction response NLL improves sharply as the instruction delta is added, from `3.612` at the base model to a best value of `1.811` near `lambda=0.75`.
- General text NLL stays nearly flat and slightly improves on this tiny slice, from `4.783` at `lambda=0` to `4.746` at `lambda=1`.
- Best average NLL occurs at `lambda=0.75`, slightly before the official instruct endpoint.
- Best worst-task NLL occurs at `lambda=1.25`, driven by the general-text NLL being the worst term and decreasing slightly across this sampled path.

Figures:

- [Qwen path sweep](results/qwen_path_sweep/qwen_path_sweep.png)
- [Qwen delta norms](results/qwen_path_sweep/delta_norms.png)

The largest base-to-instruct changes by grouped delta norm are `model.embed_tokens`, followed by late transformer blocks around layers `20-26`. This is a magnitude diagnostic only; it does not prove those layers are causally responsible for instruction following.

### Qwen GSM8K Benchmark Slice

`scripts/run_qwen_gsm8k_slice.py` adds a cached GSM8K exact-match slice. It evaluates the same Qwen2.5-1.5B base-to-instruct path at `lambda=0.0`, `0.75`, and `1.0` on `12` GSM8K test questions.

The scorer reports two metrics:

- strict exact match: the model must emit the GSM8K `#### <number>` format and the extracted number must match;
- loose exact match: if the marker is missing, the scorer falls back to the last generated number.

From `results/qwen_gsm8k_slice/metrics.csv`:

| lambda | strict exact | loose exact | hash format rate |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.000 | 0.083 | 0.000 |
| 0.75 | 0.083 | 0.250 | 0.083 |
| 1.00 | 0.083 | 0.167 | 0.083 |

Figure: [Qwen GSM8K exact-match slice](results/qwen_gsm8k_slice/gsm8k_exact_match.png)

Interpretation:

This is still a small benchmark slice, but it is a real cached GSM8K evaluation rather than a hand-written prompt set. The intermediate `lambda=0.75` again looks competitive: it ties the instruct endpoint on strict exact match and has the best loose exact match. The low hash-format rate also shows that prompt compliance is a major failure mode for this tiny generation setup, so the strict metric is the safer number to cite.

### Qwen MMLU Benchmark Slice

`scripts/run_qwen_mmlu_slice.py` adds a multiple-choice MMLU slice. It evaluates the same Qwen2.5-1.5B base-to-instruct path at `lambda=0.0`, `0.75`, and `1.0` on `24` MMLU test questions.

The scorer computes log-likelihood for answer letters A-D and selects the lowest-NLL letter.

From `results/qwen_mmlu_slice/metrics.csv`:

| lambda | accuracy | correct / total | avg gold NLL | avg predicted NLL | avg margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.292 | 7/24 | 6.048 | 3.975 | 2.628 |
| 0.75 | 0.750 | 18/24 | 0.932 | 0.292 | 3.073 |
| 1.00 | 0.667 | 16/24 | 1.282 | 0.204 | 4.443 |

Figure: [Qwen MMLU multiple-choice slice](results/qwen_mmlu_slice/mmlu_accuracy.png)

Interpretation:

This is still a small benchmark slice, but it broadens the LLM evidence beyond GSM8K generation. The intermediate `lambda=0.75` is again strongest on the sampled path: it improves over both the base and instruct endpoints by accuracy and average gold-answer NLL.

### Qwen HumanEval NLL Slice

`scripts/run_qwen_humaneval_nll_slice.py` adds a code-completion likelihood slice. It evaluates the same Qwen2.5-1.5B path at `lambda=0.0`, `0.75`, and `1.0` on `24` HumanEval tasks.

The scorer computes token-level NLL for the canonical solutions. It does not execute generated code or report pass@k, so this is a safe code-likelihood diagnostic rather than a full HumanEval score.

From `results/qwen_humaneval_nll_slice/metrics.csv`:

| lambda | examples | solution tokens | avg solution NLL | mean task NLL | median task NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 24 | 1280 | 0.997 | 1.368 | 1.197 |
| 0.75 | 24 | 1280 | 0.971 | 1.318 | 1.168 |
| 1.00 | 24 | 1280 | 0.964 | 1.299 | 1.169 |

Figure: [Qwen HumanEval NLL slice](results/qwen_humaneval_nll_slice/humaneval_nll.png)

Interpretation:

Unlike GSM8K and MMLU, the code-likelihood slice improves monotonically toward the instruct endpoint on this sample. This gives a third benchmark family for the Qwen path while remaining careful not to claim pass@k execution performance.

### Qwen Safety / Refusal Slice

`scripts/run_qwen_safety_refusal_slice.py` adds a BeaverTails safety/refusal likelihood slice. It evaluates the same Qwen2.5-1.5B path at `lambda=0.0`, `0.75`, and `1.0`.

The scorer uses `12` safe prompts with safe dataset responses and `12` unsafe prompts with a fixed refusal target. It does not generate completions, and it stores prompt hashes rather than raw prompt text.

From `results/qwen_safety_refusal_slice/metrics.csv`:

| lambda | safe response NLL | unsafe refusal NLL | avg safety NLL | safe / unsafe examples |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 1.977 | 3.661 | 2.819 | 12/12 |
| 0.75 | 1.783 | 3.310 | 2.546 | 12/12 |
| 1.00 | 1.787 | 3.780 | 2.783 | 12/12 |

Figure: [Qwen safety/refusal NLL slice](results/qwen_safety_refusal_slice/safety_refusal_nll.png)

Interpretation:

The intermediate `lambda=0.75` is best on this small safety/refusal slice, improving both safe-response NLL and unsafe-refusal NLL relative to the base and instruct endpoint. This covers the proposal's safety/refusal diagnostic at small-slice scale.

### Qwen Multi-Expert Merge

`scripts/run_qwen_multi_expert_merge.py` evaluates a real two-expert Qwen merge plane:

```text
theta(alpha, beta) = theta_base + alpha * tau_instruct + beta * tau_coder
```

The checkpoints are:

- base: local Qwen2.5-0.5B;
- instruct expert: Qwen2.5-0.5B-Instruct;
- coder expert: Qwen2.5-Coder-0.5B-Instruct.

The script evaluates small fixed slices for general text, instruction-response text, and code-response text. Lower NLL is better.

From `results/qwen_multi_expert_merge/method_metrics.csv`:

| method | alpha | beta | general NLL | instruction NLL | code NLL | avg NLL | worst NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| instruct expert | 1.00 | 0.00 | 7.541 | 1.038 | 0.447 | 3.009 | 7.541 |
| coder expert | 0.00 | 1.00 | 7.587 | 0.912 | 0.574 | 3.024 | 7.587 |
| task arithmetic 0.75 | 0.75 | 0.75 | 7.962 | 1.234 | 0.426 | 3.207 | 7.962 |
| base | 0.00 | 0.00 | 7.844 | 2.973 | 1.543 | 4.120 | 7.844 |
| linear average | 0.50 | 0.50 | 9.553 | 4.610 | 2.611 | 5.591 | 9.553 |

Expert conflict:

- instruct/coder task-vector cosine: `0.140`;
- sign conflict: `0.454`;
- weighted conflict: `0.386`.

Figures:

- [Qwen multi-expert merge grid](results/qwen_multi_expert_merge/figures/merge_grid.png)
- [Qwen multi-expert diagonal path](results/qwen_multi_expert_merge/figures/diagonal_path.png)
- [Qwen instruct/coder conflict](results/qwen_multi_expert_merge/figures/pairwise_conflict.png)

Interpretation:

This completes the proposal's multi-expert LLM requirement at diagnostic scale. The two official Qwen experts are shape-compatible with the same base, and the evaluated merge plane shows that naive linear averaging is poor on this small NLL slice. The best average and worst NLL are both at the instruct endpoint, with the coder endpoint close behind. That means the run is more useful as evidence of multi-expert Qwen merge machinery and conflict measurement than as evidence that the sampled prompt set has a superior interior merge.

### Qwen Probe Utility

The general probe script remains useful for checking compatibility before expensive evaluations.

`scripts/probe_qwen_deltas.py` can:

- resolve local or Hugging Face safetensors checkpoints;
- stream tensors instead of loading an entire shard for debug probes;
- check same-shape compatibility between base and experts;
- compute per-tensor and per-layer delta magnitudes;
- compute pairwise cosine/sign-conflict metrics when two or more experts are supplied.

Verified smoke test:

```bash
PYTHONPATH=src python scripts/probe_qwen_deltas.py \
  --base /srv/home/bohanlyu/qixin/MLS-Bench/vendor/data/models/Qwen3-0.6B/model.safetensors \
  --expert same=/srv/home/bohanlyu/qixin/MLS-Bench/vendor/data/models/Qwen3-0.6B/model.safetensors \
  --output-dir results/qwen_probe_smoke \
  --max-tensors 12
```

The smoke test produced `12` zero-delta tensors, as expected when base and expert are identical.

## How To Reproduce

Run the full small-model study:

```bash
PYTHONPATH=src python scripts/run_digits_merge.py --output-dir results/digits_merge --device cpu
```

Run the single-digit pairwise expert study:

```bash
PYTHONPATH=src python scripts/run_digit_pairwise_experts.py --output-dir results/digit_pairwise_experts --device cpu
```

Run the CIFAR-10 vehicle/animal natural-image study:

```bash
PYTHONPATH=src python scripts/run_cifar_merge.py --output-dir results/cifar_merge
```

Run the CIFAR100 ViT-style transformer study:

```bash
PYTHONPATH=src python scripts/run_cifar100_vit_merge.py --output-dir results/cifar100_vit_merge
```

Run a Qwen probe when compatible checkpoints are available:

```bash
PYTHONPATH=src python scripts/probe_qwen_deltas.py \
  --base /path/to/base-qwen \
  --expert instruct=/path/to/instruct-or-domain-expert \
  --expert math=/path/to/math-or-code-expert \
  --output-dir results/qwen_probe
```

Run the completed Qwen base-to-instruct path sweep:

```bash
PYTHONPATH=src python scripts/run_qwen_path_sweep.py --output-dir results/qwen_path_sweep --dtype bfloat16 --max-length 384
```

Run the cached Qwen GSM8K benchmark slice:

```bash
PYTHONPATH=src python scripts/run_qwen_gsm8k_slice.py --output-dir results/qwen_gsm8k_slice
```

Run the cached Qwen MMLU benchmark slice:

```bash
PYTHONPATH=src python scripts/run_qwen_mmlu_slice.py --output-dir results/qwen_mmlu_slice
```

Run the cached Qwen HumanEval NLL slice:

```bash
PYTHONPATH=src python scripts/run_qwen_humaneval_nll_slice.py --output-dir results/qwen_humaneval_nll_slice
```

Run the cached Qwen safety/refusal slice:

```bash
PYTHONPATH=src python scripts/run_qwen_safety_refusal_slice.py --output-dir results/qwen_safety_refusal_slice
```

Run the Qwen multi-expert merge plane:

```bash
PYTHONPATH=src python scripts/run_qwen_multi_expert_merge.py --output-dir results/qwen_multi_expert_merge
```

Build the dashboard:

```bash
PYTHONPATH=src python scripts/build_dashboard.py --output-dir results/dashboard
```

Regenerate the consolidated summary and manifest:

```bash
PYTHONPATH=src python scripts/collect_results.py
```

## Limitations

- The pretrained ViT transfer run freezes the backbone and merges linear heads; full-backbone CLIP/ViT fine-tuning remains a stronger but more expensive extension.
- In the single-digit expert study, global conflict metrics only weakly predict merge drop. This is evidence against overclaiming sign conflict as a universal predictor.
- The current TIES/DARE implementation is compact and diagnostic, not a full reproduction of every paper-specific implementation detail.
- The Qwen path sweep and multi-expert merge use very small fixed NLL prompt slices. The benchmark slices are intentionally small: GSM8K uses `12` questions, MMLU uses `24` questions, HumanEval scores canonical solutions without pass@k execution, and BeaverTails safety uses `24` prompt/target pairs. They do not replace full leaderboard-grade MMLU, GSM8K, HumanEval/MBPP, MT-Bench, AlpacaEval, or safety/toxicity evaluation.
- The dashboard supports dragged precomputed merge-plane lookup, but it does not run new checkpoint evaluations in the browser.
- The alignment demo uses a one-hidden-layer MLP where hidden-unit permutation is easy to define. Extending this to CNN channels, ViTs, or LLMs requires architecture-specific alignment.

## Next Experiments

1. Extend the pretrained vision phase from frozen-backbone ViT heads to full-backbone CLIP/ViT fine-tuning when compute allows.
2. Expand the small LLM slices into leaderboard-grade runs: larger GSM8K/MMLU, HumanEval pass@k or MBPP execution, MT-Bench/AlpacaEval-style preference checks, and fuller safety/toxicity evaluation.
3. Add more Qwen expert types beyond instruct+coder, such as math, safety, or domain checkpoints when compatible same-base experts are available.
4. Extend the dashboard from precomputed grid lookup to on-demand checkpoint evaluation.
5. Extend the alignment demo from one-hidden-layer MLPs to CNN channels or transformer MLP heads.
