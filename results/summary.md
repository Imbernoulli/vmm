# Result Summary

Generated at: `2026-06-19T03:53:13.920974+00:00`

## Coverage

Complete: `17`; partial: `0`; missing: `0`.

| item | status | evidence |
| --- | --- | --- |
| 2D task-vector merge landscape | complete | Digits and CIFAR grid metrics plus merge landscape figures. |
| Per-task basin overlay | complete | results/digits_merge/figures/per_task_basin_overlay.png. |
| Task-arithmetic lambda sweep | complete | Digits, CIFAR, and Qwen path/lambda sweeps. |
| Merge-method overlay | complete | Digits method table and overlay cover average, task arithmetic, SLERP, TIES, DARE, TIES+DARE, Fisher, RegMean, layer-wise task arithmetic, and validation grid search. |
| Layer-wise interference atlas | complete | Digits, CIFAR, and pairwise single-digit conflict tables/figures. |
| One-class expert surrogate | complete | Ten single-digit experts and all 45 pairwise merges. |
| Randomness and alignment analysis | complete | Independent-initialization MLP path before/after Hungarian hidden-unit alignment. |
| Natural-image small-model case study | complete | CIFAR-10 vehicle/animal GroupNorm CNN merge landscape. |
| CLIP or ViT task-vector phase | complete | CIFAR100 ViT-style from-scratch transformer and ImageNet-pretrained ViT-B/16 frozen-backbone transfer merge studies are present. |
| Qwen-compatible LLM probe | complete | Safetensors probe and same-file smoke test. |
| Real Qwen LLM path sweep | complete | Qwen2.5-1.5B base-to-instruct path is evaluated with fixed NLL prompts plus GSM8K, MMLU, and HumanEval benchmark slices. |
| Multi-expert LLM merge | complete | Qwen2.5-0.5B base, Qwen2.5-0.5B-Instruct, and Qwen2.5-Coder-0.5B-Instruct are evaluated in a two-expert merge plane. |
| Formal LLM benchmark slices | complete | Representative Qwen2.5-1.5B benchmark slices cover MMLU, GSM8K, HumanEval canonical-solution NLL, and BeaverTails safety/refusal NLL. |
| Probe-guided Average decision report | complete | results/average_decision_report/report.md converts merge grids, conflict probes, and optional MoE routing probes into same-shape average decisions. |
| MoE same-shape averaging plan | complete | results/moe_average_plan/report.md maps router/expert probes into same-shape router, shared-module, expert, and adapter averaging actions. |
| Same-shape checkpoint writer | complete | scripts/write_same_shape_average_checkpoint.py writes same-shape safetensors checkpoints; results/same_shape_writer_smoke/report.md validates Qwen2.5-0.5B base/instruct/coder dry-run compatibility. |
| Interactive explainer UI | complete | Dashboard includes a draggable precomputed merge-plane explorer with task-pair, method, objective, raw/normalized plane, alpha/beta, and lambda controls. |

## Key Metrics

| experiment | metric | value |
| --- | --- | ---: |
| digits merge | linear-average worst accuracy | 0.922 |
| digits merge | layer-wise task arithmetic worst accuracy | 0.928 |
| digits merge | RegMean linear-layer worst accuracy | 0.939 |
| digits merge | max grid worst accuracy | 0.961 |
| digits merge | global task-vector cosine | 0.138 |
| single-digit pairs | mean linear worst accuracy | 0.986 |
| single-digit pairs | weighted conflict vs drop Spearman | 0.165 |
| alignment | midpoint accuracy before to after | 0.944 to 0.971 |
| alignment | loss barrier before to after | 0.064 to 0.006 |
| CIFAR | linear-average worst accuracy | 0.249 |
| CIFAR | validation-grid best worst accuracy | 0.426 |
| CIFAR100 ViT-style | linear-average worst accuracy | 0.076 |
| CIFAR100 ViT-style | best method worst accuracy | 0.197 |
| pretrained ViT transfer | linear-average worst accuracy | 0.763 |
| pretrained ViT transfer | best method worst accuracy | 0.783 |
| Qwen path | best average-NLL lambda | 0.75 |
| Qwen path | instruction NLL at base to best | 3.612 to 1.811 |
| Qwen GSM8K slice | best strict exact match | 0.083 at lambda 0.75 |
| Qwen GSM8K slice | best loose exact match | 0.250 at lambda 0.75 |
| Qwen MMLU slice | best accuracy | 0.750 at lambda 0.75 |
| Qwen MMLU slice | best correct / total | 18/24 |
| Qwen HumanEval NLL slice | best solution NLL | 0.964 at lambda 1.00 |
| Qwen safety/refusal slice | best avg safety NLL | 2.546 at lambda 0.75 |
| Qwen multi-expert | best average-NLL method | instruct_expert (3.009) |
| Qwen multi-expert | linear-average avg / worst NLL | 5.591 / 9.553 |
| Qwen multi-expert | instruct/coder weighted conflict | 0.386 |
| Average decision report | avoid uniform average decisions | 3 |
| Average decision report | coefficient-search decisions | 2 |
| MoE average plan | router plan rows | 0 |
| MoE average plan | expert plan rows | 0 |
| same-shape writer smoke | Qwen-compatible tensors checked | 290 |
