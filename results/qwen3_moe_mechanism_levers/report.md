# Qwen3 MoE Mechanism Leverage Map

这个 artifact 把现有 Qwen3 MoE probe 结果压成一个优化优先级表：不是问哪个算法名更好，而是问哪个机制最可能解释平均失败，以及下一步应该用什么实验验证。

- Status: `mechanism_leverage_map_ready`
- Lever count: `9`
- Top lever: `source_and_candidate_downstream_eval`
- Fine calibration layers: `12,13,17,20,21,22,23,26`

## Levers

| mechanism | priority | confidence | evidence | action |
| --- | ---: | --- | --- | --- |
| `source_and_candidate_downstream_eval` | 0.98 | `high` | examples 64 -> 384; extra prompts 12800 | run budgeted one-model-at-a-time vLLM eval before accepting any average |
| `router_direct_movement` | 0.94 | `high` | allowed layers 0/48; min top1 0.0689655169844627; router rel-norm 0.7392916983133861 | freeze router for same-shape candidate; only consider calibrated router deltas |
| `routed_expert_tail_cap_0_75` | 0.86 | `high` | route->audit removes >0.75 by 675 and >1.0 by 182 | keep file-level/audit-level relative-delta cap as a mandatory safety gate |
| `risk_penalty_complexity` | 0.83 | `medium_high` | risk flag ablation tail reductions 0; summed retention loss 0.00525858; searched no-gt-0.65 rel norm 0.2475948491291486 | prefer the simpler uniform 0.65 cap unless downstream eval proves risk penalties preserve task behavior |
| `tail_cap_0_65` | 0.8 | `medium_high` | expert_only->tail_trimmed removes >0.75 by 14 and >0.65 by 286; tail max rel 0.650082822181077 | evaluate tail-trimmed as the conservative expert-only candidate |
| `route_load_trust_region` | 0.78 | `medium_high` | audit->trust removes >0.75 by 150 and >0.65 by 780 | keep trust-region as an ablation, but do not assume its extra risk flags improve utility |
| `shared_attention_delta` | 0.74 | `medium` | trust->expert-only removes attention relative norm 0.188546444938151 with routed-tail reduction 0 | freeze attention in current unified candidate, but keep trust_region vs expert_only eval as the utility test |
| `importance_guided_layer_chunking` | 0.72 | `medium` | top fine-calibration layers: 12,13,17,20,21,22,23,26; top expert-geometry layers: 12,13,14,15,16,17,22,23 | use high-sensitivity layers for future unlabeled coefficient calibration; keep low-sensitivity layers coarse |
| `expert_identity_and_subspace_probe` | 0.66 | `medium` | Qwen3 identity gate passes, but no expert-output subspace clustering artifact is tracked for candidate generation. | keep identity as required preflight; add expert-output/subspace probe before averaging unrelated downstream MoE fine-tunes |

## Next Experiments

| rank | mechanism | test or command |
| ---: | --- | --- |
| 1 | `source_and_candidate_downstream_eval` | `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh all` |
| 2 | `router_direct_movement` | `route-KD/HARC-style router calibration after frozen-router baseline and sources finish vLLM eval` |
| 3 | `routed_expert_tail_cap_0_75` | `compare route_guarded vs audit_gated under budgeted vLLM eval` |
| 4 | `risk_penalty_complexity` | `budgeted paired vLLM comparison of tail_trimmed vs searched_no_gt065 vs layer_chunk vs unified mechanism` |
| 5 | `tail_cap_0_65` | `budgeted paired vLLM comparison of expert_only vs tail_trimmed` |
| 6 | `route_load_trust_region` | `compare audit_gated vs trust_region and inspect paired task regressions` |

## Layer/Chunk Calibration Plan

这个表把 Expert Merging/importance-guided chunking 的思想落到当前 Qwen3 数据上：高敏感层给更多校准系数，低敏感层共享粗粒度系数。

| layer | score | policy | slots | route->trust | calibrate frac | router rel | geometry risk | high-geom experts |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.904783 | `per_layer_coefficients` | 4 | 0.0556817 | 1 | 0.844352 | 0.714306 | 15 |
| 22 | 0.863789 | `per_layer_coefficients` | 4 | 0.0432741 | 1 | 0.793704 | 0.671821 | 9 |
| 20 | 0.863149 | `per_layer_coefficients` | 4 | 0.0593528 | 1 | 0.808873 | 0.671361 | 10 |
| 23 | 0.85163 | `per_layer_coefficients` | 4 | 0.0519647 | 1 | 0.770017 | 0.674895 | 10 |
| 12 | 0.84651 | `per_layer_coefficients` | 4 | 0.0520805 | 1 | 0.823488 | 0.67519 | 11 |
| 26 | 0.824761 | `per_layer_coefficients` | 4 | 0.0630701 | 1 | 0.637357 | 0.641394 | 9 |
| 21 | 0.821837 | `per_layer_coefficients` | 4 | 0.0602967 | 1 | 0.786077 | 0.662217 | 8 |
| 13 | 0.809125 | `per_layer_coefficients` | 4 | 0.0563345 | 1 | 0.85527 | 0.675991 | 9 |
| 18 | 0.808771 | `two_layer_chunk_coefficients` | 2 | 0.0572632 | 1 | 0.869161 | 0.65006 | 5 |
| 24 | 0.802862 | `two_layer_chunk_coefficients` | 2 | 0.0489775 | 1 | 0.747783 | 0.661782 | 10 |
| 15 | 0.772271 | `two_layer_chunk_coefficients` | 2 | 0.0393666 | 1 | 0.824639 | 0.702701 | 13 |
| 19 | 0.760526 | `two_layer_chunk_coefficients` | 2 | 0.048112 | 1 | 0.825168 | 0.651748 | 5 |
| 16 | 0.752207 | `two_layer_chunk_coefficients` | 2 | 0.0415286 | 1 | 0.817559 | 0.688071 | 10 |
| 11 | 0.745159 | `two_layer_chunk_coefficients` | 2 | 0.0468571 | 1 | 0.819092 | 0.65972 | 3 |
| 25 | 0.724006 | `two_layer_chunk_coefficients` | 2 | 0.0431148 | 1 | 0.701719 | 0.651717 | 6 |
| 9 | 0.712906 | `two_layer_chunk_coefficients` | 2 | 0.0533227 | 0.916667 | 0.787176 | 0.636632 | 5 |

## Literature Hooks

- [When Model Merging Breaks Routing: Training-Free Calibration for MoE](https://arxiv.org/abs/2606.03391): MoE averages can fail through router/top-k dispatch breakdown; router movement needs calibration evidence.
- [Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression](https://arxiv.org/abs/2603.02217): After expert edits/merges, router-expert mismatch is a distinct failure mode; lightweight router KD can recover routing.
- [Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking](https://arxiv.org/abs/2509.25712): Layer/chunk coefficient learning on unlabeled calibration data is useful because merge sensitivity is heterogeneous across layers.
- [Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266): Expert-output similarity and shared subspaces help separate coherent expert groups from conflicting specializations.
- [MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs](https://arxiv.org/abs/2502.00997): MoE merging needs explicit interference mitigation and routing heuristics beyond unweighted expert averaging.

## Outputs

- `report`: `results/qwen3_moe_mechanism_levers/report.md`
- `summary`: `results/qwen3_moe_mechanism_levers/summary.json`
- `mechanism_levers`: `results/qwen3_moe_mechanism_levers/mechanism_levers.csv`
- `next_experiment_queue`: `results/qwen3_moe_mechanism_levers/next_experiment_queue.csv`
- `layer_chunking_plan`: `results/qwen3_moe_mechanism_levers/layer_chunking_plan.csv`
- `literature_sources`: `results/qwen3_moe_mechanism_levers/literature_sources.json`
