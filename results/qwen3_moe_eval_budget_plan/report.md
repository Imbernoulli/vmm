# Qwen3 MoE vLLM Eval Budget Plan

这份计划解决的是评测强度问题：现在 Qwen3 MoE gate 的 `64` examples 只适合 smoke，不足以支撑 final selector 的 Wilson confidence gate 和 paired prediction gate。

- Status: `ready_for_budgeted_remote_vllm_eval`
- Methods to evaluate: `16`
- Ready-to-host methods now: `12`
- Current gate max examples: `64`
- Recommended command max examples: `384`
- Total current prompt budget: `4096`
- Total recommended prompt budget: `24576`
- Additional prompt budget: `20480`
- Final core methods / prompts: `4` / `6144`
- Mechanism ablation methods / prompts: `10` / `15360`
- Canonical task manifest: `results/qwen3_moe_mechanism_eval_gate/task_manifest.json`
- Task manifest aligned methods: `16/16`
- Router calibration active / ready / plan-pruned caps: `2` / `0` / `2`

## Why This Budget

Wilson gate: for a binary task score near the worst case `p=0.5`, choose `n` so the 95% Wilson half-width is at most `0.05`. This gives `381` raw examples, rounded to `384` for batch-friendly execution.

Paired gate: final selection compares source and candidate predictions on the same examples. The planner asks for enough shared examples to make a `0.05` net source advantage significant at alpha `0.05`, assuming `0.25` paired discordance. This requires `62` discordant examples, about `248` total shared examples before rounding.

因此这里推荐的不是“静态多跑一点”，而是让下游 eval 能真正支持 source dominance、task regression、score confidence 和 paired-prediction regression 这些机制判断。

Pairing contract: every final-selection source and candidate command now uses the same task manifest path. This moves the paired-prediction requirement before the run instead of discovering mismatched examples only after vLLM output is written.

Router calibration: budget planning now reads the route-margin-gated calibration plan. Only caps that pass the planned margin gate and are enabled by the job default-run list enter the default budget; plan-pruned caps remain explicit ablations.

Queue split: the default runner request is `final`, which evaluates the two source endpoints plus the trust-region-final-selectable candidates. `mechanism` keeps the older candidates available for attribution after the core final decision is scored.

## Task Budget

| task | current | Wilson n | paired n | recommended max | achievable | half-width | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gsm8k` | 64 | 381 | 248 | 384 | 384 | 0.0497621 | `target_met` |
| `humaneval_compile` | 64 | 381 | 248 | 384 | 164 | 0.0756443 | `target_not_met_dataset_cap` |
| `mmlu` | 64 | 381 | 248 | 384 | 384 | 0.0497621 | `target_met` |
| `safety` | 64 | 381 | 248 | 384 | 384 | 0.0497621 | `target_met` |

## Queue Budget

| queue | alias | methods | ready | recommended prompts | additional prompts |
| --- | --- | ---: | ---: | ---: | ---: |
| `final_selection_core` | `final` | 4 | 4 | 6144 | 5120 |
| `mechanism_ablation` | `mechanism` | 10 | 8 | 15360 | 12800 |
| `router_calibration_pending` | `router` | 2 | 0 | 3072 | 2560 |

## Method Budget

| order | method | queue | role | serve | current | recommended | extra prompts | eval status |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | `source_qwen3_30b_instruct` | `final_selection_core` | `source` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 1 | `source_qwen3_30b_coder` | `final_selection_core` | `source` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 2 | `qwen3_moe_unified_route_guarded_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 3 | `qwen3_moe_audit_gated_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 4 | `qwen3_moe_trust_region_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 5 | `qwen3_moe_expert_only_trust_region_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 6 | `qwen3_moe_tail_trimmed_expert_only_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 7 | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 8 | `qwen3_moe_layer_chunk_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 9 | `qwen3_moe_unified_mechanism_candidate` | `mechanism_ablation` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 10 | `qwen3_moe_mechanistic_unified_candidate` | `final_selection_core` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 11 | `qwen3_moe_subspace_scaled_candidate` | `final_selection_core` | `candidate` | `ready_to_host` | 64 | 384 | 1280 | `not_run` |
| 12 | `qwen3_moe_router_coupled_candidate` | `mechanism_ablation` | `candidate` | `checkpoint_missing_until_materialized` | 64 | 384 | 1280 | `not_run` |
| 13 | `qwen3_moe_harc_router_candidate` | `mechanism_ablation` | `candidate` | `checkpoint_missing_until_harc_solver_delta` | 64 | 384 | 1280 | `not_run` |
| 14 | `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `router_calibration_pending` | `candidate` | `pending_materialization` | 64 | 384 | 1280 | `not_run` |
| 15 | `qwen3_moe_router_calibrated_searched_no_gt065_margin_profile_candidate` | `router_calibration_pending` | `candidate` | `pending_materialization` | 64 | 384 | 1280 | `not_run` |

## Task Manifest Alignment

| method | role | serve | manifest aligned | task manifest |
| --- | --- | --- | --- | --- |
| `source_qwen3_30b_instruct` | `source` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `source_qwen3_30b_coder` | `source` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_unified_route_guarded_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_audit_gated_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_trust_region_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_expert_only_trust_region_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_layer_chunk_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_unified_mechanism_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_mechanistic_unified_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_subspace_scaled_candidate` | `candidate` | `ready_to_host` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_router_coupled_candidate` | `candidate` | `checkpoint_missing_until_materialized` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_harc_router_candidate` | `candidate` | `checkpoint_missing_until_harc_solver_delta` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `candidate` | `pending_materialization` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |
| `qwen3_moe_router_calibrated_searched_no_gt065_margin_profile_candidate` | `candidate` | `pending_materialization` | `True` | `results/qwen3_moe_mechanism_eval_gate/task_manifest.json` |

## Router Calibration Budget

| cap | method | active | checkpoint | default | margin planned pass | plan-pruned | eval status | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cap001` | `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `True` | `False` | `True` | `True` | `False` | `not_run` | awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,... |
| `cap0025` | `qwen3_moe_router_calibrated_searched_no_gt065_cap0025_candidate` | `False` | `False` | `False` | `False` | `True` | `not_run` | awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,... |
| `cap005` | `qwen3_moe_router_calibrated_searched_no_gt065_cap005_candidate` | `False` | `False` | `False` | `False` | `True` | `not_run` | awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,... |
| `margin_profile` | `qwen3_moe_router_calibrated_searched_no_gt065_margin_profile_candidate` | `True` | `False` | `True` | `True` | `False` | `not_run` | awaiting_baseline_eval,awaiting_source_eval,awaiting_router_training,awaiting_candidate_eval,... |

## Mechanism Budget

| test | comparison | prompt budget | question |
| --- | --- | ---: | --- |
| `source_control_floor` | `source_qwen3_30b_instruct` -> `source_qwen3_30b_coder` | 3072 | How different are the source endpoints on the same downstream tasks? |
| `tail_delta_cap` | `qwen3_moe_unified_route_guarded_candidate` -> `qwen3_moe_audit_gated_candidate` | 3072 | Does clipping extreme routed-expert deltas help? |
| `route_load_trust_region` | `qwen3_moe_audit_gated_candidate` -> `qwen3_moe_trust_region_candidate` | 3072 | Do route/load/category/fragility probes identify the expert groups that need a tighter cap? |
| `shared_attention_ablation` | `qwen3_moe_trust_region_candidate` -> `qwen3_moe_expert_only_trust_region_candidate` | 3072 | Should the unified MoE rule move shared attention, or keep it fixed? |
| `second_stage_tail_trim` | `qwen3_moe_expert_only_trust_region_candidate` -> `qwen3_moe_tail_trimmed_expert_only_candidate` | 3072 | Does the stricter 0.65 routed-expert tail cap remove risk without removing ability? |
| `risk_penalty_simplification` | `qwen3_moe_tail_trimmed_expert_only_candidate` -> `qwen3_moe_searched_no_gt065_max_retention_candidate` | 3072 | Are hand-built risk penalties necessary after a uniform 0.65 expert cap is enforced? |
| `layer_chunk_sensitivity` | `qwen3_moe_searched_no_gt065_max_retention_candidate` -> `qwen3_moe_layer_chunk_candidate` | 3072 | Do importance-guided layer/chunk coefficients improve the unified MoE rule beyond a uniform expert cap? |
| `candidate_vs_sources` | `source_qwen3_30b_instruct` -> `qwen3_moe_unified_mechanism_candidate` | 3072 | Does any same-shape candidate avoid Pareto domination by the two source endpoints? |
| `unified_mechanism_optimizer` | `qwen3_moe_layer_chunk_candidate` -> `qwen3_moe_unified_mechanism_candidate` | 3072 | Does the router/evidence/geometry-risk optimizer improve downstream behavior beyond the layer/chunk candidate? |
| `expert_subspace_conflict_ablation` | `qwen3_moe_mechanistic_unified_candidate` -> `qwen3_moe_subspace_scaled_candidate` | 3072 | Do uncovered high subspace-conflict experts need additional non-base shrink after the unified mechanism cap? |
| `mechanistic_unified_optimizer` | `qwen3_moe_unified_mechanism_candidate` -> `qwen3_moe_mechanistic_unified_candidate` | 3072 | Does the benefit/curvature/interference objective explain a better scale law than the current risk-weighted cap search? |
| `router_coupled_boundary_ablation` | `qwen3_moe_mechanistic_unified_candidate` -> `qwen3_moe_router_coupled_candidate` | 3072 | Does the layer-level router-boundary fragility signal justify extra expert shrink after the B/H/I scale law? |
| `harc_router_calibration_ablation` | `qwen3_moe_unified_mechanism_candidate` -> `qwen3_moe_harc_router_candidate` | 3072 | Does HARC-style second-order router calibration recover downstream score beyond the frozen-router unified mechanism candidate? |

## How To Run

在 GPU host 上从仓库根目录运行：

```bash
results/qwen3_moe_eval_budget_plan/run_eval_budget.sh preflight
results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final
python scripts/audit_qwen3_moe_eval_bundle.py --output-dir results/qwen3_moe_eval_bundle_audit
python scripts/refresh_qwen3_moe_post_eval.py
```

机制归因需要时再跑 ablation 队列：

```bash
results/qwen3_moe_eval_budget_plan/run_eval_budget.sh mechanism
```

也可以只跑一个方法：

```bash
results/qwen3_moe_eval_budget_plan/run_eval_budget.sh qwen3_moe_tail_trimmed_expert_only_candidate
```

注意：原始 gate 里的 `max_examples=64` 仍是 audit floor；预算版 runner 会用更高的 `--max-examples` 覆盖 eval 命令。runner 默认请求是 `final`，不是全量 ablation；需要全量时显式运行 `all`。runner 默认先检查 manifest、GPU/vLLM/curl 和被请求方法的模型路径；确需跳过前置检查时可以设置 `EVAL_BUDGET_SKIP_PREFLIGHT=1`。HumanEval 数据集上限低于推荐值时，selector 会使用实际落盘的样本数计算区间。

## Outputs

- `results/qwen3_moe_eval_budget_plan/task_budget.csv`
- `results/qwen3_moe_eval_budget_plan/method_budget.csv`
- `results/qwen3_moe_eval_budget_plan/mechanism_budget.csv`
- `results/qwen3_moe_eval_budget_plan/router_calibration_budget.csv`
- `results/qwen3_moe_eval_budget_plan/task_manifest_alignment.csv`
- `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh`
- `results/qwen3_moe_eval_budget_plan/summary.json`
