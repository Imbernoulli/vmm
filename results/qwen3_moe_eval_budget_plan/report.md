# Qwen3 MoE vLLM Eval Budget Plan

这份计划解决的是评测强度问题：现在 Qwen3 MoE gate 的 `64` examples 只适合 smoke，不足以支撑 final selector 的 Wilson confidence gate 和 paired prediction gate。

- Status: `ready_for_budgeted_remote_vllm_eval`
- Methods to evaluate: `10`
- Current gate max examples: `64`
- Recommended command max examples: `384`
- Total current prompt budget: `2560`
- Total recommended prompt budget: `15360`
- Additional prompt budget: `12800`

## Why This Budget

Wilson gate: for a binary task score near the worst case `p=0.5`, choose `n` so the 95% Wilson half-width is at most `0.05`. This gives `381` raw examples, rounded to `384` for batch-friendly execution.

Paired gate: final selection compares source and candidate predictions on the same examples. The planner asks for enough shared examples to make a `0.05` net source advantage significant at alpha `0.05`, assuming `0.25` paired discordance. This requires `62` discordant examples, about `248` total shared examples before rounding.

因此这里推荐的不是“静态多跑一点”，而是让下游 eval 能真正支持 source dominance、task regression、score confidence 和 paired-prediction regression 这些机制判断。

## Task Budget

| task | current | Wilson n | paired n | recommended max | achievable | half-width | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gsm8k` | 64 | 381 | 248 | 384 | 384 | 0.0497621 | `target_met` |
| `humaneval_compile` | 64 | 381 | 248 | 384 | 164 | 0.0756443 | `target_not_met_dataset_cap` |
| `mmlu` | 64 | 381 | 248 | 384 | 384 | 0.0497621 | `target_met` |
| `safety` | 64 | 381 | 248 | 384 | 384 | 0.0497621 | `target_met` |

## Method Budget

| order | method | role | current | recommended | extra prompts | eval status |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | `source_qwen3_30b_instruct` | `source` | 64 | 384 | 1280 | `not_run` |
| 1 | `source_qwen3_30b_coder` | `source` | 64 | 384 | 1280 | `not_run` |
| 2 | `qwen3_moe_unified_route_guarded_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 3 | `qwen3_moe_audit_gated_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 4 | `qwen3_moe_trust_region_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 5 | `qwen3_moe_expert_only_trust_region_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 6 | `qwen3_moe_tail_trimmed_expert_only_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 7 | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 8 | `qwen3_moe_layer_chunk_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |
| 9 | `qwen3_moe_unified_mechanism_candidate` | `candidate` | 64 | 384 | 1280 | `not_run` |

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
| `unified_rule_alias_validation` | `qwen3_moe_searched_no_gt065_max_retention_candidate` -> `qwen3_moe_unified_mechanism_candidate` | 3072 | Did the unified risk/retention optimizer recover the same materialized rule as the validated searched no-gt-0.65 checkpoint? |

## How To Run

在 GPU host 上从仓库根目录运行：

```bash
results/qwen3_moe_eval_budget_plan/run_eval_budget.sh all
python scripts/audit_qwen3_moe_eval_bundle.py --output-dir results/qwen3_moe_eval_bundle_audit
python scripts/refresh_qwen3_moe_post_eval.py
```

也可以只跑一个方法：

```bash
results/qwen3_moe_eval_budget_plan/run_eval_budget.sh qwen3_moe_tail_trimmed_expert_only_candidate
```

注意：原始 gate 里的 `max_examples=64` 仍是 audit floor；预算版 runner 会用更高的 `--max-examples` 覆盖 eval 命令。HumanEval 数据集上限低于推荐值时，selector 会使用实际落盘的样本数计算区间。

## Outputs

- `results/qwen3_moe_eval_budget_plan/task_budget.csv`
- `results/qwen3_moe_eval_budget_plan/method_budget.csv`
- `results/qwen3_moe_eval_budget_plan/mechanism_budget.csv`
- `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh`
- `results/qwen3_moe_eval_budget_plan/summary.json`
