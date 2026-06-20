# Qwen3 MoE vLLM Eval Bundle Audit

这个 audit 检查远端 vLLM eval 落盘结果是否能被 downstream selector 使用，防止旧模型名、缺任务、样本数不足或缺 predictions 的结果混入 Average 选择。

- Status: `awaiting_eval`
- Usable for selection: `0/10`
- Source usable: `0/2`
- Candidate usable: `0/8`
- Unified usable: `False`

| method | status | model | tasks | examples | predictions | usable | issue preview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `source_qwen3_30b_instruct` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `source_qwen3_30b_coder` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_unified_route_guarded_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_audit_gated_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_trust_region_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_expert_only_trust_region_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_layer_chunk_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |
| `qwen3_moe_unified_mechanism_candidate` | `missing` | `False` | `False` | `False` | `False` | `False` | `eval_output_dir_missing;missing_files:summary.json,eval_plan.csv,metrics.csv,model_summary.csv,predictions.csv; +6 more` |

## Outputs

- `results/qwen3_moe_eval_bundle_audit/audit_rows.csv`
- `results/qwen3_moe_eval_bundle_audit/summary.json`
