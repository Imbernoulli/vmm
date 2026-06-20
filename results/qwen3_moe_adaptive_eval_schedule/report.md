# Qwen3 MoE Adaptive Eval Schedule

This scheduler turns mechanism evidence into a sequential vLLM budget: source controls first, high-value mechanism probes second, full-budget expansion only for candidates that are not source-dominated.

- Status: `adaptive_schedule_ready`
- Source controls complete: `False`
- Round-1 probe candidates: `5`
- Top action: `run_or_extend_source_control_probe` for `source_qwen3_30b_instruct`

## Method Schedule

| rank | method | action | status | examples | priority | reason |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | `source_qwen3_30b_instruct` | `run_or_extend_source_control_probe` | `source_control_required` | 64 | 1.200 | Both source endpoints must be scored before any same-shape average can be accepted or pruned. |
| 2 | `source_qwen3_30b_coder` | `run_or_extend_source_control_probe` | `source_control_required` | 64 | 1.200 | Both source endpoints must be scored before any same-shape average can be accepted or pruned. |
| 3 | `qwen3_moe_unified_mechanism_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | 64 | 1.090 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 4 | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | 64 | 1.000 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 5 | `qwen3_moe_layer_chunk_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | 64 | 0.980 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 6 | `qwen3_moe_subspace_scaled_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | 64 | 0.930 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 7 | `qwen3_moe_tail_trimmed_expert_only_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | 64 | 0.920 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 8 | `qwen3_moe_expert_only_trust_region_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | 0 | 0.880 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 9 | `qwen3_moe_trust_region_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | 0 | 0.840 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 10 | `qwen3_moe_audit_gated_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | 0 | 0.760 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 11 | `qwen3_moe_unified_route_guarded_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | 0 | 0.690 | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 12 | `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `materialize_checkpoint_first` | `checkpoint_missing` | 0 | 0.350 | The method is in the eval plan but its checkpoint path is not present yet. |
| 13 | `qwen3_moe_router_calibrated_searched_no_gt065_margin_profile_candidate` | `materialize_checkpoint_first` | `checkpoint_missing` | 0 | 0.350 | The method is in the eval plan but its checkpoint path is not present yet. |

## Mechanism Gates

| test | action | required method statuses |
| --- | --- | --- |
| `source_control_floor` | `hold` | `{"source_qwen3_30b_coder": "source_control_required", "source_qwen3_30b_instruct": "source_control_required"}` |
| `tail_delta_cap` | `wait_for_source_controls` | `{"qwen3_moe_audit_gated_candidate": "awaiting_source_controls", "qwen3_moe_unified_route_guarded_candidate": "awaiting_source_controls"}` |
| `route_load_trust_region` | `wait_for_source_controls` | `{"qwen3_moe_audit_gated_candidate": "awaiting_source_controls", "qwen3_moe_trust_region_candidate": "awaiting_source_controls"}` |
| `shared_attention_ablation` | `wait_for_source_controls` | `{"qwen3_moe_expert_only_trust_region_candidate": "awaiting_source_controls", "qwen3_moe_trust_region_candidate": "awaiting_source_controls"}` |
| `second_stage_tail_trim` | `wait_for_source_controls` | `{"qwen3_moe_expert_only_trust_region_candidate": "awaiting_source_controls", "qwen3_moe_tail_trimmed_expert_only_candidate": "awaiting_source_controls"}` |
| `risk_penalty_simplification` | `wait_for_source_controls` | `{"qwen3_moe_searched_no_gt065_max_retention_candidate": "awaiting_source_controls", "qwen3_moe_tail_trimmed_expert_only_candidate": "awaiting_source_controls"}` |
| `layer_chunk_sensitivity` | `wait_for_source_controls` | `{"qwen3_moe_layer_chunk_candidate": "awaiting_source_controls", "qwen3_moe_searched_no_gt065_max_retention_candidate": "awaiting_source_controls"}` |
| `candidate_vs_sources` | `wait_for_source_controls` | `{"qwen3_moe_unified_mechanism_candidate": "awaiting_source_controls", "source_qwen3_30b_instruct": "source_control_required"}` |
| `unified_mechanism_optimizer` | `wait_for_source_controls` | `{"qwen3_moe_layer_chunk_candidate": "awaiting_source_controls", "qwen3_moe_unified_mechanism_candidate": "awaiting_source_controls"}` |
| `expert_subspace_conflict_ablation` | `wait_for_source_controls` | `{"qwen3_moe_subspace_scaled_candidate": "awaiting_source_controls", "qwen3_moe_unified_mechanism_candidate": "awaiting_source_controls"}` |

## Outputs

- `results/qwen3_moe_adaptive_eval_schedule/adaptive_schedule.csv`
- `results/qwen3_moe_adaptive_eval_schedule/mechanism_schedule.csv`
- `results/qwen3_moe_adaptive_eval_schedule/run_adaptive_eval.sh`
- `results/qwen3_moe_adaptive_eval_schedule/summary.json`
- `results/qwen3_moe_adaptive_eval_schedule/report.md`
