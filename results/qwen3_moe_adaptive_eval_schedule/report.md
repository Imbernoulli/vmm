# Qwen3 MoE Adaptive Eval Schedule

This scheduler turns mechanism evidence into a sequential vLLM budget: source controls first, high-value mechanism probes second, full-budget expansion only for candidates that are not source-dominated. Candidate priority now combines mechanism coverage with the real safetensors delta frontier, so structurally redundant or riskier candidates can wait unless they answer a distinct mechanism question.

- Status: `adaptive_schedule_ready`
- Source controls complete: `False`
- Round-1 probe candidates: `6`
- Round-1 coverage policy: `greedy_mechanism_coverage_then_priority`
- Round-1 covered mechanism tests: `6`
- Structural frontier available: `True`
- Best structural method: `qwen3_moe_mechanistic_unified_candidate`
- Best structural safety score: `0.993`
- Structural dominance available: `True`
- Structural frontier members: `2`
- Structurally dominated methods: `8`
- Round-1 probe prompt budget: `1152`
- Runnable prompt budget now: `1664`
- Top action: `run_or_extend_source_control_probe` for `source_qwen3_30b_instruct`

## Method Schedule

| rank | method | action | status | tasks | examples | prompts | priority | structure | frontier | dominated | covered tests | paired gate | paired net | paired p | reason |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| 1 | `source_qwen3_30b_instruct` | `run_or_extend_source_control_probe` | `source_control_required` | `gsm8k,mmlu,safety,humaneval_compile` | 64 | 256 | 1.200 |  | `` | `` | `` | `source_control` |  |  | Both source endpoints must be scored before any same-shape average can be accepted or pruned. |
| 2 | `source_qwen3_30b_coder` | `run_or_extend_source_control_probe` | `source_control_required` | `gsm8k,mmlu,safety,humaneval_compile` | 64 | 256 | 1.200 |  | `` | `` | `` | `source_control` |  |  | Both source endpoints must be scored before any same-shape average can be accepted or pruned. |
| 3 | `qwen3_moe_mechanistic_unified_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | `gsm8k,mmlu,safety,humaneval_compile` | 64 | 256 | 1.239 | 0.993 | `True` | `False` | `expert_subspace_conflict_ablation,mechanistic_unified_optimizer` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 4 | `qwen3_moe_unified_mechanism_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | `gsm8k,mmlu,safety,humaneval_compile` | 64 | 256 | 1.125 | 0.970 | `False` | `True` | `candidate_vs_sources,mechanistic_unified_optimizer,unified_mechanism_optimizer` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 5 | `qwen3_moe_subspace_scaled_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | `gsm8k,humaneval_compile` | 64 | 128 | 1.068 | 0.985 | `True` | `False` | `expert_subspace_conflict_ablation` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 6 | `qwen3_moe_searched_no_gt065_max_retention_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | `gsm8k,mmlu,humaneval_compile` | 64 | 192 | 1.015 | 0.845 | `False` | `True` | `layer_chunk_sensitivity,risk_penalty_simplification` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 7 | `qwen3_moe_layer_chunk_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | `gsm8k,mmlu,humaneval_compile` | 64 | 192 | 1.007 | 0.918 | `False` | `True` | `layer_chunk_sensitivity,unified_mechanism_optimizer` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 8 | `qwen3_moe_tail_trimmed_expert_only_candidate` | `queue_after_source_controls` | `awaiting_source_controls` | `gsm8k,humaneval_compile` | 64 | 128 | 0.948 | 0.923 | `False` | `True` | `risk_penalty_simplification,second_stage_tail_trim` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 9 | `qwen3_moe_expert_only_trust_region_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | `` | 0 | 0 | 0.890 | 0.813 | `False` | `True` | `second_stage_tail_trim,shared_attention_ablation` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 10 | `qwen3_moe_trust_region_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | `` | 0 | 0 | 0.841 | 0.757 | `False` | `True` | `route_load_trust_region,shared_attention_ablation` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 11 | `qwen3_moe_audit_gated_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | `` | 0 | 0 | 0.712 | 0.451 | `False` | `True` | `route_load_trust_region,tail_delta_cap` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 12 | `qwen3_moe_unified_route_guarded_candidate` | `hold_until_source_controls_and_probe_slots` | `awaiting_source_controls` | `` | 0 | 0 | 0.536 | 0.040 | `False` | `True` | `tail_delta_cap` | `awaiting_source_controls` |  |  | Candidate pruning/acceptance requires both source endpoints on the same task manifest first. |
| 13 | `qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate` | `materialize_checkpoint_first` | `checkpoint_missing` | `` | 0 | 0 | 0.350 |  | `` | `` | `` | `checkpoint_missing` |  |  | The method is in the eval plan but its checkpoint path is not present yet. |
| 14 | `qwen3_moe_router_calibrated_searched_no_gt065_margin_profile_candidate` | `materialize_checkpoint_first` | `checkpoint_missing` | `` | 0 | 0 | 0.350 |  | `` | `` | `` | `checkpoint_missing` |  |  | The method is in the eval plan but its checkpoint path is not present yet. |

## Structural Frontier

| method | score | frontier | dominated | dominance reason | structural reason |
| --- | ---: | --- | --- | --- | --- |
| `qwen3_moe_mechanistic_unified_candidate` | 0.993 | `True` | `False` | structural_frontier_member; nearest=unified_mechanism@0.0209 | total=0.238226; routed=0.246441; max_routed=0.649063; gt0.65=0; attention/router_frozen |
| `qwen3_moe_subspace_scaled_candidate` | 0.985 | `True` | `False` | structural_frontier_member; nearest=unified_mechanism@0.0151 | total=0.239529; routed=0.247790; max_routed=0.623428; gt0.65=0; attention/router_frozen |
| `qwen3_moe_unified_mechanism_candidate` | 0.970 | `False` | `True` | dominated_by=subspace_scaled; nearest=subspace_scaled@0.0151 | total=0.240378; routed=0.248668; max_routed=0.643842; gt0.65=0; attention/router_frozen |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | 0.923 | `False` | `True` | dominated_by=unified_mechanism,mechanistic_unified,subspace_scaled; nearest=layer_chunk@0.0044 | total=0.243145; routed=0.251531; max_routed=0.650083; gt0.65=80; attention/router_frozen |
| `qwen3_moe_layer_chunk_candidate` | 0.918 | `False` | `True` | dominated_by=unified_mechanism,mechanistic_unified,subspace_scaled; nearest=tail_trimmed@0.0044 | total=0.243454; routed=0.251850; max_routed=0.650033; gt0.65=89; attention/router_frozen |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | 0.845 | `False` | `True` | dominated_by=layer_chunk,unified_mechanism,mechanistic_unified,subspace_scaled; nearest=layer_chunk@0.0520 | total=0.247595; routed=0.256134; max_routed=0.650081; gt0.65=245; attention/router_frozen |
| `qwen3_moe_expert_only_trust_region_candidate` | 0.813 | `False` | `True` | dominated_by=tail_trimmed,layer_chunk,unified_mechanism,mechanistic_unified,subspace_scaled; nearest=searched_no_gt065@0.0631 | total=0.246033; routed=0.254518; max_routed=0.750022; gt0.65=366; attention/router_frozen |
| `qwen3_moe_trust_region_candidate` | 0.757 | `False` | `True` | dominated_by=expert_only,tail_trimmed,layer_chunk,unified_mechanism,mechanistic_unified,subspace_scaled; nearest=expert_only@0.0911 | total=0.248661; routed=0.254518; max_routed=0.750022; gt0.65=366; attention/router_changed=288/0 |
| `qwen3_moe_audit_gated_candidate` | 0.451 | `False` | `True` | dominated_by=trust_region,expert_only,tail_trimmed,searched_no_gt065,layer_chunk,unified_mechanism,mechanistic_unified,subspace_scaled; nearest=trust_region@0.2449 | total=0.263844; routed=0.270383; max_routed=0.750042; gt0.65=1146; attention/router_changed=288/0 |
| `qwen3_moe_unified_route_guarded_candidate` | 0.040 | `False` | `True` | dominated_by=audit_gated,trust_region,expert_only,tail_trimmed,searched_no_gt065,layer_chunk,unified_mechanism,mechanistic_unified,subspace_scaled; nearest=audit_gated@0.4835 | total=0.285637; routed=0.293125; max_routed=1.327237; gt0.65=1156; attention/router_changed=288/0 |

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
| `expert_subspace_conflict_ablation` | `wait_for_source_controls` | `{"qwen3_moe_mechanistic_unified_candidate": "awaiting_source_controls", "qwen3_moe_subspace_scaled_candidate": "awaiting_source_controls"}` |
| `mechanistic_unified_optimizer` | `wait_for_source_controls` | `{"qwen3_moe_mechanistic_unified_candidate": "awaiting_source_controls", "qwen3_moe_unified_mechanism_candidate": "awaiting_source_controls"}` |

## Outputs

- `results/qwen3_moe_adaptive_eval_schedule/adaptive_schedule.csv`
- `results/qwen3_moe_adaptive_eval_schedule/mechanism_schedule.csv`
- `results/qwen3_moe_adaptive_eval_schedule/run_adaptive_eval.sh`
- `results/qwen3_moe_adaptive_eval_schedule/summary.json`
- `results/qwen3_moe_adaptive_eval_schedule/report.md`
