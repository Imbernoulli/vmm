# Average Method Gate Matrix

这个结果把常见 Dense/MoE averaging 方法从“方法名列表”改成当前证据下的执行门槛。每个方法族都必须说明：在 Dense 里是否允许、在 MoE 里是否允许、阻塞 probe 是什么，以及同构 checkpoint 约束下能做什么。

## Current Takeaway

- Accepted-by-default method families: `0`
- Conditional method families: `3`
- Active but still gated method families: `1`
- Required precondition method families: `2`
- Rejected-as-default method families: `1`
- Qwen3 final selector status: `awaiting_source_eval`

## Method Gates

| method family | gate | dense status | MoE status | allowed action | blocking probe |
| --- | --- | --- | --- | --- | --- |
| `Uniform / linear average` | `rejected_as_default` | `reject_midpoint_for_current_qwen_dense_pair` | `reject_source_to_source_midpoint_for_current_qwen3_moe_pair` | Use only as a negative baseline; do not materialize as the selected average. | `connectivity_path_must_beat_endpoint_frontier` |
| `Task arithmetic / coefficient search` | `conditional_allowed` | `allowed_only_with_heldout_lambda_selection` | `allowed_only_as_layer_expert_coefficient_rules` | Search coefficients under endpoint fallback and same-shape writer constraints. | `heldout_loss_or_vllm_gate_must_accept_coefficients` |
| `Sign / sparsity conflict methods` | `conditional_diagnostic` | `diagnostic_until_it_beats_anchor` | `expert_ffn_only_after_alignment_never_raw_router` | Emit tensor rules only for modules that pass held-out and vLLM gates. | `sparse_rule_must_improve_worst_task_and_preserve_critical_tensors` |
| `Importance / activation-aware average` | `conditional_needs_calibration_match` | `fisher_is_not_sufficient_on_current_qwen_dense_probe` | `needs_route_conditioned_sensitivity_before_acceptance` | Use Fisher/RegMean-style weights only after activation or NLL sensitivity predicts held-out behavior. | `activation_covariance_or_fisher_prediction_must_match_actual_path_loss` |
| `Output-space calibrated average` | `active_moe_lever_pending_downstream_gate` | `available_when_output_residual_is_measured` | `router_route_kd_active_but_not_accepted_yet` | Train or fit output/router residual corrections, then require source/baseline vLLM dominance gates. | `matched_baseline_source_candidate_vllm_eval_required` |
| `Alignment before averaging` | `required_precondition` | `required_when_barrier_or_permutation_symmetry_is_detected` | `required_for_expert_gauge_qwen3_identity_currently_passes` | Canonicalize or remap features/experts before any average; Qwen3 can use identity expert mapping for this pair. | `expert_or_feature_matching_must_be_known_before_tensor_average` |
| `Router-aware MoE average` | `required_for_moe` | `not_applicable` | `freeze_router_or_train_audited_route_kd_delta` | Freeze routers for expert candidate generation; treat router delta as separately audited ablation. | `hard_route_load_capacity_and_matched_vllm_eval_must_pass` |

## Probe Gates

| gate | probe | value | threshold | status | evidence |
| --- | --- | ---: | ---: | --- | --- |
| `dense_midpoint_rejection` | `dense_lambda_connectivity` | 6.0398 | 3.0727 | `failed_for_current_dense_pair` | linear midpoint worst NLL = 6.0398; best lambda-family worst NLL = 3.0727; best config = {"code": 0.5414439073133305, "general": 3.072743579141455, "lambda": 0.0, "mode": "uniform", "worst": 3.072743579141455}; best source endpoint worst NLL = 3.1737 |
| `moe_source_to_source_interpolation` | `qwen3_straight_line_connectivity` | 0.1189 | 0.0000 | `failed_for_current_qwen3_pair` | best interior worst NLL = 2.5947; best endpoint worst NLL = 2.4757; interior gap = 0.1189; general barrier = 0.1097; task-vector cosine vs base = 0.1799 |
| `moe_complementarity_claim` | `qwen3_complementary_pair_connectivity` | 0.5659 | 0.5659 | `not_supported_by_current_probe` | best merge avg NLL = 0.5659; best source avg NLL = 0.5659; merge-source gap = 0.0000; best merge t = 0.0000; merge beats both sources = False |
| `expert_gauge_alignment` | `real_olmoe_gauge_selfmerge` | 5.4910 | 0.0000 | `alignment_required` | baseline NLL = 4.1678; same-name average NLL = 9.6588; aligned average NLL = 4.1678; layers recovered = 16/16 |
| `router_movement` | `qwen3_router_move_gate` | 0.0000 | 48.0000 | `failed_freeze_or_calibrate` | allowed router layers = 0/48; top-k Jaccard mean/min = 0.4539/0.2422; top1 agreement mean/min = 0.4125/0.0690 |
| `router_coupled_direct_shrink` | `qwen3_router_coupled_retention_frontier` | 0.0103 | 0.0500 | `direct_router_boundary_term_not_default` | gate = direct_router_boundary_term_not_default; default-gate candidates = 146/770; constrained = router_q0.85_s0.00020_cap0.00010 retention_delta -0.000032 coupled_reduction 0.00001165; stress = router_q0.75_s0.01000_cap0.01000 coupled_reduction 0.001135; effect fraction = 0.0103 |
| `router_calibration` | `qwen3_router_calibration_nll_probe` | 0.2214 | 0.0000 | `awaiting_baseline_eval` | status = router_calibration_improves_linear_merge_but_needs_downstream_gate; linear worst NLL = 2.6355; router-cal worst NLL = 2.4140; worst reduction = 0.2214; code gap to best source = -0.0139 |
| `final_downstream_acceptance` | `qwen3_final_candidate_selection` | 0.0000 | 12.0000 | `awaiting_source_eval` | status = awaiting_source_eval; eligible candidates = 0/12; reason = Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection. |

## Outputs

- `results/average_method_gate_matrix/method_gate_matrix.csv`
- `results/average_method_gate_matrix/probe_gate_matrix.csv`
- `results/average_method_gate_matrix/summary.json`
- `results/average_method_gate_matrix/report.md`
