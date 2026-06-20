# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、mechanistic sensitivity、router-expert coupling、router-coupled candidate、router-coupled retention frontier、unified average optimizer、average method gate matrix、average trust-region bounds 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `44/44`
- Audit: `awaiting_eval` (`0/14` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/12` eligible)
- Candidate trust-region gate: `candidate_trust_region_gate_ready` (`2/12` final-selectable, `10` ablation-only)
- Eval budget queue: `ready_for_budgeted_remote_vllm_eval` (default `final`, final `4` methods / `6144` prompts, max examples `384`)
- Eval budget queue smoke: `passed` (`11/11` assertions)
- Attribution: `awaiting_eval` (`0/10` scored)
- Feedback optimizer: `awaiting_eval` (`0/4` scored, `0` changed groups)
- Mechanistic unified: `mechanistic_unified_candidate_ready` -> `s0.08_b1.65_h0.75_i0.75` (`retention=0.9650345047849123`, `violations=0`)
- Mechanistic evidence: `mechanistic_evidence_audit_ready` (`gradient_agreement=1.0`, `objective_improved=0.945260347129506`)
- Mechanistic sensitivity: `mechanistic_sensitivity_ready` (objective `no_category_prior` delta `0.0033572134591372538`, scale `no_subspace_conflict` shift `0.00861034446007345`)
- Router-expert coupling: `router_expert_coupling_active` (fragility->feature `0.6946899464712292`, fragility->shrink `0.5831173179896568`, shrink lift `0.013794858470413248`, top layer `L20`)
- Router-coupled candidate: `ablation_only_waiting_vllm` -> `router_q0.75_s0.0100_cap0.0100` (`retention=0.9619428055490395`, `retention_delta=-0.0030916992358726025`, `coupled_delta_reduction=0.0011349698502332028`)
- Router-coupled retention frontier: `direct_router_boundary_term_not_default` (`effect_fraction=0.010260661865575978`, candidates `146/770` pass default gate)
- Source-set complementarity: `source_dominated_not_averageable_as_final` (dominant `instruct`, frontier avg gain `0.0`, best observed gap `-0.06944444444444431`, complementary sets `2`)
- Average source-set optimizer: `probe_only_below_interference_budget` for `coder+thinking` (gain `0.0083333333333333` vs interference budget `0.0694444444444443`, surplus `-0.06111111111111099`, final-budget `0`, probe-only `2`)
- Qwen source discovery plan: `source_discovery_plan_ready` (top scenario `dense_7b_general_code_math_reasoning`, queue `measured_coder_thinking_endpoint_expansion`, additional gain needed `0.06111111111111099`, task blockers `gsm8k,humaneval,mmlu`, top task gap `gsm8k` / `no_task_frontier_gain` needs `0.0694444444444443`)
- Qwen source discovery eval plan: `source_discovery_vllm_eval_plan_ready` (`4` jobs, top `measured_coder_thinking_source_frontier`, tasks `gsm8k,humaneval_compile,mmlu,safety`, task names `passed_humaneval_compile_task_name`)
- Qwen source discovery served-model preflight: `static_preflight_ready_waiting_for_endpoint_model_list` (endpoint `not_requested`, required `12`, missing `0`, manifests `4/8`, blocker `Start the vLLM server and rerun this preflight with --base-url.`)
- Qwen source frontier eval feedback: `awaiting_vllm_source_frontier_results` (scored `0/4`, final candidates `0`, probe-only `0`, top `None` / `None`, surplus `None`, blocker `Run the planned vLLM source-frontier eval jobs, then rerun this feedback builder.`)
- Qwen source frontier eval feedback smoke: `smoke_passed` (passed `True`, scored `3/3`, final candidates `1`)
- Router calibration frontier: `router_calibration_frontier_ready` (`2/4` default, recommended `cap001,margin_profile`, blocker `baseline_eval,source_eval,candidate_eval,audit,group_validation,capacity_metrics`, nll `0.22142744874642561`, generation `0.03333333333333344`)
- HARC router stats: `harc_router_stats_missing_cache` (`0/0` routers, first-stage `0/15` `missing`, Hessian `None`, cov `None`)
- HARC router stats smoke: `harc_router_stats_ready` (`6/6` checks)
- HARC router solver: `harc_solver_missing_cache` (`0` delta tensors, KL `None` -> `None`, residual `None`)
- HARC router solver smoke: `harc_solver_ready` (`6/6` checks, KL `0.0318062212318182` -> `6.153513822937384e-06`)
- HARC router candidate: `harc_router_candidate_waiting_for_solver_delta` (checkpoint `False`, delta tensors `0`, checks `0/0`, action `collect_real_router_cache_then_rerun_harc_solver`)
- HARC router candidate smoke: `harc_router_candidate_materialized` (checkpoint `True`, checks `3/3`)
- HARC readiness gate: `harc_ready_for_curvature_collection_waiting_cache` (`6/6` preconditions, cache `missing`, top layer `L17` score `0.7379048037035392`, first-stage layers `15`, action `collect_hessian_covariance_router_stats_then_run_harc_solver`)
- HARC readiness smoke: `smoke_passed` (`4/4` cases)
- Unified average optimizer: `built_waiting_for_qwen3_vllm_eval` (top next experiment `budgeted_qwen3_moe_downstream_eval` / `blocked_on_gpu_vllm`)
- Unified algorithm contract: `blocked_on_downstream_eval` (`11/13` passed, blocking `['downstream_source_dominance_gate', 'final_unified_average_acceptance']`)
- Unified selector rank gate in optimizer: confidence band `True`, rank mode `None`, band size `0`
- Unified optimizer ledger smoke: `passed` (`5/5` cases)
- Average method gate matrix: `built_from_current_probe_evidence` (`accepted_by_default=0`, `rejected=1`, `conditional=3`)
- Average method gate smoke: `passed` (`5/5` assertions)
- Average trust-region bounds: `trust_region_bounds_ready_waiting_vllm` (`passed=2`, `rejected=7`, `waiting=2`); dense lambda bound `0.34155204135935996`, router midpoint over safe bound `25.34834674551614`
- Average trust-region smoke: `passed` (`11/11` assertions)
- Mechanism levers: `mechanism_leverage_map_ready` (top `source_task_gap_frontier_acquisition` -> `python scripts/run_vllm_downstream_eval.py --models SERVED_CODER,SERVED_THINKING --tasks gsm8k,humaneval_compile,mmlu --max-examples 256 --output-dir results/qwen_source_discovery_plan/measured_coder_thinking_vllm`, task blockers `3`, top task gap `gsm8k` / `no_task_frontier_gain` needs `0.0694444444444443`)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `build_candidate_trust_region_gate` | `gate` | `passed` | 0 | 0.40 |
| `plan_eval_budget` | `planner` | `passed` | 0 | 0.42 |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.42 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.42 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.41 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.41 |
| `build_feedback_optimizer` | `optimizer` | `passed` | 0 | 1.11 |
| `build_mechanistic_unified_candidate` | `optimizer` | `passed` | 0 | 2.59 |
| `audit_mechanistic_evidence` | `attribution` | `passed` | 0 | 1.43 |
| `analyze_mechanistic_sensitivity` | `attribution` | `passed` | 0 | 17.32 |
| `analyze_router_expert_coupling` | `attribution` | `passed` | 0 | 0.79 |
| `build_router_coupled_candidate` | `optimizer` | `passed` | 0 | 1.29 |
| `analyze_router_coupled_retention_frontier` | `attribution` | `passed` | 0 | 3.61 |
| `build_source_set_complementarity_gate` | `gate` | `passed` | 0 | 0.40 |
| `build_average_source_set_optimizer` | `optimizer` | `passed` | 0 | 0.43 |
| `build_qwen_source_discovery_plan` | `planner` | `passed` | 0 | 0.42 |
| `build_qwen_source_discovery_eval_plan` | `planner` | `passed` | 0 | 0.40 |
| `audit_qwen_source_discovery_served_model_preflight` | `gate` | `passed` | 0 | 0.46 |
| `build_qwen_source_frontier_eval_feedback` | `gate` | `passed` | 0 | 0.44 |
| `build_router_calibration_frontier` | `gate` | `passed` | 0 | 0.40 |
| `collect_qwen3_moe_harc_router_stats` | `probe` | `passed` | 0 | 2.06 |
| `solve_qwen3_moe_harc_router_delta` | `optimizer` | `passed` | 0 | 3.54 |
| `build_qwen3_moe_harc_router_candidate` | `optimizer` | `passed` | 0 | 3.52 |
| `build_qwen3_moe_harc_readiness_gate` | `gate` | `passed` | 0 | 0.42 |
| `build_unified_average_optimizer` | `optimizer` | `passed` | 0 | 0.48 |
| `build_average_method_gate_matrix` | `optimizer` | `passed` | 0 | 0.39 |
| `build_average_trust_region_bounds` | `optimizer` | `passed` | 0 | 0.43 |
| `analyze_mechanism_levers` | `attribution` | `passed` | 0 | 0.44 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.75 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.41 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.48 |
| `eval_budget_queue_smoke` | `smoke` | `passed` | 0 | 0.39 |
| `build_qwen_source_frontier_eval_feedback_smoke` | `smoke` | `passed` | 0 | 0.46 |
| `collect_qwen3_moe_harc_router_stats_smoke` | `smoke` | `passed` | 0 | 2.40 |
| `solve_qwen3_moe_harc_router_delta_smoke` | `smoke` | `passed` | 0 | 3.82 |
| `build_qwen3_moe_harc_router_candidate_smoke` | `smoke` | `passed` | 0 | 3.35 |
| `build_qwen3_moe_harc_readiness_gate_smoke` | `smoke` | `passed` | 0 | 0.51 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.41 |
| `build_feedback_optimizer_smoke` | `smoke` | `passed` | 0 | 0.49 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `passed` | 0 | 1.64 |
| `unified_average_optimizer_ledger_smoke` | `smoke` | `passed` | 0 | 0.44 |
| `average_method_gate_matrix_consistency_smoke` | `smoke` | `passed` | 0 | 0.39 |
| `average_trust_region_bounds_smoke` | `smoke` | `passed` | 0 | 0.40 |
| `collect_results` | `summary` | `passed` | 0 | 1.80 |

## Commands

- `python scripts/build_qwen3_moe_candidate_trust_region_gate.py --gate-plan results/qwen3_moe_mechanism_eval_gate/eval_gate_plan.csv --output-dir results/qwen3_moe_candidate_trust_region_gate`
- `python scripts/plan_qwen3_moe_eval_budget.py --gate-dir results/qwen3_moe_mechanism_eval_gate --candidate-trust-gate results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv --output-dir results/qwen3_moe_eval_budget_plan`
- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/select_qwen3_moe_final_candidate.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_final_candidate_selection --candidate-trust-gate results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_feedback_optimizer`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --output-dir results/qwen3_moe_mechanistic_unified_candidate`
- `python scripts/audit_qwen3_moe_mechanistic_evidence.py --output-dir results/qwen3_moe_mechanistic_evidence_audit`
- `python scripts/analyze_qwen3_moe_mechanistic_sensitivity.py --output-dir results/qwen3_moe_mechanistic_sensitivity`
- `python scripts/analyze_qwen3_moe_router_expert_coupling.py --output-dir results/qwen3_moe_router_expert_coupling`
- `python scripts/build_qwen3_moe_router_coupled_candidate.py --output-dir results/qwen3_moe_router_coupled_candidate`
- `python scripts/analyze_qwen3_moe_router_coupled_retention_frontier.py --output-dir results/qwen3_moe_router_coupled_retention_frontier`
- `python scripts/build_qwen3_source_set_complementarity_gate.py --output-dir results/qwen3_source_set_complementarity_gate`
- `python scripts/build_qwen3_average_source_set_optimizer.py --output-dir results/qwen3_average_source_set_optimizer`
- `python scripts/build_qwen_source_discovery_plan.py --output-dir results/qwen_source_discovery_plan`
- `python scripts/build_qwen_source_discovery_eval_plan.py --source-discovery-dir results/qwen_source_discovery_plan --output-dir results/qwen_source_discovery_eval_plan`
- `python scripts/audit_vllm_served_model_preflight.py --eval-jobs results/qwen_source_discovery_eval_plan/vllm_eval_jobs.csv --output-dir results/qwen_source_discovery_served_model_preflight`
- `python scripts/build_qwen_source_frontier_eval_feedback.py --eval-jobs results/qwen_source_discovery_eval_plan/vllm_eval_jobs.csv --average-source-set-optimizer results/qwen3_average_source_set_optimizer/summary.json --output-dir results/qwen_source_frontier_eval_feedback`
- `python scripts/build_qwen3_moe_router_calibration_frontier.py --output-dir results/qwen3_moe_router_calibration_frontier`
- `python scripts/collect_qwen3_moe_harc_router_stats.py --output-dir results/qwen3_moe_harc_router_stats`
- `python scripts/solve_qwen3_moe_harc_router_delta.py --output-dir results/qwen3_moe_harc_router_solver`
- `python scripts/build_qwen3_moe_harc_router_candidate.py --output-dir results/qwen3_moe_harc_router_candidate --solver-dir results/qwen3_moe_harc_router_solver --solver-summary results/qwen3_moe_harc_router_solver/summary.json`
- `python scripts/build_qwen3_moe_harc_readiness_gate.py --output-dir results/qwen3_moe_harc_readiness_gate --harc-stats-dir results/qwen3_moe_harc_router_stats --harc-stats-summary results/qwen3_moe_harc_router_stats/summary.json`
- `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer --qwen-source-discovery-plan results/qwen_source_discovery_plan/summary.json --qwen-source-discovery-eval-plan results/qwen_source_discovery_eval_plan/summary.json --qwen-source-frontier-eval-feedback results/qwen_source_frontier_eval_feedback/summary.json --qwen3-router-calibration-frontier results/qwen3_moe_router_calibration_frontier/summary.json`
- `python scripts/build_average_method_gate_matrix.py --output-dir results/average_method_gate_matrix --optimizer-summary results/unified_average_optimizer/summary.json --optimizer-features results/unified_average_optimizer/mechanism_features.csv`
- `python scripts/build_average_trust_region_bounds.py --output-dir results/average_trust_region_bounds`
- `python scripts/analyze_qwen3_moe_mechanism_levers.py --eval-budget-dir results/qwen3_moe_eval_budget_plan --qwen-source-discovery-plan results/qwen_source_discovery_plan/summary.json --qwen-source-task-gap-targets results/qwen_source_discovery_plan/task_gap_targets.csv --average-source-set-optimizer results/qwen3_average_source_set_optimizer/summary.json --output-dir results/qwen3_moe_mechanism_levers`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/smoke_qwen3_moe_eval_budget_queue.py --eval-budget-dir results/qwen3_moe_eval_budget_plan --candidate-trust-gate results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv --output-dir results/qwen3_moe_eval_budget_queue_smoke`
- `python scripts/build_qwen_source_frontier_eval_feedback.py --smoke-matrix --output-dir results/qwen_source_frontier_eval_feedback_smoke`
- `python scripts/collect_qwen3_moe_harc_router_stats.py --smoke-matrix --output-dir results/qwen3_moe_harc_router_stats_smoke`
- `python scripts/solve_qwen3_moe_harc_router_delta.py --smoke-matrix --output-dir results/qwen3_moe_harc_router_solver_smoke`
- `python scripts/build_qwen3_moe_harc_router_candidate.py --smoke-matrix --output-dir results/qwen3_moe_harc_router_candidate_smoke`
- `python scripts/build_qwen3_moe_harc_readiness_gate.py --smoke-matrix --output-dir results/qwen3_moe_harc_readiness_gate_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke`
- `python scripts/smoke_unified_average_optimizer_ledger.py --summary results/unified_average_optimizer/summary.json --output-dir results/unified_average_optimizer_ledger_smoke`
- `python scripts/smoke_average_method_gate_matrix.py --optimizer-summary results/unified_average_optimizer/summary.json --method-gate-dir results/average_method_gate_matrix --output-dir results/average_method_gate_matrix_consistency_smoke`
- `python scripts/smoke_average_trust_region_bounds.py --bounds-dir results/average_trust_region_bounds --output-dir results/average_trust_region_bounds_smoke`
- `python scripts/collect_results.py`
