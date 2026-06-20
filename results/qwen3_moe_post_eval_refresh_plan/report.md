# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、mechanistic sensitivity、router-expert coupling、router-coupled candidate、router-coupled retention frontier、unified average optimizer、average method gate matrix、average trust-region bounds 和总汇总，避免手工漏跑或用到旧结果。

- Status: `planned`
- Plan only: `True`
- Steps passed: `0/42`
- Audit: `n/a` (`n/a/n/a` usable)
- Selection: `n/a` -> `n/a`
- Final selection: `n/a` -> `n/a` (`n/a/n/a` eligible)
- Candidate trust-region gate: `n/a` (`n/a/n/a` final-selectable, `n/a` ablation-only)
- Eval budget queue: `n/a` (default `n/a`, final `n/a` methods / `n/a` prompts, max examples `n/a`)
- Eval budget queue smoke: `n/a` (`n/a/n/a` assertions)
- Attribution: `n/a` (`n/a/n/a` scored)
- Feedback optimizer: `n/a` (`n/a/n/a` scored, `n/a` changed groups)
- Mechanistic unified: `n/a` -> `n/a` (`retention=n/a`, `violations=n/a`)
- Mechanistic evidence: `n/a` (`gradient_agreement=n/a`, `objective_improved=n/a`)
- Mechanistic sensitivity: `n/a` (objective `n/a` delta `n/a`, scale `n/a` shift `n/a`)
- Router-expert coupling: `n/a` (fragility->feature `n/a`, fragility->shrink `n/a`, shrink lift `n/a`, top layer `Ln/a`)
- Router-coupled candidate: `n/a` -> `n/a` (`retention=n/a`, `retention_delta=n/a`, `coupled_delta_reduction=n/a`)
- Router-coupled retention frontier: `n/a` (`effect_fraction=n/a`, candidates `n/a/n/a` pass default gate)
- Source-set complementarity: `n/a` (dominant `n/a`, frontier avg gain `n/a`, best observed gap `n/a`, complementary sets `n/a`)
- Average source-set optimizer: `n/a` for `n/a` (gain `n/a` vs interference budget `n/a`, surplus `n/a`, final-budget `n/a`, probe-only `n/a`)
- Qwen source discovery plan: `n/a` (top scenario `n/a`, queue `n/a`, additional gain needed `n/a`)
- Qwen source discovery eval plan: `n/a` (`n/a` jobs, top `n/a`, tasks `n/a`, task names `n/a`)
- Qwen source discovery served-model preflight: `n/a` (endpoint `n/a`, required `n/a`, missing `n/a`, manifests `n/a/n/a`, blocker `n/a`)
- Qwen source frontier eval feedback: `n/a` (scored `n/a/n/a`, final candidates `n/a`, probe-only `n/a`, top `n/a` / `n/a`, surplus `n/a`, blocker `n/a`)
- Qwen source frontier eval feedback smoke: `n/a` (passed `n/a`, scored `n/a/n/a`, final candidates `n/a`)
- Router calibration frontier: `n/a` (`n/a/n/a` default, recommended `n/a`, blocker `n/a`, nll `n/a`, generation `n/a`)
- HARC router stats: `n/a` (`n/a/n/a` routers, first-stage `n/a/n/a` `n/a`, Hessian `n/a`, cov `n/a`)
- HARC router stats smoke: `n/a` (`n/a/n/a` checks)
- HARC router solver: `n/a` (`n/a` delta tensors, KL `n/a` -> `n/a`, residual `n/a`)
- HARC router solver smoke: `n/a` (`n/a/n/a` checks, KL `n/a` -> `n/a`)
- HARC readiness gate: `n/a` (`n/a/n/a` preconditions, cache `n/a`, top layer `Ln/a` score `n/a`, first-stage layers `n/a`, action `n/a`)
- HARC readiness smoke: `n/a` (`n/a/n/a` cases)
- Unified average optimizer: `n/a` (top next experiment `n/a` / `n/a`)
- Unified algorithm contract: `n/a` (`n/a/n/a` passed, blocking `[]`)
- Unified selector rank gate in optimizer: confidence band `n/a`, rank mode `n/a`, band size `n/a`
- Unified optimizer ledger smoke: `n/a` (`n/a/n/a` cases)
- Average method gate matrix: `n/a` (`accepted_by_default=n/a`, `rejected=n/a`, `conditional=n/a`)
- Average method gate smoke: `n/a` (`n/a/n/a` assertions)
- Average trust-region bounds: `n/a` (`passed=n/a`, `rejected=n/a`, `waiting=n/a`); dense lambda bound `n/a`, router midpoint over safe bound `n/a`
- Average trust-region smoke: `n/a` (`n/a/n/a` assertions)
- Mechanism levers: `n/a` (top `n/a` -> `n/a`)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `build_candidate_trust_region_gate` | `gate` | `planned` | None | 0.00 |
| `plan_eval_budget` | `planner` | `planned` | None | 0.00 |
| `audit_eval_bundles` | `gate` | `planned` | None | 0.00 |
| `select_unified_result` | `selector` | `planned` | None | 0.00 |
| `select_final_candidate` | `selector` | `planned` | None | 0.00 |
| `attribute_mechanism_effects` | `attribution` | `planned` | None | 0.00 |
| `build_feedback_optimizer` | `optimizer` | `planned` | None | 0.00 |
| `build_mechanistic_unified_candidate` | `optimizer` | `planned` | None | 0.00 |
| `audit_mechanistic_evidence` | `attribution` | `planned` | None | 0.00 |
| `analyze_mechanistic_sensitivity` | `attribution` | `planned` | None | 0.00 |
| `analyze_router_expert_coupling` | `attribution` | `planned` | None | 0.00 |
| `build_router_coupled_candidate` | `optimizer` | `planned` | None | 0.00 |
| `analyze_router_coupled_retention_frontier` | `attribution` | `planned` | None | 0.00 |
| `build_source_set_complementarity_gate` | `gate` | `planned` | None | 0.00 |
| `build_average_source_set_optimizer` | `optimizer` | `planned` | None | 0.00 |
| `build_qwen_source_discovery_plan` | `planner` | `planned` | None | 0.00 |
| `build_qwen_source_discovery_eval_plan` | `planner` | `planned` | None | 0.00 |
| `audit_qwen_source_discovery_served_model_preflight` | `gate` | `planned` | None | 0.00 |
| `build_qwen_source_frontier_eval_feedback` | `gate` | `planned` | None | 0.00 |
| `build_router_calibration_frontier` | `gate` | `planned` | None | 0.00 |
| `collect_qwen3_moe_harc_router_stats` | `probe` | `planned` | None | 0.00 |
| `solve_qwen3_moe_harc_router_delta` | `optimizer` | `planned` | None | 0.00 |
| `build_qwen3_moe_harc_readiness_gate` | `gate` | `planned` | None | 0.00 |
| `build_unified_average_optimizer` | `optimizer` | `planned` | None | 0.00 |
| `build_average_method_gate_matrix` | `optimizer` | `planned` | None | 0.00 |
| `build_average_trust_region_bounds` | `optimizer` | `planned` | None | 0.00 |
| `analyze_mechanism_levers` | `attribution` | `planned` | None | 0.00 |
| `audit_eval_bundles_smoke` | `smoke` | `planned` | None | 0.00 |
| `select_unified_result_smoke` | `smoke` | `planned` | None | 0.00 |
| `select_final_candidate_smoke` | `smoke` | `planned` | None | 0.00 |
| `eval_budget_queue_smoke` | `smoke` | `planned` | None | 0.00 |
| `build_qwen_source_frontier_eval_feedback_smoke` | `smoke` | `planned` | None | 0.00 |
| `collect_qwen3_moe_harc_router_stats_smoke` | `smoke` | `planned` | None | 0.00 |
| `solve_qwen3_moe_harc_router_delta_smoke` | `smoke` | `planned` | None | 0.00 |
| `build_qwen3_moe_harc_readiness_gate_smoke` | `smoke` | `planned` | None | 0.00 |
| `attribute_mechanism_effects_smoke` | `smoke` | `planned` | None | 0.00 |
| `build_feedback_optimizer_smoke` | `smoke` | `planned` | None | 0.00 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `planned` | None | 0.00 |
| `unified_average_optimizer_ledger_smoke` | `smoke` | `planned` | None | 0.00 |
| `average_method_gate_matrix_consistency_smoke` | `smoke` | `planned` | None | 0.00 |
| `average_trust_region_bounds_smoke` | `smoke` | `planned` | None | 0.00 |
| `collect_results` | `summary` | `planned` | None | 0.00 |

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
- `python scripts/build_qwen3_moe_harc_readiness_gate.py --output-dir results/qwen3_moe_harc_readiness_gate --harc-stats-dir results/qwen3_moe_harc_router_stats --harc-stats-summary results/qwen3_moe_harc_router_stats/summary.json`
- `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer --qwen-source-discovery-plan results/qwen_source_discovery_plan/summary.json --qwen-source-discovery-eval-plan results/qwen_source_discovery_eval_plan/summary.json --qwen-source-frontier-eval-feedback results/qwen_source_frontier_eval_feedback/summary.json --qwen3-router-calibration-frontier results/qwen3_moe_router_calibration_frontier/summary.json`
- `python scripts/build_average_method_gate_matrix.py --output-dir results/average_method_gate_matrix --optimizer-summary results/unified_average_optimizer/summary.json --optimizer-features results/unified_average_optimizer/mechanism_features.csv`
- `python scripts/build_average_trust_region_bounds.py --output-dir results/average_trust_region_bounds`
- `python scripts/analyze_qwen3_moe_mechanism_levers.py --eval-budget-dir results/qwen3_moe_eval_budget_plan --output-dir results/qwen3_moe_mechanism_levers`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/smoke_qwen3_moe_eval_budget_queue.py --eval-budget-dir results/qwen3_moe_eval_budget_plan --candidate-trust-gate results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv --output-dir results/qwen3_moe_eval_budget_queue_smoke`
- `python scripts/build_qwen_source_frontier_eval_feedback.py --smoke-matrix --output-dir results/qwen_source_frontier_eval_feedback_smoke`
- `python scripts/collect_qwen3_moe_harc_router_stats.py --smoke-matrix --output-dir results/qwen3_moe_harc_router_stats_smoke`
- `python scripts/solve_qwen3_moe_harc_router_delta.py --smoke-matrix --output-dir results/qwen3_moe_harc_router_solver_smoke`
- `python scripts/build_qwen3_moe_harc_readiness_gate.py --smoke-matrix --output-dir results/qwen3_moe_harc_readiness_gate_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke`
- `python scripts/smoke_unified_average_optimizer_ledger.py --summary results/unified_average_optimizer/summary.json --output-dir results/unified_average_optimizer_ledger_smoke`
- `python scripts/smoke_average_method_gate_matrix.py --optimizer-summary results/unified_average_optimizer/summary.json --method-gate-dir results/average_method_gate_matrix --output-dir results/average_method_gate_matrix_consistency_smoke`
- `python scripts/smoke_average_trust_region_bounds.py --bounds-dir results/average_trust_region_bounds --output-dir results/average_trust_region_bounds_smoke`
- `python scripts/collect_results.py`
