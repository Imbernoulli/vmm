# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、mechanistic sensitivity、router-expert coupling、router-coupled candidate、router-coupled retention frontier、unified average optimizer、average method gate matrix、average trust-region bounds 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `30/30`
- Audit: `awaiting_eval` (`0/13` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/11` eligible)
- Candidate trust-region gate: `candidate_trust_region_gate_ready` (`2/11` final-selectable, `9` ablation-only)
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
- Unified average optimizer: `built_waiting_for_qwen3_vllm_eval` (top next experiment `budgeted_qwen3_moe_downstream_eval` / `blocked_on_gpu_vllm`)
- Unified algorithm contract: `blocked_on_downstream_eval` (`11/13` passed, blocking `['downstream_source_dominance_gate', 'final_unified_average_acceptance']`)
- Unified selector rank gate in optimizer: confidence band `True`, rank mode `None`, band size `0`
- Unified optimizer ledger smoke: `passed` (`5/5` cases)
- Average method gate matrix: `built_from_current_probe_evidence` (`accepted_by_default=0`, `rejected=1`, `conditional=3`)
- Average method gate smoke: `passed` (`5/5` assertions)
- Average trust-region bounds: `trust_region_bounds_ready_waiting_vllm` (`passed=2`, `rejected=7`, `waiting=2`); dense lambda bound `0.34155204135935996`, router midpoint over safe bound `25.34834674551614`
- Average trust-region smoke: `passed` (`11/11` assertions)
- Mechanism levers: `mechanism_leverage_map_ready` (top `source_and_candidate_downstream_eval` -> `results/qwen3_moe_eval_budget_plan/run_eval_budget.sh final`)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `build_candidate_trust_region_gate` | `gate` | `passed` | 0 | 0.39 |
| `plan_eval_budget` | `planner` | `passed` | 0 | 0.48 |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.57 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.50 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.55 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.47 |
| `build_feedback_optimizer` | `optimizer` | `passed` | 0 | 1.15 |
| `build_mechanistic_unified_candidate` | `optimizer` | `passed` | 0 | 2.79 |
| `audit_mechanistic_evidence` | `attribution` | `passed` | 0 | 1.53 |
| `analyze_mechanistic_sensitivity` | `attribution` | `passed` | 0 | 17.04 |
| `analyze_router_expert_coupling` | `attribution` | `passed` | 0 | 0.80 |
| `build_router_coupled_candidate` | `optimizer` | `passed` | 0 | 1.35 |
| `analyze_router_coupled_retention_frontier` | `attribution` | `passed` | 0 | 3.56 |
| `build_source_set_complementarity_gate` | `gate` | `passed` | 0 | 0.40 |
| `build_average_source_set_optimizer` | `optimizer` | `passed` | 0 | 0.44 |
| `build_unified_average_optimizer` | `optimizer` | `passed` | 0 | 0.47 |
| `build_average_method_gate_matrix` | `optimizer` | `passed` | 0 | 0.42 |
| `build_average_trust_region_bounds` | `optimizer` | `passed` | 0 | 0.44 |
| `analyze_mechanism_levers` | `attribution` | `passed` | 0 | 0.43 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.69 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.40 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.52 |
| `eval_budget_queue_smoke` | `smoke` | `passed` | 0 | 0.43 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.44 |
| `build_feedback_optimizer_smoke` | `smoke` | `passed` | 0 | 0.48 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `passed` | 0 | 1.67 |
| `unified_average_optimizer_ledger_smoke` | `smoke` | `passed` | 0 | 0.82 |
| `average_method_gate_matrix_consistency_smoke` | `smoke` | `passed` | 0 | 0.60 |
| `average_trust_region_bounds_smoke` | `smoke` | `passed` | 0 | 0.69 |
| `collect_results` | `summary` | `passed` | 0 | 1.72 |

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
- `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer`
- `python scripts/build_average_method_gate_matrix.py --output-dir results/average_method_gate_matrix --optimizer-summary results/unified_average_optimizer/summary.json --optimizer-features results/unified_average_optimizer/mechanism_features.csv`
- `python scripts/build_average_trust_region_bounds.py --output-dir results/average_trust_region_bounds`
- `python scripts/analyze_qwen3_moe_mechanism_levers.py --eval-budget-dir results/qwen3_moe_eval_budget_plan --output-dir results/qwen3_moe_mechanism_levers`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/smoke_qwen3_moe_eval_budget_queue.py --eval-budget-dir results/qwen3_moe_eval_budget_plan --candidate-trust-gate results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv --output-dir results/qwen3_moe_eval_budget_queue_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke`
- `python scripts/smoke_unified_average_optimizer_ledger.py --summary results/unified_average_optimizer/summary.json --output-dir results/unified_average_optimizer_ledger_smoke`
- `python scripts/smoke_average_method_gate_matrix.py --optimizer-summary results/unified_average_optimizer/summary.json --method-gate-dir results/average_method_gate_matrix --output-dir results/average_method_gate_matrix_consistency_smoke`
- `python scripts/smoke_average_trust_region_bounds.py --bounds-dir results/average_trust_region_bounds --output-dir results/average_trust_region_bounds_smoke`
- `python scripts/collect_results.py`
