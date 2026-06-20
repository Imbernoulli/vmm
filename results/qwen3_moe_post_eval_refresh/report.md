# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、unified average optimizer、average method gate matrix、average trust-region bounds 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `19/19`
- Audit: `awaiting_eval` (`0/12` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/10` eligible)
- Attribution: `awaiting_eval` (`0/10` scored)
- Feedback optimizer: `awaiting_eval` (`0/4` scored, `0` changed groups)
- Mechanistic unified: `mechanistic_unified_candidate_ready` -> `s0.08_b1.65_h0.75_i0.75` (`retention=0.9650345047849123`, `violations=0`)
- Mechanistic evidence: `mechanistic_evidence_audit_ready` (`gradient_agreement=1.0`, `objective_improved=0.945260347129506`)
- Unified average optimizer: `built_waiting_for_qwen3_vllm_eval` (top next experiment `budgeted_qwen3_moe_downstream_eval` / `blocked_on_gpu_vllm`)
- Unified selector rank gate in optimizer: confidence band `True`, rank mode `None`, band size `0`
- Unified optimizer ledger smoke: `passed` (`5/5` cases)
- Average method gate matrix: `built_from_current_probe_evidence` (`accepted_by_default=0`, `rejected=1`, `conditional=3`)
- Average method gate smoke: `passed` (`5/5` assertions)
- Average trust-region bounds: `trust_region_bounds_ready_waiting_vllm` (`passed=2`, `rejected=7`, `waiting=2`); dense lambda bound `0.34155204135935996`, router midpoint over safe bound `25.34834674551614`

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.45 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.48 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.55 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.50 |
| `build_feedback_optimizer` | `optimizer` | `passed` | 0 | 1.28 |
| `build_mechanistic_unified_candidate` | `optimizer` | `passed` | 0 | 3.11 |
| `audit_mechanistic_evidence` | `attribution` | `passed` | 0 | 1.62 |
| `build_unified_average_optimizer` | `optimizer` | `passed` | 0 | 0.50 |
| `build_average_method_gate_matrix` | `optimizer` | `passed` | 0 | 0.61 |
| `build_average_trust_region_bounds` | `optimizer` | `passed` | 0 | 0.53 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.62 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.41 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.50 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.42 |
| `build_feedback_optimizer_smoke` | `smoke` | `passed` | 0 | 0.48 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `passed` | 0 | 1.60 |
| `unified_average_optimizer_ledger_smoke` | `smoke` | `passed` | 0 | 0.41 |
| `average_method_gate_matrix_consistency_smoke` | `smoke` | `passed` | 0 | 0.43 |
| `collect_results` | `summary` | `passed` | 0 | 1.62 |

## Commands

- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/select_qwen3_moe_final_candidate.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_final_candidate_selection`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_feedback_optimizer`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --output-dir results/qwen3_moe_mechanistic_unified_candidate`
- `python scripts/audit_qwen3_moe_mechanistic_evidence.py --output-dir results/qwen3_moe_mechanistic_evidence_audit`
- `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer`
- `python scripts/build_average_method_gate_matrix.py --output-dir results/average_method_gate_matrix --optimizer-summary results/unified_average_optimizer/summary.json --optimizer-features results/unified_average_optimizer/mechanism_features.csv`
- `python scripts/build_average_trust_region_bounds.py --output-dir results/average_trust_region_bounds`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke`
- `python scripts/smoke_unified_average_optimizer_ledger.py --summary results/unified_average_optimizer/summary.json --output-dir results/unified_average_optimizer_ledger_smoke`
- `python scripts/smoke_average_method_gate_matrix.py --optimizer-summary results/unified_average_optimizer/summary.json --method-gate-dir results/average_method_gate_matrix --output-dir results/average_method_gate_matrix_consistency_smoke`
- `python scripts/collect_results.py`
