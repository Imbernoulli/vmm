# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、unified average optimizer、average method gate matrix、average trust-region bounds 和总汇总，避免手工漏跑或用到旧结果。

- Status: `planned`
- Plan only: `True`
- Steps passed: `0/19`
- Audit: `n/a` (`n/a/n/a` usable)
- Selection: `n/a` -> `n/a`
- Final selection: `n/a` -> `n/a` (`n/a/n/a` eligible)
- Attribution: `n/a` (`n/a/n/a` scored)
- Feedback optimizer: `n/a` (`n/a/n/a` scored, `n/a` changed groups)
- Mechanistic unified: `n/a` -> `n/a` (`retention=n/a`, `violations=n/a`)
- Mechanistic evidence: `n/a` (`gradient_agreement=n/a`, `objective_improved=n/a`)
- Unified average optimizer: `n/a` (top next experiment `n/a` / `n/a`)
- Unified selector rank gate in optimizer: confidence band `n/a`, rank mode `n/a`, band size `n/a`
- Unified optimizer ledger smoke: `n/a` (`n/a/n/a` cases)
- Average method gate matrix: `n/a` (`accepted_by_default=n/a`, `rejected=n/a`, `conditional=n/a`)
- Average method gate smoke: `n/a` (`n/a/n/a` assertions)
- Average trust-region bounds: `n/a` (`passed=n/a`, `rejected=n/a`, `waiting=n/a`); dense lambda bound `n/a`, router midpoint over safe bound `n/a`

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `planned` | None | 0.00 |
| `select_unified_result` | `selector` | `planned` | None | 0.00 |
| `select_final_candidate` | `selector` | `planned` | None | 0.00 |
| `attribute_mechanism_effects` | `attribution` | `planned` | None | 0.00 |
| `build_feedback_optimizer` | `optimizer` | `planned` | None | 0.00 |
| `build_mechanistic_unified_candidate` | `optimizer` | `planned` | None | 0.00 |
| `audit_mechanistic_evidence` | `attribution` | `planned` | None | 0.00 |
| `build_unified_average_optimizer` | `optimizer` | `planned` | None | 0.00 |
| `build_average_method_gate_matrix` | `optimizer` | `planned` | None | 0.00 |
| `build_average_trust_region_bounds` | `optimizer` | `planned` | None | 0.00 |
| `audit_eval_bundles_smoke` | `smoke` | `planned` | None | 0.00 |
| `select_unified_result_smoke` | `smoke` | `planned` | None | 0.00 |
| `select_final_candidate_smoke` | `smoke` | `planned` | None | 0.00 |
| `attribute_mechanism_effects_smoke` | `smoke` | `planned` | None | 0.00 |
| `build_feedback_optimizer_smoke` | `smoke` | `planned` | None | 0.00 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `planned` | None | 0.00 |
| `unified_average_optimizer_ledger_smoke` | `smoke` | `planned` | None | 0.00 |
| `average_method_gate_matrix_consistency_smoke` | `smoke` | `planned` | None | 0.00 |
| `collect_results` | `summary` | `planned` | None | 0.00 |

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
