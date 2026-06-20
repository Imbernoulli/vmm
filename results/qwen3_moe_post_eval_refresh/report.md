# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified/final selector、mechanism attribution、feedback/mechanistic optimizer、unified average optimizer 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `16/16`
- Audit: `awaiting_eval` (`0/12` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/10` eligible)
- Attribution: `awaiting_eval` (`0/10` scored)
- Feedback optimizer: `awaiting_eval` (`0/4` scored, `0` changed groups)
- Mechanistic unified: `mechanistic_unified_candidate_ready` -> `s0.04_b1.65_h0.75_i0.75` (`retention=0.9620500430684525`, `violations=0`)
- Mechanistic evidence: `mechanistic_evidence_audit_ready` (`gradient_agreement=1.0`, `objective_improved=0.8989128361625024`)
- Unified average optimizer: `built_waiting_for_qwen3_vllm_eval` (top next experiment `budgeted_qwen3_moe_downstream_eval` / `blocked_on_gpu_vllm`)
- Unified selector rank gate in optimizer: confidence band `True`, rank mode `None`, band size `0`
- Unified optimizer ledger smoke: `passed` (`5/5` cases)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.44 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.39 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.41 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.42 |
| `build_feedback_optimizer` | `optimizer` | `passed` | 0 | 1.14 |
| `build_mechanistic_unified_candidate` | `optimizer` | `passed` | 0 | 2.66 |
| `audit_mechanistic_evidence` | `attribution` | `passed` | 0 | 1.39 |
| `build_unified_average_optimizer` | `optimizer` | `passed` | 0 | 0.44 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.65 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.48 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.48 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.41 |
| `build_feedback_optimizer_smoke` | `smoke` | `passed` | 0 | 0.47 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `passed` | 0 | 1.60 |
| `unified_average_optimizer_ledger_smoke` | `smoke` | `passed` | 0 | 0.44 |
| `collect_results` | `summary` | `passed` | 0 | 1.68 |

## Commands

- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/select_qwen3_moe_final_candidate.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_final_candidate_selection`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_feedback_optimizer`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --output-dir results/qwen3_moe_mechanistic_unified_candidate`
- `python scripts/audit_qwen3_moe_mechanistic_evidence.py --output-dir results/qwen3_moe_mechanistic_evidence_audit`
- `python scripts/build_unified_average_optimizer.py --output-dir results/unified_average_optimizer`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke`
- `python scripts/smoke_unified_average_optimizer_ledger.py --summary results/unified_average_optimizer/summary.json --output-dir results/unified_average_optimizer_ledger_smoke`
- `python scripts/collect_results.py`
