# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified selector、mechanism attribution 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `13/13`
- Audit: `awaiting_eval` (`0/12` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/10` eligible)
- Attribution: `awaiting_eval` (`0/10` scored)
- Feedback optimizer: `awaiting_eval` (`0/4` scored, `0` changed groups)
- Mechanistic unified: `mechanistic_unified_candidate_ready` -> `s0.16_b1.65_h1.25_i1.00` (`retention=0.9653481845843201`, `violations=0`)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.44 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.41 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.40 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.40 |
| `build_feedback_optimizer` | `optimizer` | `passed` | 0 | 1.09 |
| `build_mechanistic_unified_candidate` | `optimizer` | `passed` | 0 | 2.74 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.77 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.47 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.44 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.42 |
| `build_feedback_optimizer_smoke` | `smoke` | `passed` | 0 | 0.51 |
| `build_mechanistic_unified_candidate_smoke` | `smoke` | `passed` | 0 | 1.68 |
| `collect_results` | `summary` | `passed` | 0 | 1.64 |

## Commands

- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/select_qwen3_moe_final_candidate.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_final_candidate_selection`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_feedback_optimizer`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --output-dir results/qwen3_moe_mechanistic_unified_candidate`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/build_qwen3_moe_mechanistic_unified_candidate.py --smoke-matrix --output-dir results/qwen3_moe_mechanistic_unified_candidate_smoke`
- `python scripts/collect_results.py`
