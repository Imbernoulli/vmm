# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified selector、mechanism attribution 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `11/11`
- Audit: `awaiting_eval` (`0/11` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/9` eligible)
- Attribution: `awaiting_eval` (`0/9` scored)
- Feedback optimizer: `awaiting_eval` (`0/4` scored, `0` changed groups)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.41 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.42 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.39 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.39 |
| `build_feedback_optimizer` | `optimizer` | `passed` | 0 | 1.02 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.67 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.40 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.42 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.45 |
| `build_feedback_optimizer_smoke` | `smoke` | `passed` | 0 | 0.46 |
| `collect_results` | `summary` | `passed` | 0 | 1.55 |

## Commands

- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/select_qwen3_moe_final_candidate.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_final_candidate_selection`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_feedback_optimizer`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/build_qwen3_moe_feedback_optimizer.py --smoke-matrix --output-dir results/qwen3_moe_feedback_optimizer_smoke`
- `python scripts/collect_results.py`
