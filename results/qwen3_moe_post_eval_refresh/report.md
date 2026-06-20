# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified selector、mechanism attribution 和总汇总，避免手工漏跑或用到旧结果。

- Status: `passed`
- Plan only: `False`
- Steps passed: `9/9`
- Audit: `awaiting_eval` (`0/10` usable)
- Selection: `awaiting_source_eval` -> `None`
- Final selection: `awaiting_source_eval` -> `None` (`0/8` eligible)
- Attribution: `awaiting_eval` (`0/8` scored)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `passed` | 0 | 0.45 |
| `select_unified_result` | `selector` | `passed` | 0 | 0.50 |
| `select_final_candidate` | `selector` | `passed` | 0 | 0.47 |
| `attribute_mechanism_effects` | `attribution` | `passed` | 0 | 0.43 |
| `audit_eval_bundles_smoke` | `smoke` | `passed` | 0 | 0.73 |
| `select_unified_result_smoke` | `smoke` | `passed` | 0 | 0.48 |
| `select_final_candidate_smoke` | `smoke` | `passed` | 0 | 0.52 |
| `attribute_mechanism_effects_smoke` | `smoke` | `passed` | 0 | 0.50 |
| `collect_results` | `summary` | `passed` | 0 | 1.58 |

## Commands

- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/select_qwen3_moe_final_candidate.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_final_candidate_selection`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/select_qwen3_moe_final_candidate.py --smoke-matrix --output-dir results/qwen3_moe_final_candidate_selection_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/collect_results.py`
