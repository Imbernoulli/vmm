# Qwen3 MoE Post-Eval Refresh

这个脚本在远端 vLLM eval 落盘后按固定顺序刷新 eval bundle audit、unified selector、mechanism attribution 和总汇总，避免手工漏跑或用到旧结果。

- Status: `planned`
- Plan only: `True`
- Steps passed: `0/7`
- Audit: `n/a` (`n/a/n/a` usable)
- Selection: `n/a` -> `n/a`
- Attribution: `n/a` (`n/a/n/a` scored)

| step | kind | status | returncode | seconds |
| --- | --- | --- | ---: | ---: |
| `audit_eval_bundles` | `gate` | `planned` | None | 0.00 |
| `select_unified_result` | `selector` | `planned` | None | 0.00 |
| `attribute_mechanism_effects` | `attribution` | `planned` | None | 0.00 |
| `audit_eval_bundles_smoke` | `smoke` | `planned` | None | 0.00 |
| `select_unified_result_smoke` | `smoke` | `planned` | None | 0.00 |
| `attribute_mechanism_effects_smoke` | `smoke` | `planned` | None | 0.00 |
| `collect_results` | `summary` | `planned` | None | 0.00 |

## Commands

- `python scripts/audit_qwen3_moe_eval_bundle.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_eval_bundle_audit`
- `python scripts/select_qwen3_moe_unified_result.py --gate-dir results/qwen3_moe_mechanism_eval_gate --output-dir results/qwen3_moe_unified_result_selection`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --gate-dir results/qwen3_moe_mechanism_eval_gate --audit-dir results/qwen3_moe_eval_bundle_audit --output-dir results/qwen3_moe_mechanism_effect_attribution`
- `python scripts/audit_qwen3_moe_eval_bundle.py --smoke-matrix --output-dir results/qwen3_moe_eval_bundle_audit_smoke`
- `python scripts/select_qwen3_moe_unified_result.py --smoke-matrix --output-dir results/qwen3_moe_unified_result_selection_smoke`
- `python scripts/attribute_qwen3_moe_mechanism_effects.py --smoke-matrix --output-dir results/qwen3_moe_mechanism_effect_attribution_smoke`
- `python scripts/collect_results.py`
