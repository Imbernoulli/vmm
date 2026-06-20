# Unified Average Optimizer Ledger Smoke

This smoke matrix verifies that the mechanism evidence ledger and next-experiment queue change when downstream or router-selection evidence changes. It guards against treating structural/NLL probes as final acceptance.

- Status: `passed`
- Passed cases: `5/5`

## Ledger Verdicts

| case | hypothesis | expected | actual | passed |
| --- | --- | --- | --- | --- |
| `awaiting_eval_keeps_provisional` | `moe_risk_weighted_expert_caps_preserve_useful_route_mass` | `awaiting_downstream_eval` | `awaiting_downstream_eval` | `True` |
| `awaiting_eval_keeps_provisional` | `downstream_source_dominance_is_final_gate` | `awaiting_downstream_eval` | `awaiting_downstream_eval` | `True` |
| `awaiting_eval_keeps_provisional` | `router_calibration_repairs_dispatch_but_is_not_acceptance` | `promising_but_unaccepted` | `promising_but_unaccepted` | `True` |
| `unified_candidate_downstream_win` | `moe_risk_weighted_expert_caps_preserve_useful_route_mass` | `supports_current_action` | `supports_current_action` | `True` |
| `unified_candidate_downstream_win` | `downstream_source_dominance_is_final_gate` | `supports_current_action` | `supports_current_action` | `True` |
| `source_endpoint_dominates` | `moe_risk_weighted_expert_caps_preserve_useful_route_mass` | `falsified_by_downstream_eval` | `falsified_by_downstream_eval` | `True` |
| `source_endpoint_dominates` | `downstream_source_dominance_is_final_gate` | `supports_source_fallback` | `supports_source_fallback` | `True` |
| `router_calibration_selected` | `router_calibration_repairs_dispatch_but_is_not_acceptance` | `supports_current_action` | `supports_current_action` | `True` |
| `router_calibration_rejected` | `router_calibration_repairs_dispatch_but_is_not_acceptance` | `supports_freeze_router_baseline` | `supports_freeze_router_baseline` | `True` |

## Queue Assertions

| case | assertion | expected | actual | passed |
| --- | --- | --- | --- | --- |
| `awaiting_eval_keeps_provisional` | `top_experiment` | `budgeted_qwen3_moe_downstream_eval` | `budgeted_qwen3_moe_downstream_eval` | `True` |
| `awaiting_eval_keeps_provisional` | `budgeted_qwen3_moe_downstream_eval.status` | `blocked_on_gpu_vllm` | `blocked_on_gpu_vllm` | `True` |
| `awaiting_eval_keeps_provisional` | `router_calibration_active_candidates.status` | `blocked_on_gpu_vllm` | `blocked_on_gpu_vllm` | `True` |
| `unified_candidate_downstream_win` | `top_experiment` | `mechanism_effect_attribution_refresh` | `mechanism_effect_attribution_refresh` | `True` |
| `unified_candidate_downstream_win` | `budgeted_qwen3_moe_downstream_eval.status` | `completed_by_selector` | `completed_by_selector` | `True` |
| `source_endpoint_dominates` | `top_experiment` | `mechanism_effect_attribution_refresh` | `mechanism_effect_attribution_refresh` | `True` |
| `source_endpoint_dominates` | `budgeted_qwen3_moe_downstream_eval.status` | `completed_source_fallback` | `completed_source_fallback` | `True` |
| `router_calibration_selected` | `top_experiment` | `budgeted_qwen3_moe_downstream_eval` | `budgeted_qwen3_moe_downstream_eval` | `True` |
| `router_calibration_selected` | `router_calibration_active_candidates.status` | `completed_by_selector` | `completed_by_selector` | `True` |
| `router_calibration_rejected` | `top_experiment` | `budgeted_qwen3_moe_downstream_eval` | `budgeted_qwen3_moe_downstream_eval` | `True` |
| `router_calibration_rejected` | `router_calibration_active_candidates.status` | `rejected_by_selector` | `rejected_by_selector` | `True` |

## Outputs

- `results/unified_average_optimizer_ledger_smoke/ledger_matrix.csv`
- `results/unified_average_optimizer_ledger_smoke/queue_matrix.csv`
- `results/unified_average_optimizer_ledger_smoke/summary.json`
- `results/unified_average_optimizer_ledger_smoke/report.md`
