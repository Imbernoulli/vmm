# Qwen3 MoE Router Calibration Selector Matrix Smoke

这个 smoke 不是再造一个候选算法，而是把 router-calibration selector 的核心选择/拒绝边界变成一次性回归测试。

- Status: `passed`
- Cases: `6/6`

## Cases

| case | status | selected | eligible | reason marker | passed |
|---|---|---|---:|---|---|
| `positive_group_heldout` | `selected_router_calibrated_candidate` | `smoke_router_calibrated_cap0025` | 1 | `passes_all_gates` | `True` |
| `row_validation_rejected` | `awaiting_router_calibration_eval` | `None` | 0 | `router_validation_not_group_heldout` | `True` |
| `source_dominance_abstains` | `keep_frozen_router_baseline` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | 0 | `source_endpoint_dominates` | `True` |
| `no_downstream_gain_abstains` | `keep_frozen_router_baseline` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | 0 | `no_downstream_gain` | `True` |
| `task_regression_abstains` | `keep_frozen_router_baseline` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | 0 | `task_score_regression` | `True` |
| `manifest_mismatch_abstains` | `keep_frozen_router_baseline` | `qwen3_moe_searched_no_gt065_max_retention_candidate` | 0 | `task_manifest_sha_mismatch` | `True` |

## Outputs

- `results/qwen3_moe_router_calibration_selector_matrix_smoke/selector_matrix.csv`
- `results/qwen3_moe_router_calibration_selector_matrix_smoke/summary.json`
- `results/qwen3_moe_router_calibration_selector_matrix_smoke/report.md`
