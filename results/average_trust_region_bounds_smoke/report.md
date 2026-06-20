# Average Trust-Region Bounds Smoke

This smoke verifies that the trust-region bound table is synchronized with the current Dense/MoE probes, router margin gate, mechanistic cap, and final selector state.

- Status: `passed`
- Assertions: `11/11`

| assertion | expected | actual | passed |
| --- | --- | --- | --- |
| `constraint_count_matches_table` | `11` | `11` | `True` |
| `passed_count_matches_status_table` | `2` | `2` | `True` |
| `rejected_count_matches_status_table` | `7` | `7` | `True` |
| `waiting_count_matches_status_table` | `2` | `2` | `True` |
| `dense_local_bound_rejects_full_task_vector` | `bound<1 and candidate_over_bound>1` | `bound=0.3415520413593599, over=2.927811515984651, status=reject_linear_task_vector_average` | `True` |
| `dense_summary_safe_uniform_lambda_matches_constraint` | `0.0` | `0.0` | `True` |
| `router_safe_lambda_matches_margin_probe` | `0.019725152295719042` | `0.019725152295719` | `True` |
| `router_midpoint_rejected_by_margin_bound` | `candidate_over_bound>1` | `candidate=0.5, allowed=0.019725152295719, over=25.34834674551614, status=reject_direct_router_average` | `True` |
| `mechanistic_cap_matches_candidate_summary` | `0.649` | `0.649` | `True` |
| `mechanistic_candidate_passes_cap_and_retention_bound` | `cap_passed` | `status=mechanistic_cap_and_retention_passed, max=0.649, cap=0.649, retention=0.9650345047849123` | `True` |
| `final_bound_status_matches_final_selector` | `awaiting_source_eval` | `final selector status = awaiting_source_eval; eligible candidates = 0/12; reason = Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection..` | `True` |

## Outputs

- `results/average_trust_region_bounds_smoke/trust_region_bounds_smoke_matrix.csv`
- `results/average_trust_region_bounds_smoke/summary.json`
- `results/average_trust_region_bounds_smoke/report.md`
