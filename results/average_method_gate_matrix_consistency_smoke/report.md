# Average Method Gate Matrix Consistency Smoke

This smoke verifies that the method gate matrix is synchronized with the current unified optimizer and final selector state.

- Status: `passed`
- Assertions: `5/5`

| assertion | expected | actual | passed |
| --- | --- | --- | --- |
| `final_gate_threshold_matches_optimizer_candidate_count` | `11` | `11.0` | `True` |
| `final_gate_value_matches_optimizer_eligible_count` | `0` | `0.0` | `True` |
| `final_gate_status_matches_optimizer_status` | `awaiting_source_eval` | `awaiting_source_eval` | `True` |
| `no_method_family_accepted_by_default` | `0` | `0` | `True` |
| `default_rejected_count_matches_table` | `1` | `1` | `True` |

## Outputs

- `results/average_method_gate_matrix_consistency_smoke/consistency_matrix.csv`
- `results/average_method_gate_matrix_consistency_smoke/summary.json`
- `results/average_method_gate_matrix_consistency_smoke/report.md`
