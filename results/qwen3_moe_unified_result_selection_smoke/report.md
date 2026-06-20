# Qwen3 MoE Unified Result Selector Smoke

- Status: `passed`
- Cases: `4/4`

| case | status | reason | selected | passed |
| --- | --- | --- | --- | --- |
| `candidate_win` | `select_unified_candidate` | `unified_improves_source_frontier` | `qwen3_moe_unified_mechanism_candidate` | `True` |
| `source_dominance` | `keep_source_endpoint` | `source_endpoint_dominates` | `source_qwen3_30b_instruct` | `True` |
| `task_regression` | `keep_source_endpoint` | `task_score_regression` | `source_qwen3_30b_instruct` | `True` |
| `no_gain` | `keep_source_endpoint` | `no_avg_or_worst_gain` | `source_qwen3_30b_instruct` | `True` |
