# Qwen3 MoE Final Candidate Selector Smoke

- Status: `passed`
- Cases: `6/6`

| case | status | selected | eligible | passed |
| --- | --- | --- | ---: | --- |
| `candidate_win` | `select_candidate` | `qwen3_moe_tail_trimmed_expert_only_candidate` | 2 | `True` |
| `source_dominance` | `keep_source_endpoint` | `source_qwen3_30b_instruct` | 0 | `True` |
| `task_regression` | `keep_source_endpoint` | `source_qwen3_30b_instruct` | 0 | `True` |
| `uncertain_small_sample` | `awaiting_candidate_eval` | `source_qwen3_30b_instruct` | 0 | `True` |
| `paired_regression` | `awaiting_candidate_eval` | `source_qwen3_30b_instruct` | 0 | `True` |
| `partial` | `provisional_candidate` | `qwen3_moe_tail_trimmed_expert_only_candidate` | 1 | `True` |
