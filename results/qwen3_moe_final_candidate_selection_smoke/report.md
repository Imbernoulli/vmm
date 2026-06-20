# Qwen3 MoE Final Candidate Selector Smoke

- Status: `passed`
- Cases: `11/11`

| case | status | selected | eligible | rank mode | band | passed |
| --- | --- | --- | ---: | --- | ---: | --- |
| `candidate_win` | `select_candidate` | `qwen3_moe_tail_trimmed_expert_only_candidate` | 2 | `confidence_tie_band_structural` | 2 | `True` |
| `source_dominance` | `keep_source_endpoint` | `source_qwen3_30b_instruct` | 0 | `None` | 0 | `True` |
| `task_regression` | `keep_source_endpoint` | `source_qwen3_30b_instruct` | 0 | `None` | 0 | `True` |
| `uncertain_small_sample` | `awaiting_candidate_eval` | `source_qwen3_30b_instruct` | 0 | `None` | 0 | `True` |
| `paired_regression` | `awaiting_candidate_eval` | `source_qwen3_30b_instruct` | 0 | `None` | 0 | `True` |
| `paired_noisy_delta` | `provisional_candidate` | `qwen3_moe_tail_trimmed_expert_only_candidate` | 1 | `confidence_tie_band_structural` | 1 | `True` |
| `structural_tie_break` | `select_candidate` | `qwen3_moe_mechanistic_unified_candidate` | 2 | `confidence_tie_band_structural` | 2 | `True` |
| `confidence_structural_band` | `select_candidate` | `qwen3_moe_mechanistic_unified_candidate` | 2 | `confidence_tie_band_structural` | 2 | `True` |
| `confidence_separated_point_leader` | `select_candidate` | `qwen3_moe_tail_trimmed_expert_only_candidate` | 2 | `confidence_tie_band_structural` | 1 | `True` |
| `trust_region_gate` | `select_candidate` | `qwen3_moe_mechanistic_unified_candidate` | 2 | `confidence_tie_band_structural` | 2 | `True` |
| `partial` | `provisional_candidate` | `qwen3_moe_tail_trimmed_expert_only_candidate` | 1 | `confidence_tie_band_structural` | 1 | `True` |
