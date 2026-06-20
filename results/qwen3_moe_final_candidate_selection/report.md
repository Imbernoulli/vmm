# Qwen3 MoE Final Candidate Selection

这个 selector 在远端 vLLM eval 结果通过 bundle audit 后，对两个 source endpoint 和全部 same-shape Qwen3 MoE candidates 做最终选择。它不会只看内部 delta，也不会只看单个 unified 候选；candidate 必须同时通过 source dominance、task regression 和 checkpoint audit gate。

- Status: `awaiting_source_eval`
- Selected: `None`
- Reason: `Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection.`
- Sources complete: `False`
- Candidates complete: `False`
- Eligible candidates: `0/9`
- Uncertainty gate: `True`
- Paired prediction gate: `True`
- Paired alpha: `0.05`

| method | role | usable | audit | avg | avg CI | worst | worst CI | rel norm | dominated | conf dom | regressions | conf regressions | paired net | paired p | paired regressions | eligible |
| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |
| `source_qwen3_30b_instruct` | `source` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | n/a | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `source_qwen3_30b_coder` | `source` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | n/a | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_unified_route_guarded_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.285637 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_audit_gated_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.263844 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_trust_region_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.248661 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_expert_only_trust_region_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.246033 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.243145 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.247595 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_layer_chunk_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.243454 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_unified_mechanism_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.239629 | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_subspace_scaled_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.239529 | `` | `` | `` | `` | n/a | n/a | `` | `False` |

## Outputs

- `results/qwen3_moe_final_candidate_selection/selection_table.csv`
- `results/qwen3_moe_final_candidate_selection/summary.json`
- `results/qwen3_moe_final_candidate_selection/decision_rules.json`
