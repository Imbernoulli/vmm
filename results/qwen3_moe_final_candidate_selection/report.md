# Qwen3 MoE Final Candidate Selection

这个 selector 在远端 vLLM eval 结果通过 bundle audit 后，对两个 source endpoint 和全部 same-shape Qwen3 MoE candidates 做最终选择。它不会只看内部 delta，也不会只看单个 unified 候选；candidate 必须同时通过 source dominance、task regression 和 checkpoint audit gate。通过 gate 后，结构 frontier / dominance 信号只作为分数打平或 tie-band 内的机制 tie-break，用来表达“为什么这个 average 更稳”，而不是替代下游评测。

- Status: `awaiting_source_eval`
- Selected: `None`
- Reason: `Both Qwen3 source endpoints must complete audited vLLM eval before final candidate selection.`
- Sources complete: `False`
- Candidates complete: `False`
- Eligible candidates: `0/10`
- Uncertainty gate: `True`
- Paired prediction gate: `True`
- Paired alpha: `0.05`
- Selection score tie tolerance: `0.0`
- Confidence tie band: `True`
- Selection rank mode: `None`
- Selection point leader: `None`
- Selection rank band size: `0`
- Structural dominance available: `True`
- Structural-frontier eligible candidates: `0`

| method | role | usable | audit | avg | avg CI | worst | worst CI | rel norm | struct frontier | struct dom | struct safety | nearest struct | rank band | dominated | conf dom | regressions | conf regressions | paired net | paired p | paired regressions | eligible |
| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| `source_qwen3_30b_instruct` | `source` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | n/a | `n/a` | `n/a` | n/a | `n/a` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `source_qwen3_30b_coder` | `source` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | n/a | `n/a` | `n/a` | n/a | `n/a` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_unified_route_guarded_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.285637 | `False` | `True` | 0.02 | `audit_gated` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_audit_gated_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.263844 | `False` | `True` | 0.503498 | `trust_region` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_trust_region_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.248661 | `False` | `True` | 0.748353 | `expert_only` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_expert_only_trust_region_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.246033 | `False` | `True` | 0.839437 | `searched_no_gt065` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.243145 | `False` | `True` | 0.931465 | `layer_chunk` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.247595 | `False` | `True` | 0.876943 | `layer_chunk` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_layer_chunk_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.243454 | `False` | `True` | 0.928945 | `tail_trimmed` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_unified_mechanism_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.240378 | `False` | `True` | 0.974168 | `subspace_scaled` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_mechanistic_unified_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.238226 | `True` | `False` | 0.988644 | `unified_mechanism` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |
| `qwen3_moe_subspace_scaled_candidate` | `candidate` | `False` | `True` | n/a | [n/a, n/a] | n/a | [n/a, n/a] | 0.239529 | `True` | `False` | 0.989299 | `unified_mechanism` | `False` | `` | `` | `` | `` | n/a | n/a | `` | `False` |

## Outputs

- `results/qwen3_moe_final_candidate_selection/selection_table.csv`
- `results/qwen3_moe_final_candidate_selection/summary.json`
- `results/qwen3_moe_final_candidate_selection/decision_rules.json`
