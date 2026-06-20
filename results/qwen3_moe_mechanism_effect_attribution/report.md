# Qwen3 MoE Mechanism Effect Attribution

这个 attribution 把 Qwen3 MoE average 的机制链条拆成相邻对比：source frontier、route-guarded、audit-gated、trust-region、expert-only、tail-trimmed、searched cap-law 和 unified alias。只有通过 eval bundle audit 的 vLLM 结果才会进入 downstream score delta。

- Status: `awaiting_eval`
- Scored transitions: `0/8`
- Improving transitions: `0`
- Regressing transitions: `0`
- Best avg transition: `None` (`None`)
- Best worst transition: `None` (`None`)

| transition | status | avg delta | worst delta | routed >0.75 delta | attention changed delta | effect |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `source_frontier_to_route_guarded` | `awaiting_eval` | n/a | n/a | n/a | n/a | `awaiting_downstream_eval` |
| `route_guarded_to_audit_gated` | `awaiting_eval` | n/a | n/a | -675 | 0 | `structural_change_only:routed_tensors_gt_0_75:-675,routed_tensors_gt_0_65:-10` |
| `audit_gated_to_trust_region` | `awaiting_eval` | n/a | n/a | -150 | 0 | `structural_change_only:routed_tensors_gt_0_75:-150,routed_tensors_gt_0_65:-780` |
| `trust_region_to_expert_only` | `awaiting_eval` | n/a | n/a | 0 | -288 | `structural_change_only:attention_changed_tensors:-288` |
| `expert_only_to_tail_trimmed` | `awaiting_eval` | n/a | n/a | -14 | 0 | `structural_change_only:routed_tensors_gt_0_75:-14,routed_tensors_gt_0_65:-286` |
| `tail_trimmed_to_searched_cap_law` | `awaiting_eval` | n/a | n/a | 0 | 0 | `structural_change_only:routed_tensors_gt_0_65:+165` |
| `searched_cap_law_to_layer_chunk` | `awaiting_eval` | n/a | n/a | 0 | 0 | `structural_change_only:routed_tensors_gt_0_65:-158` |
| `searched_cap_law_to_unified_alias` | `awaiting_eval` | n/a | n/a | 0 | 0 | `awaiting_downstream_eval` |

## Outputs

- `results/qwen3_moe_mechanism_effect_attribution/transition_effects.csv`
- `results/qwen3_moe_mechanism_effect_attribution/summary.json`
