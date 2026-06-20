# Qwen3 MoE Candidate Trust-Region Gate

这个 gate 把结构 probe 转成 final selector 可消费的候选级约束：旧 candidate 仍可作为机制 ablation 跑 vLLM，但只有满足 router freeze、attention freeze、strict routed tail cap 和结构 frontier 的 candidate 才能进入最终默认 average 选择。

- Status: `candidate_trust_region_gate_ready`
- Candidate final-selectable: `2/10`
- Ablation-only candidates: `8`
- Structural rejects: `0`
- Strict routed max cap: `0.6505`

| method | category | selectable | reasons | frontier | safety | routed max | routed >0.65 | attn changed | router changed |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3_moe_unified_route_guarded_candidate` | `ablation_only` | `False` | `shared_attention_changed,routed_max_delta_over_strict_cap,routed_tail_over_065,structurally_dominated` | `False` | 0.0200 | 1.3272 | 1156 | 288 | 0 |
| `qwen3_moe_audit_gated_candidate` | `ablation_only` | `False` | `shared_attention_changed,routed_max_delta_over_strict_cap,routed_tail_over_065,structurally_dominated` | `False` | 0.5035 | 0.7500 | 1146 | 288 | 0 |
| `qwen3_moe_trust_region_candidate` | `ablation_only` | `False` | `shared_attention_changed,routed_max_delta_over_strict_cap,routed_tail_over_065,structurally_dominated` | `False` | 0.7484 | 0.7500 | 366 | 288 | 0 |
| `qwen3_moe_expert_only_trust_region_candidate` | `ablation_only` | `False` | `routed_max_delta_over_strict_cap,routed_tail_over_065,structurally_dominated` | `False` | 0.8394 | 0.7500 | 366 | 0 | 0 |
| `qwen3_moe_tail_trimmed_expert_only_candidate` | `ablation_only` | `False` | `routed_tail_over_065,structurally_dominated` | `False` | 0.9315 | 0.6501 | 80 | 0 | 0 |
| `qwen3_moe_searched_no_gt065_max_retention_candidate` | `ablation_only` | `False` | `routed_tail_over_065,structurally_dominated` | `False` | 0.8769 | 0.6501 | 245 | 0 | 0 |
| `qwen3_moe_layer_chunk_candidate` | `ablation_only` | `False` | `routed_tail_over_065,structurally_dominated` | `False` | 0.9289 | 0.6500 | 89 | 0 | 0 |
| `qwen3_moe_unified_mechanism_candidate` | `ablation_only` | `False` | `structurally_dominated` | `False` | 0.9742 | 0.6438 | 0 | 0 | 0 |
| `qwen3_moe_mechanistic_unified_candidate` | `final_selectable_trust_region` | `True` | `` | `True` | 0.9886 | 0.6491 | 0 | 0 | 0 |
| `qwen3_moe_subspace_scaled_candidate` | `final_selectable_trust_region` | `True` | `` | `True` | 0.9893 | 0.6234 | 0 | 0 | 0 |

## Outputs

- `results/qwen3_moe_candidate_trust_region_gate/candidate_trust_region_gate.csv`
- `results/qwen3_moe_candidate_trust_region_gate/summary.json`
- `results/qwen3_moe_candidate_trust_region_gate/report.md`
