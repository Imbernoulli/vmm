# Qwen3 MoE Audit-Gated Candidate

这个候选把 route-frequency expert weights 再经过 materialized checkpoint delta audit 约束：如果某个 expert 的实际 FFN tensor relative delta norm 超过阈值，就按比例缩小非 base source 的 delta weight。目标是保留 route-aware 机制，同时避免少数 expert projection 被移动到超过 base norm 的幅度。

- Status: `audit_gated_rules_ready`
- Target max relative delta: `0.750`
- Expert rules: `5243`
- Scaled expert rules: `302`
- Frozen by audit cap: `0`
- Mean effective nonbase weight: `0.201` -> `0.193`
- Max effective nonbase weight: `0.850` -> `0.850`
- Max audited relative delta before cap: `1.327`
- Min audit delta scale: `0.565`

## Action Counts

| action | count |
| --- | ---: |
| `keep_route_weight_rule` | 4941 |
| `scale_nonbase_delta_to_relative_norm_cap` | 302 |

## Largest Audited Expert Deltas

| layer | expert | original nonbase | new nonbase | max rel delta | scale | action |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 15 | 38 | 0.850 | 0.480 | 1.327 | 0.565 | `scale_nonbase_delta_to_relative_norm_cap` |
| 12 | 71 | 0.850 | 0.515 | 1.237 | 0.606 | `scale_nonbase_delta_to_relative_norm_cap` |
| 20 | 118 | 0.850 | 0.553 | 1.153 | 0.650 | `scale_nonbase_delta_to_relative_norm_cap` |
| 11 | 31 | 0.850 | 0.555 | 1.148 | 0.653 | `scale_nonbase_delta_to_relative_norm_cap` |
| 10 | 36 | 0.850 | 0.557 | 1.145 | 0.655 | `scale_nonbase_delta_to_relative_norm_cap` |
| 14 | 24 | 0.850 | 0.558 | 1.142 | 0.657 | `scale_nonbase_delta_to_relative_norm_cap` |
| 15 | 21 | 0.850 | 0.565 | 1.129 | 0.664 | `scale_nonbase_delta_to_relative_norm_cap` |
| 21 | 67 | 0.850 | 0.566 | 1.126 | 0.666 | `scale_nonbase_delta_to_relative_norm_cap` |
| 18 | 83 | 0.850 | 0.566 | 1.126 | 0.666 | `scale_nonbase_delta_to_relative_norm_cap` |
| 13 | 33 | 0.850 | 0.566 | 1.126 | 0.666 | `scale_nonbase_delta_to_relative_norm_cap` |
| 16 | 116 | 0.850 | 0.568 | 1.123 | 0.668 | `scale_nonbase_delta_to_relative_norm_cap` |
| 28 | 116 | 0.850 | 0.568 | 1.122 | 0.668 | `scale_nonbase_delta_to_relative_norm_cap` |
| 17 | 25 | 0.850 | 0.569 | 1.120 | 0.670 | `scale_nonbase_delta_to_relative_norm_cap` |
| 18 | 108 | 0.850 | 0.572 | 1.115 | 0.673 | `scale_nonbase_delta_to_relative_norm_cap` |
| 12 | 11 | 0.850 | 0.572 | 1.114 | 0.673 | `scale_nonbase_delta_to_relative_norm_cap` |
| 23 | 117 | 0.850 | 0.572 | 1.114 | 0.673 | `scale_nonbase_delta_to_relative_norm_cap` |
| 11 | 52 | 0.850 | 0.573 | 1.113 | 0.674 | `scale_nonbase_delta_to_relative_norm_cap` |
| 21 | 115 | 0.850 | 0.573 | 1.113 | 0.674 | `scale_nonbase_delta_to_relative_norm_cap` |
| 13 | 87 | 0.850 | 0.574 | 1.111 | 0.675 | `scale_nonbase_delta_to_relative_norm_cap` |
| 24 | 93 | 0.850 | 0.575 | 1.109 | 0.676 | `scale_nonbase_delta_to_relative_norm_cap` |

## Strongest Audit Gates

| layer | expert | original nonbase | new nonbase | max rel delta | scale | action |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 15 | 38 | 0.850 | 0.480 | 1.327 | 0.565 | `scale_nonbase_delta_to_relative_norm_cap` |
| 12 | 71 | 0.850 | 0.515 | 1.237 | 0.606 | `scale_nonbase_delta_to_relative_norm_cap` |
| 20 | 118 | 0.850 | 0.553 | 1.153 | 0.650 | `scale_nonbase_delta_to_relative_norm_cap` |
| 11 | 31 | 0.850 | 0.555 | 1.148 | 0.653 | `scale_nonbase_delta_to_relative_norm_cap` |
| 10 | 36 | 0.850 | 0.557 | 1.145 | 0.655 | `scale_nonbase_delta_to_relative_norm_cap` |
| 14 | 24 | 0.850 | 0.558 | 1.142 | 0.657 | `scale_nonbase_delta_to_relative_norm_cap` |
| 15 | 21 | 0.850 | 0.565 | 1.129 | 0.664 | `scale_nonbase_delta_to_relative_norm_cap` |
| 21 | 67 | 0.850 | 0.566 | 1.126 | 0.666 | `scale_nonbase_delta_to_relative_norm_cap` |
| 18 | 83 | 0.850 | 0.566 | 1.126 | 0.666 | `scale_nonbase_delta_to_relative_norm_cap` |
| 13 | 33 | 0.850 | 0.566 | 1.126 | 0.666 | `scale_nonbase_delta_to_relative_norm_cap` |
| 16 | 116 | 0.850 | 0.568 | 1.123 | 0.668 | `scale_nonbase_delta_to_relative_norm_cap` |
| 28 | 116 | 0.850 | 0.568 | 1.122 | 0.668 | `scale_nonbase_delta_to_relative_norm_cap` |
| 17 | 25 | 0.850 | 0.569 | 1.120 | 0.670 | `scale_nonbase_delta_to_relative_norm_cap` |
| 18 | 108 | 0.850 | 0.572 | 1.115 | 0.673 | `scale_nonbase_delta_to_relative_norm_cap` |
| 12 | 11 | 0.850 | 0.572 | 1.114 | 0.673 | `scale_nonbase_delta_to_relative_norm_cap` |
| 23 | 117 | 0.850 | 0.572 | 1.114 | 0.673 | `scale_nonbase_delta_to_relative_norm_cap` |
| 11 | 52 | 0.850 | 0.573 | 1.113 | 0.674 | `scale_nonbase_delta_to_relative_norm_cap` |
| 21 | 115 | 0.850 | 0.573 | 1.113 | 0.674 | `scale_nonbase_delta_to_relative_norm_cap` |
| 13 | 87 | 0.850 | 0.574 | 1.111 | 0.675 | `scale_nonbase_delta_to_relative_norm_cap` |
| 24 | 93 | 0.850 | 0.575 | 1.109 | 0.676 | `scale_nonbase_delta_to_relative_norm_cap` |

## Files

- `results/qwen3_moe_audit_gated_candidate/audit_gated_source_weights_by_expert.csv`
- `results/qwen3_moe_audit_gated_candidate/tensor_rules.txt`
- `results/qwen3_moe_audit_gated_candidate/writer_command.txt`
- `results/qwen3_moe_audit_gated_candidate/summary.json`
