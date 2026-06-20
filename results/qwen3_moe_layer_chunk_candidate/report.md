# Qwen3 MoE Layer/Chunk Coefficient Candidate

这个候选把 mechanism leverage map 里的 layer/chunk 敏感性变成同结构 tensor rules：router 和 shared attention 继续冻结，只对高敏感 routed experts 的 Coder contribution 做小幅 shrink。

- Status: `layer_chunk_candidate_ready`
- Selected schedule: `policy_095_098_100`
- Route-mass Coder retention: `0.985142`
- Risk-weighted delta reduction: `0.0207014`
- Fine-layer Coder retention: `0.95`
- Max predicted relative delta: `0.65`
- Selection constraints: retention >= `0.975`, max relative delta <= `0.65`
- Writer dry-run: `True` (18867 floating tensors, 3891 frozen tensors, 15729 tensor-rule hits)

## Candidate Search

| schedule | feasible | retention | fine retention | risk reduction | delta norm ratio | max rel-delta | changed groups | objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuous_importance_s0.20` | `False` | 0.885981 | 0.809548 | 0.136982 | 0.87884 | 0.65 | 5138 | 0.097075 |
| `continuous_risk_layer_s0.20` | `False` | 0.889943 | 0.838696 | 0.127042 | 0.877955 | 0.626564 | 5243 | 0.0885223 |
| `continuous_importance_s0.15` | `False` | 0.914486 | 0.857161 | 0.102736 | 0.908753 | 0.65 | 5138 | 0.0728063 |
| `policy_080_090_098` | `False` | 0.923854 | 0.8 | 0.0978801 | 0.917198 | 0.637 | 5243 | 0.0712289 |
| `continuous_risk_layer_s0.15` | `False` | 0.917457 | 0.879022 | 0.0952818 | 0.908278 | 0.632423 | 5243 | 0.0663918 |
| `policy_085_092_100` | `False` | 0.948785 | 0.85 | 0.0704917 | 0.942568 | 0.65 | 2631 | 0.0525664 |
| `continuous_importance_s0.10` | `False` | 0.942991 | 0.904774 | 0.0684908 | 0.938933 | 0.65 | 5138 | 0.0485375 |
| `continuous_risk_layer_s0.10` | `False` | 0.944971 | 0.919348 | 0.0635212 | 0.938734 | 0.638282 | 5243 | 0.0442612 |
| `policy_090_095_100` | `False` | 0.966963 | 0.9 | 0.0455966 | 0.962478 | 0.65 | 2631 | 0.0340337 |
| `policy_092_096_100` | `False` | 0.973571 | 0.92 | 0.0364772 | 0.969856 | 0.65 | 2631 | 0.027227 |
| `continuous_importance_s0.05` | `False` | 0.971495 | 0.952387 | 0.0342454 | 0.969356 | 0.65 | 5138 | 0.0242688 |
| `continuous_risk_layer_s0.05` | `False` | 0.972486 | 0.959674 | 0.0317606 | 0.969312 | 0.644141 | 5243 | 0.0221306 |
| `policy_095_098_100` | `True` | 0.985142 | 0.95 | 0.0207014 | 0.98283 | 0.65 | 2631 | 0.015501 |
| `policy_098_099_100` | `True` | 0.993393 | 0.98 | 0.00911931 | 0.992373 | 0.65 | 2631 | 0.00680674 |
| `baseline_unified` | `True` | 1 | 1 | 0 | 1 | 0.65 | 0 | 0 |

## Layer Coefficients

| layer | policy | coeff | importance | router rel | min Jaccard |
| ---: | --- | ---: | ---: | ---: | ---: |
| 13 | `two_layer_chunk_coefficients` | 0.98 | 0.807586 | 0.85527 | 0.382133 |
| 24 | `two_layer_chunk_coefficients` | 0.98 | 0.800075 | 0.747783 | 0.290635 |
| 19 | `two_layer_chunk_coefficients` | 0.98 | 0.777255 | 0.825168 | 0.322665 |
| 11 | `two_layer_chunk_coefficients` | 0.98 | 0.753925 | 0.819092 | 0.376152 |
| 15 | `two_layer_chunk_coefficients` | 0.98 | 0.734771 | 0.824639 | 0.341475 |
| 9 | `two_layer_chunk_coefficients` | 0.98 | 0.723616 | 0.787176 | 0.412437 |
| 16 | `two_layer_chunk_coefficients` | 0.98 | 0.723073 | 0.817559 | 0.337753 |
| 25 | `two_layer_chunk_coefficients` | 0.98 | 0.719829 | 0.701719 | 0.37458 |
| 4 | `two_layer_chunk_coefficients` | 0.98 | 0.711884 | 0.792725 | 0.431243 |
| 29 | `two_layer_chunk_coefficients` | 0.98 | 0.685197 | 0.562119 | 0.361532 |
| 10 | `two_layer_chunk_coefficients` | 0.98 | 0.679386 | 0.821698 | 0.39701 |
| 30 | `two_layer_chunk_coefficients` | 0.98 | 0.66073 | 0.516373 | 0.339156 |
| 6 | `two_layer_chunk_coefficients` | 0.98 | 0.644598 | 0.806417 | 0.419912 |
| 14 | `two_layer_chunk_coefficients` | 0.98 | 0.642632 | 0.841261 | 0.343251 |
| 8 | `two_layer_chunk_coefficients` | 0.98 | 0.638412 | 0.783341 | 0.464884 |
| 34 | `two_layer_chunk_coefficients` | 0.98 | 0.634043 | 0.365883 | 0.388898 |
| 17 | `per_layer_coefficients` | 0.95 | 0.886422 | 0.844352 | 0.311278 |
| 20 | `per_layer_coefficients` | 0.95 | 0.871401 | 0.808873 | 0.318842 |
| 22 | `per_layer_coefficients` | 0.95 | 0.869283 | 0.793704 | 0.242238 |
| 23 | `per_layer_coefficients` | 0.95 | 0.848024 | 0.770017 | 0.253841 |

## Why This Is A Candidate, Not A Conclusion

内部 proxy 只能说明这个 schedule 在 retention 和 delta hard-cap 约束内降低了高敏感层的风险加权 Coder delta；它不能证明下游任务更好。Candidate search 里 objective 更高但 feasible 为 false 的 schedule 是因为 route-mass Coder retention 太低，不能直接选。这个 checkpoint 必须和 source、tail-trimmed、searched no-gt-0.65 在同一套 budgeted vLLM eval 下比较。

## Literature Priors

- `expert_merging_chunking`: https://arxiv.org/abs/2509.25712
- `harc_routing_breakdown`: https://arxiv.org/abs/2606.03391
- `router_kd_calibration`: https://arxiv.org/abs/2603.02217

## Outputs

- `report`: `results/qwen3_moe_layer_chunk_candidate/report.md`
- `summary`: `results/qwen3_moe_layer_chunk_candidate/summary.json`
- `selected_schedule`: `results/qwen3_moe_layer_chunk_candidate/selected_schedule.json`
- `schedule_search`: `results/qwen3_moe_layer_chunk_candidate/schedule_search.csv`
- `layer_coefficients`: `results/qwen3_moe_layer_chunk_candidate/layer_coefficients.csv`
- `selected_group_rules`: `results/qwen3_moe_layer_chunk_candidate/selected_group_rules.csv`
- `tensor_rules`: `results/qwen3_moe_layer_chunk_candidate/tensor_rules.txt`
- `literature_sources`: `results/qwen3_moe_layer_chunk_candidate/literature_sources.json`
- `writer_command`: `results/qwen3_moe_layer_chunk_candidate/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_layer_chunk_candidate/dry_run_command.txt`
- `checkpoint_output_dir`: `results/checkpoints/qwen3_moe_layer_chunk_candidate`
