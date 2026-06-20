# Qwen3 MoE Layer/Chunk Coefficient Candidate

这个候选把 mechanism leverage map 里的 layer/chunk 敏感性变成同结构 tensor rules：router 和 shared attention 继续冻结，只对高敏感 routed experts 的 Coder contribution 做小幅 shrink。

- Status: `layer_chunk_candidate_ready`
- Selected schedule: `policy_095_098_100`
- Route-mass Coder retention: `0.985126`
- Risk-weighted delta reduction: `0.0206744`
- Fine-layer Coder retention: `0.95`
- Max predicted relative delta: `0.65`
- Selection constraints: retention >= `0.975`, max relative delta <= `0.65`
- Writer manifest validated: `True` (dry_run=`False`, 18867 floating tensors, 3891 frozen tensors, 15729 tensor-rule hits)

## Candidate Search

| schedule | feasible | retention | fine retention | risk reduction | delta norm ratio | max rel-delta | changed groups | objective |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuous_importance_s0.20` | `False` | 0.888544 | 0.814834 | 0.1342 | 0.881653 | 0.65 | 5138 | 0.0951902 |
| `continuous_risk_layer_s0.20` | `False` | 0.891352 | 0.843162 | 0.125564 | 0.879517 | 0.62736 | 5243 | 0.0875374 |
| `continuous_importance_s0.15` | `False` | 0.916408 | 0.861126 | 0.10065 | 0.910876 | 0.65 | 5138 | 0.0713927 |
| `policy_080_090_098` | `False` | 0.923802 | 0.8 | 0.0978186 | 0.917288 | 0.637 | 5243 | 0.0711494 |
| `continuous_risk_layer_s0.15` | `False` | 0.918514 | 0.882371 | 0.0941731 | 0.909454 | 0.63302 | 5243 | 0.065653 |
| `policy_085_092_100` | `False` | 0.94875 | 0.85 | 0.0705004 | 0.942677 | 0.65 | 2623 | 0.0525628 |
| `continuous_importance_s0.10` | `False` | 0.944272 | 0.907417 | 0.0670999 | 0.940357 | 0.65 | 5138 | 0.0475951 |
| `continuous_risk_layer_s0.10` | `False` | 0.945676 | 0.921581 | 0.0627821 | 0.939521 | 0.63868 | 5243 | 0.0437687 |
| `policy_090_095_100` | `False` | 0.966938 | 0.9 | 0.0455874 | 0.96254 | 0.65 | 2623 | 0.0340156 |
| `policy_092_096_100` | `False` | 0.97355 | 0.92 | 0.0364699 | 0.969906 | 0.65 | 2623 | 0.0272125 |
| `continuous_importance_s0.05` | `False` | 0.972136 | 0.953709 | 0.03355 | 0.970072 | 0.65 | 5138 | 0.0237976 |
| `continuous_risk_layer_s0.05` | `False` | 0.972838 | 0.96079 | 0.031391 | 0.969706 | 0.64434 | 5243 | 0.0218843 |
| `policy_095_098_100` | `True` | 0.985126 | 0.95 | 0.0206744 | 0.982847 | 0.65 | 2623 | 0.0154685 |
| `policy_098_099_100` | `True` | 0.993388 | 0.98 | 0.00911747 | 0.992385 | 0.65 | 2623 | 0.00680313 |
| `baseline_unified` | `True` | 1 | 1 | 0 | 1 | 0.65 | 0 | 0 |

## Layer Coefficients

| layer | policy | coeff | importance | router rel | min Jaccard | geometry risk | high-geom experts |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | `per_layer_coefficients` | 0.95 | 0.863789 | 0.793704 | 0.242238 | 0.671821 | 9 |
| 17 | `per_layer_coefficients` | 0.95 | 0.904783 | 0.844352 | 0.311278 | 0.714306 | 15 |
| 20 | `per_layer_coefficients` | 0.95 | 0.863149 | 0.808873 | 0.318842 | 0.671361 | 10 |
| 23 | `per_layer_coefficients` | 0.95 | 0.85163 | 0.770017 | 0.253841 | 0.674895 | 10 |
| 12 | `per_layer_coefficients` | 0.95 | 0.84651 | 0.823488 | 0.344755 | 0.67519 | 11 |
| 26 | `per_layer_coefficients` | 0.95 | 0.824761 | 0.637357 | 0.345504 | 0.641394 | 9 |
| 21 | `per_layer_coefficients` | 0.95 | 0.821837 | 0.786077 | 0.324809 | 0.662217 | 8 |
| 13 | `per_layer_coefficients` | 0.95 | 0.809125 | 0.85527 | 0.382133 | 0.675991 | 9 |
| 18 | `two_layer_chunk_coefficients` | 0.98 | 0.808771 | 0.869161 | 0.307368 | 0.65006 | 5 |
| 25 | `two_layer_chunk_coefficients` | 0.98 | 0.724006 | 0.701719 | 0.37458 | 0.651717 | 6 |
| 4 | `two_layer_chunk_coefficients` | 0.98 | 0.675699 | 0.792725 | 0.431243 | 0.586748 | 3 |
| 27 | `two_layer_chunk_coefficients` | 0.98 | 0.624765 | 0.601166 | 0.341909 | 0.63135 | 5 |
| 6 | `two_layer_chunk_coefficients` | 0.98 | 0.622717 | 0.806417 | 0.419912 | 0.602032 | 1 |
| 24 | `two_layer_chunk_coefficients` | 0.98 | 0.802862 | 0.747783 | 0.290635 | 0.661782 | 10 |
| 15 | `two_layer_chunk_coefficients` | 0.98 | 0.772271 | 0.824639 | 0.341475 | 0.702701 | 13 |
| 19 | `two_layer_chunk_coefficients` | 0.98 | 0.760526 | 0.825168 | 0.322665 | 0.651748 | 5 |
| 16 | `two_layer_chunk_coefficients` | 0.98 | 0.752207 | 0.817559 | 0.337753 | 0.688071 | 10 |
| 11 | `two_layer_chunk_coefficients` | 0.98 | 0.745159 | 0.819092 | 0.376152 | 0.65972 | 3 |
| 9 | `two_layer_chunk_coefficients` | 0.98 | 0.712906 | 0.787176 | 0.412437 | 0.636632 | 5 |
| 29 | `two_layer_chunk_coefficients` | 0.98 | 0.710178 | 0.562119 | 0.361532 | 0.665524 | 10 |

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
