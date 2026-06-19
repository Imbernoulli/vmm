# MoE Router Bias Plan

这个 recipe 把 routing probe 的 `expert_load.csv` 转成 writer 可用的 router-bias additive delta。它不改变 expert 数、router shape 或模型结构；只是给已有 bias tensor 的每个 expert logit 加一个离线计算出的标量。

- Status: `router_bias_delta_ready`
- Methods: `unified_confidence_blended_moe_average`
- Router dirs: `results/toy_moe_merge`
- Routers: `1`
- Delta rows: `4`
- Nonzero deltas: `4`
- Capacity factor: `1.25`
- Load statistic: `worst`
- Bias step / clip: `0.25` / `0.5`

## 规则

```text
observed_topk_fraction[e] = worst/mean/quantile over prompt-category slices
capacity_fraction = capacity_factor / num_experts
raw_delta[e] = -bias_step * log(observed_topk_fraction[e] / capacity_fraction)
delta[e] = clip(raw_delta[e] - mean_e(raw_delta[e]), -max_abs_delta, max_abs_delta)
```

过载 expert 的 logit 会被压低，低载 expert 的 logit 会被抬高；同一 router 内做中心化，避免引入无意义的整体 logit 平移。这个 CSV 是离线候选修正，仍然要用 held-out 下游任务和 capacity-aware 指标验收。

## Writer 用法

```bash
python scripts/write_same_shape_average_checkpoint.py --base MOE_BASE_OR_ANCHOR_PATH --source general=GENERAL_MODEL_PATH --source code=CODE_MODEL_PATH --source-weight general=0.0 --source-weight code=0.0 --freeze-router --tensor-add-csv results/moe_confidence_blended_router_bias_plan/router_bias_deltas.csv --output-dir results/checkpoints/moe_bias_calibrated_candidate --dry-run
```

如果真实 checkpoint 没有对应 bias tensor，writer 会在校验阶段报错；这表示该模型需要改用 router weight 小步校准或保持 router freeze，而不是强行改结构。

## 过载 Expert 预览

| method | router | expert | observed top-k | capacity | ratio | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `unified_confidence_blended_moe_average` | `toy_router` | 0 | 0.3887 | 0.3125 | 1.244 | -0.0523 |
| `unified_confidence_blended_moe_average` | `toy_router` | 1 | 0.3137 | 0.3125 | 1.004 | 0.0013 |
| `unified_confidence_blended_moe_average` | `toy_router` | 2 | 0.3075 | 0.3125 | 0.984 | 0.0063 |
| `unified_confidence_blended_moe_average` | `toy_router` | 3 | 0.2637 | 0.3125 | 0.844 | 0.0447 |

## Files

- `results/moe_confidence_blended_router_bias_plan/router_bias_plan.csv`
- `results/moe_confidence_blended_router_bias_plan/router_bias_deltas.csv`
- `results/moe_confidence_blended_router_bias_plan/summary.json`
