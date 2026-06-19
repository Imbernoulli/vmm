# Average Candidate Recipes

这个报告把 Average decision report 转成可执行或可跳过的 candidate recipes。它的核心原则是保守 materialization：endpoint-only 不算有效 average，已被 probe 否定的 uniform average 不生成 writer 命令，MoE template 必须等 routing/expert-load probe 填入权重。

Status counts: `{"candidate_weighted_average": 2, "candidate_interior_weighted_average": 2, "skip_endpoint_only": 1, "skip_rejected_by_probe": 1, "template_waiting_for_routing_probe": 1}`

| experiment | candidate | status | alpha | beta | reason |
| --- | --- | --- | ---: | ---: | --- |
| digits | best_grid_same_shape_average | `candidate_weighted_average` | 0.35 | 0.35 | probe 证据支持 coefficient-selected 同构平均。 |
| cifar10 | best_grid_same_shape_average | `candidate_interior_weighted_average` | 0.125 | 0.275 | uniform average 被否定，但 validation grid 找到了 interior 同构平均系数。 |
| cifar100_vit | best_grid_same_shape_average | `candidate_interior_weighted_average` | 0.03125 | 0.03125 | uniform average 被否定，但 validation grid 找到了 interior 同构平均系数。 |
| pretrained_vit | best_grid_same_shape_average | `candidate_weighted_average` | 0.35 | 0.125 | probe 证据支持 coefficient-selected 同构平均。 |
| qwen_instruct_coder | best_grid_endpoint | `skip_endpoint_only` | 1 | 0 | best grid 是端点；materialize 只会复制一个 source delta，不是有价值的 average。 |
| qwen_instruct_coder | uniform_average_baseline | `skip_rejected_by_probe` | 0.5 | 0.5 | 当前 probe 显示 linear average 落在高 worst-NLL ridge 上；只保留作负 baseline。 |
| qwen_moe_template | router_frozen_route_aware_average | `template_waiting_for_routing_probe` | n/a | n/a | 需要真实 MoE routing/expert-load probe 填入 expert-wise route weights；保持 router frozen 和输出结构不变。 |

## Writer Commands

### qwen_moe_template / router_frozen_route_aware_average

```bash
python scripts/write_same_shape_average_checkpoint.py --base MOE_ANCHOR_PATH --source general=MOE_GENERAL_PATH --source code=MOE_CODE_PATH --source-weight general=0.0 --source-weight code=0.0 --freeze-router --tensor-rule '.*self_attn.*::general=0.5,code=0.5' --tensor-rule '.*experts.*::general=ROUTE_WEIGHT_GENERAL,code=ROUTE_WEIGHT_CODE' --output-dir results/checkpoints/moe_route_aware_candidate --dry-run
```

## Interpretation

- `skip_endpoint_only`：best grid 是 base 或某个 endpoint，说明当前候选模型集合里还没有找到有价值的 average。
- `skip_rejected_by_probe`：已有 probe 直接显示该平均点退化，只保留作负 baseline。
- `materialize_dry_run_first`：先用 writer `--dry-run` 做同构检查，再写真实 checkpoint 并跑 held-out eval。
- `template_waiting_for_routing_probe`：MoE 需要 routing/expert-load probe 填入 expert-wise 权重后才能 materialize。
