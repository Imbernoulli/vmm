# Average Decision Report

Generated at: `2026-06-19T03:52:20.328090+00:00`

这个报告把已有 merge plane、method table、delta conflict 和可选 MoE routing probe 汇总成同构 Average 决策。这里的同构指：最终 checkpoint 的 config、tokenizer、layer 数、hidden size、router shape 和 expert 数量不变；改变的是参数来自哪些 source checkpoint 以及每组参数的平均权重。

## 当前证据

| experiment | objective | linear | best grid | gap | barrier | conflict | verdict | suggested weights |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| digits | worst_acc | 0.922 | 0.961 | 0.039 | -6.256 | 0.507 | coefficient_search | `{"alpha": 0.35, "beta": 0.35, "source": "best_grid_by_worst_objective"}` |
| cifar10 | worst_acc | 0.249 | 0.426 | 0.177 | -4.544 | 0.491 | avoid_uniform_average | `{"alpha": 0.125, "beta": 0.275, "source": "best_grid_by_worst_objective"}` |
| cifar100_vit | worst_acc | 0.076 | 0.194 | 0.118 | -2.834 | 0.638 | avoid_uniform_average | `{"alpha": 0.03125, "beta": 0.03125, "source": "best_grid_by_worst_objective"}` |
| pretrained_vit | worst_acc | 0.763 | 0.783 | 0.020 | -2.252 | 0.554 | coefficient_search | `{"alpha": 0.35, "beta": 0.125, "source": "best_grid_by_worst_objective"}` |
| qwen_instruct_coder | worst_nll | 9.553 | 7.541 | 2.012 | 1.967 | 0.386 | avoid_uniform_average | `{"alpha": 1.0, "beta": 0.0, "source": "best_grid_by_worst_objective"}` |

## 决策解释

### digits

- 判断：`coefficient_search`。
- 建议：平均可用但系数敏感；用 min-max validation objective 选 alpha/beta 或 layer-wise lambda。
- 当前建议权重：`{"alpha": 0.35, "beta": 0.35, "source": "best_grid_by_worst_objective"}`。
- 冲突最高位置：`net.6.bias`。

### cifar10

- 判断：`avoid_uniform_average`。
- 建议：不要用 0.5/0.5 或 1/n；先做 connectivity/barrier 筛选，再用验证集重学同构平均权重。
- 当前建议权重：`{"alpha": 0.125, "beta": 0.275, "source": "best_grid_by_worst_objective"}`。
- 冲突最高位置：`classifier.bias`。

### cifar100_vit

- 判断：`avoid_uniform_average`。
- 建议：不要用 0.5/0.5 或 1/n；先做 connectivity/barrier 筛选，再用验证集重学同构平均权重。
- 当前建议权重：`{"alpha": 0.03125, "beta": 0.03125, "source": "best_grid_by_worst_objective"}`。
- 冲突最高位置：`head.bias`。

### pretrained_vit

- 判断：`coefficient_search`。
- 建议：平均可用但系数敏感；用 min-max validation objective 选 alpha/beta 或 layer-wise lambda。
- 当前建议权重：`{"alpha": 0.35, "beta": 0.125, "source": "best_grid_by_worst_objective"}`。
- 冲突最高位置：`bias`。

### qwen_instruct_coder

- 判断：`avoid_uniform_average`。
- 建议：不要用 0.5/0.5 或 1/n；先做 connectivity/barrier 筛选，再用验证集重学同构平均权重。
- 当前建议权重：`{"alpha": 1.0, "beta": 0.0, "source": "best_grid_by_worst_objective"}`。
- 冲突最高位置：`instruct vs coder`。

## 参数组策略

| parameter group | trigger | same-shape average action |
| --- | --- | --- |
| embedding/lm_head | general retention 下降或 tokenizer 高频 token NLL 上升 | freeze anchor 或 Fisher-weighted average，避免全局输出分布漂移。 |
| attention | layer cosine 同向且 barrier 低 | 允许普通/task-arithmetic average；系数由 validation min-max objective 选择。 |
| MLP / dense FFN | weighted sign conflict 高 | 用 TIES/DELLA/Fisher 风格的 coordinate-wise 权重，冲突坐标降权或回 anchor。 |
| MoE router | route entropy 低、max top-1 fraction 高或 route overlap 低 | 首轮 frozen router；之后只校准 router/bias，加入 load-balance 和 route-overlap 约束。 |
| MoE experts | expert output/activation 相似但 index 不可靠 | 先 expert matching，再在同 expert 数和同 tensor shape 内做 expert-wise average。 |
| LoRA/adapters | 多个下游用户只发布 adapter | 先做 adapter delta/rank/output probe；最终压回一个同构 adapter，mixture 只作上界。 |

## MoE Routing Probe

当前没有找到 MoE routing probe 输出。下一步用 `scripts/probe_moe_routing.py` 跑 Qwen3 MoE，再把输出目录传给这个脚本的 `--router-dir`。

## 下一步

1. 对 Dense Qwen 7B 候选模型先跑 endpoint NLL/benchmark slice，再跑 delta/connectivity probe。
2. 用本报告选择全局 `alpha/beta` 或 layer/module-wise coefficient，禁止把结构改变的 union/ensemble 当最终模型。
3. 对 Qwen3 MoE 先跑 routing probe；若 route overlap 低，优先 router frozen + expert matching，再写回同构 MoE checkpoint。

## Files

- `results/average_decision_report/decision_table.csv`
- `results/average_decision_report/parameter_group_actions.csv`
- `results/average_decision_report/summary.json`
