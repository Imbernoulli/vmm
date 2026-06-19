# MoE Routing Readiness

这个报告把 MoE routing probe 的原始 CSV 转成合并前的风险诊断。它回答的不是“该不该做 MoE”，而是更具体的四个问题：router 是否会 collapse，两个 source 的路由是否漂移，top-k 边界是否脆弱，以及哪些 experts 需要 route/category-aware 权重。

- Readiness status: `waiting_for_routing_probe`
- Router dirs: `none`
- Router rows: `0`；expert rows: `0`；specialization rows: `0`

## 拓扑线索

- MoE model: `qwen3_5_35b_a3b` / `qwen3_5_moe`
- Experts: `256`；active per token: `8`；active fraction: `0.03125`
- Local weights available: `False`

## Router Readiness

当前没有真实 `router_summary.csv`。先运行 MoE routing probe，再重新生成本报告。

## Expert Load Risks

当前没有真实 `expert_load.csv`。route-weight recipes 也会保持 `waiting_for_routing_probe`。

## Category Specialization

当前没有 category-level expert specialization 证据。

## 规则依据

- [HARC / routing-breakdown](https://arxiv.org/abs/2606.03391) 分析说明 MoE router 的 softmax/top-k 对参数扰动敏感，因此 `low_topk_route_overlap`、`low_top1_route_agreement` 和 load concentration 都应阻止直接 router average。
- [Sub-MoE](https://arxiv.org/abs/2506.23266) / [MergeMoE](https://arxiv.org/abs/2510.14436) 强调 expert specialization 和 expert output alignment，因此高负载或强 category-specialized experts 应先做 route/source-aware 权重，而不是同权平均。
- [Expert Merging](https://arxiv.org/abs/2509.25712) 强调 layer/chunk-wise coefficients；本报告输出的风险表应和 `build_moe_route_weight_recipes.py` 的 tensor rules 以及后续 layer-wise coefficient search 联动。

## 下一步命令

```bash
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source general=general --category-source code=code --category-source math=general --category-source safety=general
```

## Files

- `results/moe_routing_readiness/router_readiness.csv`
- `results/moe_routing_readiness/expert_load_risks.csv`
- `results/moe_routing_readiness/category_specialization.csv`
- `results/moe_routing_readiness/summary.json`
