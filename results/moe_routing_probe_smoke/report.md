# MoE Routing Probe

这个报告由 `scripts/probe_moe_routing.py` 生成，用来把真实 MoE router 的 top-k、entropy、expert load 和可选 source-to-source route overlap 转成后续 readiness / route-weight recipe 可直接读取的 CSV。

## Run Summary

- Model: `tiny_moe_left`
- Compare model: `tiny_moe_right`
- Prompts: `3` across `3` categories
- Routers observed: `2`
- Top-k: `2`
- Chat template: `False`

## Output Rows

| file | rows | consumer |
| --- | ---: | --- |
| `results/moe_routing_probe_smoke/router_summary.csv` | 6 | routing readiness / MoE average plan |
| `results/moe_routing_probe_smoke/expert_load.csv` | 24 | route-weight recipes / expert risk analysis |
| `results/moe_routing_probe_smoke/route_overlap.csv` | 6 | router overlap gate |
| `results/moe_routing_probe_smoke/token_routes.csv` | 72 | token-level debugging |
| `results/moe_routing_probe_smoke/compare_router_summary.csv` | 6 | comparison audit |
| `results/moe_routing_probe_smoke/compare_expert_load.csv` | 24 | comparison audit |

## CSV Contract

- `router_summary.csv` records `model`, `category`, `prompt_idx`, `router`, `num_experts`, `tokens`, `top_k`, `router_entropy_mean`, `top1_margin_mean`, `unique_top1_experts`, `unique_topk_experts`, `max_top1_fraction`, and `effective_top1_experts`.
- `expert_load.csv` records per-router/per-expert `top1_fraction` and `topk_fraction`; `scripts/build_moe_route_weight_recipes.py` uses these route masses to emit same-shape tensor rules.
- `route_overlap.csv`, when a compare model is supplied, records `top1_agreement` and `topk_jaccard`; `scripts/analyze_moe_routing_readiness.py` uses them to catch routing breakdown before materialization.

## Next Commands

```bash
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe_smoke
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe_smoke --source general --source code
```

## Files

- `results/moe_routing_probe_smoke/manifest.json`
- `results/moe_routing_probe_smoke/summary.json`
- `results/moe_routing_probe_smoke/report.md`
