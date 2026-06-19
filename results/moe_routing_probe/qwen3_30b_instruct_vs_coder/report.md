# MoE Routing Probe

这个报告由 `scripts/probe_moe_routing.py` 生成，用来把真实 MoE router 的 top-k、entropy、expert load 和可选 source-to-source route overlap 转成后续 readiness / route-weight recipe 可直接读取的 CSV。

## Run Summary

- Model: `/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
- Compare model: `/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120`
- Prompts: `12` across `8` categories
- Routers observed: `48`
- Top-k: `8`
- Chat template: `True`

## Output Rows

| file | rows | consumer |
| --- | ---: | --- |
| `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/router_summary.csv` | 576 | routing readiness / MoE average plan |
| `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/expert_load.csv` | 73728 | route-weight recipes / expert risk analysis |
| `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/route_overlap.csv` | 576 | router overlap gate |
| `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/compare_router_summary.csv` | 576 | comparison audit |
| `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/compare_expert_load.csv` | 73728 | comparison audit |

## CSV Contract

- `router_summary.csv` records `model`, `category`, `prompt_idx`, `router`, `num_experts`, `tokens`, `top_k`, `router_entropy_mean`, `top1_margin_mean`, `unique_top1_experts`, `unique_topk_experts`, `max_top1_fraction`, and `effective_top1_experts`.
- `expert_load.csv` records per-router/per-expert `top1_fraction` and `topk_fraction`; `scripts/build_moe_route_weight_recipes.py` uses these route masses to emit same-shape tensor rules.
- `route_overlap.csv`, when a compare model is supplied, records `top1_agreement` and `topk_jaccard`; `scripts/analyze_moe_routing_readiness.py` uses them to catch routing breakdown before materialization.

## Next Commands

```bash
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_instruct_vs_coder
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_instruct_vs_coder --source general --source code
```

## Files

- `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/manifest.json`
- `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/summary.json`
- `results/moe_routing_probe/qwen3_30b_instruct_vs_coder/report.md`
