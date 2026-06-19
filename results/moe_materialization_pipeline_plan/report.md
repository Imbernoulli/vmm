# MoE Materialization Pipeline Plan

这个 plan 把真实 Qwen MoE average 从目标模型选择到 vLLM eval 的每个 gate 串起来。它区分 toy/template 已验证能力和真实 Qwen3 MoE 仍缺的 probe 输入，避免把 toy readiness 误当成真实 checkpoint readiness。

- Status: `waiting_for_real_moe_probe_or_paths`
- Current blocking stage: `exact_moe_topology`
- Gates: `10`
- Ready/complete gates: `3`
- Waiting/blocked gates: `7`

| stage | status | evidence | next action |
| --- | --- | --- | --- |
| `target_model_selection` | `complete` | 3 recommended p0 MoE models: Qwen/Qwen3-30B-A3B-Base, Qwen/Qwen3-30B-A3B, Qwen/Qwen3-Coder-30B-A3B-Instruct | inspect exact topology/header for every selected source model |
| `exact_moe_topology` | `partial_config_only` | inspected_moe_configs=1, weights_available=0, target_p0_moe=3 | inspect exact Qwen3-30B-A3B source checkpoints with safetensors headers |
| `real_moe_routing_probe` | `ready_to_run` | 1 routing probe command planned; expected output results/moe_routing_probe/qwen3_30b_general_vs_code | run the planned Qwen3 MoE routing probe before materialization |
| `routing_readiness_gate` | `waiting_for_routing_probe` | router_rows=0, expert_rows=0, specialization_rows=0 | run routing probe, then rerun readiness analysis on its output |
| `route_weight_tensor_rules` | `waiting_for_routing_probe` | expert_rules=0, tensor_rules=1, router_dirs=0 | run routing probe first; route weights require real expert_load.csv |
| `layerwise_expert_remap` | `template_validated` | layer_aware_rules=3, manual_review_rows=1 in smoke | run layer-wise expert-output matching on real Qwen3 MoE sources, then build source tensor aliases |
| `router_bias_capacity_delta` | `waiting_for_routing_probe` | expert_load_exists=False at results/moe_routing_probe/qwen3_30b_general_vs_code/expert_load.csv | convert real top-k load imbalance into same-shape router-bias deltas |
| `combined_same_shape_writer_recipe` | `template_validated` | tensor_rules=5, alias_rules=4, bias_delta_rows=4 | replace toy/template inputs with real route-weight rules, layer-wise aliases, and router-bias deltas |
| `checkpoint_materialization` | `blocked_by_real_source_paths` | materialized=1, blocked_by_placeholders=4 | after real source paths and recipes are ready, run writer dry-run, then materialize safetensors |
| `hosted_downstream_eval` | `waiting_for_materialized_moe_checkpoint` | checkpoint_eval_plan_status=hosted_eval_complete, completed=1, missing_checkpoint=2 | host the materialized MoE checkpoint with vLLM and run the same downstream eval harness |

## Next Commands

```bash
# target_model_selection
python scripts/build_qwen_target_model_registry.py
# exact_moe_topology
python scripts/inspect_checkpoint_topology.py --model NAME=MODEL_PATH --output-dir results/checkpoint_topology_inspect
# real_moe_routing_probe
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --max-length 768 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code
# routing_readiness_gate
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code
# route_weight_tensor_rules
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source agentic_code=code --category-source code=code --category-source finance=general --category-source general=general --category-source legal=general --category-source long_context=general --category-source math=general --category-source safety=general
# layerwise_expert_remap
PYTHONPATH=src python scripts/build_moe_expert_remap_plan.py --expert-match results/moe_expert_match/qwen3_30b_general_vs_code/expert_match.csv --output-dir results/moe_expert_remap/qwen3_30b_general_vs_code --source-name code
# router_bias_capacity_delta
PYTHONPATH=src python scripts/build_moe_router_bias_plan.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --output-dir results/moe_router_bias_plan/qwen3_30b_general_vs_code --router-bias-template '{router}.bias'
# combined_same_shape_writer_recipe
PYTHONPATH=src python scripts/build_moe_combined_materialization_recipe.py --tensor-rule-file results/moe_route_weight_recipes/tensor_rules.txt --source-tensor-alias-file results/moe_expert_remap/qwen3_30b_general_vs_code/source_tensor_aliases.txt --tensor-add-csv results/moe_router_bias_plan/qwen3_30b_general_vs_code/router_bias_deltas.csv
# checkpoint_materialization
PYTHONPATH=src python scripts/build_checkpoint_materialization_readiness.py --output-dir results/checkpoint_materialization_readiness
# hosted_downstream_eval
PYTHONPATH=src python scripts/build_vllm_checkpoint_eval_plan.py --output-dir results/vllm_checkpoint_eval_plan
```

## Files

- `results/moe_materialization_pipeline_plan/stage_gates.csv`
- `results/moe_materialization_pipeline_plan/next_commands.sh`
- `results/moe_materialization_pipeline_plan/summary.json`
