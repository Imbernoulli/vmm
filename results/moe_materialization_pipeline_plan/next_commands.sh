#!/usr/bin/env bash
set -euo pipefail

# target_model_selection [complete]
python scripts/build_qwen_target_model_registry.py

# exact_moe_topology [partial_config_only]
python scripts/inspect_checkpoint_topology.py --model NAME=MODEL_PATH --output-dir results/checkpoint_topology_inspect

# real_moe_routing_probe [ready_to_run]
python scripts/probe_moe_routing.py --model Qwen/Qwen3-30B-A3B --compare-model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts prompts/qwen_moe_route_probe_prompts.jsonl --device-map auto --dtype bfloat16 --max-length 768 --use-chat-template --output-dir results/moe_routing_probe/qwen3_30b_general_vs_code

# routing_readiness_gate [waiting_for_routing_probe]
PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code

# route_weight_tensor_rules [waiting_for_routing_probe]
PYTHONPATH=src python scripts/build_moe_route_weight_recipes.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --source general --source code --category-source agentic_code=code --category-source code=code --category-source finance=general --category-source general=general --category-source legal=general --category-source long_context=general --category-source math=general --category-source safety=general

# layerwise_expert_remap [template_validated]
PYTHONPATH=src python scripts/build_moe_expert_remap_plan.py --expert-match results/moe_expert_match/qwen3_30b_general_vs_code/expert_match.csv --output-dir results/moe_expert_remap/qwen3_30b_general_vs_code --source-name code

# router_bias_capacity_delta [waiting_for_routing_probe]
PYTHONPATH=src python scripts/build_moe_router_bias_plan.py --router-dir results/moe_routing_probe/qwen3_30b_general_vs_code --output-dir results/moe_router_bias_plan/qwen3_30b_general_vs_code --router-bias-template '{router}.bias'

# combined_same_shape_writer_recipe [template_validated]
PYTHONPATH=src python scripts/build_moe_combined_materialization_recipe.py --tensor-rule-file results/moe_route_weight_recipes/tensor_rules.txt --source-tensor-alias-file results/moe_expert_remap/qwen3_30b_general_vs_code/source_tensor_aliases.txt --tensor-add-csv results/moe_router_bias_plan/qwen3_30b_general_vs_code/router_bias_deltas.csv

# checkpoint_materialization [blocked_by_real_source_paths]
PYTHONPATH=src python scripts/build_checkpoint_materialization_readiness.py --output-dir results/checkpoint_materialization_readiness

# hosted_downstream_eval [waiting_for_materialized_moe_checkpoint]
PYTHONPATH=src python scripts/build_vllm_checkpoint_eval_plan.py --output-dir results/vllm_checkpoint_eval_plan
