#!/usr/bin/env bash
set -euo pipefail

# First check that the running vLLM endpoint exposes every served model id
# referenced by this plan.
python scripts/audit_vllm_served_model_preflight.py --eval-jobs results/qwen_source_discovery_eval_plan/vllm_eval_jobs.csv --base-url http://127.0.0.1:8000/v1 --output-dir results/qwen_source_discovery_served_model_preflight

# Prepare production task manifests first. These commands use the dataset-backed
# runner and should be executed in an environment with the required datasets/cache.

# measured_coder_thinking_source_frontier: measured_qwen3_moe_source_set
python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/measured_coder_thinking_source_frontier_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only

# dense_7b_general_code_math_reasoning: dense_7b_general_code_math_reasoning
python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_7b_general_code_math_reasoning_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only

# moe_30b_general_code_route_aware: moe_30b_general_code_route_aware
python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/moe_30b_general_code_route_aware_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only

# dense_32b_reasoning_long_reasoning: dense_32b_reasoning_long_reasoning
python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_32b_reasoning_long_reasoning_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only

# Then start the relevant vLLM servers and run the endpoint eval commands.

# measured_coder_thinking_source_frontier: Qwen/Qwen3-Coder-30B-A3B-Instruct,Qwen/Qwen3-30B-A3B-Thinking-2507
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8000/v1 --models Qwen/Qwen3-Coder-30B-A3B-Instruct,Qwen/Qwen3-30B-A3B-Thinking-2507 --tasks mmlu,gsm8k,humaneval_compile --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/measured_coder_thinking_source_frontier_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_source_discovery_eval_plan/vllm_eval/measured_coder_thinking_source_frontier

# dense_7b_general_code_math_reasoning: Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8000/v1 --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_7b_general_code_math_reasoning_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_source_discovery_eval_plan/vllm_eval/dense_7b_general_code_math_reasoning

# moe_30b_general_code_route_aware: Qwen/Qwen3-30B-A3B-Base,Qwen/Qwen3-30B-A3B,Qwen/Qwen3-Coder-30B-A3B-Instruct
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8000/v1 --models Qwen/Qwen3-30B-A3B-Base,Qwen/Qwen3-30B-A3B,Qwen/Qwen3-Coder-30B-A3B-Instruct --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/moe_30b_general_code_route_aware_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_source_discovery_eval_plan/vllm_eval/moe_30b_general_code_route_aware

# dense_32b_reasoning_long_reasoning: Qwen/Qwen2.5-32B,Qwen/Qwen2.5-32B-Instruct,a-m-team/AM-Thinking-v1,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8000/v1 --models Qwen/Qwen2.5-32B,Qwen/Qwen2.5-32B-Instruct,a-m-team/AM-Thinking-v1,deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_32b_reasoning_long_reasoning_task_manifest.json --create-task-manifest-if-missing --output-dir results/qwen_source_discovery_eval_plan/vllm_eval/dense_32b_reasoning_long_reasoning
