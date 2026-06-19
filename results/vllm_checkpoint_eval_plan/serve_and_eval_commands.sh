#!/usr/bin/env bash
set -euo pipefail

# Command plan only. Copy the relevant commands after the checkpoint exists.
# Start one vLLM server per candidate in a separate terminal or job manager,
# then run the paired eval command after /v1/models is reachable.

# [0] moe_route_aware_candidate - checkpoint_missing_until_materialized
# Checkpoint: results/checkpoints/moe_route_aware_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/moe_route_aware_candidate --served-model-name candidate_moe_route_aware_candidate --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8100/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_moe_route_aware_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/moe_route_aware_candidate

# [1] moe_bias_calibrated_candidate - checkpoint_missing_until_materialized
# Checkpoint: results/checkpoints/moe_bias_calibrated_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/moe_bias_calibrated_candidate --served-model-name candidate_moe_bias_calibrated_candidate --host 127.0.0.1 --port 8101 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8101/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8101/v1 --models candidate_moe_bias_calibrated_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/moe_bias_calibrated_candidate

# [2] toy_moe_expert_weight_candidate - not_vllm_loadable_toy_candidate
# Checkpoint: results/checkpoints/toy_moe_expert_weight_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/toy_moe_expert_weight_candidate --served-model-name candidate_toy_moe_expert_weight_candidate --host 127.0.0.1 --port 8102 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8102/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8102/v1 --models candidate_toy_moe_expert_weight_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/toy_moe_expert_weight_candidate

