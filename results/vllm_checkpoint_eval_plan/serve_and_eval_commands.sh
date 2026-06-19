#!/usr/bin/env bash
set -euo pipefail

# Command plan only. Copy the relevant commands after the checkpoint exists.
# Start one vLLM server per candidate in a separate terminal or job manager,
# then run the paired eval command after /v1/models is reachable.

# [0] source_qwen3_30b_instruct - ready_to_host
# Checkpoint: /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --served-model-name candidate_source_qwen3_30b_instruct --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8100/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_source_qwen3_30b_instruct --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/source_qwen3_30b_instruct

# [1] source_qwen3_30b_coder - ready_to_host
# Checkpoint: /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --served-model-name candidate_source_qwen3_30b_coder --host 127.0.0.1 --port 8101 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8101/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8101/v1 --models candidate_source_qwen3_30b_coder --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/source_qwen3_30b_coder

# [2] qwen3_moe_unified_route_guarded_candidate - ready_to_host
# Checkpoint: results/checkpoints/qwen3_moe_unified_route_guarded_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_unified_route_guarded_candidate --served-model-name candidate_qwen3_moe_unified_route_guarded_candidate --host 127.0.0.1 --port 8102 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8102/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8102/v1 --models candidate_qwen3_moe_unified_route_guarded_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen3_moe_unified_route_guarded_candidate

# [3] qwen3_moe_audit_gated_candidate - ready_to_host
# Checkpoint: results/checkpoints/qwen3_moe_audit_gated_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_audit_gated_candidate --served-model-name candidate_qwen3_moe_audit_gated_candidate --host 127.0.0.1 --port 8103 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8103/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8103/v1 --models candidate_qwen3_moe_audit_gated_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen3_moe_audit_gated_candidate

# [4] qwen3_moe_trust_region_candidate - ready_to_host
# Checkpoint: results/checkpoints/qwen3_moe_trust_region_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_trust_region_candidate --served-model-name candidate_qwen3_moe_trust_region_candidate --host 127.0.0.1 --port 8104 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8104/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8104/v1 --models candidate_qwen3_moe_trust_region_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen3_moe_trust_region_candidate

# [5] qwen3_moe_expert_only_trust_region_candidate - ready_to_host
# Checkpoint: results/checkpoints/qwen3_moe_expert_only_trust_region_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_expert_only_trust_region_candidate --served-model-name candidate_qwen3_moe_expert_only_trust_region_candidate --host 127.0.0.1 --port 8105 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8105/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8105/v1 --models candidate_qwen3_moe_expert_only_trust_region_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen3_moe_expert_only_trust_region_candidate

# [6] qwen3_moe_tail_trimmed_expert_only_candidate - ready_to_host
# Checkpoint: results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate --served-model-name candidate_qwen3_moe_tail_trimmed_expert_only_candidate --host 127.0.0.1 --port 8106 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8106/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8106/v1 --models candidate_qwen3_moe_tail_trimmed_expert_only_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen3_moe_tail_trimmed_expert_only_candidate

# [7] qwen3_moe_searched_no_gt065_max_retention_candidate - ready_to_host
# Checkpoint: results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate --served-model-name candidate_qwen3_moe_searched_no_gt065_max_retention_candidate --host 127.0.0.1 --port 8107 --dtype bfloat16 --tensor-parallel-size 4

# Wait:
# curl -sf http://127.0.0.1:8107/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8107/v1 --models candidate_qwen3_moe_searched_no_gt065_max_retention_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate

# [8] qwen_0_5b_instruct_coder_uniform_average - ready_to_host
# Checkpoint: results/checkpoints/qwen_0_5b_instruct_coder_uniform_average
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/qwen_0_5b_instruct_coder_uniform_average --served-model-name candidate_qwen_0_5b_instruct_coder_uniform_average --host 127.0.0.1 --port 8108 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8108/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8108/v1 --models candidate_qwen_0_5b_instruct_coder_uniform_average --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average

# [9] moe_route_aware_candidate - checkpoint_missing_until_materialized
# Checkpoint: results/checkpoints/moe_route_aware_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/moe_route_aware_candidate --served-model-name candidate_moe_route_aware_candidate --host 127.0.0.1 --port 8109 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8109/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8109/v1 --models candidate_moe_route_aware_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/moe_route_aware_candidate

# [10] moe_bias_calibrated_candidate - checkpoint_missing_until_materialized
# Checkpoint: results/checkpoints/moe_bias_calibrated_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/moe_bias_calibrated_candidate --served-model-name candidate_moe_bias_calibrated_candidate --host 127.0.0.1 --port 8110 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8110/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8110/v1 --models candidate_moe_bias_calibrated_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/moe_bias_calibrated_candidate

# [11] toy_moe_expert_weight_candidate - not_vllm_loadable_toy_candidate
# Checkpoint: results/checkpoints/toy_moe_expert_weight_candidate
# Serve:
# CUDA_VISIBLE_DEVICES=0 vllm serve results/checkpoints/toy_moe_expert_weight_candidate --served-model-name candidate_toy_moe_expert_weight_candidate --host 127.0.0.1 --port 8111 --dtype bfloat16 --tensor-parallel-size 1

# Wait:
# curl -sf http://127.0.0.1:8111/v1/models >/dev/null

# Eval:
# python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8111/v1 --models candidate_toy_moe_expert_weight_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/toy_moe_expert_weight_candidate

