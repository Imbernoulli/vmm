#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a GPU host with vLLM installed.
# Usage: results/qwen3_moe_eval_budget_plan/run_eval_budget.sh [all|method_name]
# This budgeted runner keeps the same endpoints as the mechanism gate but raises --max-examples.

requested="${1:-all}"
mkdir -p results/qwen3_moe_eval_budget_plan/logs

run_one() {
  local method="$1"
  local base_url="$2"
  local serve_cmd="$3"
  local eval_cmd="$4"
  if [[ "$requested" != "all" && "$requested" != "$method" ]]; then
    return 0
  fi
  local log_path="results/qwen3_moe_eval_budget_plan/logs/${method}.serve.log"
  echo "[serve] ${method}"
  bash -lc "$serve_cmd" >"$log_path" 2>&1 &
  local server_pid=$!
  cleanup_server() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_server RETURN
  local ready=0
  for _ in $(seq 1 "${VLLM_WAIT_ATTEMPTS:-240}"); do
    if curl -sf "${base_url}/models" >/dev/null; then
      ready=1
      break
    fi
    sleep "${VLLM_WAIT_SECONDS:-5}"
  done
  if [[ "$ready" != "1" ]]; then
    echo "vLLM did not become ready for ${method}. See ${log_path}" >&2
    return 1
  fi
  echo "[eval] ${method}"
  bash -lc "$eval_cmd"
}

run_one source_qwen3_30b_instruct http://127.0.0.1:8100/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --served-model-name candidate_source_qwen3_30b_instruct --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_source_qwen3_30b_instruct --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/source_qwen3_30b_instruct --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one source_qwen3_30b_coder http://127.0.0.1:8101/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --served-model-name candidate_source_qwen3_30b_coder --host 127.0.0.1 --port 8101 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8101/v1 --models candidate_source_qwen3_30b_coder --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/source_qwen3_30b_coder --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_unified_route_guarded_candidate http://127.0.0.1:8102/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_unified_route_guarded_candidate --served-model-name candidate_qwen3_moe_unified_route_guarded_candidate --host 127.0.0.1 --port 8102 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8102/v1 --models candidate_qwen3_moe_unified_route_guarded_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_unified_route_guarded_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_audit_gated_candidate http://127.0.0.1:8103/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_audit_gated_candidate --served-model-name candidate_qwen3_moe_audit_gated_candidate --host 127.0.0.1 --port 8103 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8103/v1 --models candidate_qwen3_moe_audit_gated_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_audit_gated_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_trust_region_candidate http://127.0.0.1:8104/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_trust_region_candidate --served-model-name candidate_qwen3_moe_trust_region_candidate --host 127.0.0.1 --port 8104 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8104/v1 --models candidate_qwen3_moe_trust_region_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_trust_region_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_expert_only_trust_region_candidate http://127.0.0.1:8105/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_expert_only_trust_region_candidate --served-model-name candidate_qwen3_moe_expert_only_trust_region_candidate --host 127.0.0.1 --port 8105 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8105/v1 --models candidate_qwen3_moe_expert_only_trust_region_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_expert_only_trust_region_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_tail_trimmed_expert_only_candidate http://127.0.0.1:8106/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate --served-model-name candidate_qwen3_moe_tail_trimmed_expert_only_candidate --host 127.0.0.1 --port 8106 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8106/v1 --models candidate_qwen3_moe_tail_trimmed_expert_only_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_tail_trimmed_expert_only_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_searched_no_gt065_max_retention_candidate http://127.0.0.1:8107/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate --served-model-name candidate_qwen3_moe_searched_no_gt065_max_retention_candidate --host 127.0.0.1 --port 8107 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8107/v1 --models candidate_qwen3_moe_searched_no_gt065_max_retention_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_layer_chunk_candidate http://127.0.0.1:8108/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_layer_chunk_candidate --served-model-name candidate_qwen3_moe_layer_chunk_candidate --host 127.0.0.1 --port 8108 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8108/v1 --models candidate_qwen3_moe_layer_chunk_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_layer_chunk_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
run_one qwen3_moe_unified_mechanism_candidate http://127.0.0.1:8109/v1 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve results/checkpoints/qwen3_moe_unified_mechanism_candidate --served-model-name candidate_qwen3_moe_unified_mechanism_candidate --host 127.0.0.1 --port 8109 --dtype bfloat16 --tensor-parallel-size 4' 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8109/v1 --models candidate_qwen3_moe_unified_mechanism_candidate --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 384 --output-dir results/vllm_checkpoint_eval/qwen3_moe_unified_mechanism_candidate --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
# Skipping qwen3_moe_router_calibrated_searched_no_gt065_cap001_candidate: serve_status=pending_materialization
