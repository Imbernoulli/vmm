#!/usr/bin/env bash
set -euo pipefail

# Adaptive Qwen3 MoE eval runner. Start only one vLLM server at a time.
# Top action: run_or_extend_source_control_probe / source_qwen3_30b_instruct

method="${1:-}"
if [[ -z "$method" ]]; then
  echo "Usage: $0 METHOD" >&2
  echo "Runnable methods:" >&2
  echo "  source_qwen3_30b_instruct (run_or_extend_source_control_probe)" >&2
  echo "  source_qwen3_30b_coder (run_or_extend_source_control_probe)" >&2
  exit 2
fi

if [[ "$method" == source_qwen3_30b_instruct ]]; then
  mkdir -p results/qwen3_moe_adaptive_eval_schedule/logs
  bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --served-model-name candidate_source_qwen3_30b_instruct --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 4' >results/qwen3_moe_adaptive_eval_schedule/logs/source_qwen3_30b_instruct.serve.log 2>&1 &
  server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8100/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8100/v1/models >/dev/null
  bash -lc 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 --models candidate_source_qwen3_30b_instruct --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/source_qwen3_30b_instruct --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
  exit 0
fi

if [[ "$method" == source_qwen3_30b_coder ]]; then
  mkdir -p results/qwen3_moe_adaptive_eval_schedule/logs
  bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 --served-model-name candidate_source_qwen3_30b_coder --host 127.0.0.1 --port 8101 --dtype bfloat16 --tensor-parallel-size 4' >results/qwen3_moe_adaptive_eval_schedule/logs/source_qwen3_30b_coder.serve.log 2>&1 &
  server_pid=$!
  cleanup() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:8101/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8101/v1/models >/dev/null
  bash -lc 'python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8101/v1 --models candidate_source_qwen3_30b_coder --tasks gsm8k,mmlu,safety,humaneval_compile --example-source datasets --max-examples 64 --output-dir results/vllm_checkpoint_eval/source_qwen3_30b_coder --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing'
  exit 0
fi

echo "Unknown method: $method" >&2
exit 2
