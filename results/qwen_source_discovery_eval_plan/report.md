# Qwen Source Discovery vLLM Eval Plan

## Result

这个计划把 source discovery 的候选源集合落成同一 task manifest 的 vLLM jobs。当前共有 `4` 个 endpoint/frontier eval jobs；最高优先 job 是 `measured_coder_thinking_source_frontier`，场景 `measured_qwen3_moe_source_set`，任务 `mmlu,gsm8k,humaneval_compile`。

关键修正：runner 的 HumanEval 任务名是 `humaneval_compile`，这里所有 vLLM 命令都使用该任务名，避免把 HumanEval 计划写成 runner 不会执行的 `humaneval`。

## vLLM Jobs

| rank | job | scenario | models | tasks | status | extra gain needed |
| ---: | --- | --- | ---: | --- | --- | ---: |
| 1 | `measured_coder_thinking_source_frontier` | `measured_qwen3_moe_source_set` | 2 | `mmlu,gsm8k,humaneval_compile` | `ready_for_probe_only_endpoint_expansion` | 0.0611 |
| 2 | `dense_7b_general_code_math_reasoning` | `dense_7b_general_code_math_reasoning` | 4 | `mmlu,gsm8k,humaneval_compile,safety` | `ready_for_vllm_endpoint_eval` | 0.0694 |
| 3 | `moe_30b_general_code_route_aware` | `moe_30b_general_code_route_aware` | 3 | `mmlu,gsm8k,humaneval_compile,safety` | `ready_for_vllm_endpoint_eval` | 0.0694 |
| 4 | `dense_32b_reasoning_long_reasoning` | `dense_32b_reasoning_long_reasoning` | 4 | `mmlu,gsm8k,humaneval_compile,safety` | `ready_for_vllm_endpoint_eval` | 0.0694 |

## Manifest Jobs

| job | kind | manifest | tasks | source | command |
| --- | --- | --- | --- | --- | --- |
| `measured_coder_thinking_source_frontier` | `production` | `results/qwen_source_discovery_eval_plan/task_manifests/measured_coder_thinking_source_frontier_task_manifest.json` | `mmlu,gsm8k,humaneval_compile` | `datasets` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/measured_coder_thinking_source_frontier_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `measured_coder_thinking_source_frontier` | `smoke` | `results/qwen_source_discovery_eval_plan/task_manifests/measured_coder_thinking_source_frontier_smoke_task_manifest.json` | `mmlu,gsm8k,humaneval_compile` | `builtin` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile --example-source builtin --max-examples 2 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/measured_coder_thinking_source_frontier_smoke_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `dense_7b_general_code_math_reasoning` | `production` | `results/qwen_source_discovery_eval_plan/task_manifests/dense_7b_general_code_math_reasoning_task_manifest.json` | `mmlu,gsm8k,humaneval_compile,safety` | `datasets` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_7b_general_code_math_reasoning_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `dense_7b_general_code_math_reasoning` | `smoke` | `results/qwen_source_discovery_eval_plan/task_manifests/dense_7b_general_code_math_reasoning_smoke_task_manifest.json` | `mmlu,gsm8k,humaneval_compile,safety` | `builtin` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source builtin --max-examples 2 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_7b_general_code_math_reasoning_smoke_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `moe_30b_general_code_route_aware` | `production` | `results/qwen_source_discovery_eval_plan/task_manifests/moe_30b_general_code_route_aware_task_manifest.json` | `mmlu,gsm8k,humaneval_compile,safety` | `datasets` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/moe_30b_general_code_route_aware_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `moe_30b_general_code_route_aware` | `smoke` | `results/qwen_source_discovery_eval_plan/task_manifests/moe_30b_general_code_route_aware_smoke_task_manifest.json` | `mmlu,gsm8k,humaneval_compile,safety` | `builtin` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source builtin --max-examples 2 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/moe_30b_general_code_route_aware_smoke_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `dense_32b_reasoning_long_reasoning` | `production` | `results/qwen_source_discovery_eval_plan/task_manifests/dense_32b_reasoning_long_reasoning_task_manifest.json` | `mmlu,gsm8k,humaneval_compile,safety` | `datasets` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source datasets --max-examples 256 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_32b_reasoning_long_reasoning_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |
| `dense_32b_reasoning_long_reasoning` | `smoke` | `results/qwen_source_discovery_eval_plan/task_manifests/dense_32b_reasoning_long_reasoning_smoke_task_manifest.json` | `mmlu,gsm8k,humaneval_compile,safety` | `builtin` | `python scripts/run_vllm_downstream_eval.py --tasks mmlu,gsm8k,humaneval_compile,safety --example-source builtin --max-examples 2 --subjects all --task-manifest results/qwen_source_discovery_eval_plan/task_manifests/dense_32b_reasoning_long_reasoning_smoke_task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only` |

## Outputs

- `results/qwen_source_discovery_eval_plan/vllm_eval_jobs.csv`
- `results/qwen_source_discovery_eval_plan/manifest_jobs.csv`
- `results/qwen_source_discovery_eval_plan/run_source_frontier_eval.sh`
- `results/qwen_source_discovery_eval_plan/summary.json`
- `results/qwen_source_discovery_eval_plan/report.md`
