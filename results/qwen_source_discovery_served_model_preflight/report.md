# vLLM Served Model Preflight

这个 preflight 在真正跑 downstream eval 前检查两件事：计划中的 served model id 是否出现在 vLLM `/models`，以及每个 job 的 task manifest 是否已经准备好并覆盖对应任务。

- Status: `static_preflight_ready_waiting_for_endpoint_model_list`
- Endpoint probe: `not_requested`
- Required served models: `12`
- Endpoint served models: `0`
- Missing required models: `0`
- Manifest ready: `4/8`

## Missing Models

Endpoint model list was not available; pass `--base-url` against the running vLLM server.

## Manifest Checks

| job | kind | exists | ready | missing tasks | insufficient tasks |
| --- | --- | --- | --- | --- | --- |
| `measured_coder_thinking_source_frontier` | `production` | `False` | `False` | `mmlu,gsm8k,humaneval_compile` | `` |
| `measured_coder_thinking_source_frontier` | `smoke` | `True` | `True` | `` | `` |
| `dense_7b_general_code_math_reasoning` | `production` | `False` | `False` | `mmlu,gsm8k,humaneval_compile,safety` | `` |
| `dense_7b_general_code_math_reasoning` | `smoke` | `True` | `True` | `` | `` |
| `moe_30b_general_code_route_aware` | `production` | `False` | `False` | `mmlu,gsm8k,humaneval_compile,safety` | `` |
| `moe_30b_general_code_route_aware` | `smoke` | `True` | `True` | `` | `` |
| `dense_32b_reasoning_long_reasoning` | `production` | `False` | `False` | `mmlu,gsm8k,humaneval_compile,safety` | `` |
| `dense_32b_reasoning_long_reasoning` | `smoke` | `True` | `True` | `` | `` |

## Outputs

- `results/qwen_source_discovery_served_model_preflight/required_served_models.csv`
- `results/qwen_source_discovery_served_model_preflight/task_manifest_checks.csv`
- `results/qwen_source_discovery_served_model_preflight/summary.json`
