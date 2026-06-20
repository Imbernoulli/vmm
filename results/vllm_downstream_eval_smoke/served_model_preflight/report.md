# vLLM Served Model Preflight

这个 preflight 在真正跑 downstream eval 前检查两件事：计划中的 served model id 是否出现在 vLLM `/models`，以及每个 job 的 task manifest 是否已经准备好并覆盖对应任务。

- Status: `served_model_preflight_ready`
- Endpoint probe: `ok`
- Required served models: `2`
- Endpoint served models: `2`
- Missing required models: `0`
- Manifest ready: `0/0`

## Missing Models

No required served models are missing from the endpoint model list.

## Manifest Checks

| job | kind | exists | ready | missing tasks | insufficient tasks |
| --- | --- | --- | --- | --- | --- |

## Outputs

- `results/vllm_downstream_eval_smoke/served_model_preflight/required_served_models.csv`
- `results/vllm_downstream_eval_smoke/served_model_preflight/task_manifest_checks.csv`
- `results/vllm_downstream_eval_smoke/served_model_preflight/summary.json`
