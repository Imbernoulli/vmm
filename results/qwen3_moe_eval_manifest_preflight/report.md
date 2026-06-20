# Qwen3 MoE Eval Task Manifest Preflight

这个 preflight 在远端 vLLM eval 前检查 final selector 需要的 canonical task manifest。它验证所有 planned rows 是否共享同一路径，并检查 manifest 里的每个任务样本数是否达到预算里的 achievable example count。

- Status: `task_manifest_ready`
- Canonical task manifest: `results/qwen3_moe_mechanism_eval_gate/task_manifest.json`
- Manifest exists: `True`
- Methods aligned: `14/14`
- Tasks sufficient: `4/4`
- Total manifest / required examples: `1316/1316`

## Prepare Command

```bash
python scripts/run_vllm_downstream_eval.py --tasks gsm8k,humaneval_compile,mmlu,safety --example-source datasets --max-examples 384 --task-manifest results/qwen3_moe_mechanism_eval_gate/task_manifest.json --create-task-manifest-if-missing --prepare-task-manifest-only
```

## Task Checks

| task | required | manifest | present | sufficient | deficit |
| --- | ---: | ---: | --- | --- | ---: |
| `gsm8k` | 384 | 384 | `True` | `True` | 0 |
| `humaneval_compile` | 164 | 164 | `True` | `True` | 0 |
| `mmlu` | 384 | 384 | `True` | `True` | 0 |
| `safety` | 384 | 384 | `True` | `True` | 0 |

## Outputs

- `results/qwen3_moe_eval_manifest_preflight/task_manifest_checks.csv`
- `results/qwen3_moe_eval_manifest_preflight/prepare_manifest_command.txt`
- `results/qwen3_moe_eval_manifest_preflight/summary.json`
