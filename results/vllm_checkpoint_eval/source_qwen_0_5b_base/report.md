# vLLM Downstream Eval

Status: `complete`; models: `source_qwen_0_5b_base`; endpoint: `http://127.0.0.1:8100/v1`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `source_qwen_0_5b_base` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| source_qwen_0_5b_base | gsm8k | 64 | strict_exact | 0.094 |
| source_qwen_0_5b_base | mmlu | 64 | accuracy | 0.312 |
| source_qwen_0_5b_base | safety | 64 | policy_accuracy | 0.484 |
| source_qwen_0_5b_base | humaneval_compile | 64 | compile_rate | 0.609 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | source_qwen_0_5b_base | 4 | 0.375 | 0.094 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
