# vLLM Downstream Eval

Status: `complete`; models: `source_qwen_0_5b_instruct`; endpoint: `http://127.0.0.1:8100/v1`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `source_qwen_0_5b_instruct` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| source_qwen_0_5b_instruct | gsm8k | 64 | strict_exact | 0.016 |
| source_qwen_0_5b_instruct | mmlu | 64 | accuracy | 0.344 |
| source_qwen_0_5b_instruct | safety | 64 | policy_accuracy | 0.547 |
| source_qwen_0_5b_instruct | humaneval_compile | 64 | compile_rate | 0.000 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | source_qwen_0_5b_instruct | 4 | 0.227 | 0.000 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
