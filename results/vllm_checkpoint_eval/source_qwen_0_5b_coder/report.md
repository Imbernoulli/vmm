# vLLM Downstream Eval

Status: `complete`; models: `source_qwen_0_5b_coder`; endpoint: `http://127.0.0.1:8100/v1`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `source_qwen_0_5b_coder` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| source_qwen_0_5b_coder | gsm8k | 64 | strict_exact | 0.031 |
| source_qwen_0_5b_coder | mmlu | 64 | accuracy | 0.234 |
| source_qwen_0_5b_coder | safety | 64 | policy_accuracy | 0.531 |
| source_qwen_0_5b_coder | humaneval_compile | 64 | compile_rate | 0.000 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | source_qwen_0_5b_coder | 4 | 0.199 | 0.000 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
