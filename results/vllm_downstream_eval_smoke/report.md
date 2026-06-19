# vLLM Downstream Eval

Status: `complete`; models: `mock-good, mock-bad`; endpoint: `MOCK_BASE_URL`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `mock-good` | manual_args |
| 1 | manual_1 | `mock-bad` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| mock-good | gsm8k | 2 | strict_exact | 1.000 |
| mock-good | mmlu | 2 | accuracy | 1.000 |
| mock-good | safety | 2 | policy_accuracy | 1.000 |
| mock-good | humaneval_compile | 2 | compile_rate | 1.000 |
| mock-bad | gsm8k | 2 | strict_exact | 0.000 |
| mock-bad | mmlu | 2 | accuracy | 0.000 |
| mock-bad | safety | 2 | policy_accuracy | 0.000 |
| mock-bad | humaneval_compile | 2 | compile_rate | 0.000 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | mock-good | 4 | 1.000 | 1.000 |
| 2 | mock-bad | 4 | 0.000 | 0.000 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
