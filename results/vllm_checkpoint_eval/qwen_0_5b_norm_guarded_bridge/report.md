# vLLM Downstream Eval

Status: `complete`; models: `candidate_qwen_0_5b_norm_guarded_bridge`; endpoint: `http://127.0.0.1:8100/v1`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `candidate_qwen_0_5b_norm_guarded_bridge` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| candidate_qwen_0_5b_norm_guarded_bridge | gsm8k | 64 | strict_exact | 0.047 |
| candidate_qwen_0_5b_norm_guarded_bridge | mmlu | 64 | accuracy | 0.219 |
| candidate_qwen_0_5b_norm_guarded_bridge | safety | 64 | policy_accuracy | 0.547 |
| candidate_qwen_0_5b_norm_guarded_bridge | humaneval_compile | 64 | compile_rate | 0.000 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | candidate_qwen_0_5b_norm_guarded_bridge | 4 | 0.203 | 0.000 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
