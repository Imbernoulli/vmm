# vLLM Downstream Eval

Status: `complete`; models: `candidate_qwen_0_5b_instruct_coder_uniform_average`; endpoint: `http://127.0.0.1:8100/v1`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `candidate_qwen_0_5b_instruct_coder_uniform_average` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| candidate_qwen_0_5b_instruct_coder_uniform_average | gsm8k | 64 | strict_exact | 0.000 |
| candidate_qwen_0_5b_instruct_coder_uniform_average | mmlu | 64 | accuracy | 0.219 |
| candidate_qwen_0_5b_instruct_coder_uniform_average | safety | 64 | policy_accuracy | 0.500 |
| candidate_qwen_0_5b_instruct_coder_uniform_average | humaneval_compile | 64 | compile_rate | 0.000 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | candidate_qwen_0_5b_instruct_coder_uniform_average | 4 | 0.180 | 0.000 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
