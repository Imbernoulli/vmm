# vLLM Downstream Eval

Status: `complete`; models: `candidate_qwen_0_5b_probe_guided_bridge_a025_b100`; endpoint: `http://127.0.0.1:8100/v1`.

## Eval Plan

| order | method | served model id | source |
| ---: | --- | --- | --- |
| 0 | manual_0 | `candidate_qwen_0_5b_probe_guided_bridge_a025_b100` | manual_args |

## Task Metrics

| model | task | examples | primary metric | value |
| --- | --- | ---: | --- | ---: |
| candidate_qwen_0_5b_probe_guided_bridge_a025_b100 | gsm8k | 64 | strict_exact | 0.062 |
| candidate_qwen_0_5b_probe_guided_bridge_a025_b100 | mmlu | 64 | accuracy | 0.250 |
| candidate_qwen_0_5b_probe_guided_bridge_a025_b100 | safety | 64 | policy_accuracy | 0.500 |
| candidate_qwen_0_5b_probe_guided_bridge_a025_b100 | humaneval_compile | 64 | compile_rate | 0.000 |

## Model Summary

| rank | model | task count | avg primary | worst primary |
| ---: | --- | ---: | ---: | ---: |
| 1 | candidate_qwen_0_5b_probe_guided_bridge_a025_b100 | 4 | 0.203 | 0.000 |

Files: `metrics.csv`, `predictions.csv`, `model_summary.csv`, `metrics.png`, `summary.json`.
