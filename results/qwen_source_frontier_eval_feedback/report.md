# Qwen Source Frontier Eval Feedback

This artifact converts completed vLLM source-frontier eval outputs into the exact signal needed by the average optimizer: task-wise endpoint frontier, best single endpoint, and surplus against the observed merge-interference budget.

## Result

- Status: `awaiting_vllm_source_frontier_results`
- Eval jobs: `4`
- Scored jobs: `0`
- Final-budget candidates: `0`
- Probe-only candidates: `0`
- Interference budget: `0.0694`
- Top scored job: `None` gate `None` surplus `n/a`

## Job Feedback

| job | status | models | tasks | avg gain | surplus | gate | action |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `measured_coder_thinking_source_frontier` | `missing_eval_output` | 0/2 | 0/3 | n/a | n/a | `awaiting_vllm_eval` | run planned vLLM eval and rerun feedback builder |
| `dense_7b_general_code_math_reasoning` | `missing_eval_output` | 0/4 | 0/4 | n/a | n/a | `awaiting_vllm_eval` | run planned vLLM eval and rerun feedback builder |
| `moe_30b_general_code_route_aware` | `missing_eval_output` | 0/3 | 0/4 | n/a | n/a | `awaiting_vllm_eval` | run planned vLLM eval and rerun feedback builder |
| `dense_32b_reasoning_long_reasoning` | `missing_eval_output` | 0/4 | 0/4 | n/a | n/a | `awaiting_vllm_eval` | run planned vLLM eval and rerun feedback builder |

## Task Frontier

| job | task | source model | score | metric |
| --- | --- | --- | ---: | --- |

## Outputs

- `results/qwen_source_frontier_eval_feedback/job_feedback.csv`
- `results/qwen_source_frontier_eval_feedback/task_frontier.csv`
- `results/qwen_source_frontier_eval_feedback/summary.json`
