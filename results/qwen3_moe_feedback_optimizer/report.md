# Qwen3 MoE Feedback Optimizer

This stage is the downstream-feedback half of the MoE averaging rule. It does not pick an algorithm name; it converts scored vLLM source-frontier regressions into bounded expert-rule updates.

- Status: `awaiting_eval`
- Candidate method: `qwen3_moe_unified_mechanism_candidate`
- Scored tasks: `0/4`
- Regression tasks: `0`
- Changed expert groups: `0`
- Materialization gate: `do_not_materialize_feedback_candidate_yet`
- Route-mass nonbase ratio after feedback: `1.0000`
- Max expected relative delta after feedback: `0.6438`
- Groups over hard cap after feedback: `0`

## Task Feedback

| task | family | status | candidate | source frontier | delta | pressure | action | paired net |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `gsm8k` | `math` | `awaiting_eval` | n/a | n/a | n/a | 0.0000 | `none` | n/a |
| `mmlu` | `general` | `awaiting_eval` | n/a | n/a | n/a | 0.0000 | `none` | n/a |
| `safety` | `safety` | `awaiting_eval` | n/a | n/a | n/a | 0.0000 | `none` | n/a |
| `humaneval_compile` | `code` | `awaiting_eval` | n/a | n/a | n/a | 0.0000 | `none` | n/a |

## Feature Updates

| task | action | affected groups | mean scale delta |
| --- | --- | ---: | ---: |
| `gsm8k` | `no_parameter_update` | 0 | 0.0000 |
| `mmlu` | `no_parameter_update` | 0 | 0.0000 |
| `safety` | `no_parameter_update` | 0 | 0.0000 |
| `humaneval_compile` | `no_parameter_update` | 0 | 0.0000 |

## Outputs

- `task_feedback`: `results/qwen3_moe_feedback_optimizer/task_feedback.csv`
- `feature_update_summary`: `results/qwen3_moe_feedback_optimizer/feature_update_summary.csv`
- `feedback_group_rules`: `results/qwen3_moe_feedback_optimizer/feedback_group_rules.csv`
- `summary`: `results/qwen3_moe_feedback_optimizer/summary.json`
- `report`: `results/qwen3_moe_feedback_optimizer/report.md`
- `tensor_rules`: `results/qwen3_moe_feedback_optimizer/tensor_rules.txt`
- `writer_command`: `results/qwen3_moe_feedback_optimizer/writer_command.txt`
- `dry_run_command`: `results/qwen3_moe_feedback_optimizer/dry_run_command.txt`
- `checkpoint_output_dir`: `results/checkpoints/qwen3_moe_feedback_mechanism_candidate`
