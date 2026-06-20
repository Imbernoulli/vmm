# Qwen Source Frontier Eval Feedback

This artifact converts completed vLLM source-frontier eval outputs into the exact signal needed by the average optimizer: task-wise endpoint frontier, best single endpoint, and surplus against the observed merge-interference budget.

## Result

- Status: `source_frontier_feedback_promotes_average_candidate`
- Eval jobs: `3`
- Scored jobs: `3`
- Final-budget candidates: `1`
- Probe-only candidates: `1`
- Interference budget: `0.1500`
- Top scored job: `final_frontier` gate `final_average_budget_candidate` surplus `0.1167`

## Job Feedback

| job | status | models | tasks | avg gain | surplus | gate | action |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `final_frontier` | `scored_complete` | 3/3 | 3/3 | 0.2667 | 0.1167 | `final_average_budget_candidate` | promote this source set to same-shape average materialization and locked candidate eval |
| `probe_frontier` | `scored_complete` | 2/2 | 2/2 | 0.1000 | -0.0500 | `probe_only_below_interference_budget` | keep as endpoint-expansion/probe-only; do not spend final average budget yet |
| `reject_frontier` | `scored_complete` | 2/2 | 2/2 | 0.0000 | -0.1500 | `source_frontier_not_better_than_endpoint` | prefer best endpoint and search more complementary sources before averaging |

## Task Frontier

| job | task | source model | score | metric |
| --- | --- | --- | ---: | --- |
| `final_frontier` | `gsm8k` | `model_a` | 0.9000 | `accuracy` |
| `final_frontier` | `humaneval_compile` | `model_b` | 0.9000 | `accuracy` |
| `final_frontier` | `mmlu` | `model_c` | 0.9000 | `accuracy` |
| `probe_frontier` | `gsm8k` | `model_a` | 0.8000 | `accuracy` |
| `probe_frontier` | `mmlu` | `model_b` | 0.8000 | `accuracy` |
| `reject_frontier` | `gsm8k` | `model_a` | 0.8000 | `accuracy` |
| `reject_frontier` | `mmlu` | `model_a` | 0.8000 | `accuracy` |

## Outputs

- `results/qwen_source_frontier_eval_feedback_smoke/job_feedback.csv`
- `results/qwen_source_frontier_eval_feedback_smoke/task_frontier.csv`
- `results/qwen_source_frontier_eval_feedback_smoke/summary.json`

## Smoke Checks

- Smoke status: `smoke_passed`

| check | passed |
| --- | --- |
| `final_frontier_promoted` | `True` |
| `probe_frontier_probe_only` | `True` |
| `reject_frontier_rejected` | `True` |
| `scored_all_jobs` | `True` |
