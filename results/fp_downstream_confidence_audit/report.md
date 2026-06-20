# Qwen3 MoE Downstream Confidence Audit

This audit adds aggregate uncertainty bounds to the auxiliary generation matrix. It uses Wilson intervals from the observed task accuracies and does not replace the final paired vLLM selector.

## Summary

- Task sample sizes: `{"gsm8k": 40, "humaneval": 40, "mmlu": 120}`
- Router-cal positive tasks vs naive: `2/3`
- Router-cal confident positive tasks vs naive: `0/3`
- Router-cal confident source-frontier wins: `0/3`
- Avg diff vs naive: `0.033` with conservative interval `[-0.170, 0.231]`
- Avg gap vs pair source frontier: `-0.069` with conservative interval `[-0.240, 0.110]`

## Comparison Intervals

| comparison | score | candidate | reference | diff | conservative diff interval | status |
| --- | --- | --- | --- | ---: | ---: | --- |
| `routercal_vs_naive` | `mmlu` | `merge_instruct+coder_routercal` | `merge_instruct+coder` | 0.000 | [-0.152, 0.152] | `tie_with_uncertainty` |
| `routercal_vs_pair_source_frontier` | `mmlu` | `merge_instruct+coder_routercal` | `instruct` | -0.083 | [-0.222, 0.060] | `directional_negative_not_confident` |
| `routercal_vs_naive` | `gsm8k` | `merge_instruct+coder_routercal` | `merge_instruct+coder` | 0.025 | [-0.203, 0.249] | `directional_positive_not_confident` |
| `routercal_vs_pair_source_frontier` | `gsm8k` | `merge_instruct+coder_routercal` | `instruct` | -0.050 | [-0.251, 0.160] | `directional_negative_not_confident` |
| `routercal_vs_naive` | `humaneval` | `merge_instruct+coder_routercal` | `merge_instruct+coder` | 0.075 | [-0.156, 0.293] | `directional_positive_not_confident` |
| `routercal_vs_pair_source_frontier` | `humaneval` | `merge_instruct+coder_routercal` | `instruct` | -0.075 | [-0.247, 0.110] | `directional_negative_not_confident` |
| `routercal_vs_naive` | `avg` | `merge_instruct+coder_routercal` | `merge_instruct+coder` | 0.033 | [-0.170, 0.231] | `directional_positive_not_confident` |
| `routercal_vs_pair_source_frontier` | `avg` | `merge_instruct+coder_routercal` | `instruct` | -0.069 | [-0.240, 0.110] | `directional_negative_not_confident` |

## Interpretation

Router calibration has positive point-estimate gains over the naive pair average on some tasks, but the conservative aggregate intervals do not yet make those gains a confident acceptance signal, and no task confidently beats the source frontier. Use it as a mechanism probe and candidate repair lever, then require matched vLLM paired prediction gates.

## Files

- `results/fp_downstream_confidence_audit/model_task_intervals.csv`
- `results/fp_downstream_confidence_audit/comparison_intervals.csv`
- `results/fp_downstream_confidence_audit/summary.json`
