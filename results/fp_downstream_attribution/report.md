# Qwen3 MoE Generation Mechanism Attribution

This report attributes the auxiliary generation matrix by task. It does not replace the final vLLM selector.

## Summary

- Best average model: `instruct` (`0.897`)
- Avg naive drop vs pair source frontier: `0.103`
- Avg router-cal gain vs naive: `0.033`
- Avg recovery fraction: `0.324`
- HumanEval recovery fraction: `0.500`
- Router-cal beats pair frontier scores: `0/5`

## Transitions

| score | pair frontier | naive | router-cal | drop | gain | recovery | gap after router-cal | best local action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mmlu` | 0.842 | 0.758 | 0.758 | 0.083 | 0.000 | 0.000 | -0.083 | `router_calibration` |
| `gsm8k` | 0.900 | 0.825 | 0.850 | 0.075 | 0.025 | 0.333 | -0.050 | `add_thinking_source` |
| `humaneval` | 0.950 | 0.800 | 0.875 | 0.150 | 0.075 | 0.500 | -0.075 | `router_calibration` |
| `avg` | 0.897 | 0.794 | 0.828 | 0.103 | 0.033 | 0.324 | -0.069 | `add_thinking_source` |
| `worst` | 0.842 | 0.758 | 0.758 | 0.083 | 0.000 | 0.000 | -0.083 | `router_calibration` |

## Interpretation

Router calibration recovers a measurable share of the naive average regression, especially on HumanEval and GSM8K, but every score remains below the relevant source frontier. The mechanism is therefore useful as a MoE-specific repair lever, not sufficient as an acceptance rule.

## Files

- `results/fp_downstream_attribution/transition_effects.csv`
- `results/fp_downstream_attribution/summary.json`
