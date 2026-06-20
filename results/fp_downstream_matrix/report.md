# Qwen3 MoE Downstream Generation Matrix

This is an auxiliary transformers-generation matrix for mechanism evidence. It is not the final vLLM selector.

## Summary

- Models: `7`
- Best average model: `instruct` (`0.897`)
- Instruct+Coder avg: `0.794`
- Instruct+Coder + router-cal avg: `0.828`
- Router-cal avg gain: `0.033`
- Router-cal HumanEval gain: `0.075`
- Router-cal gap to best parent avg: `-0.069`

## Matrix

| model | MMLU | GSM8K | HumanEval | avg | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.767 | 0.725 | 0.825 | 0.772 | 0.725 |
| `instruct` | 0.842 | 0.900 | 0.950 | 0.897 | 0.842 |
| `coder` | 0.767 | 0.900 | 0.950 | 0.872 | 0.767 |
| `thinking` | 0.792 | 0.750 | 0.900 | 0.814 | 0.750 |
| `merge_instruct+coder` | 0.758 | 0.825 | 0.800 | 0.794 | 0.758 |
| `merge_instruct+coder_routercal` | 0.758 | 0.850 | 0.875 | 0.828 | 0.758 |
| `merge_instruct+coder+thinking` | 0.750 | 0.900 | 0.850 | 0.833 | 0.750 |

## Interpretation

The generation matrix confirms the mechanism: naive Instruct+Coder averaging dilutes the dominant Instruct endpoint, while router calibration recovers part of the lost GSM8K and HumanEval performance. It still does not beat the best parent on this task set, so it is evidence for router calibration as a lever, not acceptance of an average candidate.

## Files

- `results/fp_downstream_matrix/matrix.csv`
- `results/fp_downstream_matrix/summary.json`
- `results/fp_downstream_matrix/downstream_matrix.png`
