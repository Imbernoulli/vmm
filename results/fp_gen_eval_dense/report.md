# Generation Exact-Answer Merge Eval

这个实验是一个小型生成式 smoke test：不用外部 dataset，也不执行模型生成的代码；只检查数学题和代码输出题的最终答案。

## Result

| method | math | code_output | avg | worst |
| --- | ---: | ---: | ---: | ---: |
| `base` | 0.000 | 1.000 | 0.500 | 0.000 |
| `instruct` | 0.000 | 1.000 | 0.500 | 0.000 |
| `coder` | 0.500 | 0.500 | 0.500 | 0.500 |
| `linear` | 0.000 | 0.000 | 0.000 | 0.000 |
| `unified` | 0.000 | 1.000 | 0.500 | 0.000 |

## Interpretation

The exact-answer generation smoke follows the same boundary as the NLL selector: it is a fast behavioral check, not proof that a merge beats all endpoints. The unified candidate beats the linear midpoint on this smoke slice.

## Files

- `summary.json`
- `predictions.csv`
- `method_metrics.csv`
