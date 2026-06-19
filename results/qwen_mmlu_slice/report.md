# Qwen MMLU Benchmark Slice

This run evaluates interpolated Qwen2.5-1.5B weights on a small MMLU test slice. It is a benchmark-slice diagnostic, not a full MMLU run.

The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Each multiple-choice question is scored by the log-likelihood of answer letters A-D; the predicted answer is the lowest-NLL letter.

## Key Results

- Best lambda by accuracy: `0.750` with accuracy 0.750 (18/24).

| lambda | accuracy | correct / total | avg gold NLL | avg predicted NLL | avg margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.292 | 7/24 | 6.048 | 3.975 | 2.628 |
| 0.750 | 0.750 | 18/24 | 0.932 | 0.292 | 3.073 |
| 1.000 | 0.667 | 16/24 | 1.282 | 0.204 | 4.443 |

## Files

- `metrics.csv`: per-lambda multiple-choice metrics.
- `predictions.csv`: per-example answer-letter NLLs and predictions.
- `mmlu_accuracy.png`: accuracy and NLL path plot.

## Configuration

```json
{
  "base": "/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-1.5B",
  "expert": "/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct",
  "tokenizer": "/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct",
  "lambdas": [
    0.0,
    0.75,
    1.0
  ],
  "max_examples": 24,
  "seed": 37,
  "subjects": "all",
  "max_length": 768,
  "device": "cuda:3",
  "dtype": "bfloat16",
  "use_chat_template": true
}
```
