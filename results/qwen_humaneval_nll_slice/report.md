# Qwen HumanEval NLL Slice

This run evaluates interpolated Qwen2.5-1.5B weights on a small HumanEval code-completion slice. It scores the canonical solutions by token-level negative log-likelihood; it does not execute generated code or report pass@k.

The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Lower NLL is better.

## Key Results

- Best lambda by token-weighted solution NLL: `1.000` with NLL 0.964.

| lambda | examples | solution tokens | avg solution NLL | mean task NLL | median task NLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 24 | 1280 | 0.997 | 1.368 | 1.197 |
| 0.750 | 24 | 1280 | 0.971 | 1.318 | 1.168 |
| 1.000 | 24 | 1280 | 0.964 | 1.299 | 1.169 |

## Files

- `metrics.csv`: per-lambda HumanEval NLL metrics.
- `predictions.csv`: per-task canonical-solution NLLs.
- `humaneval_nll.png`: NLL path plot.

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
  "seed": 43,
  "max_length": 1024,
  "device": "cuda:3",
  "dtype": "bfloat16",
  "use_chat_template": false,
  "add_eos": true
}
```
