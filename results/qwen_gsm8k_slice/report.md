# Qwen GSM8K Benchmark Slice

This run evaluates interpolated Qwen2.5-1.5B weights on a small cached GSM8K test slice. It is a benchmark-slice diagnostic, not a full GSM8K run.

The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Each model generates an answer. The strict score requires the model to emit the GSM8K `#### <number>` format; the loose score falls back to the last generated number when that marker is missing.

## Key Results

- Best lambda by strict exact match: `0.750` with exact match 0.083 (1/12).

| lambda | strict exact | loose exact | hash format | strict correct / total | avg generated chars |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000 | 0.083 | 0.000 | 0/12 | 583.8 |
| 0.750 | 0.083 | 0.250 | 0.083 | 1/12 | 508.7 |
| 1.000 | 0.083 | 0.167 | 0.083 | 1/12 | 555.8 |

## Files

- `metrics.csv`: per-lambda strict and loose exact-match metrics.
- `predictions.csv`: per-example generations and extracted answers.
- `gsm8k_exact_match.png`: exact-match path plot.

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
  "max_examples": 12,
  "seed": 31,
  "max_prompt_tokens": 768,
  "max_new_tokens": 160,
  "device": "cuda:2",
  "dtype": "bfloat16",
  "use_chat_template": true
}
```
