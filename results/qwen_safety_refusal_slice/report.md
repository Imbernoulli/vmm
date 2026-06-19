# Qwen Safety / Refusal Slice

This run evaluates interpolated Qwen2.5-1.5B weights on a small BeaverTails safety/refusal slice. It does not generate model completions. It scores safe prompts against safe dataset responses and unsafe prompts against a fixed refusal target by response-only NLL.

Prompts are not written to the per-example artifact; `predictions.csv` stores only a prompt hash, category summary, and NLL metrics.

## Key Results

- Best lambda by average safety NLL: `0.750` with avg NLL 2.546.

| lambda | safe response NLL | unsafe refusal NLL | avg safety NLL | safe / unsafe examples |
| ---: | ---: | ---: | ---: | ---: |
| 0.000 | 1.977 | 3.661 | 2.819 | 12/12 |
| 0.750 | 1.783 | 3.310 | 2.546 | 12/12 |
| 1.000 | 1.787 | 3.780 | 2.783 | 12/12 |

## Files

- `metrics.csv`: per-lambda safe-response and unsafe-refusal NLL metrics.
- `predictions.csv`: per-example prompt hashes and target NLLs.
- `safety_refusal_nll.png`: safety/refusal path plot.

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
  "max_safe": 12,
  "max_unsafe": 12,
  "seed": 47,
  "split": "30k_test",
  "max_length": 768,
  "device": "cuda:3",
  "dtype": "bfloat16",
  "use_chat_template": true,
  "add_eos": true
}
```
