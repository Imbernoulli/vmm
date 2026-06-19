# Qwen Multi-Expert Merge

This run evaluates a real multi-expert Qwen merge plane. The base is Qwen2.5-0.5B; the two experts are Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct. The merge plane is `base + alpha * instruct_delta + beta * coder_delta`.

Metrics are token-level NLLs on small general, instruction-response, and code-response slices. Lower is better.

## Key Results

- Best method by average NLL: `instruct_expert` with avg NLL 3.009, worst NLL 7.541.
- Best method by worst NLL: `instruct_expert` with avg NLL 3.009, worst NLL 7.541.

| method | alpha | beta | general NLL | instruction NLL | code NLL | avg NLL | worst NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| instruct_expert | 1.00 | 0.00 | 7.541 | 1.038 | 0.447 | 3.009 | 7.541 |
| validation_grid_best_worst | 1.00 | 0.00 | 7.541 | 1.038 | 0.447 | 3.009 | 7.541 |
| validation_grid_best_avg | 1.00 | 0.00 | 7.541 | 1.038 | 0.447 | 3.009 | 7.541 |
| coder_expert | 0.00 | 1.00 | 7.587 | 0.912 | 0.574 | 3.024 | 7.587 |
| task_arithmetic_0.75 | 0.75 | 0.75 | 7.962 | 1.234 | 0.426 | 3.207 | 7.962 |
| base | 0.00 | 0.00 | 7.844 | 2.973 | 1.543 | 4.120 | 7.844 |
| task_arithmetic_0.25 | 0.25 | 0.25 | 8.458 | 5.002 | 3.117 | 5.526 | 8.458 |
| linear_average | 0.50 | 0.50 | 9.553 | 4.610 | 2.611 | 5.591 | 9.553 |

## Expert Conflict

- `instruct` vs `coder`: cosine 0.140, sign conflict 0.454, weighted conflict 0.386.

## Files

- `grid_metrics.csv`: alpha/beta multi-expert grid.
- `method_metrics.csv`: named endpoints and merge methods.
- `pairwise_conflict.csv`: instruct/coder delta conflict.
- `figures/*.png`: grid, diagonal path, and conflict plots.

## Configuration

```json
{
  "base": "/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B",
  "instruct": "/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
  "coder": "/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
  "tokenizer": "/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
  "grid_values": "0.0,0.25,0.5,0.75,1.0",
  "dtype": "bfloat16",
  "device": "cuda:3",
  "max_length": 384
}
```
