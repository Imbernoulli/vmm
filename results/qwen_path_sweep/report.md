# Qwen Base-to-Instruct Path Sweep

This run evaluates a real LLM weight-space path using local Qwen2.5-1.5B base and Qwen2.5-1.5B-Instruct checkpoints.

The path is `theta(lambda) = theta_base + lambda * (theta_instruct - theta_base)`. Metrics are token-level negative log-likelihoods on a small fixed prompt slice, not full benchmark scores.

## Key Results

- Base lambda 0.0: general NLL 4.783, instruction NLL 3.612, worst NLL 4.783.
- Instruct lambda 1.0: general NLL 4.746, instruction NLL 1.874, worst NLL 4.746.
- Best average NLL lambda: 0.750, avg NLL 3.283.
- Best worst-task NLL lambda: 1.250, worst NLL 4.741.

## Largest Delta Groups

- `model.embed_tokens`: delta norm 10.84, mean relative norm 0.0282.
- `model.layers.25`: delta norm 5.93, mean relative norm 0.0082.
- `model.layers.26`: delta norm 5.82, mean relative norm 0.0074.
- `model.layers.21`: delta norm 5.78, mean relative norm 0.0078.
- `model.layers.20`: delta norm 5.78, mean relative norm 0.0077.

## Interpretation

This is the LLM analogue of the lambda-sweep plot in the image experiment. Instead of a dense 2D plane, it samples a one-dimensional task-vector path from a base model to an instruction-tuned model. A lambda with lower average or worst NLL than both endpoints would indicate a useful intermediate merge point; a monotonic tradeoff indicates that the instruction delta mainly moves the model from one behavior regime toward another.

Because this is a tiny fixed prompt slice, it should be treated as a diagnostic, not as an MMLU/GSM8K/HumanEval claim.

## Files

- `path_metrics.csv`: NLL/PPL metrics for every lambda.
- `delta_summary.csv`: per-tensor base-to-instruct delta magnitudes.
- `qwen_path_sweep.png`: path and tradeoff plot.
- `delta_norms.png`: largest parameter-change groups.

## Configuration

```json
{
  "base": "/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-1.5B",
  "expert": "/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct",
  "tokenizer": "/srv/home/bohanlyu/MLS-Bench/vendor/data/qwen2.5-1.5b-instruct",
  "lambdas": "-0.25,0.0,0.25,0.5,0.75,1.0,1.25",
  "device": "cuda:2",
  "dtype": "bfloat16",
  "max_length": 384
}
```
