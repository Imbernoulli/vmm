# Single-Digit Expert Pairwise Merge Study

This run trains ten same-base experts on individual sklearn digit classes. It evaluates all 45 digit pairs to test whether task-vector conflict metrics correlate with merge degradation.

Each expert is a full 10-way classifier fine-tuned only on one digit's examples, so it becomes a class specialist. Pairwise linear averaging tests whether two such specialists can be merged while preserving both single-digit tasks.

## Correlation With Linear Merge Drop

Spearman correlation against `linear_drop_from_base = base_worst_acc - linear_merge_worst_acc`:

- `cosine`: -0.174
- `sign_conflict`: 0.185
- `weighted_conflict`: 0.165
- `max_layer_weighted_conflict`: 0.080

## Best Linear Merges

- digits `0`/`1`: worst acc 1.000, drop 0.000, weighted conflict 0.468.
- digits `0`/`2`: worst acc 1.000, drop 0.000, weighted conflict 0.472.
- digits `0`/`3`: worst acc 1.000, drop 0.000, weighted conflict 0.418.
- digits `0`/`5`: worst acc 1.000, drop -0.054, weighted conflict 0.468.
- digits `0`/`6`: worst acc 1.000, drop 0.000, weighted conflict 0.426.

## Worst Linear Merges

- digits `3`/`9`: worst acc 0.861, drop 0.111, weighted conflict 0.442.
- digits `0`/`4`: worst acc 0.943, drop 0.057, weighted conflict 0.475.
- digits `2`/`8`: worst acc 0.943, drop -0.029, weighted conflict 0.423.
- digits `7`/`8`: worst acc 0.943, drop -0.029, weighted conflict 0.448.
- digits `4`/`6`: worst acc 0.944, drop 0.056, weighted conflict 0.426.

## Most Conflicted Layers On Average

- `net.4.bias`: mean weighted conflict 0.537, mean cosine -0.055.
- `net.4.weight`: mean weighted conflict 0.514, mean cosine -0.014.
- `net.0.weight`: mean weighted conflict 0.501, mean cosine -0.003.
- `net.0.bias`: mean weighted conflict 0.499, mean cosine 0.003.
- `net.3.weight`: mean weighted conflict 0.477, mean cosine 0.033.

## Files

- `pairwise_metrics.csv`: one row per digit pair.
- `layer_pairwise_conflict.csv`: per-layer conflict for each digit pair.
- `pairwise_heatmaps.png`: pair matrices for performance and conflict.
- `conflict_vs_drop.png`: scatter plots for conflict metrics vs merge drop.
- `layer_conflict_atlas.png`: average layer-wise conflict atlas.

## Configuration

```json
{
  "seed": 17,
  "hidden": 128,
  "base_epochs": 8,
  "expert_epochs": 90,
  "base_lr": 0.003,
  "expert_lr": 0.003,
  "weight_decay": 0.0001,
  "ties_density": 0.5,
  "dare_drop_rate": 0.5,
  "device": "cpu",
  "num_pairs": 45,
  "num_parameters": 17610
}
```
