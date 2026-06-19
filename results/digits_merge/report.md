# Digits Merge-Landscape Study

This run is a controlled image-classification surrogate for the proposal. A shared MLP base is trained on all sklearn digits, then two experts are fine-tuned from that exact base: task A uses digits 0-4 and task B uses digits 5-9. The experiment evaluates the raw task-vector plane `theta = theta0 + alpha * tau_A + beta * tau_B`.

## Key Findings

- Best observed method/checkpoint by worst-task accuracy: `regmean_linear` with task A accuracy 0.961, task B accuracy 0.939, worst-task accuracy 0.939.
- Base checkpoint worst-task accuracy: 0.917; naive linear average worst-task accuracy: 0.922.
- Best task-arithmetic lambda on the sweep: 0.400, worst-task accuracy 0.961, average loss 0.149.
- Fraction of sampled plane with worst-task accuracy >= 0.90: 0.134. This is a direct proxy for basin-overlap area in this plane.
- Worst-task loss barrier along the combined task-vector path: 0.000.

## Figures

- `figures/merge_landscape.png`: task A, task B, average, and worst-task loss surfaces with method points.
- `figures/per_task_basin_overlay.png`: task-specific contour overlays on the joint worst-task objective.
- `figures/lambda_sweep.png`: accuracy and loss along `theta0 + lambda * (tau_A + tau_B)`.
- `figures/method_overlay.png`: method points projected into the task-vector plane and their accuracies.
- `figures/interference_heatmap.png`: per-layer cosine alignment, sign conflict, weighted conflict, and delta norms.

## Most Conflicted Layers

- `net.6.bias`: cosine -0.801, sign conflict 1.000, weighted conflict 1.000.
- `net.3.weight`: cosine -0.044, sign conflict 0.538, weighted conflict 0.542.
- `net.0.weight`: cosine -0.004, sign conflict 0.481, weighted conflict 0.503.

## Interpretation

The meaningful object is the task-vector plane, not a random loss-landscape slice. Experts occupy the two coordinate axes by construction, while merge methods either stay in that plane or are projected back into it with a reported residual. When high-worst-accuracy regions appear between the two experts, merging is geometrically plausible; when the midpoint or task-arithmetic path crosses a high-loss ridge, the same plot explains why a merge fails.

The interference atlas gives a parameter-space explanation for those ridges. Low cosine alignment and high magnitude-weighted sign conflict identify tensors where the two fine-tuning runs ask the same coordinates to move in opposite directions. TIES and DARE are included because they explicitly modify those coordinates before averaging.

## Configuration

```json
{
  "output_dir": "results/digits_merge",
  "seed": 13,
  "hidden": 128,
  "base_epochs": 4,
  "expert_epochs": 120,
  "batch_size": 128,
  "base_lr": 0.003,
  "expert_lr": 0.003,
  "weight_decay": 0.0001,
  "grid_size": 41,
  "grid_min": -0.25,
  "grid_max": 1.25,
  "lambda_max": 1.5,
  "lambda_steps": 61,
  "fisher_batches": 8,
  "regmean_batches": 8,
  "regmean_ridge": 0.0001,
  "layerwise_scale_grid": "0,0.2,0.4,0.6,0.8,1.0",
  "device": "cpu",
  "num_parameters": 17610,
  "global_task_vector_cosine": 0.13830392276500614,
  "tau_a_norm": 4.283435821533203,
  "tau_b_norm": 4.661861419677734
}
```
