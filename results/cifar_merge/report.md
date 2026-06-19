# CIFAR-10 Vehicle/Animal Merge Study

This run moves beyond sklearn digits to a natural-image CIFAR-10 class-group task. A small GroupNorm CNN base is trained on a balanced CIFAR-10 subset, then two same-base experts are fine-tuned on vehicle classes and animal classes.

Vehicle classes: airplane, automobile, ship, truck. Animal classes: bird, cat, deer, dog, frog, horse.

## Key Results

- Best method by worst-task accuracy: `validation_grid_best` with vehicle accuracy 0.433, animal accuracy 0.426, worst accuracy 0.426.
- Base worst-task accuracy: 0.376; linear average worst-task accuracy: 0.249.
- Best task-arithmetic lambda: 0.112, worst accuracy 0.381.
- Fraction of sampled plane with worst-task accuracy >= 0.40: 0.016.

## Most Conflicted Tensors

- `classifier.bias`: cosine -0.406, sign conflict 0.900, weighted conflict 0.987.
- `features.15.bias`: cosine -0.528, sign conflict 0.693, weighted conflict 0.876.
- `features.14.bias`: cosine -0.301, sign conflict 0.568, weighted conflict 0.719.
- `classifier.weight`: cosine -0.044, sign conflict 0.557, weighted conflict 0.552.
- `features.14.weight`: cosine -0.026, sign conflict 0.524, weighted conflict 0.523.

## Files

- `grid_metrics.csv`: 2D task-vector plane metrics.
- `method_metrics.csv`: merge method metrics and projected coordinates.
- `lambda_sweep.csv`: task arithmetic path.
- `interference.csv`: tensor-wise conflict metrics.
- `figures/*.png`: landscape, method, lambda, and conflict plots.

## Configuration

```json
{
  "output_dir": "results/cifar_merge",
  "data_root": "/srv/home/bohanlyu/MLS-Bench/vendor/data/cifar10",
  "seed": 23,
  "width": 48,
  "base_epochs": 5,
  "expert_epochs": 6,
  "train_per_class": 700,
  "expert_train_per_class": 700,
  "eval_per_class": 180,
  "batch_size": 256,
  "base_lr": 0.002,
  "expert_lr": 0.001,
  "weight_decay": 0.0001,
  "grid_size": 21,
  "grid_min": -0.25,
  "grid_max": 1.25,
  "lambda_max": 1.5,
  "lambda_steps": 41,
  "device": "cuda:2",
  "num_parameters": 315706,
  "global_task_vector_cosine": 0.0030621524575348735,
  "tau_vehicle_norm": 4.748868942260742,
  "tau_animal_norm": 5.998348236083984
}
```
