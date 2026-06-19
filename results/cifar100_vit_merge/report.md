# CIFAR100 ViT-Style Merge Study

This run addresses the proposal's CLIP/ViT phase with a lightweight ViT-style patch transformer on CIFAR100 coarse labels. It is not a CLIP transfer run, but it uses a transformer vision architecture and the same task-vector visualization machinery.

Living superclasses: aquatic_mammals, fish, flowers, insects, large_carnivores, large_omnivores_and_herbivores, medium_mammals, non-insect_invertebrates, people, reptiles, small_mammals, trees.
Object/vehicle superclasses: food_containers, fruit_and_vegetables, household_electrical_devices, household_furniture, large_man-made_outdoor_things, large_natural_outdoor_scenes, vehicles_1, vehicles_2.

## Key Results

- Best method by worst-task accuracy: `task_arithmetic_best_lambda` with living accuracy 0.199, object accuracy 0.197, worst accuracy 0.197.
- Base worst-task accuracy: 0.189; linear average worst-task accuracy: 0.076.
- Best task-arithmetic lambda: 0.050, worst accuracy 0.197.
- Fraction of sampled plane with worst-task accuracy >= 0.15: 0.038.

## Interpretation

This experiment gives the project a ViT-style vision checkpoint family. Accuracy is intentionally modest because the model is small and trained from scratch on a limited CIFAR100 subset. The point is to test whether the merge-plane, method overlay, lambda sweep, PCA geometry, and layer-conflict atlas remain usable when the architecture is transformer-based rather than an MLP or CNN.

## Files

- `grid_metrics.csv`, `method_metrics.csv`, `lambda_sweep.csv`, `interference.csv`.
- `pca_geometry.csv`: PCA coordinates for task vectors and method projections.
- `figures/*.png`: landscape, method, lambda, conflict, and PCA figures.

## Configuration

```json
{
  "output_dir": "results/cifar100_vit_merge",
  "data_root": "/srv/home/bohanlyu/MLS-Bench/vendor/data/cifar100",
  "seed": 41,
  "patch_size": 4,
  "dim": 96,
  "depth": 2,
  "heads": 4,
  "mlp_ratio": 2.0,
  "base_epochs": 6,
  "expert_epochs": 6,
  "train_per_class": 180,
  "expert_train_per_class": 180,
  "eval_per_class": 80,
  "batch_size": 256,
  "base_lr": 0.002,
  "expert_lr": 0.001,
  "weight_decay": 0.0005,
  "grid_size": 17,
  "grid_min": -0.25,
  "grid_max": 1.25,
  "lambda_max": 1.5,
  "lambda_steps": 31,
  "device": "cuda:2",
  "num_parameters": 162740,
  "global_task_vector_cosine": -0.17608558484247044,
  "tau_living_norm": 3.7066404819488525,
  "tau_object_norm": 3.4112226963043213,
  "living_superclasses": [
    "aquatic_mammals",
    "fish",
    "flowers",
    "insects",
    "large_carnivores",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees"
  ],
  "object_superclasses": [
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "vehicles_1",
    "vehicles_2"
  ]
}
```
