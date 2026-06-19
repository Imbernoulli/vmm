# Pretrained ViT Transfer Merge

This run uses an ImageNet-pretrained torchvision ViT-B/16 as a frozen feature extractor, then trains CIFAR100 coarse-label linear heads for a base model and two class-group experts. It is a pretrained ViT transfer experiment, but not full-backbone fine-tuning.

Tasks are living superclasses versus object/vehicle superclasses. The merge plane is formed from the linear-head task vectors.

## Key Results

- Best method by worst-task accuracy: `validation_grid_best` with worst accuracy 0.783.
- Base worst accuracy: 0.740.
- Linear average worst accuracy: 0.763.
- Global head task-vector cosine: -0.068.

| method | alpha | beta | living acc | object acc | worst acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation_grid_best | 0.350 | 0.125 | 0.783 | 0.800 | 0.783 |
| task_arithmetic_best_lambda | 0.350 | 0.350 | 0.772 | 0.848 | 0.772 |
| dare_average | 0.498 | 0.499 | 0.770 | 0.858 | 0.770 |
| linear_average | 0.500 | 0.500 | 0.763 | 0.853 | 0.763 |
| slerp_experts | 0.536 | 0.532 | 0.763 | 0.853 | 0.763 |
| ties | 0.800 | 0.667 | 0.755 | 0.825 | 0.755 |
| base | 0.000 | 0.000 | 0.740 | 0.830 | 0.740 |
| ties_dare | 0.881 | 0.802 | 0.733 | 0.840 | 0.733 |
| expert_object | 0.000 | 1.000 | 0.402 | 0.877 | 0.402 |
| expert_living | 1.000 | 0.000 | 0.783 | 0.400 | 0.400 |

## Files

- `grid_metrics.csv`: alpha/beta grid over the frozen-backbone head merge plane.
- `method_metrics.csv`: endpoints and merge methods.
- `lambda_sweep.csv`: task-arithmetic path metrics.
- `interference.csv`: head weight/bias conflict metrics.
- `figures/*.png`: landscape, methods, lambda path, and conflict plots.

## Configuration

```json
{
  "data_root": "/srv/home/bohanlyu/MLS-Bench/vendor/data/cifar100",
  "output_dir": "results/pretrained_vit_transfer_merge",
  "train_per_class": 40,
  "expert_train_per_class": 40,
  "eval_per_class": 50,
  "image_batch_size": 48,
  "feature_batch_size": 256,
  "base_epochs": 80,
  "expert_epochs": 60,
  "lr": 0.002,
  "weight_decay": 0.0001,
  "grid_size": 21,
  "grid_min": -0.25,
  "grid_max": 1.25,
  "lambda_max": 1.5,
  "lambda_steps": 31,
  "seed": 53,
  "device": "cuda:3"
}
```
