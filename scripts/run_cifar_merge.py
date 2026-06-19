#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from mergeviz.merge_methods import dare_average, linear_average, slerp, task_arithmetic, ties_merge
from mergeviz.weights import VectorSpec, cosine, layer_slices, load_vector_into_model, project_to_plane, vectorize_model


VEHICLES = {0, 1, 8, 9}
ANIMALS = {2, 3, 4, 5, 6, 7}
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class SmallCifarCNN(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.GroupNorm(8, width * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 4, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    best = None
    best_free = -1
    for index in range(torch.cuda.device_count()):
        name = f"cuda:{index}"
        try:
            _ = torch.empty(1, device=name)
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:
            continue
        if int(free) > best_free:
            best_free = int(free)
            best = name
    return best or "cpu"


def indices_by_class(targets: list[int], classes: set[int]) -> dict[int, list[int]]:
    grouped = {klass: [] for klass in classes}
    for idx, target in enumerate(targets):
        if int(target) in classes:
            grouped[int(target)].append(idx)
    return grouped


def sample_indices(targets: list[int], classes: set[int], per_class: int | None, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    grouped = indices_by_class(targets, classes)
    selected: list[int] = []
    for klass in sorted(grouped):
        items = np.array(grouped[klass])
        rng.shuffle(items)
        if per_class is not None:
            items = items[: min(per_class, len(items))]
        selected.extend(items.tolist())
    rng.shuffle(selected)
    return selected


def loader(dataset: datasets.CIFAR10, indices: list[int], batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=pin_memory)


def prepare_data(args: argparse.Namespace) -> dict[str, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_set = datasets.CIFAR10(root=args.data_root, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=False, transform=transform)

    full_classes = set(range(10))
    train_full = sample_indices(train_set.targets, full_classes, args.train_per_class, args.seed)
    train_vehicle = sample_indices(train_set.targets, VEHICLES, args.expert_train_per_class, args.seed + 1)
    train_animal = sample_indices(train_set.targets, ANIMALS, args.expert_train_per_class, args.seed + 2)
    val_vehicle = sample_indices(test_set.targets, VEHICLES, args.eval_per_class, args.seed + 3)
    val_animal = sample_indices(test_set.targets, ANIMALS, args.eval_per_class, args.seed + 4)

    # This machine exposes one CUDA device that is busy/unavailable to PyTorch.
    # DataLoader's pin-memory thread can touch that device even when training on
    # another GPU, so keep pinning off for this small benchmark.
    pin_memory = False
    return {
        "train_full": loader(train_set, train_full, args.batch_size, True, pin_memory),
        "train_vehicle": loader(train_set, train_vehicle, args.batch_size, True, pin_memory),
        "train_animal": loader(train_set, train_animal, args.batch_size, True, pin_memory),
        "eval_vehicle": loader(test_set, val_vehicle, args.batch_size, False, pin_memory),
        "eval_animal": loader(test_set, val_animal, args.batch_size, False, pin_memory),
    }


def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    desc: str,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.detach().cpu())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": total_loss / total, "acc": correct / total, "n": total}


def evaluate_vector(
    model: nn.Module,
    vector: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, float]:
    load_vector_into_model(model, vector, spec, reference_state)
    model.to(device)
    vehicle = evaluate(model, eval_loaders["vehicle"], device)
    animal = evaluate(model, eval_loaders["animal"], device)
    return {
        "vehicle_loss": vehicle["loss"],
        "animal_loss": animal["loss"],
        "vehicle_acc": vehicle["acc"],
        "animal_acc": animal["acc"],
        "avg_loss": 0.5 * (vehicle["loss"] + animal["loss"]),
        "worst_loss": max(vehicle["loss"], animal["loss"]),
        "avg_acc": 0.5 * (vehicle["acc"] + animal["acc"]),
        "worst_acc": min(vehicle["acc"], animal["acc"]),
    }


def build_grid(
    model: nn.Module,
    base: Tensor,
    tau_vehicle: Tensor,
    tau_animal: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> pd.DataFrame:
    rows = []
    for alpha in tqdm(np.linspace(grid_min, grid_max, grid_size), desc="grid alpha"):
        for beta in np.linspace(grid_min, grid_max, grid_size):
            vector = base + float(alpha) * tau_vehicle + float(beta) * tau_animal
            rows.append({"alpha": float(alpha), "beta": float(beta), **evaluate_vector(model, vector, spec, reference_state, eval_loaders, device)})
    return pd.DataFrame(rows)


def lambda_sweep(
    model: nn.Module,
    base: Tensor,
    tau_vehicle: Tensor,
    tau_animal: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
    lambda_max: float,
    steps: int,
) -> pd.DataFrame:
    rows = []
    for lam in np.linspace(0.0, lambda_max, steps):
        vector = task_arithmetic(base, [tau_vehicle, tau_animal], float(lam))
        rows.append({"lambda": float(lam), **evaluate_vector(model, vector, spec, reference_state, eval_loaders, device)})
    return pd.DataFrame(rows)


def conflict_table(spec: VectorSpec, tau_vehicle: Tensor, tau_animal: Tensor) -> pd.DataFrame:
    rows = []
    for layer, sl in layer_slices(spec).items():
        a = tau_vehicle[sl]
        b = tau_animal[sl]
        active = (a.abs() > 1e-10) & (b.abs() > 1e-10)
        if int(active.sum()) == 0:
            sign_conflict = 0.0
            weighted_conflict = 0.0
        else:
            conflict = torch.sign(a[active]) != torch.sign(b[active])
            weights = (a[active].abs() * b[active].abs()).to(torch.float64)
            sign_conflict = float(conflict.to(torch.float32).mean())
            weighted_conflict = float((weights * conflict.to(torch.float64)).sum() / weights.sum().clamp_min(1e-12))
        rows.append(
            {
                "layer": layer,
                "numel": int(a.numel()),
                "cosine": cosine(a, b),
                "sign_conflict": sign_conflict,
                "weighted_conflict": weighted_conflict,
                "vehicle_norm": float(torch.linalg.norm(a)),
                "animal_norm": float(torch.linalg.norm(b)),
            }
        )
    return pd.DataFrame(rows)


def method_table(
    model: nn.Module,
    base: Tensor,
    vehicle_vec: Tensor,
    animal_vec: Tensor,
    tau_vehicle: Tensor,
    tau_animal: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
    lambda_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    best_lambda = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    best_grid = grid_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    methods = [
        ("base", base),
        ("expert_vehicle", vehicle_vec),
        ("expert_animal", animal_vec),
        ("linear_average", linear_average(base, [tau_vehicle, tau_animal])),
        ("slerp_experts", slerp(vehicle_vec, animal_vec, t=0.5)),
        ("task_arithmetic_best_lambda", task_arithmetic(base, [tau_vehicle, tau_animal], float(best_lambda["lambda"]))),
        ("ties_density_0.5", ties_merge(base, [tau_vehicle, tau_animal], density=0.5)),
        ("dare_drop_0.5", dare_average(base, [tau_vehicle, tau_animal], drop_rate=0.5, seed=seed)),
        ("validation_grid_best", base + float(best_grid["alpha"]) * tau_vehicle + float(best_grid["beta"]) * tau_animal),
    ]
    rows = []
    for name, vector in methods:
        alpha, beta, residual = project_to_plane(vector, base, tau_vehicle, tau_animal)
        rows.append({"method": name, "alpha": alpha, "beta": beta, "plane_residual": residual, **evaluate_vector(model, vector, spec, reference_state, eval_loaders, device)})
    return pd.DataFrame(rows)


def grid_to_matrix(df: pd.DataFrame, metric: str, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index="beta", columns="alpha", values=metric).sort_index()
    return pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float), pivot.to_numpy(dtype=float)


def plot_landscape(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path, grid_size: int) -> None:
    metrics = [
        ("vehicle_loss", "Vehicle loss"),
        ("animal_loss", "Animal loss"),
        ("avg_loss", "Average loss"),
        ("worst_loss", "Worst-task loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics, strict=True):
        x, y, z = grid_to_matrix(grid_df, metric, grid_size)
        levels = np.linspace(np.nanmin(z), np.nanpercentile(z, 95), 18)
        contour = ax.contourf(x, y, z, levels=levels, cmap="viridis")
        ax.contour(x, y, z, levels=levels[::3], colors="white", linewidths=0.45, alpha=0.7)
        ax.scatter(methods_df["alpha"], methods_df["beta"], c="white", s=35, edgecolors="black", linewidths=0.7)
        for _, row in methods_df.iterrows():
            if row["method"] in {"base", "expert_vehicle", "expert_animal", "linear_average", "validation_grid_best"}:
                ax.annotate(row["method"].replace("_", "\n"), (row["alpha"], row["beta"]), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("alpha: vehicle task vector")
        ax.set_ylabel("beta: animal task vector")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_methods(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path, grid_size: int) -> None:
    x, y, z = grid_to_matrix(grid_df, "worst_acc", grid_size)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax = axes[0]
    contour = ax.contourf(x, y, z, levels=np.linspace(np.nanmin(z), np.nanmax(z), 18), cmap="cividis")
    for _, row in methods_df.iterrows():
        ax.scatter(row["alpha"], row["beta"], s=70, c="white", edgecolors="black", linewidths=0.9)
        ax.annotate(row["method"].replace("_", "\n"), (row["alpha"], row["beta"]), fontsize=7)
    ax.set_title("CIFAR merge methods in task-vector plane")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="worst-task accuracy")

    ax = axes[1]
    plot_df = methods_df.sort_values("worst_acc")
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.18, plot_df["vehicle_acc"], height=0.34, color="#2a9d8f", alpha=0.78, label="vehicle")
    ax.barh(y_pos + 0.18, plot_df["animal_acc"], height=0.34, color="#e76f51", alpha=0.62, label="animal")
    ax.scatter(plot_df["worst_acc"], y_pos, color="black", label="worst", zorder=4)
    ax.set_yticks(y_pos, labels=plot_df["method"])
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("accuracy")
    ax.set_title("CIFAR method performance")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_lambda(lambda_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(lambda_df["lambda"], lambda_df["vehicle_acc"], label="vehicle acc", color="#2a9d8f")
    axes[0].plot(lambda_df["lambda"], lambda_df["animal_acc"], label="animal acc", color="#e76f51")
    axes[0].plot(lambda_df["lambda"], lambda_df["worst_acc"], label="worst acc", color="#6d597a")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("CIFAR task arithmetic path")
    axes[0].legend(fontsize=8)
    axes[1].plot(lambda_df["lambda"], lambda_df["vehicle_loss"], label="vehicle loss", color="#2a9d8f")
    axes[1].plot(lambda_df["lambda"], lambda_df["animal_loss"], label="animal loss", color="#e76f51")
    axes[1].plot(lambda_df["lambda"], lambda_df["worst_loss"], label="worst loss", color="#6d597a")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("cross entropy")
    axes[1].set_title("CIFAR loss along combined direction")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_interference(conflict_df: pd.DataFrame, out: Path) -> None:
    labels = conflict_df["layer"].tolist()
    metrics = ["cosine", "sign_conflict", "weighted_conflict"]
    matrix = conflict_df[metrics].T.to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.62 * len(labels)), 7), constrained_layout=True)
    im = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_yticks(range(len(metrics)), labels=metrics)
    axes[0].set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("CIFAR vehicle/animal task-vector conflict")
    fig.colorbar(im, ax=axes[0], shrink=0.82)
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, conflict_df["vehicle_norm"], width=0.4, label="vehicle delta norm", color="#2a9d8f")
    axes[1].bar(x + 0.2, conflict_df["animal_norm"], width=0.4, label="animal delta norm", color="#e76f51")
    axes[1].set_xticks(x, labels=labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("L2 norm")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, config: dict[str, object], methods_df: pd.DataFrame, lambda_df: pd.DataFrame, conflict_df: pd.DataFrame, grid_df: pd.DataFrame) -> None:
    best_method = methods_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    base = methods_df[methods_df["method"] == "base"].iloc[0]
    linear = methods_df[methods_df["method"] == "linear_average"].iloc[0]
    best_lambda = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    overlap = float((grid_df["worst_acc"] >= 0.4).mean())
    top_conflict = conflict_df.sort_values("weighted_conflict", ascending=False).head(5)
    lines = [
        "# CIFAR-10 Vehicle/Animal Merge Study",
        "",
        "This run moves beyond sklearn digits to a natural-image CIFAR-10 class-group task. A small GroupNorm CNN base is trained on a balanced CIFAR-10 subset, then two same-base experts are fine-tuned on vehicle classes and animal classes.",
        "",
        "Vehicle classes: airplane, automobile, ship, truck. Animal classes: bird, cat, deer, dog, frog, horse.",
        "",
        "## Key Results",
        "",
        f"- Best method by worst-task accuracy: `{best_method['method']}` with vehicle accuracy {best_method['vehicle_acc']:.3f}, animal accuracy {best_method['animal_acc']:.3f}, worst accuracy {best_method['worst_acc']:.3f}.",
        f"- Base worst-task accuracy: {base['worst_acc']:.3f}; linear average worst-task accuracy: {linear['worst_acc']:.3f}.",
        f"- Best task-arithmetic lambda: {best_lambda['lambda']:.3f}, worst accuracy {best_lambda['worst_acc']:.3f}.",
        f"- Fraction of sampled plane with worst-task accuracy >= 0.40: {overlap:.3f}.",
        "",
        "## Most Conflicted Tensors",
        "",
    ]
    for _, row in top_conflict.iterrows():
        lines.append(
            f"- `{row['layer']}`: cosine {row['cosine']:.3f}, sign conflict {row['sign_conflict']:.3f}, weighted conflict {row['weighted_conflict']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `grid_metrics.csv`: 2D task-vector plane metrics.",
            "- `method_metrics.csv`: merge method metrics and projected coordinates.",
            "- `lambda_sweep.csv`: task arithmetic path.",
            "- `interference.csv`: tensor-wise conflict metrics.",
            "- `figures/*.png`: landscape, method, lambda, and conflict plots.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(config, indent=2),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CIFAR-10 vehicle/animal model-merging landscape.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar_merge"))
    parser.add_argument("--data-root", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/cifar10")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--base-epochs", type=int, default=5)
    parser.add_argument("--expert-epochs", type=int, default=6)
    parser.add_argument("--train-per-class", type=int, default=700)
    parser.add_argument("--expert-train-per-class", type=int, default=700)
    parser.add_argument("--eval-per-class", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--expert-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--grid-min", type=float, default=-0.25)
    parser.add_argument("--grid-max", type=float, default=1.25)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-steps", type=int, default=41)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = prepare_data(args)

    base_model = SmallCifarCNN(args.width).to(device)
    train_model(base_model, data["train_full"], args.base_epochs, args.base_lr, args.weight_decay, device, "train CIFAR base")
    reference_state = deepcopy(base_model.state_dict())
    base_vec, spec = vectorize_model(base_model)

    vehicle_model = SmallCifarCNN(args.width).to(device)
    vehicle_model.load_state_dict(reference_state)
    train_model(vehicle_model, data["train_vehicle"], args.expert_epochs, args.expert_lr, args.weight_decay, device, "fine-tune vehicles")
    vehicle_vec = vectorize_model(vehicle_model)[0]

    animal_model = SmallCifarCNN(args.width).to(device)
    animal_model.load_state_dict(reference_state)
    train_model(animal_model, data["train_animal"], args.expert_epochs, args.expert_lr, args.weight_decay, device, "fine-tune animals")
    animal_vec = vectorize_model(animal_model)[0]

    tau_vehicle = vehicle_vec - base_vec
    tau_animal = animal_vec - base_vec
    eval_model = SmallCifarCNN(args.width).to(device)
    eval_loaders = {"vehicle": data["eval_vehicle"], "animal": data["eval_animal"]}

    grid_df = build_grid(eval_model, base_vec, tau_vehicle, tau_animal, spec, reference_state, eval_loaders, device, args.grid_min, args.grid_max, args.grid_size)
    lambda_df = lambda_sweep(eval_model, base_vec, tau_vehicle, tau_animal, spec, reference_state, eval_loaders, device, args.lambda_max, args.lambda_steps)
    methods_df = method_table(eval_model, base_vec, vehicle_vec, animal_vec, tau_vehicle, tau_animal, spec, reference_state, eval_loaders, device, lambda_df, grid_df, args.seed)
    conflict_df = conflict_table(spec, tau_vehicle, tau_animal)

    grid_df.to_csv(out_dir / "grid_metrics.csv", index=False)
    lambda_df.to_csv(out_dir / "lambda_sweep.csv", index=False)
    methods_df.to_csv(out_dir / "method_metrics.csv", index=False)
    conflict_df.to_csv(out_dir / "interference.csv", index=False)

    plot_landscape(grid_df, methods_df, fig_dir / "merge_landscape.png", args.grid_size)
    plot_methods(grid_df, methods_df, fig_dir / "method_overlay.png", args.grid_size)
    plot_lambda(lambda_df, fig_dir / "lambda_sweep.png")
    plot_interference(conflict_df, fig_dir / "interference_heatmap.png")

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "device": str(device),
            "num_parameters": int(base_vec.numel()),
            "global_task_vector_cosine": cosine(tau_vehicle, tau_animal),
            "tau_vehicle_norm": float(torch.linalg.norm(tau_vehicle)),
            "tau_animal_norm": float(torch.linalg.norm(tau_animal)),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(out_dir, config, methods_df, lambda_df, conflict_df, grid_df)
    print(f"Wrote CIFAR merge artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
