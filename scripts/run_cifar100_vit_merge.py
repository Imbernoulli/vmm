#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
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
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from mergeviz.merge_methods import dare_average, linear_average, slerp, task_arithmetic, ties_merge
from mergeviz.weights import VectorSpec, cosine, layer_slices, load_vector_into_model, project_to_plane, vectorize_model


LIVING_SUPERCLASSES = {0, 1, 2, 7, 8, 11, 12, 13, 14, 15, 16, 17}
OBJECT_SUPERCLASSES = {3, 4, 5, 6, 9, 10, 18, 19}


class Cifar100Coarse(Dataset):
    def __init__(self, root: str | Path, train: bool, transform: transforms.Compose | None = None) -> None:
        self.root = Path(root) / "cifar-100-python"
        split = "train" if train else "test"
        with (self.root / split).open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        with (self.root / "meta").open("rb") as handle:
            meta = pickle.load(handle, encoding="latin1")
        data = np.asarray(payload["data"], dtype=np.uint8)
        self.images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = [int(item) for item in payload["coarse_labels"]]
        self.classes = list(meta["coarse_label_names"])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


class PatchViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 96,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


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


def sample_indices(targets: list[int], classes: set[int], per_class: int | None, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by_class = {klass: [] for klass in classes}
    for index, target in enumerate(targets):
        if int(target) in by_class:
            by_class[int(target)].append(index)
    selected: list[int] = []
    for klass in sorted(by_class):
        values = np.asarray(by_class[klass])
        rng.shuffle(values)
        if per_class is not None:
            values = values[: min(per_class, len(values))]
        selected.extend(values.tolist())
    rng.shuffle(selected)
    return selected


def make_loader(dataset: Dataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=False)


def prepare_data(args: argparse.Namespace) -> tuple[dict[str, DataLoader], list[str]]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = Cifar100Coarse(args.data_root, train=True, transform=train_transform)
    eval_train_set = Cifar100Coarse(args.data_root, train=True, transform=eval_transform)
    test_set = Cifar100Coarse(args.data_root, train=False, transform=eval_transform)
    all_classes = set(range(20))
    base_indices = sample_indices(train_set.targets, all_classes, args.train_per_class, args.seed)
    living_indices = sample_indices(train_set.targets, LIVING_SUPERCLASSES, args.expert_train_per_class, args.seed + 1)
    object_indices = sample_indices(train_set.targets, OBJECT_SUPERCLASSES, args.expert_train_per_class, args.seed + 2)
    eval_living = sample_indices(test_set.targets, LIVING_SUPERCLASSES, args.eval_per_class, args.seed + 3)
    eval_object = sample_indices(test_set.targets, OBJECT_SUPERCLASSES, args.eval_per_class, args.seed + 4)
    pca_probe = sample_indices(eval_train_set.targets, all_classes, min(args.eval_per_class, 80), args.seed + 5)
    return (
        {
            "train_full": make_loader(train_set, base_indices, args.batch_size, True),
            "train_living": make_loader(train_set, living_indices, args.batch_size, True),
            "train_object": make_loader(train_set, object_indices, args.batch_size, True),
            "eval_living": make_loader(test_set, eval_living, args.batch_size, False),
            "eval_object": make_loader(test_set, eval_object, args.batch_size, False),
            "pca_probe": make_loader(eval_train_set, pca_probe, args.batch_size, False),
        },
        test_set.classes,
    )


def make_model(args: argparse.Namespace) -> PatchViT:
    return PatchViT(patch_size=args.patch_size, dim=args.dim, depth=args.depth, heads=args.heads, mlp_ratio=args.mlp_ratio)


def train_model(model: nn.Module, data_loader: DataLoader, epochs: int, lr: float, weight_decay: float, device: torch.device, desc: str) -> None:
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
    total_correct = 0
    total = 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": total_loss / total, "acc": total_correct / total, "n": total}


def evaluate_vector(
    model: nn.Module,
    vector: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, float]:
    load_vector_into_model(model, vector, spec, reference_state)
    model.to(device)
    living = evaluate(model, loaders["living"], device)
    objects = evaluate(model, loaders["object"], device)
    return {
        "living_loss": living["loss"],
        "object_loss": objects["loss"],
        "living_acc": living["acc"],
        "object_acc": objects["acc"],
        "avg_loss": 0.5 * (living["loss"] + objects["loss"]),
        "worst_loss": max(living["loss"], objects["loss"]),
        "avg_acc": 0.5 * (living["acc"] + objects["acc"]),
        "worst_acc": min(living["acc"], objects["acc"]),
    }


def build_grid(
    model: nn.Module,
    base: Tensor,
    tau_living: Tensor,
    tau_object: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> pd.DataFrame:
    rows = []
    for alpha in tqdm(np.linspace(grid_min, grid_max, grid_size), desc="ViT grid alpha"):
        for beta in np.linspace(grid_min, grid_max, grid_size):
            vector = base + float(alpha) * tau_living + float(beta) * tau_object
            rows.append({"alpha": float(alpha), "beta": float(beta), **evaluate_vector(model, vector, spec, reference_state, loaders, device)})
    return pd.DataFrame(rows)


def lambda_sweep(
    model: nn.Module,
    base: Tensor,
    tau_living: Tensor,
    tau_object: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    lambda_max: float,
    steps: int,
) -> pd.DataFrame:
    rows = []
    for lam in np.linspace(0.0, lambda_max, steps):
        vector = task_arithmetic(base, [tau_living, tau_object], float(lam))
        rows.append({"lambda": float(lam), **evaluate_vector(model, vector, spec, reference_state, loaders, device)})
    return pd.DataFrame(rows)


def conflict_table(spec: VectorSpec, tau_living: Tensor, tau_object: Tensor) -> pd.DataFrame:
    rows = []
    for layer, sl in layer_slices(spec).items():
        living = tau_living[sl]
        obj = tau_object[sl]
        active = (living.abs() > 1e-10) & (obj.abs() > 1e-10)
        if int(active.sum()) == 0:
            sign_conflict = 0.0
            weighted_conflict = 0.0
        else:
            conflict = torch.sign(living[active]) != torch.sign(obj[active])
            weights = (living[active].abs() * obj[active].abs()).to(torch.float64)
            sign_conflict = float(conflict.to(torch.float32).mean())
            weighted_conflict = float((weights * conflict.to(torch.float64)).sum() / weights.sum().clamp_min(1e-12))
        rows.append(
            {
                "layer": layer,
                "numel": int(living.numel()),
                "cosine": cosine(living, obj),
                "sign_conflict": sign_conflict,
                "weighted_conflict": weighted_conflict,
                "living_norm": float(torch.linalg.norm(living)),
                "object_norm": float(torch.linalg.norm(obj)),
            }
        )
    return pd.DataFrame(rows)


def method_table(
    model: nn.Module,
    base: Tensor,
    living_vec: Tensor,
    object_vec: Tensor,
    tau_living: Tensor,
    tau_object: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    lambda_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    best_lambda = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    best_grid = grid_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    methods = [
        ("base", base, "reference"),
        ("expert_living", living_vec, "reference"),
        ("expert_object", object_vec, "reference"),
        ("linear_average", linear_average(base, [tau_living, tau_object]), "merge"),
        ("slerp_experts", slerp(living_vec, object_vec, t=0.5), "merge"),
        ("task_arithmetic_best_lambda", task_arithmetic(base, [tau_living, tau_object], float(best_lambda["lambda"])), "merge"),
        ("ties_density_0.5", ties_merge(base, [tau_living, tau_object], density=0.5), "merge"),
        ("dare_drop_0.5", dare_average(base, [tau_living, tau_object], drop_rate=0.5, seed=seed), "merge"),
        ("validation_grid_best", base + float(best_grid["alpha"]) * tau_living + float(best_grid["beta"]) * tau_object, "oracle"),
    ]
    rows = []
    for name, vector, kind in methods:
        alpha, beta, residual = project_to_plane(vector, base, tau_living, tau_object)
        rows.append(
            {
                "method": name,
                "kind": kind,
                "alpha": alpha,
                "beta": beta,
                "plane_residual": residual,
                **evaluate_vector(model, vector, spec, reference_state, loaders, device),
            }
        )
    return pd.DataFrame(rows)


def grid_to_matrix(df: pd.DataFrame, metric: str, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index="beta", columns="alpha", values=metric).sort_index()
    matrix = pivot.to_numpy(dtype=float)
    if matrix.shape != (grid_size, grid_size):
        raise ValueError(f"Unexpected grid shape {matrix.shape}")
    return pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float), matrix


def plot_landscape(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path, grid_size: int) -> None:
    metrics = [
        ("living_loss", "Living superclass loss"),
        ("object_loss", "Object/vehicle superclass loss"),
        ("avg_loss", "Average loss"),
        ("worst_loss", "Worst-task loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics, strict=True):
        x, y, z = grid_to_matrix(grid_df, metric, grid_size)
        levels = np.linspace(np.nanmin(z), np.nanpercentile(z, 95), 18)
        contour = ax.contourf(x, y, z, levels=levels, cmap="viridis")
        ax.contour(x, y, z, levels=levels[::3], colors="white", linewidths=0.45, alpha=0.7)
        ax.scatter(methods_df["alpha"], methods_df["beta"], c="white", s=34, edgecolors="black", linewidths=0.7)
        for _, row in methods_df.iterrows():
            if row["method"] in {"base", "expert_living", "expert_object", "linear_average", "validation_grid_best"}:
                ax.annotate(row["method"].replace("_", "\n"), (row["alpha"], row["beta"]), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("alpha: living task vector")
        ax.set_ylabel("beta: object task vector")
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
    ax.set_title("ViT-style CIFAR100 merge methods")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="worst-task accuracy")

    plot_df = methods_df.sort_values("worst_acc")
    y_pos = np.arange(len(plot_df))
    axes[1].barh(y_pos - 0.18, plot_df["living_acc"], height=0.34, color="#2a9d8f", alpha=0.78, label="living")
    axes[1].barh(y_pos + 0.18, plot_df["object_acc"], height=0.34, color="#e76f51", alpha=0.62, label="object")
    axes[1].scatter(plot_df["worst_acc"], y_pos, color="black", label="worst", zorder=4)
    axes[1].set_yticks(y_pos, labels=plot_df["method"])
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_xlabel("accuracy")
    axes[1].set_title("Method performance")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_lambda(lambda_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot(lambda_df["lambda"], lambda_df["living_acc"], label="living acc", color="#2a9d8f")
    axes[0].plot(lambda_df["lambda"], lambda_df["object_acc"], label="object acc", color="#e76f51")
    axes[0].plot(lambda_df["lambda"], lambda_df["worst_acc"], label="worst acc", color="#6d597a")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("ViT-style task arithmetic path")
    axes[0].legend(fontsize=8)
    axes[1].plot(lambda_df["lambda"], lambda_df["living_loss"], label="living loss", color="#2a9d8f")
    axes[1].plot(lambda_df["lambda"], lambda_df["object_loss"], label="object loss", color="#e76f51")
    axes[1].plot(lambda_df["lambda"], lambda_df["worst_loss"], label="worst loss", color="#6d597a")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("cross entropy")
    axes[1].set_title("Loss along combined direction")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_interference(conflict_df: pd.DataFrame, out: Path) -> None:
    labels = conflict_df["layer"].tolist()
    metrics = ["cosine", "sign_conflict", "weighted_conflict"]
    matrix = conflict_df[metrics].T.to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.48 * len(labels)), 7.2), constrained_layout=True)
    im = axes[0].imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_yticks(range(len(metrics)), labels=metrics)
    axes[0].set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("ViT-style layer-wise task-vector conflict")
    fig.colorbar(im, ax=axes[0], shrink=0.82)
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, conflict_df["living_norm"], width=0.4, label="living delta norm", color="#2a9d8f")
    axes[1].bar(x + 0.2, conflict_df["object_norm"], width=0.4, label="object delta norm", color="#e76f51")
    axes[1].set_xticks(x, labels=labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("L2 norm")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def pca_geometry(base: Tensor, tau_living: Tensor, tau_object: Tensor, methods_df: pd.DataFrame) -> pd.DataFrame:
    directions = torch.stack([tau_living, tau_object], dim=0).to(torch.float64)
    centered = directions - directions.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    rows = [
        {"point": "base", "kind": "reference", "pc1": 0.0, "pc2": 0.0},
        {"point": "task_vector_living", "kind": "direction", "pc1": float(tau_living.to(torch.float64) @ components[0]), "pc2": float(tau_living.to(torch.float64) @ components[1])},
        {"point": "task_vector_object", "kind": "direction", "pc1": float(tau_object.to(torch.float64) @ components[0]), "pc2": float(tau_object.to(torch.float64) @ components[1])},
    ]
    for _, row in methods_df.iterrows():
        delta = (float(row["alpha"]) * tau_living + float(row["beta"]) * tau_object).to(torch.float64)
        rows.append({"point": row["method"], "kind": row["kind"], "pc1": float(delta @ components[0]), "pc2": float(delta @ components[1])})
    return pd.DataFrame(rows)


def plot_pca(pca_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    colors = {"reference": "#111827", "direction": "#457b9d", "merge": "#e76f51", "oracle": "#2a9d8f"}
    for _, row in pca_df.iterrows():
        ax.scatter(row["pc1"], row["pc2"], s=70, color=colors.get(row["kind"], "#6d597a"), edgecolors="white", linewidths=0.8)
        ax.annotate(str(row["point"]).replace("_", "\n"), (row["pc1"], row["pc2"]), fontsize=8)
    ax.axhline(0, color="#d7dde4", linewidth=1)
    ax.axvline(0, color="#d7dde4", linewidth=1)
    ax.set_xlabel("PC1 of task-vector matrix")
    ax.set_ylabel("PC2 of task-vector matrix")
    ax.set_title("CIFAR100 ViT-style task-vector PCA geometry")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, config: dict[str, object], methods_df: pd.DataFrame, lambda_df: pd.DataFrame, conflict_df: pd.DataFrame, grid_df: pd.DataFrame, class_names: list[str]) -> None:
    best_method = methods_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    base = methods_df[methods_df["method"] == "base"].iloc[0]
    linear = methods_df[methods_df["method"] == "linear_average"].iloc[0]
    best_lambda = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    overlap = float((grid_df["worst_acc"] >= 0.15).mean())
    living_names = [class_names[idx] for idx in sorted(LIVING_SUPERCLASSES)]
    object_names = [class_names[idx] for idx in sorted(OBJECT_SUPERCLASSES)]
    lines = [
        "# CIFAR100 ViT-Style Merge Study",
        "",
        "This run addresses the proposal's CLIP/ViT phase with a lightweight ViT-style patch transformer on CIFAR100 coarse labels. It is not a CLIP transfer run, but it uses a transformer vision architecture and the same task-vector visualization machinery.",
        "",
        f"Living superclasses: {', '.join(living_names)}.",
        f"Object/vehicle superclasses: {', '.join(object_names)}.",
        "",
        "## Key Results",
        "",
        f"- Best method by worst-task accuracy: `{best_method['method']}` with living accuracy {best_method['living_acc']:.3f}, object accuracy {best_method['object_acc']:.3f}, worst accuracy {best_method['worst_acc']:.3f}.",
        f"- Base worst-task accuracy: {base['worst_acc']:.3f}; linear average worst-task accuracy: {linear['worst_acc']:.3f}.",
        f"- Best task-arithmetic lambda: {best_lambda['lambda']:.3f}, worst accuracy {best_lambda['worst_acc']:.3f}.",
        f"- Fraction of sampled plane with worst-task accuracy >= 0.15: {overlap:.3f}.",
        "",
        "## Interpretation",
        "",
        "This experiment gives the project a ViT-style vision checkpoint family. Accuracy is intentionally modest because the model is small and trained from scratch on a limited CIFAR100 subset. The point is to test whether the merge-plane, method overlay, lambda sweep, PCA geometry, and layer-conflict atlas remain usable when the architecture is transformer-based rather than an MLP or CNN.",
        "",
        "## Files",
        "",
        "- `grid_metrics.csv`, `method_metrics.csv`, `lambda_sweep.csv`, `interference.csv`.",
        "- `pca_geometry.csv`: PCA coordinates for task vectors and method projections.",
        "- `figures/*.png`: landscape, method, lambda, conflict, and PCA figures.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CIFAR100 coarse-label ViT-style model-merging landscape.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar100_vit_merge"))
    parser.add_argument("--data-root", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/cifar100")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--base-epochs", type=int, default=6)
    parser.add_argument("--expert-epochs", type=int, default=6)
    parser.add_argument("--train-per-class", type=int, default=180)
    parser.add_argument("--expert-train-per-class", type=int, default=180)
    parser.add_argument("--eval-per-class", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--expert-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grid-size", type=int, default=17)
    parser.add_argument("--grid-min", type=float, default=-0.25)
    parser.add_argument("--grid-max", type=float, default=1.25)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-steps", type=int, default=31)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data, class_names = prepare_data(args)

    base_model = make_model(args).to(device)
    train_model(base_model, data["train_full"], args.base_epochs, args.base_lr, args.weight_decay, device, "train ViT base")
    reference_state = deepcopy(base_model.state_dict())
    base_vec, spec = vectorize_model(base_model)

    living_model = make_model(args).to(device)
    living_model.load_state_dict(reference_state)
    train_model(living_model, data["train_living"], args.expert_epochs, args.expert_lr, args.weight_decay, device, "fine-tune living")
    living_vec = vectorize_model(living_model)[0]

    object_model = make_model(args).to(device)
    object_model.load_state_dict(reference_state)
    train_model(object_model, data["train_object"], args.expert_epochs, args.expert_lr, args.weight_decay, device, "fine-tune objects")
    object_vec = vectorize_model(object_model)[0]

    tau_living = living_vec - base_vec
    tau_object = object_vec - base_vec
    eval_model = make_model(args).to(device)
    eval_loaders = {"living": data["eval_living"], "object": data["eval_object"]}

    grid_df = build_grid(eval_model, base_vec, tau_living, tau_object, spec, reference_state, eval_loaders, device, args.grid_min, args.grid_max, args.grid_size)
    lambda_df = lambda_sweep(eval_model, base_vec, tau_living, tau_object, spec, reference_state, eval_loaders, device, args.lambda_max, args.lambda_steps)
    methods_df = method_table(eval_model, base_vec, living_vec, object_vec, tau_living, tau_object, spec, reference_state, eval_loaders, device, lambda_df, grid_df, args.seed)
    conflict_df = conflict_table(spec, tau_living, tau_object)
    pca_df = pca_geometry(base_vec, tau_living, tau_object, methods_df)

    grid_df.to_csv(args.output_dir / "grid_metrics.csv", index=False)
    lambda_df.to_csv(args.output_dir / "lambda_sweep.csv", index=False)
    methods_df.to_csv(args.output_dir / "method_metrics.csv", index=False)
    conflict_df.to_csv(args.output_dir / "interference.csv", index=False)
    pca_df.to_csv(args.output_dir / "pca_geometry.csv", index=False)

    plot_landscape(grid_df, methods_df, fig_dir / "merge_landscape.png", args.grid_size)
    plot_methods(grid_df, methods_df, fig_dir / "method_overlay.png", args.grid_size)
    plot_lambda(lambda_df, fig_dir / "lambda_sweep.png")
    plot_interference(conflict_df, fig_dir / "interference_heatmap.png")
    plot_pca(pca_df, fig_dir / "pca_task_vectors.png")

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "device": str(device),
            "num_parameters": int(base_vec.numel()),
            "global_task_vector_cosine": cosine(tau_living, tau_object),
            "tau_living_norm": float(torch.linalg.norm(tau_living)),
            "tau_object_norm": float(torch.linalg.norm(tau_object)),
            "living_superclasses": [class_names[idx] for idx in sorted(LIVING_SUPERCLASSES)],
            "object_superclasses": [class_names[idx] for idx in sorted(OBJECT_SUPERCLASSES)],
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(args.output_dir, config, methods_df, lambda_df, conflict_df, grid_df, class_names)
    print(f"Wrote CIFAR100 ViT-style merge artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
