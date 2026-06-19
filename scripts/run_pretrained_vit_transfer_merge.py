#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
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
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ViT_B_16_Weights, vit_b_16
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_cifar100_vit_merge import (  # noqa: E402
    Cifar100Coarse,
    LIVING_SUPERCLASSES,
    OBJECT_SUPERCLASSES,
    default_device,
    sample_indices,
)

from mergeviz.merge_methods import dare_average, linear_average, slerp, task_arithmetic, ties_dare_merge, ties_merge
from mergeviz.weights import cosine, layer_slices, load_vector_into_model, project_to_plane, vectorize_model


def make_loader(dataset: Cifar100Coarse, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )


def prepare_image_loaders(args: argparse.Namespace, transform: nn.Module) -> dict[str, DataLoader]:
    train_set = Cifar100Coarse(args.data_root, train=True, transform=transform)
    test_set = Cifar100Coarse(args.data_root, train=False, transform=transform)
    all_classes = set(range(20))
    base_indices = sample_indices(train_set.targets, all_classes, args.train_per_class, args.seed)
    living_indices = sample_indices(train_set.targets, LIVING_SUPERCLASSES, args.expert_train_per_class, args.seed + 1)
    object_indices = sample_indices(train_set.targets, OBJECT_SUPERCLASSES, args.expert_train_per_class, args.seed + 2)
    eval_living = sample_indices(test_set.targets, LIVING_SUPERCLASSES, args.eval_per_class, args.seed + 3)
    eval_object = sample_indices(test_set.targets, OBJECT_SUPERCLASSES, args.eval_per_class, args.seed + 4)
    return {
        "train_full": make_loader(train_set, base_indices, args.image_batch_size),
        "train_living": make_loader(train_set, living_indices, args.image_batch_size),
        "train_object": make_loader(train_set, object_indices, args.image_batch_size),
        "eval_living": make_loader(test_set, eval_living, args.image_batch_size),
        "eval_object": make_loader(test_set, eval_object, args.image_batch_size),
    }


@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> TensorDataset:
    features = []
    labels = []
    backbone.eval()
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        features.append(backbone(x).detach().cpu().to(torch.float32))
        labels.append(y.detach().cpu().to(torch.long))
    return TensorDataset(torch.cat(features, dim=0), torch.cat(labels, dim=0))


def feature_loader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_head(feature_dim: int, num_classes: int = 20) -> nn.Linear:
    head = nn.Linear(feature_dim, num_classes)
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, mean=0.0, std=0.01)
    return head


def train_head(head: nn.Module, loader: DataLoader, epochs: int, lr: float, weight_decay: float, device: torch.device, desc: str) -> None:
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    head.to(device)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        head.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_head(head: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = head(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total), "n": total}


def evaluate_vector(
    head: nn.Module,
    vector: Tensor,
    spec,
    reference: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> dict[str, float]:
    load_vector_into_model(head, vector, spec, reference)
    head.to(device)
    living = evaluate_head(head, loaders["eval_living"], device)
    obj = evaluate_head(head, loaders["eval_object"], device)
    return {
        "living_loss": living["loss"],
        "object_loss": obj["loss"],
        "living_acc": living["acc"],
        "object_acc": obj["acc"],
        "avg_loss": 0.5 * (living["loss"] + obj["loss"]),
        "worst_loss": max(living["loss"], obj["loss"]),
        "avg_acc": 0.5 * (living["acc"] + obj["acc"]),
        "worst_acc": min(living["acc"], obj["acc"]),
    }


def grid_metrics(base_vec: Tensor, tau_living: Tensor, tau_object: Tensor, head: nn.Module, spec, reference, loaders, args, device):
    rows = []
    for alpha in tqdm(np.linspace(args.grid_min, args.grid_max, args.grid_size), desc="pretrained ViT grid alpha"):
        for beta in np.linspace(args.grid_min, args.grid_max, args.grid_size):
            vector = base_vec + float(alpha) * tau_living + float(beta) * tau_object
            rows.append({"alpha": float(alpha), "beta": float(beta), **evaluate_vector(head, vector, spec, reference, loaders, device)})
    return pd.DataFrame(rows)


def lambda_sweep(base_vec: Tensor, tau_living: Tensor, tau_object: Tensor, head, spec, reference, loaders, args, device):
    rows = []
    for lam in tqdm(np.linspace(0.0, args.lambda_max, args.lambda_steps), desc="pretrained ViT lambda"):
        vector = task_arithmetic(base_vec, [tau_living, tau_object], float(lam))
        rows.append({"lambda": float(lam), **evaluate_vector(head, vector, spec, reference, loaders, device)})
    return pd.DataFrame(rows)


def sign_conflict(a: Tensor, b: Tensor) -> tuple[float, float]:
    nonzero = (a != 0) & (b != 0)
    if int(nonzero.sum()) == 0:
        return 0.0, 0.0
    conflict = (torch.sign(a[nonzero]) != torch.sign(b[nonzero])).to(torch.float32)
    weights = torch.minimum(a[nonzero].abs(), b[nonzero].abs())
    return float(conflict.mean()), float((conflict * weights).sum() / weights.sum().clamp_min(1e-12))


def interference(spec, tau_living: Tensor, tau_object: Tensor) -> pd.DataFrame:
    rows = []
    for name, slc in layer_slices(spec).items():
        left = tau_living[slc]
        right = tau_object[slc]
        sign, weighted = sign_conflict(left, right)
        rows.append(
            {
                "layer": name,
                "numel": int(left.numel()),
                "cosine": cosine(left, right),
                "sign_conflict": sign,
                "weighted_conflict": weighted,
                "living_norm": float(torch.linalg.norm(left)),
                "object_norm": float(torch.linalg.norm(right)),
            }
        )
    return pd.DataFrame(rows)


def method_metrics(base_vec, living_vec, object_vec, tau_living, tau_object, lambda_df, grid_df, head, spec, reference, loaders, device) -> pd.DataFrame:
    candidates: list[tuple[str, Tensor, str]] = [
        ("base", base_vec, "reference"),
        ("expert_living", living_vec, "reference"),
        ("expert_object", object_vec, "reference"),
        ("linear_average", linear_average(base_vec, [tau_living, tau_object]), "merge"),
        ("slerp_experts", slerp(living_vec, object_vec, 0.5), "merge"),
        ("ties", ties_merge(base_vec, [tau_living, tau_object], density=0.5), "merge"),
        ("dare_average", dare_average(base_vec, [tau_living, tau_object], drop_rate=0.5, seed=17), "merge"),
        ("ties_dare", ties_dare_merge(base_vec, [tau_living, tau_object], density=0.5, drop_rate=0.5, seed=19), "merge"),
    ]
    best_lambda = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    candidates.append(
        (
            "task_arithmetic_best_lambda",
            task_arithmetic(base_vec, [tau_living, tau_object], float(best_lambda["lambda"])),
            "merge",
        )
    )
    best_grid = grid_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    candidates.append(
        (
            "validation_grid_best",
            base_vec + float(best_grid["alpha"]) * tau_living + float(best_grid["beta"]) * tau_object,
            "merge",
        )
    )
    rows = []
    for name, vector, kind in candidates:
        alpha, beta, residual = project_to_plane(vector, base_vec, tau_living, tau_object)
        rows.append(
            {
                "method": name,
                "kind": kind,
                "alpha": alpha,
                "beta": beta,
                "plane_residual": residual,
                **evaluate_vector(head, vector, spec, reference, loaders, device),
            }
        )
    return pd.DataFrame(rows)


def plot_landscape(grid_df: pd.DataFrame, out: Path) -> None:
    pivot = grid_df.pivot(index="beta", columns="alpha", values="worst_acc").sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(6.3, 5.2), constrained_layout=True)
    im = ax.imshow(
        pivot.values,
        origin="lower",
        extent=[grid_df["alpha"].min(), grid_df["alpha"].max(), grid_df["beta"].min(), grid_df["beta"].max()],
        aspect="auto",
        cmap="viridis",
    )
    ax.scatter([0, 1, 0, 0.5], [0, 0, 1, 0.5], color=["white", "#e76f51", "#2a9d8f", "black"], s=42)
    ax.set_xlabel("alpha living")
    ax.set_ylabel("beta object")
    ax.set_title("Pretrained ViT frozen-backbone merge landscape")
    fig.colorbar(im, ax=ax, label="worst-task accuracy")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_methods(grid_df: pd.DataFrame, method_df: pd.DataFrame, out: Path) -> None:
    pivot = grid_df.pivot(index="beta", columns="alpha", values="worst_acc").sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.imshow(
        pivot.values,
        origin="lower",
        extent=[grid_df["alpha"].min(), grid_df["alpha"].max(), grid_df["beta"].min(), grid_df["beta"].max()],
        aspect="auto",
        cmap="viridis",
        alpha=0.82,
    )
    for _, row in method_df.iterrows():
        ax.scatter(row["alpha"], row["beta"], s=40, color="white", edgecolor="black", linewidth=0.9)
        ax.annotate(row["method"], (row["alpha"], row["beta"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("alpha living")
    ax.set_ylabel("beta object")
    ax.set_title("Pretrained ViT merge methods")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_lambda(lambda_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(lambda_df["lambda"], lambda_df["living_acc"], marker="o", label="living acc", color="#2a9d8f")
    ax.plot(lambda_df["lambda"], lambda_df["object_acc"], marker="o", label="object acc", color="#e76f51")
    ax.plot(lambda_df["lambda"], lambda_df["worst_acc"], marker="o", label="worst acc", color="#6d597a")
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Pretrained ViT task-arithmetic path")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_interference(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.bar(df["layer"], df["weighted_conflict"], color="#457b9d")
    ax.set_ylabel("weighted sign conflict")
    ax.set_ylim(0, 1)
    ax.set_title("Pretrained ViT head-vector conflict")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, summary: dict[str, object], method_df: pd.DataFrame) -> None:
    best = method_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    base = method_df[method_df["method"] == "base"].iloc[0]
    linear = method_df[method_df["method"] == "linear_average"].iloc[0]
    lines = [
        "# Pretrained ViT Transfer Merge",
        "",
        "This run uses an ImageNet-pretrained torchvision ViT-B/16 as a frozen feature extractor, then trains CIFAR100 coarse-label linear heads for a base model and two class-group experts. It is a pretrained ViT transfer experiment, but not full-backbone fine-tuning.",
        "",
        "Tasks are living superclasses versus object/vehicle superclasses. The merge plane is formed from the linear-head task vectors.",
        "",
        "## Key Results",
        "",
        f"- Best method by worst-task accuracy: `{best['method']}` with worst accuracy {best['worst_acc']:.3f}.",
        f"- Base worst accuracy: {base['worst_acc']:.3f}.",
        f"- Linear average worst accuracy: {linear['worst_acc']:.3f}.",
        f"- Global head task-vector cosine: {summary['global_task_vector_cosine']:.3f}.",
        "",
        "| method | alpha | beta | living acc | object acc | worst acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in method_df.sort_values("worst_acc", ascending=False).iterrows():
        lines.append(
            f"| {row['method']} | {row['alpha']:.3f} | {row['beta']:.3f} | {row['living_acc']:.3f} | {row['object_acc']:.3f} | {row['worst_acc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `grid_metrics.csv`: alpha/beta grid over the frozen-backbone head merge plane.",
            "- `method_metrics.csv`: endpoints and merge methods.",
            "- `lambda_sweep.csv`: task-arithmetic path metrics.",
            "- `interference.csv`: head weight/bias conflict metrics.",
            "- `figures/*.png`: landscape, methods, lambda path, and conflict plots.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(vars(args), indent=2, default=str),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen-backbone pretrained ViT transfer merge on CIFAR100 coarse labels.")
    parser.add_argument("--data-root", default="/srv/home/bohanlyu/MLS-Bench/vendor/data/cifar100")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pretrained_vit_transfer_merge"))
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--expert-train-per-class", type=int, default=40)
    parser.add_argument("--eval-per-class", type=int, default=50)
    parser.add_argument("--image-batch-size", type=int, default=48)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--base-epochs", type=int, default=80)
    parser.add_argument("--expert-epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--grid-min", type=float, default=-0.25)
    parser.add_argument("--grid-max", type=float, default=1.25)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-steps", type=int, default=31)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    weights = ViT_B_16_Weights.DEFAULT
    backbone = vit_b_16(weights=weights)
    backbone.heads = nn.Identity()
    backbone.to(device)
    backbone.eval()
    loaders = prepare_image_loaders(args, weights.transforms())
    feature_sets = {
        name: extract_features(backbone, loader, device, f"extract {name}")
        for name, loader in loaders.items()
    }
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    feature_dim = int(feature_sets["train_full"].tensors[0].shape[1])
    feature_loaders = {
        "train_full": feature_loader(feature_sets["train_full"], args.feature_batch_size, True),
        "train_living": feature_loader(feature_sets["train_living"], args.feature_batch_size, True),
        "train_object": feature_loader(feature_sets["train_object"], args.feature_batch_size, True),
        "eval_living": feature_loader(feature_sets["eval_living"], args.feature_batch_size, False),
        "eval_object": feature_loader(feature_sets["eval_object"], args.feature_batch_size, False),
    }

    base_head = make_head(feature_dim)
    train_head(base_head, feature_loaders["train_full"], args.base_epochs, args.lr, args.weight_decay, device, "train pretrained ViT base head")
    living_head = deepcopy(base_head)
    object_head = deepcopy(base_head)
    train_head(living_head, feature_loaders["train_living"], args.expert_epochs, args.lr, args.weight_decay, device, "train living head")
    train_head(object_head, feature_loaders["train_object"], args.expert_epochs, args.lr, args.weight_decay, device, "train object head")

    base_vec, spec = vectorize_model(base_head.cpu())
    living_vec, _ = vectorize_model(living_head.cpu())
    object_vec, _ = vectorize_model(object_head.cpu())
    reference = base_head.state_dict()
    tau_living = living_vec - base_vec
    tau_object = object_vec - base_vec
    eval_head = make_head(feature_dim)

    grid_df = grid_metrics(base_vec, tau_living, tau_object, eval_head, spec, reference, feature_loaders, args, device)
    lambda_df = lambda_sweep(base_vec, tau_living, tau_object, eval_head, spec, reference, feature_loaders, args, device)
    method_df = method_metrics(
        base_vec,
        living_vec,
        object_vec,
        tau_living,
        tau_object,
        lambda_df,
        grid_df,
        eval_head,
        spec,
        reference,
        feature_loaders,
        device,
    )
    interference_df = interference(spec, tau_living, tau_object)

    grid_df.to_csv(args.output_dir / "grid_metrics.csv", index=False)
    lambda_df.to_csv(args.output_dir / "lambda_sweep.csv", index=False)
    method_df.to_csv(args.output_dir / "method_metrics.csv", index=False)
    interference_df.to_csv(args.output_dir / "interference.csv", index=False)

    plot_landscape(grid_df, fig_dir / "merge_landscape.png")
    plot_methods(grid_df, method_df, fig_dir / "method_overlay.png")
    plot_lambda(lambda_df, fig_dir / "lambda_sweep.png")
    plot_interference(interference_df, fig_dir / "interference_heatmap.png")

    summary = {
        "pretrained_backbone": "torchvision ViT_B_16_Weights.IMAGENET1K_V1",
        "frozen_backbone": True,
        "feature_dim": feature_dim,
        "device": str(device),
        "train_per_class": args.train_per_class,
        "expert_train_per_class": args.expert_train_per_class,
        "eval_per_class": args.eval_per_class,
        "grid_size": args.grid_size,
        "global_task_vector_cosine": cosine(tau_living, tau_object),
        "tau_living_norm": float(torch.linalg.norm(tau_living)),
        "tau_object_norm": float(torch.linalg.norm(tau_object)),
        "best_method": str(method_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]["method"]),
        "best_worst_acc": float(method_df["worst_acc"].max()),
        "linear_worst_acc": float(method_df[method_df["method"] == "linear_average"].iloc[0]["worst_acc"]),
        "base_worst_acc": float(method_df[method_df["method"] == "base"].iloc[0]["worst_acc"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.output_dir, args, summary, method_df)
    print(f"Wrote pretrained ViT transfer merge artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
