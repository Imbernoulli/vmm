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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mergeviz.merge_methods import dare_average, linear_average, ties_merge
from mergeviz.weights import VectorSpec, cosine, layer_slices, load_vector_into_model, vectorize_model


class DigitMLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_by_digits(dataset: TensorDataset, digits: set[int]) -> TensorDataset:
    xs, ys = dataset.tensors
    mask = torch.tensor([int(y.item()) in digits for y in ys], dtype=torch.bool)
    return TensorDataset(xs[mask], ys[mask])


def prepare_digits(seed: int, batch_size: int) -> dict[str, DataLoader | TensorDataset]:
    digits = load_digits()
    x = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=seed,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=0.5,
        random_state=seed + 1,
        stratify=y_tmp,
    )

    def dataset(xs: np.ndarray, ys: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))

    train = dataset(x_train, y_train)
    val = dataset(x_val, y_val)
    test = dataset(x_test, y_test)

    def loader(ds: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    out: dict[str, DataLoader | TensorDataset] = {
        "train": train,
        "val": val,
        "test": test,
        "full_train_loader": loader(train, True),
    }
    for digit in range(10):
        out[f"digit_{digit}_train_loader"] = loader(subset_by_digits(train, {digit}), True)
        out[f"digit_{digit}_val_loader"] = loader(subset_by_digits(val, {digit}), False)
        out[f"digit_{digit}_test_loader"] = loader(subset_by_digits(test, {digit}), False)
    return out


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    desc: str,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return {"loss": loss_sum / total, "acc": correct / total, "n": total}


def evaluate_digit_vector(
    model: nn.Module,
    vector: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    digits: tuple[int, int],
    device: torch.device,
) -> dict[str, float]:
    load_vector_into_model(model, vector, spec, reference_state)
    model.to(device)
    left = evaluate(model, loaders[f"digit_{digits[0]}"], device)
    right = evaluate(model, loaders[f"digit_{digits[1]}"], device)
    return {
        "left_loss": left["loss"],
        "right_loss": right["loss"],
        "left_acc": left["acc"],
        "right_acc": right["acc"],
        "avg_loss": 0.5 * (left["loss"] + right["loss"]),
        "worst_loss": max(left["loss"], right["loss"]),
        "avg_acc": 0.5 * (left["acc"] + right["acc"]),
        "worst_acc": min(left["acc"], right["acc"]),
    }


def conflict_metrics(tau_a: Tensor, tau_b: Tensor) -> dict[str, float]:
    active = (tau_a.abs() > 1e-10) & (tau_b.abs() > 1e-10)
    if int(active.sum()) == 0:
        return {"cosine": 0.0, "sign_conflict": 0.0, "weighted_conflict": 0.0}
    conflict = torch.sign(tau_a[active]) != torch.sign(tau_b[active])
    weights = (tau_a[active].abs() * tau_b[active].abs()).to(torch.float64)
    return {
        "cosine": cosine(tau_a, tau_b),
        "sign_conflict": float(conflict.to(torch.float32).mean()),
        "weighted_conflict": float((weights * conflict.to(torch.float64)).sum() / weights.sum().clamp_min(1e-12)),
    }


def layer_conflict_table(spec: VectorSpec, tau_a: Tensor, tau_b: Tensor, left: int, right: int) -> pd.DataFrame:
    rows = []
    for layer, sl in layer_slices(spec).items():
        metrics = conflict_metrics(tau_a[sl], tau_b[sl])
        rows.append({"left_digit": left, "right_digit": right, "layer": layer, **metrics})
    return pd.DataFrame(rows)


def matrix_from_pairs(df: pd.DataFrame, value: str) -> np.ndarray:
    mat = np.full((10, 10), np.nan)
    for _, row in df.iterrows():
        i = int(row["left_digit"])
        j = int(row["right_digit"])
        mat[i, j] = mat[j, i] = float(row[value])
    return mat


def plot_pair_heatmaps(pair_df: pd.DataFrame, out: Path) -> None:
    items = [
        ("linear_worst_acc", "Linear merge worst accuracy", "viridis", 0.0, 1.0),
        ("linear_drop_from_base", "Linear merge drop from base", "magma", None, None),
        ("cosine", "Task-vector cosine", "coolwarm", -1.0, 1.0),
        ("weighted_conflict", "Weighted sign conflict", "magma", 0.0, 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes.ravel(), items, strict=True):
        mat = matrix_from_pairs(pair_df, metric)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("digit")
        ax.set_ylabel("digit")
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                value = mat[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_scatter(pair_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    specs = [
        ("cosine", "linear_drop_from_base", "Cosine vs merge drop"),
        ("sign_conflict", "linear_drop_from_base", "Sign conflict vs merge drop"),
        ("weighted_conflict", "linear_drop_from_base", "Weighted conflict vs merge drop"),
    ]
    for ax, (x_col, y_col, title) in zip(axes, specs, strict=True):
        x = pair_df[x_col].to_numpy(dtype=float)
        y = pair_df[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, color="#457b9d", edgecolors="black", linewidths=0.5)
        if len(x) >= 2 and np.nanstd(x) > 1e-12:
            coeff = np.polyfit(x, y, deg=1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="#e76f51", linewidth=1.4)
        ax.set_xlabel(x_col)
        ax.set_ylabel("base worst acc - merge worst acc")
        ax.set_title(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_layer_atlas(layer_df: pd.DataFrame, out: Path) -> None:
    grouped = (
        layer_df.groupby("layer", as_index=False)
        .agg(cosine=("cosine", "mean"), sign_conflict=("sign_conflict", "mean"), weighted_conflict=("weighted_conflict", "mean"))
    )
    layers = grouped["layer"].tolist()
    matrix = grouped[["cosine", "sign_conflict", "weighted_conflict"]].T.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(layers)), 4.2), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_yticks(range(3), labels=["cosine", "sign conflict", "weighted conflict"])
    ax.set_xticks(range(len(layers)), labels=layers, rotation=45, ha="right", fontsize=8)
    ax.set_title("Average layer-wise conflict across 45 digit-expert pairs")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, pair_df: pd.DataFrame, layer_df: pd.DataFrame, config: dict[str, int | float | str]) -> None:
    corr = pair_df[
        [
            "linear_drop_from_base",
            "cosine",
            "sign_conflict",
            "weighted_conflict",
            "max_layer_weighted_conflict",
        ]
    ].corr(method="spearman")
    best = pair_df.sort_values(["linear_worst_acc", "linear_avg_acc"], ascending=False).head(5)
    worst = pair_df.sort_values(["linear_worst_acc", "linear_avg_acc"], ascending=True).head(5)
    layer_summary = (
        layer_df.groupby("layer", as_index=False)
        .agg(weighted_conflict=("weighted_conflict", "mean"), cosine=("cosine", "mean"))
        .sort_values("weighted_conflict", ascending=False)
        .head(5)
    )

    lines = [
        "# Single-Digit Expert Pairwise Merge Study",
        "",
        "This run trains ten same-base experts on individual sklearn digit classes. It evaluates all 45 digit pairs to test whether task-vector conflict metrics correlate with merge degradation.",
        "",
        "Each expert is a full 10-way classifier fine-tuned only on one digit's examples, so it becomes a class specialist. Pairwise linear averaging tests whether two such specialists can be merged while preserving both single-digit tasks.",
        "",
        "## Correlation With Linear Merge Drop",
        "",
        "Spearman correlation against `linear_drop_from_base = base_worst_acc - linear_merge_worst_acc`:",
        "",
    ]
    for metric in ["cosine", "sign_conflict", "weighted_conflict", "max_layer_weighted_conflict"]:
        value = corr.loc["linear_drop_from_base", metric]
        lines.append(f"- `{metric}`: {value:.3f}")
    lines.extend(["", "## Best Linear Merges", ""])
    for _, row in best.iterrows():
        lines.append(
            f"- digits `{int(row['left_digit'])}`/`{int(row['right_digit'])}`: worst acc {row['linear_worst_acc']:.3f}, drop {row['linear_drop_from_base']:.3f}, weighted conflict {row['weighted_conflict']:.3f}."
        )
    lines.extend(["", "## Worst Linear Merges", ""])
    for _, row in worst.iterrows():
        lines.append(
            f"- digits `{int(row['left_digit'])}`/`{int(row['right_digit'])}`: worst acc {row['linear_worst_acc']:.3f}, drop {row['linear_drop_from_base']:.3f}, weighted conflict {row['weighted_conflict']:.3f}."
        )
    lines.extend(["", "## Most Conflicted Layers On Average", ""])
    for _, row in layer_summary.iterrows():
        lines.append(
            f"- `{row['layer']}`: mean weighted conflict {row['weighted_conflict']:.3f}, mean cosine {row['cosine']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `pairwise_metrics.csv`: one row per digit pair.",
            "- `layer_pairwise_conflict.csv`: per-layer conflict for each digit pair.",
            "- `pairwise_heatmaps.png`: pair matrices for performance and conflict.",
            "- `conflict_vs_drop.png`: scatter plots for conflict metrics vs merge drop.",
            "- `layer_conflict_atlas.png`: average layer-wise conflict atlas.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(config, indent=2),
            "```",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    corr.to_csv(out_dir / "correlations.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-digit experts and evaluate pairwise merge/conflict correlations.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/digit_pairwise_experts"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--base-epochs", type=int, default=8)
    parser.add_argument("--expert-epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-lr", type=float, default=3e-3)
    parser.add_argument("--expert-lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ties-density", type=float, default=0.5)
    parser.add_argument("--dare-drop-rate", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = prepare_digits(args.seed, args.batch_size)

    base_model = DigitMLP(args.hidden).to(device)
    train_model(base_model, data["full_train_loader"], args.base_epochs, args.base_lr, args.weight_decay, device, "train base")
    reference_state = deepcopy(base_model.state_dict())
    base_vec, spec = vectorize_model(base_model)

    experts: dict[int, Tensor] = {}
    for digit in range(10):
        expert = DigitMLP(args.hidden).to(device)
        expert.load_state_dict(reference_state)
        train_model(
            expert,
            data[f"digit_{digit}_train_loader"],
            args.expert_epochs,
            args.expert_lr,
            args.weight_decay,
            device,
            f"fine-tune digit {digit}",
        )
        experts[digit] = vectorize_model(expert)[0]

    eval_model = DigitMLP(args.hidden).to(device)
    loaders = {f"digit_{digit}": data[f"digit_{digit}_test_loader"] for digit in range(10)}

    pair_rows = []
    layer_rows = []
    for left in tqdm(range(10), desc="pair left"):
        for right in range(left + 1, 10):
            tau_left = experts[left] - base_vec
            tau_right = experts[right] - base_vec
            pair_conflict = conflict_metrics(tau_left, tau_right)
            layer_df = layer_conflict_table(spec, tau_left, tau_right, left, right)
            layer_rows.append(layer_df)

            base_metrics = evaluate_digit_vector(eval_model, base_vec, spec, reference_state, loaders, (left, right), device)
            left_expert_metrics = evaluate_digit_vector(eval_model, experts[left], spec, reference_state, loaders, (left, right), device)
            right_expert_metrics = evaluate_digit_vector(eval_model, experts[right], spec, reference_state, loaders, (left, right), device)
            linear_vec = linear_average(base_vec, [tau_left, tau_right])
            ties_vec = ties_merge(base_vec, [tau_left, tau_right], density=args.ties_density)
            dare_vec = dare_average(base_vec, [tau_left, tau_right], drop_rate=args.dare_drop_rate, seed=args.seed + left * 10 + right)
            linear_metrics = evaluate_digit_vector(eval_model, linear_vec, spec, reference_state, loaders, (left, right), device)
            ties_metrics = evaluate_digit_vector(eval_model, ties_vec, spec, reference_state, loaders, (left, right), device)
            dare_metrics = evaluate_digit_vector(eval_model, dare_vec, spec, reference_state, loaders, (left, right), device)

            oracle_worst = min(left_expert_metrics["left_acc"], right_expert_metrics["right_acc"])
            max_layer_weighted = float(layer_df["weighted_conflict"].max())
            pair_rows.append(
                {
                    "left_digit": left,
                    "right_digit": right,
                    **pair_conflict,
                    "max_layer_weighted_conflict": max_layer_weighted,
                    "base_left_acc": base_metrics["left_acc"],
                    "base_right_acc": base_metrics["right_acc"],
                    "base_worst_acc": base_metrics["worst_acc"],
                    "left_expert_left_acc": left_expert_metrics["left_acc"],
                    "left_expert_right_acc": left_expert_metrics["right_acc"],
                    "right_expert_left_acc": right_expert_metrics["left_acc"],
                    "right_expert_right_acc": right_expert_metrics["right_acc"],
                    "oracle_worst_acc": oracle_worst,
                    "linear_left_acc": linear_metrics["left_acc"],
                    "linear_right_acc": linear_metrics["right_acc"],
                    "linear_avg_acc": linear_metrics["avg_acc"],
                    "linear_worst_acc": linear_metrics["worst_acc"],
                    "linear_avg_loss": linear_metrics["avg_loss"],
                    "linear_worst_loss": linear_metrics["worst_loss"],
                    "linear_drop_from_base": base_metrics["worst_acc"] - linear_metrics["worst_acc"],
                    "linear_drop_from_oracle": oracle_worst - linear_metrics["worst_acc"],
                    "ties_worst_acc": ties_metrics["worst_acc"],
                    "dare_worst_acc": dare_metrics["worst_acc"],
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    layer_df = pd.concat(layer_rows, ignore_index=True)
    pair_df.to_csv(out_dir / "pairwise_metrics.csv", index=False)
    layer_df.to_csv(out_dir / "layer_pairwise_conflict.csv", index=False)

    plot_pair_heatmaps(pair_df, out_dir / "pairwise_heatmaps.png")
    plot_scatter(pair_df, out_dir / "conflict_vs_drop.png")
    plot_layer_atlas(layer_df, out_dir / "layer_conflict_atlas.png")

    config = {
        "seed": args.seed,
        "hidden": args.hidden,
        "base_epochs": args.base_epochs,
        "expert_epochs": args.expert_epochs,
        "base_lr": args.base_lr,
        "expert_lr": args.expert_lr,
        "weight_decay": args.weight_decay,
        "ties_density": args.ties_density,
        "dare_drop_rate": args.dare_drop_rate,
        "device": str(device),
        "num_pairs": int(len(pair_df)),
        "num_parameters": int(base_vec.numel()),
    }
    (out_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(out_dir, pair_df, layer_df, config)
    print(f"Wrote digit pairwise expert artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
