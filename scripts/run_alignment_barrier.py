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
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class OneHiddenMLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(seed: int, batch_size: int) -> tuple[DataLoader, DataLoader]:
    digits = load_digits()
    x = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, weight_decay: float, device: torch.device, desc: str) -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


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
    return {"loss": loss_sum / total, "acc": correct / total}


def interpolate_state(left: dict[str, Tensor], right: dict[str, Tensor], t: float) -> dict[str, Tensor]:
    return {
        name: (1.0 - t) * left[name] + t * right[name]
        for name in left.keys()
    }


def unit_features(model: OneHiddenMLP) -> Tensor:
    with torch.no_grad():
        incoming = model.fc1.weight.detach().cpu()
        bias = model.fc1.bias.detach().cpu().unsqueeze(1)
        outgoing = model.fc2.weight.detach().cpu().T
        features = torch.cat([incoming, bias, outgoing], dim=1)
        return F.normalize(features, dim=1)


def align_hidden_units(reference: OneHiddenMLP, target: OneHiddenMLP) -> OneHiddenMLP:
    ref_features = unit_features(reference)
    target_features = unit_features(target)
    similarity = ref_features @ target_features.T
    rows, cols = linear_sum_assignment((-similarity).numpy())
    order = np.empty_like(cols)
    order[rows] = cols
    order_tensor = torch.from_numpy(order).long()

    aligned = deepcopy(target).cpu()
    with torch.no_grad():
        aligned.fc1.weight.copy_(target.fc1.weight.detach().cpu()[order_tensor])
        aligned.fc1.bias.copy_(target.fc1.bias.detach().cpu()[order_tensor])
        aligned.fc2.weight.copy_(target.fc2.weight.detach().cpu()[:, order_tensor])
        aligned.fc2.bias.copy_(target.fc2.bias.detach().cpu())
    return aligned


def evaluate_path(
    model_template: OneHiddenMLP,
    left: dict[str, Tensor],
    right: dict[str, Tensor],
    loader: DataLoader,
    device: torch.device,
    steps: int,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    model = deepcopy(model_template).to(device)
    for t in np.linspace(0.0, 1.0, steps):
        model.load_state_dict(interpolate_state(left, right, float(t)))
        metrics = evaluate(model, loader, device)
        rows.append({"t": float(t), f"{prefix}_loss": metrics["loss"], f"{prefix}_acc": metrics["acc"]})
    return pd.DataFrame(rows)


def barrier(losses: pd.Series) -> float:
    endpoints = max(float(losses.iloc[0]), float(losses.iloc[-1]))
    return float(losses.max() - endpoints)


def plot_paths(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(df["t"], df["before_acc"], label="before alignment", color="#e76f51")
    ax.plot(df["t"], df["after_acc"], label="after alignment", color="#2a9d8f")
    ax.set_xlabel("interpolation t")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Accuracy along linear path")
    ax.legend()
    ax = axes[1]
    ax.plot(df["t"], df["before_loss"], label="before alignment", color="#e76f51")
    ax.plot(df["t"], df["after_loss"], label="after alignment", color="#2a9d8f")
    ax.set_xlabel("interpolation t")
    ax.set_ylabel("cross entropy")
    ax.set_title("Loss barrier from permutation mismatch")
    ax.legend()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict[str, float | int | str]) -> None:
    lines = [
        "# Independent-Initialization Alignment Barrier",
        "",
        "This is a small surrogate for the proposal's alignment-before-merge visualization. Two one-hidden-layer MLPs are trained on the same sklearn digits task from different random initializations. The second model is then permutation-aligned to the first by matching hidden-unit feature vectors with the Hungarian algorithm.",
        "",
        "## Result",
        "",
        f"- Model A accuracy: {summary['model_a_acc']:.3f}.",
        f"- Model B accuracy: {summary['model_b_acc']:.3f}.",
        f"- Midpoint accuracy before alignment: {summary['midpoint_before_acc']:.3f}.",
        f"- Midpoint accuracy after alignment: {summary['midpoint_after_acc']:.3f}.",
        f"- Loss barrier before alignment: {summary['barrier_before']:.3f}.",
        f"- Loss barrier after alignment: {summary['barrier_after']:.3f}.",
        "",
        "## Interpretation",
        "",
        "If the before-alignment midpoint is poor while the after-alignment midpoint improves, the apparent barrier was partly a coordinate/permutation artifact rather than pure functional incompatibility. This is the small-model analogue of why Git Re-Basin style alignment matters before weight-space merging.",
        "",
        "See `interpolation_alignment.png` and `path_metrics.csv` in this directory.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an independent-initialization alignment barrier demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/alignment_barrier"))
    parser.add_argument("--seed-a", type=int, default=101)
    parser.add_argument("--seed-b", type=int, default=202)
    parser.add_argument("--data-seed", type=int, default=13)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=41)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_loader, test_loader = prepare_data(args.data_seed, args.batch_size)

    set_seed(args.seed_a)
    model_a = OneHiddenMLP(args.hidden)
    train(model_a, train_loader, args.epochs, args.lr, args.weight_decay, device, "train model A")
    model_a.cpu()

    set_seed(args.seed_b)
    model_b = OneHiddenMLP(args.hidden)
    train(model_b, train_loader, args.epochs, args.lr, args.weight_decay, device, "train model B")
    model_b.cpu()

    aligned_b = align_hidden_units(model_a, model_b)
    template = OneHiddenMLP(args.hidden)
    before = evaluate_path(template, model_a.state_dict(), model_b.state_dict(), test_loader, device, args.steps, "before")
    after = evaluate_path(template, model_a.state_dict(), aligned_b.state_dict(), test_loader, device, args.steps, "after")
    df = before.merge(after, on="t")
    df.to_csv(args.output_dir / "path_metrics.csv", index=False)
    plot_paths(df, args.output_dir / "interpolation_alignment.png")

    model_a.to(device)
    model_b.to(device)
    a_metrics = evaluate(model_a, test_loader, device)
    b_metrics = evaluate(model_b, test_loader, device)
    midpoint = df.iloc[len(df) // 2]
    summary: dict[str, float | int | str] = {
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "steps": args.steps,
        "device": str(device),
        "model_a_acc": a_metrics["acc"],
        "model_b_acc": b_metrics["acc"],
        "midpoint_before_acc": float(midpoint["before_acc"]),
        "midpoint_after_acc": float(midpoint["after_acc"]),
        "barrier_before": barrier(df["before_loss"]),
        "barrier_after": barrier(df["after_loss"]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.output_dir, summary)
    print(f"Wrote alignment barrier artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
