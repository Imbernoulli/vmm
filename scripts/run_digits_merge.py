#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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

from mergeviz.merge_methods import (
    dare_average,
    fisher_weighted_average,
    linear_average,
    slerp,
    task_arithmetic,
    ties_dare_merge,
    ties_merge,
)
from mergeviz.weights import (
    VectorSpec,
    cosine,
    interpolation_barrier,
    layer_slices,
    load_vector_into_model,
    project_to_plane,
    vectorize_state_dict,
    vectorize_model,
)


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

    low_train = subset_by_digits(train, set(range(5)))
    high_train = subset_by_digits(train, set(range(5, 10)))
    low_val = subset_by_digits(val, set(range(5)))
    high_val = subset_by_digits(val, set(range(5, 10)))
    low_test = subset_by_digits(test, set(range(5)))
    high_test = subset_by_digits(test, set(range(5, 10)))

    def loader(ds: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return {
        "train": train,
        "val": val,
        "test": test,
        "low_train_loader": loader(low_train, True),
        "high_train_loader": loader(high_train, True),
        "full_train_loader": loader(train, True),
        "low_val_loader": loader(low_val, False),
        "high_val_loader": loader(high_val, False),
        "low_test_loader": loader(low_test, False),
        "high_test_loader": loader(high_test, False),
    }


def subset_by_digits(dataset: TensorDataset, digits: set[int]) -> TensorDataset:
    xs, ys = dataset.tensors
    mask = torch.tensor([int(y.item()) in digits for y in ys], dtype=torch.bool)
    return TensorDataset(xs[mask], ys[mask])


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
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
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
    low = evaluate(model, loaders["low"], device)
    high = evaluate(model, loaders["high"], device)
    return {
        "task_a_loss": low["loss"],
        "task_b_loss": high["loss"],
        "task_a_acc": low["acc"],
        "task_b_acc": high["acc"],
        "avg_loss": 0.5 * (low["loss"] + high["loss"]),
        "worst_loss": max(low["loss"], high["loss"]),
        "avg_acc": 0.5 * (low["acc"] + high["acc"]),
        "worst_acc": min(low["acc"], high["acc"]),
    }


def compute_fisher(
    model: nn.Module,
    vector: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tensor:
    load_vector_into_model(model, vector, spec, reference_state)
    model.to(device)
    model.train()
    fisher = torch.zeros_like(vector, dtype=torch.float32)
    batches = 0
    names = [item.name for item in spec.tensors]
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        grads = []
        state_named_params = dict(model.named_parameters())
        for name in names:
            param = state_named_params.get(name)
            if param is None or param.grad is None:
                grads.append(torch.zeros(spec.tensors[names.index(name)].numel, dtype=torch.float32))
            else:
                grads.append(param.grad.detach().cpu().to(torch.float32).reshape(-1).pow(2))
        fisher += torch.cat(grads)
        batches += 1
        if batches >= max_batches:
            break
    return fisher / max(1, batches)


@torch.no_grad()
def collect_linear_covariances(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    model.to(device)
    model.eval()
    covariances: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float64).cpu()
            ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
            augmented = torch.cat([x, ones], dim=1)
            covariances[name] = covariances.get(name, torch.zeros((augmented.shape[1], augmented.shape[1]), dtype=torch.float64))
            covariances[name] = covariances[name] + augmented.T @ augmented
            counts[name] = counts.get(name, 0) + int(augmented.shape[0])

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook_for(name)))

    batches = 0
    for x, _y in loader:
        model(x.to(device))
        batches += 1
        if batches >= max_batches:
            break

    for handle in handles:
        handle.remove()

    rows = []
    for name, cov in covariances.items():
        rows.append(
            {
                "layer": name,
                "examples": counts[name],
                "augmented_dim": int(cov.shape[0]),
                "trace": float(torch.trace(cov)),
                "condition": float(torch.linalg.cond(cov + torch.eye(cov.shape[0], dtype=cov.dtype) * 1e-8)),
            }
        )
    return covariances, pd.DataFrame(rows)


def regmean_merge_state_dict(
    expert_states: list[dict[str, Tensor]],
    covariances: list[dict[str, Tensor]],
    reference_state: dict[str, Tensor],
    ridge: float,
) -> tuple[dict[str, Tensor], pd.DataFrame]:
    merged: dict[str, Tensor] = {}
    for name, value in reference_state.items():
        if torch.is_floating_point(value) and all(name in state for state in expert_states):
            stacked = torch.stack([state[name].detach().cpu().to(torch.float32) for state in expert_states])
            merged[name] = stacked.mean(dim=0).to(dtype=value.dtype)
        else:
            merged[name] = value.detach().clone()

    rows = []
    linear_names = sorted({name.rsplit(".", 1)[0] for name in reference_state if name.endswith(".weight")})
    for layer in linear_names:
        weight_name = f"{layer}.weight"
        bias_name = f"{layer}.bias"
        if bias_name not in reference_state:
            continue
        if not all(layer in cov for cov in covariances):
            continue
        if not all(weight_name in state and bias_name in state for state in expert_states):
            continue

        denom = None
        numerator = None
        for state, cov_by_layer in zip(expert_states, covariances, strict=True):
            cov = cov_by_layer[layer].to(torch.float64)
            weight = state[weight_name].detach().cpu().to(torch.float64)
            bias = state[bias_name].detach().cpu().to(torch.float64).reshape(-1, 1)
            augmented_weight = torch.cat([weight, bias], dim=1)
            denom = cov if denom is None else denom + cov
            part = augmented_weight @ cov
            numerator = part if numerator is None else numerator + part

        assert denom is not None and numerator is not None
        dim = denom.shape[0]
        ridge_value = ridge * float(torch.trace(denom)) / max(1, dim)
        system = denom + torch.eye(dim, dtype=torch.float64) * max(ridge_value, ridge)
        merged_augmented = torch.linalg.solve(system.T, numerator.T).T
        merged[weight_name] = merged_augmented[:, :-1].to(dtype=reference_state[weight_name].dtype)
        merged[bias_name] = merged_augmented[:, -1].to(dtype=reference_state[bias_name].dtype)
        rows.append(
            {
                "layer": layer,
                "out_features": int(merged[weight_name].shape[0]),
                "in_features": int(merged[weight_name].shape[1]),
                "ridge_value": float(max(ridge_value, ridge)),
            }
        )
    return merged, pd.DataFrame(rows)


def vector_from_state_dict(state: dict[str, Tensor], spec: VectorSpec) -> Tensor:
    names = [item.name for item in spec.tensors]
    vector, _ = vectorize_state_dict(state, names=names)
    return vector


def parse_scale_grid(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("--layerwise-scale-grid must contain at least one float")
    return values


def layerwise_task_arithmetic_search(
    model: nn.Module,
    base: Tensor,
    tau_a: Tensor,
    tau_b: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    scale_candidates: list[float],
) -> tuple[Tensor, pd.DataFrame]:
    combined_delta = tau_a + tau_b
    current = base.clone()
    rows = []
    for tensor_spec in tqdm(spec.tensors, desc="layerwise task arithmetic"):
        sl = slice(tensor_spec.start, tensor_spec.end)
        candidates = []
        for scale in scale_candidates:
            candidate = current.clone()
            candidate[sl] = base[sl] + float(scale) * combined_delta[sl]
            metrics = evaluate_vector(model, candidate, spec, reference_state, loaders, device)
            candidates.append((float(scale), candidate, metrics))
        scale, best_candidate, best_metrics = sorted(
            candidates,
            key=lambda item: (item[2]["worst_acc"], item[2]["avg_acc"], -item[2]["worst_loss"]),
            reverse=True,
        )[0]
        current = best_candidate
        rows.append({"layer": tensor_spec.name, "scale": scale, **best_metrics})
    return current, pd.DataFrame(rows)


def build_grid(
    model: nn.Module,
    base: Tensor,
    tau_a: Tensor,
    tau_b: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    grid_min: float,
    grid_max: float,
    grid_size: int,
) -> pd.DataFrame:
    rows = []
    values = np.linspace(grid_min, grid_max, grid_size)
    for alpha in tqdm(values, desc="grid alpha"):
        for beta in values:
            point = base + float(alpha) * tau_a + float(beta) * tau_b
            metrics = evaluate_vector(model, point, spec, reference_state, loaders, device)
            rows.append({"alpha": float(alpha), "beta": float(beta), **metrics})
    return pd.DataFrame(rows)


def lambda_sweep(
    model: nn.Module,
    base: Tensor,
    tau_a: Tensor,
    tau_b: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    lambda_max: float,
    steps: int,
) -> pd.DataFrame:
    rows = []
    for lam in np.linspace(0.0, lambda_max, steps):
        point = task_arithmetic(base, [tau_a, tau_b], float(lam))
        metrics = evaluate_vector(model, point, spec, reference_state, loaders, device)
        rows.append({"lambda": float(lam), **metrics})
    return pd.DataFrame(rows)


def method_table(
    model: nn.Module,
    base: Tensor,
    expert_a: Tensor,
    expert_b: Tensor,
    tau_a: Tensor,
    tau_b: Tensor,
    spec: VectorSpec,
    reference_state: dict[str, Tensor],
    loaders: dict[str, DataLoader],
    device: torch.device,
    lambda_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    fisher_a: Tensor,
    fisher_b: Tensor,
    seed: int,
    extra_methods: list[tuple[str, Tensor, dict[str, float | str]]] | None = None,
) -> pd.DataFrame:
    best_lam_row = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    best_grid_row = grid_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    methods: list[tuple[str, Tensor, dict[str, float | str]]] = [
        ("base", base, {"kind": "reference"}),
        ("expert_a_low_digits", expert_a, {"kind": "reference"}),
        ("expert_b_high_digits", expert_b, {"kind": "reference"}),
        ("linear_average", linear_average(base, [tau_a, tau_b]), {"kind": "merge"}),
        (
            "task_arithmetic_best_lambda",
            task_arithmetic(base, [tau_a, tau_b], float(best_lam_row["lambda"])),
            {"kind": "merge", "lambda": float(best_lam_row["lambda"])},
        ),
        ("slerp_experts", slerp(expert_a, expert_b, t=0.5), {"kind": "merge"}),
        ("ties_density_0.5", ties_merge(base, [tau_a, tau_b], density=0.5), {"kind": "merge", "density": 0.5}),
        ("dare_drop_0.5", dare_average(base, [tau_a, tau_b], drop_rate=0.5, seed=seed), {"kind": "merge", "drop_rate": 0.5}),
        (
            "ties_dare_0.5",
            ties_dare_merge(base, [tau_a, tau_b], density=0.5, drop_rate=0.5, seed=seed),
            {"kind": "merge", "density": 0.5, "drop_rate": 0.5},
        ),
        (
            "fisher_weighted",
            fisher_weighted_average([expert_a, expert_b], [fisher_a, fisher_b]),
            {"kind": "merge"},
        ),
    ]
    if extra_methods:
        methods.extend(extra_methods)
    methods.append(
        (
            "validation_grid_best",
            base + float(best_grid_row["alpha"]) * tau_a + float(best_grid_row["beta"]) * tau_b,
            {"kind": "oracle", "alpha": float(best_grid_row["alpha"]), "beta": float(best_grid_row["beta"])},
        )
    )
    rows = []
    for name, vector, details in methods:
        metrics = evaluate_vector(model, vector, spec, reference_state, loaders, device)
        alpha, beta, residual = project_to_plane(vector, base, tau_a, tau_b)
        rows.append(
            {
                "method": name,
                "alpha": alpha,
                "beta": beta,
                "plane_residual": residual,
                **metrics,
                **details,
            }
        )
    return pd.DataFrame(rows)


def interference_table(spec: VectorSpec, tau_a: Tensor, tau_b: Tensor) -> pd.DataFrame:
    rows = []
    for layer, sl in layer_slices(spec).items():
        a = tau_a[sl]
        b = tau_b[sl]
        active = (a.abs() > 1e-10) & (b.abs() > 1e-10)
        if int(active.sum()) == 0:
            sign_conflict = 0.0
            weighted_conflict = 0.0
        else:
            conflict = torch.sign(a[active]) != torch.sign(b[active])
            sign_conflict = float(conflict.to(torch.float32).mean())
            weights = (a[active].abs() * b[active].abs()).to(torch.float64)
            weighted_conflict = float((weights * conflict.to(torch.float64)).sum() / weights.sum().clamp_min(1e-12))
        rows.append(
            {
                "layer": layer,
                "numel": int(a.numel()),
                "cosine": cosine(a, b),
                "sign_conflict": sign_conflict,
                "weighted_conflict": weighted_conflict,
                "tau_a_norm": float(torch.linalg.norm(a)),
                "tau_b_norm": float(torch.linalg.norm(b)),
            }
        )
    return pd.DataFrame(rows)


def grid_to_matrix(df: pd.DataFrame, metric: str, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index="beta", columns="alpha", values=metric).sort_index(ascending=True)
    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    assert z.shape == (grid_size, grid_size)
    return xs, ys, z


def plot_landscape(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path, grid_size: int) -> None:
    metrics = [
        ("task_a_loss", "Task A low-digits loss"),
        ("task_b_loss", "Task B high-digits loss"),
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
            if row["method"] in {"base", "expert_a_low_digits", "expert_b_high_digits", "linear_average", "validation_grid_best"}:
                ax.annotate(row["method"].replace("_", "\n"), (row["alpha"], row["beta"]), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("alpha: low-digit task vector")
        ax.set_ylabel("beta: high-digit task vector")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(contour, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_overlay(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path, grid_size: int) -> None:
    x, y, worst = grid_to_matrix(grid_df, "worst_loss", grid_size)
    _, _, a_loss = grid_to_matrix(grid_df, "task_a_loss", grid_size)
    _, _, b_loss = grid_to_matrix(grid_df, "task_b_loss", grid_size)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    bg_levels = np.linspace(np.nanmin(worst), np.nanpercentile(worst, 90), 20)
    bg = ax.contourf(x, y, worst, levels=bg_levels, cmap="magma_r")
    a_levels = np.quantile(a_loss, [0.1, 0.2, 0.35, 0.5])
    b_levels = np.quantile(b_loss, [0.1, 0.2, 0.35, 0.5])
    cs_a = ax.contour(x, y, a_loss, levels=np.unique(a_levels), colors="#2a9d8f", linewidths=1.3)
    cs_b = ax.contour(x, y, b_loss, levels=np.unique(b_levels), colors="#e76f51", linewidths=1.3, linestyles="--")
    ax.clabel(cs_a, inline=True, fontsize=7, fmt="A %.2f")
    ax.clabel(cs_b, inline=True, fontsize=7, fmt="B %.2f")
    colors = {
        "reference": "#f8f9fa",
        "merge": "#ffd166",
        "oracle": "#8ecae6",
    }
    for _, row in methods_df.iterrows():
        ax.scatter(
            row["alpha"],
            row["beta"],
            s=75,
            c=colors.get(str(row.get("kind", "merge")), "#ffd166"),
            edgecolors="black",
            linewidths=0.9,
            zorder=4,
        )
        ax.annotate(row["method"].replace("_", "\n"), (row["alpha"], row["beta"]), fontsize=7, zorder=5)
    ax.set_title("Per-task basin overlay on worst-task loss")
    ax.set_xlabel("alpha: low-digit task vector")
    ax.set_ylabel("beta: high-digit task vector")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(bg, ax=ax, label="worst-task loss")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_lambda(lambda_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(lambda_df["lambda"], lambda_df["task_a_acc"], label="task A acc", color="#2a9d8f")
    ax.plot(lambda_df["lambda"], lambda_df["task_b_acc"], label="task B acc", color="#e76f51")
    ax.plot(lambda_df["lambda"], lambda_df["avg_acc"], label="avg acc", color="#264653")
    ax.plot(lambda_df["lambda"], lambda_df["worst_acc"], label="worst acc", color="#6d597a")
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Task arithmetic path")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(lambda_df["lambda"], lambda_df["task_a_loss"], label="task A loss", color="#2a9d8f")
    ax.plot(lambda_df["lambda"], lambda_df["task_b_loss"], label="task B loss", color="#e76f51")
    ax.plot(lambda_df["lambda"], lambda_df["worst_loss"], label="worst loss", color="#6d597a")
    ax.set_xlabel("lambda")
    ax.set_ylabel("cross entropy")
    ax.set_title("Loss barrier along combined direction")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_methods(grid_df: pd.DataFrame, methods_df: pd.DataFrame, out: Path, grid_size: int) -> None:
    x, y, z = grid_to_matrix(grid_df, "worst_acc", grid_size)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax = axes[0]
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 18)
    bg = ax.contourf(x, y, z, levels=levels, cmap="cividis")
    short_labels = {
        "base": "base",
        "expert_a_low_digits": "A",
        "expert_b_high_digits": "B",
        "linear_average": "avg",
        "task_arithmetic_best_lambda": "task",
        "slerp_experts": "slerp",
        "ties_density_0.5": "TIES",
        "dare_drop_0.5": "DARE",
        "ties_dare_0.5": "T+D",
        "fisher_weighted": "Fisher",
        "regmean_linear": "RegMean",
        "layerwise_task_arithmetic": "layerwise",
        "validation_grid_best": "grid",
    }
    offsets = {
        "base": (6, -16),
        "expert_a_low_digits": (-42, 10),
        "expert_b_high_digits": (4, 10),
        "linear_average": (7, 8),
        "task_arithmetic_best_lambda": (-58, -18),
        "slerp_experts": (8, 20),
        "ties_density_0.5": (6, 8),
        "dare_drop_0.5": (-42, 20),
        "ties_dare_0.5": (6, 10),
        "fisher_weighted": (8, -16),
        "regmean_linear": (8, 10),
        "layerwise_task_arithmetic": (-62, 10),
        "validation_grid_best": (-52, 4),
    }
    for _, row in methods_df.iterrows():
        ax.scatter(row["alpha"], row["beta"], s=70, c="white", edgecolors="black", linewidths=0.9)
        label = short_labels.get(row["method"], row["method"])
        dx, dy = offsets.get(row["method"], (6, 6))
        ax.annotate(
            label,
            (row["alpha"], row["beta"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax.set_title("Merge methods projected into task-vector plane")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(bg, ax=ax, label="worst-task accuracy")

    ax = axes[1]
    plot_df = methods_df.sort_values("worst_acc", ascending=True)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.18, plot_df["task_a_acc"], height=0.34, color="#2a9d8f", alpha=0.78, label="task A")
    ax.barh(y_pos + 0.18, plot_df["task_b_acc"], height=0.34, color="#e76f51", alpha=0.62, label="task B")
    ax.scatter(plot_df["worst_acc"], y_pos, color="black", label="worst", zorder=4)
    ax.set_yticks(y_pos, labels=plot_df["method"])
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("accuracy")
    ax.set_title("Method performance")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_interference(interference_df: pd.DataFrame, out: Path) -> None:
    labels = interference_df["layer"].tolist()
    metrics = ["cosine", "sign_conflict", "weighted_conflict"]
    matrix = interference_df[metrics].T.to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.8), 6.6), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_yticks(range(len(metrics)), labels=metrics)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Layer-wise task-vector alignment and conflict")
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[1]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, interference_df["tau_a_norm"], width=0.4, label="task A delta norm", color="#2a9d8f")
    ax.bar(x + 0.2, interference_df["tau_b_norm"], width=0.4, label="task B delta norm", color="#e76f51")
    ax.set_xticks(x, labels=labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("L2 norm")
    ax.set_title("Where the experts changed the base")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    config: dict[str, int | float | str],
    method_df: pd.DataFrame,
    lambda_df: pd.DataFrame,
    interference_df: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> None:
    best_method = method_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    linear = method_df[method_df["method"] == "linear_average"].iloc[0]
    base = method_df[method_df["method"] == "base"].iloc[0]
    best_lambda = lambda_df.sort_values(["worst_acc", "avg_acc"], ascending=False).iloc[0]
    worst_barrier = interpolation_barrier(lambda_df["worst_loss"].tolist())
    overlap = float((grid_df["worst_acc"] >= 0.9).mean())
    high_conflict = interference_df.sort_values("weighted_conflict", ascending=False).head(3)

    lines = [
        "# Digits Merge-Landscape Study",
        "",
        "This run is a controlled image-classification surrogate for the proposal. A shared MLP base is trained on all sklearn digits, then two experts are fine-tuned from that exact base: task A uses digits 0-4 and task B uses digits 5-9. The experiment evaluates the raw task-vector plane `theta = theta0 + alpha * tau_A + beta * tau_B`.",
        "",
        "## Key Findings",
        "",
        f"- Best observed method/checkpoint by worst-task accuracy: `{best_method['method']}` with task A accuracy {best_method['task_a_acc']:.3f}, task B accuracy {best_method['task_b_acc']:.3f}, worst-task accuracy {best_method['worst_acc']:.3f}.",
        f"- Base checkpoint worst-task accuracy: {base['worst_acc']:.3f}; naive linear average worst-task accuracy: {linear['worst_acc']:.3f}.",
        f"- Best task-arithmetic lambda on the sweep: {best_lambda['lambda']:.3f}, worst-task accuracy {best_lambda['worst_acc']:.3f}, average loss {best_lambda['avg_loss']:.3f}.",
        f"- Fraction of sampled plane with worst-task accuracy >= 0.90: {overlap:.3f}. This is a direct proxy for basin-overlap area in this plane.",
        f"- Worst-task loss barrier along the combined task-vector path: {worst_barrier:.3f}.",
        "",
        "## Figures",
        "",
        "- `figures/merge_landscape.png`: task A, task B, average, and worst-task loss surfaces with method points.",
        "- `figures/per_task_basin_overlay.png`: task-specific contour overlays on the joint worst-task objective.",
        "- `figures/lambda_sweep.png`: accuracy and loss along `theta0 + lambda * (tau_A + tau_B)`.",
        "- `figures/method_overlay.png`: method points projected into the task-vector plane and their accuracies.",
        "- `figures/interference_heatmap.png`: per-layer cosine alignment, sign conflict, weighted conflict, and delta norms.",
        "",
        "## Most Conflicted Layers",
        "",
    ]
    for _, row in high_conflict.iterrows():
        lines.append(
            f"- `{row['layer']}`: cosine {row['cosine']:.3f}, sign conflict {row['sign_conflict']:.3f}, weighted conflict {row['weighted_conflict']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The meaningful object is the task-vector plane, not a random loss-landscape slice. Experts occupy the two coordinate axes by construction, while merge methods either stay in that plane or are projected back into it with a reported residual. When high-worst-accuracy regions appear between the two experts, merging is geometrically plausible; when the midpoint or task-arithmetic path crosses a high-loss ridge, the same plot explains why a merge fails.",
            "",
            "The interference atlas gives a parameter-space explanation for those ridges. Low cosine alignment and high magnitude-weighted sign conflict identify tensors where the two fine-tuning runs ask the same coordinates to move in opposite directions. TIES and DARE are included because they explicitly modify those coordinates before averaging.",
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
    parser = argparse.ArgumentParser(description="Run a controlled digits model-merging visualization study.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/digits_merge"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--expert-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-lr", type=float, default=3e-3)
    parser.add_argument("--expert-lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--grid-min", type=float, default=-0.25)
    parser.add_argument("--grid-max", type=float, default=1.25)
    parser.add_argument("--lambda-max", type=float, default=1.5)
    parser.add_argument("--lambda-steps", type=int, default=61)
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--regmean-batches", type=int, default=8)
    parser.add_argument("--regmean-ridge", type=float, default=1e-4)
    parser.add_argument("--layerwise-scale-grid", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    ckpt_dir = out_dir / "checkpoints"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data = prepare_digits(args.seed, args.batch_size)
    base_model = DigitMLP(hidden=args.hidden).to(device)
    train_model(
        base_model,
        data["full_train_loader"],
        args.base_epochs,
        args.base_lr,
        args.weight_decay,
        device,
        "train base",
    )
    reference_state = deepcopy(base_model.state_dict())
    base_vec, spec = vectorize_model(base_model)

    expert_a = DigitMLP(hidden=args.hidden).to(device)
    expert_a.load_state_dict(reference_state)
    train_model(
        expert_a,
        data["low_train_loader"],
        args.expert_epochs,
        args.expert_lr,
        args.weight_decay,
        device,
        "fine-tune low digits",
    )
    expert_a_vec, _ = vectorize_model(expert_a)

    expert_b = DigitMLP(hidden=args.hidden).to(device)
    expert_b.load_state_dict(reference_state)
    train_model(
        expert_b,
        data["high_train_loader"],
        args.expert_epochs,
        args.expert_lr,
        args.weight_decay,
        device,
        "fine-tune high digits",
    )
    expert_b_vec, _ = vectorize_model(expert_b)

    torch.save({"state_dict": reference_state, "vector": base_vec, "spec": spec}, ckpt_dir / "base.pt")
    torch.save({"state_dict": expert_a.state_dict(), "vector": expert_a_vec, "spec": spec}, ckpt_dir / "expert_low_digits.pt")
    torch.save({"state_dict": expert_b.state_dict(), "vector": expert_b_vec, "spec": spec}, ckpt_dir / "expert_high_digits.pt")

    tau_a = expert_a_vec - base_vec
    tau_b = expert_b_vec - base_vec
    eval_model = DigitMLP(hidden=args.hidden).to(device)
    val_loaders = {"low": data["low_val_loader"], "high": data["high_val_loader"]}
    test_loaders = {"low": data["low_test_loader"], "high": data["high_test_loader"]}

    grid_df = build_grid(
        eval_model,
        base_vec,
        tau_a,
        tau_b,
        spec,
        reference_state,
        val_loaders,
        device,
        args.grid_min,
        args.grid_max,
        args.grid_size,
    )
    grid_df.to_csv(out_dir / "grid_metrics.csv", index=False)

    lambda_df = lambda_sweep(
        eval_model,
        base_vec,
        tau_a,
        tau_b,
        spec,
        reference_state,
        val_loaders,
        device,
        args.lambda_max,
        args.lambda_steps,
    )
    lambda_df.to_csv(out_dir / "lambda_sweep.csv", index=False)

    fisher_a = compute_fisher(
        eval_model,
        expert_a_vec,
        spec,
        reference_state,
        data["low_train_loader"],
        device,
        args.fisher_batches,
    )
    fisher_b = compute_fisher(
        eval_model,
        expert_b_vec,
        spec,
        reference_state,
        data["high_train_loader"],
        device,
        args.fisher_batches,
    )

    scale_candidates = parse_scale_grid(args.layerwise_scale_grid)
    layerwise_vec, layerwise_df = layerwise_task_arithmetic_search(
        eval_model,
        base_vec,
        tau_a,
        tau_b,
        spec,
        reference_state,
        val_loaders,
        device,
        scale_candidates,
    )
    layerwise_df.to_csv(out_dir / "layerwise_task_arithmetic.csv", index=False)

    cov_a, cov_a_df = collect_linear_covariances(
        expert_a,
        data["low_train_loader"],
        device,
        args.regmean_batches,
    )
    cov_b, cov_b_df = collect_linear_covariances(
        expert_b,
        data["high_train_loader"],
        device,
        args.regmean_batches,
    )
    cov_a_df.insert(0, "expert", "low_digits")
    cov_b_df.insert(0, "expert", "high_digits")
    pd.concat([cov_a_df, cov_b_df], ignore_index=True).to_csv(out_dir / "regmean_covariances.csv", index=False)
    regmean_state, regmean_layers_df = regmean_merge_state_dict(
        [expert_a.state_dict(), expert_b.state_dict()],
        [cov_a, cov_b],
        reference_state,
        args.regmean_ridge,
    )
    regmean_layers_df.to_csv(out_dir / "regmean_linear_layers.csv", index=False)
    regmean_vec = vector_from_state_dict(regmean_state, spec)

    extra_methods = [
        (
            "layerwise_task_arithmetic",
            layerwise_vec,
            {
                "kind": "merge",
                "mean_scale": float(layerwise_df["scale"].mean()),
                "nonzero_layers": float((layerwise_df["scale"] != 0.0).sum()),
            },
        ),
        (
            "regmean_linear",
            regmean_vec,
            {
                "kind": "merge",
                "regmean_ridge": float(args.regmean_ridge),
                "regmean_layers": float(len(regmean_layers_df)),
            },
        ),
    ]

    methods_df = method_table(
        eval_model,
        base_vec,
        expert_a_vec,
        expert_b_vec,
        tau_a,
        tau_b,
        spec,
        reference_state,
        test_loaders,
        device,
        lambda_df,
        grid_df,
        fisher_a,
        fisher_b,
        args.seed,
        extra_methods,
    )
    methods_df.to_csv(out_dir / "method_metrics.csv", index=False)

    interference_df = interference_table(spec, tau_a, tau_b)
    interference_df.to_csv(out_dir / "interference.csv", index=False)

    plot_landscape(grid_df, methods_df, fig_dir / "merge_landscape.png", args.grid_size)
    plot_overlay(grid_df, methods_df, fig_dir / "per_task_basin_overlay.png", args.grid_size)
    plot_lambda(lambda_df, fig_dir / "lambda_sweep.png")
    plot_methods(grid_df, methods_df, fig_dir / "method_overlay.png", args.grid_size)
    plot_interference(interference_df, fig_dir / "interference_heatmap.png")

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["device"] = str(device)
    config["num_parameters"] = int(base_vec.numel())
    config["global_task_vector_cosine"] = cosine(tau_a, tau_b)
    config["tau_a_norm"] = float(torch.linalg.norm(tau_a))
    config["tau_b_norm"] = float(torch.linalg.norm(tau_b))
    (out_dir / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_report(out_dir, config, methods_df, lambda_df, interference_df, grid_df)
    print(f"Wrote study artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
