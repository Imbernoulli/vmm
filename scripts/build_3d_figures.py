#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    {
        "name": "digits",
        "title": "Digits merge plane: worst loss surface",
        "grid": "results/digits_merge/grid_metrics.csv",
        "methods": "results/digits_merge/method_metrics.csv",
        "z": "worst_loss",
        "z_label": "worst loss",
        "output": "digits_worst_loss_surface.png",
    },
    {
        "name": "cifar10",
        "title": "CIFAR-10 vehicle/animal: worst loss surface",
        "grid": "results/cifar_merge/grid_metrics.csv",
        "methods": "results/cifar_merge/method_metrics.csv",
        "z": "worst_loss",
        "z_label": "worst loss",
        "output": "cifar10_worst_loss_surface.png",
    },
    {
        "name": "pretrained_vit",
        "title": "Pretrained ViT transfer: worst loss surface",
        "grid": "results/pretrained_vit_transfer_merge/grid_metrics.csv",
        "methods": "results/pretrained_vit_transfer_merge/method_metrics.csv",
        "z": "worst_loss",
        "z_label": "worst loss",
        "output": "pretrained_vit_worst_loss_surface.png",
    },
    {
        "name": "qwen_multi",
        "title": "Qwen instruct/coder merge plane: worst NLL surface",
        "grid": "results/qwen_multi_expert_merge/grid_metrics.csv",
        "methods": "results/qwen_multi_expert_merge/method_metrics.csv",
        "z": "worst_nll",
        "z_label": "worst NLL",
        "output": "qwen_multi_worst_nll_surface.png",
    },
]

METHOD_LABELS = {
    "base": "base",
    "expert_a": "A",
    "expert_b": "B",
    "living_expert": "A",
    "object_expert": "B",
    "vehicle_expert": "A",
    "animal_expert": "B",
    "instruct_expert": "instruct",
    "coder_expert": "coder",
    "linear_average": "avg",
    "validation_grid_best": "best",
}

METHOD_COLORS = {
    "base": "#111827",
    "expert_a": "#2563a8",
    "expert_b": "#e76f51",
    "living_expert": "#2563a8",
    "object_expert": "#e76f51",
    "vehicle_expert": "#2563a8",
    "animal_expert": "#e76f51",
    "instruct_expert": "#2563a8",
    "coder_expert": "#e76f51",
    "linear_average": "#2a9d8f",
    "validation_grid_best": "#8b5cf6",
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def nearest_surface_value(grid: pd.DataFrame, alpha: float, beta: float, z_column: str) -> float:
    distances = (grid["alpha"].astype(float) - alpha) ** 2 + (grid["beta"].astype(float) - beta) ** 2
    return float(grid.loc[distances.idxmin(), z_column])


def build_surface(experiment: dict[str, str], output_dir: Path) -> Path:
    grid = pd.read_csv(repo_path(experiment["grid"]))
    methods = pd.read_csv(repo_path(experiment["methods"]))
    z_column = experiment["z"]

    alphas = sorted(grid["alpha"].astype(float).unique())
    betas = sorted(grid["beta"].astype(float).unique())
    surface = (
        grid.assign(alpha=grid["alpha"].astype(float), beta=grid["beta"].astype(float))
        .pivot(index="beta", columns="alpha", values=z_column)
        .reindex(index=betas, columns=alphas)
    )
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    z_grid = surface.to_numpy(dtype=float)

    fig = plt.figure(figsize=(8.5, 6.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        alpha_grid,
        beta_grid,
        z_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.88,
    )
    z_min = float(np.nanmin(z_grid))
    ax.contour(alpha_grid, beta_grid, z_grid, zdir="z", offset=z_min, cmap="viridis", linewidths=0.7)

    for _, row in methods.iterrows():
        method = str(row["method"])
        if method not in METHOD_LABELS:
            continue
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        z_value = float(row[z_column]) if z_column in row and pd.notna(row[z_column]) else nearest_surface_value(grid, alpha, beta, z_column)
        ax.scatter(
            [alpha],
            [beta],
            [z_value],
            s=38,
            c=METHOD_COLORS.get(method, "#111827"),
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )
        ax.text(alpha, beta, z_value, f" {METHOD_LABELS[method]}", fontsize=8, color="#111827")

    ax.set_title(experiment["title"], pad=16)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel(experiment["z_label"])
    ax.view_init(elev=28, azim=-128)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / experiment["output"]
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3D surface figures for checked-in merge-plane grids.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures_3d"))
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    for experiment in EXPERIMENTS:
        output = build_surface(experiment, output_dir)
        print(f"Wrote {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
