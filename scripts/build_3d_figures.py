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
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]

METHOD_STYLES = {
    "base": {"label": "Base", "color": "#111827", "marker": "X", "offset": (9, -14)},
    "expert_a_low_digits": {"label": "Expert A", "color": "#2563eb", "marker": "^", "offset": (8, 10)},
    "expert_b_high_digits": {"label": "Expert B", "color": "#f97316", "marker": "^", "offset": (8, 10)},
    "expert_vehicle": {"label": "Vehicle expert", "color": "#2563eb", "marker": "^", "offset": (8, 10)},
    "expert_animal": {"label": "Animal expert", "color": "#f97316", "marker": "^", "offset": (8, 10)},
    "expert_living": {"label": "Living expert", "color": "#2563eb", "marker": "^", "offset": (8, 10)},
    "expert_object": {"label": "Object expert", "color": "#f97316", "marker": "^", "offset": (8, 10)},
    "instruct_expert": {"label": "Instruct expert", "color": "#2563eb", "marker": "^", "offset": (8, 10)},
    "coder_expert": {"label": "Coder expert", "color": "#f97316", "marker": "^", "offset": (8, 10)},
    "linear_average": {"label": "Linear average", "color": "#16a34a", "marker": "o", "offset": (9, 8)},
    "regmean_linear": {"label": "RegMean", "color": "#dc2626", "marker": "D", "offset": (9, 10)},
    "validation_grid_best": {"label": "Best grid", "color": "#7c3aed", "marker": "*", "offset": (10, -16)},
    "validation_grid_best_worst": {"label": "Best worst-NLL", "color": "#7c3aed", "marker": "*", "offset": (10, -16)},
}

EXPERIMENTS = [
    {
        "title": "Digits merge plane",
        "subtitle": "worst loss, lower is better",
        "grid": "results/digits_merge/grid_metrics.csv",
        "methods": "results/digits_merge/method_metrics.csv",
        "z": "worst_loss",
        "z_label": "worst loss",
        "output": "digits_worst_loss_surface.png",
        "methods_to_show": [
            "base",
            "expert_a_low_digits",
            "expert_b_high_digits",
            "linear_average",
            "regmean_linear",
            "validation_grid_best",
        ],
    },
    {
        "title": "CIFAR-10 vehicle/animal",
        "subtitle": "worst loss, lower is better",
        "grid": "results/cifar_merge/grid_metrics.csv",
        "methods": "results/cifar_merge/method_metrics.csv",
        "z": "worst_loss",
        "z_label": "worst loss",
        "output": "cifar10_worst_loss_surface.png",
        "methods_to_show": [
            "base",
            "expert_vehicle",
            "expert_animal",
            "linear_average",
            "validation_grid_best",
        ],
    },
    {
        "title": "Pretrained ViT transfer",
        "subtitle": "worst loss, lower is better",
        "grid": "results/pretrained_vit_transfer_merge/grid_metrics.csv",
        "methods": "results/pretrained_vit_transfer_merge/method_metrics.csv",
        "z": "worst_loss",
        "z_label": "worst loss",
        "output": "pretrained_vit_worst_loss_surface.png",
        "methods_to_show": [
            "base",
            "expert_living",
            "expert_object",
            "linear_average",
            "validation_grid_best",
        ],
    },
    {
        "title": "Qwen instruct/coder",
        "subtitle": "worst NLL, lower is better",
        "grid": "results/qwen_multi_expert_merge/grid_metrics.csv",
        "methods": "results/qwen_multi_expert_merge/method_metrics.csv",
        "z": "worst_nll",
        "z_label": "worst NLL",
        "output": "qwen_multi_worst_nll_surface.png",
        "methods_to_show": [
            "base",
            "instruct_expert",
            "coder_expert",
            "linear_average",
        ],
        "label_overrides": {"instruct_expert": "Instruct / best"},
    },
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def surface_grid(grid: pd.DataFrame, z_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = sorted(grid["alpha"].astype(float).unique())
    betas = sorted(grid["beta"].astype(float).unique())
    surface = (
        grid.assign(alpha=grid["alpha"].astype(float), beta=grid["beta"].astype(float))
        .pivot(index="beta", columns="alpha", values=z_column)
        .reindex(index=betas, columns=alphas)
    )
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    return alpha_grid, beta_grid, surface.to_numpy(dtype=float)


def nearest_surface_value(grid: pd.DataFrame, alpha: float, beta: float, z_column: str) -> float:
    distances = (grid["alpha"].astype(float) - alpha) ** 2 + (grid["beta"].astype(float) - beta) ** 2
    return float(grid.loc[distances.idxmin(), z_column])


def method_points(experiment: dict[str, object], grid: pd.DataFrame, methods: pd.DataFrame) -> list[dict[str, object]]:
    z_column = str(experiment["z"])
    points: list[dict[str, object]] = []
    for method in experiment["methods_to_show"]:
        rows = methods[methods["method"] == method]
        if rows.empty:
            continue
        row = rows.iloc[0]
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        z_value = (
            float(row[z_column])
            if z_column in row.index and pd.notna(row[z_column])
            else nearest_surface_value(grid, alpha, beta, z_column)
        )
        residual = float(row["plane_residual"]) if "plane_residual" in row.index and pd.notna(row["plane_residual"]) else 0.0
        style = METHOD_STYLES[str(method)]
        label = experiment.get("label_overrides", {}).get(str(method), style["label"])
        offplane = residual > 1e-3
        if offplane:
            label = f"{label} projected"
        points.append({"method": method, "alpha": alpha, "beta": beta, "z": z_value, **style})
        points[-1]["label"] = label
        points[-1]["plane_residual"] = residual
        points[-1]["offplane"] = offplane
    return points


def annotation_props(alpha: float, beta: float, alpha_grid: np.ndarray, beta_grid: np.ndarray) -> tuple[tuple[int, int], str, str]:
    min_alpha = float(np.nanmin(alpha_grid))
    max_alpha = float(np.nanmax(alpha_grid))
    min_beta = float(np.nanmin(beta_grid))
    max_beta = float(np.nanmax(beta_grid))
    alpha_span = max(max_alpha - min_alpha, 1e-9)
    beta_span = max(max_beta - min_beta, 1e-9)

    near_right = alpha > max_alpha - 0.16 * alpha_span
    near_left = alpha < min_alpha + 0.16 * alpha_span
    near_top = beta > max_beta - 0.16 * beta_span
    near_bottom = beta < min_beta + 0.16 * beta_span

    x_offset = -10 if near_right else 10
    y_offset = -12 if near_top else 10
    if near_left and near_top:
        x_offset, y_offset = 10, -14
    elif near_right and near_bottom:
        x_offset, y_offset = -10, 10
    elif near_right and near_top:
        x_offset, y_offset = -10, -14
    elif near_left and near_bottom:
        x_offset, y_offset = 10, 10

    ha = "right" if near_right else "left"
    va = "top" if near_top else "bottom"
    return (x_offset, y_offset), ha, va


def draw_2d_panel(ax: plt.Axes, alpha_grid: np.ndarray, beta_grid: np.ndarray, z_grid: np.ndarray, points: list[dict[str, object]], experiment: dict[str, object]) -> None:
    levels = np.linspace(float(np.nanmin(z_grid)), float(np.nanmax(z_grid)), 18)
    heat = ax.contourf(alpha_grid, beta_grid, z_grid, levels=levels, cmap="viridis", alpha=0.95)
    contour = ax.contour(alpha_grid, beta_grid, z_grid, levels=levels[::3], colors="white", linewidths=0.8, alpha=0.75)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2g")

    for point in points:
        offplane = bool(point.get("offplane", False))
        ax.scatter(
            point["alpha"],
            point["beta"],
            s=170 if point["marker"] == "*" else 92,
            marker=point["marker"],
            facecolor="white" if offplane else point["color"],
            edgecolor=point["color"] if offplane else "white",
            linewidth=2.4 if offplane else 1.8,
            zorder=6,
        )
        offset, ha, va = annotation_props(float(point["alpha"]), float(point["beta"]), alpha_grid, beta_grid)
        ax.annotate(
            str(point["label"]),
            xy=(point["alpha"], point["beta"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="#111827",
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#d1d5db", "alpha": 0.92},
            zorder=7,
        )

    ax.set_title("2D contour map of the same grid", fontsize=13, fontweight="bold")
    ax.set_xlabel("alpha = amount of expert A delta")
    ax.set_ylabel("beta = amount of expert B delta")
    ax.set_aspect("equal", adjustable="box")
    alpha_min = float(np.nanmin(alpha_grid))
    alpha_max = float(np.nanmax(alpha_grid))
    beta_min = float(np.nanmin(beta_grid))
    beta_max = float(np.nanmax(beta_grid))
    ax.set_xlim(alpha_min - 0.035 * (alpha_max - alpha_min), alpha_max + 0.035 * (alpha_max - alpha_min))
    ax.set_ylim(beta_min - 0.035 * (beta_max - beta_min), beta_max + 0.035 * (beta_max - beta_min))
    ax.grid(color="white", alpha=0.25, linewidth=0.8)
    cbar = plt.colorbar(heat, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(str(experiment["z_label"]))


def draw_3d_panel(ax: plt.Axes, alpha_grid: np.ndarray, beta_grid: np.ndarray, z_grid: np.ndarray, points: list[dict[str, object]], experiment: dict[str, object]) -> None:
    z_min = float(np.nanmin(z_grid))
    z_max = float(np.nanmax(z_grid))
    z_pad = max((z_max - z_min) * 0.035, 1e-3)
    ax.plot_surface(
        alpha_grid,
        beta_grid,
        z_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.78,
        shade=True,
    )

    for point in points:
        if bool(point.get("offplane", False)):
            continue
        z = float(point["z"])
        ax.scatter(
            [point["alpha"]],
            [point["beta"]],
            [z + z_pad * 1.25],
            s=145 if point["marker"] == "*" else 96,
            marker=point["marker"],
            c=point["color"],
            edgecolors="#ffffff",
            linewidths=1.8,
            depthshade=False,
        )

    ax.set_title("3D surface of loss over alpha/beta", fontsize=13, fontweight="bold")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("")
    ax.set_zlim(z_min - z_pad, z_max + z_pad * 4)
    ax.view_init(elev=25, azim=-135)
    ax.grid(True, linewidth=0.4, alpha=0.45)


def build_surface(experiment: dict[str, object], output_dir: Path) -> Path:
    grid = pd.read_csv(repo_path(str(experiment["grid"])))
    methods = pd.read_csv(repo_path(str(experiment["methods"])))
    alpha_grid, beta_grid, z_grid = surface_grid(grid, str(experiment["z"]))
    points = method_points(experiment, grid, methods)

    fig = plt.figure(figsize=(15.5, 7.4), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.12)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")

    fig.suptitle(str(experiment["title"]), fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.936, str(experiment["subtitle"]), ha="center", va="center", fontsize=11, color="#4b5563")

    draw_2d_panel(ax2d, alpha_grid, beta_grid, z_grid, points, experiment)
    draw_3d_panel(ax3d, alpha_grid, beta_grid, z_grid, points, experiment)

    handles = [
        Line2D(
            [0],
            [0],
            marker=point["marker"],
            color="none",
            markerfacecolor="white" if bool(point.get("offplane", False)) else point["color"],
            markeredgecolor=point["color"] if bool(point.get("offplane", False)) else "white",
            markeredgewidth=1.8 if bool(point.get("offplane", False)) else 1.2,
            markersize=10 if point["marker"] == "*" else 8,
            label=str(point["label"]),
        )
        for point in points
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.88, bottom=0.12, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / str(experiment["output"])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build presentation 2D+3D surface figures for checked-in merge-plane grids.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures_3d"))
    args = parser.parse_args()

    output_dir = repo_path(args.output_dir)
    for experiment in EXPERIMENTS:
        output = build_surface(experiment, output_dir)
        print(f"Wrote {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
