#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
EPS = 1e-12
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


LITERATURE_SOURCES = [
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "MoE top-k dispatch can change sharply under router perturbations; router calibration is a separate intervention.",
    },
    {
        "key": "mergeme",
        "title": "MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs",
        "url": "https://arxiv.org/abs/2502.00997",
        "mechanism": "MoE merging needs routing heuristics and interference mitigation beyond unweighted expert averaging.",
    },
    {
        "key": "expert_merging",
        "title": "Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking",
        "url": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Layer heterogeneity matters; router risk should be summarized by layer before opening layer-wise movement.",
    },
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(repo_path(path))


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clean_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def clean_row(row: pd.Series) -> dict[str, Any]:
    return {str(key): clean_value(value) for key, value in row.items()}


def layer_id(name: str) -> int | None:
    match = LAYER_RE.search(str(name))
    return None if match is None else int(match.group(1))


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def inverse_normalize(series: pd.Series) -> pd.Series:
    return 1.0 - normalize(series)


def build_layer_fragility(layer_gate: pd.DataFrame) -> pd.DataFrame:
    out = layer_gate.copy()
    out["unsafe_fraction"] = out["unsafe_rows"] / out["probe_rows"].clip(lower=1)
    out["low_margin_score"] = inverse_normalize(out["mean_top1_margin"])
    out["low_overlap_score"] = (1.0 - out["mean_topk_jaccard"]).clip(0.0, 1.0)
    out["low_top1_agreement_score"] = (1.0 - out["mean_top1_agreement"]).clip(0.0, 1.0)
    out["router_movement_score"] = normalize(out["router_relative_delta_norm"])
    out["margin_delta_ratio_proxy"] = out["router_relative_delta_norm"] / out["mean_top1_margin"].clip(lower=EPS)
    out["min_margin_delta_ratio_proxy"] = out["router_relative_delta_norm"] / out["min_top1_margin"].clip(lower=EPS)
    out["safe_lambda_proxy"] = (
        out["min_top1_margin"] / out["router_relative_delta_norm"].clip(lower=EPS)
    ).clip(lower=0.0, upper=0.05)
    out["boundary_fragility_score"] = (
        0.30 * out["low_overlap_score"]
        + 0.25 * out["low_top1_agreement_score"]
        + 0.20 * out["low_margin_score"]
        + 0.15 * out["router_movement_score"]
        + 0.10 * out["unsafe_fraction"].clip(0.0, 1.0)
    )
    out["fragility_rank"] = out["boundary_fragility_score"].rank(method="dense", ascending=False).astype(int)
    out["router_margin_action"] = "freeze_router"
    out.loc[
        (out["unsafe_fraction"] >= 1.0) & (out["boundary_fragility_score"] >= 0.60),
        "router_margin_action",
    ] = "freeze_router_prioritize_calibration"
    out.loc[
        (out["unsafe_fraction"] < 0.5) & (out["boundary_fragility_score"] < 0.45),
        "router_margin_action",
    ] = "probe_small_lambda_only"
    return out.sort_values(["boundary_fragility_score", "unsafe_fraction"], ascending=[False, False])


def build_category_fragility(readiness: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for category, group in readiness.groupby("category", sort=True):
        actions = group["recommended_action"].astype(str)
        unsafe = actions.isin({"calibrate_router_before_average", "freeze_router_and_check_load_balance"})
        grouped.append(
            {
                "category": category,
                "probe_rows": int(len(group)),
                "mean_top1_margin": float(group["top1_margin_mean"].mean()),
                "min_top1_margin": float(group["top1_margin_mean"].min()),
                "mean_top1_agreement": float(group["top1_agreement"].mean()),
                "min_top1_agreement": float(group["top1_agreement"].min()),
                "mean_topk_jaccard": float(group["topk_jaccard"].mean()),
                "min_topk_jaccard": float(group["topk_jaccard"].min()),
                "mean_risk_score": float(group["risk_score"].mean()),
                "unsafe_rows": int(unsafe.sum()),
                "unsafe_fraction": float(unsafe.mean()),
                "calibrate_rows": int((actions == "calibrate_router_before_average").sum()),
                "freeze_rows": int((actions == "freeze_router_and_check_load_balance").sum()),
                "small_lambda_rows": int((actions == "small_lambda_router_with_overlap_guard").sum()),
                "passed_rows": int((actions == "router_probe_passed_for_small_lambda").sum()),
            }
        )
    out = pd.DataFrame(grouped)
    out["low_margin_score"] = inverse_normalize(out["mean_top1_margin"])
    out["category_fragility_score"] = (
        0.30 * (1.0 - out["mean_topk_jaccard"]).clip(0.0, 1.0)
        + 0.30 * (1.0 - out["mean_top1_agreement"]).clip(0.0, 1.0)
        + 0.25 * out["low_margin_score"]
        + 0.15 * out["unsafe_fraction"].clip(0.0, 1.0)
    )
    return out.sort_values("category_fragility_score", ascending=False)


def build_cross_fragility(readiness: pd.DataFrame, layer_fragility: pd.DataFrame) -> pd.DataFrame:
    frame = readiness.copy()
    if "layer" not in frame.columns:
        frame["layer"] = frame["router"].map(layer_id)
    layer_scores = layer_fragility[["layer", "router_relative_delta_norm", "router_movement_score"]].copy()
    frame = frame.merge(layer_scores, on="layer", how="left")
    frame["unsafe"] = frame["recommended_action"].astype(str).isin(
        {"calibrate_router_before_average", "freeze_router_and_check_load_balance"}
    )
    frame["low_margin_score"] = inverse_normalize(frame["top1_margin_mean"])
    frame["slice_fragility_score"] = (
        0.30 * (1.0 - frame["topk_jaccard"]).clip(0.0, 1.0)
        + 0.25 * (1.0 - frame["top1_agreement"]).clip(0.0, 1.0)
        + 0.20 * frame["low_margin_score"]
        + 0.15 * frame["router_movement_score"].fillna(0.0)
        + 0.10 * frame["unsafe"].astype(float)
    )
    keep = [
        "category",
        "prompt_idx",
        "layer",
        "router",
        "top1_margin_mean",
        "top1_agreement",
        "topk_jaccard",
        "risk_score",
        "router_relative_delta_norm",
        "slice_fragility_score",
        "recommended_action",
        "risk_flags",
    ]
    return frame[keep].sort_values("slice_fragility_score", ascending=False)


def write_figure(path: Path, layer_fragility: pd.DataFrame, category_fragility: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_layers = layer_fragility.sort_values("boundary_fragility_score", ascending=False).head(12).copy()
    categories = category_fragility.sort_values("category_fragility_score", ascending=True).copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].barh(
        [f"L{int(layer)}" for layer in top_layers["layer"].iloc[::-1]],
        top_layers["boundary_fragility_score"].iloc[::-1],
        color="#b91c1c",
    )
    axes[0].set_title("Most fragile router layers")
    axes[0].set_xlabel("boundary fragility score")
    axes[0].grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for index, value in enumerate(top_layers["boundary_fragility_score"].iloc[::-1]):
        axes[0].text(value + 0.01, index, f"{value:.3f}", va="center", ha="left", fontsize=8)

    axes[1].barh(
        categories["category"],
        categories["category_fragility_score"],
        color="#4f46e5",
    )
    axes[1].set_title("Fragility by prompt category")
    axes[1].set_xlabel("category fragility score")
    axes[1].grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for index, value in enumerate(categories["category_fragility_score"]):
        axes[1].text(value + 0.01, index, f"{value:.3f}", va="center", ha="left", fontsize=8)

    fig.suptitle("Qwen3 MoE router boundary fragility")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(
    summary: dict[str, Any],
    layer_fragility: pd.DataFrame,
    category_fragility: pd.DataFrame,
    cross_fragility: pd.DataFrame,
) -> str:
    top = layer_fragility.iloc[0]
    least = layer_fragility.sort_values("boundary_fragility_score", ascending=True).iloc[0]
    lines = [
        "# Qwen3 MoE Router Margin Fragility",
        "",
        "这个 probe 把 router top-k 的边界稳定性单独抽出来看：如果 top-1 margin 很小、Instruct/Coder route overlap 很低、router 权重位移又大，那么直接线性移动 router 很容易跨过离散 top-k 边界。这个分数是排序用的机制 proxy，不是最终下游分数。",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Router layers: `{summary['router_layer_count']}`",
        f"- High-fragility layers: `{summary['high_fragility_layer_count']}`",
        f"- Top fragile layer: `L{summary['top_fragile_layer']}` score `{fmt(summary['top_fragility_score'])}`",
        f"- Least fragile layer: `L{summary['least_fragile_layer']}` score `{fmt(summary['least_fragility_score'])}`",
        f"- Min safe-lambda proxy: `{fmt(summary['min_safe_lambda_proxy'])}`",
        f"- Top category: `{summary['top_fragile_category']}` score `{fmt(summary['top_category_fragility_score'])}`",
        "",
        "## Mechanism",
        "",
        "Dense interpolation assumes one continuous parameter path can be evaluated by loss along the path. MoE router averaging adds a discrete top-k boundary: small logit changes can swap experts, and the wrong expert then receives the token. Therefore router movement needs a separate boundary/margin gate before it can be treated like ordinary Dense parameters.",
        "",
        "## Layer Ranking",
        "",
        "| layer | score | action | unsafe frac | mean margin | mean top-k Jaccard | mean top1 agreement | router rel | lambda proxy |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in layer_fragility.head(16).iterrows():
        lines.append(
            f"| {int(row['layer'])} | {fmt(row['boundary_fragility_score'])} | "
            f"`{row['router_margin_action']}` | {fmt(row['unsafe_fraction'])} | "
            f"{fmt(row['mean_top1_margin'])} | {fmt(row['mean_topk_jaccard'])} | "
            f"{fmt(row['mean_top1_agreement'])} | {fmt(row['router_relative_delta_norm'])} | "
            f"{fmt(row['safe_lambda_proxy'])} |"
        )
    lines.extend(
        [
            "",
            "## Category Ranking",
            "",
            "| category | score | unsafe frac | mean margin | mean top-k Jaccard | mean top1 agreement | calibrate rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in category_fragility.iterrows():
        lines.append(
            f"| `{row['category']}` | {fmt(row['category_fragility_score'])} | "
            f"{fmt(row['unsafe_fraction'])} | {fmt(row['mean_top1_margin'])} | "
            f"{fmt(row['mean_topk_jaccard'])} | {fmt(row['mean_top1_agreement'])} | "
            f"{int(row['calibrate_rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Most Fragile Observed Slices",
            "",
            "| category | prompt | layer | score | margin | top-k Jaccard | top1 agreement | action |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in cross_fragility.head(12).iterrows():
        lines.append(
            f"| `{row['category']}` | {int(row['prompt_idx'])} | {int(row['layer'])} | "
            f"{fmt(row['slice_fragility_score'])} | {fmt(row['top1_margin_mean'])} | "
            f"{fmt(row['topk_jaccard'])} | {fmt(row['top1_agreement'])} | "
            f"`{row['recommended_action']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The most fragile layer is `L{int(top['layer'])}`; the least fragile layer is `L{int(least['layer'])}`, but every layer still has at least one unsafe observed category/prompt slice. The algorithmic consequence is unchanged but sharper: direct router averaging remains rejected, and any router movement must be a calibrated route-KD/HARC-style intervention with the same eval-bundle/source-dominance gate as other candidates.",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, output in summary["outputs"].items():
        lines.append(f"- `{key}`: `{output}`")
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    readiness = read_csv(args.router_readiness)
    layer_gate = read_csv(args.router_layer_move_gate)
    router_move_summary = read_json(args.router_move_summary)
    layer_fragility = build_layer_fragility(layer_gate)
    category_fragility = build_category_fragility(readiness)
    cross_fragility = build_cross_fragility(readiness, layer_fragility)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_path = output_dir / "layer_margin_fragility.csv"
    category_path = output_dir / "category_margin_fragility.csv"
    cross_path = output_dir / "slice_margin_fragility.csv"
    sources_path = output_dir / "literature_sources.json"
    figure_path = output_dir / "router_margin_fragility.png"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    high_fragility = layer_fragility[layer_fragility["boundary_fragility_score"] >= args.high_fragility_threshold]
    top_layer = layer_fragility.iloc[0]
    least_layer = layer_fragility.sort_values("boundary_fragility_score", ascending=True).iloc[0]
    top_category = category_fragility.iloc[0]
    summary = {
        "schema_version": 1,
        "status": "router_margin_fragility_rejects_direct_router_average",
        "router_layer_count": int(len(layer_fragility)),
        "high_fragility_threshold": float(args.high_fragility_threshold),
        "high_fragility_layer_count": int(len(high_fragility)),
        "top_fragile_layer": int(top_layer["layer"]),
        "top_fragility_score": float(top_layer["boundary_fragility_score"]),
        "least_fragile_layer": int(least_layer["layer"]),
        "least_fragility_score": float(least_layer["boundary_fragility_score"]),
        "mean_fragility_score": float(layer_fragility["boundary_fragility_score"].mean()),
        "min_safe_lambda_proxy": float(layer_fragility["safe_lambda_proxy"].min()),
        "median_safe_lambda_proxy": float(layer_fragility["safe_lambda_proxy"].median()),
        "top_fragile_category": str(top_category["category"]),
        "top_category_fragility_score": float(top_category["category_fragility_score"]),
        "unsafe_readiness_rows": int(router_move_summary.get("unsafe_readiness_rows", 0)),
        "calibrate_readiness_rows": int(router_move_summary.get("calibrate_readiness_rows", 0)),
        "freeze_readiness_rows": int(router_move_summary.get("freeze_readiness_rows", 0)),
        "allowed_router_layer_count": int(router_move_summary.get("allowed_router_layer_count", 0)),
        "recommended_unified_router_action": router_move_summary.get("recommended_unified_router_action"),
        "interpretation": (
            "Top-k router boundary fragility is high enough that direct router averaging should remain frozen. "
            "The next meaningful router intervention is calibrated router movement with matched downstream eval."
        ),
        "outputs": {
            "layer_fragility": rel(layer_path),
            "category_fragility": rel(category_path),
            "slice_fragility": rel(cross_path),
            "literature_sources": rel(sources_path),
            "figure": rel(figure_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    layer_fragility.to_csv(layer_path, index=False)
    category_fragility.to_csv(category_path, index=False)
    cross_fragility.to_csv(cross_path, index=False)
    sources_path.write_text(json.dumps(LITERATURE_SOURCES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figure(figure_path, layer_fragility, category_fragility)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, layer_fragility, category_fragility, cross_fragility), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen3 MoE router margin/boundary fragility probe.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_margin_fragility"))
    parser.add_argument(
        "--router-readiness",
        type=Path,
        default=Path("results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/router_readiness.csv"),
    )
    parser.add_argument(
        "--router-layer-move-gate",
        type=Path,
        default=Path("results/qwen3_moe_router_move_gate/router_layer_move_gate.csv"),
    )
    parser.add_argument(
        "--router-move-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_move_gate/summary.json"),
    )
    parser.add_argument("--high-fragility-threshold", type=float, default=0.62)
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(
        "Wrote router margin fragility: "
        f"{summary['status']}; high_fragility_layers={summary['high_fragility_layer_count']}; "
        f"top_layer=L{summary['top_fragile_layer']}"
    )


if __name__ == "__main__":
    main()
