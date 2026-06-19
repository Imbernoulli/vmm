#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS = [
    ("digits", "results/digits_merge"),
    ("cifar10", "results/cifar_merge"),
    ("cifar100_vit", "results/cifar100_vit_merge"),
    ("pretrained_vit", "results/pretrained_vit_transfer_merge"),
    ("qwen_instruct_coder", "results/qwen_multi_expert_merge"),
]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    return str(repo_path(path).relative_to(REPO_ROOT))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def choose_metric(grid: pd.DataFrame) -> dict[str, str]:
    if "worst_nll" in grid.columns:
        return {"objective": "worst_nll", "average": "avg_nll", "orientation": "lower", "family": "nll"}
    if "worst_acc" in grid.columns:
        return {"objective": "worst_acc", "average": "avg_acc", "orientation": "higher", "family": "accuracy"}
    if "worst_loss" in grid.columns:
        return {"objective": "worst_loss", "average": "avg_loss", "orientation": "lower", "family": "loss"}
    raise ValueError("Could not infer objective column from grid metrics")


def sort_by_metric(df: pd.DataFrame, metric: str, orientation: str) -> pd.DataFrame:
    return df.sort_values(metric, ascending=(orientation == "lower"))


def pick_point(df: pd.DataFrame, alpha: float, beta: float, max_distance: float = 1e-3) -> pd.Series | None:
    if "alpha" not in df.columns or "beta" not in df.columns:
        return None
    rows = df[(df["alpha"].sub(alpha).abs() < 1e-6) & (df["beta"].sub(beta).abs() < 1e-6)]
    if not rows.empty:
        return rows.iloc[0]
    distance = df["alpha"].sub(alpha).abs() + df["beta"].sub(beta).abs()
    nearest_idx = distance.idxmin()
    if float(distance.loc[nearest_idx]) > max_distance:
        return None
    return df.loc[nearest_idx]


def pick_method(methods: pd.DataFrame | None, method: str) -> pd.Series | None:
    if methods is None or "method" not in methods.columns:
        return None
    rows = methods[methods["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]


def metric_gain(new: float | None, old: float | None, orientation: str) -> float | None:
    if not finite(new) or not finite(old):
        return None
    if orientation == "lower":
        return float(old - new)
    return float(new - old)


def metric_gap(candidate: float | None, best: float | None, orientation: str) -> float | None:
    if not finite(candidate) or not finite(best):
        return None
    if orientation == "lower":
        return float(candidate - best)
    return float(best - candidate)


def loss_barrier(grid: pd.DataFrame) -> float | None:
    loss_col = None
    for candidate in ("worst_nll", "worst_loss"):
        if candidate in grid.columns:
            loss_col = candidate
            break
    if loss_col is None:
        return None
    midpoint = pick_point(grid, 0.5, 0.5, max_distance=0.15)
    endpoint_a = pick_point(grid, 1.0, 0.0, max_distance=0.15)
    endpoint_b = pick_point(grid, 0.0, 1.0, max_distance=0.15)
    if midpoint is None or endpoint_a is None or endpoint_b is None:
        return None
    midpoint_value = clean_float(midpoint[loss_col])
    endpoint_a_value = clean_float(endpoint_a[loss_col])
    endpoint_b_value = clean_float(endpoint_b[loss_col])
    if not all(finite(value) for value in (midpoint_value, endpoint_a_value, endpoint_b_value)):
        return None
    return float(midpoint_value - max(endpoint_a_value, endpoint_b_value))


def summarize_conflict(exp_dir: Path) -> dict[str, float | str | None]:
    candidates = [
        exp_dir / "pairwise_conflict.csv",
        exp_dir / "interference.csv",
    ]
    for path in candidates:
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            continue
        if "weighted_conflict" not in df.columns:
            continue
        weight_col = "numel" if "numel" in df.columns else None
        if weight_col is not None and float(df[weight_col].sum()) > 0:
            weighted_conflict = float((df["weighted_conflict"] * df[weight_col]).sum() / df[weight_col].sum())
            cosine = float((df["cosine"] * df[weight_col]).sum() / df[weight_col].sum()) if "cosine" in df.columns else None
        else:
            weighted_conflict = float(df["weighted_conflict"].mean())
            cosine = float(df["cosine"].mean()) if "cosine" in df.columns else None
        top = df.sort_values("weighted_conflict", ascending=False).iloc[0]
        top_group = None
        for key in ("group", "layer", "left"):
            if key in top.index:
                top_group = str(top[key])
                break
        if "left" in top.index and "right" in top.index:
            top_group = f"{top['left']} vs {top['right']}"
        return {
            "conflict_file": rel(path),
            "cosine": cosine,
            "weighted_conflict": weighted_conflict,
            "max_weighted_conflict": float(top["weighted_conflict"]),
            "top_conflict_group": top_group,
        }
    return {
        "conflict_file": None,
        "cosine": None,
        "weighted_conflict": None,
        "max_weighted_conflict": None,
        "top_conflict_group": None,
    }


def recommendation(
    *,
    family: str,
    orientation: str,
    linear_gap_to_best: float | None,
    linear_gain_vs_base: float | None,
    midpoint_barrier: float | None,
    weighted_conflict: float | None,
) -> tuple[str, str]:
    tolerance = 0.02 if family == "accuracy" else 0.25
    high_gap = finite(linear_gap_to_best) and linear_gap_to_best > (2.0 * tolerance)
    modest_gap = finite(linear_gap_to_best) and linear_gap_to_best > tolerance
    worse_than_base = finite(linear_gain_vs_base) and linear_gain_vs_base < -tolerance
    high_barrier = finite(midpoint_barrier) and midpoint_barrier > tolerance
    high_conflict = finite(weighted_conflict) and weighted_conflict >= 0.35

    if high_barrier or (high_gap and worse_than_base):
        return (
            "avoid_uniform_average",
            "不要用 0.5/0.5 或 1/n；先做 connectivity/barrier 筛选，再用验证集重学同构平均权重。",
        )
    if high_gap and high_conflict:
        return (
            "structured_average",
            "直接平均有明显剩余损失；优先做 layer/module/expert-wise 权重，冲突坐标降权或回到 anchor。",
        )
    if modest_gap:
        return (
            "coefficient_search",
            "平均可用但系数敏感；用 min-max validation objective 选 alpha/beta 或 layer-wise lambda。",
        )
    if high_conflict:
        return (
            "average_with_conflict_probe",
            "当前均匀平均没有明显失败，但冲突偏高；保留 TIES/Fisher/RegMean 作为结构化平均备选。",
        )
    return (
        "uniform_average_ok",
        "当前证据支持同构均匀平均作为 baseline；仍需在 held-out 任务上确认 general retention。",
    )


def summarize_experiment(name: str, directory: str | Path) -> dict[str, Any]:
    exp_dir = repo_path(directory)
    grid = read_csv_if_exists(exp_dir / "grid_metrics.csv")
    methods = read_csv_if_exists(exp_dir / "method_metrics.csv")
    if grid is None or grid.empty:
        raise FileNotFoundError(f"Missing grid_metrics.csv in {exp_dir}")
    metric = choose_metric(grid)
    objective = metric["objective"]
    orientation = metric["orientation"]
    best_grid = sort_by_metric(grid, objective, orientation).iloc[0]
    base = pick_method(methods, "base")
    if base is None:
        base = pick_point(grid, 0.0, 0.0)
    linear = pick_method(methods, "linear_average")
    if linear is None:
        linear = pick_point(grid, 0.5, 0.5)
    best_method = sort_by_metric(methods, objective, orientation).iloc[0] if methods is not None and objective in methods.columns else None
    conflict = summarize_conflict(exp_dir)

    base_value = clean_float(base[objective]) if base is not None and objective in base.index else None
    linear_value = clean_float(linear[objective]) if linear is not None and objective in linear.index else None
    best_grid_value = clean_float(best_grid[objective])
    best_method_value = clean_float(best_method[objective]) if best_method is not None and objective in best_method.index else None
    linear_gain = metric_gain(linear_value, base_value, orientation)
    linear_gap = metric_gap(linear_value, best_grid_value, orientation)
    barrier = loss_barrier(grid)
    verdict, action = recommendation(
        family=metric["family"],
        orientation=orientation,
        linear_gap_to_best=linear_gap,
        linear_gain_vs_base=linear_gain,
        midpoint_barrier=barrier,
        weighted_conflict=clean_float(conflict["weighted_conflict"]),
    )

    weights = {
        "alpha": clean_float(best_grid["alpha"]) if "alpha" in best_grid.index else None,
        "beta": clean_float(best_grid["beta"]) if "beta" in best_grid.index else None,
        "source": "best_grid_by_worst_objective",
    }
    return {
        "experiment": name,
        "directory": rel(exp_dir),
        "metric_family": metric["family"],
        "objective": objective,
        "orientation": orientation,
        "base_value": base_value,
        "linear_average_value": linear_value,
        "best_grid_value": best_grid_value,
        "best_grid_alpha": weights["alpha"],
        "best_grid_beta": weights["beta"],
        "best_method": str(best_method["method"]) if best_method is not None and "method" in best_method.index else None,
        "best_method_value": best_method_value,
        "linear_gain_vs_base": linear_gain,
        "linear_gap_to_best_grid": linear_gap,
        "midpoint_loss_barrier": barrier,
        "conflict_cosine": clean_float(conflict["cosine"]),
        "weighted_conflict": clean_float(conflict["weighted_conflict"]),
        "max_weighted_conflict": clean_float(conflict["max_weighted_conflict"]),
        "top_conflict_group": conflict["top_conflict_group"],
        "verdict": verdict,
        "average_weight_suggestion": json.dumps(weights, ensure_ascii=False),
        "recommended_action": action,
    }


def summarize_router_dir(path: Path) -> dict[str, Any] | None:
    router = read_csv_if_exists(path / "router_summary.csv")
    expert = read_csv_if_exists(path / "expert_load.csv")
    overlap = read_csv_if_exists(path / "route_overlap.csv")
    if router is None and expert is None and overlap is None:
        return None
    result: dict[str, Any] = {"directory": rel(path)}
    if router is not None and not router.empty:
        for col in ("router_entropy_mean", "top1_margin_mean", "max_top1_fraction", "effective_top1_experts"):
            if col in router.columns:
                result[f"mean_{col}"] = float(router[col].mean())
        result["num_router_rows"] = int(len(router))
    if expert is not None and not expert.empty:
        result["num_expert_load_rows"] = int(len(expert))
        if "top1_fraction" in expert.columns:
            result["max_expert_top1_fraction"] = float(expert["top1_fraction"].max())
    if overlap is not None and not overlap.empty:
        for col in ("top1_overlap", "topk_jaccard"):
            if col in overlap.columns:
                result[f"mean_{col}"] = float(overlap[col].mean())
        result["num_overlap_rows"] = int(len(overlap))

    actions = []
    max_top1 = clean_float(result.get("mean_max_top1_fraction")) or clean_float(result.get("max_expert_top1_fraction"))
    if finite(max_top1) and max_top1 > 0.50:
        actions.append("router 有 collapse 风险：先 frozen router，再试 load-balance regularized calibration。")
    topk_overlap = clean_float(result.get("mean_topk_jaccard"))
    if finite(topk_overlap) and topk_overlap < 0.50:
        actions.append("route overlap 偏低：不要直接平均 router，先做 route-overlap regularized average 或 expert matching。")
    if not actions:
        actions.append("routing probe 未显示强 collapse/漂移；可以进入 expert-wise average 权重估计。")
    result["recommended_actions"] = actions
    return result


def decision_actions() -> list[dict[str, str]]:
    return [
        {
            "parameter_group": "embedding/lm_head",
            "trigger": "general retention 下降或 tokenizer 高频 token NLL 上升",
            "same_shape_average_action": "freeze anchor 或 Fisher-weighted average，避免全局输出分布漂移。",
        },
        {
            "parameter_group": "attention",
            "trigger": "layer cosine 同向且 barrier 低",
            "same_shape_average_action": "允许普通/task-arithmetic average；系数由 validation min-max objective 选择。",
        },
        {
            "parameter_group": "MLP / dense FFN",
            "trigger": "weighted sign conflict 高",
            "same_shape_average_action": "用 TIES/DELLA/Fisher 风格的 coordinate-wise 权重，冲突坐标降权或回 anchor。",
        },
        {
            "parameter_group": "MoE router",
            "trigger": "route entropy 低、max top-1 fraction 高或 route overlap 低",
            "same_shape_average_action": "首轮 frozen router；之后只校准 router/bias，加入 load-balance 和 route-overlap 约束。",
        },
        {
            "parameter_group": "MoE experts",
            "trigger": "expert output/activation 相似但 index 不可靠",
            "same_shape_average_action": "先 expert matching，再在同 expert 数和同 tensor shape 内做 expert-wise average。",
        },
        {
            "parameter_group": "LoRA/adapters",
            "trigger": "多个下游用户只发布 adapter",
            "same_shape_average_action": "先做 adapter delta/rank/output probe；最终压回一个同构 adapter，mixture 只作上界。",
        },
    ]


def build_report(decisions: list[dict[str, Any]], router_summaries: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# Average Decision Report",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "这个报告把已有 merge plane、method table、delta conflict 和可选 MoE routing probe 汇总成同构 Average 决策。这里的同构指：最终 checkpoint 的 config、tokenizer、layer 数、hidden size、router shape 和 expert 数量不变；改变的是参数来自哪些 source checkpoint 以及每组参数的平均权重。",
        "",
        "## 当前证据",
        "",
        "| experiment | objective | linear | best grid | gap | barrier | conflict | verdict | suggested weights |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in decisions:
        lines.append(
            "| {experiment} | {objective} | {linear} | {best} | {gap} | {barrier} | {conflict} | {verdict} | `{weights}` |".format(
                experiment=row["experiment"],
                objective=row["objective"],
                linear=format_number(row["linear_average_value"]),
                best=format_number(row["best_grid_value"]),
                gap=format_number(row["linear_gap_to_best_grid"]),
                barrier=format_number(row["midpoint_loss_barrier"]),
                conflict=format_number(row["weighted_conflict"]),
                verdict=row["verdict"],
                weights=row["average_weight_suggestion"],
            )
        )
    lines.extend(["", "## 决策解释", ""])
    for row in decisions:
        lines.extend(
            [
                f"### {row['experiment']}",
                "",
                f"- 判断：`{row['verdict']}`。",
                f"- 建议：{row['recommended_action']}",
                f"- 当前建议权重：`{row['average_weight_suggestion']}`。",
            ]
        )
        if row.get("top_conflict_group"):
            lines.append(f"- 冲突最高位置：`{row['top_conflict_group']}`。")
        lines.append("")

    lines.extend(
        [
            "## 参数组策略",
            "",
            "| parameter group | trigger | same-shape average action |",
            "| --- | --- | --- |",
        ]
    )
    for action in decision_actions():
        lines.append(
            f"| {action['parameter_group']} | {action['trigger']} | {action['same_shape_average_action']} |"
        )

    lines.extend(["", "## MoE Routing Probe", ""])
    if router_summaries:
        for summary in router_summaries:
            lines.append(f"- `{summary['directory']}`: " + "; ".join(summary["recommended_actions"]))
    else:
        lines.append("当前没有找到 MoE routing probe 输出。下一步用 `scripts/probe_moe_routing.py` 跑 Qwen3 MoE，再把输出目录传给这个脚本的 `--router-dir`。")

    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 对 Dense Qwen 7B 候选模型先跑 endpoint NLL/benchmark slice，再跑 delta/connectivity probe。",
            "2. 用本报告选择全局 `alpha/beta` 或 layer/module-wise coefficient，禁止把结构改变的 union/ensemble 当最终模型。",
            "3. 对 Qwen3 MoE 先跑 routing probe；若 route overlap 低，优先 router frozen + expert matching，再写回同构 MoE checkpoint。",
            "",
            "## Files",
            "",
            f"- `{rel(output_dir / 'decision_table.csv')}`",
            f"- `{rel(output_dir / 'parameter_group_actions.csv')}`",
            f"- `{rel(output_dir / 'summary.json')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def format_number(value: Any) -> str:
    number = clean_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"


def parse_experiment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        path = raw.strip()
        return Path(path).name, path
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a same-shape Average decision report from merge/probe artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/average_decision_report"))
    parser.add_argument(
        "--experiment",
        action="append",
        help="Experiment spec NAME=DIR. Defaults to the current dense/Qwen merge artifacts.",
    )
    parser.add_argument(
        "--router-dir",
        action="append",
        default=[],
        help="Optional MoE routing probe output directory containing router_summary.csv/expert_load.csv/route_overlap.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [parse_experiment(item) for item in args.experiment] if args.experiment else DEFAULT_EXPERIMENTS
    decisions = [summarize_experiment(name, path) for name, path in experiments]
    router_summaries = []
    for router_dir in args.router_dir:
        summary = summarize_router_dir(repo_path(router_dir))
        if summary is not None:
            router_summaries.append(summary)

    pd.DataFrame(decisions).to_csv(output_dir / "decision_table.csv", index=False)
    pd.DataFrame(decision_actions()).to_csv(output_dir / "parameter_group_actions.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments,
        "decisions": decisions,
        "router_summaries": router_summaries,
        "same_shape_constraint": "Final Average must keep the input model architecture/config/tokenizer/router/expert count unchanged.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(decisions, router_summaries, output_dir), encoding="utf-8")
    print(f"Wrote Average decision report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
