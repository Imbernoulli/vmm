#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_PLAN_COLUMNS = [
    "router_dir",
    "category",
    "prompt_idx",
    "router",
    "num_experts",
    "max_top1_fraction",
    "effective_top1_experts",
    "top1_margin_mean",
    "topk_jaccard",
    "same_shape_action",
    "reason",
]
EXPERT_PLAN_COLUMNS = [
    "router_dir",
    "category",
    "prompt_idx",
    "router",
    "expert_id",
    "top1_fraction",
    "topk_fraction",
    "importance_within_router",
    "same_shape_action",
    "reason",
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


def read_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    path = repo_path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def normalize(series: pd.Series) -> pd.Series:
    total = float(series.sum())
    if total <= 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return series / total


def summarize_average_decisions(decision_table: pd.DataFrame | None) -> dict[str, Any]:
    if decision_table is None or decision_table.empty:
        return {
            "global_verdict": "unknown",
            "default_weight_source": "missing_decision_table",
            "avoid_uniform_count": 0,
            "coefficient_search_count": 0,
        }
    verdicts = decision_table["verdict"].astype(str) if "verdict" in decision_table.columns else pd.Series([], dtype=str)
    avoid = int((verdicts == "avoid_uniform_average").sum())
    search = int((verdicts == "coefficient_search").sum())
    if avoid > 0:
        global_verdict = "avoid_uniform_average"
        source = "validation_grid_or_layerwise_coefficients"
    elif search > 0:
        global_verdict = "coefficient_search"
        source = "validation_grid_coefficients"
    else:
        global_verdict = "uniform_average_ok"
        source = "uniform_or_soup_coefficients"
    return {
        "global_verdict": global_verdict,
        "default_weight_source": source,
        "avoid_uniform_count": avoid,
        "coefficient_search_count": search,
    }


def summarize_router(router_dir: Path) -> dict[str, Any] | None:
    router_summary = read_csv_if_exists(router_dir / "router_summary.csv")
    expert_load = read_csv_if_exists(router_dir / "expert_load.csv")
    route_overlap = read_csv_if_exists(router_dir / "route_overlap.csv")
    if router_summary is None and expert_load is None and route_overlap is None:
        return None
    summary: dict[str, Any] = {
        "router_dir": rel(router_dir),
        "has_router_summary": router_summary is not None,
        "has_expert_load": expert_load is not None,
        "has_route_overlap": route_overlap is not None,
    }
    if router_summary is not None and not router_summary.empty:
        for column in (
            "router_entropy_mean",
            "top1_margin_mean",
            "max_top1_fraction",
            "effective_top1_experts",
            "unique_top1_experts",
            "unique_topk_experts",
        ):
            if column in router_summary.columns:
                summary[f"mean_{column}"] = float(router_summary[column].mean())
        if "num_experts" in router_summary.columns:
            summary["median_num_experts"] = float(router_summary["num_experts"].median())
    if route_overlap is not None and not route_overlap.empty:
        for column in ("top1_overlap", "topk_jaccard"):
            if column in route_overlap.columns:
                summary[f"mean_{column}"] = float(route_overlap[column].mean())
    if expert_load is not None and not expert_load.empty:
        if "top1_fraction" in expert_load.columns:
            summary["max_expert_top1_fraction"] = float(expert_load["top1_fraction"].max())
        if "topk_fraction" in expert_load.columns:
            summary["max_expert_topk_fraction"] = float(expert_load["topk_fraction"].max())
    return summary


def router_action(row: dict[str, Any], collapse_threshold: float, overlap_threshold: float) -> tuple[str, str]:
    max_top1 = maybe_float(row.get("max_top1_fraction"))
    if max_top1 is None:
        max_top1 = maybe_float(row.get("mean_max_top1_fraction"))
    topk_overlap = maybe_float(row.get("topk_jaccard"))
    if topk_overlap is None:
        topk_overlap = maybe_float(row.get("mean_topk_jaccard"))
    if max_top1 is not None and max_top1 >= collapse_threshold:
        return (
            "freeze_then_calibrate",
            "router top-1 load 过于集中；先冻结 anchor router，再用 load-balance 和 route-overlap 约束校准 router bias/linear weights。",
        )
    if topk_overlap is not None and topk_overlap < overlap_threshold:
        return (
            "route_overlap_regularized_average",
            "source routers 分歧较大；避免直接平均 router，改用 route-overlap regularized average 或 HARC-style calibration。",
        )
    if topk_overlap is not None:
        return (
            "small_lambda_router_average",
            "route overlap 尚可；允许小 lambda router delta，但必须验证 routing entropy 和 load balance。",
        )
    return (
        "router_frozen_until_probe",
        "目前没有 route-overlap 证据；在 routing probe 可用前保留 anchor/general router。",
    )


def build_parameter_group_plan(
    decision_summary: dict[str, Any],
    router_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    global_verdict = decision_summary["global_verdict"]
    default_source = decision_summary["default_weight_source"]
    if global_verdict == "avoid_uniform_average":
        shared_policy = "coefficient_search_or_layerwise_average"
        shared_reason = "现有 Dense/Qwen probe 显示均匀平均可能穿过高 loss 区域。"
    elif global_verdict == "coefficient_search":
        shared_policy = "validation_weighted_average"
        shared_reason = "均匀平均可作为起点，但系数敏感，需要验证集选择。"
    else:
        shared_policy = "uniform_or_soup_average"
        shared_reason = "当前 probe 没有否定均匀平均。"

    router_policy = "router_frozen_until_probe"
    router_reason = "没有传入 MoE routing probe。"
    if router_summaries:
        if any(router_action(row, 0.50, 0.50)[0] == "freeze_then_calibrate" for row in router_summaries):
            router_policy = "freeze_then_calibrate"
            router_reason = "至少一个 routing probe 显示 expert load 过于集中。"
        elif any(router_action(row, 0.50, 0.50)[0] == "route_overlap_regularized_average" for row in router_summaries):
            router_policy = "route_overlap_regularized_average"
            router_reason = "至少一个 routing probe 显示 route overlap 偏低。"
        else:
            router_policy = "small_lambda_router_average"
            router_reason = "routing probe 暂未显示明显 collapse 或漂移。"

    return [
        {
            "parameter_group": "embedding / lm_head",
            "same_shape_action": "anchor_or_fisher_weighted_average",
            "weight_source": "general_retention_and_token_nll",
            "reason": "这些张量控制全局 token 分布；只有 general retention 不下降时才平均。",
        },
        {
            "parameter_group": "shared attention / norms",
            "same_shape_action": shared_policy,
            "weight_source": default_source,
            "reason": shared_reason,
        },
        {
            "parameter_group": "shared dense MLP",
            "same_shape_action": "conflict_aware_coordinate_average",
            "weight_source": "weighted_sign_conflict + Fisher/activation statistics",
            "reason": "MLP delta 往往包含专长冲突；冲突高时用 TIES/DELLA/Fisher 风格的 coordinate weighting。",
        },
        {
            "parameter_group": "MoE router",
            "same_shape_action": router_policy,
            "weight_source": "router_entropy + load_balance + route_overlap",
            "reason": router_reason,
        },
        {
            "parameter_group": "MoE experts",
            "same_shape_action": "expert_matched_route_frequency_average",
            "weight_source": "expert output similarity + route frequency + NLL sensitivity",
            "reason": "保持 expert 数和 tensor shape 不变；如果 expert index 语义不确定，先匹配再平均。",
        },
        {
            "parameter_group": "LoRA / adapters",
            "same_shape_action": "adapter_delta_average_or_distill_back",
            "weight_source": "rank overlap + adapter output similarity",
            "reason": "mixture 可作上界，但最终 artifact 应压回一个同构 adapter/checkpoint。",
        },
    ]


def build_router_plan(
    router_dirs: list[Path],
    collapse_threshold: float,
    overlap_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for router_dir in router_dirs:
        router_summary = read_csv_if_exists(router_dir / "router_summary.csv")
        overlap = read_csv_if_exists(router_dir / "route_overlap.csv")
        if router_summary is None or router_summary.empty:
            continue
        merged = router_summary.copy()
        if overlap is not None and not overlap.empty:
            join_cols = [col for col in ("category", "prompt_idx", "router") if col in merged.columns and col in overlap.columns]
            if join_cols:
                merged = merged.merge(overlap, on=join_cols, how="left", suffixes=("", "_overlap"))
        for _, item in merged.iterrows():
            action, reason = router_action(item.to_dict(), collapse_threshold, overlap_threshold)
            rows.append(
                {
                    "router_dir": rel(router_dir),
                    "category": item.get("category"),
                    "prompt_idx": item.get("prompt_idx"),
                    "router": item.get("router"),
                    "num_experts": item.get("num_experts"),
                    "max_top1_fraction": item.get("max_top1_fraction"),
                    "effective_top1_experts": item.get("effective_top1_experts"),
                    "top1_margin_mean": item.get("top1_margin_mean"),
                    "topk_jaccard": item.get("topk_jaccard"),
                    "same_shape_action": action,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=ROUTER_PLAN_COLUMNS)


def build_expert_plan(router_dirs: list[Path], high_frequency_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for router_dir in router_dirs:
        expert_load = read_csv_if_exists(router_dir / "expert_load.csv")
        if expert_load is None or expert_load.empty:
            continue
        group_cols = [col for col in ("category", "prompt_idx", "router") if col in expert_load.columns]
        if group_cols and "topk_fraction" in expert_load.columns:
            expert_load = expert_load.copy()
            expert_load["importance_within_router"] = expert_load.groupby(group_cols)["topk_fraction"].transform(normalize)
        else:
            expert_load["importance_within_router"] = expert_load.get("topk_fraction", 0.0)
        for _, item in expert_load.iterrows():
            topk_fraction = maybe_float(item.get("topk_fraction")) or 0.0
            top1_fraction = maybe_float(item.get("top1_fraction")) or 0.0
            importance = maybe_float(item.get("importance_within_router")) or 0.0
            if topk_fraction >= high_frequency_threshold or top1_fraction >= high_frequency_threshold:
                action = "route_frequency_weighted_average"
                reason = "高频 expert；只用 route/task-aware 权重平均，并验证 NLL sensitivity。"
            elif importance <= 0.01:
                action = "anchor_heavy_or_freeze"
                reason = "低使用率 expert；除非 task-specific probe 显示稀有能力，否则保留 anchor/general 权重。"
            else:
                action = "low_lambda_average"
                reason = "中等使用率 expert；使用较小 lambda 和 conflict-aware 权重。"
            rows.append(
                {
                    "router_dir": rel(router_dir),
                    "category": item.get("category"),
                    "prompt_idx": item.get("prompt_idx"),
                    "router": item.get("router"),
                    "expert_id": item.get("expert_id"),
                    "top1_fraction": top1_fraction,
                    "topk_fraction": topk_fraction,
                    "importance_within_router": importance,
                    "same_shape_action": action,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=EXPERT_PLAN_COLUMNS)


def build_report(
    *,
    output_dir: Path,
    decision_summary: dict[str, Any],
    router_summaries: list[dict[str, Any]],
    parameter_plan: list[dict[str, Any]],
    router_plan: pd.DataFrame,
    expert_plan: pd.DataFrame,
) -> str:
    lines = [
        "# MoE Same-Shape Average Plan",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "这个报告把 Dense/Qwen Average 决策、MoE router probe 和 expert load probe 转成同构 MoE checkpoint 的合并计划。这里的同构约束是硬约束：不增加 experts，不改 router shape，不做运行时 ensemble；所有策略最后都要写回原模型结构。",
        "",
        "## 全局判断",
        "",
        f"- Dense/Qwen decision verdict: `{decision_summary['global_verdict']}`。",
        f"- 默认平均权重来源：`{decision_summary['default_weight_source']}`。",
        f"- `avoid_uniform_average` 数量：`{decision_summary['avoid_uniform_count']}`；`coefficient_search` 数量：`{decision_summary['coefficient_search_count']}`。",
        "",
        "## 参数组计划",
        "",
        "| parameter group | same-shape action | weight source | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in parameter_plan:
        lines.append(
            f"| {row['parameter_group']} | `{row['same_shape_action']}` | {row['weight_source']} | {row['reason']} |"
        )

    lines.extend(["", "## Router Probe Summary", ""])
    if router_summaries:
        for summary in router_summaries:
            lines.append(f"- `{summary['router_dir']}`: {json.dumps(summary, ensure_ascii=False)}")
    else:
        lines.append("当前没有传入 MoE routing probe 输出；router 策略默认是 `router_frozen_until_probe`。")

    lines.extend(["", "## Router Plan", ""])
    if router_plan.empty:
        lines.append("没有 router-level CSV。运行 `scripts/probe_moe_routing.py` 后用 `--router-dir` 传入输出目录。")
    else:
        action_counts = router_plan["same_shape_action"].value_counts().to_dict()
        lines.append(f"Router action counts: `{json.dumps(action_counts, ensure_ascii=False)}`。")

    lines.extend(["", "## Expert Plan", ""])
    if expert_plan.empty:
        lines.append("没有 expert-load CSV。运行 `scripts/probe_moe_routing.py` 后会生成 per-expert route-frequency plan。")
    else:
        action_counts = expert_plan["same_shape_action"].value_counts().to_dict()
        lines.append(f"Expert action counts: `{json.dumps(action_counts, ensure_ascii=False)}`。")

    lines.extend(
        [
            "",
            "## 研究依据到实现规则",
            "",
            "- HARC / routing-breakdown 方向说明 MoE router 对 softmax/top-k 扰动非常敏感，所以 router 不应默认同权平均。",
            "- Sub-MoE 方向说明 expert specialization 会造成参数冲突，因此要先按输出相似度/路由频率聚类或匹配，再做 expert-wise average。",
            "- Expert Merging / layer-wise coefficient 方向说明不同层的重要性不同，因此 shared attention、MLP、router、expert FFN 应分组设权重。",
            "- MergeME/WEMoE 类方法可以作为上界或启发，但本项目最终仍要压回同构 checkpoint。",
            "",
            "## Files",
            "",
            f"- `{rel(output_dir / 'parameter_group_plan.csv')}`",
            f"- `{rel(output_dir / 'router_plan.csv')}`",
            f"- `{rel(output_dir / 'expert_plan.csv')}`",
            f"- `{rel(output_dir / 'summary.json')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a same-shape MoE averaging plan from routing/expert probes.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_average_plan"))
    parser.add_argument(
        "--decision-table",
        default="results/average_decision_report/decision_table.csv",
        help="Dense/Average decision table used to choose global/shared-module policies.",
    )
    parser.add_argument(
        "--router-dir",
        action="append",
        default=[],
        help="MoE routing probe output directory from scripts/probe_moe_routing.py.",
    )
    parser.add_argument("--collapse-threshold", type=float, default=0.50)
    parser.add_argument("--overlap-threshold", type=float, default=0.50)
    parser.add_argument("--high-frequency-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_table = read_csv_if_exists(args.decision_table)
    decision_summary = summarize_average_decisions(decision_table)
    router_dirs = [repo_path(path) for path in args.router_dir]
    router_summaries = [summary for path in router_dirs if (summary := summarize_router(path)) is not None]
    parameter_plan = build_parameter_group_plan(decision_summary, router_summaries)
    router_plan = build_router_plan(router_dirs, args.collapse_threshold, args.overlap_threshold)
    expert_plan = build_expert_plan(router_dirs, args.high_frequency_threshold)

    pd.DataFrame(parameter_plan).to_csv(output_dir / "parameter_group_plan.csv", index=False)
    router_plan.to_csv(output_dir / "router_plan.csv", index=False)
    expert_plan.to_csv(output_dir / "expert_plan.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "same_shape_constraint": "Do not change router shape, expert count, hidden size, layer count, tokenizer, or model class.",
        "decision_summary": decision_summary,
        "router_dirs": [rel(path) for path in router_dirs],
        "router_summaries": router_summaries,
        "parameter_plan": parameter_plan,
        "router_plan_rows": int(len(router_plan)),
        "expert_plan_rows": int(len(expert_plan)),
        "thresholds": {
            "collapse_threshold": args.collapse_threshold,
            "overlap_threshold": args.overlap_threshold,
            "high_frequency_threshold": args.high_frequency_threshold,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(
            output_dir=output_dir,
            decision_summary=decision_summary,
            router_summaries=router_summaries,
            parameter_plan=parameter_plan,
            router_plan=router_plan,
            expert_plan=expert_plan,
        ),
        encoding="utf-8",
    )
    print(f"Wrote MoE same-shape average plan to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
