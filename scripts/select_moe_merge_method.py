#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_METHODS = {"base"}


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


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def method_kind(method: str) -> str:
    if method == "base":
        return "base_baseline"
    if "endpoint" in method:
        return "endpoint_baseline"
    if "average" in method or "merge" in method:
        return "merge_candidate"
    return "other_candidate"


def aggregate_readiness(router_readiness: pd.DataFrame) -> pd.DataFrame:
    if router_readiness.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "router_rows",
                "max_risk_score",
                "mean_risk_score",
                "calibrate_router_count",
                "low_overlap_count",
                "low_top1_agreement_count",
                "load_concentration_count",
                "min_topk_jaccard",
                "mean_topk_jaccard",
                "min_top1_agreement",
            ]
        )
    rows = []
    for method, group in router_readiness.groupby("method", dropna=False):
        flags = group["risk_flags"].fillna("").astype(str)
        topk = group["topk_jaccard"].dropna()
        top1 = group["top1_agreement"].dropna()
        rows.append(
            {
                "method": str(method),
                "router_rows": int(len(group)),
                "max_risk_score": int(group["risk_score"].max()) if "risk_score" in group else 0,
                "mean_risk_score": float(group["risk_score"].mean()) if "risk_score" in group else 0.0,
                "calibrate_router_count": int((group["recommended_action"] == "calibrate_router_before_average").sum()),
                "low_overlap_count": int(flags.str.contains("low_topk_route_overlap", regex=False).sum()),
                "low_top1_agreement_count": int(flags.str.contains("low_top1_route_agreement", regex=False).sum()),
                "load_concentration_count": int(flags.str.contains("top1_load_concentration", regex=False).sum()),
                "min_topk_jaccard": float(topk.min()) if not topk.empty else None,
                "mean_topk_jaccard": float(topk.mean()) if not topk.empty else None,
                "min_top1_agreement": float(top1.min()) if not top1.empty else None,
            }
        )
    return pd.DataFrame(rows)


def decision_for_row(
    row: pd.Series,
    *,
    base_worst_acc: float | None,
    min_worst_acc_delta_vs_base: float,
    max_calibrate_count: int,
) -> tuple[str, str, float]:
    method = str(row["method"])
    kind = str(row["method_kind"])
    worst_acc = float(row["worst_acc"])
    calibrate_count = int(row.get("calibrate_router_count", 0) or 0)
    low_overlap_count = int(row.get("low_overlap_count", 0) or 0)
    risk_score = float(row.get("mean_risk_score", 0.0) or 0.0)

    if kind == "base_baseline":
        return "baseline_only", "base is the anchor/reference, not a merged candidate.", -1.0
    if kind == "endpoint_baseline":
        return "baseline_only", "endpoint shows source capability but does not materialize an average.", -0.5
    if calibrate_count > max_calibrate_count:
        return (
            "reject_routing_breakdown",
            "routing readiness triggered calibrate_router_before_average; do not materialize without router calibration or expert alignment.",
            worst_acc - 2.0 - 0.25 * low_overlap_count,
        )
    if base_worst_acc is not None and worst_acc < base_worst_acc + min_worst_acc_delta_vs_base:
        return (
            "reject_underperforms_base",
            "worst accuracy does not beat the base/anchor threshold.",
            worst_acc - 1.0 - 0.1 * risk_score,
        )
    score = worst_acc - 0.01 * risk_score - 0.05 * low_overlap_count
    if int(row.get("load_concentration_count", 0) or 0) > 0:
        if "calibrated" in method:
            reason = (
                "candidate passes routing-overlap gate after router calibration; keep load-balance and held-out route checks."
            )
        elif "router_frozen" in method:
            reason = "candidate passes routing-overlap gate with a frozen router; keep load-balance checks."
        else:
            reason = "candidate passes routing-overlap gate but needs router/load-balance guard checks."
        return "candidate_with_router_guard", reason, score
    return "candidate_materialize", "candidate passes performance and routing readiness gates.", score


def build_selection_table(
    method_metrics: pd.DataFrame,
    router_readiness: pd.DataFrame,
    *,
    min_worst_acc_delta_vs_base: float,
    max_calibrate_count: int,
) -> pd.DataFrame:
    readiness = aggregate_readiness(router_readiness)
    table = method_metrics.merge(readiness, on="method", how="left")
    table["method_kind"] = table["method"].astype(str).map(method_kind)
    for column in (
        "router_rows",
        "max_risk_score",
        "mean_risk_score",
        "calibrate_router_count",
        "low_overlap_count",
        "low_top1_agreement_count",
        "load_concentration_count",
    ):
        if column not in table:
            table[column] = 0
        table[column] = table[column].fillna(0)
    base_rows = table[table["method"] == "base"]
    base_worst_acc = float(base_rows.iloc[0]["worst_acc"]) if not base_rows.empty else None
    decisions = []
    reasons = []
    scores = []
    for _, row in table.iterrows():
        decision, reason, score = decision_for_row(
            row,
            base_worst_acc=base_worst_acc,
            min_worst_acc_delta_vs_base=min_worst_acc_delta_vs_base,
            max_calibrate_count=max_calibrate_count,
        )
        decisions.append(decision)
        reasons.append(reason)
        scores.append(score)
    table["decision"] = decisions
    table["decision_reason"] = reasons
    table["selection_score"] = scores
    return table.sort_values(["decision", "selection_score"], ascending=[True, False])


def attach_dispatch_metrics(selection: pd.DataFrame, dispatch_metrics: pd.DataFrame) -> pd.DataFrame:
    if dispatch_metrics.empty:
        return selection
    dispatch_worst = dispatch_metrics.pivot_table(
        index="method",
        columns="dispatch_mode",
        values="worst_acc",
        aggfunc="max",
    )
    dispatch_worst = dispatch_worst.rename(
        columns={mode: f"dispatch_{mode}_worst_acc" for mode in dispatch_worst.columns}
    )
    return selection.merge(dispatch_worst.reset_index(), on="method", how="left")


def attach_capacity_metrics(selection: pd.DataFrame, capacity_metrics: pd.DataFrame) -> pd.DataFrame:
    if capacity_metrics.empty:
        return selection
    rows = []
    for method, group in capacity_metrics.groupby("method", dropna=False):
        topk_overflow = pd.to_numeric(group.get("topk_overflow_fraction"), errors="coerce")
        top1_overflow = pd.to_numeric(group.get("top1_overflow_fraction"), errors="coerce")
        topk_ratio = pd.to_numeric(group.get("max_topk_capacity_ratio"), errors="coerce")
        top1_ratio = pd.to_numeric(group.get("max_top1_capacity_ratio"), errors="coerce")
        worst_idx = topk_overflow.idxmax() if topk_overflow.notna().any() else None
        worst_row = group.loc[worst_idx] if worst_idx is not None else None
        rows.append(
            {
                "method": str(method),
                "capacity_rows": int(len(group)),
                "capacity_overflow_risk_count": int((group.get("capacity_action") == "capacity_overflow_risk").sum())
                if "capacity_action" in group
                else 0,
                "capacity_max_topk_overflow_fraction": float(topk_overflow.max())
                if topk_overflow.notna().any()
                else None,
                "capacity_max_top1_overflow_fraction": float(top1_overflow.max())
                if top1_overflow.notna().any()
                else None,
                "capacity_max_topk_capacity_ratio": float(topk_ratio.max()) if topk_ratio.notna().any() else None,
                "capacity_max_top1_capacity_ratio": float(top1_ratio.max()) if top1_ratio.notna().any() else None,
                "capacity_worst_category": None if worst_row is None else str(worst_row.get("category")),
                "capacity_worst_action": None if worst_row is None else str(worst_row.get("capacity_action")),
            }
        )
    return selection.merge(pd.DataFrame(rows), on="method", how="left")


def choose_recommendation(selection: pd.DataFrame) -> pd.Series | None:
    candidates = selection[selection["decision"].isin(["candidate_materialize", "candidate_with_router_guard"])]
    if candidates.empty:
        return None
    return candidates.sort_values(["selection_score", "worst_acc", "avg_acc"], ascending=False).iloc[0]


def choose_sparse_recommendation(selection: pd.DataFrame, dispatch_mode: str) -> pd.Series | None:
    metric = f"dispatch_{dispatch_mode}_worst_acc"
    if metric not in selection:
        return None
    candidates = selection[selection["decision"].isin(["candidate_materialize", "candidate_with_router_guard"])].copy()
    candidates = candidates[candidates[metric].notna()]
    if candidates.empty:
        return None
    candidates[f"{dispatch_mode}_selection_score"] = (
        candidates[metric].astype(float)
        - 0.01 * candidates["mean_risk_score"].fillna(0.0).astype(float)
        - 0.05 * candidates["low_overlap_count"].fillna(0.0).astype(float)
    )
    return candidates.sort_values([f"{dispatch_mode}_selection_score", metric, "worst_acc"], ascending=False).iloc[0]


def choose_capacity_aware_sparse_recommendation(
    selection: pd.DataFrame,
    dispatch_mode: str,
    *,
    overflow_penalty: float,
) -> pd.Series | None:
    metric = f"dispatch_{dispatch_mode}_worst_acc"
    overflow = "capacity_max_topk_overflow_fraction"
    if metric not in selection or overflow not in selection:
        return None
    candidates = selection[selection["decision"].isin(["candidate_materialize", "candidate_with_router_guard"])].copy()
    candidates = candidates[candidates[metric].notna() & candidates[overflow].notna()]
    if candidates.empty:
        return None
    candidates[f"{dispatch_mode}_capacity_aware_score"] = (
        candidates[metric].astype(float)
        - overflow_penalty * candidates[overflow].astype(float)
        - 0.01 * candidates["mean_risk_score"].fillna(0.0).astype(float)
        - 0.05 * candidates["low_overlap_count"].fillna(0.0).astype(float)
    )
    return candidates.sort_values(
        [f"{dispatch_mode}_capacity_aware_score", metric, overflow, "worst_acc"],
        ascending=[False, False, True, False],
    ).iloc[0]


def build_sparse_pareto_frontier(selection: pd.DataFrame, dispatch_mode: str) -> pd.DataFrame:
    metric = f"dispatch_{dispatch_mode}_worst_acc"
    overflow = "capacity_max_topk_overflow_fraction"
    columns = [
        "method",
        "decision",
        "worst_acc",
        metric,
        overflow,
        "capacity_worst_category",
        "capacity_worst_action",
        "capacity_max_topk_capacity_ratio",
        "min_topk_jaccard",
        "mean_risk_score",
    ]
    if metric not in selection or overflow not in selection:
        return pd.DataFrame(columns=columns)
    candidates = selection[selection["decision"].isin(["candidate_materialize", "candidate_with_router_guard"])].copy()
    candidates = candidates[candidates[metric].notna() & candidates[overflow].notna()]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    acc = candidates[metric].astype(float)
    cap = candidates[overflow].astype(float)
    keep_indices = []
    for idx in candidates.index:
        dominated = False
        for other_idx in candidates.index:
            if other_idx == idx:
                continue
            at_least_as_accurate = acc.loc[other_idx] >= acc.loc[idx]
            no_more_overflow = cap.loc[other_idx] <= cap.loc[idx]
            strictly_better = acc.loc[other_idx] > acc.loc[idx] or cap.loc[other_idx] < cap.loc[idx]
            if at_least_as_accurate and no_more_overflow and strictly_better:
                dominated = True
                break
        if not dominated:
            keep_indices.append(idx)
    frontier = candidates.loc[keep_indices].copy()
    frontier[f"{dispatch_mode}_accuracy_minus_overflow"] = frontier[metric].astype(float) - frontier[overflow].astype(float)
    columns.append(f"{dispatch_mode}_accuracy_minus_overflow")
    return frontier.sort_values([metric, overflow], ascending=[False, True])[
        [column for column in columns if column in frontier]
    ]


def build_report(
    *,
    output_dir: Path,
    selection: pd.DataFrame,
    summary: dict[str, Any],
    sparse_frontier: pd.DataFrame,
) -> str:
    recommendation = summary.get("recommended_method")
    sparse_recommendation = summary.get("recommended_sparse_method")
    sparse_capacity_recommendation = summary.get("recommended_sparse_capacity_aware_method")
    sparse_dispatch_mode = summary.get("recommended_sparse_dispatch_mode")
    frontier_methods = summary.get("sparse_pareto_frontier_methods", [])
    lines = [
        "# MoE Merge Method Selection",
        "",
        "这个报告把 MoE 方法分数和 routing readiness 合在一起，给出是否 materialize 的决策。它的边界是保守的：endpoint 只能作 baseline，低 route-overlap / 低 top-1 agreement 的 average 会被拒绝，能过性能和 routing gate 的方法才进入 checkpoint writer 或下一轮 held-out eval。",
        "",
        f"- Recommended soft-router method: `{recommendation or 'none'}`",
        f"- Recommended sparse `{sparse_dispatch_mode or 'n/a'}` method: `{sparse_recommendation or 'none'}`",
        f"- Capacity-aware sparse `{sparse_dispatch_mode or 'n/a'}` method: `{sparse_capacity_recommendation or 'none'}`",
        f"- Sparse accuracy/overflow Pareto frontier: `{', '.join(frontier_methods) if frontier_methods else 'none'}`",
        f"- Selection status: `{summary['selection_status']}`",
        f"- Base worst accuracy: `{summary.get('base_worst_acc')}`",
        "",
        "## Decision Table",
        "",
        "| method | kind | soft worst acc | hard top-2 worst acc | top-k overflow | avg acc | calibrate flags | min top-k Jaccard | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in selection.sort_values(["selection_score"], ascending=False).iterrows():
        min_jaccard = row.get("min_topk_jaccard")
        min_jaccard_text = "n/a" if pd.isna(min_jaccard) else f"{float(min_jaccard):.4g}"
        hard_top2 = row.get("dispatch_hard_top2_worst_acc")
        hard_top2_text = "n/a" if pd.isna(hard_top2) else f"{float(hard_top2):.3f}"
        overflow = row.get("capacity_max_topk_overflow_fraction")
        overflow_text = "n/a" if pd.isna(overflow) else f"{float(overflow):.4f}"
        lines.append(
            f"| {row['method']} | {row['method_kind']} | {float(row['worst_acc']):.3f} | {hard_top2_text} | {overflow_text} | {float(row['avg_acc']):.3f} | "
            f"{int(row['calibrate_router_count'])} | {min_jaccard_text} | `{row['decision']}` |"
        )
    lines.extend(["", "## Recommendation", ""])
    if recommendation:
        rec = selection[selection["method"] == recommendation].iloc[0]
        lines.extend(
            [
                f"推荐先 materialize/复评 `{recommendation}`。",
                "",
                f"- worst accuracy: `{float(rec['worst_acc']):.3f}`",
                f"- avg accuracy: `{float(rec['avg_acc']):.3f}`",
                f"- decision: `{rec['decision']}`",
                f"- reason: {rec['decision_reason']}",
            ]
        )
    else:
        lines.append("当前没有方法同时通过性能和 routing readiness gate；应先做 router calibration、expert matching 或重新选 source。")
    if sparse_recommendation:
        rec = selection[selection["method"] == sparse_recommendation].iloc[0]
        sparse_metric = f"dispatch_{sparse_dispatch_mode}_worst_acc"
        lines.extend(
            [
                "",
                f"如果部署路径使用 `{sparse_dispatch_mode}` sparse dispatch，优先复评 `{sparse_recommendation}`。",
                "",
                f"- {sparse_dispatch_mode} worst accuracy: `{float(rec[sparse_metric]):.3f}`",
                f"- soft worst accuracy: `{float(rec['worst_acc']):.3f}`",
                f"- decision: `{rec['decision']}`",
            ]
        )
    if sparse_capacity_recommendation:
        rec = selection[selection["method"] == sparse_capacity_recommendation].iloc[0]
        sparse_metric = f"dispatch_{sparse_dispatch_mode}_worst_acc"
        lines.extend(
            [
                "",
                f"如果同时惩罚 `{sparse_dispatch_mode}` accuracy loss 和 capacity overflow，当前优先复评 `{sparse_capacity_recommendation}`。",
                "",
                f"- capacity-aware score: `{summary.get('recommended_sparse_capacity_aware_score'):.4f}`",
                f"- {sparse_dispatch_mode} worst accuracy: `{float(rec[sparse_metric]):.3f}`",
                f"- max top-k overflow fraction: `{float(rec['capacity_max_topk_overflow_fraction']):.4f}`",
                f"- worst overflow category: `{rec['capacity_worst_category']}`",
            ]
        )
    if not sparse_frontier.empty:
        sparse_metric = f"dispatch_{sparse_dispatch_mode}_worst_acc"
        lines.extend(
            [
                "",
                "## Sparse Pareto Frontier",
                "",
                "这些点在 hard top-2 accuracy 和 top-k capacity overflow 上互不支配；部署前应围绕这几个点做 vLLM 下游评测，而不是只看 soft-router 分数。",
                "",
                "| method | hard top-2 worst acc | top-k overflow | worst category | soft worst acc |",
                "| --- | ---: | ---: | --- | ---: |",
            ]
        )
        for _, row in sparse_frontier.iterrows():
            lines.append(
                f"| {row['method']} | {float(row[sparse_metric]):.3f} | "
                f"{float(row['capacity_max_topk_overflow_fraction']):.4f} | "
                f"{row.get('capacity_worst_category', 'n/a')} | {float(row['worst_acc']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## 规则",
            "",
            "- `base` 和 endpoint 只能说明 anchor/source 能力，不算平均结果。",
            "- `calibrate_router_before_average` 计数大于阈值时，拒绝直接 materialize。",
            "- 候选方法必须至少不低于 base worst accuracy 阈值。",
            "- 通过 route-overlap 但有 load concentration 的方法标为 `candidate_with_router_guard`，表示 materialize 后仍需做 load-balance 和 held-out route-overlap 检查。",
            "",
            "## Files",
            "",
            f"- `{rel(output_dir / 'method_selection.csv')}`",
            f"- `{rel(output_dir / 'sparse_pareto_frontier.csv')}`",
            f"- `{rel(output_dir / 'summary.json')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select MoE merge methods from performance metrics and routing readiness probes.")
    parser.add_argument("--method-metrics", default="results/toy_moe_merge/method_metrics.csv")
    parser.add_argument("--dispatch-metrics", default="results/toy_moe_merge/dispatch_mode_metrics.csv")
    parser.add_argument("--capacity-metrics", default="results/toy_moe_merge/router_capacity_metrics.csv")
    parser.add_argument("--sparse-dispatch-mode", default="hard_top2")
    parser.add_argument("--sparse-overflow-penalty", type=float, default=1.0)
    parser.add_argument("--router-readiness", default="results/toy_moe_routing_readiness/router_readiness.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_moe_method_selection"))
    parser.add_argument("--min-worst-acc-delta-vs-base", type=float, default=0.0)
    parser.add_argument("--max-calibrate-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_metrics = read_csv(args.method_metrics)
    router_readiness = read_csv(args.router_readiness)
    dispatch_metrics = read_csv(args.dispatch_metrics) if repo_path(args.dispatch_metrics).exists() else pd.DataFrame()
    capacity_metrics = read_csv(args.capacity_metrics) if repo_path(args.capacity_metrics).exists() else pd.DataFrame()
    selection = build_selection_table(
        method_metrics,
        router_readiness,
        min_worst_acc_delta_vs_base=args.min_worst_acc_delta_vs_base,
        max_calibrate_count=args.max_calibrate_count,
    )
    selection = attach_dispatch_metrics(selection, dispatch_metrics)
    selection = attach_capacity_metrics(selection, capacity_metrics)
    recommended = choose_recommendation(selection)
    sparse_recommended = choose_sparse_recommendation(selection, args.sparse_dispatch_mode)
    sparse_capacity_recommended = choose_capacity_aware_sparse_recommendation(
        selection,
        args.sparse_dispatch_mode,
        overflow_penalty=args.sparse_overflow_penalty,
    )
    sparse_frontier = build_sparse_pareto_frontier(selection, args.sparse_dispatch_mode)
    base_rows = selection[selection["method"] == "base"]
    sparse_capacity_metric = f"{args.sparse_dispatch_mode}_capacity_aware_score"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_status": "has_candidate" if recommended is not None else "no_candidate",
        "recommended_method": None if recommended is None else str(recommended["method"]),
        "recommended_decision": None if recommended is None else str(recommended["decision"]),
        "recommended_sparse_dispatch_mode": args.sparse_dispatch_mode,
        "recommended_sparse_method": None if sparse_recommended is None else str(sparse_recommended["method"]),
        "recommended_sparse_decision": None if sparse_recommended is None else str(sparse_recommended["decision"]),
        "recommended_sparse_worst_acc": None
        if sparse_recommended is None
        else float(sparse_recommended[f"dispatch_{args.sparse_dispatch_mode}_worst_acc"]),
        "recommended_sparse_capacity_aware_method": None
        if sparse_capacity_recommended is None
        else str(sparse_capacity_recommended["method"]),
        "recommended_sparse_capacity_aware_decision": None
        if sparse_capacity_recommended is None
        else str(sparse_capacity_recommended["decision"]),
        "recommended_sparse_capacity_aware_score": None
        if sparse_capacity_recommended is None
        else float(sparse_capacity_recommended[sparse_capacity_metric]),
        "recommended_sparse_capacity_aware_worst_acc": None
        if sparse_capacity_recommended is None
        else float(sparse_capacity_recommended[f"dispatch_{args.sparse_dispatch_mode}_worst_acc"]),
        "recommended_sparse_capacity_aware_topk_overflow_fraction": None
        if sparse_capacity_recommended is None
        else float(sparse_capacity_recommended["capacity_max_topk_overflow_fraction"]),
        "sparse_overflow_penalty": args.sparse_overflow_penalty,
        "sparse_pareto_frontier_rows": int(len(sparse_frontier)),
        "sparse_pareto_frontier_methods": []
        if sparse_frontier.empty
        else [str(method) for method in sparse_frontier["method"].tolist()],
        "base_worst_acc": None if base_rows.empty else float(base_rows.iloc[0]["worst_acc"]),
        "thresholds": {
            "min_worst_acc_delta_vs_base": args.min_worst_acc_delta_vs_base,
            "max_calibrate_count": args.max_calibrate_count,
            "sparse_overflow_penalty": args.sparse_overflow_penalty,
        },
        "inputs": {
            "method_metrics": rel(args.method_metrics),
            "dispatch_metrics": rel(args.dispatch_metrics),
            "capacity_metrics": rel(args.capacity_metrics),
            "router_readiness": rel(args.router_readiness),
        },
        "outputs": {
            "method_selection": rel(output_dir / "method_selection.csv"),
            "sparse_pareto_frontier": rel(output_dir / "sparse_pareto_frontier.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    selection.to_csv(output_dir / "method_selection.csv", index=False)
    sparse_frontier.to_csv(output_dir / "sparse_pareto_frontier.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(output_dir=output_dir, selection=selection, summary=summary, sparse_frontier=sparse_frontier),
        encoding="utf-8",
    )
    print(f"Wrote MoE merge method selection to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
