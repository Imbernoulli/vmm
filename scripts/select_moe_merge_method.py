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


def choose_recommendation(selection: pd.DataFrame) -> pd.Series | None:
    candidates = selection[selection["decision"].isin(["candidate_materialize", "candidate_with_router_guard"])]
    if candidates.empty:
        return None
    return candidates.sort_values(["selection_score", "worst_acc", "avg_acc"], ascending=False).iloc[0]


def build_report(
    *,
    output_dir: Path,
    selection: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    recommendation = summary.get("recommended_method")
    lines = [
        "# MoE Merge Method Selection",
        "",
        "这个报告把 MoE 方法分数和 routing readiness 合在一起，给出是否 materialize 的决策。它的边界是保守的：endpoint 只能作 baseline，低 route-overlap / 低 top-1 agreement 的 average 会被拒绝，能过性能和 routing gate 的方法才进入 checkpoint writer 或下一轮 held-out eval。",
        "",
        f"- Recommended method: `{recommendation or 'none'}`",
        f"- Selection status: `{summary['selection_status']}`",
        f"- Base worst accuracy: `{summary.get('base_worst_acc')}`",
        "",
        "## Decision Table",
        "",
        "| method | kind | worst acc | avg acc | calibrate flags | min top-k Jaccard | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in selection.sort_values(["selection_score"], ascending=False).iterrows():
        min_jaccard = row.get("min_topk_jaccard")
        min_jaccard_text = "n/a" if pd.isna(min_jaccard) else f"{float(min_jaccard):.4g}"
        lines.append(
            f"| {row['method']} | {row['method_kind']} | {float(row['worst_acc']):.3f} | {float(row['avg_acc']):.3f} | "
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
            f"- `{rel(output_dir / 'summary.json')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select MoE merge methods from performance metrics and routing readiness probes.")
    parser.add_argument("--method-metrics", default="results/toy_moe_merge/method_metrics.csv")
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
    selection = build_selection_table(
        method_metrics,
        router_readiness,
        min_worst_acc_delta_vs_base=args.min_worst_acc_delta_vs_base,
        max_calibrate_count=args.max_calibrate_count,
    )
    recommended = choose_recommendation(selection)
    base_rows = selection[selection["method"] == "base"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_status": "has_candidate" if recommended is not None else "no_candidate",
        "recommended_method": None if recommended is None else str(recommended["method"]),
        "recommended_decision": None if recommended is None else str(recommended["decision"]),
        "base_worst_acc": None if base_rows.empty else float(base_rows.iloc[0]["worst_acc"]),
        "thresholds": {
            "min_worst_acc_delta_vs_base": args.min_worst_acc_delta_vs_base,
            "max_calibrate_count": args.max_calibrate_count,
        },
        "inputs": {
            "method_metrics": rel(args.method_metrics),
            "router_readiness": rel(args.router_readiness),
        },
        "outputs": {
            "method_selection": rel(output_dir / "method_selection.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    selection.to_csv(output_dir / "method_selection.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(output_dir=output_dir, selection=selection, summary=summary), encoding="utf-8")
    print(f"Wrote MoE merge method selection to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
