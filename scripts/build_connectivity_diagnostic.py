#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def common_task_keys(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    skip = {"worst", "avg"}
    return sorted(key for key in a if key in b and key not in skip and fnum(a.get(key)) is not None)


def task_winners(endpoints: dict[str, dict[str, Any]]) -> dict[str, str]:
    names = list(endpoints)
    if len(names) < 2:
        return {}
    a_name, b_name = names[:2]
    a = endpoints[a_name]
    b = endpoints[b_name]
    winners = {}
    for key in common_task_keys(a, b):
        av = fnum(a.get(key))
        bv = fnum(b.get(key))
        if av is None or bv is None:
            continue
        winners[key] = a_name if av <= bv else b_name
    return winners


def row_from_path_case(
    *,
    case: str,
    domain: str,
    path_kind: str,
    summary: dict[str, Any],
    metric: str = "worst",
    midpoint_key: str = "t",
    midpoint_value: float = 0.5,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    path = summary.get("interpolation") or []
    endpoints = summary.get("endpoints") or {}
    endpoint_values = [fnum(item.get(metric)) for item in endpoints.values() if fnum(item.get(metric)) is not None]
    endpoint_best = min(endpoint_values) if endpoint_values else fnum(summary.get(f"endpoint_best_{metric}"))
    endpoint_worst = max(endpoint_values) if endpoint_values else None
    interior = [item for item in path if fnum(item.get(midpoint_key)) not in {0.0, 1.0}]
    best_interior = min((fnum(item.get(metric)) for item in interior if fnum(item.get(metric)) is not None), default=None)
    max_path = max((fnum(item.get(metric)) for item in path if fnum(item.get(metric)) is not None), default=None)
    midpoint = min(
        path,
        key=lambda item: abs((fnum(item.get(midpoint_key)) or 0.0) - midpoint_value),
        default={},
    )
    midpoint_metric = fnum(midpoint.get(metric))
    endpoint_frontier_gap = (
        None if best_interior is None or endpoint_best is None else best_interior - endpoint_best
    )
    midpoint_gap = None if midpoint_metric is None or endpoint_best is None else midpoint_metric - endpoint_best
    barrier_vs_worst_endpoint = None if max_path is None or endpoint_worst is None else max_path - endpoint_worst
    winners = task_winners(endpoints)
    winner_set = set(winners.values())
    complementarity_observed = len(winner_set) >= 2
    if endpoint_frontier_gap is not None and endpoint_frontier_gap <= -tolerance:
        decision = "interior_beats_endpoint_frontier"
    elif endpoint_frontier_gap is not None and endpoint_frontier_gap <= tolerance:
        decision = "interior_ties_endpoint_frontier"
    else:
        decision = "reject_interior_average"
    if midpoint_gap is not None and midpoint_gap > tolerance:
        midpoint_decision = "reject_midpoint"
    elif midpoint_gap is not None:
        midpoint_decision = "midpoint_not_worse_than_frontier"
    else:
        midpoint_decision = "midpoint_not_measured"
    return {
        "case": case,
        "domain": domain,
        "path_kind": path_kind,
        "primary_metric": metric,
        "endpoint_best": endpoint_best,
        "endpoint_worst": endpoint_worst,
        "best_interior": best_interior,
        "endpoint_frontier_gap": endpoint_frontier_gap,
        "midpoint_metric": midpoint_metric,
        "midpoint_gap": midpoint_gap,
        "max_path_metric": max_path,
        "barrier_vs_worst_endpoint": barrier_vs_worst_endpoint,
        "task_winners": json.dumps(winners, sort_keys=True),
        "complementarity_observed": complementarity_observed,
        "decision": decision,
        "midpoint_decision": midpoint_decision,
        "same_shape_implication": "allow only if interior beats endpoint frontier and midpoint/path barrier is within tolerance",
    }


def row_from_dense_lambda(summary: dict[str, Any], tolerance: float = 0.01) -> dict[str, Any]:
    rows = summary.get("rows") or []
    linear = fnum(summary.get("linear_worst"))
    best_family = fnum(summary.get("unified_best_worst"))
    best_endpoint = fnum(summary.get("best_endpoint_worst"))
    best_config = summary.get("unified_best_config") or {}
    midpoint_gap = None if linear is None or best_endpoint is None else linear - best_endpoint
    family_gap = None if best_family is None or best_endpoint is None else best_family - best_endpoint
    decision = "anchor_or_abstention_selected"
    if family_gap is not None and family_gap <= -tolerance:
        decision = "family_beats_endpoint_but_not_as_fixed_average"
    elif family_gap is not None and family_gap <= tolerance:
        decision = "family_ties_endpoint_frontier"
    return {
        "case": "dense_base_anchored_lambda_family",
        "domain": "dense",
        "path_kind": "base_plus_lambda_average_delta",
        "primary_metric": "worst",
        "endpoint_best": best_endpoint,
        "endpoint_worst": best_endpoint,
        "best_interior": best_family,
        "endpoint_frontier_gap": family_gap,
        "midpoint_metric": linear,
        "midpoint_gap": midpoint_gap,
        "max_path_metric": max((fnum(item.get("worst")) for item in rows if fnum(item.get("worst")) is not None), default=None),
        "barrier_vs_worst_endpoint": midpoint_gap,
        "task_winners": "{}",
        "complementarity_observed": False,
        "decision": decision,
        "midpoint_decision": "reject_midpoint" if midpoint_gap is not None and midpoint_gap > tolerance else "midpoint_not_worse_than_frontier",
        "same_shape_implication": (
            f"best config {json.dumps(best_config, sort_keys=True)} behaves as endpoint/anchor fallback; "
            "fixed 0.5/0.5 average remains rejected"
        ),
    }


def dense_curvature_row(summary: dict[str, Any]) -> dict[str, Any]:
    law = summary.get("curvature_law") or {}
    ratio_general = fnum(law.get("ratio_general"))
    ratio_code = fnum(law.get("ratio_code"))
    max_ratio = max(value for value in [ratio_general, ratio_code] if value is not None)
    return {
        "case": "dense_fisher_local_quadratic_check",
        "domain": "dense",
        "path_kind": "second_order_prediction",
        "primary_metric": "actual_over_predicted_degradation",
        "endpoint_best": None,
        "endpoint_worst": None,
        "best_interior": None,
        "endpoint_frontier_gap": None,
        "midpoint_metric": max_ratio,
        "midpoint_gap": max_ratio - 5.0 if max_ratio is not None else None,
        "max_path_metric": max_ratio,
        "barrier_vs_worst_endpoint": None,
        "task_winners": "{}",
        "complementarity_observed": False,
        "decision": "reject_local_quadratic_as_sufficient_gate" if max_ratio and max_ratio > 5.0 else "local_quadratic_plausible",
        "midpoint_decision": "fisher_prediction_underestimates_barrier" if max_ratio and max_ratio > 5.0 else "fisher_prediction_plausible",
        "same_shape_implication": (
            f"actual/predicted ratios general/code = {fmt(ratio_general)}/{fmt(ratio_code)}; "
            "Fisher/RegMean-style methods need held-out validation before materialization"
        ),
    }


def build_rules(tolerance: float) -> list[dict[str, Any]]:
    return [
        {
            "rule": "endpoint_frontier_rule",
            "accept_if": f"best interior score <= best endpoint score - {tolerance}",
            "reject_if": "best interior score is worse than the best endpoint frontier",
            "meaning": "A straight-line or coefficient average must beat the source frontier, not only look smooth.",
        },
        {
            "rule": "midpoint_safety_rule",
            "accept_if": f"midpoint score <= best endpoint score + {tolerance}",
            "reject_if": "fixed 0.5/0.5 midpoint has a positive endpoint-frontier gap",
            "meaning": "Uniform averaging is a special case and should be rejected separately from coefficient search.",
        },
        {
            "rule": "complementarity_rule",
            "accept_if": "different endpoints win different tasks and an interior point beats the best source average",
            "reject_if": "one endpoint dominates all measured tasks or the best interior only ties the best source",
            "meaning": "Specialist labels are not enough; complementarity must appear in measured task losses.",
        },
        {
            "rule": "local_quadratic_rule",
            "accept_if": "curvature prediction matches actual midpoint degradation within a small ratio",
            "reject_if": "actual/predicted degradation ratio is large",
            "meaning": "Fisher/RegMean-style local arguments cannot justify a nonlocal merge by themselves.",
        },
    ]


def build_report(summary: dict[str, Any], diagnostics: pd.DataFrame, rules: list[dict[str, Any]]) -> str:
    lines = [
        "# Average Connectivity Diagnostic",
        "",
        "这个诊断把 Dense 和 MoE 的直线路径证据放到同一个判据里：一个 average 是否可以接受，不看方法名，而看它有没有越过 endpoint frontier、midpoint 是否安全、是否真的观测到任务互补，以及局部二阶近似是否足够可信。",
        "",
        "## Current Result",
        "",
        f"- Cases: `{summary['case_count']}`",
        f"- Rejected path/family cases: `{summary['path_rejected_count']}`",
        f"- Rejected fixed-midpoint cases: `{summary['midpoint_rejected_count']}`",
        f"- Endpoint-frontier wins: `{summary['endpoint_frontier_win_count']}`",
        f"- Complementary cases observed: `{summary['complementarity_observed_count']}`",
        f"- Dense midpoint gap: `{fmt(summary['dense_source_midpoint_gap'])}`",
        f"- Qwen3 MoE Instruct/Coder interior gap: `{fmt(summary['qwen3_instruct_coder_gap'])}`",
        "",
        "## Diagnostics",
        "",
        "| case | domain | metric | endpoint best | best interior | endpoint gap | midpoint gap | barrier | complementarity | decision |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in diagnostics.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['domain']}` | `{row['primary_metric']}` | "
            f"{fmt(row['endpoint_best'])} | {fmt(row['best_interior'])} | "
            f"{fmt(row['endpoint_frontier_gap'])} | {fmt(row['midpoint_gap'])} | "
            f"{fmt(row['barrier_vs_worst_endpoint'])} | `{row['complementarity_observed']}` | `{row['decision']}` |"
        )
    lines.extend(["", "## Rules", "", "| rule | accept if | reject if | meaning |", "| --- | --- | --- | --- |"])
    for rule in rules:
        lines.append(f"| `{rule['rule']}` | {rule['accept_if']} | {rule['reject_if']} | {rule['meaning']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['path_diagnostics']}`",
            f"- `{summary['outputs']['rules']}`",
            f"- `{summary['outputs']['figure']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_figure(path: Path, diagnostics: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = diagnostics[diagnostics["endpoint_frontier_gap"].notna()].copy()
    plot_rows = plot_rows[~plot_rows["case"].str.contains("local_quadratic")]
    if plot_rows.empty:
        return
    label_map = {
        "dense_instruct_coder_source_path": "Dense Instruct/Coder source path",
        "dense_base_anchored_lambda_family": "Dense base-anchored lambda family",
        "qwen3_moe_instruct_coder_source_path": "Qwen3 MoE Instruct/Coder source path",
        "qwen3_moe_base_coder_source_path": "Qwen3 MoE Base/Coder source path",
        "qwen3_moe_thinking_coder_complementary_path": "Qwen3 MoE Thinking/Coder complementary path",
    }
    values = [float(value) for value in plot_rows["endpoint_frontier_gap"]]
    labels = [label_map.get(str(item), str(item).replace("_", " ")) for item in plot_rows["case"]]
    colors = ["#b91c1c" if value > 0 else "#15803d" for value in values]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.barh(labels, values, color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("best interior - best endpoint NLL (negative is better)")
    ax.set_title("Endpoint-frontier gaps for Dense and MoE averaging paths")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    x_min = min(values + [0.0])
    x_max = max(values + [0.0])
    pad = max(0.03, (x_max - x_min) * 0.12)
    ax.set_xlim(x_min - pad, x_max + pad)
    for index, value in enumerate(values):
        ha = "left" if value >= 0 else "right"
        offset = pad * 0.18 if value >= 0 else -pad * 0.18
        ax.text(value + offset, index, f"{value:+.4f}", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(args: argparse.Namespace) -> dict[str, Any]:
    dense_curvature = read_json(args.dense_curvature)
    dense_lambda = read_json(args.dense_lambda)
    moe_instruct_coder = read_json(args.moe_instruct_coder)
    moe_base_coder = read_json(args.moe_base_coder)
    moe_complementary = read_json(args.moe_complementary)

    rows = [
        row_from_path_case(
            case="dense_instruct_coder_source_path",
            domain="dense",
            path_kind="source_to_source_interpolation",
            summary=dense_curvature,
            metric="worst",
            tolerance=args.tolerance,
        ),
        row_from_dense_lambda(dense_lambda, tolerance=args.tolerance),
        dense_curvature_row(dense_curvature),
        row_from_path_case(
            case="qwen3_moe_instruct_coder_source_path",
            domain="moe",
            path_kind="source_to_source_interpolation",
            summary=moe_instruct_coder,
            metric="worst",
            tolerance=args.tolerance,
        ),
        row_from_path_case(
            case="qwen3_moe_base_coder_source_path",
            domain="moe",
            path_kind="base_to_source_interpolation",
            summary=moe_base_coder,
            metric="worst",
            tolerance=args.tolerance,
        ),
        row_from_path_case(
            case="qwen3_moe_thinking_coder_complementary_path",
            domain="moe",
            path_kind="specialist_to_specialist_interpolation",
            summary=moe_complementary,
            metric="avg",
            tolerance=args.tolerance,
        ),
    ]
    diagnostics = pd.DataFrame(rows)
    rules = build_rules(args.tolerance)

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "path_diagnostics.csv"
    rules_path = output_dir / "acceptance_rules.json"
    figure_path = output_dir / "connectivity_gaps.png"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    rejected_paths = diagnostics[diagnostics["decision"].astype(str).str.contains("reject")]
    rejected_midpoints = diagnostics[diagnostics["midpoint_decision"].astype(str).str.contains("reject")]
    frontier_wins = diagnostics[
        diagnostics["endpoint_frontier_gap"].notna()
        & (diagnostics["endpoint_frontier_gap"].astype(float) < -args.tolerance)
    ]
    dense_source = diagnostics[diagnostics["case"] == "dense_instruct_coder_source_path"].iloc[0]
    qwen3_source = diagnostics[diagnostics["case"] == "qwen3_moe_instruct_coder_source_path"].iloc[0]
    summary = {
        "schema_version": 1,
        "status": "connectivity_rejects_default_midpoints",
        "case_count": int(len(diagnostics)),
        "path_rejected_count": int(len(rejected_paths)),
        "midpoint_rejected_count": int(len(rejected_midpoints)),
        "endpoint_frontier_win_count": int(len(frontier_wins)),
        "complementarity_observed_count": int(diagnostics["complementarity_observed"].astype(bool).sum()),
        "dense_source_midpoint_gap": fnum(dense_source.get("midpoint_gap")),
        "dense_source_endpoint_gap": fnum(dense_source.get("endpoint_frontier_gap")),
        "qwen3_instruct_coder_gap": fnum(qwen3_source.get("endpoint_frontier_gap")),
        "qwen3_instruct_coder_midpoint_gap": fnum(qwen3_source.get("midpoint_gap")),
        "tolerance": float(args.tolerance),
        "outputs": {
            "path_diagnostics": rel(diagnostics_path),
            "rules": rel(rules_path),
            "figure": rel(figure_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    diagnostics.to_csv(diagnostics_path, index=False)
    rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figure(figure_path, diagnostics)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, diagnostics, rules), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified Dense/MoE connectivity diagnostic.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/average_connectivity_diagnostic"))
    parser.add_argument("--dense-curvature", type=Path, default=Path("results/fp_curvature_law/summary.json"))
    parser.add_argument("--dense-lambda", type=Path, default=Path("results/fp_dense_lambda/summary.json"))
    parser.add_argument("--moe-instruct-coder", type=Path, default=Path("results/fp_moe_barrier/summary.json"))
    parser.add_argument(
        "--moe-base-coder",
        type=Path,
        default=Path("results/fp_moe_forgetting_base_coder/summary.json"),
    )
    parser.add_argument("--moe-complementary", type=Path, default=Path("results/fp_moe_complementary/summary.json"))
    parser.add_argument("--tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote connectivity diagnostic to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; path_rejected={summary['path_rejected_count']}/{summary['case_count']}; "
        f"midpoint_rejected={summary['midpoint_rejected_count']}/{summary['case_count']}; "
        f"frontier_wins={summary['endpoint_frontier_win_count']}"
    )


if __name__ == "__main__":
    main()
