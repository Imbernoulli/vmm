#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METHODS = ["source_qwen3_30b_instruct", "source_qwen3_30b_coder"]
TASK_SCORE_COLUMNS = [
    "task_gsm8k_score",
    "task_mmlu_score",
    "task_safety_score",
    "task_humaneval_compile_score",
]
SCORE_COLUMNS = ["avg_primary_score", "worst_primary_score", *TASK_SCORE_COLUMNS]
RISK_COLUMNS = [
    "total_relative_delta_norm",
    "routed_relative_delta_norm",
    "routed_max_tensor_relative_delta",
    "routed_tensors_gt_1_0",
    "routed_tensors_gt_0_75",
    "routed_tensors_gt_0_65",
    "attention_relative_delta_norm",
    "attention_changed_tensors",
    "router_changed_tensors",
]
TRANSITIONS = [
    {
        "transition": "source_frontier_to_route_guarded",
        "from_method": "source_frontier",
        "to_method": "qwen3_moe_unified_route_guarded_candidate",
        "mechanism": "enter same-shape average: freeze router, route-conditioned expert weights, small attention step",
    },
    {
        "transition": "route_guarded_to_audit_gated",
        "from_method": "qwen3_moe_unified_route_guarded_candidate",
        "to_method": "qwen3_moe_audit_gated_candidate",
        "mechanism": "clip largest routed-expert file-level relative deltas",
    },
    {
        "transition": "audit_gated_to_trust_region",
        "from_method": "qwen3_moe_audit_gated_candidate",
        "to_method": "qwen3_moe_trust_region_candidate",
        "mechanism": "add route/load/category/router-fragility trust-region caps",
    },
    {
        "transition": "trust_region_to_expert_only",
        "from_method": "qwen3_moe_trust_region_candidate",
        "to_method": "qwen3_moe_expert_only_trust_region_candidate",
        "mechanism": "freeze shared attention while keeping trust-region expert rules",
    },
    {
        "transition": "expert_only_to_tail_trimmed",
        "from_method": "qwen3_moe_expert_only_trust_region_candidate",
        "to_method": "qwen3_moe_tail_trimmed_expert_only_candidate",
        "mechanism": "second-stage trim of remaining high-tail routed expert deltas",
    },
    {
        "transition": "tail_trimmed_to_searched_cap_law",
        "from_method": "qwen3_moe_tail_trimmed_expert_only_candidate",
        "to_method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
        "mechanism": "replace hand-built risk penalties with a uniform searched 0.65 routed-expert cap",
    },
    {
        "transition": "searched_cap_law_to_layer_chunk",
        "from_method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
        "to_method": "qwen3_moe_layer_chunk_candidate",
        "mechanism": "apply importance-guided layer/chunk coefficients to high-sensitivity routed experts",
    },
    {
        "transition": "layer_chunk_to_unified_mechanism",
        "from_method": "qwen3_moe_layer_chunk_candidate",
        "to_method": "qwen3_moe_unified_mechanism_candidate",
        "mechanism": "apply router/evidence/geometry-risk optimizer under a retention constraint",
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


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float] | None:
    for column in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = maybe_float(row.get(column))
        if value is not None:
            return column, value
    return None


def read_eval_scores(eval_dir: str | Path) -> dict[str, Any]:
    root = repo_path(eval_dir)
    summary = read_json(root / "summary.json")
    model_summary = read_csv(root / "model_summary.csv")
    metrics = read_csv(root / "metrics.csv")
    model_row = model_summary.iloc[0].to_dict() if not model_summary.empty else {}
    task_scores: dict[str, float] = {}
    if not metrics.empty:
        for _, row in metrics.iterrows():
            task = str(row.get("task", ""))
            primary = primary_metric(row)
            if primary is not None:
                task_scores[task] = primary[1]
    return {
        "eval_status": summary.get("status", "missing"),
        "avg_primary_score": maybe_float(model_row.get("avg_primary_score")),
        "worst_primary_score": maybe_float(model_row.get("worst_primary_score")),
        "task_gsm8k_score": task_scores.get("gsm8k"),
        "task_mmlu_score": task_scores.get("mmlu"),
        "task_safety_score": task_scores.get("safety"),
        "task_humaneval_compile_score": task_scores.get("humaneval_compile"),
    }


def bundle_usable_map(audit_dir: Path) -> dict[str, bool]:
    rows = read_csv(repo_path(audit_dir) / "audit_rows.csv")
    if rows.empty:
        return {}
    return {str(row["method"]): bool(row.get("usable_for_selection", False)) for _, row in rows.iterrows()}


def source_frontier(rows_by_method: dict[str, dict[str, Any]]) -> dict[str, Any]:
    frontier: dict[str, Any] = {"method": "source_frontier", "role": "source_frontier"}
    for column in SCORE_COLUMNS:
        values = [maybe_float(rows_by_method.get(method, {}).get(column)) for method in SOURCE_METHODS]
        values = [value for value in values if value is not None]
        frontier[column] = max(values) if values else None
    frontier["eval_usable"] = all(bool(rows_by_method.get(method, {}).get("eval_usable")) for method in SOURCE_METHODS)
    return frontier


def load_real_rows(gate_dir: Path, audit_dir: Path) -> dict[str, dict[str, Any]]:
    gate = read_csv(repo_path(gate_dir) / "eval_gate_plan.csv")
    usable = bundle_usable_map(audit_dir)
    rows: dict[str, dict[str, Any]] = {}
    for _, row in gate.iterrows():
        method = str(row["method"])
        item = row.to_dict()
        item["eval_usable"] = bool(usable.get(method, False))
        if item["eval_usable"]:
            item.update(read_eval_scores(str(row.get("eval_output_dir", ""))))
        else:
            for column in SCORE_COLUMNS:
                item[column] = None
        rows[method] = item
    rows["source_frontier"] = source_frontier(rows)
    return rows


def delta(to_row: dict[str, Any], from_row: dict[str, Any], column: str) -> float | None:
    to_value = maybe_float(to_row.get(column))
    from_value = maybe_float(from_row.get(column))
    if to_value is None or from_value is None:
        return None
    return to_value - from_value


def transition_status(from_row: dict[str, Any], to_row: dict[str, Any]) -> str:
    if not bool(from_row.get("eval_usable")) or not bool(to_row.get("eval_usable")):
        return "awaiting_eval"
    return "scored"


def build_transition_rows(rows_by_method: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in TRANSITIONS:
        from_row = rows_by_method.get(spec["from_method"], {})
        to_row = rows_by_method.get(spec["to_method"], {})
        status = transition_status(from_row, to_row)
        row: dict[str, Any] = {
            "transition": spec["transition"],
            "from_method": spec["from_method"],
            "to_method": spec["to_method"],
            "mechanism": spec["mechanism"],
            "status": status,
            "from_eval_usable": bool(from_row.get("eval_usable", False)),
            "to_eval_usable": bool(to_row.get("eval_usable", False)),
        }
        for column in SCORE_COLUMNS:
            row[f"from_{column}"] = from_row.get(column)
            row[f"to_{column}"] = to_row.get(column)
            row[f"delta_{column}"] = delta(to_row, from_row, column)
        for column in RISK_COLUMNS:
            row[f"from_{column}"] = from_row.get(column)
            row[f"to_{column}"] = to_row.get(column)
            row[f"delta_{column}"] = delta(to_row, from_row, column)
        rows.append(row)
    return pd.DataFrame(rows)


def infer_effect(row: pd.Series) -> str:
    if row.get("status") != "scored":
        risk_terms = []
        for column in ("routed_tensors_gt_0_75", "routed_tensors_gt_0_65", "attention_changed_tensors"):
            value = maybe_float(row.get(f"delta_{column}"))
            if value is not None and value != 0:
                risk_terms.append(f"{column}:{value:+.0f}")
        return "awaiting_downstream_eval" if not risk_terms else "structural_change_only:" + ",".join(risk_terms)
    avg_delta = maybe_float(row.get("delta_avg_primary_score")) or 0.0
    worst_delta = maybe_float(row.get("delta_worst_primary_score")) or 0.0
    if avg_delta > 0 and worst_delta >= 0:
        return "improves_avg_without_worst_regression"
    if worst_delta > 0 and avg_delta >= 0:
        return "improves_worst_without_avg_regression"
    if avg_delta < 0 or worst_delta < 0:
        return "downstream_regression"
    return "neutral"


def summarize_effects(transitions: pd.DataFrame, output_dir: Path, *, smoke_case: str | None = None) -> dict[str, Any]:
    scored = transitions[transitions["status"] == "scored"] if "status" in transitions else pd.DataFrame()
    improved = scored[
        (scored["delta_avg_primary_score"].fillna(0.0) > 0.0)
        | (scored["delta_worst_primary_score"].fillna(0.0) > 0.0)
    ] if not scored.empty else pd.DataFrame()
    regressions = scored[
        (scored["delta_avg_primary_score"].fillna(0.0) < 0.0)
        | (scored["delta_worst_primary_score"].fillna(0.0) < 0.0)
    ] if not scored.empty else pd.DataFrame()
    if len(scored) == len(transitions) and len(transitions) > 0:
        status = "complete"
    elif not scored.empty:
        status = "partial"
    else:
        status = "awaiting_eval"
    best_avg_row = None if scored.empty else scored.sort_values("delta_avg_primary_score", ascending=False).iloc[0]
    best_worst_row = None if scored.empty else scored.sort_values("delta_worst_primary_score", ascending=False).iloc[0]
    output_dir = repo_path(output_dir)
    return {
        "schema_version": 1,
        "status": status,
        "smoke_case": smoke_case,
        "transition_count": int(len(transitions)),
        "scored_transition_count": int(len(scored)),
        "improving_transition_count": int(len(improved)),
        "regressing_transition_count": int(len(regressions)),
        "best_avg_transition": None if best_avg_row is None else best_avg_row.get("transition"),
        "best_avg_delta": None if best_avg_row is None else maybe_float(best_avg_row.get("delta_avg_primary_score")),
        "best_worst_transition": None if best_worst_row is None else best_worst_row.get("transition"),
        "best_worst_delta": None if best_worst_row is None else maybe_float(best_worst_row.get("delta_worst_primary_score")),
        "outputs": {
            "transition_effects": rel(output_dir / "transition_effects.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }


def build_report(summary: dict[str, Any], transitions: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        value = clean_value(value)
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "# Qwen3 MoE Mechanism Effect Attribution",
        "",
        "这个 attribution 把 Qwen3 MoE average 的机制链条拆成相邻对比：source frontier、route-guarded、audit-gated、trust-region、expert-only、tail-trimmed、searched cap-law 和 unified alias。只有通过 eval bundle audit 的 vLLM 结果才会进入 downstream score delta。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scored transitions: `{summary['scored_transition_count']}/{summary['transition_count']}`",
        f"- Improving transitions: `{summary['improving_transition_count']}`",
        f"- Regressing transitions: `{summary['regressing_transition_count']}`",
        f"- Best avg transition: `{summary['best_avg_transition']}` (`{summary['best_avg_delta']}`)",
        f"- Best worst transition: `{summary['best_worst_transition']}` (`{summary['best_worst_delta']}`)",
        "",
        "| transition | status | avg delta | worst delta | routed >0.75 delta | attention changed delta | effect |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in transitions.iterrows():
        lines.append(
            f"| `{row['transition']}` | `{row['status']}` | "
            f"{fmt(row.get('delta_avg_primary_score'))} | {fmt(row.get('delta_worst_primary_score'))} | "
            f"{fmt(row.get('delta_routed_tensors_gt_0_75'))} | {fmt(row.get('delta_attention_changed_tensors'))} | "
            f"`{row.get('effect', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['transition_effects']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(transitions: pd.DataFrame, output_dir: Path, *, smoke_case: str | None = None) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions = transitions.copy()
    transitions["effect"] = [infer_effect(row) for _, row in transitions.iterrows()]
    transition_path = output_dir / "transition_effects.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    transitions.to_csv(transition_path, index=False)
    summary = summarize_effects(transitions, output_dir, smoke_case=smoke_case)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, transitions), encoding="utf-8")
    return summary


def synthetic_rows(case: str) -> dict[str, dict[str, Any]]:
    methods = [*SOURCE_METHODS, *[spec["to_method"] for spec in TRANSITIONS]]
    base_scores = {
        "source_qwen3_30b_instruct": (0.62, 0.40, [0.55, 0.60, 0.70, 0.35]),
        "source_qwen3_30b_coder": (0.58, 0.42, [0.50, 0.55, 0.45, 0.60]),
        "qwen3_moe_unified_route_guarded_candidate": (0.63, 0.43, [0.56, 0.60, 0.67, 0.53]),
        "qwen3_moe_audit_gated_candidate": (0.63, 0.43, [0.56, 0.59, 0.67, 0.55]),
        "qwen3_moe_trust_region_candidate": (0.65, 0.46, [0.58, 0.61, 0.68, 0.57]),
        "qwen3_moe_expert_only_trust_region_candidate": (0.66, 0.47, [0.59, 0.62, 0.69, 0.58]),
        "qwen3_moe_tail_trimmed_expert_only_candidate": (0.67, 0.49, [0.60, 0.63, 0.70, 0.59]),
        "qwen3_moe_searched_no_gt065_max_retention_candidate": (0.68, 0.50, [0.61, 0.64, 0.70, 0.60]),
        "qwen3_moe_layer_chunk_candidate": (0.69, 0.51, [0.62, 0.65, 0.70, 0.61]),
        "qwen3_moe_unified_mechanism_candidate": (0.70, 0.52, [0.63, 0.66, 0.70, 0.62]),
    }
    if case == "regression":
        base_scores["qwen3_moe_expert_only_trust_region_candidate"] = (0.60, 0.38, [0.52, 0.54, 0.62, 0.50])
    elif case == "partial":
        pass
    elif case != "complete":
        raise ValueError(f"Unknown smoke case: {case}")
    risk = {
        "qwen3_moe_unified_route_guarded_candidate": (0.286, 839, 288, 0),
        "qwen3_moe_audit_gated_candidate": (0.264, 164, 288, 0),
        "qwen3_moe_trust_region_candidate": (0.249, 14, 288, 0),
        "qwen3_moe_expert_only_trust_region_candidate": (0.246, 14, 0, 0),
        "qwen3_moe_tail_trimmed_expert_only_candidate": (0.243, 0, 0, 0),
        "qwen3_moe_searched_no_gt065_max_retention_candidate": (0.248, 0, 0, 0),
        "qwen3_moe_layer_chunk_candidate": (0.243, 0, 0, 0),
        "qwen3_moe_unified_mechanism_candidate": (0.240, 0, 0, 0),
    }
    rows: dict[str, dict[str, Any]] = {}
    for method in methods:
        avg, worst, task_values = base_scores[method]
        if case == "partial" and method in {
            "qwen3_moe_tail_trimmed_expert_only_candidate",
            "qwen3_moe_searched_no_gt065_max_retention_candidate",
            "qwen3_moe_layer_chunk_candidate",
            "qwen3_moe_unified_mechanism_candidate",
        }:
            eval_usable = False
            avg = worst = None
            task_values = [None, None, None, None]
        else:
            eval_usable = True
        row: dict[str, Any] = {
            "method": method,
            "role": "source" if method in SOURCE_METHODS else "candidate",
            "eval_usable": eval_usable,
            "avg_primary_score": avg,
            "worst_primary_score": worst,
        }
        for column, value in zip(TASK_SCORE_COLUMNS, task_values, strict=True):
            row[column] = value
        total_norm, routed_gt075, attention_changed, router_changed = risk.get(method, (0, 0, 0, 0))
        row.update(
            {
                "total_relative_delta_norm": total_norm,
                "routed_relative_delta_norm": total_norm,
                "routed_max_tensor_relative_delta": 0.65 if routed_gt075 == 0 else 0.75,
                "routed_tensors_gt_1_0": 0,
                "routed_tensors_gt_0_75": routed_gt075,
                "routed_tensors_gt_0_65": routed_gt075,
                "attention_relative_delta_norm": 0.189 if attention_changed else 0.0,
                "attention_changed_tensors": attention_changed,
                "router_changed_tensors": router_changed,
            }
        )
        rows[method] = row
    rows["source_frontier"] = source_frontier(rows)
    return rows


def run_smoke_matrix(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "complete": ("complete", 8),
        "partial": ("partial", 4),
        "regression": ("complete", 8),
    }
    rows = []
    for case, (expected_status, expected_scored) in expected.items():
        transitions = build_transition_rows(synthetic_rows(case))
        summary = summarize_effects(transitions, repo_path(args.output_dir) / case, smoke_case=case)
        regression_count = 0
        transitions = transitions.copy()
        transitions["effect"] = [infer_effect(row) for _, row in transitions.iterrows()]
        regression_count = int((transitions["effect"] == "downstream_regression").sum())
        rows.append(
            {
                "case": case,
                "status": summary["status"],
                "expected_status": expected_status,
                "scored_transition_count": summary["scored_transition_count"],
                "expected_scored_transition_count": expected_scored,
                "regression_count": regression_count,
                "passed": summary["status"] == expected_status
                and summary["scored_transition_count"] == expected_scored
                and (case != "regression" or regression_count > 0),
            }
        )
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(rows)
    matrix_path = output_dir / "attribution_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    passed = int(matrix["passed"].astype(bool).sum())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed == len(matrix) else "failed",
        "case_count": int(len(matrix)),
        "passed_case_count": passed,
        "failed_case_count": int(len(matrix) - passed),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
        "cases": matrix.to_dict(orient="records"),
    }
    lines = [
        "# Qwen3 MoE Mechanism Effect Attribution Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{passed}/{len(matrix)}`",
        "",
        "| case | status | scored | regressions | passed |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['status']}` | {int(row['scored_transition_count'])} | "
            f"{int(row['regression_count'])} | `{bool(row['passed'])}` |"
        )
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribute Qwen3 MoE downstream score changes to mechanism steps.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_mechanism_effect_attribution"))
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        summary = run_smoke_matrix(args)
        print(f"Wrote Qwen3 MoE mechanism attribution smoke to {repo_path(args.output_dir).resolve()}")
        print(f"Status: {summary['status']}; cases {summary['passed_case_count']}/{summary['case_count']}")
        return
    rows_by_method = load_real_rows(args.gate_dir, args.audit_dir)
    transitions = build_transition_rows(rows_by_method)
    summary = write_outputs(transitions, args.output_dir)
    print(f"Wrote Qwen3 MoE mechanism effect attribution to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; scored={summary['scored_transition_count']}/{summary['transition_count']}")


if __name__ == "__main__":
    main()
