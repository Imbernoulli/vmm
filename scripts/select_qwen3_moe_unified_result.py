#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METHODS = ["source_qwen3_30b_instruct", "source_qwen3_30b_coder"]
UNIFIED_METHOD = "qwen3_moe_unified_mechanism_candidate"
TASK_SCORE_COLUMNS = [
    "task_gsm8k_score",
    "task_mmlu_score",
    "task_safety_score",
    "task_humaneval_compile_score",
]
SCORE_COLUMNS = ["avg_primary_score", "worst_primary_score", *TASK_SCORE_COLUMNS]


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


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float] | None:
    for key in ("strict_exact", "accuracy", "policy_accuracy", "compile_rate"):
        value = maybe_float(row.get(key))
        if value is not None:
            return key, value
    return None


def read_eval_state(eval_dir: str | Path) -> dict[str, Any]:
    root = repo_path(eval_dir)
    summary = read_json_if_exists(root / "summary.json")
    model_summary = read_csv_if_exists(root / "model_summary.csv")
    metrics = read_csv_if_exists(root / "metrics.csv")
    status = str(summary.get("status", "not_run"))
    if not model_summary.empty:
        model_row = model_summary.iloc[0].to_dict()
    else:
        rows = summary.get("model_summary") or []
        model_row = rows[0] if rows else {}

    task_scores: dict[str, float] = {}
    if not metrics.empty:
        for _, metric_row in metrics.iterrows():
            task = str(metric_row.get("task", "unknown"))
            primary = primary_metric(metric_row)
            if primary is not None:
                _, score = primary
                task_scores[task] = score

    state = {
        "eval_status": status,
        "eval_completed": status == "complete",
        "avg_primary_score": maybe_float(model_row.get("avg_primary_score")),
        "worst_primary_score": maybe_float(model_row.get("worst_primary_score")),
        "task_gsm8k_score": task_scores.get("gsm8k"),
        "task_mmlu_score": task_scores.get("mmlu"),
        "task_safety_score": task_scores.get("safety"),
        "task_humaneval_compile_score": task_scores.get("humaneval_compile"),
        "eval_output_dir": rel(root),
    }
    return state


def load_real_rows(gate_dir: Path) -> tuple[pd.DataFrame, str | None]:
    gate_dir = repo_path(gate_dir)
    gate = read_csv_if_exists(gate_dir / "eval_gate_plan.csv")
    tests = read_csv_if_exists(gate_dir / "mechanism_tests.csv")
    if gate.empty:
        return pd.DataFrame(), None
    keep = gate[gate["method"].isin([*SOURCE_METHODS, UNIFIED_METHOD])].copy()
    rows = []
    for _, row in keep.iterrows():
        item = row.to_dict()
        eval_state = read_eval_state(item.get("eval_output_dir", ""))
        item.update(eval_state)
        rows.append(item)
    alias_status = None
    if not tests.empty:
        alias_rows = tests[tests["test"] == "unified_rule_alias_validation"]
        if not alias_rows.empty:
            alias_status = str(alias_rows.iloc[0].get("current_status"))
    return pd.DataFrame(rows), alias_status


def score_pairs(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any]) -> list[tuple[float, float]]:
    pairs = []
    for column in SCORE_COLUMNS:
        left_value = maybe_float(left.get(column))
        right_value = maybe_float(right.get(column))
        if left_value is not None and right_value is not None:
            pairs.append((left_value, right_value))
    return pairs


def dominates(left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any], eps: float = 1e-9) -> bool:
    pairs = score_pairs(left, right)
    if not pairs:
        return False
    return all(left_value >= right_value - eps for left_value, right_value in pairs) and any(
        left_value > right_value + eps for left_value, right_value in pairs
    )


def best_source(sources: pd.DataFrame) -> dict[str, Any]:
    if sources.empty:
        return {}
    return sources.sort_values(["avg_primary_score", "worst_primary_score"], ascending=False).iloc[0].to_dict()


def task_regressions(sources: pd.DataFrame, unified: pd.Series, tolerance: float) -> list[str]:
    regressions = []
    for column in TASK_SCORE_COLUMNS:
        unified_value = maybe_float(unified.get(column))
        source_values = [maybe_float(value) for value in sources[column].tolist() if maybe_float(value) is not None]
        if unified_value is None or len(source_values) < len(SOURCE_METHODS):
            continue
        if unified_value < min(source_values) - tolerance:
            regressions.append(column)
    return regressions


def build_selection(
    rows: pd.DataFrame,
    *,
    alias_status: str | None,
    min_avg_gain: float,
    min_worst_gain: float,
    task_regression_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty:
        selection = {
            "status": "awaiting_eval_gate",
            "selected_method": None,
            "reason": "qwen3_moe_mechanism_eval_gate/eval_gate_plan.csv is missing or empty.",
        }
        return pd.DataFrame(), selection

    rows = rows.copy()
    sources = rows[rows["method"].isin(SOURCE_METHODS)].copy()
    unified_rows = rows[rows["method"] == UNIFIED_METHOD].copy()
    unified = pd.Series(dtype=object) if unified_rows.empty else unified_rows.iloc[0]
    sources_complete = bool(len(sources) == len(SOURCE_METHODS) and sources["eval_completed"].astype(bool).all())
    unified_completed = bool(not unified.empty and bool(unified.get("eval_completed")))
    unified_audit_passed = bool(unified.get("audit_passed", False)) if not unified.empty else False
    alias_supported = alias_status in {None, "mechanism_supported"}
    best_source_row = best_source(sources)

    dominated_by = []
    if unified_completed and sources_complete:
        for _, source in sources.iterrows():
            if dominates(source, unified):
                dominated_by.append(str(source["method"]))
    regressions = task_regressions(sources, unified, task_regression_tolerance) if unified_completed else []
    best_avg = maybe_float(best_source_row.get("avg_primary_score"))
    best_worst = maybe_float(best_source_row.get("worst_primary_score"))
    unified_avg = maybe_float(unified.get("avg_primary_score")) if not unified.empty else None
    unified_worst = maybe_float(unified.get("worst_primary_score")) if not unified.empty else None
    avg_gain = None if unified_avg is None or best_avg is None else unified_avg - best_avg
    worst_gain = None if unified_worst is None or best_worst is None else unified_worst - best_worst
    has_frontier_gain = bool(
        (avg_gain is not None and avg_gain >= min_avg_gain)
        or (worst_gain is not None and worst_gain >= min_worst_gain)
    )

    table_rows = []
    for _, row in rows.iterrows():
        is_unified = str(row["method"]) == UNIFIED_METHOD
        table_rows.append(
            {
                "method": row.get("method"),
                "role": row.get("role"),
                "eval_completed": bool(row.get("eval_completed")),
                "audit_passed": bool(row.get("audit_passed", False)),
                "avg_primary_score": row.get("avg_primary_score"),
                "worst_primary_score": row.get("worst_primary_score"),
                "task_gsm8k_score": row.get("task_gsm8k_score"),
                "task_mmlu_score": row.get("task_mmlu_score"),
                "task_safety_score": row.get("task_safety_score"),
                "task_humaneval_compile_score": row.get("task_humaneval_compile_score"),
                "dominated_by_source": ",".join(dominated_by) if is_unified else "",
                "task_regression_columns": ",".join(regressions) if is_unified else "",
                "selection_eligible": bool(
                    is_unified
                    and sources_complete
                    and unified_completed
                    and unified_audit_passed
                    and alias_supported
                    and not dominated_by
                    and not regressions
                    and has_frontier_gain
                ),
            }
        )
    selection_table = pd.DataFrame(table_rows)

    if not sources_complete:
        selection = {
            "status": "awaiting_source_eval",
            "selected_method": None,
            "reason": "Both Qwen3 source endpoints must complete matched vLLM downstream eval before accepting an average.",
        }
    elif not unified_completed:
        selection = {
            "status": "awaiting_unified_eval",
            "selected_method": None,
            "reason": "Unified mechanism candidate is in the vLLM gate but has not completed downstream eval.",
        }
    elif not unified_audit_passed:
        selection = {
            "status": "keep_source_endpoint",
            "selected_method": best_source_row.get("method"),
            "reason": "unified_audit_failed",
        }
    elif not alias_supported:
        selection = {
            "status": "keep_source_endpoint",
            "selected_method": best_source_row.get("method"),
            "reason": "unified_alias_not_supported",
        }
    elif regressions:
        selection = {
            "status": "keep_source_endpoint",
            "selected_method": best_source_row.get("method"),
            "reason": "task_score_regression",
            "regression_columns": regressions,
        }
    elif dominated_by:
        selection = {
            "status": "keep_source_endpoint",
            "selected_method": best_source_row.get("method"),
            "reason": "source_endpoint_dominates",
            "dominated_by_source": dominated_by,
        }
    elif has_frontier_gain:
        selection = {
            "status": "select_unified_candidate",
            "selected_method": UNIFIED_METHOD,
            "reason": "unified_improves_source_frontier",
        }
    else:
        selection = {
            "status": "keep_source_endpoint",
            "selected_method": best_source_row.get("method"),
            "reason": "no_avg_or_worst_gain",
        }

    selection.update(
        {
            "sources_complete": sources_complete,
            "unified_completed": unified_completed,
            "unified_audit_passed": unified_audit_passed,
            "alias_status": alias_status,
            "best_source_by_avg": best_source_row.get("method"),
            "avg_gain_vs_best_source": avg_gain,
            "worst_gain_vs_best_source": worst_gain,
            "source_count": int(len(sources)),
            "score_columns": SCORE_COLUMNS,
        }
    )
    return selection_table, selection


def smoke_rows(case: str) -> tuple[pd.DataFrame, str]:
    base = [
        {
            "method": "source_qwen3_30b_instruct",
            "role": "source",
            "eval_completed": True,
            "audit_passed": True,
            "avg_primary_score": 0.65,
            "worst_primary_score": 0.40,
            "task_gsm8k_score": 0.60,
            "task_mmlu_score": 0.60,
            "task_safety_score": 0.50,
            "task_humaneval_compile_score": 0.30,
        },
        {
            "method": "source_qwen3_30b_coder",
            "role": "source",
            "eval_completed": True,
            "audit_passed": True,
            "avg_primary_score": 0.62,
            "worst_primary_score": 0.35,
            "task_gsm8k_score": 0.55,
            "task_mmlu_score": 0.55,
            "task_safety_score": 0.45,
            "task_humaneval_compile_score": 0.35,
        },
    ]
    unified = {
        "method": UNIFIED_METHOD,
        "role": "candidate",
        "eval_completed": True,
        "audit_passed": True,
        "avg_primary_score": 0.70,
        "worst_primary_score": 0.55,
        "task_gsm8k_score": 0.62,
        "task_mmlu_score": 0.61,
        "task_safety_score": 0.52,
        "task_humaneval_compile_score": 0.36,
    }
    if case == "source_dominance":
        base[0].update(
            avg_primary_score=0.80,
            worst_primary_score=0.70,
            task_gsm8k_score=0.80,
            task_mmlu_score=0.80,
            task_safety_score=0.80,
            task_humaneval_compile_score=0.80,
        )
        unified.update(
            avg_primary_score=0.70,
            worst_primary_score=0.60,
            task_gsm8k_score=0.70,
            task_mmlu_score=0.70,
            task_safety_score=0.70,
            task_humaneval_compile_score=0.70,
        )
    elif case == "task_regression":
        unified.update(
            avg_primary_score=0.72,
            worst_primary_score=0.50,
            task_gsm8k_score=0.65,
            task_mmlu_score=0.62,
            task_safety_score=0.40,
            task_humaneval_compile_score=0.36,
        )
    elif case == "no_gain":
        unified.update(
            avg_primary_score=0.64,
            worst_primary_score=0.39,
            task_gsm8k_score=0.58,
            task_mmlu_score=0.58,
            task_safety_score=0.50,
            task_humaneval_compile_score=0.35,
        )
    elif case != "candidate_win":
        raise ValueError(f"Unknown smoke case: {case}")
    return pd.DataFrame([*base, unified]), "mechanism_supported"


def build_report(summary: dict[str, Any], selection_table: pd.DataFrame) -> str:
    selection = summary["current_selection"]
    lines = [
        "# Qwen3 MoE Unified Result Selector",
        "",
        "这个 selector 只回答一个问题：source endpoints 和 unified mechanism candidate 完成同一套 vLLM 下游任务后，是否接受这个 Average，还是回退到同结构 source endpoint。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selected: `{selection.get('selected_method')}`",
        f"- Reason: `{selection.get('reason')}`",
        f"- Source eval complete: `{selection.get('sources_complete')}`",
        f"- Unified eval complete: `{selection.get('unified_completed')}`",
        f"- Unified audit passed: `{selection.get('unified_audit_passed')}`",
        f"- Alias status: `{selection.get('alias_status')}`",
        "",
        "## Selection Table",
        "",
        "| method | eval | audit | avg | worst | gsm8k | mmlu | safety | humaneval | dominated | regression | eligible |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for _, row in selection_table.iterrows():
        lines.append(
            f"| `{row['method']}` | `{bool(row['eval_completed'])}` | `{bool(row['audit_passed'])}` | "
            f"{row.get('avg_primary_score', '')} | {row.get('worst_primary_score', '')} | "
            f"{row.get('task_gsm8k_score', '')} | {row.get('task_mmlu_score', '')} | "
            f"{row.get('task_safety_score', '')} | {row.get('task_humaneval_compile_score', '')} | "
            f"`{row.get('dominated_by_source', '')}` | `{row.get('task_regression_columns', '')}` | "
            f"`{bool(row.get('selection_eligible'))}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['selection_table']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['decision_rules']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    selection_table: pd.DataFrame,
    selection: dict[str, Any],
    *,
    smoke_case: str | None,
) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "selection_table.csv"
    rules_path = output_dir / "decision_rules.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    selection_table.to_csv(table_path, index=False)
    rules = {
        "schema_version": 1,
        "target_contract": "same-shape Qwen3 MoE output; no ensemble, no added experts, no architecture change",
        "required_controls": SOURCE_METHODS,
        "candidate": UNIFIED_METHOD,
        "acceptance": [
            "both source endpoints complete matched vLLM eval",
            "unified candidate completes matched vLLM eval",
            "unified candidate audit passes",
            "unified alias validation is supported when present",
            "no source endpoint dominates the unified candidate",
            "no task score regresses below both source endpoints",
            "unified improves avg or worst source frontier",
        ],
    }
    summary = {
        "schema_version": 1,
        "status": selection["status"],
        "smoke_case": smoke_case,
        "current_selection": selection,
        "candidate_count": int(len(selection_table)),
        "outputs": {
            "selection_table": rel(table_path),
            "decision_rules": rel(rules_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    rules_path.write_text(json.dumps(json_safe(rules), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, selection_table), encoding="utf-8")
    return summary


def run_smoke_matrix(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "candidate_win": ("select_unified_candidate", "unified_improves_source_frontier"),
        "source_dominance": ("keep_source_endpoint", "source_endpoint_dominates"),
        "task_regression": ("keep_source_endpoint", "task_score_regression"),
        "no_gain": ("keep_source_endpoint", "no_avg_or_worst_gain"),
    }
    rows = []
    for case, (expected_status, expected_reason) in expected.items():
        data, alias_status = smoke_rows(case)
        _, selection = build_selection(
            data,
            alias_status=alias_status,
            min_avg_gain=args.min_avg_gain,
            min_worst_gain=args.min_worst_gain,
            task_regression_tolerance=args.task_regression_tolerance,
        )
        rows.append(
            {
                "case": case,
                "status": selection["status"],
                "expected_status": expected_status,
                "reason": selection.get("reason"),
                "expected_reason": expected_reason,
                "passed": selection["status"] == expected_status and selection.get("reason") == expected_reason,
                "selected_method": selection.get("selected_method"),
            }
        )
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(rows)
    matrix_path = output_dir / "selector_matrix.csv"
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
        "# Qwen3 MoE Unified Result Selector Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{passed}/{len(matrix)}`",
        "",
        "| case | status | reason | selected | passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['status']}` | `{row['reason']}` | "
            f"`{row['selected_method']}` | `{bool(row['passed'])}` |"
        )
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select or reject the Qwen3 MoE unified mechanism candidate after vLLM eval.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_unified_result_selection"))
    parser.add_argument("--smoke-case", choices=["candidate_win", "source_dominance", "task_regression", "no_gain"])
    parser.add_argument("--smoke-matrix", action="store_true")
    parser.add_argument("--min-avg-gain", type=float, default=1e-9)
    parser.add_argument("--min-worst-gain", type=float, default=1e-9)
    parser.add_argument("--task-regression-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        summary = run_smoke_matrix(args)
        print(f"Wrote smoke matrix to {repo_path(args.output_dir).resolve()}")
        print(f"Status: {summary['status']}; cases {summary['passed_case_count']}/{summary['case_count']}")
        return
    if args.smoke_case:
        rows, alias_status = smoke_rows(args.smoke_case)
    else:
        rows, alias_status = load_real_rows(args.gate_dir)
    selection_table, selection = build_selection(
        rows,
        alias_status=alias_status,
        min_avg_gain=args.min_avg_gain,
        min_worst_gain=args.min_worst_gain,
        task_regression_tolerance=args.task_regression_tolerance,
    )
    summary = write_outputs(args.output_dir, selection_table, selection, smoke_case=args.smoke_case)
    print(f"Wrote Qwen3 MoE unified result selection to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; selected={summary['current_selection'].get('selected_method')}")


if __name__ == "__main__":
    main()
