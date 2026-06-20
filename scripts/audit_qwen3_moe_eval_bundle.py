#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ["summary.json", "eval_plan.csv", "metrics.csv", "model_summary.csv", "predictions.csv"]
PRIMARY_SCORE_COLUMNS = ["strict_exact", "accuracy", "policy_accuracy", "compile_rate"]


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


def parse_tasks(raw: Any) -> list[str]:
    raw = clean_value(raw)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def unique_non_null(values: pd.Series) -> set[str]:
    return {str(value) for value in values.tolist() if clean_value(value) is not None}


def primary_score(row: pd.Series) -> float | None:
    for column in PRIMARY_SCORE_COLUMNS:
        value = clean_value(row.get(column))
        if value is not None:
            return float(value)
    return None


def expected_model_id(row: pd.Series) -> str:
    return str(row.get("served_model_id") or f"candidate_{row['method']}")


def audit_eval_row(row: pd.Series, *, min_examples: int | None = None) -> dict[str, Any]:
    method = str(row["method"])
    expected_model = expected_model_id(row)
    expected_tasks = parse_tasks(row.get("tasks"))
    expected_examples = int(row.get("max_examples") or min_examples or 0)
    if min_examples is not None:
        expected_examples = max(expected_examples, min_examples)
    output_dir = repo_path(row.get("eval_output_dir", ""))
    files_present = {name: (output_dir / name).exists() for name in REQUIRED_FILES}
    missing_files = [name for name, present in files_present.items() if not present]
    summary = read_json(output_dir / "summary.json")
    eval_plan = read_csv(output_dir / "eval_plan.csv")
    metrics = read_csv(output_dir / "metrics.csv")
    model_summary = read_csv(output_dir / "model_summary.csv")
    predictions = read_csv(output_dir / "predictions.csv")

    issues: list[str] = []
    if not output_dir.exists():
        issues.append("eval_output_dir_missing")
    if missing_files:
        issues.append("missing_files:" + ",".join(missing_files))

    summary_status = str(summary.get("status", "missing"))
    if summary_status != "complete":
        issues.append(f"summary_status:{summary_status}")

    summary_models = set(str(model) for model in summary.get("models", []) if model)
    if summary.get("model"):
        summary_models.add(str(summary["model"]))
    plan_models = unique_non_null(eval_plan["served_model_id"]) if "served_model_id" in eval_plan else set()
    metric_models = unique_non_null(metrics["model"]) if "model" in metrics else set()
    summary_table_models = unique_non_null(model_summary["model"]) if "model" in model_summary else set()
    prediction_models = unique_non_null(predictions["model"]) if "model" in predictions else set()
    observed_models = summary_models | plan_models | metric_models | summary_table_models | prediction_models
    model_match = observed_models == {expected_model} if observed_models else False
    if not model_match:
        issues.append("model_mismatch")

    metric_tasks = unique_non_null(metrics["task"]) if "task" in metrics else set()
    task_coverage = set(expected_tasks).issubset(metric_tasks) and len(metric_tasks) == len(expected_tasks)
    if not task_coverage:
        issues.append("task_coverage_mismatch")

    examples_ok = True
    low_example_tasks = []
    if expected_examples > 0 and not metrics.empty and "examples" in metrics:
        for _, metric_row in metrics.iterrows():
            task = str(metric_row.get("task"))
            examples = clean_value(metric_row.get("examples"))
            if examples is None or int(examples) < expected_examples:
                examples_ok = False
                low_example_tasks.append(task)
    elif expected_tasks:
        examples_ok = False
    if not examples_ok:
        issues.append("insufficient_examples:" + ",".join(low_example_tasks or expected_tasks))

    prediction_counts_ok = True
    low_prediction_tasks = []
    if not predictions.empty and "task" in predictions and expected_examples > 0:
        counts = predictions.groupby("task").size().to_dict()
        for task in expected_tasks:
            if int(counts.get(task, 0)) < expected_examples:
                prediction_counts_ok = False
                low_prediction_tasks.append(task)
    elif expected_tasks:
        prediction_counts_ok = False
    if not prediction_counts_ok:
        issues.append("insufficient_predictions:" + ",".join(low_prediction_tasks or expected_tasks))

    primary_scores_present = True
    missing_score_tasks = []
    if not metrics.empty:
        for _, metric_row in metrics.iterrows():
            task = str(metric_row.get("task"))
            if task in expected_tasks and primary_score(metric_row) is None:
                primary_scores_present = False
                missing_score_tasks.append(task)
    elif expected_tasks:
        primary_scores_present = False
    if not primary_scores_present:
        issues.append("missing_primary_scores:" + ",".join(missing_score_tasks or expected_tasks))

    usable = (
        summary_status == "complete"
        and not missing_files
        and model_match
        and task_coverage
        and examples_ok
        and prediction_counts_ok
        and primary_scores_present
    )
    return {
        "method": method,
        "role": row.get("role"),
        "expected_model": expected_model,
        "eval_output_dir": rel(output_dir),
        "summary_status": summary_status,
        "required_task_count": len(expected_tasks),
        "observed_task_count": len(metric_tasks),
        "required_examples_per_task": expected_examples,
        "model_match": model_match,
        "task_coverage": task_coverage,
        "example_coverage": examples_ok,
        "prediction_coverage": prediction_counts_ok,
        "primary_scores_present": primary_scores_present,
        "usable_for_selection": usable,
        "issues": ";".join(issues) if issues else "",
    }


def audit_gate(gate_dir: Path, *, min_examples: int | None = None) -> pd.DataFrame:
    plan = read_csv(repo_path(gate_dir) / "eval_gate_plan.csv")
    if plan.empty:
        raise ValueError(f"Missing or empty eval gate plan: {repo_path(gate_dir) / 'eval_gate_plan.csv'}")
    rows = [audit_eval_row(row, min_examples=min_examples) for _, row in plan.iterrows()]
    return pd.DataFrame(rows)


def summarize_audit(rows: pd.DataFrame, output_dir: Path, *, smoke_case: str | None = None) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    usable = rows[rows["usable_for_selection"].astype(bool)] if "usable_for_selection" in rows else pd.DataFrame()
    complete = rows[rows["summary_status"] == "complete"] if "summary_status" in rows else pd.DataFrame()
    invalid_complete = complete[~complete["usable_for_selection"].astype(bool)] if not complete.empty else pd.DataFrame()
    sources = rows[rows["role"] == "source"] if "role" in rows else pd.DataFrame()
    candidates = rows[rows["role"] == "candidate"] if "role" in rows else pd.DataFrame()
    unified = rows[rows["method"] == "qwen3_moe_unified_mechanism_candidate"] if "method" in rows else pd.DataFrame()
    if not invalid_complete.empty:
        status = "invalid_bundle"
    elif len(usable) == len(rows) and len(rows) > 0:
        status = "passed"
    elif len(usable) > 0:
        status = "partial"
    else:
        status = "awaiting_eval"
    return {
        "schema_version": 1,
        "status": status,
        "smoke_case": smoke_case,
        "method_count": int(len(rows)),
        "complete_eval_count": int(len(complete)),
        "usable_for_selection_count": int(len(usable)),
        "invalid_complete_count": int(len(invalid_complete)),
        "source_usable_count": int(sources["usable_for_selection"].astype(bool).sum()) if not sources.empty else 0,
        "source_count": int(len(sources)),
        "candidate_usable_count": (
            int(candidates["usable_for_selection"].astype(bool).sum()) if not candidates.empty else 0
        ),
        "candidate_count": int(len(candidates)),
        "unified_usable": bool(not unified.empty and bool(unified.iloc[0]["usable_for_selection"])),
        "outputs": {
            "audit_rows": rel(output_dir / "audit_rows.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }


def build_report(summary: dict[str, Any], rows: pd.DataFrame) -> str:
    def issue_preview(raw: Any, limit: int = 2) -> str:
        raw = str(raw or "")
        if not raw:
            return ""
        parts = raw.split(";")
        suffix = "" if len(parts) <= limit else f"; +{len(parts) - limit} more"
        return ";".join(parts[:limit]) + suffix

    lines = [
        "# Qwen3 MoE vLLM Eval Bundle Audit",
        "",
        "这个 audit 检查远端 vLLM eval 落盘结果是否能被 downstream selector 使用，防止旧模型名、缺任务、样本数不足或缺 predictions 的结果混入 Average 选择。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Usable for selection: `{summary['usable_for_selection_count']}/{summary['method_count']}`",
        f"- Source usable: `{summary['source_usable_count']}/{summary['source_count']}`",
        f"- Candidate usable: `{summary['candidate_usable_count']}/{summary['candidate_count']}`",
        f"- Unified usable: `{summary['unified_usable']}`",
        "",
        "| method | status | model | tasks | examples | predictions | usable | issue preview |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| `{row['method']}` | `{row['summary_status']}` | `{bool(row['model_match'])}` | "
            f"`{bool(row['task_coverage'])}` | `{bool(row['example_coverage'])}` | "
            f"`{bool(row['prediction_coverage'])}` | `{bool(row['usable_for_selection'])}` | "
            f"`{issue_preview(row.get('issues', ''))}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['audit_rows']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(rows: pd.DataFrame, output_dir: Path, *, smoke_case: str | None = None) -> dict[str, Any]:
    output_dir = repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "audit_rows.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    rows.to_csv(rows_path, index=False)
    summary = summarize_audit(rows, output_dir, smoke_case=smoke_case)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, rows), encoding="utf-8")
    return summary


def write_eval_fixture(
    root: Path,
    *,
    method: str,
    model: str,
    tasks: list[str],
    examples: int,
    status: str = "complete",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"eval_order": 0, "candidate_source": "smoke", "method": method, "served_model_id": model}]
    ).to_csv(root / "eval_plan.csv", index=False)
    metric_rows = []
    prediction_rows = []
    for task in tasks:
        metric = {"task": task, "examples": examples, "model": model}
        if task == "gsm8k":
            metric["strict_exact"] = 0.6
        elif task == "mmlu":
            metric["accuracy"] = 0.5
        elif task == "safety":
            metric["policy_accuracy"] = 0.75
        elif task == "humaneval_compile":
            metric["compile_rate"] = 0.4
        else:
            metric["accuracy"] = 0.5
        metric_rows.append(metric)
        for idx in range(examples):
            prediction_rows.append({"task": task, "index": idx, "model": model, "correct": True, "output": "ok"})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(root / "predictions.csv", index=False)
    scores = [primary_score(row) or 0.0 for _, row in metrics.iterrows()]
    pd.DataFrame(
        [
            {
                "model": model,
                "task_count": len(tasks),
                "avg_primary_score": sum(scores) / max(1, len(scores)),
                "worst_primary_score": min(scores) if scores else 0.0,
                "rank": 1,
            }
        ]
    ).to_csv(root / "model_summary.csv", index=False)
    summary = {
        "schema_version": 1,
        "status": status,
        "model": model,
        "models": [model],
        "model_count": 1,
        "tasks": ",".join(tasks),
        "example_source": "smoke",
        "max_examples_per_task": examples,
        "eval_plan": [{"eval_order": 0, "candidate_source": "smoke", "method": method, "served_model_id": model}],
        "metrics": metrics.to_dict(orient="records"),
    }
    (root / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_smoke_fixture(output_dir: Path, case: str) -> Path:
    fixture_root = repo_path(output_dir) / "fixtures" / case
    tasks = ["gsm8k", "mmlu", "safety", "humaneval_compile"]
    plan_rows = [
        {
            "gate_order": 0,
            "method": "source_qwen3_30b_instruct",
            "role": "source",
            "served_model_id": "candidate_source_qwen3_30b_instruct",
            "tasks": ",".join(tasks),
            "max_examples": 4,
            "eval_output_dir": rel(fixture_root / "source_qwen3_30b_instruct"),
        },
        {
            "gate_order": 1,
            "method": "source_qwen3_30b_coder",
            "role": "source",
            "served_model_id": "candidate_source_qwen3_30b_coder",
            "tasks": ",".join(tasks),
            "max_examples": 4,
            "eval_output_dir": rel(fixture_root / "source_qwen3_30b_coder"),
        },
        {
            "gate_order": 2,
            "method": "qwen3_moe_unified_mechanism_candidate",
            "role": "candidate",
            "served_model_id": "candidate_qwen3_moe_unified_mechanism_candidate",
            "tasks": ",".join(tasks),
            "max_examples": 4,
            "eval_output_dir": rel(fixture_root / "qwen3_moe_unified_mechanism_candidate"),
        },
    ]
    if case == "missing_task":
        candidate_tasks = tasks[:-1]
    else:
        candidate_tasks = tasks
    if case == "low_examples":
        candidate_examples = 3
    else:
        candidate_examples = 4
    candidate_model = "candidate_qwen3_moe_unified_mechanism_candidate"
    if case == "stale_model":
        candidate_model = "candidate_old_model"
    for plan_row in plan_rows:
        method = plan_row["method"]
        model = plan_row["served_model_id"]
        fixture_tasks = tasks
        examples = 4
        if method == "qwen3_moe_unified_mechanism_candidate":
            model = candidate_model
            fixture_tasks = candidate_tasks
            examples = candidate_examples
        write_eval_fixture(
            repo_path(plan_row["eval_output_dir"]),
            method=method,
            model=model,
            tasks=fixture_tasks,
            examples=examples,
        )
    gate_dir = fixture_root / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plan_rows).to_csv(gate_dir / "eval_gate_plan.csv", index=False)
    return gate_dir


def run_smoke_matrix(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        "valid_bundle": ("passed", 3),
        "stale_model": ("invalid_bundle", 2),
        "missing_task": ("invalid_bundle", 2),
        "low_examples": ("invalid_bundle", 2),
    }
    fixture_root = repo_path(args.output_dir) / "fixtures"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    rows = []
    for case, (expected_status, expected_usable) in expected.items():
        gate_dir = build_smoke_fixture(args.output_dir, case)
        audit_rows = audit_gate(gate_dir)
        summary = summarize_audit(audit_rows, repo_path(args.output_dir) / case, smoke_case=case)
        rows.append(
            {
                "case": case,
                "status": summary["status"],
                "expected_status": expected_status,
                "usable_for_selection_count": summary["usable_for_selection_count"],
                "expected_usable_for_selection_count": expected_usable,
                "passed": (
                    summary["status"] == expected_status
                    and summary["usable_for_selection_count"] == expected_usable
                ),
            }
        )
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(rows)
    matrix_path = output_dir / "audit_matrix.csv"
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
        "# Qwen3 MoE vLLM Eval Bundle Audit Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Cases: `{passed}/{len(matrix)}`",
        "",
        "| case | status | usable | expected | passed |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['status']}` | "
            f"{int(row['usable_for_selection_count'])} | "
            f"{int(row['expected_usable_for_selection_count'])} | `{bool(row['passed'])}` |"
        )
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    if summary["status"] != "passed":
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Qwen3 MoE vLLM eval artifacts before downstream selection.")
    parser.add_argument("--gate-dir", type=Path, default=Path("results/qwen3_moe_mechanism_eval_gate"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_eval_bundle_audit"))
    parser.add_argument("--min-examples", type=int, default=None)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_matrix:
        summary = run_smoke_matrix(args)
        print(f"Wrote Qwen3 MoE eval bundle audit smoke to {repo_path(args.output_dir).resolve()}")
        print(f"Status: {summary['status']}; cases {summary['passed_case_count']}/{summary['case_count']}")
        return
    rows = audit_gate(args.gate_dir, min_examples=args.min_examples)
    summary = write_outputs(rows, args.output_dir)
    print(f"Wrote Qwen3 MoE eval bundle audit to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; usable={summary['usable_for_selection_count']}/{summary['method_count']}")


if __name__ == "__main__":
    main()
