#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


PRIMARY_METRICS = ("strict_exact", "accuracy", "policy_accuracy", "compile_rate")


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


def split_csv(value: Any) -> list[str]:
    value = clean_value(value)
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def fnum(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    value = fnum(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def primary_metric(row: pd.Series | dict[str, Any]) -> tuple[str, float | None]:
    for metric in PRIMARY_METRICS:
        value = row.get(metric)
        if value is not None and pd.notna(value):
            return metric, float(value)
    return "score", None


def metric_score_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(rows)
    for _, row in metrics.iterrows():
        metric_name, score = primary_metric(row)
        rows.append(
            {
                "model": str(row.get("model", "")),
                "task": str(row.get("task", "")),
                "primary_metric": metric_name,
                "primary_score": score,
                "examples": int(row.get("examples", 0)) if pd.notna(row.get("examples", None)) else 0,
            }
        )
    return pd.DataFrame(rows)


def score_eval_output(job: pd.Series, *, interference_budget: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_dir = repo_path(job.get("output_dir", ""))
    summary = read_json(eval_dir / "summary.json")
    metrics = read_csv(eval_dir / "metrics.csv")
    expected_models = split_csv(job.get("served_models"))
    expected_tasks = split_csv(job.get("tasks"))

    if not summary and not metrics.empty:
        summary = {"status": "metrics_without_summary"}
    if not summary:
        return (
            {
                "job_id": job.get("job_id"),
                "scenario_id": job.get("scenario_id"),
                "status": "missing_eval_output",
                "eval_dir": rel(eval_dir),
                "expected_model_count": len(expected_models),
                "observed_model_count": 0,
                "expected_task_count": len(expected_tasks),
                "observed_task_count": 0,
                "source_frontier_avg_score": None,
                "best_single_avg_score": None,
                "source_frontier_worst_score": None,
                "best_single_worst_score": None,
                "frontier_avg_gain_vs_best_single": None,
                "frontier_worst_gain_vs_best_single": None,
                "surplus_vs_interference": None,
                "interference_budget": interference_budget,
                "decision_gate": "awaiting_vllm_eval",
                "recommended_action": "run planned vLLM eval and rerun feedback builder",
            },
            [],
        )

    score_rows = metric_score_rows(metrics)
    if score_rows.empty:
        return (
            {
                "job_id": job.get("job_id"),
                "scenario_id": job.get("scenario_id"),
                "status": "empty_metrics",
                "eval_dir": rel(eval_dir),
                "expected_model_count": len(expected_models),
                "observed_model_count": 0,
                "expected_task_count": len(expected_tasks),
                "observed_task_count": 0,
                "source_frontier_avg_score": None,
                "best_single_avg_score": None,
                "source_frontier_worst_score": None,
                "best_single_worst_score": None,
                "frontier_avg_gain_vs_best_single": None,
                "frontier_worst_gain_vs_best_single": None,
                "surplus_vs_interference": None,
                "interference_budget": interference_budget,
                "decision_gate": "blocked_empty_metrics",
                "recommended_action": "inspect eval output because summary exists but no task metrics were found",
            },
            [],
        )

    observed_models = sorted(set(score_rows["model"].astype(str)))
    observed_tasks = sorted(set(score_rows["task"].astype(str)))
    missing_models = sorted(set(expected_models) - set(observed_models))
    missing_tasks = sorted(set(expected_tasks) - set(observed_tasks))

    task_rows: list[dict[str, Any]] = []
    for task, group in score_rows.groupby("task", sort=True):
        ranked = group.dropna(subset=["primary_score"]).sort_values("primary_score", ascending=False)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        task_rows.append(
            {
                "job_id": job.get("job_id"),
                "scenario_id": job.get("scenario_id"),
                "task": task,
                "frontier_source_model": best["model"],
                "frontier_score": float(best["primary_score"]),
                "primary_metric": best["primary_metric"],
                "observed_model_count": int(len(group)),
                "expected_models": ",".join(expected_models),
            }
        )

    model_rows = []
    for model, group in score_rows.groupby("model", sort=True):
        values = pd.to_numeric(group["primary_score"], errors="coerce").dropna()
        model_rows.append(
            {
                "model": model,
                "avg": float(values.mean()) if len(values) else None,
                "worst": float(values.min()) if len(values) else None,
                "task_count": int(len(values)),
            }
        )
    model_df = pd.DataFrame(model_rows)
    task_df = pd.DataFrame(task_rows)
    frontier_avg = float(task_df["frontier_score"].mean()) if not task_df.empty else None
    frontier_worst = float(task_df["frontier_score"].min()) if not task_df.empty else None
    best_single_avg = float(model_df["avg"].max()) if not model_df.empty else None
    best_single_worst = float(model_df["worst"].max()) if not model_df.empty else None
    avg_gain = None if frontier_avg is None or best_single_avg is None else frontier_avg - best_single_avg
    worst_gain = None if frontier_worst is None or best_single_worst is None else frontier_worst - best_single_worst
    surplus = None if avg_gain is None else avg_gain - interference_budget

    complete = not missing_models and not missing_tasks and summary.get("status") == "complete"
    if not complete:
        decision_gate = "incomplete_vllm_eval"
        action = "finish matched endpoint eval before feeding average optimizer"
    elif surplus is not None and surplus >= 0.0 and (worst_gain is None or worst_gain >= 0.0):
        decision_gate = "final_average_budget_candidate"
        action = "promote this source set to same-shape average materialization and locked candidate eval"
    elif avg_gain is not None and avg_gain > 0.0:
        decision_gate = "probe_only_below_interference_budget"
        action = "keep as endpoint-expansion/probe-only; do not spend final average budget yet"
    else:
        decision_gate = "source_frontier_not_better_than_endpoint"
        action = "prefer best endpoint and search more complementary sources before averaging"

    return (
        {
            "job_id": job.get("job_id"),
            "scenario_id": job.get("scenario_id"),
            "status": "scored_complete" if complete else "scored_incomplete",
            "eval_status": summary.get("status"),
            "eval_dir": rel(eval_dir),
            "expected_model_count": len(expected_models),
            "observed_model_count": len(observed_models),
            "missing_models": ",".join(missing_models),
            "expected_task_count": len(expected_tasks),
            "observed_task_count": len(observed_tasks),
            "missing_tasks": ",".join(missing_tasks),
            "source_frontier_avg_score": frontier_avg,
            "best_single_avg_score": best_single_avg,
            "source_frontier_worst_score": frontier_worst,
            "best_single_worst_score": best_single_worst,
            "frontier_avg_gain_vs_best_single": avg_gain,
            "frontier_worst_gain_vs_best_single": worst_gain,
            "surplus_vs_interference": surplus,
            "interference_budget": interference_budget,
            "decision_gate": decision_gate,
            "recommended_action": action,
            "task_frontier_assignment": json.dumps(
                {str(row["task"]): str(row["frontier_source_model"]) for row in task_rows},
                sort_keys=True,
            ),
        },
        task_rows,
    )


def build_report(summary: dict[str, Any], jobs: pd.DataFrame, tasks: pd.DataFrame) -> str:
    top = summary.get("top_scored_job") or {}
    lines = [
        "# Qwen Source Frontier Eval Feedback",
        "",
        "This artifact converts completed vLLM source-frontier eval outputs into the exact signal needed by the average optimizer: task-wise endpoint frontier, best single endpoint, and surplus against the observed merge-interference budget.",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Eval jobs: `{summary['job_count']}`",
        f"- Scored jobs: `{summary['scored_job_count']}`",
        f"- Final-budget candidates: `{summary['final_average_budget_candidate_count']}`",
        f"- Probe-only candidates: `{summary['probe_only_candidate_count']}`",
        f"- Interference budget: `{fmt(summary['interference_budget'])}`",
        f"- Top scored job: `{top.get('job_id')}` gate `{top.get('decision_gate')}` surplus `{fmt(top.get('surplus_vs_interference'))}`",
        "",
        "## Job Feedback",
        "",
        "| job | status | models | tasks | avg gain | surplus | gate | action |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in jobs.iterrows():
        lines.append(
            f"| `{row['job_id']}` | `{row['status']}` | {int(row['observed_model_count'])}/"
            f"{int(row['expected_model_count'])} | {int(row['observed_task_count'])}/"
            f"{int(row['expected_task_count'])} | {fmt(row['frontier_avg_gain_vs_best_single'])} | "
            f"{fmt(row['surplus_vs_interference'])} | `{row['decision_gate']}` | {row['recommended_action']} |"
        )
    lines.extend(
        [
            "",
            "## Task Frontier",
            "",
            "| job | task | source model | score | metric |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for _, row in tasks.iterrows():
        lines.append(
            f"| `{row['job_id']}` | `{row['task']}` | `{row['frontier_source_model']}` | "
            f"{fmt(row['frontier_score'])} | `{row['primary_metric']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['job_feedback']}`",
            f"- `{summary['outputs']['task_frontier']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_jobs = read_csv(args.eval_jobs)
    optimizer = read_json(args.average_source_set_optimizer)
    interference_budget = fnum(optimizer.get("interference_budget"))
    if interference_budget is None:
        interference_budget = float(args.default_interference_budget)

    job_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for _, job in eval_jobs.iterrows():
        job_feedback, task_feedback = score_eval_output(job, interference_budget=interference_budget)
        job_rows.append(job_feedback)
        task_rows.extend(task_feedback)

    jobs = pd.DataFrame(job_rows)
    tasks = pd.DataFrame(task_rows)
    if tasks.empty:
        tasks = pd.DataFrame(
            columns=[
                "job_id",
                "scenario_id",
                "task",
                "frontier_source_model",
                "frontier_score",
                "primary_metric",
                "observed_model_count",
                "expected_models",
            ]
        )
    scored = jobs[jobs["status"].astype(str).str.startswith("scored_")] if not jobs.empty else pd.DataFrame()
    final_count = int((jobs["decision_gate"] == "final_average_budget_candidate").sum()) if not jobs.empty else 0
    probe_count = int((jobs["decision_gate"] == "probe_only_below_interference_budget").sum()) if not jobs.empty else 0
    if final_count:
        status = "source_frontier_feedback_promotes_average_candidate"
    elif len(scored):
        status = "source_frontier_feedback_scored_no_final_candidate"
    else:
        status = "awaiting_vllm_source_frontier_results"

    top_scored = {}
    if not scored.empty:
        ranked = scored.sort_values(
            ["surplus_vs_interference", "frontier_avg_gain_vs_best_single"],
            ascending=[False, False],
            na_position="last",
        )
        top_scored = ranked.iloc[0].to_dict()

    summary = {
        "schema_version": 1,
        "status": status,
        "eval_jobs": rel(args.eval_jobs),
        "job_count": int(len(jobs)),
        "scored_job_count": int(len(scored)),
        "final_average_budget_candidate_count": final_count,
        "probe_only_candidate_count": probe_count,
        "interference_budget": float(interference_budget),
        "top_scored_job": {
            "job_id": top_scored.get("job_id"),
            "scenario_id": top_scored.get("scenario_id"),
            "decision_gate": top_scored.get("decision_gate"),
            "frontier_avg_gain_vs_best_single": fnum(top_scored.get("frontier_avg_gain_vs_best_single")),
            "frontier_worst_gain_vs_best_single": fnum(top_scored.get("frontier_worst_gain_vs_best_single")),
            "surplus_vs_interference": fnum(top_scored.get("surplus_vs_interference")),
            "recommended_action": top_scored.get("recommended_action"),
        },
        "blocking_reason": (
            "Run the planned vLLM source-frontier eval jobs, then rerun this feedback builder."
            if not len(scored)
            else ""
        ),
        "outputs": {
            "job_feedback": rel(output_dir / "job_feedback.csv"),
            "task_frontier": rel(output_dir / "task_frontier.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    jobs.to_csv(output_dir / "job_feedback.csv", index=False)
    tasks.to_csv(output_dir / "task_frontier.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, jobs, tasks), encoding="utf-8")
    return summary


def write_mock_eval(eval_dir: Path, models: list[str], metrics_by_model: dict[str, dict[str, float]]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tasks = sorted({task for task_scores in metrics_by_model.values() for task in task_scores})
    for model in models:
        for task in tasks:
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "examples": 8,
                    "accuracy": metrics_by_model[model][task],
                }
            )
    pd.DataFrame(rows).to_csv(eval_dir / "metrics.csv", index=False)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "models": models,
                "model_count": len(models),
                "tasks": ",".join(tasks),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    input_dir = output_dir / "mock_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    eval_root = input_dir / "eval_outputs"
    jobs = pd.DataFrame(
        [
            {
                "job_id": "final_frontier",
                "scenario_id": "mock_final",
                "served_models": "model_a,model_b,model_c",
                "tasks": "gsm8k,humaneval_compile,mmlu",
                "output_dir": rel(eval_root / "final_frontier"),
            },
            {
                "job_id": "probe_frontier",
                "scenario_id": "mock_probe",
                "served_models": "model_a,model_b",
                "tasks": "gsm8k,mmlu",
                "output_dir": rel(eval_root / "probe_frontier"),
            },
            {
                "job_id": "reject_frontier",
                "scenario_id": "mock_reject",
                "served_models": "model_a,model_b",
                "tasks": "gsm8k,mmlu",
                "output_dir": rel(eval_root / "reject_frontier"),
            },
        ]
    )
    jobs_path = input_dir / "vllm_eval_jobs.csv"
    jobs.to_csv(jobs_path, index=False)
    optimizer_path = input_dir / "average_source_set_optimizer_summary.json"
    optimizer_path.write_text(
        json.dumps({"schema_version": 1, "interference_budget": 0.15}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_mock_eval(
        eval_root / "final_frontier",
        ["model_a", "model_b", "model_c"],
        {
            "model_a": {"gsm8k": 0.90, "humaneval_compile": 0.50, "mmlu": 0.50},
            "model_b": {"gsm8k": 0.50, "humaneval_compile": 0.90, "mmlu": 0.50},
            "model_c": {"gsm8k": 0.50, "humaneval_compile": 0.50, "mmlu": 0.90},
        },
    )
    write_mock_eval(
        eval_root / "probe_frontier",
        ["model_a", "model_b"],
        {
            "model_a": {"gsm8k": 0.80, "mmlu": 0.60},
            "model_b": {"gsm8k": 0.60, "mmlu": 0.80},
        },
    )
    write_mock_eval(
        eval_root / "reject_frontier",
        ["model_a", "model_b"],
        {
            "model_a": {"gsm8k": 0.80, "mmlu": 0.80},
            "model_b": {"gsm8k": 0.70, "mmlu": 0.70},
        },
    )
    smoke_args = argparse.Namespace(
        eval_jobs=jobs_path,
        average_source_set_optimizer=optimizer_path,
        output_dir=output_dir,
        default_interference_budget=args.default_interference_budget,
    )
    summary = build(smoke_args)
    job_feedback = read_csv(output_dir / "job_feedback.csv")
    gates = {str(row["job_id"]): str(row["decision_gate"]) for _, row in job_feedback.iterrows()}
    checks = {
        "final_frontier_promoted": gates.get("final_frontier") == "final_average_budget_candidate",
        "probe_frontier_probe_only": gates.get("probe_frontier") == "probe_only_below_interference_budget",
        "reject_frontier_rejected": gates.get("reject_frontier") == "source_frontier_not_better_than_endpoint",
        "scored_all_jobs": int(summary.get("scored_job_count") or 0) == 3,
    }
    checks["passed"] = all(checks.values())
    summary.update(
        {
            "status": "smoke_passed" if checks["passed"] else "smoke_failed",
            "smoke_checks": checks,
            "smoke_input_dir": rel(input_dir),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    smoke_lines = [
        "",
        "## Smoke Checks",
        "",
        f"- Smoke status: `{summary['status']}`",
        "",
        "| check | passed |",
        "| --- | --- |",
    ]
    for key, value in checks.items():
        if key != "passed":
            smoke_lines.append(f"| `{key}` | `{value}` |")
    (output_dir / "report.md").write_text(report + "\n".join(smoke_lines) + "\n", encoding="utf-8")
    if not checks["passed"]:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-frontier average feedback from completed vLLM eval outputs.")
    parser.add_argument("--eval-jobs", type=Path, default=Path("results/qwen_source_discovery_eval_plan/vllm_eval_jobs.csv"))
    parser.add_argument(
        "--average-source-set-optimizer",
        type=Path,
        default=Path("results/qwen3_average_source_set_optimizer/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_source_frontier_eval_feedback"))
    parser.add_argument("--default-interference-budget", type=float, default=0.03)
    parser.add_argument("--smoke-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_smoke(args) if args.smoke_matrix else build(args)
    print(f"Wrote Qwen source frontier eval feedback to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; scored={summary['scored_job_count']}/{summary['job_count']}; "
        f"final={summary['final_average_budget_candidate_count']}"
    )


if __name__ == "__main__":
    main()
