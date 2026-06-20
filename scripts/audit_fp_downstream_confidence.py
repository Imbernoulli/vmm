#!/usr/bin/env python
"""Audit aggregate confidence for the Qwen3 MoE generation matrix."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_SOURCES = ["instruct", "coder"]
ALL_SOURCES = ["base", "instruct", "coder", "thinking"]
PAIR_MERGE = "merge_instruct+coder"
PAIR_ROUTERCAL = "merge_instruct+coder_routercal"
TASK_SAMPLE_SIZES = {"mmlu": 120, "gsm8k": 40, "humaneval": 40}


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


def fmt(value: Any, digits: int = 3) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def wilson_interval(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    p = min(1.0, max(0.0, float(score)))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_task_n(raw: list[str]) -> dict[str, int]:
    sizes = dict(TASK_SAMPLE_SIZES)
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Task size must look like task=n, got {item!r}")
        task, value = item.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Task name is empty in {item!r}")
        sizes[task] = int(value)
    return sizes


def load_matrix(path: Path, tasks: list[str]) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"name", *tasks, "avg", "worst"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required matrix columns: {sorted(missing)}")
    return rows


def row_by_name(rows: pd.DataFrame, name: str) -> pd.Series:
    match = rows[rows["name"] == name]
    if match.empty:
        raise ValueError(f"Missing model row {name!r}")
    return match.iloc[0]


def best_model_for_score(rows: pd.DataFrame, names: list[str], score: str) -> str:
    candidates = [row_by_name(rows, name) for name in names if not rows[rows["name"] == name].empty]
    if not candidates:
        raise ValueError(f"No frontier candidates found for {score}")
    best = max(candidates, key=lambda row: float(row[score]))
    return str(best["name"])


def interval_rows(rows: pd.DataFrame, task_sizes: dict[str, int], z: float) -> pd.DataFrame:
    tasks = list(task_sizes)
    out: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        model = str(row["name"])
        task_lowers: list[float] = []
        task_uppers: list[float] = []
        for task, n in task_sizes.items():
            score = float(row[task])
            lower, upper = wilson_interval(score, n, z)
            task_lowers.append(lower)
            task_uppers.append(upper)
            out.append(
                {
                    "model": model,
                    "score": task,
                    "score_kind": "task_accuracy",
                    "value": score,
                    "n": n,
                    "lower": lower,
                    "upper": upper,
                    "half_width": (upper - lower) / 2.0,
                }
            )
        avg_lower = sum(task_lowers) / len(tasks)
        avg_upper = sum(task_uppers) / len(tasks)
        out.append(
            {
                "model": model,
                "score": "avg",
                "score_kind": "macro_average_of_task_intervals",
                "value": float(row["avg"]),
                "n": sum(task_sizes.values()),
                "lower": avg_lower,
                "upper": avg_upper,
                "half_width": (avg_upper - avg_lower) / 2.0,
            }
        )
        out.append(
            {
                "model": model,
                "score": "worst",
                "score_kind": "minimum_of_task_intervals",
                "value": float(row["worst"]),
                "n": sum(task_sizes.values()),
                "lower": min(task_lowers),
                "upper": min(task_uppers),
                "half_width": (min(task_uppers) - min(task_lowers)) / 2.0,
            }
        )
    return pd.DataFrame(out)


def interval_lookup(intervals: pd.DataFrame, model: str, score: str) -> dict[str, Any]:
    match = intervals[(intervals["model"] == model) & (intervals["score"] == score)]
    if match.empty:
        raise ValueError(f"Missing interval for model={model!r}, score={score!r}")
    return match.iloc[0].to_dict()


def compare_intervals(
    intervals: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    task_sizes: dict[str, int],
) -> pd.DataFrame:
    score_names = list(task_sizes) + ["avg", "worst"]
    comparisons: list[dict[str, Any]] = []
    comparison_specs = [
        ("routercal_vs_naive", PAIR_ROUTERCAL, PAIR_MERGE),
        ("routercal_vs_pair_source_frontier", PAIR_ROUTERCAL, "pair_source_frontier"),
        ("routercal_vs_all_source_frontier", PAIR_ROUTERCAL, "all_source_frontier"),
    ]
    for score in score_names:
        for comparison, candidate_model, reference_model in comparison_specs:
            if reference_model == "pair_source_frontier":
                actual_reference = best_model_for_score(rows, PAIR_SOURCES, score)
            elif reference_model == "all_source_frontier":
                actual_reference = best_model_for_score(rows, ALL_SOURCES, score)
            else:
                actual_reference = reference_model
            candidate = interval_lookup(intervals, candidate_model, score)
            reference = interval_lookup(intervals, actual_reference, score)
            diff = float(candidate["value"]) - float(reference["value"])
            diff_lower = float(candidate["lower"]) - float(reference["upper"])
            diff_upper = float(candidate["upper"]) - float(reference["lower"])
            if diff_lower > 0:
                status = "confident_positive"
            elif diff_upper < 0:
                status = "confident_negative"
            elif diff > 0:
                status = "directional_positive_not_confident"
            elif diff < 0:
                status = "directional_negative_not_confident"
            else:
                status = "tie_with_uncertainty"
            comparisons.append(
                {
                    "comparison": comparison,
                    "score": score,
                    "candidate_model": candidate_model,
                    "reference_model": actual_reference,
                    "candidate_value": float(candidate["value"]),
                    "reference_value": float(reference["value"]),
                    "diff": diff,
                    "diff_lower_conservative": diff_lower,
                    "diff_upper_conservative": diff_upper,
                    "status": status,
                    "confidence_rule": "conservative_independent_interval_difference",
                }
            )
    return pd.DataFrame(comparisons)


def first_comparison(comparisons: pd.DataFrame, comparison: str, score: str) -> dict[str, Any]:
    match = comparisons[(comparisons["comparison"] == comparison) & (comparisons["score"] == score)]
    if match.empty:
        raise ValueError(f"Missing comparison={comparison!r}, score={score!r}")
    return match.iloc[0].to_dict()


def build_summary(
    intervals: pd.DataFrame,
    comparisons: pd.DataFrame,
    rows: pd.DataFrame,
    task_sizes: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    task_scores = list(task_sizes)
    routercal_vs_naive_tasks = comparisons[
        (comparisons["comparison"] == "routercal_vs_naive") & (comparisons["score"].isin(task_scores))
    ]
    routercal_vs_pair_tasks = comparisons[
        (comparisons["comparison"] == "routercal_vs_pair_source_frontier")
        & (comparisons["score"].isin(task_scores))
    ]
    routercal_vs_all_tasks = comparisons[
        (comparisons["comparison"] == "routercal_vs_all_source_frontier")
        & (comparisons["score"].isin(task_scores))
    ]
    avg_naive = first_comparison(comparisons, "routercal_vs_naive", "avg")
    avg_pair = first_comparison(comparisons, "routercal_vs_pair_source_frontier", "avg")
    avg_all = first_comparison(comparisons, "routercal_vs_all_source_frontier", "avg")
    best_model = rows.sort_values("avg", ascending=False).iloc[0]
    return {
        "schema_version": 1,
        "status": "generation_confidence_audit_ready",
        "role": "aggregate_confidence_audit_not_paired_vllm_selector",
        "task_sample_sizes": task_sizes,
        "task_count": len(task_scores),
        "model_count": int(len(rows)),
        "best_avg_model": str(best_model["name"]),
        "best_avg": float(best_model["avg"]),
        "routercal_positive_task_count_vs_naive": int((routercal_vs_naive_tasks["diff"] > 0).sum()),
        "routercal_confident_positive_task_count_vs_naive": int(
            (routercal_vs_naive_tasks["diff_lower_conservative"] > 0).sum()
        ),
        "routercal_confident_negative_task_count_vs_naive": int(
            (routercal_vs_naive_tasks["diff_upper_conservative"] < 0).sum()
        ),
        "routercal_positive_task_count_vs_pair_frontier": int((routercal_vs_pair_tasks["diff"] > 0).sum()),
        "routercal_confident_beats_pair_frontier_task_count": int(
            (routercal_vs_pair_tasks["diff_lower_conservative"] > 0).sum()
        ),
        "routercal_confident_loses_pair_frontier_task_count": int(
            (routercal_vs_pair_tasks["diff_upper_conservative"] < 0).sum()
        ),
        "routercal_confident_beats_all_source_frontier_task_count": int(
            (routercal_vs_all_tasks["diff_lower_conservative"] > 0).sum()
        ),
        "routercal_avg_diff_vs_naive": float(avg_naive["diff"]),
        "routercal_avg_diff_lower_vs_naive": float(avg_naive["diff_lower_conservative"]),
        "routercal_avg_diff_upper_vs_naive": float(avg_naive["diff_upper_conservative"]),
        "routercal_avg_gap_vs_pair_frontier": float(avg_pair["diff"]),
        "routercal_avg_gap_lower_vs_pair_frontier": float(avg_pair["diff_lower_conservative"]),
        "routercal_avg_gap_upper_vs_pair_frontier": float(avg_pair["diff_upper_conservative"]),
        "routercal_avg_gap_vs_all_source_frontier": float(avg_all["diff"]),
        "routercal_avg_gap_lower_vs_all_source_frontier": float(avg_all["diff_lower_conservative"]),
        "routercal_avg_gap_upper_vs_all_source_frontier": float(avg_all["diff_upper_conservative"]),
        "interpretation": (
            "Router calibration has positive point-estimate gains over the naive pair average on some "
            "tasks, but the conservative aggregate intervals do not yet make those gains a confident "
            "acceptance signal, and no task confidently beats the source frontier. Use it as a mechanism "
            "probe and candidate repair lever, then require matched vLLM paired prediction gates."
        ),
        "outputs": {
            "model_task_intervals": rel(output_dir / "model_task_intervals.csv"),
            "comparison_intervals": rel(output_dir / "comparison_intervals.csv"),
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
        },
    }


def build_report(comparisons: pd.DataFrame, summary: dict[str, Any]) -> str:
    interesting = comparisons[
        comparisons["comparison"].isin(
            ["routercal_vs_naive", "routercal_vs_pair_source_frontier"]
        )
        & comparisons["score"].isin(["mmlu", "gsm8k", "humaneval", "avg"])
    ]
    lines = [
        "# Qwen3 MoE Downstream Confidence Audit",
        "",
        "This audit adds aggregate uncertainty bounds to the auxiliary generation matrix. It uses Wilson intervals from the observed task accuracies and does not replace the final paired vLLM selector.",
        "",
        "## Summary",
        "",
        f"- Task sample sizes: `{json.dumps(summary['task_sample_sizes'], sort_keys=True)}`",
        f"- Router-cal positive tasks vs naive: `{summary['routercal_positive_task_count_vs_naive']}/{summary['task_count']}`",
        f"- Router-cal confident positive tasks vs naive: `{summary['routercal_confident_positive_task_count_vs_naive']}/{summary['task_count']}`",
        f"- Router-cal confident source-frontier wins: `{summary['routercal_confident_beats_pair_frontier_task_count']}/{summary['task_count']}`",
        f"- Avg diff vs naive: `{fmt(summary['routercal_avg_diff_vs_naive'])}` with conservative interval `[{fmt(summary['routercal_avg_diff_lower_vs_naive'])}, {fmt(summary['routercal_avg_diff_upper_vs_naive'])}]`",
        f"- Avg gap vs pair source frontier: `{fmt(summary['routercal_avg_gap_vs_pair_frontier'])}` with conservative interval `[{fmt(summary['routercal_avg_gap_lower_vs_pair_frontier'])}, {fmt(summary['routercal_avg_gap_upper_vs_pair_frontier'])}]`",
        "",
        "## Comparison Intervals",
        "",
        "| comparison | score | candidate | reference | diff | conservative diff interval | status |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in interesting.iterrows():
        lines.append(
            f"| `{row['comparison']}` | `{row['score']}` | `{row['candidate_model']}` | "
            f"`{row['reference_model']}` | {fmt(row['diff'])} | "
            f"[{fmt(row['diff_lower_conservative'])}, {fmt(row['diff_upper_conservative'])}] | "
            f"`{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['model_task_intervals']}`",
            f"- `{summary['outputs']['comparison_intervals']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    task_sizes = parse_task_n(args.task_n)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_matrix(repo_path(args.matrix), list(task_sizes))
    intervals = interval_rows(rows, task_sizes, args.confidence_z)
    comparisons = compare_intervals(intervals, rows, task_sizes=task_sizes)
    intervals.to_csv(output_dir / "model_task_intervals.csv", index=False)
    comparisons.to_csv(output_dir / "comparison_intervals.csv", index=False)
    summary = build_summary(intervals, comparisons, rows, task_sizes, output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(comparisons, summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit confidence intervals for the Qwen3 MoE generation matrix.")
    parser.add_argument("--matrix", type=Path, default=Path("results/fp_downstream_matrix/matrix.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fp_downstream_confidence_audit"))
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--task-n", action="append", default=[], help="Override task sample size, e.g. mmlu=384")
    args = parser.parse_args()
    summary = build(args)
    print(f"Wrote downstream confidence audit to {repo_path(args.output_dir).resolve()}")
    print(
        f"Status: {summary['status']}; positive_vs_naive="
        f"{summary['routercal_positive_task_count_vs_naive']}/{summary['task_count']}; "
        f"confident_source_wins={summary['routercal_confident_beats_pair_frontier_task_count']}/"
        f"{summary['task_count']}"
    )


if __name__ == "__main__":
    main()
