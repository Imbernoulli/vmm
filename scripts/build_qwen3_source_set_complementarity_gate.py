#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def source_key(sources: list[str] | tuple[str, ...]) -> str:
    return "+".join(sorted(str(source) for source in sources))


def source_label(sources: list[str] | tuple[str, ...]) -> str:
    return "+".join(str(source) for source in sources)


def parse_merge_name(name: str) -> tuple[str, str, list[str]] | None:
    if not name.startswith("merge_"):
        return None
    body = name[len("merge_") :]
    variant = "naive_average"
    if body.endswith("_routercal"):
        body = body[: -len("_routercal")]
        variant = "router_calibrated"
    sources = [part for part in body.split("+") if part]
    if len(sources) < 2:
        return None
    return source_key(sources), variant, sources


def task_columns(matrix: pd.DataFrame) -> list[str]:
    ignored = {"name", "avg", "worst"}
    return [column for column in matrix.columns if column not in ignored]


def dominant_source(
    source_scores: dict[str, dict[str, float]],
    sources: tuple[str, ...],
    tasks: list[str],
    *,
    margin: float,
) -> str | None:
    for source in sources:
        dominates_all = True
        strict_somewhere = False
        for other in sources:
            if other == source:
                continue
            for task in tasks:
                delta = source_scores[source][task] - source_scores[other][task]
                if delta < -margin:
                    dominates_all = False
                    break
                if delta > margin:
                    strict_somewhere = True
            if not dominates_all:
                break
        if dominates_all and strict_somewhere:
            return source
    return None


def strict_task_winners(
    source_scores: dict[str, dict[str, float]],
    sources: tuple[str, ...],
    tasks: list[str],
    *,
    margin: float,
) -> dict[str, str]:
    winners: dict[str, str] = {}
    for task in tasks:
        ordered = sorted(
            ((source, source_scores[source][task]) for source in sources),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(ordered) == 1 or ordered[0][1] - ordered[1][1] > margin:
            winners[task] = ordered[0][0]
        else:
            winners[task] = "tie"
    return winners


def build_source_set_rows(
    matrix: pd.DataFrame,
    *,
    min_avg_frontier_gain: float,
    min_worst_frontier_gain: float,
    task_win_margin: float,
) -> pd.DataFrame:
    tasks = task_columns(matrix)
    source_df = matrix[~matrix["name"].astype(str).str.startswith("merge_")].copy()
    source_names = [str(name) for name in source_df["name"].tolist()]
    source_scores = {
        str(row["name"]): {task: float(row[task]) for task in tasks}
        for _, row in source_df.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for size in range(2, len(source_names) + 1):
        for combo in itertools.combinations(source_names, size):
            frontier_scores = {task: max(source_scores[source][task] for source in combo) for task in tasks}
            frontier_avg = sum(frontier_scores.values()) / len(tasks)
            frontier_worst = min(frontier_scores.values())
            source_summary = []
            for source in combo:
                values = [source_scores[source][task] for task in tasks]
                source_summary.append(
                    {
                        "source": source,
                        "avg": sum(values) / len(values),
                        "worst": min(values),
                    }
                )
            best_avg_source = max(source_summary, key=lambda item: item["avg"])
            best_worst_source = max(source_summary, key=lambda item: item["worst"])
            winners = strict_task_winners(source_scores, combo, tasks, margin=task_win_margin)
            strict_sources = sorted({winner for winner in winners.values() if winner != "tie"})
            dominant = dominant_source(source_scores, combo, tasks, margin=task_win_margin)
            avg_gain = frontier_avg - best_avg_source["avg"]
            worst_gain = frontier_worst - best_worst_source["worst"]
            if dominant:
                gate = "source_dominated_not_averageable_as_final"
                action = "use_dominant_source_as_frontier; keep averages as repair_or_negative_control"
            elif avg_gain >= min_avg_frontier_gain or worst_gain >= min_worst_frontier_gain:
                gate = "complementary_source_set_candidate"
                action = "spend_average_budget_after_topology_and_connectivity_gates"
            elif len(strict_sources) >= 2:
                gate = "weak_complementarity_needs_more_or_larger_eval"
                action = "run endpoint_eval_first; average only after frontier gain is statistically material"
            else:
                gate = "no_clear_task_complementarity"
                action = "do_not_prioritize_average; search better source set"
            rows.append(
                {
                    "source_set": source_label(combo),
                    "source_set_key": source_key(combo),
                    "source_count": len(combo),
                    "gate": gate,
                    "recommended_action": action,
                    "dominant_source": dominant,
                    "strict_task_winner_count": len(strict_sources),
                    "strict_task_winners": json.dumps(winners, sort_keys=True),
                    "frontier_avg": frontier_avg,
                    "frontier_worst": frontier_worst,
                    "best_single_avg_source": best_avg_source["source"],
                    "best_single_avg": best_avg_source["avg"],
                    "best_single_worst_source": best_worst_source["source"],
                    "best_single_worst": best_worst_source["worst"],
                    "frontier_avg_gain_vs_best_single": avg_gain,
                    "frontier_worst_gain_vs_best_single": worst_gain,
                }
            )
    priority = {
        "complementary_source_set_candidate": 0,
        "weak_complementarity_needs_more_or_larger_eval": 1,
        "no_clear_task_complementarity": 2,
        "source_dominated_not_averageable_as_final": 3,
    }
    out = pd.DataFrame(rows)
    out["_priority"] = out["gate"].map(priority).fillna(9)
    out = out.sort_values(
        ["_priority", "frontier_avg_gain_vs_best_single", "frontier_worst_gain_vs_best_single"],
        ascending=[True, False, False],
    ).drop(columns=["_priority"])
    return out.reset_index(drop=True)


def build_observed_merge_rows(matrix: pd.DataFrame, source_sets: pd.DataFrame) -> pd.DataFrame:
    tasks = task_columns(matrix)
    by_key = {str(row["source_set_key"]): row for _, row in source_sets.iterrows()}
    rows: list[dict[str, Any]] = []
    for _, row in matrix.iterrows():
        parsed = parse_merge_name(str(row["name"]))
        if not parsed:
            continue
        key, variant, sources = parsed
        source_set = by_key.get(key)
        if source_set is None:
            continue
        task_gaps = {
            task: float(row[task]) - max(float(matrix[matrix["name"] == source][task].iloc[0]) for source in sources)
            for task in tasks
        }
        rows.append(
            {
                "merge_model": str(row["name"]),
                "source_set": "+".join(sources),
                "source_set_key": key,
                "variant": variant,
                "avg": float(row["avg"]),
                "worst": float(row["worst"]),
                "frontier_avg": float(source_set["frontier_avg"]),
                "frontier_worst": float(source_set["frontier_worst"]),
                "avg_gap_to_source_frontier": float(row["avg"]) - float(source_set["frontier_avg"]),
                "worst_gap_to_source_frontier": float(row["worst"]) - float(source_set["frontier_worst"]),
                "task_regression_count_vs_frontier": sum(1 for value in task_gaps.values() if value < 0.0),
                "task_gaps_to_frontier": json.dumps(task_gaps, sort_keys=True),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["source_set_key", "avg_gap_to_source_frontier", "worst_gap_to_source_frontier"],
        ascending=[True, False, False],
    )


def build_recommended_scenarios(scenario_matrix: pd.DataFrame) -> pd.DataFrame:
    if scenario_matrix.empty:
        return pd.DataFrame()
    rows = []
    for _, row in scenario_matrix.iterrows():
        scenario_id = str(row["scenario_id"])
        if scenario_id == "negative_controls":
            action = "keep_negative_controls"
        elif str(row["priority"]).startswith("p0"):
            action = "run_endpoint_eval_then_source_set_gate"
        else:
            action = "queue_after_p0_source_set_gate"
        rows.append(
            {
                "scenario_id": scenario_id,
                "priority": row["priority"],
                "input_roles": row.get("input_roles"),
                "required_models": row.get("required_models"),
                "first_average_candidate": row.get("first_average_candidate"),
                "source_set_gate_action": action,
            }
        )
    return pd.DataFrame(rows)


def build_report(summary: dict[str, Any], source_sets: pd.DataFrame, observed: pd.DataFrame, scenarios: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 Source-Set Complementarity Gate",
        "",
        "This gate decides whether a source set is worth averaging before spending candidate/eval budget. It separates the source frontier from the merge method: if one source already dominates the measured tasks, averaging is treated as repair or ablation rather than a final-candidate prior.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Current source set: `{summary['current_source_set']}`",
        f"- Current gate: `{summary['current_gate']}`",
        f"- Current dominant source: `{summary['current_dominant_source']}`",
        f"- Current frontier avg gain vs best single: `{fmt(summary['current_frontier_avg_gain_vs_best_single'])}`",
        f"- Best observed merge: `{summary['current_best_observed_merge']}`",
        f"- Best observed avg gap to source frontier: `{fmt(summary['current_best_observed_avg_gap_to_frontier'])}`",
        f"- Strong complementary measured sets: `{summary['complementary_source_set_count']}`",
        f"- Weak complementary measured sets: `{summary['weak_complementary_source_set_count']}`",
        f"- Recommended action: `{summary['recommended_action']}`",
        "",
        "## Measured Source Sets",
        "",
        "| source set | gate | dominant | strict winners | frontier avg gain | frontier worst gain | action |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in source_sets.iterrows():
        lines.append(
            f"| `{row['source_set']}` | `{row['gate']}` | `{row['dominant_source']}` | "
            f"{int(row['strict_task_winner_count'])} | {fmt(row['frontier_avg_gain_vs_best_single'])} | "
            f"{fmt(row['frontier_worst_gain_vs_best_single'])} | {row['recommended_action']} |"
        )
    lines.extend(
        [
            "",
            "## Observed Merge Gaps",
            "",
            "| merge | source set | variant | avg | gap to frontier | task regressions |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in observed.iterrows():
        lines.append(
            f"| `{row['merge_model']}` | `{row['source_set']}` | `{row['variant']}` | "
            f"{fmt(row['avg'])} | {fmt(row['avg_gap_to_source_frontier'])} | "
            f"{int(row['task_regression_count_vs_frontier'])} |"
        )
    if not scenarios.empty:
        lines.extend(
            [
                "",
                "## Registry Scenarios",
                "",
                "| scenario | priority | source-set gate action | first average candidate |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in scenarios.iterrows():
            lines.append(
                f"| `{row['scenario_id']}` | `{row['priority']}` | "
                f"`{row['source_set_gate_action']}` | `{row['first_average_candidate']}` |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `source_set_gate`: `{summary['outputs']['source_set_gate']}`",
            f"- `observed_merge_gaps`: `{summary['outputs']['observed_merge_gaps']}`",
            f"- `recommended_scenarios`: `{summary['outputs']['recommended_scenarios']}`",
            f"- `summary`: `{summary['outputs']['summary']}`",
            f"- `report`: `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    matrix = read_csv(args.matrix)
    if matrix.empty:
        raise ValueError(f"Missing downstream matrix: {args.matrix}")
    source_sets = build_source_set_rows(
        matrix,
        min_avg_frontier_gain=args.min_avg_frontier_gain,
        min_worst_frontier_gain=args.min_worst_frontier_gain,
        task_win_margin=args.task_win_margin,
    )
    observed = build_observed_merge_rows(matrix, source_sets)
    scenarios = build_recommended_scenarios(read_csv(args.scenario_matrix))

    current_sources = [source.strip() for source in args.current_source_set.split("+") if source.strip()]
    current_key = source_key(current_sources)
    current_rows = source_sets[source_sets["source_set_key"] == current_key]
    if current_rows.empty:
        raise ValueError(f"Current source set not found in matrix: {args.current_source_set}")
    current = current_rows.iloc[0].to_dict()
    current_observed = observed[observed["source_set_key"] == current_key] if not observed.empty else pd.DataFrame()
    best_observed: dict[str, Any] = {}
    if not current_observed.empty:
        best_observed = current_observed.sort_values(
            ["avg_gap_to_source_frontier", "worst_gap_to_source_frontier"],
            ascending=[False, False],
        ).iloc[0].to_dict()

    complementary_count = int((source_sets["gate"] == "complementary_source_set_candidate").sum())
    weak_count = int((source_sets["gate"] == "weak_complementarity_needs_more_or_larger_eval").sum())
    dominated_count = int((source_sets["gate"] == "source_dominated_not_averageable_as_final").sum())
    if str(current["gate"]) == "source_dominated_not_averageable_as_final":
        recommended_action = "do_not_expect_average_to_beat_current_source_frontier; use average only as repair_ablation_until_new_complementary_sources_pass_endpoint_eval"
    elif str(current["gate"]) == "weak_complementarity_needs_more_or_larger_eval":
        recommended_action = "run_larger_endpoint_eval_before_spending_final_average_budget"
    else:
        recommended_action = "run_topology_connectivity_and_same_manifest_vllm_eval_for_this_source_set"

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_set_gate.csv"
    observed_path = output_dir / "observed_merge_gaps.csv"
    scenario_path = output_dir / "recommended_scenarios.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    summary = {
        "schema_version": 1,
        "status": "source_set_complementarity_gate_ready",
        "matrix": rel(args.matrix),
        "source_set_count": int(len(source_sets)),
        "observed_merge_count": int(len(observed)),
        "source_dominated_set_count": dominated_count,
        "complementary_source_set_count": complementary_count,
        "weak_complementary_source_set_count": weak_count,
        "min_avg_frontier_gain": float(args.min_avg_frontier_gain),
        "min_worst_frontier_gain": float(args.min_worst_frontier_gain),
        "task_win_margin": float(args.task_win_margin),
        "current_source_set": args.current_source_set,
        "current_source_set_key": current_key,
        "current_gate": current["gate"],
        "current_dominant_source": current["dominant_source"],
        "current_strict_task_winner_count": int(current["strict_task_winner_count"]),
        "current_frontier_avg": fnum(current["frontier_avg"]),
        "current_frontier_worst": fnum(current["frontier_worst"]),
        "current_best_single_avg_source": current["best_single_avg_source"],
        "current_best_single_avg": fnum(current["best_single_avg"]),
        "current_frontier_avg_gain_vs_best_single": fnum(current["frontier_avg_gain_vs_best_single"]),
        "current_frontier_worst_gain_vs_best_single": fnum(current["frontier_worst_gain_vs_best_single"]),
        "current_best_observed_merge": best_observed.get("merge_model"),
        "current_best_observed_variant": best_observed.get("variant"),
        "current_best_observed_avg": fnum(best_observed.get("avg")),
        "current_best_observed_avg_gap_to_frontier": fnum(best_observed.get("avg_gap_to_source_frontier")),
        "current_best_observed_worst_gap_to_frontier": fnum(best_observed.get("worst_gap_to_source_frontier")),
        "recommended_action": recommended_action,
        "outputs": {
            "source_set_gate": rel(source_path),
            "observed_merge_gaps": rel(observed_path),
            "recommended_scenarios": rel(scenario_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }

    source_sets.to_csv(source_path, index=False)
    observed.to_csv(observed_path, index=False)
    scenarios.to_csv(scenario_path, index=False)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, source_sets, observed, scenarios), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen3 source-set complementarity gate for averaging.")
    parser.add_argument("--matrix", type=Path, default=Path("results/fp_downstream_matrix/matrix.csv"))
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("results/qwen_target_model_registry/scenario_matrix.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_source_set_complementarity_gate"))
    parser.add_argument("--current-source-set", default="instruct+coder")
    parser.add_argument("--min-avg-frontier-gain", type=float, default=0.02)
    parser.add_argument("--min-worst-frontier-gain", type=float, default=0.02)
    parser.add_argument("--task-win-margin", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote Qwen3 source-set complementarity gate to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; current={summary['current_source_set']} gate={summary['current_gate']}"
    )


if __name__ == "__main__":
    main()
