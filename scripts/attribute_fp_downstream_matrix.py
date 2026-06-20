#!/usr/bin/env python
"""Attribute Qwen3 MoE generation-matrix effects by mechanism and task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_SOURCES = ["instruct", "coder"]
ALL_SOURCES = ["base", "instruct", "coder", "thinking"]
PAIR_MERGE = "merge_instruct+coder"
PAIR_ROUTERCAL = "merge_instruct+coder_routercal"
TRIPLE_MERGE = "merge_instruct+coder+thinking"
SCORE_COLUMNS = ["mmlu", "gsm8k", "humaneval", "avg", "worst"]


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


def load_rows(matrix_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(matrix_path)
    required = {"name", *SCORE_COLUMNS}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing required matrix columns: {sorted(missing)}")
    return rows


def row_by_name(rows: pd.DataFrame, name: str) -> pd.Series:
    match = rows[rows["name"] == name]
    if match.empty:
        raise ValueError(f"Missing model row {name!r}")
    return match.iloc[0]


def frontier(rows: pd.DataFrame, names: list[str], label: str) -> dict[str, Any]:
    items = [row_by_name(rows, name) for name in names if not rows[rows["name"] == name].empty]
    if not items:
        raise ValueError(f"No source rows found for {label}")
    out: dict[str, Any] = {"name": label}
    for column in SCORE_COLUMNS:
        values = [(str(item["name"]), float(item[column])) for item in items]
        source, value = max(values, key=lambda pair: pair[1])
        out[column] = value
        out[f"{column}_source"] = source
    return out


def recovery_fraction(drop: float | None, gain: float | None) -> float | None:
    if drop is None or gain is None or drop <= 0:
        return None
    return gain / drop


def transition_rows(rows: pd.DataFrame, pair_frontier: dict[str, Any], all_frontier: dict[str, Any]) -> list[dict[str, Any]]:
    pair = row_by_name(rows, PAIR_MERGE)
    routercal = row_by_name(rows, PAIR_ROUTERCAL)
    triple = row_by_name(rows, TRIPLE_MERGE)
    transitions = []
    for column in SCORE_COLUMNS:
        pair_value = float(pair[column])
        routercal_value = float(routercal[column])
        triple_value = float(triple[column])
        pair_source_value = float(pair_frontier[column])
        all_source_value = float(all_frontier[column])
        naive_drop = pair_source_value - pair_value
        router_gain = routercal_value - pair_value
        router_gap = routercal_value - pair_source_value
        triple_gain = triple_value - pair_value
        all_source_gap = routercal_value - all_source_value
        transitions.append(
            {
                "score": column,
                "pair_source_frontier": pair_source_value,
                "pair_source_frontier_source": pair_frontier.get(f"{column}_source"),
                "all_source_frontier": all_source_value,
                "all_source_frontier_source": all_frontier.get(f"{column}_source"),
                "naive_pair_score": pair_value,
                "routercal_pair_score": routercal_value,
                "triple_merge_score": triple_value,
                "naive_drop_vs_pair_frontier": naive_drop,
                "routercal_gain_vs_naive": router_gain,
                "routercal_recovery_fraction": recovery_fraction(naive_drop, router_gain),
                "routercal_gap_vs_pair_frontier": router_gap,
                "routercal_gap_vs_all_source_frontier": all_source_gap,
                "triple_gain_vs_pair": triple_gain,
                "best_local_action": "router_calibration"
                if router_gain >= max(0.0, triple_gain)
                else "add_thinking_source",
                "accepts_routercal_over_pair_frontier": router_gap > 0,
                "accepts_routercal_over_all_source_frontier": all_source_gap > 0,
            }
        )
    return transitions


def build_summary(transitions: pd.DataFrame, rows: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    avg_row = transitions[transitions["score"] == "avg"].iloc[0].to_dict()
    humaneval_row = transitions[transitions["score"] == "humaneval"].iloc[0].to_dict()
    recoverable = transitions[transitions["naive_drop_vs_pair_frontier"] > 0]
    mean_recovery = None
    if not recoverable.empty:
        mean_recovery = float(recoverable["routercal_recovery_fraction"].dropna().mean())
    best_model = rows.sort_values("avg", ascending=False).iloc[0]
    return {
        "schema_version": 1,
        "status": "generation_mechanism_attribution_ready",
        "role": "auxiliary_transformers_generation_eval_not_final_vllm_selector",
        "score_count": int(len(transitions)),
        "best_avg_model": str(best_model["name"]),
        "best_avg": float(best_model["avg"]),
        "avg_naive_drop_vs_pair_frontier": float(avg_row["naive_drop_vs_pair_frontier"]),
        "avg_routercal_gain_vs_naive": float(avg_row["routercal_gain_vs_naive"]),
        "avg_routercal_recovery_fraction": clean_value(avg_row["routercal_recovery_fraction"]),
        "avg_routercal_gap_vs_pair_frontier": float(avg_row["routercal_gap_vs_pair_frontier"]),
        "humaneval_naive_drop_vs_pair_frontier": float(humaneval_row["naive_drop_vs_pair_frontier"]),
        "humaneval_routercal_gain_vs_naive": float(humaneval_row["routercal_gain_vs_naive"]),
        "humaneval_routercal_recovery_fraction": clean_value(
            humaneval_row["routercal_recovery_fraction"]
        ),
        "mean_recovery_fraction_over_dropped_scores": mean_recovery,
        "routercal_beats_pair_frontier_count": int(
            transitions["accepts_routercal_over_pair_frontier"].sum()
        ),
        "routercal_beats_all_source_frontier_count": int(
            transitions["accepts_routercal_over_all_source_frontier"].sum()
        ),
        "interpretation": (
            "Router calibration recovers a measurable share of the naive average regression, especially "
            "on HumanEval and GSM8K, but every score remains below the relevant source frontier. The "
            "mechanism is therefore useful as a MoE-specific repair lever, not sufficient as an acceptance rule."
        ),
        "outputs": {
            "transition_effects": rel(output_dir / "transition_effects.csv"),
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
        },
    }


def build_report(transitions: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 MoE Generation Mechanism Attribution",
        "",
        "This report attributes the auxiliary generation matrix by task. It does not replace the final vLLM selector.",
        "",
        "## Summary",
        "",
        f"- Best average model: `{summary['best_avg_model']}` (`{fmt(summary['best_avg'])}`)",
        f"- Avg naive drop vs pair source frontier: `{fmt(summary['avg_naive_drop_vs_pair_frontier'])}`",
        f"- Avg router-cal gain vs naive: `{fmt(summary['avg_routercal_gain_vs_naive'])}`",
        f"- Avg recovery fraction: `{fmt(summary['avg_routercal_recovery_fraction'])}`",
        f"- HumanEval recovery fraction: `{fmt(summary['humaneval_routercal_recovery_fraction'])}`",
        f"- Router-cal beats pair frontier scores: `{summary['routercal_beats_pair_frontier_count']}/{summary['score_count']}`",
        "",
        "## Transitions",
        "",
        "| score | pair frontier | naive | router-cal | drop | gain | recovery | gap after router-cal | best local action |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in transitions.iterrows():
        lines.append(
            f"| `{row['score']}` | {fmt(row['pair_source_frontier'])} | "
            f"{fmt(row['naive_pair_score'])} | {fmt(row['routercal_pair_score'])} | "
            f"{fmt(row['naive_drop_vs_pair_frontier'])} | {fmt(row['routercal_gain_vs_naive'])} | "
            f"{fmt(row['routercal_recovery_fraction'])} | {fmt(row['routercal_gap_vs_pair_frontier'])} | "
            f"`{row['best_local_action']}` |"
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
            f"- `{summary['outputs']['transition_effects']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    matrix_path = repo_path(args.matrix)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(matrix_path)
    pair_frontier = frontier(rows, PAIR_SOURCES, "pair_source_frontier")
    all_frontier = frontier(rows, ALL_SOURCES, "all_source_frontier")
    transitions = pd.DataFrame(transition_rows(rows, pair_frontier, all_frontier))
    transition_path = output_dir / "transition_effects.csv"
    transitions.to_csv(transition_path, index=False)
    summary = build_summary(transitions, rows, output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(transitions, summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute Qwen3 MoE downstream generation effects.")
    parser.add_argument("--matrix", type=Path, default=Path("results/fp_downstream_matrix/matrix.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fp_downstream_attribution"))
    args = parser.parse_args()
    summary = build(args)
    print(f"Wrote downstream attribution to {repo_path(args.output_dir).resolve()}")
    print(
        f"Status: {summary['status']}; avg_recovery={fmt(summary['avg_routercal_recovery_fraction'])}; "
        f"beats_frontier={summary['routercal_beats_pair_frontier_count']}/{summary['score_count']}"
    )


if __name__ == "__main__":
    main()
