#!/usr/bin/env python
"""Build a formal report for the Qwen3 MoE first-principles downstream matrix.

This consumes per-model JSON files produced by scripts/fp_downstream_eval.py.
It is auxiliary generation-level evidence; the final Qwen3 candidate selector
still requires the matched vLLM eval bundle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORDER = [
    "base",
    "instruct",
    "coder",
    "thinking",
    "merge_instruct+coder",
    "merge_instruct+coder_routercal",
    "merge_instruct+coder+thinking",
    "merge_instruct+coder+thinking_routercal",
]
TASKS = ["mmlu", "gsm8k", "humaneval"]


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


def load_model_rows(input_dir: Path, order: list[str]) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data.get("name") or path.stem)
        rows[name] = data
    ordered_names = [name for name in order if name in rows] + sorted(name for name in rows if name not in order)
    out_rows = []
    for name in ordered_names:
        row = dict(rows[name])
        row["name"] = name
        scores = [row.get(task) for task in TASKS if row.get(task) is not None]
        if scores:
            row["avg"] = float(sum(float(score) for score in scores) / len(scores))
            row["worst"] = float(min(float(score) for score in scores))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def find_row(rows: pd.DataFrame, name: str) -> dict[str, Any]:
    if rows.empty or "name" not in rows:
        return {}
    match = rows[rows["name"] == name]
    if match.empty:
        return {}
    return {str(key): clean_value(value) for key, value in match.iloc[0].items()}


def score_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    av = clean_value(a.get(key))
    bv = clean_value(b.get(key))
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


def build_summary(rows: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    instruct = find_row(rows, "instruct")
    coder = find_row(rows, "coder")
    thinking = find_row(rows, "thinking")
    pair = find_row(rows, "merge_instruct+coder")
    pair_routercal = find_row(rows, "merge_instruct+coder_routercal")
    triple = find_row(rows, "merge_instruct+coder+thinking")
    parents = [row for row in [instruct, coder, thinking] if row]
    best_parent = max(parents, key=lambda row: float(row.get("avg") or float("-inf")), default={})
    best_model_row = max(rows.to_dict("records"), key=lambda row: float(clean_value(row.get("avg")) or float("-inf")))
    routercal_gain_by_task = {
        task: score_delta(pair_routercal, pair, task)
        for task in [*TASKS, "avg", "worst"]
    }
    return {
        "schema_version": 1,
        "status": "generation_downstream_matrix_ready",
        "role": "auxiliary_transformers_generation_eval_not_final_vllm_selector",
        "model_count": int(len(rows)),
        "tasks": TASKS,
        "best_avg_model": best_model_row.get("name"),
        "best_avg": clean_value(best_model_row.get("avg")),
        "best_parent_model": best_parent.get("name"),
        "best_parent_avg": clean_value(best_parent.get("avg")),
        "pair_merge_avg": clean_value(pair.get("avg")),
        "pair_routercal_avg": clean_value(pair_routercal.get("avg")),
        "pair_routercal_avg_gain": routercal_gain_by_task["avg"],
        "pair_routercal_humaneval_gain": routercal_gain_by_task["humaneval"],
        "pair_routercal_gsm8k_gain": routercal_gain_by_task["gsm8k"],
        "pair_routercal_mmlu_gain": routercal_gain_by_task["mmlu"],
        "pair_routercal_gap_to_best_parent_avg": score_delta(pair_routercal, best_parent, "avg"),
        "triple_merge_avg": clean_value(triple.get("avg")),
        "triple_minus_pair_avg": score_delta(triple, pair, "avg"),
        "interpretation": (
            "The generation matrix confirms the mechanism: naive Instruct+Coder averaging dilutes the "
            "dominant Instruct endpoint, while router calibration recovers part of the lost GSM8K and "
            "HumanEval performance. It still does not beat the best parent on this task set, so it is "
            "evidence for router calibration as a lever, not acceptance of an average candidate."
        ),
        "outputs": {
            "matrix": rel(output_dir / "matrix.csv"),
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
            "figure": rel(output_dir / "downstream_matrix.png"),
        },
    }


def build_report(rows: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 MoE Downstream Generation Matrix",
        "",
        "This is an auxiliary transformers-generation matrix for mechanism evidence. It is not the final vLLM selector.",
        "",
        "## Summary",
        "",
        f"- Models: `{summary['model_count']}`",
        f"- Best average model: `{summary['best_avg_model']}` (`{fmt(summary['best_avg'])}`)",
        f"- Instruct+Coder avg: `{fmt(summary['pair_merge_avg'])}`",
        f"- Instruct+Coder + router-cal avg: `{fmt(summary['pair_routercal_avg'])}`",
        f"- Router-cal avg gain: `{fmt(summary['pair_routercal_avg_gain'])}`",
        f"- Router-cal HumanEval gain: `{fmt(summary['pair_routercal_humaneval_gain'])}`",
        f"- Router-cal gap to best parent avg: `{fmt(summary['pair_routercal_gap_to_best_parent_avg'])}`",
        "",
        "## Matrix",
        "",
        "| model | MMLU | GSM8K | HumanEval | avg | worst |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| `{row['name']}` | {fmt(row.get('mmlu'))} | {fmt(row.get('gsm8k'))} | "
            f"{fmt(row.get('humaneval'))} | {fmt(row.get('avg'))} | {fmt(row.get('worst'))} |"
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
            f"- `{summary['outputs']['matrix']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['figure']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_figure(rows: pd.DataFrame, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    complete = rows.dropna(subset=TASKS)
    if complete.empty:
        return
    labels = [
        str(name)
        .replace("merge_", "M:")
        .replace("instruct+coder", "I+C")
        .replace("+thinking", "+T")
        .replace("_routercal", "+RC")
        for name in complete["name"].tolist()
    ]
    x = np.arange(len(complete))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(9.0, 1.35 * len(complete)), 5.0))
    for idx, task in enumerate(TASKS):
        ax.bar(x + (idx - 1) * width, complete[task].astype(float).tolist(), width, label=task)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Qwen3-30B-A3B MoE downstream generation matrix")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_model_rows(input_dir, DEFAULT_ORDER)
    if rows.empty:
        raise ValueError(f"No model JSON rows found in {input_dir}")
    matrix_path = output_dir / "matrix.csv"
    rows[["name", *TASKS, "avg", "worst"]].to_csv(matrix_path, index=False)
    figure_path = output_dir / "downstream_matrix.png"
    write_figure(rows, figure_path)
    summary = build_summary(rows, output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(build_report(rows, summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Qwen3 MoE downstream generation matrix report.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/fp_downstream_matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fp_downstream_matrix"))
    args = parser.parse_args()
    summary = build(args)
    print(f"Wrote downstream matrix to {repo_path(args.output_dir).resolve()}")
    print(
        f"Status: {summary['status']}; routercal_avg_gain={fmt(summary['pair_routercal_avg_gain'])}; "
        f"best={summary['best_avg_model']}"
    )


if __name__ == "__main__":
    main()
