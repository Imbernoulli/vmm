#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "results" / "vllm_checkpoint_eval"

PRIMARY_METRICS = {
    "gsm8k": "strict_exact",
    "mmlu": "accuracy",
    "safety": "policy_accuracy",
    "humaneval_compile": "compile_rate",
}

MODEL_DIRS = [
    {
        "model_key": "source_qwen_0_5b_base",
        "display_name": "Qwen2.5-0.5B Base",
        "role": "source",
        "directory": "source_qwen_0_5b_base",
    },
    {
        "model_key": "source_qwen_0_5b_instruct",
        "display_name": "Qwen2.5-0.5B Instruct",
        "role": "source",
        "directory": "source_qwen_0_5b_instruct",
    },
    {
        "model_key": "source_qwen_0_5b_coder",
        "display_name": "Qwen2.5-Coder-0.5B-Instruct",
        "role": "source",
        "directory": "source_qwen_0_5b_coder",
    },
    {
        "model_key": "qwen_0_5b_instruct_coder_uniform_average",
        "display_name": "Instruct/Coder 0.5/0.5 Uniform Average",
        "role": "merge",
        "directory": "qwen_0_5b_instruct_coder_uniform_average",
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
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return clean_value(value)


def fmt(value: Any, digits: int = 3) -> str:
    value = clean_value(value)
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def load_eval(meta: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir = EVAL_ROOT / meta["directory"]
    summary_path = output_dir / "summary.json"
    model_summary_path = output_dir / "model_summary.csv"
    metrics_path = output_dir / "metrics.csv"
    missing = [path for path in (summary_path, model_summary_path, metrics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing vLLM eval artifacts for {meta['model_key']}: {missing}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_summary = pd.read_csv(model_summary_path)
    metrics = pd.read_csv(metrics_path)
    if model_summary.empty:
        raise ValueError(f"Empty model_summary.csv for {meta['model_key']}")

    first = model_summary.iloc[0].to_dict()
    model_row = {
        "model_key": meta["model_key"],
        "display_name": meta["display_name"],
        "role": meta["role"],
        "eval_dir": rel(output_dir),
        "status": summary.get("status"),
        "served_model": first.get("model"),
        "task_count": int(first.get("task_count", len(metrics))),
        "avg_primary_score": clean_value(first.get("avg_primary_score")),
        "worst_primary_score": clean_value(first.get("worst_primary_score")),
    }

    task_rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        task = str(row["task"])
        primary_metric = PRIMARY_METRICS.get(task)
        if primary_metric is None:
            continue
        score = clean_value(row.get(primary_metric))
        task_rows.append(
            {
                "model_key": meta["model_key"],
                "display_name": meta["display_name"],
                "role": meta["role"],
                "task": task,
                "examples": int(row["examples"]),
                "primary_metric": primary_metric,
                "primary_score": score,
                "strict_exact": clean_value(row.get("strict_exact")),
                "loose_exact": clean_value(row.get("loose_exact")),
                "accuracy": clean_value(row.get("accuracy")),
                "safe_non_refusal_rate": clean_value(row.get("safe_non_refusal_rate")),
                "unsafe_refusal_rate": clean_value(row.get("unsafe_refusal_rate")),
                "policy_accuracy": clean_value(row.get("policy_accuracy")),
                "compile_rate": clean_value(row.get("compile_rate")),
            }
        )
    return model_row, task_rows


def add_delta_columns(model_scores: pd.DataFrame, task_metrics: pd.DataFrame) -> pd.DataFrame:
    source_scores = model_scores[model_scores["role"] == "source"]
    best_source_avg = float(source_scores["avg_primary_score"].max())
    best_source_worst = float(source_scores["worst_primary_score"].max())
    model_scores["delta_vs_best_source_avg_primary"] = model_scores["avg_primary_score"] - best_source_avg
    model_scores["delta_vs_best_source_worst_primary"] = model_scores["worst_primary_score"] - best_source_worst

    best_by_task = (
        task_metrics[task_metrics["role"] == "source"]
        .groupby("task", as_index=False)["primary_score"]
        .max()
        .rename(columns={"primary_score": "best_source_task_score"})
    )
    task_metrics_with_delta = task_metrics.merge(best_by_task, on="task", how="left")
    task_metrics_with_delta["delta_vs_best_source_task_score"] = (
        task_metrics_with_delta["primary_score"] - task_metrics_with_delta["best_source_task_score"]
    )
    return task_metrics_with_delta


def build_summary(model_scores: pd.DataFrame, task_metrics: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    source_scores = model_scores[model_scores["role"] == "source"].copy()
    merge_scores = model_scores[model_scores["role"] == "merge"].copy()
    if len(merge_scores) != 1:
        raise ValueError("Expected exactly one merge row")
    merge = merge_scores.iloc[0]
    best_source = source_scores.sort_values(["avg_primary_score", "worst_primary_score"], ascending=False).iloc[0]

    best_source_by_task: dict[str, Any] = {}
    merge_by_task: dict[str, Any] = {}
    for task, task_df in task_metrics.groupby("task"):
        source_task = task_df[task_df["role"] == "source"].sort_values("primary_score", ascending=False).iloc[0]
        merge_task = task_df[task_df["role"] == "merge"].iloc[0]
        best_source_by_task[task] = {
            "model_key": source_task["model_key"],
            "display_name": source_task["display_name"],
            "primary_metric": source_task["primary_metric"],
            "primary_score": clean_value(source_task["primary_score"]),
        }
        merge_by_task[task] = {
            "primary_metric": merge_task["primary_metric"],
            "primary_score": clean_value(merge_task["primary_score"]),
            "delta_vs_best_source_task_score": clean_value(merge_task["delta_vs_best_source_task_score"]),
        }

    source_better_count = int((source_scores["avg_primary_score"] > float(merge["avg_primary_score"])).sum())
    status = "merge_underperforms_all_sources" if source_better_count == len(source_scores) else "merge_not_dominated_by_sources"
    merge_rank = int(
        model_scores["avg_primary_score"].rank(method="min", ascending=False)[merge.name]
    )

    return {
        "schema_version": 1,
        "status": status,
        "source_model_count": int(len(source_scores)),
        "merge_model": str(merge["model_key"]),
        "merge_rank_by_avg_primary": merge_rank,
        "source_models_better_than_merge_count": source_better_count,
        "best_source_model": str(best_source["model_key"]),
        "best_source_display_name": str(best_source["display_name"]),
        "best_source_avg_primary_score": clean_value(best_source["avg_primary_score"]),
        "best_source_worst_primary_score": clean_value(best_source["worst_primary_score"]),
        "merge_avg_primary_score": clean_value(merge["avg_primary_score"]),
        "merge_worst_primary_score": clean_value(merge["worst_primary_score"]),
        "merge_delta_vs_best_source_avg_primary": clean_value(merge["delta_vs_best_source_avg_primary"]),
        "merge_delta_vs_best_source_worst_primary": clean_value(merge["delta_vs_best_source_worst_primary"]),
        "best_source_by_task": json_safe(best_source_by_task),
        "merge_by_task": json_safe(merge_by_task),
        "mechanism_findings": [
            "Uniform averaging is dominated by all three source endpoints on avg primary score in this slice.",
            "The base model wins the aggregate mostly because HumanEval compile is a syntax/loadability metric here, not because it is instruction-following aligned.",
            "The uniform merge loses MMLU and GSM8K relative to the best endpoint and collapses HumanEval compile to zero, matching the earlier NLL ridge probe.",
            "Safety policy accuracy of 0.500 is not a benign middle point: the merge keeps safe non-refusal at 1.0 but unsafe refusal at 0.0, so it almost never refuses unsafe prompts.",
        ],
        "recommended_next_method": "probe_guided_same_shape_average",
        "next_method_hypothesis": (
            "Use source/task probes to learn non-uniform coefficients and module groups; do not average "
            "instruction/coder deltas uniformly when endpoint skills and safety behavior are not connected in the same basin."
        ),
        "artifacts": {
            "report": rel(output_dir / "report.md"),
            "model_scores": rel(output_dir / "model_scores.csv"),
            "task_metrics": rel(output_dir / "task_metrics.csv"),
            "figure": rel(output_dir / "source_vs_merge_primary_scores.png"),
        },
    }


def write_report(
    output_dir: Path,
    model_scores: pd.DataFrame,
    task_metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    merge = model_scores[model_scores["role"] == "merge"].iloc[0]
    source_scores = model_scores[model_scores["role"] == "source"].sort_values(
        ["avg_primary_score", "worst_primary_score"], ascending=False
    )
    best_source = source_scores.iloc[0]
    merge_task = task_metrics[task_metrics["role"] == "merge"].copy()

    lines = [
        "# Qwen 0.5B Source-vs-Merge vLLM Comparison",
        "",
        "## 结论",
        "",
        (
            f"同一套 vLLM 下游评测里，`{merge['display_name']}` 不是有效折中点："
            f"avg primary `{fmt(merge['avg_primary_score'])}`，"
            f"比最佳源模型 `{best_source['display_name']}` 的 `{fmt(best_source['avg_primary_score'])}` "
            f"低 `{fmt(abs(summary['merge_delta_vs_best_source_avg_primary']))}`；"
            f"worst primary `{fmt(merge['worst_primary_score'])}`，仍然是零。"
        ),
        "",
        (
            "这说明这里的失败不是静态 probe 的误判，而是在真实 endpoint 上也成立："
            "uniform average 没有同时保留 instruct/coder/base 的能力，反而落在一个多任务都不强的区域。"
        ),
        "",
        "## Model Scores",
        "",
        "| model | role | avg primary | worst primary | delta vs best source avg |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in model_scores.sort_values("avg_primary_score", ascending=False).iterrows():
        lines.append(
            f"| {row['display_name']} | {row['role']} | {fmt(row['avg_primary_score'])} | "
            f"{fmt(row['worst_primary_score'])} | {fmt(row['delta_vs_best_source_avg_primary'])} |"
        )

    lines.extend(
        [
            "",
            "## Task-Level Primary Metrics",
            "",
            "| task | merge score | best source | best source score | merge delta |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for _, row in merge_task.sort_values("task").iterrows():
        task = row["task"]
        best = summary["best_source_by_task"][task]
        lines.append(
            f"| {task} | {fmt(row['primary_score'])} | {best['display_name']} | "
            f"{fmt(best['primary_score'])} | {fmt(row['delta_vs_best_source_task_score'])} |"
        )

    safety = merge_task[merge_task["task"] == "safety"].iloc[0]
    lines.extend(
        [
            "",
            "## 机理解释",
            "",
            (
                "- Base 的 aggregate 分数最高，主要来自 `humaneval_compile=0.609`；这个指标只检查生成片段能否编译，"
                "不等价于 instruction/code benchmark 的真实 pass rate，也不能说明 base 已经是目标平均模型。"
            ),
            (
                "- Instruct 在 MMLU 和 safety 上强于 coder；coder 的 GSM8K strict 只高于 instruct，"
                "但这个 64-sample slice 里 base 仍是 GSM8K 和 compile 的最高点。这些 endpoint skill "
                "没有被 `0.5/0.5` 权重同时继承。"
            ),
            (
                f"- Uniform merge 的 safety policy accuracy 是 `{fmt(safety['primary_score'])}`，但 safe_non_refusal "
                f"是 `{fmt(safety['safe_non_refusal_rate'])}`、unsafe_refusal 是 `{fmt(safety['unsafe_refusal_rate'])}`。"
                "这不是安全行为变好，而是几乎不拒绝 unsafe prompts。"
            ),
            (
                "- 这个结果和前面的 Qwen multi-expert NLL plane 一致：`alpha=0.5,beta=0.5` 位于高 worst-NLL ridge，"
                "因此真实生成评测也退化。"
            ),
            (
                "- 下一步的统一算法不能只问“哪个算法在哪个场景最好”，而要把 source endpoint ability、"
                "任务向量连通性、模块级 conflict、输出空间投影残差、MoE router/expert load 一起作为 gate，"
                "再写回同构 checkpoint。"
            ),
            "",
            "## Artifacts",
            "",
            f"- Model score CSV: `{rel(output_dir / 'model_scores.csv')}`",
            f"- Task metric CSV: `{rel(output_dir / 'task_metrics.csv')}`",
            f"- Figure: `{rel(output_dir / 'source_vs_merge_primary_scores.png')}`",
            f"- Summary JSON: `{rel(output_dir / 'summary.json')}`",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_figure(output_dir: Path, model_scores: pd.DataFrame, task_metrics: pd.DataFrame) -> None:
    ordered = model_scores.sort_values("avg_primary_score", ascending=True)
    colors = ["#6b7280" if role == "source" else "#c2410c" for role in ordered["role"]]
    labels = list(ordered["display_name"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].barh(labels, ordered["avg_primary_score"], color=colors)
    axes[0].set_xlabel("Avg primary score")
    axes[0].set_title("Aggregate")
    axes[0].set_xlim(0.0, max(0.45, float(model_scores["avg_primary_score"].max()) * 1.15))
    for idx, value in enumerate(ordered["avg_primary_score"]):
        axes[0].text(float(value) + 0.006, idx, fmt(value), va="center", fontsize=9)

    pivot = task_metrics.pivot(index="task", columns="display_name", values="primary_score")
    pivot = pivot.loc[["gsm8k", "mmlu", "safety", "humaneval_compile"]]
    pivot.plot(kind="bar", ax=axes[1], width=0.78)
    axes[1].set_ylabel("Primary score")
    axes[1].set_xlabel("Task")
    axes[1].set_title("Task primary metrics")
    axes[1].set_ylim(0.0, 0.70)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Qwen 0.5B source endpoints vs uniform average under vLLM", fontsize=13)
    fig.savefig(output_dir / "source_vs_merge_primary_scores.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/vllm_source_merge_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for meta in MODEL_DIRS:
        model_row, loaded_task_rows = load_eval(meta)
        model_rows.append(model_row)
        task_rows.extend(loaded_task_rows)

    model_scores = pd.DataFrame(model_rows)
    task_metrics = pd.DataFrame(task_rows)
    task_metrics = add_delta_columns(model_scores, task_metrics)
    model_scores = model_scores.sort_values("avg_primary_score", ascending=False).reset_index(drop=True)
    task_metrics = task_metrics.sort_values(["task", "role", "model_key"]).reset_index(drop=True)

    summary = build_summary(model_scores, task_metrics, output_dir)
    model_scores.to_csv(output_dir / "model_scores.csv", index=False)
    task_metrics.to_csv(output_dir / "task_metrics.csv", index=False)
    write_figure(output_dir, model_scores, task_metrics)
    write_report(output_dir, model_scores, task_metrics, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {rel(output_dir / 'summary.json')}")
    print(f"Wrote {rel(output_dir / 'report.md')}")


if __name__ == "__main__":
    main()
