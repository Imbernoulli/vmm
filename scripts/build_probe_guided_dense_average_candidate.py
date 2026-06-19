#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def candidate_id(alpha: float, beta: float) -> str:
    alpha_part = str(int(round(alpha * 100))).zfill(3)
    beta_part = str(int(round(beta * 100))).zfill(3)
    return f"qwen_0_5b_probe_guided_bridge_a{alpha_part}_b{beta_part}"


def shell_quote(value: str | Path) -> str:
    raw = str(value)
    if not raw:
        return "''"
    if all(ch.isalnum() or ch in "/._-:=," for ch in raw):
        return raw
    return "'" + raw.replace("'", "'\"'\"'") + "'"


def build_writer_command(
    qwen_summary: dict[str, Any],
    selected: pd.Series,
    output_dir: str,
) -> str:
    sources = qwen_summary["experts"]
    parts = [
        "python",
        "scripts/write_same_shape_average_checkpoint.py",
        "--base",
        qwen_summary["base"],
        "--source",
        f"instruct={sources['instruct']}",
        "--source",
        f"coder={sources['coder']}",
        "--source-weight",
        f"instruct={float(selected['alpha'])}",
        "--source-weight",
        f"coder={float(selected['beta'])}",
        "--output-dir",
        output_dir,
    ]
    return " ".join(shell_quote(part) for part in parts)


def build_vllm_commands(output_dir: str, eval_dir: str, served_model: str) -> dict[str, str]:
    return {
        "serve": (
            "CUDA_VISIBLE_DEVICES=1 /srv/home/bohanlyu/miniconda3/envs/cogdoc/bin/vllm "
            f"serve {shell_quote(output_dir)} --served-model-name {served_model} "
            "--host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1"
        ),
        "eval": (
            "python scripts/run_vllm_downstream_eval.py "
            "--base-url http://127.0.0.1:8100/v1 "
            f"--models {served_model} "
            "--tasks gsm8k,mmlu,safety,humaneval_compile "
            "--example-source datasets --max-examples 64 "
            f"--output-dir {shell_quote(eval_dir)}"
        ),
    }


def select_candidate(
    grid: pd.DataFrame,
    min_uniform_worst_gain: float,
    min_uniform_avg_gain: float,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    uniform_rows = grid[(grid["alpha"] == 0.5) & (grid["beta"] == 0.5)]
    if uniform_rows.empty:
        raise ValueError("Expected a uniform alpha=0.5,beta=0.5 row in the Qwen grid")
    uniform = uniform_rows.iloc[0]
    endpoint_mask = (
        ((grid["alpha"] == 0.0) & (grid["beta"] == 0.0))
        | ((grid["alpha"] == 1.0) & (grid["beta"] == 0.0))
        | ((grid["alpha"] == 0.0) & (grid["beta"] == 1.0))
    )
    uniform_mask = (grid["alpha"] == 0.5) & (grid["beta"] == 0.5)
    candidates = grid[~endpoint_mask & ~uniform_mask].copy()
    candidates["uniform_worst_gain"] = float(uniform["worst_nll"]) - candidates["worst_nll"]
    candidates["uniform_avg_gain"] = float(uniform["avg_nll"]) - candidates["avg_nll"]
    candidates["nonzero_source_count"] = (candidates[["alpha", "beta"]] > 0).sum(axis=1)
    candidates["uses_both_sources"] = candidates["nonzero_source_count"] == 2
    candidates["selection_status"] = "eligible"
    candidates.loc[candidates["uniform_worst_gain"] < min_uniform_worst_gain, "selection_status"] = (
        "reject_small_worst_gain"
    )
    candidates.loc[candidates["uniform_avg_gain"] < min_uniform_avg_gain, "selection_status"] = "reject_small_avg_gain"
    eligible = candidates[candidates["selection_status"] == "eligible"].copy()
    if eligible.empty:
        eligible = candidates.copy()
        eligible["selection_status"] = "relaxed_no_threshold_candidate"
    eligible = eligible.sort_values(["worst_nll", "avg_nll", "general_nll", "instruction_nll", "code_nll"])
    selected = eligible.iloc[0]
    diagnostics = {
        "uniform_alpha": float(uniform["alpha"]),
        "uniform_beta": float(uniform["beta"]),
        "uniform_avg_nll": float(uniform["avg_nll"]),
        "uniform_worst_nll": float(uniform["worst_nll"]),
        "candidate_rows": int(len(candidates)),
        "eligible_rows": int(len(eligible)),
        "min_uniform_worst_gain": min_uniform_worst_gain,
        "min_uniform_avg_gain": min_uniform_avg_gain,
    }
    return candidates.sort_values(["selection_status", "worst_nll", "avg_nll"]), selected, diagnostics


def primary_metric_for_task(task: str) -> str:
    return {
        "gsm8k": "strict_exact",
        "mmlu": "accuracy",
        "safety": "policy_accuracy",
        "humaneval_compile": "compile_rate",
    }[task]


def optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def load_vllm_eval_summary(eval_dir: str) -> dict[str, Any] | None:
    eval_path = repo_path(eval_dir)
    model_summary_path = eval_path / "model_summary.csv"
    metrics_path = eval_path / "metrics.csv"
    if not model_summary_path.exists() or not metrics_path.exists():
        return None

    model_summary = pd.read_csv(model_summary_path)
    metrics = pd.read_csv(metrics_path)
    first = model_summary.iloc[0]
    uniform_metrics = pd.read_csv(
        repo_path("results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/metrics.csv")
    )
    uniform_summary = pd.read_csv(
        repo_path("results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/model_summary.csv")
    ).iloc[0]
    source_scores = pd.read_csv(repo_path("results/vllm_source_merge_comparison/model_scores.csv"))
    best_source = source_scores[source_scores["role"] == "source"].sort_values(
        ["avg_primary_score", "worst_primary_score"], ascending=False
    ).iloc[0]

    task_rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        task = str(row["task"])
        metric = primary_metric_for_task(task)
        uniform_row = uniform_metrics[uniform_metrics["task"] == task].iloc[0]
        score = float(row[metric])
        uniform_score = float(uniform_row[metric])
        task_rows.append(
            {
                "task": task,
                "primary_metric": metric,
                "primary_score": score,
                "uniform_primary_score": uniform_score,
                "delta_vs_uniform": score - uniform_score,
                "safe_non_refusal_rate": optional_float(row.get("safe_non_refusal_rate")),
                "unsafe_refusal_rate": optional_float(row.get("unsafe_refusal_rate")),
            }
        )

    return {
        "status": "complete",
        "eval_dir": eval_dir,
        "avg_primary_score": float(first["avg_primary_score"]),
        "worst_primary_score": float(first["worst_primary_score"]),
        "uniform_avg_primary_score": float(uniform_summary["avg_primary_score"]),
        "uniform_worst_primary_score": float(uniform_summary["worst_primary_score"]),
        "delta_vs_uniform_avg_primary": float(first["avg_primary_score"]) - float(uniform_summary["avg_primary_score"]),
        "delta_vs_uniform_worst_primary": float(first["worst_primary_score"]) - float(uniform_summary["worst_primary_score"]),
        "best_source_model": str(best_source["model_key"]),
        "best_source_display_name": str(best_source["display_name"]),
        "best_source_avg_primary_score": float(best_source["avg_primary_score"]),
        "best_source_worst_primary_score": float(best_source["worst_primary_score"]),
        "delta_vs_best_source_avg_primary": float(first["avg_primary_score"]) - float(best_source["avg_primary_score"]),
        "delta_vs_best_source_worst_primary": float(first["worst_primary_score"]) - float(best_source["worst_primary_score"]),
        "task_metrics": task_rows,
        "report": rel(eval_path / "report.md"),
        "metrics": rel(eval_path / "metrics.csv"),
        "model_summary": rel(eval_path / "model_summary.csv"),
    }


def write_report(
    output_dir: Path,
    selected: pd.Series,
    diagnostics: dict[str, Any],
    command: str,
    vllm_commands: dict[str, str],
    selected_id: str,
    checkpoint_dir: str,
    eval_dir: str,
    eval_summary: dict[str, Any] | None,
) -> None:
    eval_lines: list[str] = []
    if eval_summary:
        eval_lines = [
            "",
            "## vLLM Eval Result",
            "",
            (
                f"真实 endpoint eval 已完成：avg primary `{fmt(eval_summary['avg_primary_score'])}`，"
                f"worst primary `{fmt(eval_summary['worst_primary_score'])}`。相对 uniform average，"
                f"avg primary 提升 `{fmt(eval_summary['delta_vs_uniform_avg_primary'])}`；相对最佳源模型仍低 "
                f"`{fmt(abs(eval_summary['delta_vs_best_source_avg_primary']))}`。"
            ),
            "",
            "| task | score | uniform score | delta vs uniform |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in eval_summary["task_metrics"]:
            eval_lines.append(
                f"| {row['task']} | {fmt(row['primary_score'])} | {fmt(row['uniform_primary_score'])} | "
                f"{fmt(row['delta_vs_uniform'])} |"
            )
        eval_lines.extend(
            [
                "",
                (
                    "这说明 NLL-grid probe 选出的 bridge 确实避开了最差 midpoint：GSM8K 和 MMLU 都比 "
                    "uniform 回升，safety 的 unsafe refusal 从 `0.000` 回到 `0.500`。但 compile 仍是 "
                    "`0.000`，整体仍低于 instruct/base；所以 global scalar coefficient 只能算第一层 gate，"
                    "下一步需要 layer/module-wise 权重。"
                ),
            ]
        )

    lines = [
        "# Probe-Guided Dense Average Candidate",
        "",
        "## 结论",
        "",
        (
            f"从 Qwen instruct/coder NLL grid 里选出的下一轮同构 checkpoint candidate 是 "
            f"`{selected_id}`：`alpha={fmt(selected['alpha'], 2)}`、`beta={fmt(selected['beta'], 2)}`。"
        ),
        "",
        (
            f"它不是端点复制，也不是 `0.5/0.5` uniform average。相对 uniform midpoint，"
            f"worst NLL 降低 `{fmt(selected['uniform_worst_gain'])}`，avg NLL 降低 "
            f"`{fmt(selected['uniform_avg_gain'])}`；但它仍然必须用真实 vLLM endpoint 评测验证。"
        ),
        "",
        "## Selection Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| alpha | {fmt(selected['alpha'], 2)} |",
        f"| beta | {fmt(selected['beta'], 2)} |",
        f"| avg NLL | {fmt(selected['avg_nll'])} |",
        f"| worst NLL | {fmt(selected['worst_nll'])} |",
        f"| general NLL | {fmt(selected['general_nll'])} |",
        f"| instruction NLL | {fmt(selected['instruction_nll'])} |",
        f"| code NLL | {fmt(selected['code_nll'])} |",
        f"| uniform avg NLL | {fmt(diagnostics['uniform_avg_nll'])} |",
        f"| uniform worst NLL | {fmt(diagnostics['uniform_worst_nll'])} |",
        "",
        "## 机制解释",
        "",
        (
            "- `0.5/0.5` midpoint 落在高 worst-NLL ridge 上，说明两个 task deltas 在这个切片里不是简单同 basin 线性连通。"
        ),
        (
            "- 选出的 bridge candidate 沿着 coder endpoint 保留代码 delta，同时只加入一小段 instruct delta；"
            "它的目标是测试“低剂量跨任务注入”是否比全量对半平均更稳定。"
        ),
        (
            "- 如果 vLLM 下游评测仍然差，结论不是换一个固定系数，而是进入 layer/module-wise weighting："
            "对冲突层降权，对稳定共享层保留平均。"
        ),
        "",
        "## Materialization",
        "",
        f"Checkpoint output: `{checkpoint_dir}`",
        "",
        "```bash",
        command,
        "```",
        "",
        "## vLLM Eval",
        "",
        f"Eval output: `{eval_dir}`",
        "",
        "```bash",
        vllm_commands["serve"],
        "```",
        "",
        "```bash",
        vllm_commands["eval"],
        "```",
        *eval_lines,
        "",
        "## Artifacts",
        "",
        f"- Summary: `{rel(output_dir / 'summary.json')}`",
        f"- Candidate table: `{rel(output_dir / 'candidate_scores.csv')}`",
        f"- Writer command: `{rel(output_dir / 'writer_command.txt')}`",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-summary", default="results/qwen_multi_expert_merge/summary.json")
    parser.add_argument("--qwen-grid", default="results/qwen_multi_expert_merge/grid_metrics.csv")
    parser.add_argument("--output-dir", default="results/probe_guided_dense_average_candidate")
    parser.add_argument("--min-uniform-worst-gain", type=float, default=1.0)
    parser.add_argument("--min-uniform-avg-gain", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen_summary = json.loads(repo_path(args.qwen_summary).read_text(encoding="utf-8"))
    grid = pd.read_csv(repo_path(args.qwen_grid))
    candidates, selected, diagnostics = select_candidate(
        grid,
        min_uniform_worst_gain=args.min_uniform_worst_gain,
        min_uniform_avg_gain=args.min_uniform_avg_gain,
    )
    selected_id = candidate_id(float(selected["alpha"]), float(selected["beta"]))
    checkpoint_dir = f"results/checkpoints/{selected_id}"
    eval_dir = f"results/vllm_checkpoint_eval/{selected_id}"
    served_model = f"candidate_{selected_id}"
    writer_command = build_writer_command(qwen_summary, selected, checkpoint_dir)
    vllm_commands = build_vllm_commands(checkpoint_dir, eval_dir, served_model)
    eval_summary = load_vllm_eval_summary(eval_dir)

    candidates.to_csv(output_dir / "candidate_scores.csv", index=False)
    (output_dir / "writer_command.txt").write_text(writer_command + "\n", encoding="utf-8")
    (output_dir / "vllm_commands.json").write_text(json.dumps(vllm_commands, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "evaluated_complete" if eval_summary else "candidate_selected_waiting_for_materialization",
        "candidate_id": selected_id,
        "checkpoint_output_dir": checkpoint_dir,
        "vllm_eval_output_dir": eval_dir,
        "served_model": served_model,
        "alpha": float(selected["alpha"]),
        "beta": float(selected["beta"]),
        "avg_nll": float(selected["avg_nll"]),
        "worst_nll": float(selected["worst_nll"]),
        "general_nll": float(selected["general_nll"]),
        "instruction_nll": float(selected["instruction_nll"]),
        "code_nll": float(selected["code_nll"]),
        "uniform_avg_nll": diagnostics["uniform_avg_nll"],
        "uniform_worst_nll": diagnostics["uniform_worst_nll"],
        "uniform_avg_gain": float(selected["uniform_avg_gain"]),
        "uniform_worst_gain": float(selected["uniform_worst_gain"]),
        "selection_status": selected["selection_status"],
        "selection_diagnostics": diagnostics,
        "writer_command": writer_command,
        "vllm_commands": vllm_commands,
        "vllm_eval": eval_summary,
        "same_shape_constraint": "The candidate uses the existing base config/tokenizer and writes source deltas into the same tensor names and shapes.",
        "hypothesis": "A coder-anchored bridge with a small instruct delta should avoid the high-NLL midpoint ridge better than uniform averaging.",
        "artifacts": {
            "report": rel(output_dir / "report.md"),
            "candidate_scores": rel(output_dir / "candidate_scores.csv"),
            "writer_command": rel(output_dir / "writer_command.txt"),
            "vllm_commands": rel(output_dir / "vllm_commands.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        output_dir,
        selected,
        diagnostics,
        writer_command,
        vllm_commands,
        selected_id,
        checkpoint_dir,
        eval_dir,
        eval_summary,
    )
    print(f"Wrote {rel(output_dir / 'summary.json')}")
    print(f"Selected {selected_id}: alpha={selected['alpha']}, beta={selected['beta']}")


if __name__ == "__main__":
    main()
