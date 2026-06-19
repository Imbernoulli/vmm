#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import Counter
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
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def shell_quote(value: str | Path) -> str:
    raw = str(value)
    if not raw:
        return "''"
    if all(ch.isalnum() or ch in "/._-:=," for ch in raw):
        return raw
    return "'" + raw.replace("'", "'\"'\"'") + "'"


def replace_option(parts: list[str], option: str, value: str) -> list[str]:
    out: list[str] = []
    idx = 0
    replaced = False
    while idx < len(parts):
        if parts[idx] == option and idx + 1 < len(parts):
            out.extend([option, value])
            idx += 2
            replaced = True
            continue
        out.append(parts[idx])
        idx += 1
    if not replaced:
        out.extend([option, value])
    return out


def remove_flag(parts: list[str], flag: str) -> list[str]:
    return [part for part in parts if part != flag]


def add_option_before_output(parts: list[str], option: str, value: str) -> list[str]:
    if option in parts:
        return parts
    try:
        output_idx = parts.index("--output-dir")
    except ValueError:
        return parts + [option, value]
    return parts[:output_idx] + [option, value] + parts[output_idx:]


def build_writer_command(
    base_command: str,
    *,
    method_rule_file: Path,
    output_dir: str,
    dry_run: bool,
) -> str:
    parts = shlex.split(base_command)
    parts = remove_flag(parts, "--dry-run")
    parts = add_option_before_output(parts, "--tensor-method-rule-file", rel(method_rule_file))
    parts = replace_option(parts, "--output-dir", output_dir)
    if dry_run:
        parts.append("--dry-run")
    return " ".join(shell_quote(part) for part in parts)


def build_vllm_commands(output_dir: str, eval_dir: str, served_model: str) -> dict[str, str]:
    return {
        "serve": (
            "CUDA_VISIBLE_DEVICES=1 /srv/home/bohanlyu/miniconda3/envs/cogdoc/bin/vllm "
            f"serve {shell_quote(output_dir)} --served-model-name {served_model} "
            "--host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 1"
        ),
        "eval": (
            "python scripts/run_vllm_downstream_eval.py --base-url http://127.0.0.1:8100/v1 "
            f"--models {served_model} --tasks gsm8k,mmlu,safety,humaneval_compile "
            "--example-source datasets --max-examples 64 "
            f"--output-dir {shell_quote(eval_dir)}"
        ),
    }


def primary_metric(row: dict[str, Any]) -> tuple[str, float | None]:
    task = str(row.get("task"))
    if task == "gsm8k":
        return "strict_exact", maybe_float(row.get("strict_exact"))
    if task == "mmlu":
        return "accuracy", maybe_float(row.get("accuracy"))
    if task == "safety":
        return "policy_accuracy", maybe_float(row.get("policy_accuracy"))
    if task == "humaneval_compile":
        return "compile_rate", maybe_float(row.get("compile_rate"))
    return "score", None


def load_vllm_eval(eval_dir: str, bridge_summary: dict[str, Any]) -> dict[str, Any] | None:
    eval_root = repo_path(eval_dir)
    eval_summary_path = eval_root / "summary.json"
    if not eval_summary_path.exists():
        return None
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    model_summary = eval_summary.get("model_summary") or []
    model_row = model_summary[0] if model_summary else {}
    source_merge_path = repo_path("results/vllm_source_merge_comparison/summary.json")
    source_merge = json.loads(source_merge_path.read_text(encoding="utf-8")) if source_merge_path.exists() else {}
    base_eval = bridge_summary.get("vllm_eval") or {}
    avg_primary = maybe_float(model_row.get("avg_primary_score"))
    worst_primary = maybe_float(model_row.get("worst_primary_score"))
    base_bridge_avg = maybe_float(base_eval.get("avg_primary_score"))
    uniform_avg = maybe_float(source_merge.get("merge_avg_primary_score"))
    best_source_avg = maybe_float(source_merge.get("best_source_avg_primary_score"))
    base_tasks = {row.get("task"): row for row in base_eval.get("task_metrics", [])}
    uniform_tasks = source_merge.get("merge_by_task", {})
    task_metrics = []
    for row in eval_summary.get("metrics", []):
        metric_name, value = primary_metric(row)
        task = row.get("task")
        base_task = base_tasks.get(task, {})
        uniform_task = uniform_tasks.get(task, {})
        task_metrics.append(
            {
                "task": task,
                "primary_metric": metric_name,
                "primary_score": value,
                "delta_vs_base_bridge": None
                if value is None or base_task.get("primary_score") is None
                else value - float(base_task["primary_score"]),
                "delta_vs_uniform": None
                if value is None or uniform_task.get("primary_score") is None
                else value - float(uniform_task["primary_score"]),
                "safe_non_refusal_rate": maybe_float(row.get("safe_non_refusal_rate")),
                "unsafe_refusal_rate": maybe_float(row.get("unsafe_refusal_rate")),
            }
        )
    return {
        "status": eval_summary.get("status"),
        "eval_dir": rel(eval_root),
        "report": rel(eval_root / "report.md"),
        "metrics": rel(eval_root / "metrics.csv"),
        "model_summary": rel(eval_root / "model_summary.csv"),
        "avg_primary_score": avg_primary,
        "worst_primary_score": worst_primary,
        "base_bridge_avg_primary_score": base_bridge_avg,
        "uniform_avg_primary_score": uniform_avg,
        "best_source_avg_primary_score": best_source_avg,
        "delta_vs_base_bridge_avg_primary": None
        if avg_primary is None or base_bridge_avg is None
        else avg_primary - base_bridge_avg,
        "delta_vs_uniform_avg_primary": None if avg_primary is None or uniform_avg is None else avg_primary - uniform_avg,
        "delta_vs_best_source_avg_primary": None
        if avg_primary is None or best_source_avg is None
        else avg_primary - best_source_avg,
        "task_metrics": task_metrics,
    }


def select_tensors(
    conflict: pd.DataFrame,
    *,
    groups: set[str],
    min_sign_conflict: float,
    max_delta_cosine: float,
    min_numel: int,
    max_rules: int | None,
) -> pd.DataFrame:
    selected = conflict[
        conflict["group"].astype(str).isin(groups)
        & (conflict["sign_conflict_rate"].astype(float) >= min_sign_conflict)
        & (conflict["delta_cosine"].astype(float) <= max_delta_cosine)
        & (conflict["numel"].astype(int) >= min_numel)
    ].copy()
    selected = selected.sort_values(["sign_conflict_rate", "numel"], ascending=[False, False])
    if max_rules is not None:
        selected = selected.head(max_rules).copy()
    return selected


def write_method_rules(path: Path, selected: pd.DataFrame, method: str, density: float, drop_rate: float, seed: int) -> None:
    lines = [
        "# PATTERN::METHOD[,KEY=VALUE]",
        "# First matching method rule wins in write_same_shape_average_checkpoint.py.",
        "# These exact-tensor rules come from Qwen instruct/coder sign-conflict probes.",
    ]
    for _, row in selected.iterrows():
        tensor = str(row["tensor"])
        options = f"{method},density={density}"
        if method in {"dare", "ties_dare"}:
            options += f",drop_rate={drop_rate},seed={seed}"
        lines.append(
            "# group={group}; projection={projection}; sign_conflict={sign}; cosine={cos}; numel={numel}".format(
                group=row["group"],
                projection=row["projection"],
                sign=fmt(row["sign_conflict_rate"]),
                cos=fmt(row["delta_cosine"]),
                numel=int(row["numel"]),
            )
        )
        lines.append(f"^{re.escape(tensor)}$::{options}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    selected: pd.DataFrame,
    projection_counts: dict[str, int],
) -> None:
    lines = [
        "# Qwen Dense Sparse-Method Candidate",
        "",
        "## 结论",
        "",
        (
            f"`{summary['candidate_id']}` 把已有 global bridge 的 source weights "
            f"`instruct={summary['source_weights']['instruct']}`、`coder={summary['source_weights']['coder']}` 保留不变，"
            f"但对 high-conflict {','.join(summary['groups'])} tensor 加上 `{summary['method']}` coordinate rule。"
        ),
        "",
        (
            "这不是再做模块级 freeze。它对应 TIES/DARE/DELLA/Breadcrumbs 这类 sparse task-vector 机制："
            "先用 probe 找到符号冲突集中的坐标，再在同构 writer 中对这些坐标做 trim/sign-elect/merge。"
        ),
        "",
        "## Selection",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| selected tensors | {summary['selected_tensor_count']} |",
        f"| selected parameters | {summary['selected_numel']} |",
        f"| selected parameter fraction | {fmt(summary['selected_numel_fraction'], 4)} |",
        f"| min sign conflict | {fmt(summary['min_sign_conflict'])} |",
        f"| max cosine | {fmt(summary['max_delta_cosine'])} |",
        f"| density | {fmt(summary['density'])} |",
        f"| dry-run status | {summary['dry_run_status']} |",
        f"| dry-run sparse method tensors | {summary['dry_run_tensor_method_applied_count']} |",
        "",
        "Projection counts: `" + json.dumps(projection_counts, sort_keys=True) + "`",
        "",
    ]
    eval_summary = summary.get("vllm_eval")
    if eval_summary:
        lines.extend(
            [
                "## vLLM Eval Result",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| status | {eval_summary['status']} |",
                f"| avg primary | {fmt(eval_summary['avg_primary_score'])} |",
                f"| worst primary | {fmt(eval_summary['worst_primary_score'])} |",
                f"| delta vs global bridge avg | {fmt(eval_summary['delta_vs_base_bridge_avg_primary'])} |",
                f"| delta vs uniform avg | {fmt(eval_summary['delta_vs_uniform_avg_primary'])} |",
                f"| delta vs best source avg | {fmt(eval_summary['delta_vs_best_source_avg_primary'])} |",
                "",
                "| task | primary metric | score | delta vs global bridge | delta vs uniform |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in eval_summary["task_metrics"]:
            lines.append(
                f"| {row['task']} | {row['primary_metric']} | {fmt(row['primary_score'])} | "
                f"{fmt(row['delta_vs_base_bridge'])} | {fmt(row['delta_vs_uniform'])} |"
            )
        lines.extend(["", f"Full vLLM report: `{eval_summary['report']}`", ""])
    lines.extend(
        [
            "## Selected Tensor Preview",
            "",
            "| tensor | projection | numel | cosine | sign conflict |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in selected.head(20).iterrows():
        lines.append(
            f"| `{row['tensor']}` | {row['projection']} | {int(row['numel'])} | "
            f"{fmt(row['delta_cosine'])} | {fmt(row['sign_conflict_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Writer Commands",
            "",
            "Dry-run validation:",
            "",
            "```bash",
            summary["dry_run_command"],
            "```",
            "",
            "Materialize candidate:",
            "",
            "```bash",
            summary["writer_command"],
            "```",
            "",
            "## vLLM Eval",
            "",
            "```bash",
            summary["vllm_commands"]["serve"],
            "```",
            "",
            "```bash",
            summary["vllm_commands"]["eval"],
            "```",
            "",
            "## Files",
            "",
            f"- `{summary['artifacts']['selected_tensors']}`",
            f"- `{summary['artifacts']['tensor_method_rules']}`",
            f"- `{summary['artifacts']['dry_run_manifest']}`",
            f"- `{summary['artifacts']['summary']}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen dense sparse tensor-method candidate from conflict probes.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_dense_sparse_method_candidate"))
    parser.add_argument("--conflict-csv", type=Path, default=Path("results/qwen_dense_module_guarded_candidate/tensor_conflict.csv"))
    parser.add_argument("--bridge-summary", type=Path, default=Path("results/probe_guided_dense_average_candidate/summary.json"))
    parser.add_argument("--candidate-id", default="qwen_0_5b_sparse_method_bridge")
    parser.add_argument("--method", choices=["ties", "dare", "ties_dare"], default="ties")
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--drop-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-sign-conflict", type=float, default=0.44)
    parser.add_argument("--max-delta-cosine", type=float, default=0.16)
    parser.add_argument("--min-numel", type=int, default=100_000)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument("--groups", default="attention,mlp")
    parser.add_argument("--skip-dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conflict = pd.read_csv(repo_path(args.conflict_csv))
    bridge_summary = json.loads(repo_path(args.bridge_summary).read_text(encoding="utf-8"))
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    selected = select_tensors(
        conflict,
        groups=groups,
        min_sign_conflict=args.min_sign_conflict,
        max_delta_cosine=args.max_delta_cosine,
        min_numel=args.min_numel,
        max_rules=args.max_rules,
    )
    if selected.empty:
        raise ValueError("No tensors selected for sparse method rules; relax thresholds or groups.")

    method_rule_file = output_dir / "tensor_method_rules.txt"
    write_method_rules(method_rule_file, selected, args.method, args.density, args.drop_rate, args.seed)
    selected_path = output_dir / "selected_tensors.csv"
    selected.to_csv(selected_path, index=False)

    checkpoint_dir = f"results/checkpoints/{args.candidate_id}"
    dry_run_dir = str(output_dir / "dry_run")
    eval_dir = f"results/vllm_checkpoint_eval/{args.candidate_id}"
    served_model = f"candidate_{args.candidate_id}"
    writer_command = build_writer_command(
        bridge_summary["writer_command"],
        method_rule_file=method_rule_file,
        output_dir=checkpoint_dir,
        dry_run=False,
    )
    dry_run_command = build_writer_command(
        bridge_summary["writer_command"],
        method_rule_file=method_rule_file,
        output_dir=dry_run_dir,
        dry_run=True,
    )
    dry_run_status = "skipped"
    dry_run_method_counts: dict[str, int] = {}
    dry_run_tensor_method_applied_count = 0
    dry_run_manifest = output_dir / "dry_run" / "merge_manifest.json"
    if not args.skip_dry_run:
        completed = subprocess.run(shlex.split(dry_run_command), cwd=REPO_ROOT, check=True, text=True, capture_output=True)
        dry_run_status = "passed" if dry_run_manifest.exists() else "missing_manifest"
        (output_dir / "dry_run_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "dry_run_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if dry_run_manifest.exists():
            manifest = json.loads(dry_run_manifest.read_text(encoding="utf-8"))
            dry_run_method_counts = {str(key): int(value) for key, value in manifest.get("method_counts", {}).items()}
            dry_run_tensor_method_applied_count = sum(
                count for method, count in dry_run_method_counts.items() if method.startswith("tensor_method:")
            )
            if dry_run_tensor_method_applied_count != len(selected):
                raise RuntimeError(
                    "Sparse method dry-run rule mismatch: selected "
                    f"{len(selected)} tensors but writer applied {dry_run_tensor_method_applied_count}. "
                    f"method_counts={dry_run_method_counts}"
                )

    projection_counts = dict(Counter(selected["projection"].astype(str)))
    total_numel = int(conflict["numel"].sum())
    selected_numel = int(selected["numel"].sum())
    source_weights = {
        "instruct": bridge_summary.get("alpha"),
        "coder": bridge_summary.get("beta"),
    }
    vllm_commands = build_vllm_commands(checkpoint_dir, eval_dir, served_model)
    vllm_eval = load_vllm_eval(eval_dir, bridge_summary)
    status = "dry_run_passed_waiting_for_materialization_eval" if dry_run_status == "passed" else "candidate_selected_waiting_for_dry_run"
    if vllm_eval and vllm_eval.get("status") == "complete":
        status = "evaluated_complete"
    summary = {
        "schema_version": 1,
        "status": status,
        "candidate_id": args.candidate_id,
        "method": args.method,
        "density": args.density,
        "drop_rate": args.drop_rate if args.method in {"dare", "ties_dare"} else 0.0,
        "seed": args.seed,
        "groups": sorted(groups),
        "min_sign_conflict": args.min_sign_conflict,
        "max_delta_cosine": args.max_delta_cosine,
        "min_numel": args.min_numel,
        "selected_tensor_count": int(len(selected)),
        "selected_numel": selected_numel,
        "selected_numel_fraction": selected_numel / max(1, total_numel),
        "projection_counts": projection_counts,
        "source_weights": source_weights,
        "base_bridge_candidate": bridge_summary.get("candidate_id"),
        "checkpoint_output_dir": checkpoint_dir,
        "dry_run_output_dir": dry_run_dir,
        "dry_run_status": dry_run_status,
        "dry_run_method_counts": dry_run_method_counts,
        "dry_run_tensor_method_applied_count": dry_run_tensor_method_applied_count,
        "writer_command": writer_command,
        "dry_run_command": dry_run_command,
        "vllm_commands": vllm_commands,
        "vllm_eval": vllm_eval,
        "hypothesis": "High sign-conflict tensors should use coordinate-wise sparse merge instead of whole-tensor freeze or scalar damping.",
        "artifacts": {
            "report": rel(output_dir / "report.md"),
            "selected_tensors": rel(selected_path),
            "tensor_method_rules": rel(method_rule_file),
            "dry_run_manifest": rel(dry_run_manifest),
            "dry_run_stdout": rel(output_dir / "dry_run_stdout.txt"),
            "dry_run_stderr": rel(output_dir / "dry_run_stderr.txt"),
            "summary": rel(output_dir / "summary.json"),
        },
    }
    (output_dir / "writer_command.txt").write_text(writer_command + "\n", encoding="utf-8")
    (output_dir / "dry_run_command.txt").write_text(dry_run_command + "\n", encoding="utf-8")
    (output_dir / "vllm_commands.json").write_text(json.dumps(vllm_commands, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir, summary, selected, projection_counts)
    print(f"Wrote {rel(output_dir / 'summary.json')}")
    print(f"Selected {len(selected)} tensors for {args.method} sparse method rules")


if __name__ == "__main__":
    main()
