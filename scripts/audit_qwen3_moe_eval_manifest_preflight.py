#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
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


def canonical_json(data: Any) -> str:
    return json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n"


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def shell_join(tokens: list[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def task_requirements(task_budget: pd.DataFrame) -> dict[str, int]:
    requirements = {}
    for _, row in task_budget.iterrows():
        task = str(row.get("task", "")).strip()
        if not task:
            continue
        achievable = clean_value(row.get("achievable_examples"))
        requirements[task] = int(achievable) if achievable is not None else 0
    return requirements


def prepare_manifest_command(
    *,
    manifest_path: str,
    tasks: list[str],
    max_examples: int,
    example_source: str,
) -> str:
    return shell_join(
        [
            "python",
            "scripts/run_vllm_downstream_eval.py",
            "--tasks",
            ",".join(tasks),
            "--example-source",
            example_source,
            "--max-examples",
            str(max_examples),
            "--task-manifest",
            manifest_path,
            "--create-task-manifest-if-missing",
            "--prepare-task-manifest-only",
        ]
    )


def manifest_counts(payload: dict[str, Any]) -> dict[str, int]:
    examples = payload.get("examples") or {}
    return {str(task): len(rows) for task, rows in examples.items() if isinstance(rows, list)}


def build_rows(
    *,
    manifest_exists: bool,
    manifest_payload: dict[str, Any],
    requirements: dict[str, int],
) -> pd.DataFrame:
    counts = manifest_counts(manifest_payload) if manifest_exists else {}
    rows = []
    for task, required in requirements.items():
        observed = counts.get(task, 0)
        rows.append(
            {
                "task": task,
                "required_examples": required,
                "manifest_examples": observed,
                "task_present": task in counts,
                "count_sufficient": observed >= required,
                "deficit": max(0, required - observed),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    *,
    output_dir: Path,
    budget_summary: dict[str, Any],
    task_rows: pd.DataFrame,
    alignment: pd.DataFrame,
    manifest_path: str,
    manifest_payload: dict[str, Any],
    prepare_command: str,
) -> dict[str, Any]:
    absolute_manifest_path = repo_path(manifest_path)
    manifest_exists = absolute_manifest_path.exists()
    aligned_count = int((alignment["task_manifest_aligned"].astype(bool)).sum()) if not alignment.empty else 0
    method_count = int(budget_summary.get("method_count") or len(alignment))
    all_aligned = aligned_count == method_count and method_count > 0
    all_tasks_sufficient = bool(not task_rows.empty and task_rows["count_sufficient"].astype(bool).all())
    if not all_aligned:
        status = "task_manifest_path_alignment_failed"
    elif not manifest_exists:
        status = "task_manifest_missing"
    elif not all_tasks_sufficient:
        status = "task_manifest_incomplete"
    else:
        status = "task_manifest_ready"
    return {
        "schema_version": 1,
        "status": status,
        "canonical_task_manifest": manifest_path,
        "manifest_exists": manifest_exists,
        "manifest_sha256": sha256_json(manifest_payload) if manifest_exists else None,
        "method_count": method_count,
        "task_manifest_aligned_method_count": aligned_count,
        "task_manifest_unaligned_method_count": max(0, method_count - aligned_count),
        "task_count": int(len(task_rows)),
        "task_sufficient_count": int(task_rows["count_sufficient"].astype(bool).sum()) if not task_rows.empty else 0,
        "total_required_examples": int(task_rows["required_examples"].sum()) if not task_rows.empty else 0,
        "total_manifest_examples": int(task_rows["manifest_examples"].sum()) if not task_rows.empty else 0,
        "prepare_manifest_command": prepare_command,
        "blocking_reason": (
            "Run the prepare_manifest_command before launching vLLM evals."
            if not manifest_exists
            else "Regenerate the task manifest with enough examples for every budgeted task."
            if not all_tasks_sufficient
            else ""
        ),
        "outputs": {
            "task_manifest_checks": rel(output_dir / "task_manifest_checks.csv"),
            "prepare_manifest_command": rel(output_dir / "prepare_manifest_command.txt"),
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
        },
    }


def build_report(summary: dict[str, Any], task_rows: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Eval Task Manifest Preflight",
        "",
        "这个 preflight 在远端 vLLM eval 前检查 final selector 需要的 canonical task manifest。它验证所有 planned rows 是否共享同一路径，并检查 manifest 里的每个任务样本数是否达到预算里的 achievable example count。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Canonical task manifest: `{summary['canonical_task_manifest']}`",
        f"- Manifest exists: `{summary['manifest_exists']}`",
        f"- Methods aligned: `{summary['task_manifest_aligned_method_count']}/{summary['method_count']}`",
        f"- Tasks sufficient: `{summary['task_sufficient_count']}/{summary['task_count']}`",
        f"- Total manifest / required examples: `{summary['total_manifest_examples']}/{summary['total_required_examples']}`",
        "",
        "## Prepare Command",
        "",
        "```bash",
        summary["prepare_manifest_command"],
        "```",
        "",
        "## Task Checks",
        "",
        "| task | required | manifest | present | sufficient | deficit |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for _, row in task_rows.iterrows():
        lines.append(
            f"| `{row['task']}` | {int(row['required_examples'])} | {int(row['manifest_examples'])} | "
            f"`{bool(row['task_present'])}` | `{bool(row['count_sufficient'])}` | {int(row['deficit'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['task_manifest_checks']}`",
            f"- `{summary['outputs']['prepare_manifest_command']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    budget_dir = repo_path(args.eval_budget_dir)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_summary = read_json(budget_dir / "summary.json")
    task_budget = read_csv(budget_dir / "task_budget.csv")
    alignment = read_csv(budget_dir / "task_manifest_alignment.csv")
    manifest_path = str(budget_summary.get("canonical_task_manifest") or args.task_manifest)
    requirements = task_requirements(task_budget)
    tasks = list(requirements)
    max_examples = int(budget_summary.get("recommended_max_examples") or max(requirements.values()))
    manifest_file = repo_path(manifest_path)
    manifest_payload = read_json(manifest_file)
    task_rows = build_rows(
        manifest_exists=manifest_file.exists(),
        manifest_payload=manifest_payload,
        requirements=requirements,
    )
    prepare_command = prepare_manifest_command(
        manifest_path=manifest_path,
        tasks=tasks,
        max_examples=max_examples,
        example_source=args.example_source,
    )
    summary = build_summary(
        output_dir=output_dir,
        budget_summary=budget_summary,
        task_rows=task_rows,
        alignment=alignment,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
        prepare_command=prepare_command,
    )
    task_rows.to_csv(output_dir / "task_manifest_checks.csv", index=False)
    (output_dir / "prepare_manifest_command.txt").write_text(prepare_command + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, task_rows), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Qwen3 MoE eval task manifest readiness.")
    parser.add_argument("--eval-budget-dir", type=Path, default=Path("results/qwen3_moe_eval_budget_plan"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_eval_manifest_preflight"))
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("results/qwen3_moe_mechanism_eval_gate/task_manifest.json"),
    )
    parser.add_argument("--example-source", default="datasets")
    args = parser.parse_args()
    summary = build(args)
    print(f"Wrote Qwen3 MoE eval manifest preflight to {repo_path(args.output_dir).resolve()}")
    print(
        f"Status: {summary['status']}; tasks={summary['task_sufficient_count']}/"
        f"{summary['task_count']}; methods={summary['task_manifest_aligned_method_count']}/"
        f"{summary['method_count']}"
    )


if __name__ == "__main__":
    main()
