#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ALIAS_TO_MODEL = {
    "base": "Qwen/Qwen3-30B-A3B-Base",
    "instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "coder": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def maybe_int(value: Any) -> int | None:
    value = clean_value(value)
    return None if value is None else int(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def split_semicolon(text: Any) -> list[str]:
    text = "" if clean_value(text) is None else str(text)
    return [part.strip() for part in text.split(";") if part.strip()]


def shell_join(tokens: list[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens if str(token) != "")


def normalize_task(task: str) -> str:
    return "humaneval_compile" if task == "humaneval" else task


def tasks_for_row(row: pd.Series) -> list[str]:
    scenario_id = str(row.get("scenario_id", ""))
    architecture = str(row.get("architecture_kind", ""))
    measured = str(row.get("source_set_type", "")) == "measured_generation_frontier"
    tasks = ["mmlu", "gsm8k", "humaneval_compile"]
    if not measured and (architecture == "dense" or "general_code_route_aware" in scenario_id):
        tasks.append("safety")
    return [normalize_task(task) for task in tasks]


def source_alias_models(source_set: Any) -> list[str]:
    aliases = [part.strip() for part in str(source_set).split("+") if part.strip()]
    return [SOURCE_ALIAS_TO_MODEL.get(alias, alias) for alias in aliases]


def registry_source_models(required_models: Any) -> list[str]:
    models = []
    for model in split_semicolon(required_models):
        lower = model.lower()
        if lower.startswith("optional "):
            continue
        if "verified " in lower or " plus " in lower:
            continue
        models.append(model)
    return models


def served_models_for_row(row: pd.Series) -> list[str]:
    if str(row.get("source_set_type", "")) == "measured_generation_frontier":
        return source_alias_models(row.get("source_set"))
    return registry_source_models(row.get("required_models"))


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    out = []
    for char in text:
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    while "__" in "".join(out):
        text = "".join(out).replace("__", "_")
        out = list(text)
    return "".join(out).strip("_") or "job"


def task_manifest_path(output_dir: Path, job_id: str, *, smoke: bool = False) -> str:
    name = f"{job_id}_{'smoke_' if smoke else ''}task_manifest.json"
    return rel(output_dir / "task_manifests" / name)


def prepare_manifest_command(
    *,
    manifest_path: str,
    tasks: list[str],
    example_source: str,
    max_examples: int,
    subjects: str,
) -> str:
    tokens = [
        "python",
        "scripts/run_vllm_downstream_eval.py",
        "--tasks",
        ",".join(tasks),
        "--example-source",
        example_source,
        "--max-examples",
        str(max_examples),
        "--subjects",
        subjects,
        "--task-manifest",
        manifest_path,
        "--create-task-manifest-if-missing",
        "--prepare-task-manifest-only",
    ]
    return shell_join(tokens)


def eval_command(
    *,
    base_url: str,
    models: list[str],
    tasks: list[str],
    example_source: str,
    max_examples: int,
    subjects: str,
    manifest_path: str,
    output_dir: str,
    allow_unavailable: bool = False,
) -> str:
    tokens = [
        "python",
        "scripts/run_vllm_downstream_eval.py",
        "--base-url",
        base_url,
        "--models",
        ",".join(models),
        "--tasks",
        ",".join(tasks),
        "--example-source",
        example_source,
        "--max-examples",
        str(max_examples),
        "--subjects",
        subjects,
        "--task-manifest",
        manifest_path,
        "--create-task-manifest-if-missing",
        "--output-dir",
        output_dir,
    ]
    if allow_unavailable:
        tokens.append("--allow-unavailable")
    return shell_join(tokens)


def build_eval_jobs(args: argparse.Namespace, candidates: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    queue_by_scenario = {
        str(row["scenario_id"]): row.to_dict() for _, row in queue.iterrows() if "scenario_id" in row
    }
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        readiness = str(candidate.get("readiness_gate", ""))
        if readiness == "manual_model_resolution_first":
            continue
        models = served_models_for_row(candidate)
        if len(models) < 2:
            continue
        scenario_id = str(candidate["scenario_id"])
        job_id = slug(scenario_id)
        if str(candidate.get("source_set_type", "")) == "measured_generation_frontier":
            job_id = "measured_coder_thinking_source_frontier"
        tasks = tasks_for_row(candidate)
        queue_row = queue_by_scenario.get(scenario_id, {})
        manifest = task_manifest_path(args.output_dir, job_id, smoke=False)
        smoke_manifest = task_manifest_path(args.output_dir, job_id, smoke=True)
        eval_dir = rel(args.output_dir / "vllm_eval" / job_id)
        smoke_eval_dir = rel(args.output_dir / "smoke_endpoint_probe" / job_id)
        production_prepare = prepare_manifest_command(
            manifest_path=manifest,
            tasks=tasks,
            example_source=args.example_source,
            max_examples=args.max_examples,
            subjects=args.subjects,
        )
        smoke_prepare = prepare_manifest_command(
            manifest_path=smoke_manifest,
            tasks=tasks,
            example_source=args.smoke_example_source,
            max_examples=args.smoke_max_examples,
            subjects=args.subjects,
        )
        rows.append(
            {
                "job_id": job_id,
                "scenario_id": scenario_id,
                "candidate_id": candidate.get("candidate_id"),
                "architecture_kind": candidate.get("architecture_kind"),
                "source_set_type": candidate.get("source_set_type"),
                "served_models": ",".join(models),
                "model_count": len(models),
                "tasks": ",".join(tasks),
                "task_count": len(tasks),
                "source_discovery_rank": maybe_int(queue_row.get("rank")) or 999,
                "source_discovery_priority_score": maybe_float(queue_row.get("priority_score")),
                "max_examples": args.max_examples,
                "task_manifest": manifest,
                "smoke_task_manifest": smoke_manifest,
                "output_dir": eval_dir,
                "readiness_gate": readiness,
                "optimizer_gate": candidate.get("optimizer_gate"),
                "frontier_avg_gain_vs_best_single": maybe_float(
                    candidate.get("frontier_avg_gain_vs_best_single")
                ),
                "interference_budget": maybe_float(candidate.get("interference_budget")),
                "additional_frontier_avg_gain_needed": maybe_float(
                    candidate.get("additional_frontier_avg_gain_needed")
                ),
                "status": "ready_for_vllm_endpoint_eval"
                if str(candidate.get("source_set_type", "")) != "measured_generation_frontier"
                else "ready_for_probe_only_endpoint_expansion",
                "prepare_manifest_command": production_prepare,
                "smoke_manifest_command": smoke_prepare,
                "eval_command": eval_command(
                    base_url=args.base_url,
                    models=models,
                    tasks=tasks,
                    example_source=args.example_source,
                    max_examples=args.max_examples,
                    subjects=args.subjects,
                    manifest_path=manifest,
                    output_dir=eval_dir,
                ),
                "smoke_endpoint_probe_command": eval_command(
                    base_url=args.base_url,
                    models=models,
                    tasks=tasks,
                    example_source=args.smoke_example_source,
                    max_examples=args.smoke_max_examples,
                    subjects=args.subjects,
                    manifest_path=smoke_manifest,
                    output_dir=smoke_eval_dir,
                    allow_unavailable=True,
                ),
                "expected_decision_update": (
                    "refresh qwen3_average_source_set_optimizer and promote only if surplus is non-negative"
                    if str(candidate.get("source_set_type", "")) == "measured_generation_frontier"
                    else "compute source endpoint frontier before any same-shape average materialization"
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["source_discovery_rank", "job_id"], ascending=[True, True]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out[
        [
            "rank",
            "job_id",
            "scenario_id",
            "candidate_id",
            "architecture_kind",
            "source_set_type",
            "served_models",
            "model_count",
            "tasks",
            "task_count",
            "source_discovery_rank",
            "source_discovery_priority_score",
            "max_examples",
            "task_manifest",
            "smoke_task_manifest",
            "output_dir",
            "readiness_gate",
            "optimizer_gate",
            "frontier_avg_gain_vs_best_single",
            "interference_budget",
            "additional_frontier_avg_gain_needed",
            "status",
            "prepare_manifest_command",
            "smoke_manifest_command",
            "eval_command",
            "smoke_endpoint_probe_command",
            "expected_decision_update",
        ]
    ]


def build_manifest_jobs(eval_jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, job in eval_jobs.iterrows():
        rows.append(
            {
                "job_id": job["job_id"],
                "scenario_id": job["scenario_id"],
                "manifest_kind": "production",
                "task_manifest": job["task_manifest"],
                "tasks": job["tasks"],
                "max_examples": job["max_examples"],
                "example_source": "datasets",
                "command": job["prepare_manifest_command"],
            }
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "scenario_id": job["scenario_id"],
                "manifest_kind": "smoke",
                "task_manifest": job["smoke_task_manifest"],
                "tasks": job["tasks"],
                "max_examples": 2,
                "example_source": "builtin",
                "command": job["smoke_manifest_command"],
            }
        )
    return pd.DataFrame(rows)


def build_run_script(eval_jobs: pd.DataFrame) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Prepare production task manifests first. These commands use the dataset-backed",
        "# runner and should be executed in an environment with the required datasets/cache.",
    ]
    for _, job in eval_jobs.iterrows():
        lines.extend(["", f"# {job['job_id']}: {job['scenario_id']}", job["prepare_manifest_command"]])
    lines.extend(["", "# Then start the relevant vLLM servers and run the endpoint eval commands."])
    for _, job in eval_jobs.iterrows():
        lines.extend(["", f"# {job['job_id']}: {job['served_models']}", job["eval_command"]])
    return "\n".join(lines) + "\n"


def build_report(summary: dict[str, Any], eval_jobs: pd.DataFrame, manifest_jobs: pd.DataFrame) -> str:
    top_job = summary.get("top_eval_job") or {}
    lines = [
        "# Qwen Source Discovery vLLM Eval Plan",
        "",
        "## Result",
        "",
        (
            f"这个计划把 source discovery 的候选源集合落成同一 task manifest 的 vLLM jobs。"
            f"当前共有 `{summary['eval_job_count']}` 个 endpoint/frontier eval jobs；最高优先 job 是 "
            f"`{top_job.get('job_id')}`，场景 `{top_job.get('scenario_id')}`，任务 `{top_job.get('tasks')}`。"
        ),
        "",
        (
            "关键修正：runner 的 HumanEval 任务名是 `humaneval_compile`，这里所有 vLLM 命令都使用该任务名，"
            "避免把 HumanEval 计划写成 runner 不会执行的 `humaneval`。"
        ),
        "",
        "## vLLM Jobs",
        "",
        "| rank | job | scenario | models | tasks | status | extra gain needed |",
        "| ---: | --- | --- | ---: | --- | --- | ---: |",
    ]
    for _, row in eval_jobs.iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['job_id']}` | `{row['scenario_id']}` | "
            f"{int(row['model_count'])} | `{row['tasks']}` | `{row['status']}` | "
            f"{fmt(row['additional_frontier_avg_gain_needed'])} |"
        )
    lines.extend(
        [
            "",
            "## Manifest Jobs",
            "",
            "| job | kind | manifest | tasks | source | command |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in manifest_jobs.iterrows():
        lines.append(
            f"| `{row['job_id']}` | `{row['manifest_kind']}` | `{row['task_manifest']}` | "
            f"`{row['tasks']}` | `{row['example_source']}` | `{row['command']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['vllm_eval_jobs']}`",
            f"- `{summary['outputs']['manifest_jobs']}`",
            f"- `{summary['outputs']['run_script']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = repo_path(args.source_discovery_dir)
    source_summary = read_json(source_dir / "summary.json")
    candidates = read_csv(source_dir / "candidate_source_sets.csv")
    queue = read_csv(source_dir / "source_discovery_queue.csv")
    eval_jobs = build_eval_jobs(args, candidates, queue)
    manifest_jobs = build_manifest_jobs(eval_jobs)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_manifests").mkdir(parents=True, exist_ok=True)

    top_job = eval_jobs.iloc[0].to_dict() if not eval_jobs.empty else {}
    task_names = sorted({task for raw in eval_jobs.get("tasks", pd.Series(dtype=str)) for task in str(raw).split(",") if task})
    task_compatibility_status = (
        "passed_humaneval_compile_task_name"
        if "humaneval_compile" in task_names and "humaneval" not in task_names
        else "check_task_names"
    )
    summary = {
        "schema_version": 1,
        "status": "source_discovery_vllm_eval_plan_ready",
        "source_discovery_status": source_summary.get("status"),
        "eval_job_count": int(len(eval_jobs)),
        "manifest_job_count": int(len(manifest_jobs)),
        "task_names": task_names,
        "task_name_compatibility_status": task_compatibility_status,
        "top_eval_job": {
            "job_id": top_job.get("job_id"),
            "scenario_id": top_job.get("scenario_id"),
            "status": top_job.get("status"),
            "tasks": top_job.get("tasks"),
            "served_models": top_job.get("served_models"),
            "additional_frontier_avg_gain_needed": maybe_float(
                top_job.get("additional_frontier_avg_gain_needed")
            ),
            "eval_command": top_job.get("eval_command"),
        },
        "measured_additional_frontier_avg_gain_needed": source_summary.get(
            "measured_additional_frontier_avg_gain_needed"
        ),
        "outputs": {
            "vllm_eval_jobs": rel(output_dir / "vllm_eval_jobs.csv"),
            "manifest_jobs": rel(output_dir / "manifest_jobs.csv"),
            "run_script": rel(output_dir / "run_source_frontier_eval.sh"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    eval_jobs.to_csv(output_dir / "vllm_eval_jobs.csv", index=False)
    manifest_jobs.to_csv(output_dir / "manifest_jobs.csv", index=False)
    (output_dir / "run_source_frontier_eval.sh").write_text(build_run_script(eval_jobs), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, eval_jobs, manifest_jobs),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build executable vLLM eval jobs for Qwen source discovery.")
    parser.add_argument("--source-discovery-dir", type=Path, default=Path("results/qwen_source_discovery_plan"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_source_discovery_eval_plan"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--example-source", choices=["datasets", "builtin"], default="datasets")
    parser.add_argument("--smoke-example-source", choices=["datasets", "builtin"], default="builtin")
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--smoke-max-examples", type=int, default=2)
    parser.add_argument("--subjects", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    top = summary.get("top_eval_job") or {}
    print(f"Wrote Qwen source discovery eval plan to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; jobs={summary['eval_job_count']}; "
        f"top={top.get('job_id')}; task_names={','.join(summary['task_names'])}"
    )


if __name__ == "__main__":
    main()
