#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

ALIAS_SOURCES = [
    {
        "phase": "moe_30b_a3b",
        "priority": "p0",
        "role": "moe_thinking_reasoning_anchor",
        "model_id": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "model_family": "Qwen3",
        "architecture": "qwen3_moe",
        "scale": "30B-A3B",
        "source_type": "official_posttrained_moe",
        "merge_role": "expert",
        "scenario": "measured_qwen3_moe_source_set",
        "topology_action": "inspect_moe_config_router_expert_tensors",
        "eval_focus": "MMLU, GSM8K, long reasoning, thinking-mode retention",
        "probe_focus": "router entropy, thinking-route overlap, expert-load specialization",
        "materialization_status": "ready_for_endpoint_eval",
        "source_url": "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507",
        "evidence_note": "Measured source alias used by the current Coder+Thinking frontier; verify served id before vLLM launch.",
    },
]

CAPABILITY_KEYWORDS = {
    "math_reasoning": ["math", "gsm8k", "aime", "gpqa", "reasoning", "cot", "thinking"],
    "code_generation_agentic": ["code", "coder", "humaneval", "mbpp", "livecodebench", "tool", "agent"],
    "general_knowledge_instruction": ["general", "instruction", "mmlu", "c-eval", "cmmlu", "ifeval", "chat"],
    "safety_refusal": ["safety", "refusal", "advbench", "beavertails"],
}

RETENTION_TASKS = {
    "gsm8k": ["gsm8k", "mmlu"],
    "humaneval_compile": ["humaneval_compile", "mmlu"],
    "mmlu": ["mmlu", "gsm8k", "humaneval_compile"],
    "safety": ["safety", "mmlu"],
}

LITERATURE_SOURCES = [
    {
        "key": "model_soups",
        "title": "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Weight averaging is most plausible when fine-tuned models occupy a shared low-loss basin and source endpoints have complementary validation signal.",
    },
    {
        "key": "ties",
        "title": "TIES-Merging: Resolving Interference When Merging Models",
        "url": "https://arxiv.org/abs/2306.01708",
        "mechanism": "Sign and redundancy interference mean weak source complementarity should block final average promotion before delta surgery.",
    },
    {
        "key": "expert_merging",
        "title": "Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking",
        "url": "https://arxiv.org/abs/2509.25712",
        "mechanism": "For MoE, source and task evidence should decide which same-shape expert families deserve layer/chunk coefficient calibration.",
    },
    {
        "key": "harc",
        "title": "When Model Merging Breaks Routing: Training-Free Calibration for MoE",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Router calibration is useful only after source/candidate eval shows the average is worth repairing.",
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


def maybe_float(value: Any, default: float = 0.0) -> float:
    value = clean_value(value)
    return default if value is None else float(value)


def fmt(value: Any) -> str:
    return f"{maybe_float(value):.4f}"


def shell_join(tokens: list[Any]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens if str(token) != "")


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    slugged = "".join(out)
    while "__" in slugged:
        slugged = slugged.replace("__", "_")
    return slugged.strip("_") or "item"


def same_shape_group(row: pd.Series) -> str:
    architecture = str(row.get("architecture", "")).lower()
    scale = str(row.get("scale", "")).lower()
    family = str(row.get("model_family", "")).lower()
    if "qwen3" in family and "moe" in architecture:
        return "qwen3_moe_30b_a3b"
    if "qwen2" in architecture and "7b" in scale:
        return "qwen2_dense_7b"
    if "qwen2" in architecture and "32b" in scale:
        return "qwen2_dense_32b"
    return slug(f"{row.get('architecture')}_{row.get('scale')}")


def capability_match(row: pd.Series, capability: str) -> tuple[float, str]:
    text = " ".join(
        str(row.get(column, ""))
        for column in ["role", "model_id", "model_family", "source_type", "merge_role", "eval_focus", "probe_focus", "evidence_note"]
    ).lower()
    keywords = CAPABILITY_KEYWORDS.get(capability, [])
    hits = [keyword for keyword in keywords if keyword in text]
    if not hits:
        return 0.0, ""
    score = min(1.0, 0.32 + 0.17 * len(hits))
    if "official_specialist" in text or "downstream_expert" in text:
        score += 0.12
    if "anchor" in text and capability == "general_knowledge_instruction":
        score += 0.08
    return min(score, 1.0), ",".join(hits)


def source_type_score(row: pd.Series) -> float:
    priority = str(row.get("priority", "p9"))
    source_type = str(row.get("source_type", "")).lower()
    status = str(row.get("materialization_status", "")).lower()
    score = 0.0
    if priority == "p0":
        score += 0.18
    elif priority == "p1":
        score += 0.08
    if "official" in source_type:
        score += 0.12
    if "third_party" in source_type or "downstream" in source_type:
        score += 0.10
    if "manual" in status or "resolution_required" in status:
        score -= 0.18
    if "ready" in status:
        score += 0.08
    return score


def combined_model_pool(registry: pd.DataFrame) -> pd.DataFrame:
    extra = pd.DataFrame(ALIAS_SOURCES)
    if registry.empty:
        return extra
    return pd.concat([registry, extra], ignore_index=True, sort=False)


def build_candidate_sources(
    registry: pd.DataFrame,
    task_gap_targets: pd.DataFrame,
    task_gap_policy: pd.DataFrame,
    max_models_per_task_group: int,
) -> pd.DataFrame:
    if task_gap_targets.empty:
        return pd.DataFrame()
    policy_by_task = {
        str(row["task"]): row.to_dict() for _, row in task_gap_policy.iterrows()
    } if not task_gap_policy.empty else {}
    models = combined_model_pool(registry)
    rows: list[dict[str, Any]] = []
    for _, gap in task_gap_targets.iterrows():
        task = str(gap["task"])
        capability = str(gap["capability"])
        policy = policy_by_task.get(task, {})
        for _, model in models.iterrows():
            match_score, matched_keywords = capability_match(model, capability)
            if match_score <= 0.0:
                continue
            group = same_shape_group(model)
            score = match_score + source_type_score(model)
            if group == "qwen3_moe_30b_a3b":
                score += 0.10
            if str(model.get("role", "")).lower().endswith("_pool"):
                score -= 0.12
            rows.append(
                {
                    "task": task,
                    "runner_task": gap.get("runner_task"),
                    "capability": capability,
                    "status": gap.get("status"),
                    "additional_task_gain_needed": maybe_float(gap.get("additional_task_gain_needed")),
                    "frontier_source": gap.get("frontier_source"),
                    "model_id": model.get("model_id"),
                    "role": model.get("role"),
                    "model_family": model.get("model_family"),
                    "architecture": model.get("architecture"),
                    "scale": model.get("scale"),
                    "same_shape_group": group,
                    "source_type": model.get("source_type"),
                    "merge_role": model.get("merge_role"),
                    "scenario": model.get("scenario"),
                    "materialization_status": model.get("materialization_status"),
                    "source_url": model.get("source_url"),
                    "matched_keywords": matched_keywords,
                    "capability_match_score": match_score,
                    "candidate_priority_score": score,
                    "average_policy": policy.get("average_policy"),
                    "recommended_probe": policy.get("recommended_probe"),
                    "acceptance_signal": (
                        "task_source_frontier_gain_minus_observed_interference_budget_must_be_nonnegative"
                    ),
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        ["task", "same_shape_group", "candidate_priority_score"],
        ascending=[True, True, False],
    )
    return (
        candidates.groupby(["task", "same_shape_group"], group_keys=False)
        .head(max_models_per_task_group)
        .reset_index(drop=True)
    )


def eval_tasks_for_runner_task(runner_task: Any) -> list[str]:
    runner_task = str(runner_task)
    return RETENTION_TASKS.get(runner_task, [runner_task, "mmlu"])


def eval_command(
    *,
    base_url: str,
    models: list[str],
    tasks: list[str],
    output_dir: str,
    max_examples: int,
    subjects: str,
    example_source: str,
    manifest_path: str,
) -> str:
    return shell_join(
        [
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
    )


def build_eval_jobs(
    args: argparse.Namespace,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (task, group), table in candidates.groupby(["task", "same_shape_group"], sort=True):
        concrete = table[
            ~table["model_id"].astype(str).str.startswith("hf_tree:")
            & ~table["materialization_status"].astype(str).str.contains("manual", case=False, na=False)
            & ~table["materialization_status"].astype(str).str.contains("resolution_required", case=False, na=False)
        ].copy()
        models = concrete.sort_values("candidate_priority_score", ascending=False)["model_id"].astype(str).tolist()
        if len(models) < 2:
            continue
        runner_task = str(table.iloc[0]["runner_task"])
        tasks = eval_tasks_for_runner_task(runner_task)
        job_id = slug(f"{task}_{group}_source_acquisition")
        output_dir = rel(args.output_dir / "vllm_eval" / job_id)
        manifest_path = rel(args.output_dir / "task_manifests" / f"{job_id}_task_manifest.json")
        rows.append(
            {
                "job_id": job_id,
                "task": task,
                "runner_task": runner_task,
                "same_shape_group": group,
                "model_count": len(models),
                "candidate_rows": len(table),
                "served_models": ",".join(models),
                "tasks": ",".join(tasks),
                "additional_task_gain_needed": maybe_float(table.iloc[0]["additional_task_gain_needed"]),
                "job_priority_score": maybe_float(table.iloc[0]["additional_task_gain_needed"])
                + (0.010 if group == "qwen3_moe_30b_a3b" else 0.0)
                + (0.004 if task in {"gsm8k", "humaneval"} else 0.0),
                "acceptance_signal": (
                    "promote_group_to_average_source_set_only_if_task_frontier_gain_covers_interference"
                ),
                "task_manifest": manifest_path,
                "eval_output_dir": output_dir,
                "eval_command": eval_command(
                    base_url=args.base_url,
                    models=models,
                    tasks=tasks,
                    output_dir=output_dir,
                    max_examples=args.max_examples,
                    subjects=args.subjects,
                    example_source=args.example_source,
                    manifest_path=manifest_path,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["job_priority_score", "task", "same_shape_group"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_group_gate(candidates: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    if jobs.empty:
        return pd.DataFrame()
    rows = []
    candidate_counts = (
        candidates.groupby(["task", "same_shape_group"], as_index=False)
        .agg(
            top_candidate=("model_id", "first"),
            max_priority=("candidate_priority_score", "max"),
            candidate_roles=("role", lambda values: ",".join(str(value) for value in values)),
        )
    )
    by_key = {
        (str(row["task"]), str(row["same_shape_group"])): row.to_dict()
        for _, row in candidate_counts.iterrows()
    }
    for _, job in jobs.iterrows():
        key = (str(job["task"]), str(job["same_shape_group"]))
        info = by_key.get(key, {})
        rows.append(
            {
                "task": job["task"],
                "same_shape_group": job["same_shape_group"],
                "top_candidate": info.get("top_candidate"),
                "model_count": job["model_count"],
                "additional_task_gain_needed": job["additional_task_gain_needed"],
                "minimum_acceptance_gain": job["additional_task_gain_needed"],
                "gate": "awaiting_vllm_source_frontier_eval",
                "if_passes": "add_group_to_source_set_surplus_optimizer_then_run_same_shape_average_probe",
                "if_fails": "do_not_spend_average_budget_on_this_group_for_this_task",
            }
        )
    return pd.DataFrame(rows)


def build_report(
    summary: dict[str, Any],
    candidates: pd.DataFrame,
    jobs: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen Task Source Acquisition Plan",
        "",
        "这个 artifact 把 task-level source gap 翻译成具体候选源模型和 vLLM jobs。它的目的不是直接平均更多模型，而是先证明每个 same-shape group 的 endpoint frontier 有足够 task surplus。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Task gaps: `{summary['task_gap_count']}`",
        f"- Candidate rows: `{summary['candidate_source_count']}`",
        f"- Eval jobs: `{summary['eval_job_count']}`",
        f"- Top job: `{summary['top_eval_job']}`",
        f"- Top task: `{summary['top_task']}`",
        "",
        "## Eval Jobs",
        "",
        "| job | task | group | models | tasks | needed | command |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for _, row in jobs.iterrows():
        lines.append(
            f"| `{row['job_id']}` | `{row['task']}` | `{row['same_shape_group']}` | "
            f"{int(row['model_count'])} | `{row['tasks']}` | {fmt(row['additional_task_gain_needed'])} | "
            f"`{row['eval_command']}` |"
        )
    lines.extend(
        [
            "",
            "## Top Candidate Sources",
            "",
            "| task | group | model | role | score | status |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for _, row in candidates.head(24).iterrows():
        lines.append(
            f"| `{row['task']}` | `{row['same_shape_group']}` | `{row['model_id']}` | "
            f"`{row['role']}` | {fmt(row['candidate_priority_score'])} | "
            f"`{row['materialization_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Acceptance Gates",
            "",
            "| task | group | top candidate | gate | pass action | fail action |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in gates.iterrows():
        lines.append(
            f"| `{row['task']}` | `{row['same_shape_group']}` | `{row['top_candidate']}` | "
            f"`{row['gate']}` | {row['if_passes']} | {row['if_fails']} |"
        )
    lines.extend(["", "## Literature Hooks", ""])
    for source in LITERATURE_SOURCES:
        lines.append(f"- [{source['title']}]({source['url']}): {source['mechanism']}")
    lines.extend(["", "## Outputs", ""])
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def write_outputs(
    args: argparse.Namespace,
    candidates: pd.DataFrame,
    jobs: pd.DataFrame,
    gates: pd.DataFrame,
    task_gap_targets: pd.DataFrame,
) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_task_sources.csv"
    jobs_path = output_dir / "vllm_source_acquisition_jobs.csv"
    gates_path = output_dir / "source_acquisition_gates.csv"
    literature_path = output_dir / "literature_sources.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    candidates.to_csv(candidate_path, index=False)
    jobs.to_csv(jobs_path, index=False)
    gates.to_csv(gates_path, index=False)
    literature_path.write_text(
        json.dumps(json_safe(LITERATURE_SOURCES), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    top_job = jobs.iloc[0].to_dict() if not jobs.empty else {}
    summary = {
        "schema_version": 1,
        "status": "task_source_acquisition_plan_ready" if not jobs.empty else "task_source_acquisition_plan_waiting_for_candidates",
        "task_gap_count": int(len(task_gap_targets)),
        "candidate_source_count": int(len(candidates)),
        "eval_job_count": int(len(jobs)),
        "same_shape_group_count": int(candidates["same_shape_group"].nunique()) if not candidates.empty else 0,
        "top_eval_job": top_job.get("job_id"),
        "top_task": top_job.get("task"),
        "top_same_shape_group": top_job.get("same_shape_group"),
        "top_eval_command": top_job.get("eval_command"),
        "literature_count": int(len(LITERATURE_SOURCES)),
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "candidate_task_sources": rel(candidate_path),
            "vllm_source_acquisition_jobs": rel(jobs_path),
            "source_acquisition_gates": rel(gates_path),
            "literature_sources": rel(literature_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, candidates, jobs, gates), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-gap-driven Qwen source acquisition eval jobs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_task_source_acquisition_plan"))
    parser.add_argument("--model-registry", type=Path, default=Path("results/qwen_target_model_registry/model_registry.csv"))
    parser.add_argument("--task-gap-targets", type=Path, default=Path("results/qwen_source_discovery_plan/task_gap_targets.csv"))
    parser.add_argument("--task-gap-policy", type=Path, default=Path("results/qwen3_moe_mechanism_levers/task_gap_policy.csv"))
    parser.add_argument("--base-url", default="http://HOST:PORT/v1")
    parser.add_argument("--example-source", default="datasets")
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--max-models-per-task-group", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = read_csv(args.model_registry)
    task_gap_targets = read_csv(args.task_gap_targets)
    task_gap_policy = read_csv(args.task_gap_policy)
    candidates = build_candidate_sources(
        registry,
        task_gap_targets,
        task_gap_policy,
        args.max_models_per_task_group,
    )
    jobs = build_eval_jobs(args, candidates)
    gates = build_group_gate(candidates, jobs)
    summary = write_outputs(args, candidates, jobs, gates, task_gap_targets)
    print(f"Wrote Qwen task source acquisition plan to {repo_path(args.output_dir).resolve()}")
    print(f"Status: {summary['status']}; jobs={summary['eval_job_count']}; top={summary['top_eval_job']}")


if __name__ == "__main__":
    main()
