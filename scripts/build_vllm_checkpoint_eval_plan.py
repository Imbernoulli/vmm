#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFETENSOR_PATTERNS = ("*.safetensors", "model.safetensors.index.json")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_pair(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected METHOD=CHECKPOINT_PATH, got: {raw}")
    method, path = raw.split("=", 1)
    method = method.strip()
    path = path.strip()
    if not method or not path:
        raise ValueError(f"Expected non-empty METHOD=CHECKPOINT_PATH, got: {raw}")
    return method, path


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def has_safetensors(path: Path) -> bool:
    if path.is_file() and path.name.endswith(".safetensors"):
        return True
    if not path.is_dir():
        return False
    for pattern in SAFETENSOR_PATTERNS:
        if any(path.glob(pattern)):
            return True
    return False


def load_manual_candidates(raw_candidates: list[str]) -> list[dict[str, Any]]:
    rows = []
    for raw in raw_candidates:
        method, checkpoint_path = parse_pair(raw)
        rows.append(
            {
                "candidate_source": "manual_args",
                "method": method,
                "checkpoint_path": checkpoint_path,
            }
        )
    return rows


def load_candidate_table(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.candidate_table:
        return []
    path = repo_path(args.candidate_table)
    table = read_csv_if_exists(path)
    if table is None or table.empty:
        raise ValueError(f"Candidate table is missing or empty: {path}")
    if args.candidate_query:
        table = table.query(args.candidate_query).copy()
    if args.max_candidates > 0:
        table = table.head(args.max_candidates).copy()
    missing = [col for col in (args.method_column, args.checkpoint_column) if col not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows = []
    for _, item in table.iterrows():
        checkpoint_path = str(item[args.checkpoint_column]).strip()
        if not checkpoint_path:
            continue
        row = {
            "candidate_source": rel(path),
            "method": str(item[args.method_column]).strip(),
            "checkpoint_path": checkpoint_path,
        }
        for optional in ("served_model_id", "dtype", "tensor_parallel_size", "gpu", "notes", "materialization_status"):
            if optional in item and pd.notna(item[optional]):
                row[optional] = item[optional]
        rows.append(row)
    return rows


def default_candidates() -> list[dict[str, Any]]:
    return [
        {
            "candidate_source": "qwen3_source_endpoint",
            "method": "source_qwen3_30b_instruct",
            "checkpoint_path": "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "source_checkpoint_ready_if_local_cache_is_complete",
            "notes": "Source endpoint for Qwen3-30B-A3B-Instruct-2507; compare candidate quality against this anchor on the same vLLM eval tasks.",
        },
        {
            "candidate_source": "qwen3_source_endpoint",
            "method": "source_qwen3_30b_coder",
            "checkpoint_path": "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "source_checkpoint_ready_if_local_cache_is_complete",
            "notes": "Source endpoint for Qwen3-Coder-30B-A3B-Instruct; needed to measure whether the route-guarded average preserves coding behavior.",
        },
        {
            "candidate_source": "results/qwen3_moe_unified_route_guarded_candidate/writer_command.txt",
            "method": "qwen3_moe_unified_route_guarded_candidate",
            "checkpoint_path": "results/checkpoints/qwen3_moe_unified_route_guarded_candidate",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "local_checkpoint_materialized_if_safetensors_present",
            "notes": "Qwen3-30B Instruct/Coder same-shape route-guarded candidate. It has been materialized locally; host it if the ignored safetensors checkpoint is present.",
        },
        {
            "candidate_source": "results/qwen3_moe_audit_gated_candidate/writer_command.txt",
            "method": "qwen3_moe_audit_gated_candidate",
            "checkpoint_path": "results/checkpoints/qwen3_moe_audit_gated_candidate",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "checkpoint_missing_until_audit_gated_candidate_materialized",
            "notes": "Qwen3-30B route-guarded candidate with additional safetensors-delta audit cap on high-relative-move expert rules. Writer dry-run is validated; materialize before vLLM eval.",
        },
        {
            "candidate_source": "results/qwen3_moe_trust_region_candidate/writer_command.txt",
            "method": "qwen3_moe_trust_region_candidate",
            "checkpoint_path": "results/checkpoints/qwen3_moe_trust_region_candidate",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "checkpoint_missing_until_trust_region_candidate_materialized",
            "notes": "Qwen3-30B route-guarded candidate with MoE trust-region caps from route load, category specialization, router fragility, and materialized delta audit probes. Dry-run is validated; materialize before vLLM eval.",
        },
        {
            "candidate_source": "results/qwen3_moe_expert_only_trust_region_candidate/writer_command.txt",
            "method": "qwen3_moe_expert_only_trust_region_candidate",
            "checkpoint_path": "results/checkpoints/qwen3_moe_expert_only_trust_region_candidate",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "checkpoint_missing_until_expert_only_ablation_materialized",
            "notes": "Ablation of the trust-region candidate that freezes shared attention while keeping routed expert trust-region rules. Use it to test whether Coder attention deltas help or hurt under the same vLLM tasks.",
        },
        {
            "candidate_source": "results/qwen3_moe_tail_trimmed_expert_only_candidate/writer_command.txt",
            "method": "qwen3_moe_tail_trimmed_expert_only_candidate",
            "checkpoint_path": "results/checkpoints/qwen3_moe_tail_trimmed_expert_only_candidate",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "checkpoint_missing_until_tail_trimmed_candidate_materialized",
            "notes": "Second-stage routed-expert tail trim from the expert-only candidate. Shared attention and router remain frozen; remaining high-tail expert groups are scaled toward a 0.65 relative-delta cap.",
        },
        {
            "candidate_source": "results/qwen3_moe_trust_region_cap_search/searched_no_gt065_max_retention_writer_command.txt",
            "method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
            "checkpoint_path": "results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate",
            "tensor_parallel_size": 4,
            "gpu": "0,1,2,3",
            "materialization_status": "checkpoint_missing_until_searched_cap_law_candidate_materialized",
            "notes": "Searched Qwen3 MoE cap-law candidate: freeze router and shared attention, keep source-route-conditioned expert weights, and replace hand-built risk penalties with a simple global 0.65 routed-expert relative-delta cap.",
        },
        {
            "candidate_source": "local_materialized_dense_baseline",
            "method": "qwen_0_5b_instruct_coder_uniform_average",
            "checkpoint_path": "results/checkpoints/qwen_0_5b_instruct_coder_uniform_average",
            "materialization_status": "local_ignored_checkpoint_materialized",
            "notes": "Dense Qwen2.5-0.5B Instruct/Coder 0.5/0.5 uniform-average negative baseline. The checkpoint is a local ignored artifact and is not committed to git.",
        },
        {
            "candidate_source": "default_same_shape_recipes",
            "method": "moe_route_aware_candidate",
            "checkpoint_path": "results/checkpoints/moe_route_aware_candidate",
            "materialization_status": "checkpoint_missing_until_route_weight_recipe_is_materialized",
            "notes": "Generated by results/moe_route_weight_recipes/writer_command.txt after real MoE routing probe fills expert weights.",
        },
        {
            "candidate_source": "default_same_shape_recipes",
            "method": "moe_bias_calibrated_candidate",
            "checkpoint_path": "results/checkpoints/moe_bias_calibrated_candidate",
            "materialization_status": "checkpoint_missing_until_tensor_add_csv_is_applied",
            "notes": "Use scripts/write_same_shape_average_checkpoint.py with --tensor-add-csv results/moe_router_bias_plan/router_bias_deltas.csv after materializing compatible MoE sources.",
        },
        {
            "candidate_source": "default_same_shape_recipes",
            "method": "toy_moe_expert_weight_candidate",
            "checkpoint_path": "results/checkpoints/toy_moe_expert_weight_candidate",
            "materialization_status": "toy_recipe_not_a_real_qwen_vllm_checkpoint",
            "notes": "Toy recipe validates writer logic only; it is not a vLLM-loadable Qwen checkpoint.",
        },
    ]


def served_model_id(method: str, template: str) -> str:
    return template.format(method=method).replace("/", "_")


def build_plan_rows(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, candidate in enumerate(candidates):
        method = str(candidate["method"])
        checkpoint_path = str(candidate["checkpoint_path"])
        checkpoint = repo_path(checkpoint_path)
        port = int(args.start_port) + idx
        base_url = f"http://{args.host}:{port}/v1"
        served_id = str(candidate.get("served_model_id") or served_model_id(method, args.served_model_template))
        dtype = str(candidate.get("dtype") or args.dtype)
        tensor_parallel_size = int(candidate.get("tensor_parallel_size") or args.tensor_parallel_size)
        gpu = str(candidate.get("gpu") if "gpu" in candidate else args.gpu)
        output_dir = repo_path(args.eval_output_root) / method
        eval_summary = read_json_if_exists(output_dir / "summary.json")
        model_summary = eval_summary.get("model_summary", [])
        first_model_summary = model_summary[0] if model_summary else {}
        checkpoint_display = rel(checkpoint)
        checkpoint_loadable = has_safetensors(checkpoint)
        serve_parts = []
        if gpu:
            serve_parts.append(f"CUDA_VISIBLE_DEVICES={gpu}")
        serve_parts.extend(
            [
                "vllm",
                "serve",
                checkpoint_display,
                "--served-model-name",
                served_id,
                "--host",
                args.host,
                "--port",
                str(port),
                "--dtype",
                dtype,
                "--tensor-parallel-size",
                str(tensor_parallel_size),
            ]
        )
        if args.trust_remote_code:
            serve_parts.append("--trust-remote-code")
        if args.max_model_len:
            serve_parts.extend(["--max-model-len", str(args.max_model_len)])
        if args.gpu_memory_utilization:
            serve_parts.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
        if args.vllm_extra_args:
            serve_parts.extend(shlex.split(args.vllm_extra_args))

        eval_parts = [
            "python",
            "scripts/run_vllm_downstream_eval.py",
            "--base-url",
            base_url,
            "--models",
            served_id,
            "--tasks",
            args.tasks,
            "--example-source",
            args.example_source,
            "--max-examples",
            str(args.max_examples),
            "--output-dir",
            rel(output_dir),
        ]
        if args.eval_extra_args:
            eval_parts.extend(shlex.split(args.eval_extra_args))

        status = "ready_to_host" if checkpoint_loadable else "checkpoint_missing_until_materialized"
        if str(candidate.get("materialization_status", "")).startswith("toy_"):
            status = "not_vllm_loadable_toy_candidate"
        eval_status = str(eval_summary.get("status", "not_run"))
        rows.append(
            {
                "eval_order": idx,
                "candidate_source": candidate.get("candidate_source", ""),
                "method": method,
                "checkpoint_path": checkpoint_display,
                "checkpoint_exists": checkpoint_loadable,
                "serve_status": status,
                "served_model_id": served_id,
                "host": args.host,
                "port": port,
                "base_url": base_url,
                "dtype": dtype,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu": gpu,
                "tasks": args.tasks,
                "example_source": args.example_source,
                "max_examples": args.max_examples,
                "eval_output_dir": rel(output_dir),
                "eval_status": eval_status,
                "eval_completed": eval_status == "complete",
                "eval_avg_primary_score": first_model_summary.get("avg_primary_score"),
                "eval_worst_primary_score": first_model_summary.get("worst_primary_score"),
                "serve_command": shell_join(serve_parts),
                "eval_command": shell_join(eval_parts),
                "notes": candidate.get("notes", ""),
            }
        )
    return rows


def build_shell_script(rows: list[dict[str, Any]], *, wait_command: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Command plan only. Copy the relevant commands after the checkpoint exists.",
        "# Start one vLLM server per candidate in a separate terminal or job manager,",
        "# then run the paired eval command after /v1/models is reachable.",
        "",
    ]
    for row in rows:
        command_prefix = "# "
        lines.extend(
            [
                f"# [{row['eval_order']}] {row['method']} - {row['serve_status']}",
                f"# Checkpoint: {row['checkpoint_path']}",
                f"# Serve:",
                f"{command_prefix}{row['serve_command']}",
                "",
                f"# Wait:",
                f"{command_prefix}{wait_command.format(base_url=row['base_url'], served_model_id=row['served_model_id'])}",
                "",
                f"# Eval:",
                f"{command_prefix}{row['eval_command']}",
                "",
            ]
        )
    return "\n".join(lines)


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# vLLM Checkpoint Eval Plan",
        "",
        "这个计划把 source baseline 和 same-shape checkpoint 候选转成逐个 `vllm serve` 和 `run_vllm_downstream_eval.py` 命令。它不声称已经完成真实性能评测；只有 `serve_status = ready_to_host` 且目录内存在 safetensors/index 的模型才能进入 GPU/vLLM 下游评测。",
        "",
        f"- Plan status: `{summary['status']}`",
        f"- Candidate rows: `{summary['candidate_count']}`",
        f"- Ready to host: `{summary['ready_to_host_count']}`",
        f"- Completed evals: `{summary['completed_eval_count']}`",
        f"- Missing checkpoints: `{summary['missing_checkpoint_count']}`",
        f"- Tasks: `{summary['tasks']}`",
        "",
        "## Plan",
        "",
        "| order | method | serve status | eval status | avg primary | worst primary | checkpoint | port | output |",
        "| ---: | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        avg_score = row.get("eval_avg_primary_score")
        worst_score = row.get("eval_worst_primary_score")
        avg_text = "" if avg_score is None or pd.isna(avg_score) else f"{float(avg_score):.3f}"
        worst_text = "" if worst_score is None or pd.isna(worst_score) else f"{float(worst_score):.3f}"
        lines.append(
            f"| {int(row['eval_order'])} | `{row['method']}` | `{row['serve_status']}` | "
            f"`{row['eval_status']}` | {avg_text} | {worst_text} | "
            f"`{row['checkpoint_path']}` | {int(row['port'])} | `{row['eval_output_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- `{summary['outputs']['shell_script']}`",
            f"- `{summary['outputs']['plan_csv']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a vLLM serve/eval plan for source baselines and same-shape checkpoint candidates.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/vllm_checkpoint_eval_plan"))
    parser.add_argument("--candidate", action="append", default=[], help="Candidate METHOD=CHECKPOINT_PATH. Repeatable.")
    parser.add_argument("--candidate-table", default=None, help="Optional CSV with method/checkpoint_path columns.")
    parser.add_argument("--candidate-query", default=None)
    parser.add_argument("--method-column", default="method")
    parser.add_argument("--checkpoint-column", default="checkpoint_path")
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--start-port", type=int, default=8100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--served-model-template", default="candidate_{method}")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.0)
    parser.add_argument("--vllm-extra-args", default="")
    parser.add_argument("--tasks", default="gsm8k,mmlu,safety,humaneval_compile")
    parser.add_argument("--example-source", choices=["datasets", "builtin"], default="datasets")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--eval-output-root", default="results/vllm_checkpoint_eval")
    parser.add_argument("--eval-extra-args", default="")
    parser.add_argument(
        "--wait-command",
        default="curl -sf {base_url}/models >/dev/null",
        help="Template shown between serve and eval commands. Supports {base_url} and {served_model_id}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_manual_candidates(args.candidate) + load_candidate_table(args)
    if not candidates:
        candidates = default_candidates()
    rows = build_plan_rows(args, candidates)
    plan_df = pd.DataFrame(rows)
    plan_path = output_dir / "checkpoint_eval_plan.csv"
    shell_path = output_dir / "serve_and_eval_commands.sh"
    plan_df.to_csv(plan_path, index=False)
    shell_path.write_text(build_shell_script(rows, wait_command=args.wait_command) + "\n", encoding="utf-8")
    shell_path.chmod(0o755)

    ready_count = int(sum(row["serve_status"] == "ready_to_host" for row in rows))
    missing_count = int(sum(row["serve_status"] == "checkpoint_missing_until_materialized" for row in rows))
    not_loadable_count = int(sum(row["serve_status"] == "not_vllm_loadable_toy_candidate" for row in rows))
    completed_eval_count = int(sum(bool(row["eval_completed"]) for row in rows))
    summary = {
        "schema_version": 1,
        "status": "hosted_eval_complete"
        if completed_eval_count
        else "ready_to_host"
        if ready_count
        else "waiting_for_checkpoint_materialization",
        "candidate_count": len(rows),
        "ready_to_host_count": ready_count,
        "completed_eval_count": completed_eval_count,
        "missing_checkpoint_count": missing_count,
        "not_vllm_loadable_count": not_loadable_count,
        "tasks": args.tasks,
        "example_source": args.example_source,
        "same_shape_constraint": "The plan evaluates one materialized checkpoint at a time through vLLM; it does not use ensembles or change model structure.",
        "outputs": {
            "plan_csv": rel(plan_path),
            "shell_script": rel(shell_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, rows), encoding="utf-8")
    print(f"Wrote vLLM checkpoint eval plan to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
