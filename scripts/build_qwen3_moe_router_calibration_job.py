#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT = Path("results/checkpoints/qwen3_moe_searched_no_gt065_max_retention_candidate")
DEFAULT_TEACHER = Path(
    "/srv/home/bohanlyu/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/"
    "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
)
DEFAULT_TOKENIZER = Path(
    "/srv/home/bohanlyu/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/"
    "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)
DEFAULT_PROMPTS = Path("prompts/qwen_moe_route_probe_prompts.jsonl")
DEFAULT_SOURCE_CONTROLS = [
    ("source_qwen3_30b_instruct", DEFAULT_TOKENIZER),
    ("source_qwen3_30b_coder", DEFAULT_TEACHER),
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


def shell_join(parts: list[str | Path | int | float]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def parse_source_control(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected METHOD=CHECKPOINT_PATH, got: {raw}")
    method, path = raw.split("=", 1)
    method = method.strip()
    path = path.strip()
    if not method or not path:
        raise ValueError(f"Expected METHOD=CHECKPOINT_PATH, got: {raw}")
    return method, Path(path)


def shell_func_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def has_safetensors(path: Path) -> bool:
    if path.is_file() and path.name.endswith(".safetensors"):
        return True
    if not path.is_dir():
        return False
    return any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").exists()


def gpu_status() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia_smi_missing"
    try:
        completed = subprocess.run(["nvidia-smi"], text=True, capture_output=True, timeout=10, check=False)
    except Exception as exc:  # pragma: no cover - defensive host check
        return f"nvidia_smi_error:{type(exc).__name__}"
    return "available" if completed.returncode == 0 else "unavailable"


def cap_label(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    digits = text.replace(".", "")
    return f"cap{digits}"


def build_source_control_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_controls = args.source_control
    if raw_controls is None:
        raw_controls = [f"{method}={path}" for method, path in DEFAULT_SOURCE_CONTROLS]

    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_controls):
        method, checkpoint = parse_source_control(raw)
        eval_dir = repo_path("results/vllm_checkpoint_eval") / method
        served_model = f"candidate_{method}"
        port = int(args.source_start_port) + idx
        rows.append(
            {
                "rank": idx,
                "method": method,
                "checkpoint_dir": rel(checkpoint),
                "checkpoint_exists": has_safetensors(repo_path(checkpoint)),
                "eval_dir": rel(eval_dir),
                "served_model": served_model,
                "port": port,
                "base_url": f"http://{args.host}:{port}/v1",
                "purpose": "Source endpoint control for downstream dominance checks.",
            }
        )
    return rows


def build_candidate_rows(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache_dir = output_dir / "cache"
    for idx, cap in enumerate(args.router_cap):
        label = cap_label(float(cap))
        method = f"qwen3_moe_router_calibrated_searched_no_gt065_{label}_candidate"
        delta_dir = output_dir / f"delta_{label}"
        checkpoint_dir = repo_path("results/checkpoints") / method
        audit_dir = output_dir / f"audit_{label}"
        eval_dir = repo_path("results/vllm_checkpoint_eval") / method
        served_model = f"candidate_{method}"
        port = int(args.start_port) + idx
        rows.append(
            {
                "rank": idx,
                "cap_label": label,
                "router_max_relative_norm": float(cap),
                "method": method,
                "student_checkpoint": rel(args.student),
                "teacher_checkpoint": rel(args.teacher),
                "cache_dir": rel(cache_dir),
                "cache_path": rel(cache_dir / "router_calibration_cache.pt"),
                "delta_dir": rel(delta_dir),
                "delta_safetensors": rel(delta_dir / "router_delta.safetensors"),
                "checkpoint_dir": rel(checkpoint_dir),
                "audit_dir": rel(audit_dir),
                "eval_dir": rel(eval_dir),
                "served_model": served_model,
                "port": port,
                "base_url": f"http://{args.host}:{port}/v1",
                "selection_question": (
                    "Does route-KD router calibration improve the searched no-gt-0.65 expert-only candidate "
                    "without breaking source-control downstream scores?"
                ),
            }
        )
    return rows


def collect_command(args: argparse.Namespace, output_dir: Path) -> str:
    return shell_join(
        [
            "PYTHONPATH=scripts",
            "python",
            "scripts/collect_moe_router_calibration_cache.py",
            "--student-model",
            args.student,
            "--teacher-model",
            args.teacher,
            "--tokenizer",
            args.tokenizer,
            "--prompts",
            args.prompts,
            "--device-map",
            args.device_map,
            "--dtype",
            args.dtype,
            "--max-length",
            args.max_length,
            "--top-k",
            args.top_k,
            "--capacity-factor",
            args.capacity_factor,
            "--max-samples-per-router",
            args.max_samples_per_router,
            "--use-chat-template",
            "--local-files-only",
            "--output-dir",
            output_dir / "cache",
        ]
    )


def train_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    return shell_join(
        [
            "python",
            "scripts/train_moe_router_delta_calibration.py",
            "--base",
            args.student,
            "--cache",
            row["cache_path"],
            "--output-dir",
            row["delta_dir"],
            "--epochs",
            args.epochs,
            "--lr",
            args.lr,
            "--temperature",
            args.temperature,
            "--top-k",
            args.top_k,
            "--capacity-factor",
            args.capacity_factor,
            "--top1-loss-coef",
            args.top1_loss_coef,
            "--capacity-loss-coef",
            args.capacity_loss_coef,
            "--trust-l2-coef",
            args.trust_l2_coef,
            "--max-relative-norm",
            row["router_max_relative_norm"],
        ]
    )


def writer_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    return shell_join(
        [
            "python",
            "scripts/write_same_shape_average_checkpoint.py",
            "--base",
            args.student,
            "--source",
            f"same={repo_path(args.student)}",
            "--source-weight",
            "same=0.0",
            "--freeze-router",
            "--tensor-delta-safetensors",
            row["delta_safetensors"],
            "--output-dir",
            row["checkpoint_dir"],
        ]
    )


def audit_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    return shell_join(
        [
            "python",
            "scripts/audit_materialized_checkpoint_delta.py",
            "--base",
            args.student,
            "--candidate",
            row["checkpoint_dir"],
            "--output-dir",
            row["audit_dir"],
        ]
    )


def eval_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    return shell_join(
        [
            "python",
            "scripts/run_vllm_downstream_eval.py",
            "--base-url",
            row["base_url"],
            "--models",
            row["served_model"],
            "--tasks",
            args.tasks,
            "--example-source",
            args.example_source,
            "--max-examples",
            args.max_examples,
            "--output-dir",
            row["eval_dir"],
        ]
    )


def source_eval_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    return shell_join(
        [
            "python",
            "scripts/run_vllm_downstream_eval.py",
            "--base-url",
            row["base_url"],
            "--models",
            row["served_model"],
            "--tasks",
            args.tasks,
            "--example-source",
            args.example_source,
            "--max-examples",
            args.max_examples,
            "--output-dir",
            row["eval_dir"],
        ]
    )


def baseline_eval_command(args: argparse.Namespace) -> str:
    return shell_join(
        [
            "python",
            "scripts/run_vllm_downstream_eval.py",
            "--base-url",
            f"http://{args.host}:{args.baseline_port}/v1",
            "--models",
            args.baseline_served_model,
            "--tasks",
            args.tasks,
            "--example-source",
            args.example_source,
            "--max-examples",
            args.max_examples,
            "--output-dir",
            args.baseline_eval_dir,
        ]
    )


def select_command(args: argparse.Namespace, output_dir: Path, source_rows: list[dict[str, Any]]) -> str:
    parts: list[str | Path | int | float] = [
        "python",
        "scripts/select_qwen3_moe_router_calibration_result.py",
        "--job-dir",
        output_dir,
        "--output-dir",
        repo_path("results/qwen3_moe_router_calibration_selection"),
        "--baseline-eval-dir",
        args.baseline_eval_dir,
    ]
    for row in source_rows:
        parts.extend(["--source-eval-dir", row["eval_dir"]])
    return shell_join(parts)


def serve_command(row: dict[str, Any], args: argparse.Namespace) -> str:
    return shell_join(
        [
            "CUDA_VISIBLE_DEVICES=" + args.gpus,
            "vllm",
            "serve",
            row["checkpoint_dir"],
            "--served-model-name",
            row["served_model"],
            "--host",
            args.host,
            "--port",
            row["port"],
            "--dtype",
            args.dtype,
            "--tensor-parallel-size",
            args.tensor_parallel_size,
        ]
    )


def build_stage_rows(
    args: argparse.Namespace,
    output_dir: Path,
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "stage": "collect_router_cache",
            "cap_label": "shared",
            "method": "shared_cache",
            "command": collect_command(args, output_dir),
            "expected_output": rel(output_dir / "cache/router_calibration_cache.pt"),
            "purpose": "Capture student router hidden states and teacher router logits on the shared prompt pack.",
        }
    ]
    for row in source_rows:
        rows.append(
            {
                "stage": "vllm_eval_source_control",
                "cap_label": "source",
                "method": row["method"],
                "command": source_eval_command(row, args),
                "expected_output": row["eval_dir"],
                "purpose": "Run a source endpoint control on the same downstream task set.",
            }
        )
    rows.append(
        {
            "stage": "vllm_eval_baseline",
            "cap_label": "baseline",
            "method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
            "command": baseline_eval_command(args),
            "expected_output": rel(repo_path(args.baseline_eval_dir)),
            "purpose": "Run the frozen-router searched no-gt-0.65 baseline on the same downstream tasks.",
        }
    )
    for row in candidate_rows:
        rows.extend(
            [
                {
                    "stage": "train_router_delta",
                    "cap_label": row["cap_label"],
                    "method": row["method"],
                    "command": train_command(row, args),
                    "expected_output": row["delta_safetensors"],
                    "purpose": "Train same-shape additive router delta under the cap-specific relative norm.",
                },
                {
                    "stage": "materialize_checkpoint",
                    "cap_label": row["cap_label"],
                    "method": row["method"],
                    "command": writer_command(row, args),
                    "expected_output": row["checkpoint_dir"],
                    "purpose": "Apply only the calibrated router delta to the searched no-gt-0.65 student checkpoint.",
                },
                {
                    "stage": "audit_delta",
                    "cap_label": row["cap_label"],
                    "method": row["method"],
                    "command": audit_command(row, args),
                    "expected_output": row["audit_dir"],
                    "purpose": "Verify the new checkpoint only moves intended tensors and records delta norms.",
                },
                {
                    "stage": "vllm_eval",
                    "cap_label": row["cap_label"],
                    "method": row["method"],
                    "command": eval_command(row, args),
                    "expected_output": row["eval_dir"],
                    "purpose": "Run downstream tasks against the router-calibrated ablation candidate.",
                },
            ]
        )
    rows.append(
        {
            "stage": "select_router_calibration_result",
            "cap_label": "all",
            "method": "router_calibration_selection",
            "command": select_command(args, output_dir, source_rows),
            "expected_output": rel(repo_path("results/qwen3_moe_router_calibration_selection") / "summary.json"),
            "purpose": "Select or reject router calibration using matched vLLM scores, router-only audit, and cap gates.",
        }
    )
    return rows


def build_run_script(
    args: argparse.Namespace,
    output_dir: Path,
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/build_qwen3_moe_router_calibration_job.py",
        "# Usage: results/qwen3_moe_router_calibration_job/run_router_calibration_job.sh [all|collect|sources|baseline|cap001|cap0025|cap005|eval_cap001|eval_cap0025|eval_cap005|select]",
        "",
        f"GPUS=\"${{GPUS:-{args.gpus}}}\"",
        f"HOST=\"${{HOST:-{args.host}}}\"",
        "mkdir -p results/qwen3_moe_router_calibration_job/logs",
        "",
        "wait_for_server() {",
        "  local port=\"$1\"",
        "  for _ in $(seq 1 240); do",
        "    if curl -fsS \"http://${HOST}:${port}/v1/models\" >/dev/null 2>&1; then",
        "      return 0",
        "    fi",
        "    sleep 2",
        "  done",
        "  echo \"server on port ${port} did not become ready\" >&2",
        "  return 1",
        "}",
        "",
        "stop_server() {",
        "  local pid=\"$1\"",
        "  if kill -0 \"${pid}\" >/dev/null 2>&1; then",
        "    kill \"${pid}\" || true",
        "    wait \"${pid}\" || true",
        "  fi",
        "}",
        "",
        "collect_cache() {",
        f"  {collect_command(args, output_dir)}",
        "}",
        "",
    ]
    for row in source_rows:
        func = "eval_source_" + shell_func_name(str(row["method"]))
        lines.extend(
            [
                f"{func}() {{",
                f"  CUDA_VISIBLE_DEVICES=\"${{GPUS}}\" vllm serve {shlex.quote(row['checkpoint_dir'])} "
                f"--served-model-name {shlex.quote(row['served_model'])} --host \"${{HOST}}\" "
                f"--port {int(row['port'])} --dtype {shlex.quote(args.dtype)} "
                f"--tensor-parallel-size {int(args.tensor_parallel_size)} "
                f"> results/qwen3_moe_router_calibration_job/logs/{shell_func_name(str(row['method']))}.serve.log 2>&1 &",
                "  local serve_pid=$!",
                "  trap 'stop_server ${serve_pid}' RETURN",
                f"  wait_for_server {int(row['port'])}",
                (
                    "  python scripts/run_vllm_downstream_eval.py "
                    f"--base-url \"http://${{HOST}}:{int(row['port'])}/v1\" "
                    f"--models {shlex.quote(row['served_model'])} "
                    f"--tasks {shlex.quote(args.tasks)} "
                    f"--example-source {shlex.quote(args.example_source)} "
                    f"--max-examples {int(args.max_examples)} "
                    f"--output-dir {shlex.quote(row['eval_dir'])}"
                ),
                "}",
                "",
            ]
        )
    lines.extend(
        [
            "eval_sources() {",
        ]
    )
    for row in source_rows:
        lines.append("  eval_source_" + shell_func_name(str(row["method"])))
    lines.extend(
        [
            "}",
            "",
        "eval_baseline() {",
        f"  CUDA_VISIBLE_DEVICES=\"${{GPUS}}\" vllm serve {shlex.quote(rel(args.student))} "
        f"--served-model-name {shlex.quote(args.baseline_served_model)} --host \"${{HOST}}\" "
        f"--port {int(args.baseline_port)} --dtype {shlex.quote(args.dtype)} "
        f"--tensor-parallel-size {int(args.tensor_parallel_size)} "
        f"> results/qwen3_moe_router_calibration_job/logs/baseline.serve.log 2>&1 &",
        "  local serve_pid=$!",
        "  trap 'stop_server ${serve_pid}' RETURN",
        f"  wait_for_server {int(args.baseline_port)}",
        (
            "  python scripts/run_vllm_downstream_eval.py "
            f"--base-url \"http://${{HOST}}:{int(args.baseline_port)}/v1\" "
            f"--models {shlex.quote(args.baseline_served_model)} "
            f"--tasks {shlex.quote(args.tasks)} "
            f"--example-source {shlex.quote(args.example_source)} "
            f"--max-examples {int(args.max_examples)} "
            f"--output-dir {shlex.quote(args.baseline_eval_dir)}"
        ),
        "}",
        "",
        "select_result() {",
        f"  {select_command(args, output_dir, source_rows)}",
        "}",
        "",
        ]
    )
    for row in candidate_rows:
        label = row["cap_label"]
        lines.extend(
            [
                f"train_{label}() {{",
                f"  {train_command(row, args)}",
                "}",
                "",
                f"materialize_{label}() {{",
                f"  {writer_command(row, args)}",
                f"  {audit_command(row, args)}",
                "}",
                "",
                f"eval_{label}() {{",
                f"  CUDA_VISIBLE_DEVICES=\"${{GPUS}}\" vllm serve {shlex.quote(row['checkpoint_dir'])} "
                f"--served-model-name {shlex.quote(row['served_model'])} --host \"${{HOST}}\" "
                f"--port {int(row['port'])} --dtype {shlex.quote(args.dtype)} "
                f"--tensor-parallel-size {int(args.tensor_parallel_size)} "
                f"> results/qwen3_moe_router_calibration_job/logs/{label}.serve.log 2>&1 &",
                "  local serve_pid=$!",
                "  trap 'stop_server ${serve_pid}' RETURN",
                f"  wait_for_server {int(row['port'])}",
                (
                    "  python scripts/run_vllm_downstream_eval.py "
                    f"--base-url \"http://${{HOST}}:{int(row['port'])}/v1\" "
                    f"--models {shlex.quote(row['served_model'])} "
                    f"--tasks {shlex.quote(args.tasks)} "
                    f"--example-source {shlex.quote(args.example_source)} "
                    f"--max-examples {int(args.max_examples)} "
                    f"--output-dir {shlex.quote(row['eval_dir'])}"
                ),
                "}",
                "",
                f"run_{label}() {{",
                f"  train_{label}",
                f"  materialize_{label}",
                f"  eval_{label}",
                "}",
                "",
            ]
        )
    labels = [row["cap_label"] for row in candidate_rows]
    lines.extend(
        [
            "run_all() {",
            "  collect_cache",
            "  eval_sources",
            "  eval_baseline",
        ]
    )
    for label in labels:
        lines.append(f"  run_{label}")
    lines.append("  select_result")
    lines.extend(
        [
            "}",
            "",
            "case \"${1:-all}\" in",
            "  all) run_all ;;",
            "  collect) collect_cache ;;",
            "  sources) eval_sources ;;",
            "  baseline) eval_baseline ;;",
        ]
    )
    for row in source_rows:
        lines.append(f"  {row['method']}) eval_source_{shell_func_name(str(row['method']))} ;;")
    for label in labels:
        lines.extend(
            [
                f"  {label}) run_{label} ;;",
                f"  eval_{label}) eval_{label} ;;",
            ]
        )
    lines.append("  select) select_result ;;")
    lines.extend(
        [
            "  *)",
            "    echo \"unknown target: ${1:-all}\" >&2",
            "    exit 2",
            "    ;;",
            "esac",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    summary: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Qwen3 MoE Router Calibration Job",
        "",
        "这个作业把已完成的 router calibration smoke 扩展到真实 Qwen3 candidate：先在 `searched_no_gt065` checkpoint 上收集 student router hidden states，用 Coder source 作为 teacher router logits，再训练 capped route-KD router delta，最后写出 router-calibrated ablation checkpoint 并进入 vLLM 下游评测。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Student checkpoint exists: `{summary['student_exists']}`",
        f"- Teacher checkpoint exists: `{summary['teacher_exists']}`",
        f"- Prompt pack exists: `{summary['prompts_exists']}`",
        f"- Local GPU status: `{summary['local_gpu_status']}`",
        f"- Router caps: `{', '.join(str(item) for item in summary['router_caps'])}`",
        f"- Baseline eval dir: `{summary['baseline_eval_dir']}`",
        f"- Source control count: `{summary['source_control_count']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Router validation gate: `{summary['router_validation_gate']}`",
        f"- Selection output: `{summary['outputs']['selection']}`",
        "",
        "## Why This Ablation",
        "",
        "Direct Instruct/Coder router weight movement was rejected by the router move gate. This job tests a narrower mechanism: keep the best frozen-router expert candidate fixed, then add a small route-KD router delta learned from real hidden states and teacher logits. If downstream scores improve without routing collapse, router calibration becomes a valid next component; otherwise the unified rule keeps router frozen.",
        "",
        "The selector requires group-heldout route-KD validation by default: prompt/batch groups used for selection must be absent from router-delta training rows. This prevents a row-level random split from overstating router generalization.",
        "",
        "## Source Controls",
        "",
        "| method | checkpoint | port | eval output |",
        "|---|---|---:|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| `{row['method']}` | `{row['checkpoint_dir']}` | {int(row['port'])} | `{row['eval_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| cap | method | checkpoint | port |",
            "|---:|---|---|---:|",
        ]
    )
    for row in candidate_rows:
        lines.append(
            f"| {row['router_max_relative_norm']:.3f} | `{row['method']}` | "
            f"`{row['checkpoint_dir']}` | {int(row['port'])} |"
        )
    lines.extend(
        [
            "",
            "## Stages",
            "",
            "| stage | cap | expected output |",
            "|---|---:|---|",
        ]
    )
    for row in stage_rows:
        lines.append(f"| `{row['stage']}` | `{row['cap_label']}` | `{row['expected_output']}` |")
    lines.extend(
        [
            "",
            "## Run",
            "",
            "```bash",
            summary["run_script_command"],
            "```",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen3 MoE router-calibration job spec and run script.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_calibration_job"))
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--router-cap", type=float, action="append", default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--top1-loss-coef", type=float, default=0.25)
    parser.add_argument("--capacity-loss-coef", type=float, default=0.1)
    parser.add_argument("--trust-l2-coef", type=float, default=0.001)
    parser.add_argument("--max-samples-per-router", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--source-control",
        action="append",
        default=None,
        help="Optional METHOD=CHECKPOINT_PATH source endpoint control. Defaults to Qwen3 Instruct and Coder.",
    )
    parser.add_argument("--source-start-port", type=int, default=8100)
    parser.add_argument("--start-port", type=int, default=8108)
    parser.add_argument("--tasks", default="gsm8k,mmlu,safety,humaneval_compile")
    parser.add_argument("--example-source", default="datasets")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument(
        "--baseline-eval-dir",
        default="results/vllm_checkpoint_eval/qwen3_moe_searched_no_gt065_max_retention_candidate",
    )
    parser.add_argument("--baseline-served-model", default="baseline_qwen3_moe_searched_no_gt065")
    parser.add_argument("--baseline-port", type=int, default=8107)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.router_cap is None:
        args.router_cap = [0.01, 0.025, 0.05]
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = build_source_control_rows(args)
    candidate_rows = build_candidate_rows(args, output_dir)
    stage_rows = build_stage_rows(args, output_dir, source_rows, candidate_rows)
    run_script = output_dir / "run_router_calibration_job.sh"
    run_script.write_text(build_run_script(args, output_dir, source_rows, candidate_rows), encoding="utf-8")
    run_script.chmod(0o755)
    pd.DataFrame(source_rows).to_csv(output_dir / "source_control_plan.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_plan.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output_dir / "stage_plan.csv", index=False)

    student_exists = has_safetensors(repo_path(args.student))
    teacher_exists = has_safetensors(repo_path(args.teacher))
    tokenizer_exists = repo_path(args.tokenizer).exists()
    prompts_exists = repo_path(args.prompts).exists()
    source_controls_ready = all(bool(row["checkpoint_exists"]) for row in source_rows)
    local_gpu_status = gpu_status()
    status = (
        "job_ready_awaiting_gpu"
        if student_exists
        and teacher_exists
        and tokenizer_exists
        and prompts_exists
        and source_controls_ready
        and local_gpu_status != "available"
        else "job_ready"
        if student_exists and teacher_exists and tokenizer_exists and prompts_exists and source_controls_ready
        else "missing_inputs"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "student": rel(args.student),
        "teacher": rel(args.teacher),
        "tokenizer": rel(args.tokenizer),
        "prompts": rel(args.prompts),
        "student_exists": student_exists,
        "teacher_exists": teacher_exists,
        "tokenizer_exists": tokenizer_exists,
        "prompts_exists": prompts_exists,
        "local_gpu_status": local_gpu_status,
        "router_caps": [float(item) for item in args.router_cap],
        "baseline_eval_dir": rel(args.baseline_eval_dir),
        "baseline_served_model": args.baseline_served_model,
        "baseline_port": int(args.baseline_port),
        "source_control_count": int(len(source_rows)),
        "source_controls_ready": source_controls_ready,
        "candidate_count": int(len(candidate_rows)),
        "stage_count": int(len(stage_rows)),
        "run_script_command": rel(run_script) + " all",
        "mechanism": "route_kd_router_delta_after_frozen_router_expert_cap_law_candidate",
        "router_validation_gate": "require_group_heldout_prompt_batch_validation",
        "outputs": {
            "source_control_plan": rel(output_dir / "source_control_plan.csv"),
            "candidate_plan": rel(output_dir / "candidate_plan.csv"),
            "stage_plan": rel(output_dir / "stage_plan.csv"),
            "run_script": rel(run_script),
            "selection": rel(repo_path("results/qwen3_moe_router_calibration_selection") / "summary.json"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, source_rows, candidate_rows, stage_rows), encoding="utf-8")
    print(f"Wrote Qwen3 MoE router calibration job to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
