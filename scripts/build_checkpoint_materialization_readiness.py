#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_RE = re.compile(r"\b(?:MOE_[A-Z0-9_]+|GENERAL_MODEL_PATH|CODE_MODEL_PATH|BASE|EXPERT|ROUTE_WEIGHT_[A-Z]+)\b")
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


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
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


def command_output_dir(command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    for idx, part in enumerate(parts):
        if part == "--output-dir" and idx + 1 < len(parts):
            return parts[idx + 1]
        if part.startswith("--output-dir="):
            return part.split("=", 1)[1]
    return ""


def classify_writer_command(command: str) -> dict[str, Any]:
    placeholders = sorted(set(PLACEHOLDER_RE.findall(command)))
    output_dir = command_output_dir(command)
    output_path = repo_path(output_dir) if output_dir else None
    checkpoint_exists = bool(output_path and has_safetensors(output_path))
    dry_run_only = "--dry-run" in command.split()
    if checkpoint_exists:
        status = "materialized_checkpoint_exists"
        next_action = "host_with_vllm_and_run_downstream_eval"
    elif placeholders:
        status = "blocked_by_placeholder_inputs"
        next_action = "replace placeholder model paths/route weights, run writer dry-run, then materialize"
    elif dry_run_only:
        status = "dry_run_command_ready"
        next_action = "remove --dry-run after compatibility check and write checkpoint shards"
    elif output_dir:
        status = "materialization_command_ready"
        next_action = "run writer command and verify safetensors output"
    else:
        status = "missing_writer_output_dir"
        next_action = "define output checkpoint directory"
    return {
        "writer_command": command,
        "writer_output_dir": output_dir,
        "writer_output_exists": checkpoint_exists,
        "dry_run_only": dry_run_only,
        "placeholder_count": len(placeholders),
        "placeholders": ",".join(placeholders),
        "writer_status": status,
        "next_writer_action": next_action,
    }


def writer_command_candidates() -> list[dict[str, Any]]:
    specs = [
        {
            "candidate": "moe_route_aware_candidate",
            "source": "results/moe_route_weight_recipes/writer_command.txt",
            "loadability": "qwen_moe_if_materialized",
        },
        {
            "candidate": "toy_moe_expert_weight_candidate",
            "source": "results/toy_moe_expert_weight_recipes/writer_command.txt",
            "loadability": "not_vllm_loadable_toy",
        },
        {
            "candidate": "toy_moe_expert_matched_candidate",
            "source": "results/toy_moe_expert_remap_plan/writer_command.txt",
            "loadability": "not_vllm_loadable_toy",
        },
    ]
    rows = []
    for spec in specs:
        source_path = repo_path(spec["source"])
        command = read_text_if_exists(source_path)
        classified = classify_writer_command(command)
        rows.append(
            {
                "candidate": spec["candidate"],
                "source_kind": "writer_command",
                "source_path": spec["source"],
                "loadability": spec["loadability"],
                **classified,
            }
        )
    return rows


def bias_candidate() -> dict[str, Any]:
    plan = read_json_if_exists(repo_path("results/moe_router_bias_plan/summary.json"))
    checkpoint_path = repo_path("results/checkpoints/moe_bias_calibrated_candidate")
    checkpoint_exists = has_safetensors(checkpoint_path)
    writer_ready = bool(plan.get("writer_csv_ready", False))
    if checkpoint_exists:
        status = "materialized_checkpoint_exists"
        next_action = "host_with_vllm_and_run_downstream_eval"
    elif writer_ready:
        status = "needs_real_moe_source_paths_for_tensor_add_writer"
        next_action = "run write_same_shape_average_checkpoint.py with real MoE sources and --tensor-add-csv results/moe_router_bias_plan/router_bias_deltas.csv"
    else:
        status = "bias_delta_not_ready"
        next_action = "generate router-bias deltas from a single selected MoE method"
    return {
        "candidate": "moe_bias_calibrated_candidate",
        "source_kind": "router_bias_plan",
        "source_path": "results/moe_router_bias_plan/summary.json",
        "loadability": "qwen_moe_if_materialized",
        "writer_command": "",
        "writer_output_dir": "results/checkpoints/moe_bias_calibrated_candidate",
        "writer_output_exists": checkpoint_exists,
        "dry_run_only": False,
        "placeholder_count": 0,
        "placeholders": "",
        "writer_status": status,
        "next_writer_action": next_action,
    }


def same_shape_smoke_candidate() -> dict[str, Any]:
    manifest = read_json_if_exists(repo_path("results/same_shape_writer_smoke/merge_manifest.json"))
    output_dir = "results/same_shape_writer_smoke"
    return {
        "candidate": "qwen_0_5b_writer_compatibility",
        "source_kind": "dry_run_manifest",
        "source_path": "results/same_shape_writer_smoke/merge_manifest.json",
        "loadability": "qwen_dense_if_materialized",
        "writer_command": "",
        "writer_output_dir": output_dir,
        "writer_output_exists": has_safetensors(repo_path(output_dir)),
        "dry_run_only": bool(manifest.get("dry_run", True)),
        "placeholder_count": 0,
        "placeholders": "",
        "writer_status": "dry_run_compatible_no_checkpoint_written",
        "next_writer_action": "choose a non-rejected dense average coefficient, run writer without --dry-run, then run vLLM eval plan",
    }


def vllm_rows() -> pd.DataFrame:
    path = repo_path("results/vllm_checkpoint_eval_plan/checkpoint_eval_plan.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def attach_vllm_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vllm = vllm_rows()
    by_method = {}
    if not vllm.empty:
        by_method = {str(row["method"]): row for _, row in vllm.iterrows()}
    out = []
    for row in rows:
        vllm_row = by_method.get(row["candidate"])
        if vllm_row is None:
            row["vllm_plan_status"] = "not_in_vllm_plan"
            row["vllm_eval_output_dir"] = ""
            row["vllm_eval_command"] = ""
        else:
            row["vllm_plan_status"] = str(vllm_row.get("serve_status", ""))
            row["vllm_eval_output_dir"] = str(vllm_row.get("eval_output_dir", ""))
            row["vllm_eval_command"] = str(vllm_row.get("eval_command", ""))
        if row["writer_output_exists"] and row["vllm_plan_status"] in {"ready_to_host", "not_in_vllm_plan"}:
            row["end_to_end_status"] = "ready_for_vllm_eval"
        elif row["loadability"] == "not_vllm_loadable_toy":
            row["end_to_end_status"] = "toy_writer_validation_only"
        elif row["writer_status"] == "blocked_by_placeholder_inputs":
            row["end_to_end_status"] = "blocked_before_materialization"
        elif row["writer_status"].startswith("dry_run"):
            row["end_to_end_status"] = "needs_materialization_after_dry_run"
        else:
            row["end_to_end_status"] = "needs_checkpoint_materialization"
        out.append(row)
    return out


def build_rows() -> list[dict[str, Any]]:
    return attach_vllm_status(writer_command_candidates() + [bias_candidate(), same_shape_smoke_candidate()])


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Checkpoint Materialization Readiness",
        "",
        "这个 audit 把 same-shape writer 命令、router-bias plan、dry-run manifest 和 vLLM eval plan 串起来。目标是区分三件事：方法是否有 recipe、checkpoint 是否已经写出、是否可以进入真实 vLLM 下游评测。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Materialized checkpoints: `{summary['materialized_count']}`",
        f"- Blocked by placeholders: `{summary['blocked_by_placeholder_count']}`",
        f"- Ready for vLLM eval: `{summary['ready_for_vllm_eval_count']}`",
        "",
        "| candidate | writer status | vLLM status | end-to-end status | next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate']}` | `{row['writer_status']}` | `{row['vllm_plan_status']}` | "
            f"`{row['end_to_end_status']}` | {row['next_writer_action']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['readiness_csv']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit checkpoint materialization readiness before vLLM downstream evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/checkpoint_materialization_readiness"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    df = pd.DataFrame(rows)
    csv_path = output_dir / "candidate_readiness.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "schema_version": 1,
        "status": "ready_for_vllm_eval" if (df["end_to_end_status"] == "ready_for_vllm_eval").any() else "waiting_for_checkpoint_materialization",
        "candidate_count": int(len(df)),
        "materialized_count": int(df["writer_output_exists"].sum()) if "writer_output_exists" in df else 0,
        "blocked_by_placeholder_count": int((df["writer_status"] == "blocked_by_placeholder_inputs").sum()),
        "ready_for_vllm_eval_count": int((df["end_to_end_status"] == "ready_for_vllm_eval").sum()),
        "toy_validation_only_count": int((df["end_to_end_status"] == "toy_writer_validation_only").sum()),
        "outputs": {
            "readiness_csv": rel(csv_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, rows), encoding="utf-8")
    print(f"Wrote checkpoint materialization readiness audit to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
