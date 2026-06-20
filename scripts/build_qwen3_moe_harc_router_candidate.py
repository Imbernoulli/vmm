#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file

from write_same_shape_average_checkpoint import discover_safetensors, load_tensors, write_average_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8


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
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    value = clean_value(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_float(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def writer_args(base_dir: Path, delta_path: Path, checkpoint_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        base=str(base_dir),
        source=[f"same={base_dir}"],
        source_weight=["same=0.0"],
        tensor_rule=[],
        tensor_rule_file=[],
        tensor_method_rule=[],
        tensor_method_rule_file=[],
        source_tensor_alias=[],
        source_tensor_alias_file=[],
        tensor_add_csv=[],
        tensor_delta_safetensors=[str(delta_path)],
        packed_expert_rule_csv=[],
        freeze_regex=[],
        freeze_router=True,
        allow_missing_source_tensors=False,
        output_dtype="base",
        output_dir=str(checkpoint_dir),
        copy_metadata=True,
        dry_run=False,
    )


def writer_command(base_dir: Path, delta_path: Path, checkpoint_dir: Path) -> str:
    return (
        "python scripts/write_same_shape_average_checkpoint.py "
        f"--base {rel(base_dir)} "
        f"--source same={rel(base_dir)} "
        "--source-weight same=0.0 "
        "--freeze-router "
        f"--tensor-delta-safetensors {rel(delta_path)} "
        f"--output-dir {rel(checkpoint_dir)}"
    )


def materialize_checkpoint(base_dir: Path, delta_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    return write_average_checkpoint(writer_args(base_dir, delta_path, checkpoint_dir))


def load_selected_tensors(path: Path, names: list[str]) -> dict[str, torch.Tensor]:
    index = discover_safetensors(path)
    missing = [name for name in names if name not in index.tensor_info]
    if missing:
        raise KeyError(f"Missing tensors in {path}: {missing}")
    return load_tensors(index, names)


def first_control_tensor(base_dir: Path, delta_names: set[str]) -> str | None:
    index = discover_safetensors(base_dir)
    for name in sorted(index.tensor_info):
        if name not in delta_names:
            return name
    return None


def verify_materialization(
    *,
    base_dir: Path,
    checkpoint_dir: Path,
    delta_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    delta_index = discover_safetensors(delta_path)
    delta_names = sorted(delta_index.tensor_info)
    delta_tensors = load_tensors(delta_index, delta_names)
    control = first_control_tensor(base_dir, set(delta_names))
    check_names = delta_names + ([control] if control else [])
    base_tensors = load_selected_tensors(base_dir, check_names)
    candidate_tensors = load_selected_tensors(checkpoint_dir, check_names)

    rows: list[dict[str, Any]] = []
    for name in delta_names:
        base_tensor = base_tensors[name]
        candidate = candidate_tensors[name]
        delta = delta_tensors[name]
        expected = (base_tensor.to(torch.float32) + delta.to(torch.float32)).to(candidate.dtype).to(torch.float32)
        actual = candidate.to(torch.float32)
        max_abs_error = float((actual - expected).abs().max().item())
        shape_preserved = tuple(base_tensor.shape) == tuple(candidate.shape)
        rows.append(
            {
                "tensor": name,
                "role": "router_delta_applied",
                "base_shape": list(base_tensor.shape),
                "candidate_shape": list(candidate.shape),
                "delta_shape": list(delta.shape),
                "max_abs_error": max_abs_error,
                "shape_preserved": shape_preserved,
                "passed": bool(shape_preserved and max_abs_error <= 1e-6),
            }
        )

    if control:
        base_tensor = base_tensors[control]
        candidate = candidate_tensors[control]
        expected = base_tensor.to(candidate.dtype).to(torch.float32)
        actual = candidate.to(torch.float32)
        max_abs_error = float((actual - expected).abs().max().item())
        shape_preserved = tuple(base_tensor.shape) == tuple(candidate.shape)
        rows.append(
            {
                "tensor": control,
                "role": "non_router_unchanged",
                "base_shape": list(base_tensor.shape),
                "candidate_shape": list(candidate.shape),
                "delta_shape": None,
                "max_abs_error": max_abs_error,
                "shape_preserved": shape_preserved,
                "passed": bool(shape_preserved and max_abs_error <= 1e-6),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "materialization_checks.csv", index=False)
    router_rows = frame[frame["role"] == "router_delta_applied"] if not frame.empty else pd.DataFrame()
    unchanged_rows = frame[frame["role"] == "non_router_unchanged"] if not frame.empty else pd.DataFrame()
    return {
        "check_count": int(len(frame)),
        "passed_check_count": int(frame["passed"].astype(bool).sum()) if not frame.empty else 0,
        "router_delta_check_count": int(len(router_rows)),
        "router_delta_passed_count": int(router_rows["passed"].astype(bool).sum()) if not router_rows.empty else 0,
        "non_router_check_count": int(len(unchanged_rows)),
        "non_router_passed_count": int(unchanged_rows["passed"].astype(bool).sum()) if not unchanged_rows.empty else 0,
        "max_abs_error": float(frame["max_abs_error"].max()) if not frame.empty else None,
        "checks": rows,
    }


def build_requirements(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement": "solver_ready",
                "passed": summary.get("solver_status") in {"harc_solver_ready", "smoke_ready"},
                "evidence": summary.get("solver_status"),
            },
            {
                "requirement": "router_delta_exists",
                "passed": bool(summary.get("router_delta_exists")),
                "evidence": summary.get("router_delta_safetensors"),
            },
            {
                "requirement": "base_checkpoint_exists",
                "passed": bool(summary.get("base_exists")),
                "evidence": summary.get("base_path"),
            },
            {
                "requirement": "checkpoint_materialized",
                "passed": bool(summary.get("candidate_checkpoint_exists")),
                "evidence": summary.get("candidate_checkpoint_dir"),
            },
            {
                "requirement": "router_delta_applied",
                "passed": summary.get("router_delta_passed_count") == summary.get("router_delta_check_count")
                and bool(summary.get("router_delta_check_count")),
                "evidence": f"{summary.get('router_delta_passed_count')}/{summary.get('router_delta_check_count')}",
            },
            {
                "requirement": "non_router_unchanged",
                "passed": summary.get("non_router_passed_count") == summary.get("non_router_check_count")
                and bool(summary.get("non_router_check_count")),
                "evidence": f"{summary.get('non_router_passed_count')}/{summary.get('non_router_check_count')}",
            },
        ]
    )


def build_report(summary: dict[str, Any], requirements: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE HARC Router Candidate",
        "",
        "这个 gate 把 HARC router solver 产生的同形状 `router_delta.safetensors` 写入一个可直接进入 vLLM 评测队列的同结构 checkpoint。它验证的不是“router 不动”，而是“router 只按 HARC delta 改动，非 router 权重保持不变”。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Solver status: `{summary.get('solver_status')}`",
        f"- Base checkpoint: `{summary.get('base_path')}` (exists `{summary.get('base_exists')}`)",
        f"- Router delta: `{summary.get('router_delta_safetensors')}` (exists `{summary.get('router_delta_exists')}`)",
        f"- Candidate checkpoint: `{summary.get('candidate_checkpoint_dir')}` (exists `{summary.get('candidate_checkpoint_exists')}`)",
        f"- Delta tensors: `{summary.get('delta_tensor_count')}`",
        f"- Materialization checks: `{summary.get('passed_check_count')}/{summary.get('check_count')}` passed, max error `{fmt(summary.get('max_abs_error'))}`",
        f"- Recommended action: `{summary.get('recommended_action')}`",
        "",
        "## Requirements",
        "",
        "| requirement | passed | evidence |",
        "| --- | ---: | --- |",
    ]
    for _, row in requirements.iterrows():
        lines.append(f"| `{row['requirement']}` | `{bool(row['passed'])}` | `{row['evidence']}` |")
    lines.extend(
        [
            "",
            "## Writer Command",
            "",
            "```bash",
            str(summary.get("writer_command") or ""),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def output_paths(output_dir: Path, checkpoint_dir: Path) -> dict[str, str]:
    return {
        "summary": rel(output_dir / "summary.json"),
        "report": rel(output_dir / "report.md"),
        "requirements": rel(output_dir / "harc_candidate_requirements.csv"),
        "materialization_checks": rel(output_dir / "materialization_checks.csv"),
        "writer_command": rel(output_dir / "writer_command.txt"),
        "checkpoint": rel(checkpoint_dir),
    }


def write_outputs(output_dir: Path, checkpoint_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = output_paths(output_dir, checkpoint_dir)
    requirements = build_requirements(summary)
    requirements.to_csv(output_dir / "harc_candidate_requirements.csv", index=False)
    (output_dir / "writer_command.txt").write_text(str(summary.get("writer_command") or "") + "\n", encoding="utf-8")
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(build_report(summary, requirements), encoding="utf-8")
    return summary


def waiting_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_dir: Path,
    solver_summary: dict[str, Any],
    delta_path: Path,
    base_dir: Path,
    status: str,
    recommended_action: str,
) -> dict[str, Any]:
    pd.DataFrame(
        columns=[
            "tensor",
            "role",
            "base_shape",
            "candidate_shape",
            "delta_shape",
            "max_abs_error",
            "shape_preserved",
            "passed",
        ]
    ).to_csv(output_dir / "materialization_checks.csv", index=False)
    summary = {
        "schema_version": 1,
        "status": status,
        "smoke_status": None,
        "solver_summary": rel(args.solver_summary),
        "solver_status": solver_summary.get("status"),
        "base_path": rel(base_dir),
        "base_exists": base_dir.exists(),
        "router_delta_safetensors": rel(delta_path),
        "router_delta_exists": delta_path.exists(),
        "candidate_checkpoint_dir": rel(checkpoint_dir),
        "candidate_checkpoint_exists": checkpoint_dir.exists()
        and any(checkpoint_dir.glob("*.safetensors")),
        "delta_tensor_count": 0,
        "writer_command": writer_command(base_dir, delta_path, checkpoint_dir),
        "check_count": 0,
        "passed_check_count": 0,
        "router_delta_check_count": 0,
        "router_delta_passed_count": 0,
        "non_router_check_count": 0,
        "non_router_passed_count": 0,
        "max_abs_error": None,
        "recommended_action": recommended_action,
    }
    return write_outputs(output_dir, checkpoint_dir, summary)


def build_candidate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = repo_path(args.checkpoint_dir)
    solver_summary = read_json(args.solver_summary)
    base_dir = repo_path(args.base)
    raw_delta = args.router_delta or (solver_summary.get("outputs") or {}).get("router_delta_safetensors")
    delta_path = repo_path(raw_delta) if raw_delta else repo_path(args.solver_dir) / "router_delta.safetensors"

    solver_status = solver_summary.get("status")
    if solver_status != "harc_solver_ready":
        return waiting_summary(
            args=args,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            solver_summary=solver_summary,
            delta_path=delta_path,
            base_dir=base_dir,
            status="harc_router_candidate_waiting_for_solver_delta",
            recommended_action="collect_real_router_cache_then_rerun_harc_solver",
        )
    if not delta_path.exists():
        return waiting_summary(
            args=args,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            solver_summary=solver_summary,
            delta_path=delta_path,
            base_dir=base_dir,
            status="harc_router_candidate_waiting_for_solver_delta",
            recommended_action="rerun_harc_solver_to_write_router_delta_safetensors",
        )
    if not base_dir.exists():
        return waiting_summary(
            args=args,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            solver_summary=solver_summary,
            delta_path=delta_path,
            base_dir=base_dir,
            status="harc_router_candidate_waiting_for_base_checkpoint",
            recommended_action="materialize_or_sync_the_base_same_shape_checkpoint_before_harc_candidate",
        )

    manifest = materialize_checkpoint(base_dir, delta_path, checkpoint_dir)
    checks = verify_materialization(
        base_dir=base_dir,
        checkpoint_dir=checkpoint_dir,
        delta_path=delta_path,
        output_dir=output_dir,
    )
    delta_index = discover_safetensors(delta_path)
    all_passed = checks["passed_check_count"] == checks["check_count"] and checks["check_count"] > 0
    summary = {
        "schema_version": 1,
        "status": "harc_router_candidate_materialized" if all_passed else "harc_router_candidate_check_failed",
        "smoke_status": None,
        "solver_summary": rel(args.solver_summary),
        "solver_status": solver_status,
        "base_path": rel(base_dir),
        "base_exists": True,
        "router_delta_safetensors": rel(delta_path),
        "router_delta_exists": True,
        "candidate_checkpoint_dir": rel(checkpoint_dir),
        "candidate_checkpoint_exists": any(checkpoint_dir.glob("*.safetensors")),
        "delta_tensor_count": int(len(delta_index.tensor_info)),
        "writer_command": writer_command(base_dir, delta_path, checkpoint_dir),
        "writer_manifest": {
            "floating_tensors": manifest.get("floating_tensors"),
            "frozen_tensors": manifest.get("frozen_tensors"),
            "tensor_delta_safetensors_tensors": manifest.get("tensor_delta_safetensors_tensors"),
            "tensor_delta_safetensors_entries": manifest.get("tensor_delta_safetensors_entries"),
            "tensor_delta_safetensors_values": manifest.get("tensor_delta_safetensors_values"),
            "merge_manifest": rel(checkpoint_dir / "merge_manifest.json"),
        },
        **{key: value for key, value in checks.items() if key != "checks"},
        "recommended_action": "run_matched_vllm_eval_for_harc_router_candidate"
        if all_passed
        else "inspect_materialization_checks_before_vllm_eval",
    }
    return write_outputs(output_dir, checkpoint_dir, summary)


def build_smoke_inputs(output_dir: Path, seed: int) -> tuple[Path, Path]:
    generator = torch.Generator().manual_seed(seed)
    mock_dir = output_dir / "mock_inputs"
    base_dir = mock_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    router0 = 0.2 * torch.randn(4, 6, generator=generator)
    router1 = 0.2 * torch.randn(4, 6, generator=generator)
    control = torch.randn(6, 6, generator=generator)
    base_tensors = {
        "model.layers.0.mlp.gate.weight": router0,
        "model.layers.1.mlp.gate.weight": router1,
        "model.layers.0.self_attn.q_proj.weight": control,
    }
    delta_tensors = {
        "model.layers.0.mlp.gate.weight": 0.03 * torch.randn(4, 6, generator=generator),
        "model.layers.1.mlp.gate.weight": 0.03 * torch.randn(4, 6, generator=generator),
    }
    delta_tensors["model.layers.0.mlp.gate.weight"][0, 0] += 0.05
    delta_tensors["model.layers.1.mlp.gate.weight"][1, 1] -= 0.04
    save_file(base_tensors, str(base_dir / "model.safetensors"), metadata={"format": "pt"})
    delta_path = mock_dir / "router_delta.safetensors"
    save_file(delta_tensors, str(delta_path), metadata={"format": "pt"})
    return base_dir, delta_path


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint_with_harc_router"
    base_dir, delta_path = build_smoke_inputs(output_dir, args.seed)
    manifest = materialize_checkpoint(base_dir, delta_path, checkpoint_dir)
    checks = verify_materialization(
        base_dir=base_dir,
        checkpoint_dir=checkpoint_dir,
        delta_path=delta_path,
        output_dir=output_dir,
    )
    all_passed = checks["passed_check_count"] == checks["check_count"] and checks["check_count"] > 0
    summary = {
        "schema_version": 1,
        "status": "harc_router_candidate_materialized" if all_passed else "harc_router_candidate_check_failed",
        "smoke_status": "smoke_passed" if all_passed else "smoke_failed",
        "solver_summary": None,
        "solver_status": "smoke_ready",
        "base_path": rel(base_dir),
        "base_exists": True,
        "router_delta_safetensors": rel(delta_path),
        "router_delta_exists": True,
        "candidate_checkpoint_dir": rel(checkpoint_dir),
        "candidate_checkpoint_exists": any(checkpoint_dir.glob("*.safetensors")),
        "delta_tensor_count": 2,
        "writer_command": writer_command(base_dir, delta_path, checkpoint_dir),
        "writer_manifest": {
            "floating_tensors": manifest.get("floating_tensors"),
            "frozen_tensors": manifest.get("frozen_tensors"),
            "tensor_delta_safetensors_tensors": manifest.get("tensor_delta_safetensors_tensors"),
            "tensor_delta_safetensors_entries": manifest.get("tensor_delta_safetensors_entries"),
            "tensor_delta_safetensors_values": manifest.get("tensor_delta_safetensors_values"),
            "merge_manifest": rel(checkpoint_dir / "merge_manifest.json"),
        },
        **{key: value for key, value in checks.items() if key != "checks"},
        "recommended_action": "smoke_passed_materialization_contract_ready"
        if all_passed
        else "fix_harc_router_candidate_materialization_contract",
    }
    write_outputs(output_dir, checkpoint_dir, summary)
    if not all_passed:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a same-shape HARC router candidate checkpoint.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("results/checkpoints/qwen3_moe_unified_mechanism_candidate"),
        help="Base same-shape checkpoint that receives the HARC router delta.",
    )
    parser.add_argument(
        "--solver-dir",
        type=Path,
        default=Path("results/qwen3_moe_harc_router_solver"),
        help="Directory containing the HARC router solver outputs.",
    )
    parser.add_argument(
        "--solver-summary",
        type=Path,
        default=Path("results/qwen3_moe_harc_router_solver/summary.json"),
        help="HARC solver summary.json.",
    )
    parser.add_argument("--router-delta", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/checkpoints/qwen3_moe_harc_router_candidate"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_harc_router_candidate"))
    parser.add_argument("--smoke-matrix", action="store_true")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(args) if args.smoke_matrix else build_candidate(args)
    print(f"Wrote Qwen3 MoE HARC router candidate gate to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; checks={summary.get('passed_check_count')}/{summary.get('check_count')}; "
        f"checkpoint={summary.get('candidate_checkpoint_exists')}"
    )


if __name__ == "__main__":
    main()
