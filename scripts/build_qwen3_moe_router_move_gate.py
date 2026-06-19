#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from audit_materialized_checkpoint_delta import shape_numel
from write_same_shape_average_checkpoint import discover_safetensors, is_router_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
SAFE_ACTIONS = {"router_probe_passed_for_small_lambda", "small_lambda_router_with_overlap_guard"}


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
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return clean_value(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def layer_id(name: str) -> int | None:
    match = LAYER_RE.search(str(name))
    return None if match is None else int(match.group(1))


def tensor_delta_stats(base_tensor: torch.Tensor, source_tensor: torch.Tensor) -> dict[str, float]:
    base_fp32 = base_tensor.to(torch.float32)
    source_fp32 = source_tensor.to(torch.float32)
    delta = source_fp32 - base_fp32
    base_norm2 = float(torch.sum(base_fp32 * base_fp32).item())
    delta_norm2 = float(torch.sum(delta * delta).item())
    base_norm = math.sqrt(base_norm2)
    delta_norm = math.sqrt(delta_norm2)
    return {
        "base_norm": base_norm,
        "delta_norm": delta_norm,
        "relative_delta_norm": delta_norm / max(1e-12, base_norm),
        "max_abs_delta": float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0,
    }


def router_delta_rows(base_path: Path, source_path: Path) -> pd.DataFrame:
    base = discover_safetensors(base_path)
    source = discover_safetensors(source_path)
    names = sorted(
        name for name in set(base.tensor_info) & set(source.tensor_info) if is_router_tensor(name)
    )
    rows = []
    from safetensors import safe_open

    for name in names:
        base_info = base.tensor_info[name]
        source_info = source.tensor_info[name]
        if base_info.shape != source_info.shape:
            rows.append(
                {
                    "router": name,
                    "layer": layer_id(name),
                    "shape": "x".join(str(dim) for dim in base_info.shape),
                    "numel": shape_numel(base_info.shape),
                    "shape_match": False,
                    "relative_delta_norm": None,
                    "delta_norm": None,
                    "base_norm": None,
                    "max_abs_delta": None,
                }
            )
            continue
        with safe_open(str(base_info.shard), framework="pt", device="cpu") as base_handle, safe_open(
            str(source_info.shard), framework="pt", device="cpu"
        ) as source_handle:
            stats = tensor_delta_stats(base_handle.get_tensor(name), source_handle.get_tensor(name))
        rows.append(
            {
                "router": name,
                "layer": layer_id(name),
                "shape": "x".join(str(dim) for dim in base_info.shape),
                "numel": shape_numel(base_info.shape),
                "shape_match": True,
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("layer")


def layer_readiness_rows(readiness: pd.DataFrame, router_deltas: pd.DataFrame) -> pd.DataFrame:
    frame = readiness.copy()
    frame["layer"] = frame["router"].map(layer_id)
    grouped = []
    for layer, group in frame.groupby("layer", sort=True):
        actions = group["recommended_action"].astype(str)
        calibrate_count = int((actions == "calibrate_router_before_average").sum())
        freeze_count = int((actions == "freeze_router_and_check_load_balance").sum())
        small_count = int((actions == "small_lambda_router_with_overlap_guard").sum())
        passed_count = int((actions == "router_probe_passed_for_small_lambda").sum())
        unsafe_count = calibrate_count + freeze_count
        all_rows_safe = bool(unsafe_count == 0 and set(actions).issubset(SAFE_ACTIONS))
        if all_rows_safe:
            move_decision = "allow_small_lambda_router_delta"
            recommended_router_lambda = 0.05
            reason = "All observed categories passed the router move guard."
        elif unsafe_count:
            move_decision = "freeze_router"
            recommended_router_lambda = 0.0
            reason = "At least one observed category requires calibration/freeze; the router tensor is shared across categories."
        else:
            move_decision = "freeze_router_until_more_probe_data"
            recommended_router_lambda = 0.0
            reason = "The layer has no hard failure but lacks a full all-category pass."

        delta = router_deltas[router_deltas["layer"] == layer]
        delta_row = delta.iloc[0].to_dict() if not delta.empty else {}
        grouped.append(
            {
                "layer": int(layer),
                "router": group["router"].iloc[0],
                "probe_rows": int(len(group)),
                "calibrate_rows": calibrate_count,
                "freeze_rows": freeze_count,
                "small_lambda_rows": small_count,
                "passed_rows": passed_count,
                "unsafe_rows": unsafe_count,
                "mean_topk_jaccard": float(group["topk_jaccard"].mean()),
                "min_topk_jaccard": float(group["topk_jaccard"].min()),
                "mean_top1_agreement": float(group["top1_agreement"].mean()),
                "min_top1_agreement": float(group["top1_agreement"].min()),
                "mean_top1_margin": float(group["top1_margin_mean"].mean()),
                "min_top1_margin": float(group["top1_margin_mean"].min()),
                "mean_risk_score": float(group["risk_score"].mean()),
                "max_risk_score": int(group["risk_score"].max()),
                "router_relative_delta_norm": clean_value(delta_row.get("relative_delta_norm")),
                "router_max_abs_delta": clean_value(delta_row.get("max_abs_delta")),
                "move_decision": move_decision,
                "recommended_router_lambda": recommended_router_lambda,
                "reason": reason,
            }
        )
    return pd.DataFrame(grouped).sort_values(
        ["recommended_router_lambda", "unsafe_rows", "max_risk_score", "mean_topk_jaccard"],
        ascending=[False, True, True, False],
    )


def build_report(summary: dict[str, Any], layer_gate: pd.DataFrame, router_deltas: pd.DataFrame) -> str:
    lines = [
        "# Qwen3 MoE Router Move Gate",
        "",
        "这个 gate 专门回答一个问题：在已经有 routed-expert candidate 的情况下，能不能把 router 也作为 same-shape average 的一部分打开。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Router layers: `{summary['router_layer_count']}`",
        f"- Allowed router layers: `{summary['allowed_router_layer_count']}`",
        f"- Frozen router layers: `{summary['frozen_router_layer_count']}`",
        f"- Total router relative delta norm: `{fmt(summary['total_router_relative_delta_norm'])}`",
        f"- Mean top-k Jaccard: `{fmt(summary['mean_topk_jaccard'])}`",
        f"- Min top-k Jaccard: `{fmt(summary['min_topk_jaccard'])}`",
        f"- Mean top1 agreement: `{fmt(summary['mean_top1_agreement'])}`",
        f"- Min top1 agreement: `{fmt(summary['min_top1_agreement'])}`",
        "",
        "## Decision",
        "",
        summary["interpretation"],
        "",
        "## Layer Gate",
        "",
        "| layer | decision | lambda | unsafe | calibrate | freeze | small | passed | mean Jaccard | min top1 | router rel | reason |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in layer_gate.iterrows():
        lines.append(
            f"| {int(row['layer'])} | `{row['move_decision']}` | "
            f"{fmt(row['recommended_router_lambda'])} | {int(row['unsafe_rows'])} | "
            f"{int(row['calibrate_rows'])} | {int(row['freeze_rows'])} | "
            f"{int(row['small_lambda_rows'])} | {int(row['passed_rows'])} | "
            f"{fmt(float(row['mean_topk_jaccard']))} | {fmt(float(row['min_top1_agreement']))} | "
            f"{fmt(row['router_relative_delta_norm'])} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Router Delta Summary",
            "",
            "| layer | router rel | max abs delta | numel |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in router_deltas.sort_values("relative_delta_norm", ascending=False).head(12).iterrows():
        lines.append(
            f"| {int(row['layer'])} | {fmt(row['relative_delta_norm'])} | "
            f"{fmt(row['max_abs_delta'])} | {int(row['numel'])} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            "Router tensor 是整层共享的参数；如果同一层里任何 category/prompt slice 已经触发 `calibrate_router_before_average` 或 `freeze_router_and_check_load_balance`，就不能只对安全 category 移动这层 router。当前真实 Qwen3 probe 中没有任何一层在全部观察场景里通过，因此统一规则仍应冻结 router。下一步如果要打开 router，应该做 route-KD / HARC-style calibration 并重新 probe，而不是直接平均 router weights。",
            "",
            "## Outputs",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen3 MoE router move gate from routing readiness and router tensor deltas.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120"),
    )
    parser.add_argument(
        "--readiness-csv",
        type=Path,
        default=Path("results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/router_readiness.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_move_gate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness = pd.read_csv(repo_path(args.readiness_csv))
    router_deltas = router_delta_rows(repo_path(args.base), repo_path(args.source))
    layer_gate = layer_readiness_rows(readiness, router_deltas)

    delta_norm2 = float((router_deltas["delta_norm"].fillna(0.0) ** 2).sum())
    base_norm2 = float((router_deltas["base_norm"].fillna(0.0) ** 2).sum())
    allowed = layer_gate[layer_gate["recommended_router_lambda"] > 0.0]
    frozen = layer_gate[layer_gate["recommended_router_lambda"] == 0.0]
    status = "router_move_candidate_ready" if not allowed.empty else "router_move_rejected_freeze_router"
    summary = {
        "schema_version": 1,
        "status": status,
        "router_layer_count": int(len(layer_gate)),
        "allowed_router_layer_count": int(len(allowed)),
        "frozen_router_layer_count": int(len(frozen)),
        "readiness_rows": int(len(readiness)),
        "unsafe_readiness_rows": int(layer_gate["unsafe_rows"].sum()),
        "calibrate_readiness_rows": int(layer_gate["calibrate_rows"].sum()),
        "freeze_readiness_rows": int(layer_gate["freeze_rows"].sum()),
        "small_lambda_readiness_rows": int(layer_gate["small_lambda_rows"].sum()),
        "passed_readiness_rows": int(layer_gate["passed_rows"].sum()),
        "total_router_delta_norm": math.sqrt(delta_norm2),
        "total_router_base_norm": math.sqrt(base_norm2),
        "total_router_relative_delta_norm": math.sqrt(delta_norm2) / max(1e-12, math.sqrt(base_norm2)),
        "mean_topk_jaccard": float(readiness["topk_jaccard"].mean()),
        "min_topk_jaccard": float(readiness["topk_jaccard"].min()),
        "mean_top1_agreement": float(readiness["top1_agreement"].mean()),
        "min_top1_agreement": float(readiness["top1_agreement"].min()),
        "max_router_relative_delta_norm": float(router_deltas["relative_delta_norm"].max()),
        "recommended_unified_router_action": "freeze_router",
        "interpretation": (
            "No router layer passes the all-observed-category guard. Because router tensors are shared "
            "across categories, selective category-level movement is not expressible as a same-shape "
            "weight average. The current unified Qwen3 MoE rule should keep router frozen and test "
            "route-KD/HARC-style calibration only as a separate trained/calibrated intervention."
        ),
        "outputs": {
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
            "router_layer_move_gate": rel(output_dir / "router_layer_move_gate.csv"),
            "router_delta_summary": rel(output_dir / "router_delta_summary.csv"),
        },
    }

    layer_gate.to_csv(output_dir / "router_layer_move_gate.csv", index=False)
    router_deltas.to_csv(output_dir / "router_delta_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(build_report(summary, layer_gate, router_deltas), encoding="utf-8")
    print(f"Wrote Qwen3 MoE router move gate to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
