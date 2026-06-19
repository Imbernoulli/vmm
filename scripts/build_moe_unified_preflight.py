#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from inspect_checkpoint_topology import (
    config_summary,
    find_config,
    load_config,
    read_headers,
    rel,
    repo_path,
    summarize_headers,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FIELDS = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
]


def parse_model_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = repo_path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    return name.strip(), repo_path(path.strip())


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def shape_key(value: Any) -> str:
    return "" if value is None else str(value)


def read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    full = repo_path(path)
    if not full.exists() or not full.is_file():
        return None
    return json.loads(full.read_text(encoding="utf-8"))


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def inspect_model(name: str, path: Path, output_dir: Path, *, write_model_tensor_tables: bool) -> dict[str, Any]:
    config = load_config(path)
    cfg = config_summary(config)
    config_path = find_config(path)
    weights_root, headers, has_index = read_headers(path)
    tensors, groups, experts, header_summary = summarize_headers(headers)

    model_slug = safe_name(name)
    if write_model_tensor_tables and not tensors.empty:
        tensors.to_csv(output_dir / f"{model_slug}_tensors.csv", index=False)
        groups.to_csv(output_dir / f"{model_slug}_groups.csv", index=False)
        experts.to_csv(output_dir / f"{model_slug}_experts.csv", index=False)

    router = tensors[tensors["group"] == "router"].copy() if not tensors.empty else pd.DataFrame()
    routed = tensors[tensors["group"] == "routed_expert"].copy() if not tensors.empty else pd.DataFrame()
    shared = tensors[tensors["group"] == "shared_expert"].copy() if not tensors.empty else pd.DataFrame()

    return {
        "name": name,
        "path": str(path),
        "config_path": str(config_path) if config_path else None,
        "weights_root": str(weights_root) if weights_root else None,
        "has_safetensors_index": bool(has_index),
        "config": cfg,
        "headers": header_summary,
        "router_tensors": router,
        "routed_expert_tensors": routed,
        "shared_expert_tensors": shared,
        "artifacts": {
            "tensors": rel(output_dir / f"{model_slug}_tensors.csv")
            if write_model_tensor_tables and not tensors.empty
            else None,
            "groups": rel(output_dir / f"{model_slug}_groups.csv")
            if write_model_tensor_tables and not groups.empty
            else None,
            "experts": rel(output_dir / f"{model_slug}_experts.csv")
            if write_model_tensor_tables and not experts.empty
            else None,
        },
    }


def compare_config(left: dict[str, Any], right: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for field in CONFIG_FIELDS:
        left_value = left["config"].get(field)
        right_value = right["config"].get(field)
        rows.append(
            {
                "field": field,
                "left": left_value,
                "right": right_value,
                "match": left_value == right_value,
            }
        )
    return pd.DataFrame(rows)


def compare_tensor_contract(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    group: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    left_map = {str(row["tensor"]): row for _, row in left.iterrows()}
    right_map = {str(row["tensor"]): row for _, row in right.iterrows()}
    for tensor in sorted(set(left_map) | set(right_map)):
        left_row = left_map.get(tensor)
        right_row = right_map.get(tensor)
        left_shape = clean_value(left_row["shape"]) if left_row is not None else None
        right_shape = clean_value(right_row["shape"]) if right_row is not None else None
        rows.append(
            {
                "group": group,
                "tensor": tensor,
                "layer": clean_value(left_row["layer"]) if left_row is not None else clean_value(right_row["layer"]),
                "left_model": left_name,
                "right_model": right_name,
                "left_shape": left_shape,
                "right_shape": right_shape,
                "shape_match": left_shape == right_shape and left_row is not None and right_row is not None,
                "left_dtype": clean_value(left_row["dtype"]) if left_row is not None else None,
                "right_dtype": clean_value(right_row["dtype"]) if right_row is not None else None,
                "left_shard": clean_value(left_row["shard"]) if left_row is not None else None,
                "right_shard": clean_value(right_row["shard"]) if right_row is not None else None,
            }
        )
    return pd.DataFrame(rows)


def layer_layout(model: dict[str, Any], tensors: pd.DataFrame, group: str) -> pd.DataFrame:
    if tensors.empty:
        return pd.DataFrame(
            columns=["model", "group", "layer", "tensor_count", "shape_signature", "packed_expert_count"]
        )
    rows = []
    for layer, layer_rows in tensors.groupby("layer", dropna=False):
        shapes = sorted({str(value) for value in layer_rows["shape"].dropna().tolist()})
        packed_values = [clean_value(value) for value in layer_rows.get("packed_expert_count", pd.Series()).dropna()]
        rows.append(
            {
                "model": model["name"],
                "group": group,
                "layer": clean_value(layer),
                "tensor_count": int(len(layer_rows)),
                "shape_signature": "|".join(shapes),
                "packed_expert_count": max(packed_values) if packed_values else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "layer"], na_position="last")


def compare_layer_layout(left_layout: pd.DataFrame, right_layout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = ["group", "layer"]
    left_map = {
        (str(row["group"]), str(clean_value(row["layer"]))): row for _, row in left_layout.iterrows()
    }
    right_map = {
        (str(row["group"]), str(clean_value(row["layer"]))): row for _, row in right_layout.iterrows()
    }
    for key in sorted(set(left_map) | set(right_map)):
        left_row = left_map.get(key)
        right_row = right_map.get(key)
        rows.append(
            {
                "group": key[0],
                "layer": key[1],
                "left_tensor_count": clean_value(left_row["tensor_count"]) if left_row is not None else None,
                "right_tensor_count": clean_value(right_row["tensor_count"]) if right_row is not None else None,
                "left_shape_signature": shape_key(clean_value(left_row["shape_signature"])) if left_row is not None else None,
                "right_shape_signature": shape_key(clean_value(right_row["shape_signature"])) if right_row is not None else None,
                "left_packed_expert_count": clean_value(left_row["packed_expert_count"]) if left_row is not None else None,
                "right_packed_expert_count": clean_value(right_row["packed_expert_count"]) if right_row is not None else None,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    left_packed = out["left_packed_expert_count"].map(lambda value: -1 if pd.isna(value) else value)
    right_packed = out["right_packed_expert_count"].map(lambda value: -1 if pd.isna(value) else value)
    out["layout_match"] = (
        (out["left_tensor_count"] == out["right_tensor_count"])
        & (out["left_shape_signature"] == out["right_shape_signature"])
        & (left_packed == right_packed)
    )
    return out


def concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    kept = [frame for frame in frames if not frame.empty]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def qwen_identity_gate(cross: dict[str, Any] | None, expected_layers: int | None) -> dict[str, Any]:
    if not cross:
        return {
            "status": "missing_cross_correspondence",
            "pass": False,
            "reason": "No cross-model expert correspondence file was supplied.",
        }
    identity = maybe_float(cross.get("frac_layers_identity_optimal"))
    argmax = maybe_float(cross.get("mean_argmax_is_identity_frac"))
    layers = int(cross.get("n_layers", 0))
    layer_match = expected_layers is None or layers == expected_layers
    passed = bool((identity or 0.0) >= 0.99 and (argmax or 0.0) >= 0.99 and layer_match)
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "n_layers": layers,
        "expected_layers": expected_layers,
        "frac_layers_identity_optimal": identity,
        "mean_argmax_is_identity_frac": argmax,
        "mean_diag_cos": maybe_float(cross.get("mean_diag_cos")),
        "mean_offdiag_cos": maybe_float(cross.get("mean_offdiag_cos")),
        "reason": "Expert identity is stable enough to use identity slices first."
        if passed
        else "Expert identity is not established; layer-wise remap is required before averaging.",
    }


def build_gate_rows(
    *,
    config_table: pd.DataFrame,
    router_contract: pd.DataFrame,
    expert_contract: pd.DataFrame,
    layout_contract: pd.DataFrame,
    left: dict[str, Any],
    right: dict[str, Any],
    identity_gate: dict[str, Any],
    cuda_available: bool,
) -> list[dict[str, Any]]:
    config_pass = bool(config_table["match"].all()) and bool(left["config"].get("is_moe_config")) and bool(
        right["config"].get("is_moe_config")
    )
    router_pass = bool(not router_contract.empty and router_contract["shape_match"].all())
    expert_pass = bool(not expert_contract.empty and expert_contract["shape_match"].all())
    layout_pass = bool(not layout_contract.empty and layout_contract["layout_match"].all())
    header_pass = bool(left["headers"].get("weights_available") and right["headers"].get("weights_available"))
    return [
        {
            "stage": "same_shape_config",
            "status": "pass" if config_pass else "fail",
            "evidence": "All average-critical config fields match." if config_pass else "Config fields differ.",
            "action": "continue" if config_pass else "do_not_materialize",
        },
        {
            "stage": "router_tensor_contract",
            "status": "pass" if router_pass else "fail",
            "evidence": f"{len(router_contract)} router tensors compared; all shapes match={router_pass}.",
            "action": "allow_router_probe" if router_pass else "fix_router_name_or_target_pair",
        },
        {
            "stage": "routed_expert_layout",
            "status": "pass" if expert_pass and layout_pass else "fail",
            "evidence": f"{len(expert_contract)} routed expert tensors compared; per-layer layout match={layout_pass}.",
            "action": "allow_identity_or_remap_gate" if expert_pass and layout_pass else "do_not_average_experts",
        },
        {
            "stage": "expert_identity",
            "status": "pass" if identity_gate["pass"] else "blocked",
            "evidence": identity_gate["reason"],
            "action": "use_identity_expert_slices_first" if identity_gate["pass"] else "run_layerwise_expert_correspondence",
        },
        {
            "stage": "runtime_route_probe",
            "status": "ready" if cuda_available else "blocked_in_this_process",
            "evidence": f"torch.cuda.is_available={cuda_available}.",
            "action": "run_transformers_or_vllm_route_probe"
            if cuda_available
            else "run the emitted command on a GPU/vLLM host; keep materialization blocked here",
        },
        {
            "stage": "behavior_eval",
            "status": "waiting",
            "evidence": "No same-shape Qwen3 MoE candidate should be published before route overlap, load, and downstream eval.",
            "action": "host candidate with vLLM after route/load gates pass",
        },
    ]


def write_command(
    *,
    output_dir: Path,
    left: dict[str, Any],
    right: dict[str, Any],
    prompt_file: str,
    top_k: int | None,
) -> str:
    k = top_k or left["config"].get("num_experts_per_tok") or 8
    command = (
        "python scripts/probe_moe_routing.py "
        f"--model {left['path']} "
        f"--compare-model {right['path']} "
        f"--tokenizer {left['path']} "
        f"--prompts {prompt_file} "
        "--device-map auto --dtype bfloat16 --use-chat-template --local-files-only "
        f"--top-k {k} "
        "--output-dir results/moe_routing_probe/qwen3_30b_instruct_vs_coder"
    )
    (output_dir / "routing_probe_command.txt").write_text(command + "\n", encoding="utf-8")
    return command


def build_summary(
    *,
    output_dir: Path,
    left: dict[str, Any],
    right: dict[str, Any],
    config_table: pd.DataFrame,
    router_contract: pd.DataFrame,
    expert_contract: pd.DataFrame,
    layout_contract: pd.DataFrame,
    gates: pd.DataFrame,
    identity_gate: dict[str, Any],
    routing_command: str,
    selector_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    pass_count = int((gates["status"] == "pass").sum())
    blocked_count = int(gates["status"].isin(["blocked", "blocked_in_this_process"]).sum())
    config_pass = bool(config_table["match"].all())
    same_shape_ready = bool(
        config_pass
        and not router_contract.empty
        and router_contract["shape_match"].all()
        and not expert_contract.empty
        and expert_contract["shape_match"].all()
        and not layout_contract.empty
        and layout_contract["layout_match"].all()
    )
    status = "ready_for_real_route_probe" if same_shape_ready and identity_gate["pass"] else "blocked_before_route_probe"
    if same_shape_ready and identity_gate["pass"] and blocked_count:
        status = "same_shape_and_identity_ready_route_runtime_blocked_here"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "left_model": left["name"],
        "right_model": right["name"],
        "same_shape_contract_pass": same_shape_ready,
        "expert_identity_gate": identity_gate,
        "cuda_available": bool(torch.cuda.is_available()),
        "router_tensor_rows": int(len(router_contract)),
        "routed_expert_tensor_rows": int(len(expert_contract)),
        "layout_rows": int(len(layout_contract)),
        "gate_pass_count": pass_count,
        "gate_blocked_count": blocked_count,
        "selector_prior_status": None if selector_summary is None else selector_summary.get("status"),
        "selector_next_blocking_probe": None if selector_summary is None else selector_summary.get("next_blocking_probe"),
        "unified_mechanism": {
            "objective": "same-shape model average that can choose endpoint/base, layer/tensor coefficients, expert remap, guarded router, and capacity bias without changing architecture",
            "moe_error_decomposition": "expert gauge/identity + expert functional residual + router route-overlap drift + sparse capacity overflow + downstream behavior",
            "current_qwen3_action": "identity expert slices are allowed by correspondence; router and load remain blocked until real route probe.",
        },
        "routing_probe_command": routing_command,
        "outputs": {
            "config_contract": rel(output_dir / "config_contract.csv"),
            "router_contract": rel(output_dir / "router_contract.csv"),
            "expert_contract": rel(output_dir / "expert_contract.csv"),
            "layout_contract": rel(output_dir / "layout_contract.csv"),
            "gate_table": rel(output_dir / "unified_gate_table.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
            "routing_probe_command": rel(output_dir / "routing_probe_command.txt"),
        },
    }


def build_report(
    *,
    summary: dict[str, Any],
    config_table: pd.DataFrame,
    gates: pd.DataFrame,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str:
    left_cfg = left["config"]
    lines = [
        "# Qwen3 MoE Unified Average Preflight",
        "",
        "这一步不是继续做静态方法排名，而是把 unified average 的必要条件拆成可验证合同：同构 topology、router tensor、packed expert layout、expert identity、真实 route/load，以及最终 vLLM 行为评测。当前脚本只读 config 和 safetensors header，不加载 30B 权重内容。",
        "",
        "## Current Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Pair: `{summary['left_model']}` vs `{summary['right_model']}`",
        f"- Same-shape contract: `{summary['same_shape_contract_pass']}`",
        f"- Expert identity gate: `{summary['expert_identity_gate']['status']}`",
        f"- CUDA available in this process: `{summary['cuda_available']}`",
        f"- Router tensors compared: `{summary['router_tensor_rows']}`",
        f"- Routed expert tensors compared: `{summary['routed_expert_tensor_rows']}`",
        "",
        "## Why This Is The Unified Algorithm Gate",
        "",
        "MoE 的输出可以粗略写成 `shared(x) + sum_e route_e(x; R) * expert_e(x; W_e)`。因此 average 失败不是一个单一原因：同名 expert 可能只是 gauge index，expert 函数可能不在同一个语义坐标，router 的 top-k 边界可能漂移，serving 时还会出现 capacity/load 问题。unified 方法应该先验证这些机制，再决定 expert remap、expert weights、router freeze/small-step/route-KD 和 router-bias capacity correction；最终 checkpoint 仍保持同结构。",
        "",
        "## Model Contract",
        "",
        f"- Model type: `{left_cfg.get('model_type')}`",
        f"- Layers: `{left_cfg.get('num_hidden_layers')}`",
        f"- Experts per layer: `{left_cfg.get('num_experts')}`",
        f"- Active experts per token: `{left_cfg.get('num_experts_per_tok')}`",
        "",
        "| field | left | right | match |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in config_table.iterrows():
        lines.append(f"| `{row['field']}` | `{row['left']}` | `{row['right']}` | `{row['match']}` |")
    lines.extend(["", "## Gate Table", "", "| stage | status | action | evidence |", "| --- | --- | --- | --- |"])
    for _, row in gates.iterrows():
        lines.append(f"| `{row['stage']}` | `{row['status']}` | `{row['action']}` | {row['evidence']} |")
    lines.extend(
        [
            "",
            "## Next Executable Probe",
            "",
            "```bash",
            summary["routing_probe_command"],
            "PYTHONPATH=src python scripts/analyze_moe_routing_readiness.py --router-dir results/moe_routing_probe/qwen3_30b_instruct_vs_coder --output-dir results/moe_routing_readiness/qwen3_30b_instruct_vs_coder",
            "```",
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['config_contract']}`",
            f"- `{summary['outputs']['router_contract']}`",
            f"- `{summary['outputs']['expert_contract']}`",
            f"- `{summary['outputs']['layout_contract']}`",
            f"- `{summary['outputs']['gate_table']}`",
            f"- `{summary['outputs']['routing_probe_command']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a same-shape MoE unified-average preflight for a model pair.")
    parser.add_argument("--model", action="append", required=True, help="Model spec NAME=PATH. Pass exactly two.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_unified_preflight_qwen3_30b"))
    parser.add_argument(
        "--cross-correspondence",
        default="results/fp_moe_real_probe/qwen3_instruct_coder/cross_correspondence.json",
    )
    parser.add_argument("--selector-summary", default="results/moe_probe_gated_selector/summary.json")
    parser.add_argument("--prompt-file", default="prompts/qwen_moe_route_probe_prompts.jsonl")
    parser.add_argument(
        "--write-model-tensor-tables",
        action="store_true",
        help="Also write full per-model tensor/group/expert header dumps. Contract tables are always written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.model) != 2:
        raise SystemExit("--model must be supplied exactly twice: left and right.")
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    left_name, left_path = parse_model_spec(args.model[0])
    right_name, right_path = parse_model_spec(args.model[1])
    left = inspect_model(
        left_name,
        left_path,
        output_dir,
        write_model_tensor_tables=args.write_model_tensor_tables,
    )
    right = inspect_model(
        right_name,
        right_path,
        output_dir,
        write_model_tensor_tables=args.write_model_tensor_tables,
    )

    config_table = compare_config(left, right)
    router_contract = compare_tensor_contract(
        left["router_tensors"],
        right["router_tensors"],
        group="router",
        left_name=left_name,
        right_name=right_name,
    )
    expert_contract = compare_tensor_contract(
        left["routed_expert_tensors"],
        right["routed_expert_tensors"],
        group="routed_expert",
        left_name=left_name,
        right_name=right_name,
    )
    shared_contract = compare_tensor_contract(
        left["shared_expert_tensors"],
        right["shared_expert_tensors"],
        group="shared_expert",
        left_name=left_name,
        right_name=right_name,
    )
    layout_contract = compare_layer_layout(
        concat_nonempty(
            [
                layer_layout(left, left["router_tensors"], "router"),
                layer_layout(left, left["routed_expert_tensors"], "routed_expert"),
                layer_layout(left, left["shared_expert_tensors"], "shared_expert"),
            ],
        ),
        concat_nonempty(
            [
                layer_layout(right, right["router_tensors"], "router"),
                layer_layout(right, right["routed_expert_tensors"], "routed_expert"),
                layer_layout(right, right["shared_expert_tensors"], "shared_expert"),
            ],
        ),
    )
    identity_gate = qwen_identity_gate(
        read_optional_json(args.cross_correspondence),
        expected_layers=left["config"].get("num_hidden_layers"),
    )
    gates = pd.DataFrame(
        build_gate_rows(
            config_table=config_table,
            router_contract=router_contract,
            expert_contract=expert_contract,
            layout_contract=layout_contract,
            left=left,
            right=right,
            identity_gate=identity_gate,
            cuda_available=torch.cuda.is_available(),
        )
    )
    routing_command = write_command(
        output_dir=output_dir,
        left=left,
        right=right,
        prompt_file=args.prompt_file,
        top_k=left["config"].get("num_experts_per_tok"),
    )
    selector_summary = read_optional_json(args.selector_summary)
    summary = build_summary(
        output_dir=output_dir,
        left=left,
        right=right,
        config_table=config_table,
        router_contract=router_contract,
        expert_contract=expert_contract,
        layout_contract=layout_contract,
        gates=gates,
        identity_gate=identity_gate,
        routing_command=routing_command,
        selector_summary=selector_summary,
    )

    config_table.to_csv(output_dir / "config_contract.csv", index=False)
    router_contract.to_csv(output_dir / "router_contract.csv", index=False)
    expert_contract.to_csv(output_dir / "expert_contract.csv", index=False)
    shared_contract.to_csv(output_dir / "shared_expert_contract.csv", index=False)
    layout_contract.to_csv(output_dir / "layout_contract.csv", index=False)
    gates.to_csv(output_dir / "unified_gate_table.csv", index=False)
    compact_models = [
        {
            key: value
            for key, value in model.items()
            if key not in {"router_tensors", "routed_expert_tensors", "shared_expert_tensors"}
        }
        for model in (left, right)
    ]
    (output_dir / "model_contracts.json").write_text(
        json.dumps(jsonable(compact_models), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary=summary, config_table=config_table, gates=gates, left=left, right=right),
        encoding="utf-8",
    )
    print(f"Wrote MoE unified preflight to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
