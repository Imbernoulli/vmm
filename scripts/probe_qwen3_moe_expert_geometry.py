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
from safetensors import safe_open
from tqdm import tqdm

from write_same_shape_average_checkpoint import discover_safetensors


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
DEFAULT_INSTRUCT = Path(
    "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/"
    "snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)
DEFAULT_CODER = Path(
    "/srv/home/bohanlyu/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/"
    "snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
)
EXPERT_TENSOR_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
PROJECTION_ORDER = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}


LITERATURE_PRIORS = [
    {
        "key": "git_re_basin",
        "url": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Permutation symmetry makes same-name averaging unsafe unless hidden units or experts are aligned.",
    },
    {
        "key": "model_soups",
        "url": "https://arxiv.org/abs/2203.05482",
        "mechanism": "Weight averaging is most plausible inside a shared basin; geometry probes gate the averaging radius.",
    },
    {
        "key": "mergeme",
        "url": "https://arxiv.org/abs/2502.00997",
        "mechanism": "MoE merging needs interference mitigation and routing heuristics beyond unweighted expert averaging.",
    },
    {
        "key": "harc_routing_breakdown",
        "url": "https://arxiv.org/abs/2606.03391",
        "mechanism": "Router/top-k perturbations can break MoE dispatch; expert edits need a separate router gate.",
    },
    {
        "key": "router_kd_calibration",
        "url": "https://arxiv.org/abs/2603.02217",
        "mechanism": "Post-merge degradation often comes from router-expert mismatch, motivating router-only calibration.",
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


def fmt(value: Any, digits: int = 4) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_expert_tensor(name: str) -> tuple[int, int, str] | None:
    match = EXPERT_TENSOR_RE.search(name)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("expert")), match.group("projection")


def shape_numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def robust01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.quantile(0.05))
    hi = float(values.quantile(0.95))
    if hi <= lo + EPS:
        return pd.Series(0.0, index=values.index)
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def top_fraction(values: torch.Tensor, fraction: float) -> float:
    if values.numel() == 0:
        return 0.0
    total = float(values.sum().item())
    if total <= EPS:
        return 0.0
    k = max(1, int(math.ceil(values.numel() * fraction)))
    return float(torch.topk(values, k=k).values.sum().item()) / total


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, q).item())


def chunk_bounds(size: int, chunks: int) -> list[tuple[int, int]]:
    chunks = max(1, min(chunks, size))
    return [
        (int(round(idx * size / chunks)), int(round((idx + 1) * size / chunks)))
        for idx in range(chunks)
    ]


def channel_axis_for_projection(projection: str) -> int:
    return 1 if projection == "down_proj" else 0


def tensor_geometry(
    *,
    name: str,
    layer_id: int,
    expert_id: int,
    projection: str,
    instruct_tensor: torch.Tensor,
    coder_tensor: torch.Tensor,
    chunks: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    instruct = instruct_tensor.to(torch.float32)
    coder = coder_tensor.to(torch.float32)
    delta = coder - instruct

    instruct_norm2 = float(torch.sum(instruct * instruct).item())
    coder_norm2 = float(torch.sum(coder * coder).item())
    delta_norm2 = float(torch.sum(delta * delta).item())
    dot = float(torch.sum(instruct * coder).item())
    instruct_norm = math.sqrt(max(0.0, instruct_norm2))
    coder_norm = math.sqrt(max(0.0, coder_norm2))
    delta_norm = math.sqrt(max(0.0, delta_norm2))
    cosine = dot / max(EPS, instruct_norm * coder_norm)
    relative_delta = delta_norm / max(EPS, instruct_norm)
    midpoint_norm = torch.linalg.vector_norm((instruct + coder) * 0.5).item()
    midpoint_relative_move = 0.5 * delta_norm / max(EPS, float(midpoint_norm))

    axis = channel_axis_for_projection(projection)
    channel_count = int(instruct.shape[axis])
    moved_instruct = instruct.movedim(axis, 0).reshape(channel_count, -1)
    moved_coder = coder.movedim(axis, 0).reshape(channel_count, -1)
    moved_delta = delta.movedim(axis, 0).reshape(channel_count, -1)
    channel_instruct_norm2 = torch.sum(moved_instruct * moved_instruct, dim=1)
    channel_coder_norm2 = torch.sum(moved_coder * moved_coder, dim=1)
    channel_delta_norm2 = torch.sum(moved_delta * moved_delta, dim=1)
    channel_dot = torch.sum(moved_instruct * moved_coder, dim=1)
    channel_rel = torch.sqrt(channel_delta_norm2) / torch.sqrt(channel_instruct_norm2.clamp_min(EPS))
    channel_cos = channel_dot / torch.sqrt((channel_instruct_norm2 * channel_coder_norm2).clamp_min(EPS))

    chunk_rows: list[dict[str, Any]] = []
    chunk_relative_deltas = []
    chunk_cosines = []
    chunk_delta_shares = []
    for chunk_id, (start, end) in enumerate(chunk_bounds(channel_count, chunks)):
        if end <= start:
            continue
        c_instruct_norm2 = float(channel_instruct_norm2[start:end].sum().item())
        c_coder_norm2 = float(channel_coder_norm2[start:end].sum().item())
        c_delta_norm2 = float(channel_delta_norm2[start:end].sum().item())
        c_dot = float(channel_dot[start:end].sum().item())
        c_instruct_norm = math.sqrt(max(0.0, c_instruct_norm2))
        c_coder_norm = math.sqrt(max(0.0, c_coder_norm2))
        c_delta_norm = math.sqrt(max(0.0, c_delta_norm2))
        c_relative_delta = c_delta_norm / max(EPS, c_instruct_norm)
        c_cosine = c_dot / max(EPS, c_instruct_norm * c_coder_norm)
        c_delta_share = c_delta_norm2 / max(EPS, delta_norm2)
        chunk_relative_deltas.append(c_relative_delta)
        chunk_cosines.append(c_cosine)
        chunk_delta_shares.append(c_delta_share)
        chunk_rows.append(
            {
                "tensor": name,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "projection": projection,
                "channel_axis": axis,
                "chunk_id": chunk_id,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_size": end - start,
                "chunk_instruct_norm": c_instruct_norm,
                "chunk_delta_norm": c_delta_norm,
                "chunk_relative_delta": c_relative_delta,
                "chunk_cosine": c_cosine,
                "chunk_delta_energy_share": c_delta_share,
                "chunk_relative_delta_excess": c_relative_delta - relative_delta,
            }
        )

    projection_row = {
        "tensor": name,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "projection": projection,
        "shape": "x".join(str(dim) for dim in instruct.shape),
        "numel": shape_numel(tuple(instruct.shape)),
        "channel_axis": axis,
        "channel_count": channel_count,
        "instruct_norm2": instruct_norm2,
        "coder_norm2": coder_norm2,
        "delta_norm2": delta_norm2,
        "dot": dot,
        "instruct_norm": instruct_norm,
        "coder_norm": coder_norm,
        "delta_norm": delta_norm,
        "cosine": cosine,
        "one_minus_cosine": 1.0 - cosine,
        "relative_delta": relative_delta,
        "midpoint_relative_move": midpoint_relative_move,
        "channel_relative_delta_max": float(channel_rel.max().item()),
        "channel_relative_delta_p95": quantile(channel_rel, 0.95),
        "channel_cosine_min": float(channel_cos.min().item()),
        "channel_cosine_p05": quantile(channel_cos, 0.05),
        "top_10pct_channel_delta_energy_fraction": top_fraction(channel_delta_norm2, 0.10),
        "top_25pct_channel_delta_energy_fraction": top_fraction(channel_delta_norm2, 0.25),
        "chunk_relative_delta_max": max(chunk_relative_deltas) if chunk_relative_deltas else 0.0,
        "chunk_relative_delta_p95": float(pd.Series(chunk_relative_deltas).quantile(0.95))
        if chunk_relative_deltas
        else 0.0,
        "chunk_cosine_min": min(chunk_cosines) if chunk_cosines else 0.0,
        "chunk_delta_energy_max_share": max(chunk_delta_shares) if chunk_delta_shares else 0.0,
    }
    return projection_row, chunk_rows


def load_route_context(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["layer_id", "expert_id"])
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["layer_id", "expert_id"])
    keep = [
        "layer_id",
        "expert_id",
        "total_topk_fraction",
        "dominant_source",
        "dominant_weight",
        "weight_instruct",
        "weight_coder",
        "route_mass_instruct",
        "route_mass_coder",
        "audit_max_relative_delta_norm",
        "audit_mean_relative_delta_norm",
        "audit_delta_norm",
        "max_topk_over_uniform",
        "mean_topk_over_uniform",
        "max_topk_fraction",
        "mean_topk_fraction",
        "dominant_category",
        "max_dominant_category_share",
        "mean_dominant_category_share",
        "min_topk_jaccard",
        "mean_topk_jaccard",
        "min_top1_agreement",
        "mean_top1_agreement",
        "max_router_risk_score",
        "mean_router_risk_score",
        "trust_delta_scale",
        "trust_risk_flags",
        "expected_max_relative_delta_norm",
        "effective_nonbase_weight",
    ]
    present = [column for column in keep if column in frame.columns]
    return frame[present].copy()


def build_expert_geometry(projections: pd.DataFrame, route_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer, expert), group in projections.groupby(["layer_id", "expert_id"], sort=True):
        instruct_norm2 = float(group["instruct_norm2"].sum())
        coder_norm2 = float(group["coder_norm2"].sum())
        delta_norm2 = float(group["delta_norm2"].sum())
        dot = float(group["dot"].sum())
        combined_relative_delta = math.sqrt(delta_norm2) / max(EPS, math.sqrt(instruct_norm2))
        combined_cosine = dot / max(EPS, math.sqrt(instruct_norm2 * coder_norm2))
        rows.append(
            {
                "layer_id": int(layer),
                "expert_id": int(expert),
                "projection_count": int(len(group)),
                "instruct_norm2": instruct_norm2,
                "coder_norm2": coder_norm2,
                "delta_norm2": delta_norm2,
                "combined_relative_delta": combined_relative_delta,
                "combined_cosine": combined_cosine,
                "combined_one_minus_cosine": 1.0 - combined_cosine,
                "max_projection_relative_delta": float(group["relative_delta"].max()),
                "mean_projection_relative_delta": float(group["relative_delta"].mean()),
                "min_projection_cosine": float(group["cosine"].min()),
                "mean_projection_cosine": float(group["cosine"].mean()),
                "max_channel_relative_delta": float(group["channel_relative_delta_max"].max()),
                "max_chunk_relative_delta": float(group["chunk_relative_delta_max"].max()),
                "max_chunk_delta_energy_share": float(group["chunk_delta_energy_max_share"].max()),
                "max_top_10pct_channel_delta_energy_fraction": float(
                    group["top_10pct_channel_delta_energy_fraction"].max()
                ),
            }
        )
    experts = pd.DataFrame(rows)
    if experts.empty:
        return experts
    experts["geometry_delta_pressure"] = robust01(experts["combined_relative_delta"])
    experts["geometry_angle_pressure"] = robust01(experts["combined_one_minus_cosine"])
    experts["geometry_chunk_pressure"] = robust01(experts["max_chunk_relative_delta"])
    experts["geometry_concentration_pressure"] = robust01(
        experts["max_top_10pct_channel_delta_energy_fraction"]
    )
    experts["internal_geometry_risk_score"] = (
        0.40 * experts["geometry_delta_pressure"]
        + 0.25 * experts["geometry_angle_pressure"]
        + 0.20 * experts["geometry_chunk_pressure"]
        + 0.15 * experts["geometry_concentration_pressure"]
    ).clip(0.0, 1.0)
    experts = experts.merge(route_context, on=["layer_id", "expert_id"], how="left")
    for column in [
        "total_topk_fraction",
        "max_topk_over_uniform",
        "min_topk_jaccard",
        "min_top1_agreement",
        "max_router_risk_score",
    ]:
        if column not in experts:
            experts[column] = 0.0
    experts["route_mass_pressure"] = robust01(experts["total_topk_fraction"].fillna(0.0))
    experts["load_pressure"] = robust01(experts["max_topk_over_uniform"].fillna(0.0))
    experts["router_instability_pressure"] = (
        robust01(1.0 - experts["min_topk_jaccard"].fillna(1.0))
        + robust01(1.0 - experts["min_top1_agreement"].fillna(1.0))
        + robust01(experts["max_router_risk_score"].fillna(0.0))
    ) / 3.0
    experts["route_geometry_risk_score"] = (
        0.50 * experts["internal_geometry_risk_score"]
        + 0.25 * experts["route_mass_pressure"]
        + 0.15 * experts["router_instability_pressure"]
        + 0.10 * experts["load_pressure"]
    ).clip(0.0, 1.0)
    experts["geometry_action"] = "identity_average_geometry_ok"
    experts.loc[
        experts["internal_geometry_risk_score"] >= 0.75,
        "geometry_action",
    ] = "cap_expert_delta_or_reduce_nonbase_weight"
    experts.loc[
        experts["route_geometry_risk_score"] >= 0.75,
        "geometry_action",
    ] = "route_important_geometry_risk_use_layer_chunk_or_lower_cap"
    return experts.sort_values(
        ["route_geometry_risk_score", "internal_geometry_risk_score", "combined_relative_delta"],
        ascending=[False, False, False],
    )


def build_layer_geometry(experts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, group in experts.groupby("layer_id", sort=True):
        route_mass = pd.to_numeric(group.get("total_topk_fraction", 0.0), errors="coerce").fillna(0.0)
        route_mass_sum = float(route_mass.sum())
        internal = pd.to_numeric(group["internal_geometry_risk_score"], errors="coerce").fillna(0.0)
        route_geometry = pd.to_numeric(group["route_geometry_risk_score"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "layer_id": int(layer),
                "expert_count": int(len(group)),
                "route_observed_expert_count": int((route_mass > 0.0).sum()),
                "mean_combined_relative_delta": float(group["combined_relative_delta"].mean()),
                "p95_combined_relative_delta": float(group["combined_relative_delta"].quantile(0.95)),
                "max_combined_relative_delta": float(group["combined_relative_delta"].max()),
                "mean_combined_cosine": float(group["combined_cosine"].mean()),
                "min_combined_cosine": float(group["combined_cosine"].min()),
                "mean_internal_geometry_risk_score": float(internal.mean()),
                "max_internal_geometry_risk_score": float(internal.max()),
                "mean_route_geometry_risk_score": float(route_geometry.mean()),
                "max_route_geometry_risk_score": float(route_geometry.max()),
                "route_mass_weighted_internal_geometry_risk_score": float(
                    (internal * route_mass).sum() / max(EPS, route_mass_sum)
                ),
                "route_mass_weighted_route_geometry_risk_score": float(
                    (route_geometry * route_mass).sum() / max(EPS, route_mass_sum)
                ),
                "high_internal_geometry_risk_experts": int((internal >= 0.75).sum()),
                "high_route_geometry_risk_experts": int((route_geometry >= 0.75).sum()),
                "route_mass_sum": route_mass_sum,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "route_mass_weighted_route_geometry_risk_score",
            "high_route_geometry_risk_experts",
            "max_combined_relative_delta",
        ],
        ascending=[False, False, False],
    )


def projection_summary(projections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for projection, group in projections.groupby("projection", sort=True):
        rows.append(
            {
                "projection": projection,
                "tensor_count": int(len(group)),
                "mean_relative_delta": float(group["relative_delta"].mean()),
                "p95_relative_delta": float(group["relative_delta"].quantile(0.95)),
                "max_relative_delta": float(group["relative_delta"].max()),
                "mean_cosine": float(group["cosine"].mean()),
                "min_cosine": float(group["cosine"].min()),
                "max_channel_relative_delta": float(group["channel_relative_delta_max"].max()),
                "max_chunk_relative_delta": float(group["chunk_relative_delta_max"].max()),
                "max_top_10pct_channel_delta_energy_fraction": float(
                    group["top_10pct_channel_delta_energy_fraction"].max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("max_relative_delta", ascending=False)


def discover_common_expert_tensors(args: argparse.Namespace) -> tuple[Any, Any, list[str]]:
    instruct = discover_safetensors(args.instruct)
    coder = discover_safetensors(args.coder)
    common = sorted(set(instruct.tensor_info) & set(coder.tensor_info))
    names = []
    selected_layers = None if args.layers == "all" else {int(item) for item in args.layers.split(",") if item}
    for name in common:
        parsed = parse_expert_tensor(name)
        if parsed is None:
            continue
        layer, expert, projection = parsed
        if selected_layers is not None and layer not in selected_layers:
            continue
        if args.max_experts_per_layer is not None and expert >= args.max_experts_per_layer:
            continue
        if instruct.tensor_info[name].shape != coder.tensor_info[name].shape:
            continue
        names.append(name)
    names.sort(key=lambda item: (*parse_expert_tensor(item)[:2], PROJECTION_ORDER[parse_expert_tensor(item)[2]]))
    if args.max_tensors is not None:
        names = names[: args.max_tensors]
    return instruct, coder, names


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instruct, coder, names = discover_common_expert_tensors(args)
    route_context = load_route_context(repo_path(args.route_context))

    shard_pairs: dict[tuple[Path, Path], list[str]] = {}
    for name in names:
        shard_pairs.setdefault((instruct.tensor_info[name].shard, coder.tensor_info[name].shard), []).append(name)

    projection_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for (instruct_shard, coder_shard), shard_names in tqdm(
        sorted(shard_pairs.items()),
        desc="expert geometry shard pairs",
    ):
        with torch.no_grad():
            with safe_open(str(instruct_shard), framework="pt", device="cpu") as instruct_handle, safe_open(
                str(coder_shard), framework="pt", device="cpu"
            ) as coder_handle:
                for name in shard_names:
                    parsed = parse_expert_tensor(name)
                    if parsed is None:
                        continue
                    layer, expert, projection = parsed
                    projection_row, tensor_chunk_rows = tensor_geometry(
                        name=name,
                        layer_id=layer,
                        expert_id=expert,
                        projection=projection,
                        instruct_tensor=instruct_handle.get_tensor(name),
                        coder_tensor=coder_handle.get_tensor(name),
                        chunks=args.chunks,
                    )
                    projection_rows.append(projection_row)
                    chunk_rows.extend(tensor_chunk_rows)

    projections = pd.DataFrame(projection_rows).sort_values(["layer_id", "expert_id", "projection"])
    chunks = pd.DataFrame(chunk_rows)
    experts = build_expert_geometry(projections, route_context)
    layers = build_layer_geometry(experts)
    proj_summary = projection_summary(projections)
    top_chunks = chunks.sort_values(
        ["chunk_relative_delta", "chunk_delta_energy_share"],
        ascending=[False, False],
    ).head(args.top_chunks)

    projection_path = output_dir / "projection_geometry.csv"
    expert_path = output_dir / "expert_geometry.csv"
    layer_path = output_dir / "layer_geometry.csv"
    projection_summary_path = output_dir / "projection_summary.csv"
    top_chunks_path = output_dir / "top_chunk_geometry.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    projections.to_csv(projection_path, index=False)
    experts.to_csv(expert_path, index=False)
    layers.to_csv(layer_path, index=False)
    proj_summary.to_csv(projection_summary_path, index=False)
    top_chunks.to_csv(top_chunks_path, index=False)

    top_layer = layers.iloc[0].to_dict() if not layers.empty else {}
    top_expert = experts.iloc[0].to_dict() if not experts.empty else {}
    summary = {
        "schema_version": 1,
        "status": "passed" if len(projections) > 0 else "no_expert_tensors",
        "instruct": str(args.instruct),
        "coder": str(args.coder),
        "route_context": rel(args.route_context),
        "projection_tensor_count": int(len(projections)),
        "expert_count": int(len(experts)),
        "layer_count": int(layers["layer_id"].nunique()) if not layers.empty else 0,
        "chunks_per_projection": int(args.chunks),
        "top_chunk_rows": int(len(top_chunks)),
        "mean_projection_cosine": float(projections["cosine"].mean()) if not projections.empty else None,
        "p05_projection_cosine": float(projections["cosine"].quantile(0.05)) if not projections.empty else None,
        "mean_projection_relative_delta": float(projections["relative_delta"].mean()) if not projections.empty else None,
        "p95_projection_relative_delta": float(projections["relative_delta"].quantile(0.95)) if not projections.empty else None,
        "max_projection_relative_delta": float(projections["relative_delta"].max()) if not projections.empty else None,
        "mean_expert_combined_relative_delta": float(experts["combined_relative_delta"].mean())
        if not experts.empty
        else None,
        "p95_expert_combined_relative_delta": float(experts["combined_relative_delta"].quantile(0.95))
        if not experts.empty
        else None,
        "max_expert_combined_relative_delta": float(experts["combined_relative_delta"].max())
        if not experts.empty
        else None,
        "mean_expert_combined_cosine": float(experts["combined_cosine"].mean()) if not experts.empty else None,
        "min_expert_combined_cosine": float(experts["combined_cosine"].min()) if not experts.empty else None,
        "high_internal_geometry_risk_expert_count": int((experts["internal_geometry_risk_score"] >= 0.75).sum())
        if not experts.empty
        else 0,
        "high_route_geometry_risk_expert_count": int((experts["route_geometry_risk_score"] >= 0.75).sum())
        if not experts.empty
        else 0,
        "route_observed_expert_count": int((experts.get("total_topk_fraction", pd.Series()).fillna(0.0) > 0).sum())
        if not experts.empty
        else 0,
        "top_layer_by_route_geometry_risk": None if not top_layer else int(top_layer["layer_id"]),
        "top_layer_route_mass_weighted_route_geometry_risk": None
        if not top_layer
        else float(top_layer["route_mass_weighted_route_geometry_risk_score"]),
        "top_expert_layer": None if not top_expert else int(top_expert["layer_id"]),
        "top_expert_id": None if not top_expert else int(top_expert["expert_id"]),
        "top_expert_route_geometry_risk_score": None
        if not top_expert
        else float(top_expert["route_geometry_risk_score"]),
        "top_expert_combined_relative_delta": None
        if not top_expert
        else float(top_expert["combined_relative_delta"]),
        "recommended_unified_action": (
            "keep_router_frozen_and_use_geometry_route_risk_as_layer_chunk_or_cap_weight"
        ),
        "literature_priors": LITERATURE_PRIORS,
        "outputs": {
            "report": rel(report_path),
            "summary": rel(summary_path),
            "projection_geometry": rel(projection_path),
            "expert_geometry": rel(expert_path),
            "layer_geometry": rel(layer_path),
            "projection_summary": rel(projection_summary_path),
            "top_chunk_geometry": rel(top_chunks_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, proj_summary, layers, experts, top_chunks), encoding="utf-8")
    return summary


def build_report(
    summary: dict[str, Any],
    proj_summary: pd.DataFrame,
    layers: pd.DataFrame,
    experts: pd.DataFrame,
    top_chunks: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Expert Geometry Probe",
        "",
        "这个 probe 直接读取 Qwen3-30B-A3B Instruct 和 Coder 的 routed expert 权重，按 `gate_proj/up_proj/down_proj`、expert、layer 和 channel chunk 计算参数几何。它回答的不是哪个算法名最好，而是为什么某些 expert average 需要更保守：identity 可以成立，但专家内部方向、delta tail 和 route mass 可能仍然冲突。",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Projection tensors: `{summary['projection_tensor_count']}`",
        f"- Experts: `{summary['expert_count']}` across `{summary['layer_count']}` layers",
        f"- Mean / p05 projection cosine: `{fmt(summary['mean_projection_cosine'])}` / `{fmt(summary['p05_projection_cosine'])}`",
        f"- Mean / p95 projection relative delta: `{fmt(summary['mean_projection_relative_delta'])}` / `{fmt(summary['p95_projection_relative_delta'])}`",
        f"- High internal-geometry-risk experts: `{summary['high_internal_geometry_risk_expert_count']}`",
        f"- High route+geometry-risk experts: `{summary['high_route_geometry_risk_expert_count']}`",
        f"- Top route-geometry layer: `{summary['top_layer_by_route_geometry_risk']}`",
        f"- Top route-geometry expert: layer `{summary['top_expert_layer']}`, expert `{summary['top_expert_id']}`",
        "",
        "## Why This Matters",
        "",
        "MoE average 有两个不同层面的风险。第一，expert identity/gauge 要先对齐；否则同名 expert 不代表同一个函数。第二，即使 identity 通过，某些 expert 的 `gate/up/down` 内部参数方向和 channel chunk delta 仍然比其他 expert 更尖锐；如果这些 expert 又有高 route mass 或高 router fragility，就不能用同一个全局平均系数处理。",
        "",
        "因此新的 unified rule 应把 `route_geometry_risk_score` 当作 cap-law 或 layer/chunk 系数的输入：router 仍冻结，expert identity 仍保持 same-shape，同一结构内只调整每层/每专家的非 base 权重或 delta cap。",
        "",
        "## Projection Summary",
        "",
        "| projection | tensors | mean rel-delta | p95 rel-delta | max rel-delta | mean cosine | min cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in proj_summary.iterrows():
        lines.append(
            f"| `{row['projection']}` | {int(row['tensor_count'])} | "
            f"{fmt(row['mean_relative_delta'])} | {fmt(row['p95_relative_delta'])} | "
            f"{fmt(row['max_relative_delta'])} | {fmt(row['mean_cosine'])} | {fmt(row['min_cosine'])} |"
        )
    lines.extend(
        [
            "",
            "## Highest-Risk Layers",
            "",
            "| layer | route-weighted risk | max risk | p95 rel-delta | min cosine | high-risk experts |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in layers.head(12).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | "
            f"{fmt(row['route_mass_weighted_route_geometry_risk_score'])} | "
            f"{fmt(row['max_route_geometry_risk_score'])} | "
            f"{fmt(row['p95_combined_relative_delta'])} | "
            f"{fmt(row['min_combined_cosine'])} | "
            f"{int(row['high_route_geometry_risk_experts'])} |"
        )
    lines.extend(
        [
            "",
            "## Highest-Risk Experts",
            "",
            "| layer | expert | action | route+geometry risk | internal risk | rel-delta | cosine | route mass |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in experts.head(16).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['geometry_action']}` | "
            f"{fmt(row['route_geometry_risk_score'])} | "
            f"{fmt(row['internal_geometry_risk_score'])} | "
            f"{fmt(row['combined_relative_delta'])} | "
            f"{fmt(row['combined_cosine'])} | "
            f"{fmt(row.get('total_topk_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "## Top Channel Chunks",
            "",
            "| layer | expert | projection | chunk | rel-delta | cosine | delta share |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in top_chunks.head(16).iterrows():
        lines.append(
            f"| {int(row['layer_id'])} | {int(row['expert_id'])} | `{row['projection']}` | "
            f"{int(row['chunk_id'])} | {fmt(row['chunk_relative_delta'])} | "
            f"{fmt(row['chunk_cosine'])} | {fmt(row['chunk_delta_energy_share'])} |"
        )
    lines.extend(
        [
            "",
            "## Literature Priors",
            "",
        ]
    )
    for item in LITERATURE_PRIORS:
        lines.append(f"- `{item['key']}`: {item['url']}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe internal expert geometry for Qwen3 MoE source models.")
    parser.add_argument("--instruct", type=Path, default=DEFAULT_INSTRUCT)
    parser.add_argument("--coder", type=Path, default=DEFAULT_CODER)
    parser.add_argument(
        "--route-context",
        type=Path,
        default=Path("results/qwen3_moe_trust_region_candidate/trust_region_source_weights_by_expert.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_expert_geometry_probe"))
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--top-chunks", type=int, default=5000)
    parser.add_argument("--layers", default="all", help="Comma-separated layer ids, or 'all'.")
    parser.add_argument("--max-experts-per-layer", type=int, default=None)
    parser.add_argument("--max-tensors", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    summary = run_probe(parse_args())
    print(
        "Wrote Qwen3 MoE expert geometry probe: "
        f"experts={summary['expert_count']}, "
        f"high_route_geometry={summary['high_route_geometry_risk_expert_count']}, "
        f"top_layer={summary['top_layer_by_route_geometry_risk']}"
    )


if __name__ == "__main__":
    main()
