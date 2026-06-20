#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = [
    {
        "candidate": "route_guarded",
        "method": "qwen3_moe_unified_route_guarded_candidate",
        "audit_dir": "results/qwen3_moe_materialized_delta_audit",
        "rule": "freeze_router_route_conditioned_experts_small_attention",
    },
    {
        "candidate": "audit_gated",
        "method": "qwen3_moe_audit_gated_candidate",
        "audit_dir": "results/qwen3_moe_audit_gated_delta_audit",
        "rule": "route_conditioned_experts_with_delta_cap",
    },
    {
        "candidate": "trust_region",
        "method": "qwen3_moe_trust_region_candidate",
        "audit_dir": "results/qwen3_moe_trust_region_delta_audit",
        "rule": "route_load_category_fragility_trust_region",
    },
    {
        "candidate": "expert_only",
        "method": "qwen3_moe_expert_only_trust_region_candidate",
        "audit_dir": "results/qwen3_moe_expert_only_delta_audit",
        "rule": "trust_region_experts_freeze_attention",
    },
    {
        "candidate": "tail_trimmed",
        "method": "qwen3_moe_tail_trimmed_expert_only_candidate",
        "audit_dir": "results/qwen3_moe_tail_trimmed_delta_audit",
        "rule": "expert_only_second_stage_tail_cap_0_65",
    },
    {
        "candidate": "searched_no_gt065",
        "method": "qwen3_moe_searched_no_gt065_max_retention_candidate",
        "audit_dir": "results/qwen3_moe_searched_no_gt065_delta_audit",
        "rule": "searched_source_route_expert_weights_uniform_cap_0_65",
    },
    {
        "candidate": "layer_chunk",
        "method": "qwen3_moe_layer_chunk_candidate",
        "audit_dir": "results/qwen3_moe_layer_chunk_delta_audit",
        "rule": "importance_guided_layer_chunk_coefficients",
    },
    {
        "candidate": "unified_mechanism",
        "method": "qwen3_moe_unified_mechanism_candidate",
        "audit_dir": "results/qwen3_moe_unified_mechanism_delta_audit",
        "rule": "unified_router_evidence_geometry_risk_cap",
    },
    {
        "candidate": "mechanistic_unified",
        "method": "qwen3_moe_mechanistic_unified_candidate",
        "audit_dir": "results/qwen3_moe_mechanistic_unified_delta_audit",
        "rule": "benefit_curvature_interference_scale_law",
    },
    {
        "candidate": "subspace_scaled",
        "method": "qwen3_moe_subspace_scaled_candidate",
        "audit_dir": "results/qwen3_moe_subspace_scaled_delta_audit",
        "rule": "unified_plus_uncovered_subspace_conflict_shrink",
    },
]
THRESHOLDS = [1.0, 0.75, 0.6505, 0.65, 0.5]
STRUCTURAL_RISK_COLUMNS = [
    "total_relative_delta_norm",
    "routed_relative_delta_norm",
    "routed_max_tensor_relative_delta",
    "routed_p99_tensor_relative_delta",
    "routed_tensors_gt_0_75",
    "routed_tensors_gt_0_65",
    "attention_relative_delta_norm",
    "attention_changed_tensors",
    "router_changed_tensors",
]
STRUCTURAL_DISTANCE_WEIGHTS = {
    "total_relative_delta_norm": 0.20,
    "routed_relative_delta_norm": 0.18,
    "routed_max_tensor_relative_delta": 0.16,
    "routed_p99_tensor_relative_delta": 0.12,
    "routed_tensors_gt_0_75": 0.12,
    "routed_tensors_gt_0_65": 0.12,
    "attention_relative_delta_norm": 0.04,
    "attention_changed_tensors": 0.04,
    "router_changed_tensors": 0.02,
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def clean(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def group_metric(groups: pd.DataFrame, group: str, column: str, default: float = 0.0) -> float:
    rows = groups[groups["group"] == group]
    if rows.empty:
        return default
    return float(rows.iloc[0][column])


def group_int(groups: pd.DataFrame, group: str, column: str) -> int:
    rows = groups[groups["group"] == group]
    if rows.empty:
        return 0
    return int(rows.iloc[0][column])


def summarize_candidate(spec: dict[str, str]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_dir = repo_path(spec["audit_dir"])
    summary_path = audit_dir / "summary.json"
    if not summary_path.exists():
        row = {
            "candidate": spec["candidate"],
            "method": spec["method"],
            "rule": spec["rule"],
            "audit_dir": rel(audit_dir),
            "status": "pending_delta_audit",
            "delta_audit_available": False,
            "tensor_count": 0,
            "changed_tensors": 0,
            "changed_numel_fraction": 0.0,
            "total_relative_delta_norm": 0.0,
            "total_delta_norm": 0.0,
            "max_abs_delta": 0.0,
            "routed_relative_delta_norm": 0.0,
            "routed_changed_tensors": 0,
            "routed_changed_numel_fraction": 0.0,
            "routed_max_tensor_relative_delta": 0.0,
            "routed_p99_tensor_relative_delta": 0.0,
            "routed_p95_tensor_relative_delta": 0.0,
            "attention_relative_delta_norm": 0.0,
            "attention_changed_tensors": 0,
            "attention_changed_numel_fraction": 0.0,
            "attention_max_tensor_relative_delta": 0.0,
            "router_changed_tensors": 0,
            "router_tensors": 0,
        }
        for threshold in THRESHOLDS:
            suffix = str(threshold).replace(".", "_")
            row[f"routed_tensors_gt_{suffix}"] = 0
        return row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = read_json(audit_dir / "summary.json")
    groups = pd.read_csv(audit_dir / "group_delta_summary.csv")
    layers = pd.read_csv(audit_dir / "layer_delta_summary.csv")
    tensors = pd.read_csv(audit_dir / "tensor_delta_audit.csv")
    routed = tensors[tensors["group"] == "routed_expert_ffn"].copy()
    attention = tensors[tensors["group"] == "attention"].copy()

    row = {
        "candidate": spec["candidate"],
        "method": spec["method"],
        "rule": spec["rule"],
        "audit_dir": rel(audit_dir),
        "status": summary.get("status"),
        "delta_audit_available": True,
        "tensor_count": int(summary.get("tensor_count", 0)),
        "changed_tensors": int(summary.get("changed_tensors", 0)),
        "changed_numel_fraction": float(summary.get("changed_numel", 0))
        / max(1, int(summary.get("total_numel", 1))),
        "total_relative_delta_norm": float(summary.get("relative_delta_norm", 0.0)),
        "total_delta_norm": float(summary.get("total_delta_norm", 0.0)),
        "max_abs_delta": float(summary.get("max_abs_delta", 0.0)),
        "routed_relative_delta_norm": group_metric(groups, "routed_expert_ffn", "relative_delta_norm"),
        "routed_changed_tensors": group_int(groups, "routed_expert_ffn", "changed_tensors"),
        "routed_changed_numel_fraction": group_metric(
            groups, "routed_expert_ffn", "changed_numel_fraction"
        ),
        "routed_max_tensor_relative_delta": 0.0
        if routed.empty
        else float(routed["relative_delta_norm"].max()),
        "routed_p99_tensor_relative_delta": 0.0
        if routed.empty
        else float(routed["relative_delta_norm"].quantile(0.99)),
        "routed_p95_tensor_relative_delta": 0.0
        if routed.empty
        else float(routed["relative_delta_norm"].quantile(0.95)),
        "attention_relative_delta_norm": group_metric(groups, "attention", "relative_delta_norm"),
        "attention_changed_tensors": group_int(groups, "attention", "changed_tensors"),
        "attention_changed_numel_fraction": group_metric(groups, "attention", "changed_numel_fraction"),
        "attention_max_tensor_relative_delta": 0.0
        if attention.empty
        else float(attention["relative_delta_norm"].max()),
        "router_changed_tensors": int(summary.get("router_changed_tensors", 0)),
        "router_tensors": int(summary.get("router_tensors", 0)),
    }
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        row[f"routed_tensors_gt_{suffix}"] = int((routed["relative_delta_norm"] > threshold).sum())
    group_rows = groups.copy()
    group_rows.insert(0, "candidate", spec["candidate"])
    layer_rows = layers.copy()
    layer_rows.insert(0, "candidate", spec["candidate"])
    tensor_rows = tensors.copy()
    tensor_rows.insert(0, "candidate", spec["candidate"])
    return row, group_rows, layer_rows, tensor_rows


def build_pairwise(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = candidate_rows[candidate_rows.get("delta_audit_available", True).astype(bool)].to_dict("records")
    for before, after in zip(ordered, ordered[1:]):
        rows.append(
            {
                "from_candidate": before["candidate"],
                "to_candidate": after["candidate"],
                "delta_rule_change": f"{before['rule']} -> {after['rule']}",
                "total_relative_delta_norm_reduction": before["total_relative_delta_norm"]
                - after["total_relative_delta_norm"],
                "routed_relative_delta_norm_reduction": before["routed_relative_delta_norm"]
                - after["routed_relative_delta_norm"],
                "attention_relative_delta_norm_reduction": before["attention_relative_delta_norm"]
                - after["attention_relative_delta_norm"],
                "routed_max_tensor_relative_delta_reduction": before["routed_max_tensor_relative_delta"]
                - after["routed_max_tensor_relative_delta"],
                "routed_gt_1_reduction": int(before["routed_tensors_gt_1_0"])
                - int(after["routed_tensors_gt_1_0"]),
                "routed_gt_075_reduction": int(before["routed_tensors_gt_0_75"])
                - int(after["routed_tensors_gt_0_75"]),
                "routed_gt_065_reduction": int(before["routed_tensors_gt_0_65"])
                - int(after["routed_tensors_gt_0_65"]),
                "changed_tensors_reduction": int(before["changed_tensors"]) - int(after["changed_tensors"]),
            }
        )
    return pd.DataFrame(rows)


def normalized_structural_values(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    frame = candidate_rows.copy()
    for column in STRUCTURAL_RISK_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        lo = float(frame[column].min())
        hi = float(frame[column].max())
        if hi <= lo + 1e-12:
            frame[f"{column}_norm"] = 0.0
        else:
            frame[f"{column}_norm"] = (frame[column] - lo) / (hi - lo)
    frame["structural_risk_score"] = 0.0
    for column, weight in STRUCTURAL_DISTANCE_WEIGHTS.items():
        frame["structural_risk_score"] += weight * frame[f"{column}_norm"]
    frame["structural_safety_score"] = (1.0 - frame["structural_risk_score"]).clip(0.0, 1.0)
    return frame


def build_structural_pairwise(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    frame = normalized_structural_values(
        candidate_rows[candidate_rows.get("delta_audit_available", True).astype(bool)]
    )
    rows = []
    for _, left in frame.iterrows():
        for _, right in frame.iterrows():
            if left["candidate"] == right["candidate"]:
                continue
            weighted_distance = 0.0
            max_component = 0.0
            for column, weight in STRUCTURAL_DISTANCE_WEIGHTS.items():
                component = abs(float(left[f"{column}_norm"]) - float(right[f"{column}_norm"]))
                weighted_distance += weight * component
                max_component = max(max_component, component)
            rows.append(
                {
                    "from_candidate": left["candidate"],
                    "to_candidate": right["candidate"],
                    "from_method": left["method"],
                    "to_method": right["method"],
                    "structural_distance": weighted_distance,
                    "max_normalized_component_distance": max_component,
                    "from_structural_safety_score": float(left["structural_safety_score"]),
                    "to_structural_safety_score": float(right["structural_safety_score"]),
                    "to_minus_from_structural_safety": float(
                        right["structural_safety_score"] - left["structural_safety_score"]
                    ),
                    "total_relative_delta_norm_delta": float(
                        right["total_relative_delta_norm"] - left["total_relative_delta_norm"]
                    ),
                    "routed_relative_delta_norm_delta": float(
                        right["routed_relative_delta_norm"] - left["routed_relative_delta_norm"]
                    ),
                    "routed_max_tensor_relative_delta_delta": float(
                        right["routed_max_tensor_relative_delta"] - left["routed_max_tensor_relative_delta"]
                    ),
                    "routed_gt_0_65_delta": int(right["routed_tensors_gt_0_65"])
                    - int(left["routed_tensors_gt_0_65"]),
                    "attention_changed_delta": int(right["attention_changed_tensors"])
                    - int(left["attention_changed_tensors"]),
                    "router_changed_delta": int(right["router_changed_tensors"])
                    - int(left["router_changed_tensors"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["structural_distance", "from_candidate", "to_candidate"],
        ascending=[True, True, True],
    )


def dominates_structurally(left: pd.Series, right: pd.Series) -> bool:
    pairs = [(float(left[column]), float(right[column])) for column in STRUCTURAL_RISK_COLUMNS]
    return all(left_value <= right_value + 1e-12 for left_value, right_value in pairs) and any(
        left_value < right_value - 1e-12 for left_value, right_value in pairs
    )


def build_structural_dominance(candidate_rows: pd.DataFrame, structural_pairwise: pd.DataFrame) -> pd.DataFrame:
    frame = normalized_structural_values(
        candidate_rows[candidate_rows.get("delta_audit_available", True).astype(bool)]
    )
    nearest = (
        structural_pairwise.sort_values(["from_candidate", "structural_distance"])
        .groupby("from_candidate", as_index=False)
        .first()
        if not structural_pairwise.empty
        else pd.DataFrame()
    )
    rows = []
    for _, target in frame.iterrows():
        dominators = []
        for _, candidate in frame.iterrows():
            if candidate["candidate"] == target["candidate"]:
                continue
            if dominates_structurally(candidate, target):
                dominators.append(str(candidate["candidate"]))
        nearest_row = nearest[nearest["from_candidate"] == target["candidate"]]
        nearest_candidate = None if nearest_row.empty else str(nearest_row.iloc[0]["to_candidate"])
        nearest_distance = None if nearest_row.empty else float(nearest_row.iloc[0]["structural_distance"])
        rows.append(
            {
                "candidate": target["candidate"],
                "method": target["method"],
                "structural_safety_score": float(target["structural_safety_score"]),
                "structural_risk_score": float(target["structural_risk_score"]),
                "structurally_dominated": bool(dominators),
                "dominating_candidates": ",".join(dominators),
                "nearest_structural_candidate": nearest_candidate,
                "nearest_structural_distance": nearest_distance,
                "total_relative_delta_norm": float(target["total_relative_delta_norm"]),
                "routed_relative_delta_norm": float(target["routed_relative_delta_norm"]),
                "routed_max_tensor_relative_delta": float(target["routed_max_tensor_relative_delta"]),
                "routed_tensors_gt_0_65": int(target["routed_tensors_gt_0_65"]),
                "attention_changed_tensors": int(target["attention_changed_tensors"]),
                "router_changed_tensors": int(target["router_changed_tensors"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["structurally_dominated", "structural_safety_score"],
        ascending=[True, False],
    )


def build_thresholds(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidate_rows.iterrows():
        for threshold in THRESHOLDS:
            suffix = str(threshold).replace(".", "_")
            rows.append(
                {
                    "candidate": row["candidate"],
                    "threshold": threshold,
                    "routed_tensor_count": int(row[f"routed_tensors_gt_{suffix}"]),
                }
            )
    return pd.DataFrame(rows)


def build_layer_frontier(layer_rows: pd.DataFrame) -> pd.DataFrame:
    layers = layer_rows.copy()
    layers = layers[layers["layer"].notna()].copy()
    layers["layer"] = layers["layer"].astype(int)
    rows = []
    for layer, group in layers.groupby("layer", sort=True):
        values = {str(row["candidate"]): float(row["relative_delta_norm"]) for _, row in group.iterrows()}
        route = values.get("route_guarded")
        trust = values.get("trust_region")
        expert = values.get("expert_only")
        rows.append(
            {
                "layer": int(layer),
                "route_guarded_relative_delta_norm": route,
                "trust_region_relative_delta_norm": trust,
                "expert_only_relative_delta_norm": expert,
                "route_to_trust_reduction": None if route is None or trust is None else route - trust,
                "trust_to_expert_only_reduction": None if trust is None or expert is None else trust - expert,
            }
        )
    return pd.DataFrame(rows).sort_values("trust_region_relative_delta_norm", ascending=False)


def build_summary(
    candidate_rows: pd.DataFrame,
    pairwise_rows: pd.DataFrame,
    structural_pairwise: pd.DataFrame,
    structural_dominance: pd.DataFrame,
    layer_frontier: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    available = candidate_rows[candidate_rows.get("delta_audit_available", True).astype(bool)].copy()
    best_safety = available.sort_values(
        ["router_changed_tensors", "routed_tensors_gt_1_0", "routed_tensors_gt_0_75", "total_relative_delta_norm"],
        ascending=[True, True, True, True],
    ).iloc[0]
    trust = candidate_rows[candidate_rows["candidate"] == "trust_region"].iloc[0]
    expert = candidate_rows[candidate_rows["candidate"] == "expert_only"].iloc[0]
    tail = candidate_rows[candidate_rows["candidate"] == "tail_trimmed"].iloc[0]
    searched = candidate_rows[candidate_rows["candidate"] == "searched_no_gt065"].iloc[0]
    layer_chunk = candidate_rows[candidate_rows["candidate"] == "layer_chunk"].iloc[0]
    unified = candidate_rows[candidate_rows["candidate"] == "unified_mechanism"].iloc[0]
    mechanistic = candidate_rows[candidate_rows["candidate"] == "mechanistic_unified"].iloc[0]
    subspace = candidate_rows[candidate_rows["candidate"] == "subspace_scaled"].iloc[0]
    route_to_trust = pairwise_rows[
        (pairwise_rows["from_candidate"] == "audit_gated")
        & (pairwise_rows["to_candidate"] == "trust_region")
    ]
    trust_to_expert = pairwise_rows[
        (pairwise_rows["from_candidate"] == "trust_region")
        & (pairwise_rows["to_candidate"] == "expert_only")
    ]
    expert_to_tail = pairwise_rows[
        (pairwise_rows["from_candidate"] == "expert_only")
        & (pairwise_rows["to_candidate"] == "tail_trimmed")
    ]
    tail_to_searched = pairwise_rows[
        (pairwise_rows["from_candidate"] == "tail_trimmed")
        & (pairwise_rows["to_candidate"] == "searched_no_gt065")
    ]
    searched_to_layer_chunk = pairwise_rows[
        (pairwise_rows["from_candidate"] == "searched_no_gt065")
        & (pairwise_rows["to_candidate"] == "layer_chunk")
    ]
    layer_chunk_to_unified = pairwise_rows[
        (pairwise_rows["from_candidate"] == "layer_chunk")
        & (pairwise_rows["to_candidate"] == "unified_mechanism")
    ]
    unified_to_mechanistic = pairwise_rows[
        (pairwise_rows["from_candidate"] == "unified_mechanism")
        & (pairwise_rows["to_candidate"] == "mechanistic_unified")
    ]
    mechanistic_to_subspace = pairwise_rows[
        (pairwise_rows["from_candidate"] == "mechanistic_unified")
        & (pairwise_rows["to_candidate"] == "subspace_scaled")
    ]
    closest_structural = structural_pairwise.iloc[0] if not structural_pairwise.empty else {}
    mechanistic_neighbors = structural_pairwise[
        structural_pairwise["from_candidate"].eq("mechanistic_unified")
    ]
    closest_mechanistic_neighbor = mechanistic_neighbors.iloc[0] if not mechanistic_neighbors.empty else {}
    dominated_rows = structural_dominance[structural_dominance["structurally_dominated"].astype(bool)]
    return {
        "schema_version": 1,
        "status": "delta_frontier_ready",
        "candidate_count": int(len(candidate_rows)),
        "pending_delta_audit_candidate_count": int((~candidate_rows.get("delta_audit_available", True).astype(bool)).sum()),
        "best_delta_safety_candidate": str(best_safety["candidate"]),
        "structural_dominated_candidate_count": int(len(dominated_rows)),
        "structural_dominated_candidates": [str(item) for item in dominated_rows["candidate"].tolist()],
        "closest_structural_pair": None
        if not isinstance(closest_structural, pd.Series)
        else {
            "from_candidate": str(closest_structural["from_candidate"]),
            "to_candidate": str(closest_structural["to_candidate"]),
            "structural_distance": float(closest_structural["structural_distance"]),
            "to_minus_from_structural_safety": float(
                closest_structural["to_minus_from_structural_safety"]
            ),
        },
        "mechanistic_nearest_structural_candidate": None
        if not isinstance(closest_mechanistic_neighbor, pd.Series)
        else str(closest_mechanistic_neighbor["to_candidate"]),
        "mechanistic_nearest_structural_distance": None
        if not isinstance(closest_mechanistic_neighbor, pd.Series)
        else float(closest_mechanistic_neighbor["structural_distance"]),
        "mechanistic_nearest_structural_safety_delta": None
        if not isinstance(closest_mechanistic_neighbor, pd.Series)
        else float(closest_mechanistic_neighbor["to_minus_from_structural_safety"]),
        "router_changed_all_candidates": int(candidate_rows["router_changed_tensors"].sum()),
        "trust_region_total_relative_delta_norm": float(trust["total_relative_delta_norm"]),
        "expert_only_total_relative_delta_norm": float(expert["total_relative_delta_norm"]),
        "tail_trimmed_total_relative_delta_norm": float(tail["total_relative_delta_norm"]),
        "searched_no_gt065_total_relative_delta_norm": float(searched["total_relative_delta_norm"]),
        "layer_chunk_total_relative_delta_norm": float(layer_chunk["total_relative_delta_norm"]),
        "unified_mechanism_total_relative_delta_norm": float(unified["total_relative_delta_norm"]),
        "mechanistic_unified_total_relative_delta_norm": float(mechanistic["total_relative_delta_norm"]),
        "subspace_scaled_total_relative_delta_norm": float(subspace["total_relative_delta_norm"]),
        "expert_only_attention_changed_tensors": int(expert["attention_changed_tensors"]),
        "tail_trimmed_attention_changed_tensors": int(tail["attention_changed_tensors"]),
        "searched_no_gt065_attention_changed_tensors": int(searched["attention_changed_tensors"]),
        "layer_chunk_attention_changed_tensors": int(layer_chunk["attention_changed_tensors"]),
        "expert_only_router_changed_tensors": int(expert["router_changed_tensors"]),
        "tail_trimmed_router_changed_tensors": int(tail["router_changed_tensors"]),
        "searched_no_gt065_router_changed_tensors": int(searched["router_changed_tensors"]),
        "layer_chunk_router_changed_tensors": int(layer_chunk["router_changed_tensors"]),
        "unified_mechanism_router_changed_tensors": int(unified["router_changed_tensors"]),
        "mechanistic_unified_router_changed_tensors": int(mechanistic["router_changed_tensors"]),
        "subspace_scaled_router_changed_tensors": int(subspace["router_changed_tensors"]),
        "unified_mechanism_matches_searched_no_gt065_delta": bool(
            abs(float(unified["total_relative_delta_norm"]) - float(searched["total_relative_delta_norm"])) <= 1e-12
            and abs(float(unified["routed_relative_delta_norm"]) - float(searched["routed_relative_delta_norm"])) <= 1e-12
            and int(unified["routed_tensors_gt_0_65"]) == int(searched["routed_tensors_gt_0_65"])
            and int(unified["router_changed_tensors"]) == int(searched["router_changed_tensors"])
        ),
        "trust_to_expert_only_relative_norm_reduction": float(
            trust["total_relative_delta_norm"] - expert["total_relative_delta_norm"]
        ),
        "expert_only_to_tail_trimmed_relative_norm_reduction": float(
            expert["total_relative_delta_norm"] - tail["total_relative_delta_norm"]
        ),
        "tail_trimmed_to_searched_no_gt065_relative_norm_delta": float(
            searched["total_relative_delta_norm"] - tail["total_relative_delta_norm"]
        ),
        "searched_no_gt065_to_layer_chunk_relative_norm_reduction": 0.0
        if searched_to_layer_chunk.empty
        else float(searched_to_layer_chunk.iloc[0]["total_relative_delta_norm_reduction"]),
        "searched_no_gt065_to_layer_chunk_routed_gt_065_reduction": 0
        if searched_to_layer_chunk.empty
        else int(searched_to_layer_chunk.iloc[0].get("routed_gt_065_reduction", 0)),
        "trust_to_expert_only_attention_norm_reduction": float(
            trust["attention_relative_delta_norm"] - expert["attention_relative_delta_norm"]
        ),
        "trust_to_expert_only_routed_gt_075_reduction": 0
        if trust_to_expert.empty
        else int(trust_to_expert.iloc[0]["routed_gt_075_reduction"]),
        "expert_only_to_tail_trimmed_routed_gt_065_reduction": 0
        if expert_to_tail.empty
        else int(expert_to_tail.iloc[0].get("routed_gt_065_reduction", 0)),
        "expert_only_to_tail_trimmed_routed_gt_075_reduction": 0
        if expert_to_tail.empty
        else int(expert_to_tail.iloc[0]["routed_gt_075_reduction"]),
        "tail_trimmed_to_searched_no_gt065_routed_gt_065_delta": 0
        if tail_to_searched.empty
        else int(searched["routed_tensors_gt_0_65"] - tail["routed_tensors_gt_0_65"]),
        "tail_trimmed_routed_gt_0_6505": int(tail["routed_tensors_gt_0_6505"]),
        "searched_no_gt065_routed_gt_0_6505": int(searched["routed_tensors_gt_0_6505"]),
        "layer_chunk_routed_gt_0_6505": int(layer_chunk["routed_tensors_gt_0_6505"]),
        "layer_chunk_routed_gt_0_65": int(layer_chunk["routed_tensors_gt_0_65"]),
        "unified_mechanism_routed_gt_0_6505": int(unified["routed_tensors_gt_0_6505"]),
        "unified_mechanism_routed_gt_0_65": int(unified["routed_tensors_gt_0_65"]),
        "mechanistic_unified_routed_gt_0_6505": int(mechanistic["routed_tensors_gt_0_6505"]),
        "mechanistic_unified_routed_gt_0_65": int(mechanistic["routed_tensors_gt_0_65"]),
        "subspace_scaled_routed_gt_0_6505": int(subspace["routed_tensors_gt_0_6505"]),
        "subspace_scaled_routed_gt_0_65": int(subspace["routed_tensors_gt_0_65"]),
        "layer_chunk_to_unified_relative_norm_reduction": 0.0
        if layer_chunk_to_unified.empty
        else float(layer_chunk_to_unified.iloc[0]["total_relative_delta_norm_reduction"]),
        "layer_chunk_to_unified_routed_gt_065_reduction": 0
        if layer_chunk_to_unified.empty
        else int(layer_chunk_to_unified.iloc[0].get("routed_gt_065_reduction", 0)),
        "unified_to_mechanistic_relative_norm_reduction": 0.0
        if unified_to_mechanistic.empty
        else float(unified_to_mechanistic.iloc[0]["total_relative_delta_norm_reduction"]),
        "unified_to_mechanistic_routed_gt_065_reduction": 0
        if unified_to_mechanistic.empty
        else int(unified_to_mechanistic.iloc[0].get("routed_gt_065_reduction", 0)),
        "mechanistic_to_subspace_relative_norm_delta": float(
            subspace["total_relative_delta_norm"] - mechanistic["total_relative_delta_norm"]
        ),
        "mechanistic_to_subspace_routed_gt_065_reduction": 0
        if mechanistic_to_subspace.empty
        else int(mechanistic_to_subspace.iloc[0].get("routed_gt_065_reduction", 0)),
        "audit_to_trust_routed_gt_075_reduction": 0
        if route_to_trust.empty
        else int(route_to_trust.iloc[0]["routed_gt_075_reduction"]),
        "highest_trust_region_layers": [
            {str(k): clean(v) for k, v in row.items()}
            for row in layer_frontier.head(10).to_dict("records")
        ],
        "next_required_gate": "vllm_downstream_eval_trust_region_vs_expert_only_tail_trimmed_vs_searched_cap_law_vs_layer_chunk_vs_unified_vs_mechanistic_vs_subspace_scaled",
        "interpretation": (
            "Trust-region rules control the routed-expert delta tail; expert-only freezes attention "
            "without changing routed tail risk. Tail-trimmed then reduces the remaining routed tail "
            "while preserving the frozen attention/router contract. The searched no-gt-0.65 candidate "
            "tests whether the hand-built route/load/category risk penalties can be replaced by a simpler "
            "global expert cap. The layer/chunk candidate then tests whether layer sensitivity coefficients "
            "can reduce structural delta further without removing useful Coder specialization. The unified "
            "mechanism candidate now uses router/evidence/geometry risk to lower the routed tail below the "
            "uniform 0.65 cap while staying same-shape. The mechanistic unified candidate then turns the same "
            "signals into a benefit/curvature/interference scale law and is now materialized plus delta-audited. "
            "The subspace-scaled ablation then applies a "
            "small extra shrink to uncovered channel/chunk conflict experts, trading a slightly higher total norm than "
            "mechanistic unified for a lower routed tail. Attention, cap-law complexity, "
            "layer sensitivity, geometry-aware shrink, mechanistic scale law, and subspace-conflict shrink should therefore be "
            "decided by downstream eval, not by delta safety alone."
        ),
        "outputs": {
            "candidate_frontier": rel(output_dir / "candidate_delta_frontier.csv"),
            "group_frontier": rel(output_dir / "group_delta_frontier.csv"),
            "pairwise_reductions": rel(output_dir / "pairwise_delta_reductions.csv"),
            "structural_pairwise": rel(output_dir / "structural_pairwise_distances.csv"),
            "structural_dominance": rel(output_dir / "structural_dominance.csv"),
            "tail_thresholds": rel(output_dir / "tail_thresholds.csv"),
            "layer_frontier": rel(output_dir / "layer_delta_frontier.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }


def build_report(
    summary: dict[str, Any],
    candidate_rows: pd.DataFrame,
    pairwise_rows: pd.DataFrame,
    structural_pairwise: pd.DataFrame,
    structural_dominance: pd.DataFrame,
    layer_frontier: pd.DataFrame,
) -> str:
    closest_pair = summary.get("closest_structural_pair") or {}
    lines = [
        "# Qwen3 MoE Delta Frontier Probe",
        "",
        "这个 probe 只读已物化 checkpoint 的 delta audit，不重新加载模型权重。",
        "目的不是替代 vLLM 下游评测，而是回答：当前几版规则到底改变了哪些参数组，下一版算法应该把风险预算放在哪里。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Pending delta-audit candidates: `{summary['pending_delta_audit_candidate_count']}`",
        f"- Best delta-safety candidate: `{summary['best_delta_safety_candidate']}`",
        f"- Structurally dominated candidates: `{summary['structural_dominated_candidate_count']}`",
        f"- Closest structural pair: `{closest_pair.get('from_candidate')}` -> `{closest_pair.get('to_candidate')}` (`{fmt(closest_pair.get('structural_distance'))}`)",
        f"- Mechanistic nearest structural candidate: `{summary['mechanistic_nearest_structural_candidate']}` (`{fmt(summary['mechanistic_nearest_structural_distance'])}`)",
        f"- Trust-region total relative delta norm: `{fmt(summary['trust_region_total_relative_delta_norm'])}`",
        f"- Expert-only total relative delta norm: `{fmt(summary['expert_only_total_relative_delta_norm'])}`",
        f"- Tail-trimmed total relative delta norm: `{fmt(summary['tail_trimmed_total_relative_delta_norm'])}`",
        f"- Searched no-gt-0.65 total relative delta norm: `{fmt(summary['searched_no_gt065_total_relative_delta_norm'])}`",
        f"- Layer/chunk total relative delta norm: `{fmt(summary['layer_chunk_total_relative_delta_norm'])}`",
        f"- Unified mechanism total relative delta norm: `{fmt(summary['unified_mechanism_total_relative_delta_norm'])}`",
        f"- Mechanistic unified total relative delta norm: `{fmt(summary['mechanistic_unified_total_relative_delta_norm'])}`",
        f"- Subspace-scaled total relative delta norm: `{fmt(summary['subspace_scaled_total_relative_delta_norm'])}`",
        f"- Unified mechanism matches searched no-gt-0.65 delta: `{summary['unified_mechanism_matches_searched_no_gt065_delta']}`",
        f"- Trust -> expert-only relative norm reduction: `{fmt(summary['trust_to_expert_only_relative_norm_reduction'])}`",
        f"- Expert-only -> tail-trimmed relative norm reduction: `{fmt(summary['expert_only_to_tail_trimmed_relative_norm_reduction'])}`",
        f"- Tail-trimmed -> searched no-gt-0.65 relative norm delta: `{fmt(summary['tail_trimmed_to_searched_no_gt065_relative_norm_delta'])}`",
        f"- Searched no-gt-0.65 -> layer/chunk relative norm reduction: `{fmt(summary['searched_no_gt065_to_layer_chunk_relative_norm_reduction'])}`",
        f"- Layer/chunk -> unified relative norm reduction: `{fmt(summary['layer_chunk_to_unified_relative_norm_reduction'])}`",
        f"- Unified -> mechanistic relative norm reduction: `{fmt(summary['unified_to_mechanistic_relative_norm_reduction'])}`",
        f"- Mechanistic -> subspace-scaled relative norm delta: `{fmt(summary['mechanistic_to_subspace_relative_norm_delta'])}`",
        f"- Tail-trimmed / searched / layer-chunk / unified / mechanistic / subspace routed tensors >0.6505: `{summary['tail_trimmed_routed_gt_0_6505']}` / `{summary['searched_no_gt065_routed_gt_0_6505']}` / `{summary['layer_chunk_routed_gt_0_6505']}` / `{summary['unified_mechanism_routed_gt_0_6505']}` / `{summary['mechanistic_unified_routed_gt_0_6505']}` / `{summary['subspace_scaled_routed_gt_0_6505']}`",
        f"- Expert-only attention changed tensors: `{summary['expert_only_attention_changed_tensors']}`",
        f"- Tail-trimmed attention changed tensors: `{summary['tail_trimmed_attention_changed_tensors']}`",
        f"- Layer/chunk attention changed tensors: `{summary['layer_chunk_attention_changed_tensors']}`",
        f"- Expert-only router changed tensors: `{summary['expert_only_router_changed_tensors']}`",
        f"- Tail-trimmed router changed tensors: `{summary['tail_trimmed_router_changed_tensors']}`",
        f"- Layer/chunk router changed tensors: `{summary['layer_chunk_router_changed_tensors']}`",
        f"- Next required gate: `{summary['next_required_gate']}`",
        "",
        "## Candidate Frontier",
        "",
        "| candidate | total rel | routed rel | attention rel | router changed | max routed rel | routed >1 | routed >0.75 | routed >0.65 | routed >0.6505 | changed tensors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in candidate_rows.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {fmt(float(row['total_relative_delta_norm']))} | "
            f"{fmt(float(row['routed_relative_delta_norm']))} | "
            f"{fmt(float(row['attention_relative_delta_norm']))} | "
            f"{int(row['router_changed_tensors'])}/{int(row['router_tensors'])} | "
            f"{fmt(float(row['routed_max_tensor_relative_delta']))} | "
            f"{int(row['routed_tensors_gt_1_0'])} | "
            f"{int(row['routed_tensors_gt_0_75'])} | "
            f"{int(row['routed_tensors_gt_0_65'])} | "
            f"{int(row['routed_tensors_gt_0_6505'])} | "
            f"{int(row['changed_tensors'])} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Reductions",
            "",
            "| from | to | total rel reduction | routed rel reduction | attention rel reduction | routed >1 reduction | routed >0.75 reduction | routed >0.65 reduction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in pairwise_rows.iterrows():
        lines.append(
            "| "
            f"`{row['from_candidate']}` | `{row['to_candidate']}` | "
            f"{fmt(float(row['total_relative_delta_norm_reduction']))} | "
            f"{fmt(float(row['routed_relative_delta_norm_reduction']))} | "
            f"{fmt(float(row['attention_relative_delta_norm_reduction']))} | "
            f"{int(row['routed_gt_1_reduction'])} | "
            f"{int(row['routed_gt_075_reduction'])} | "
            f"{int(row.get('routed_gt_065_reduction', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Structural Pairwise Distance",
            "",
            "| from | to | distance | safety delta | total delta | routed delta | max routed delta | routed >0.65 delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in structural_pairwise.head(16).iterrows():
        lines.append(
            "| "
            f"`{row['from_candidate']}` | `{row['to_candidate']}` | "
            f"{fmt(float(row['structural_distance']))} | "
            f"{fmt(float(row['to_minus_from_structural_safety']))} | "
            f"{fmt(float(row['total_relative_delta_norm_delta']))} | "
            f"{fmt(float(row['routed_relative_delta_norm_delta']))} | "
            f"{fmt(float(row['routed_max_tensor_relative_delta_delta']))} | "
            f"{int(row['routed_gt_0_65_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Structural Dominance",
            "",
            "| candidate | safety | dominated | dominators | nearest | distance | total rel | routed rel | max routed | routed >0.65 |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in structural_dominance.iterrows():
        lines.append(
            "| "
            f"`{row['candidate']}` | {fmt(float(row['structural_safety_score']))} | "
            f"`{bool(row['structurally_dominated'])}` | `{row['dominating_candidates']}` | "
            f"`{row['nearest_structural_candidate']}` | {fmt(row['nearest_structural_distance'])} | "
            f"{fmt(float(row['total_relative_delta_norm']))} | "
            f"{fmt(float(row['routed_relative_delta_norm']))} | "
            f"{fmt(float(row['routed_max_tensor_relative_delta']))} | "
            f"{int(row['routed_tensors_gt_0_65'])} |"
        )
    lines.extend(
        [
            "",
            "## Highest Trust-Region Layers",
            "",
            "| layer | route rel | trust rel | expert-only rel | route->trust reduction | trust->expert-only reduction |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in layer_frontier.head(12).iterrows():
        lines.append(
            "| "
            f"{int(row['layer'])} | "
            f"{fmt(row['route_guarded_relative_delta_norm'])} | "
            f"{fmt(row['trust_region_relative_delta_norm'])} | "
            f"{fmt(row['expert_only_relative_delta_norm'])} | "
            f"{fmt(row['route_to_trust_reduction'])} | "
            f"{fmt(row['trust_to_expert_only_reduction'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "实际含义：trust-region/audit-gated 的价值主要是压 routed expert 的高 relative-delta tail；"
            "expert-only 只是把 shared attention 从候选里拿掉，几乎不改变 routed expert 风险；"
            "tail-trimmed 才继续压剩余 routed tail。"
            "searched no-gt-0.65 则把复杂风险 penalty 换成统一 cap，给下一轮 eval 一个更简单的候选。"
            "layer/chunk candidate 再把机制 leverage 里的层敏感度转成系数，给下一轮 eval 一个更细粒度的候选。"
            "unified mechanism candidate 进一步把 router/evidence/geometry risk 放进同一个约束优化器，成为当前最保守的 same-shape average 候选。"
            "mechanistic unified 把同样的内部信号改写成 benefit/curvature/interference scale law，并已通过真实 checkpoint delta audit。"
            "subspace-scaled ablation 只在 mechanistic/unified 链路之后额外压少数 uncovered 子空间冲突 expert。"
            "所以 attention 是否保留、risk penalty 是否保留、layer sensitivity 是否有用、subspace shrink 是否值得默认启用，都不能靠 delta safety 单独判断，必须靠同任务 vLLM 下游结果决定。",
            "",
            "## Files",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def build_frontier(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_records = []
    group_frames = []
    layer_frames = []
    for spec in DEFAULT_CANDIDATES:
        record, groups, layers, _ = summarize_candidate(spec)
        candidate_records.append(record)
        group_frames.append(groups)
        layer_frames.append(layers)
    candidate_rows = pd.DataFrame(candidate_records)
    group_rows = pd.concat(group_frames, ignore_index=True)
    layer_rows = pd.concat(layer_frames, ignore_index=True)
    pairwise_rows = build_pairwise(candidate_rows)
    structural_pairwise = build_structural_pairwise(candidate_rows)
    structural_dominance = build_structural_dominance(candidate_rows, structural_pairwise)
    threshold_rows = build_thresholds(candidate_rows)
    layer_frontier = build_layer_frontier(layer_rows)
    summary = build_summary(
        candidate_rows,
        pairwise_rows,
        structural_pairwise,
        structural_dominance,
        layer_frontier,
        output_dir,
    )

    candidate_rows.to_csv(output_dir / "candidate_delta_frontier.csv", index=False)
    group_rows.to_csv(output_dir / "group_delta_frontier.csv", index=False)
    pairwise_rows.to_csv(output_dir / "pairwise_delta_reductions.csv", index=False)
    structural_pairwise.to_csv(output_dir / "structural_pairwise_distances.csv", index=False)
    structural_dominance.to_csv(output_dir / "structural_dominance.csv", index=False)
    threshold_rows.to_csv(output_dir / "tail_thresholds.csv", index=False)
    layer_frontier.to_csv(output_dir / "layer_delta_frontier.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, candidate_rows, pairwise_rows, structural_pairwise, structural_dominance, layer_frontier),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/qwen3_moe_delta_frontier")
    return parser.parse_args()


def main() -> None:
    summary = build_frontier(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
