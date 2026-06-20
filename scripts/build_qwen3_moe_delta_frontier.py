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
]
THRESHOLDS = [1.0, 0.75, 0.6505, 0.65, 0.5]


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
    ordered = candidate_rows.to_dict("records")
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
    layer_frontier: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    best_safety = candidate_rows.sort_values(
        ["router_changed_tensors", "routed_tensors_gt_1_0", "routed_tensors_gt_0_75", "total_relative_delta_norm"],
        ascending=[True, True, True, True],
    ).iloc[0]
    trust = candidate_rows[candidate_rows["candidate"] == "trust_region"].iloc[0]
    expert = candidate_rows[candidate_rows["candidate"] == "expert_only"].iloc[0]
    tail = candidate_rows[candidate_rows["candidate"] == "tail_trimmed"].iloc[0]
    searched = candidate_rows[candidate_rows["candidate"] == "searched_no_gt065"].iloc[0]
    layer_chunk = candidate_rows[candidate_rows["candidate"] == "layer_chunk"].iloc[0]
    unified = candidate_rows[candidate_rows["candidate"] == "unified_mechanism"].iloc[0]
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
    return {
        "schema_version": 1,
        "status": "delta_frontier_ready",
        "candidate_count": int(len(candidate_rows)),
        "best_delta_safety_candidate": str(best_safety["candidate"]),
        "router_changed_all_candidates": int(candidate_rows["router_changed_tensors"].sum()),
        "trust_region_total_relative_delta_norm": float(trust["total_relative_delta_norm"]),
        "expert_only_total_relative_delta_norm": float(expert["total_relative_delta_norm"]),
        "tail_trimmed_total_relative_delta_norm": float(tail["total_relative_delta_norm"]),
        "searched_no_gt065_total_relative_delta_norm": float(searched["total_relative_delta_norm"]),
        "layer_chunk_total_relative_delta_norm": float(layer_chunk["total_relative_delta_norm"]),
        "unified_mechanism_total_relative_delta_norm": float(unified["total_relative_delta_norm"]),
        "expert_only_attention_changed_tensors": int(expert["attention_changed_tensors"]),
        "tail_trimmed_attention_changed_tensors": int(tail["attention_changed_tensors"]),
        "searched_no_gt065_attention_changed_tensors": int(searched["attention_changed_tensors"]),
        "layer_chunk_attention_changed_tensors": int(layer_chunk["attention_changed_tensors"]),
        "expert_only_router_changed_tensors": int(expert["router_changed_tensors"]),
        "tail_trimmed_router_changed_tensors": int(tail["router_changed_tensors"]),
        "searched_no_gt065_router_changed_tensors": int(searched["router_changed_tensors"]),
        "layer_chunk_router_changed_tensors": int(layer_chunk["router_changed_tensors"]),
        "unified_mechanism_router_changed_tensors": int(unified["router_changed_tensors"]),
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
        "layer_chunk_to_unified_relative_norm_reduction": 0.0
        if layer_chunk_to_unified.empty
        else float(layer_chunk_to_unified.iloc[0]["total_relative_delta_norm_reduction"]),
        "layer_chunk_to_unified_routed_gt_065_reduction": 0
        if layer_chunk_to_unified.empty
        else int(layer_chunk_to_unified.iloc[0].get("routed_gt_065_reduction", 0)),
        "audit_to_trust_routed_gt_075_reduction": 0
        if route_to_trust.empty
        else int(route_to_trust.iloc[0]["routed_gt_075_reduction"]),
        "highest_trust_region_layers": [
            {str(k): clean(v) for k, v in row.items()}
            for row in layer_frontier.head(10).to_dict("records")
        ],
        "next_required_gate": "vllm_downstream_eval_trust_region_vs_expert_only_tail_trimmed_vs_searched_cap_law_vs_layer_chunk_vs_unified",
        "interpretation": (
            "Trust-region rules control the routed-expert delta tail; expert-only freezes attention "
            "without changing routed tail risk. Tail-trimmed then reduces the remaining routed tail "
            "while preserving the frozen attention/router contract. The searched no-gt-0.65 candidate "
            "tests whether the hand-built route/load/category risk penalties can be replaced by a simpler "
            "global expert cap. The layer/chunk candidate then tests whether layer sensitivity coefficients "
            "can reduce structural delta further without removing useful Coder specialization. The unified "
            "mechanism candidate now uses router/evidence/geometry risk to lower the routed tail below the "
            "uniform 0.65 cap while staying same-shape. Attention, cap-law complexity, layer sensitivity, "
            "and geometry-aware shrink should therefore be decided by downstream eval, not by delta safety alone."
        ),
        "outputs": {
            "candidate_frontier": rel(output_dir / "candidate_delta_frontier.csv"),
            "group_frontier": rel(output_dir / "group_delta_frontier.csv"),
            "pairwise_reductions": rel(output_dir / "pairwise_delta_reductions.csv"),
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
    layer_frontier: pd.DataFrame,
) -> str:
    lines = [
        "# Qwen3 MoE Delta Frontier Probe",
        "",
        "这个 probe 只读已物化 checkpoint 的 delta audit，不重新加载模型权重。",
        "目的不是替代 vLLM 下游评测，而是回答：当前几版规则到底改变了哪些参数组，下一版算法应该把风险预算放在哪里。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Best delta-safety candidate: `{summary['best_delta_safety_candidate']}`",
        f"- Trust-region total relative delta norm: `{fmt(summary['trust_region_total_relative_delta_norm'])}`",
        f"- Expert-only total relative delta norm: `{fmt(summary['expert_only_total_relative_delta_norm'])}`",
        f"- Tail-trimmed total relative delta norm: `{fmt(summary['tail_trimmed_total_relative_delta_norm'])}`",
        f"- Searched no-gt-0.65 total relative delta norm: `{fmt(summary['searched_no_gt065_total_relative_delta_norm'])}`",
        f"- Layer/chunk total relative delta norm: `{fmt(summary['layer_chunk_total_relative_delta_norm'])}`",
        f"- Unified mechanism total relative delta norm: `{fmt(summary['unified_mechanism_total_relative_delta_norm'])}`",
        f"- Unified mechanism matches searched no-gt-0.65 delta: `{summary['unified_mechanism_matches_searched_no_gt065_delta']}`",
        f"- Trust -> expert-only relative norm reduction: `{fmt(summary['trust_to_expert_only_relative_norm_reduction'])}`",
        f"- Expert-only -> tail-trimmed relative norm reduction: `{fmt(summary['expert_only_to_tail_trimmed_relative_norm_reduction'])}`",
        f"- Tail-trimmed -> searched no-gt-0.65 relative norm delta: `{fmt(summary['tail_trimmed_to_searched_no_gt065_relative_norm_delta'])}`",
        f"- Searched no-gt-0.65 -> layer/chunk relative norm reduction: `{fmt(summary['searched_no_gt065_to_layer_chunk_relative_norm_reduction'])}`",
        f"- Layer/chunk -> unified relative norm reduction: `{fmt(summary['layer_chunk_to_unified_relative_norm_reduction'])}`",
        f"- Tail-trimmed / searched / layer-chunk / unified routed tensors >0.6505: `{summary['tail_trimmed_routed_gt_0_6505']}` / `{summary['searched_no_gt065_routed_gt_0_6505']}` / `{summary['layer_chunk_routed_gt_0_6505']}` / `{summary['unified_mechanism_routed_gt_0_6505']}`",
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
            "所以 attention 是否保留、risk penalty 是否保留、layer sensitivity 是否有用，都不能靠 delta safety 单独判断，必须靠同任务 vLLM 下游结果决定。",
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
    threshold_rows = build_thresholds(candidate_rows)
    layer_frontier = build_layer_frontier(layer_rows)
    summary = build_summary(candidate_rows, pairwise_rows, layer_frontier, output_dir)

    candidate_rows.to_csv(output_dir / "candidate_delta_frontier.csv", index=False)
    group_rows.to_csv(output_dir / "group_delta_frontier.csv", index=False)
    pairwise_rows.to_csv(output_dir / "pairwise_delta_reductions.csv", index=False)
    threshold_rows.to_csv(output_dir / "tail_thresholds.csv", index=False)
    layer_frontier.to_csv(output_dir / "layer_delta_frontier.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_report(summary, candidate_rows, pairwise_rows, layer_frontier),
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
