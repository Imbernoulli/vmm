#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(repo_path(path))


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def method_row(selection: pd.DataFrame, method: str) -> dict[str, Any]:
    rows = selection[selection["method"] == method]
    if rows.empty:
        return {}
    return {str(key): clean_value(value) for key, value in rows.iloc[0].items()}


def build_case_table(
    *,
    real_gauge: dict[str, Any],
    qwen_xcorr: dict[str, Any],
    toy_selection_summary: dict[str, Any],
    toy_selection: pd.DataFrame,
    max_aligned_degradation: float,
    min_identity_fraction: float,
    min_argmax_identity_fraction: float,
    max_capacity_overflow: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    naive_deg = float(real_gauge["naive_degradation_vs_baseline"])
    aligned_deg = float(real_gauge["aligned_degradation_vs_baseline"])
    gauge_risk = naive_deg > max_aligned_degradation and abs(aligned_deg) <= max_aligned_degradation
    rows.append(
        {
            "case": "real_olmoe_gauge_selfmerge",
            "model_or_pair": real_gauge["model"],
            "probe": "function-preserving expert/router permutation",
            "evidence": (
                f"same-name NLL degradation={naive_deg:.3f}; "
                f"aligned degradation={aligned_deg:.6f}; "
                f"permutation recovery={real_gauge['layers_perm_recovered']}/{real_gauge['n_moe_layers']}"
            ),
            "expert_identity_gate": "required",
            "router_gate": "permute_router_rows_with_experts",
            "capacity_gate": "not_tested_in_this_probe",
            "decision": "reject_same_name_average_without_alignment" if gauge_risk else "same_name_not_disproved",
            "same_shape_action": "match expert identity first; apply any expert remap as source-slice alias, not as a shape change",
            "blocking_probe": "cross_model_expert_correspondence_for_target_pair",
        }
    )

    identity_ok = (
        float(qwen_xcorr["frac_layers_identity_optimal"]) >= min_identity_fraction
        and float(qwen_xcorr["mean_argmax_is_identity_frac"]) >= min_argmax_identity_fraction
    )
    diag = float(qwen_xcorr["mean_diag_cos"])
    offdiag = float(qwen_xcorr["mean_offdiag_cos"])
    diag_ratio = diag / max(offdiag, 1e-12)
    rows.append(
        {
            "case": "qwen3_instruct_coder_cross_correspondence",
            "model_or_pair": f"{qwen_xcorr['model_a']} :: {qwen_xcorr['model_b']}",
            "probe": "base-subtracted expert-delta Hungarian correspondence",
            "evidence": (
                f"identity layers={qwen_xcorr['frac_layers_identity_optimal']:.3f}; "
                f"argmax identity={qwen_xcorr['mean_argmax_is_identity_frac']:.3f}; "
                f"diag/offdiag cosine ratio={diag_ratio:.1f}"
            ),
            "expert_identity_gate": "identity_mapping_allowed" if identity_ok else "remap_required",
            "router_gate": "routing_probe_required_before_router_average",
            "capacity_gate": "expert_load_probe_required",
            "decision": "identity_expert_average_allowed_with_routing_gate" if identity_ok else "compute_layerwise_expert_remap",
            "same_shape_action": "use identity expert slices first; freeze or small-step router until route overlap/load is measured",
            "blocking_probe": "real_qwen3_route_overlap_and_expert_load",
        }
    )

    all_weight = method_row(toy_selection, "all_weight_average")
    soft = toy_selection_summary.get("recommended_method")
    sparse = toy_selection_summary.get("recommended_sparse_method")
    capacity = toy_selection_summary.get("recommended_sparse_capacity_aware_method")
    cap_overflow = float(toy_selection_summary.get("recommended_sparse_capacity_aware_topk_overflow_fraction", 1.0))
    rows.append(
        {
            "case": "toy_moe_method_selector",
            "model_or_pair": "synthetic two-domain MoE",
            "probe": "task loss + route readiness + dispatch + capacity",
            "evidence": (
                f"all_weight decision={all_weight.get('decision', 'missing')}; "
                f"soft={soft}; hard_top2={sparse}; capacity={capacity}; "
                f"capacity overflow={cap_overflow:.4f}"
            ),
            "expert_identity_gate": "expert_match_or_confidence_blend",
            "router_gate": "calibrate_or_route_kd_with_overlap_guard",
            "capacity_gate": "pass" if cap_overflow <= max_capacity_overflow else "bias_capacity_correction_required",
            "decision": "use_dispatch_aware_selector_not_static_average",
            "same_shape_action": "select expert weights and router policy by serving mode; never expand expert count",
            "blocking_probe": "target_model_generation_or_vllm_eval",
        }
    )
    return pd.DataFrame(rows)


def build_stage_table(cases: pd.DataFrame) -> pd.DataFrame:
    qwen = cases[cases["case"] == "qwen3_instruct_coder_cross_correspondence"].iloc[0]
    toy = cases[cases["case"] == "toy_moe_method_selector"].iloc[0]
    identity_action = (
        "use identity mapping for Qwen3 pair"
        if qwen["expert_identity_gate"] == "identity_mapping_allowed"
        else "generate layerwise source expert remap"
    )
    return pd.DataFrame(
        [
            {
                "stage": "topology",
                "gate": "same_shape_config_and_packed_expert_layout",
                "required_evidence": "tensor names, tensor shapes, router rows, expert dimension, tokenizer/model class",
                "action_if_pass": "continue",
                "action_if_fail": "do_not_materialize",
            },
            {
                "stage": "expert_identity",
                "gate": "expert correspondence or gauge self-merge evidence",
                "required_evidence": "identity-optimal fraction or layerwise Hungarian permutation",
                "action_if_pass": identity_action,
                "action_if_fail": "apply layerwise expert alias/remap before any expert averaging",
            },
            {
                "stage": "expert_weighting",
                "gate": "route-conditioned output or held-out task loss",
                "required_evidence": "expert captured_fraction, expert load, or per-expert held-out search",
                "action_if_pass": "confidence-blend searched and output-projection expert weights",
                "action_if_fail": "freeze expert source or stay at endpoint/base",
            },
            {
                "stage": "router",
                "gate": "route overlap, top1 agreement, and router loss",
                "required_evidence": "router_summary.csv and route_overlap.csv",
                "action_if_pass": "small-step router calibration, route-KD, or freeze-router policy",
                "action_if_fail": "freeze router and reject router averaging",
            },
            {
                "stage": "capacity",
                "gate": "top-k expert load overflow under serving dispatch",
                "required_evidence": "expert_load.csv or router_capacity_metrics.csv",
                "action_if_pass": str(toy["capacity_gate"]),
                "action_if_fail": "router-bias capacity correction or reject sparse-serving candidate",
            },
            {
                "stage": "behavior_eval",
                "gate": "NLL plus generation/vLLM held-out tasks",
                "required_evidence": "downstream metrics with endpoints and bad-average baselines",
                "action_if_pass": "materialize same-shape checkpoint",
                "action_if_fail": "do_not_publish_candidate_as_average",
            },
        ]
    )


def build_summary(
    *,
    cases: pd.DataFrame,
    stage_table: pd.DataFrame,
    qwen_xcorr: dict[str, Any],
    real_gauge: dict[str, Any],
    toy_selection_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    qwen_case = cases[cases["case"] == "qwen3_instruct_coder_cross_correspondence"].iloc[0].to_dict()
    real_case = cases[cases["case"] == "real_olmoe_gauge_selfmerge"].iloc[0].to_dict()
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "selector_ready_waiting_for_real_qwen_routing_probe",
        "same_shape_invariant": "No selector stage changes model class, tokenizer, hidden size, router shape, expert count, or tensor shape.",
        "global_moe_gauge_decision": real_case["decision"],
        "qwen3_expert_identity_decision": qwen_case["decision"],
        "qwen3_identity_fraction": qwen_xcorr.get("frac_layers_identity_optimal"),
        "qwen3_argmax_identity_fraction": qwen_xcorr.get("mean_argmax_is_identity_frac"),
        "real_gauge_naive_degradation": real_gauge.get("naive_degradation_vs_baseline"),
        "real_gauge_aligned_degradation": real_gauge.get("aligned_degradation_vs_baseline"),
        "toy_soft_recommendation": toy_selection_summary.get("recommended_method"),
        "toy_sparse_recommendation": toy_selection_summary.get("recommended_sparse_method"),
        "toy_capacity_recommendation": toy_selection_summary.get("recommended_sparse_capacity_aware_method"),
        "next_blocking_probe": "real_qwen3_route_overlap_and_expert_load",
        "outputs": {
            "case_table": rel(output_dir / "selector_cases.csv"),
            "stage_table": rel(output_dir / "selector_stages.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }


def build_report(summary: dict[str, Any], cases: pd.DataFrame, stages: pd.DataFrame) -> str:
    lines = [
        "# MoE Probe-Gated Selector",
        "",
        "这份结果把真实 MoE gauge 反事实、Qwen3 expert correspondence 和 toy MoE route/capacity selector 合成一套同构 MoE average gate。它不是静态说某个算法最好，而是决定什么时候必须对齐 experts、什么时候可以暂用 identity、什么时候必须等 routing/load probe。",
        "",
        "## Current Decision",
        "",
        f"- Global MoE gauge rule: `{summary['global_moe_gauge_decision']}`",
        f"- Qwen3 Instruct/Coder expert identity: `{summary['qwen3_expert_identity_decision']}`",
        f"- Next blocking probe: `{summary['next_blocking_probe']}`",
        f"- Same-shape invariant: `{summary['same_shape_invariant']}`",
        "",
        "## Evidence Cases",
        "",
        "| case | decision | expert gate | router gate | capacity gate | evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in cases.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['decision']}` | `{row['expert_identity_gate']}` | "
            f"`{row['router_gate']}` | `{row['capacity_gate']}` | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Selector Stages",
            "",
            "| stage | gate | action if pass | action if fail |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in stages.iterrows():
        lines.append(
            f"| `{row['stage']}` | {row['gate']} | {row['action_if_pass']} | {row['action_if_fail']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "真实 OLMoE 反事实说明 expert index 是 gauge，不是稳定语义；Qwen3 cross-correspondence 说明这对官方同族 checkpoint 目前 identity mapping 可信，但这只能通过 expert identity gate，不能替代 routing/load gate。toy MoE selector 进一步说明 router policy 和 capacity overflow 是独立失败模式。因此真实 Qwen3 MoE materialization 的下一步不是直接 average，而是先跑 route overlap 和 expert load，再决定 freeze-router、small-step calibration、route-KD 或 router-bias capacity correction。",
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['case_table']}`",
            f"- `{summary['outputs']['stage_table']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a probe-gated MoE selector from real and synthetic MoE evidence.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_probe_gated_selector"))
    parser.add_argument("--real-gauge-summary", type=Path, default=Path("results/fp_moe_real_probe/summary.json"))
    parser.add_argument(
        "--qwen-cross-correspondence",
        type=Path,
        default=Path("results/fp_moe_real_probe/qwen3_instruct_coder/cross_correspondence.json"),
    )
    parser.add_argument("--toy-selection-summary", type=Path, default=Path("results/toy_moe_method_selection/summary.json"))
    parser.add_argument("--toy-selection-csv", type=Path, default=Path("results/toy_moe_method_selection/method_selection.csv"))
    parser.add_argument("--max-aligned-degradation", type=float, default=0.05)
    parser.add_argument("--min-identity-fraction", type=float, default=0.95)
    parser.add_argument("--min-argmax-identity-fraction", type=float, default=0.95)
    parser.add_argument("--max-capacity-overflow", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_gauge = read_json(args.real_gauge_summary)
    qwen_xcorr = read_json(args.qwen_cross_correspondence)
    toy_selection_summary = read_json(args.toy_selection_summary)
    toy_selection = read_csv(args.toy_selection_csv)

    cases = build_case_table(
        real_gauge=real_gauge,
        qwen_xcorr=qwen_xcorr,
        toy_selection_summary=toy_selection_summary,
        toy_selection=toy_selection,
        max_aligned_degradation=args.max_aligned_degradation,
        min_identity_fraction=args.min_identity_fraction,
        min_argmax_identity_fraction=args.min_argmax_identity_fraction,
        max_capacity_overflow=args.max_capacity_overflow,
    )
    stages = build_stage_table(cases)
    summary = build_summary(
        cases=cases,
        stage_table=stages,
        qwen_xcorr=qwen_xcorr,
        real_gauge=real_gauge,
        toy_selection_summary=toy_selection_summary,
        output_dir=output_dir,
    )

    cases.to_csv(output_dir / "selector_cases.csv", index=False)
    stages.to_csv(output_dir / "selector_stages.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, cases, stages), encoding="utf-8")
    print(f"Wrote MoE probe-gated selector to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
