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


def maybe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if value is None:
        return None
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    value = maybe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def row_for_method(selection: pd.DataFrame, method: str) -> pd.Series:
    rows = selection[selection["method"] == method]
    if rows.empty:
        raise KeyError(f"Missing method in selection table: {method}")
    return rows.iloc[0]


def dense_contrasts() -> pd.DataFrame:
    source_merge = read_json("results/vllm_source_merge_comparison/summary.json")
    bridge = read_json("results/probe_guided_dense_average_candidate/summary.json")
    module_guard = read_json("results/qwen_dense_module_guarded_candidate/summary.json")
    norm_guard = read_json("results/qwen_dense_norm_guarded_candidate/summary.json")
    selective_norm = read_json("results/qwen_dense_selective_norm_guarded_candidate/summary.json")

    rows = [
        {
            "contrast": "uniform_average_vs_best_source",
            "baseline": source_merge.get("best_source_model"),
            "candidate": source_merge.get("merge_model"),
            "avg_primary_delta": source_merge.get("merge_delta_vs_best_source_avg_primary"),
            "worst_primary_delta": source_merge.get("merge_delta_vs_best_source_worst_primary"),
            "probe_delta": None,
            "mechanism": "midpoint ridge and endpoint skill mismatch",
            "interpretation": "Uniform Dense averaging is not a safe default when task skills are not in the same low-loss basin.",
            "action": "reject_uniform_average",
        },
        {
            "contrast": "probe_guided_bridge_vs_uniform",
            "baseline": "qwen_0_5b_instruct_coder_uniform_average",
            "candidate": bridge.get("candidate_id"),
            "avg_primary_delta": bridge.get("vllm_eval", {}).get("delta_vs_uniform_avg_primary"),
            "worst_primary_delta": bridge.get("vllm_eval", {}).get("delta_vs_uniform_worst_primary"),
            "probe_delta": bridge.get("uniform_worst_gain"),
            "mechanism": "validation NLL selects a coder-anchored bridge away from the midpoint ridge",
            "interpretation": "Global coefficient search transfers from NLL probe to a small vLLM gain, but still does not beat the best source.",
            "action": "keep_global_probe_guided_bridge_as_dense_baseline",
        },
        {
            "contrast": "aggressive_module_guard_vs_global_bridge",
            "baseline": bridge.get("candidate_id"),
            "candidate": module_guard.get("candidate_id"),
            "avg_primary_delta": module_guard.get("vllm_eval", {}).get("delta_vs_global_bridge_avg_primary"),
            "worst_primary_delta": None,
            "probe_delta": None,
            "mechanism": "module-level conflict is too coarse; freezing anchors and damping MLP removes useful adaptation",
            "interpretation": "A high conflict score identifies a mechanism, but the intervention must be narrower than a whole-module rule.",
            "action": "reject_aggressive_module_guard",
        },
        {
            "contrast": "norm_only_guard_vs_global_bridge",
            "baseline": bridge.get("candidate_id"),
            "candidate": norm_guard.get("candidate_id"),
            "avg_primary_delta": norm_guard.get("vllm_eval", {}).get("delta_vs_global_bridge_avg_primary"),
            "worst_primary_delta": None,
            "probe_delta": None,
            "mechanism": "normalization deltas shift task distribution rather than uniformly improving all tasks",
            "interpretation": "Norm freezing is neutral on aggregate here, so it should remain an ablation knob rather than a default.",
            "action": "hold_norm_guard_for_targeted_ablation_only",
        },
        {
            "contrast": "selective_norm_guard_vs_global_bridge",
            "baseline": bridge.get("candidate_id"),
            "candidate": selective_norm.get("candidate_id"),
            "avg_primary_delta": selective_norm.get("vllm_eval", {}).get("delta_vs_global_bridge_avg_primary"),
            "worst_primary_delta": None,
            "probe_delta": None,
            "mechanism": "highest-conflict norm tensors are not sufficient causal levers",
            "interpretation": "Freezing only extreme sign-conflict norm tensors hurts the real endpoint score.",
            "action": "reject_static_high_conflict_tensor_freeze",
        },
    ]
    return pd.DataFrame(rows)


def moe_contrasts() -> pd.DataFrame:
    toy = read_json("results/toy_moe_merge/summary.json")
    selection = read_csv("results/toy_moe_method_selection/method_selection.csv")
    confidence = read_csv("results/toy_moe_merge/confidence_blended_expert_weights_by_expert.csv")

    def contrast(
        name: str,
        baseline_method: str,
        candidate_method: str,
        mechanism: str,
        interpretation: str,
        action: str,
    ) -> dict[str, Any]:
        base = row_for_method(selection, baseline_method)
        cand = row_for_method(selection, candidate_method)
        return {
            "contrast": name,
            "baseline": baseline_method,
            "candidate": candidate_method,
            "soft_worst_acc_delta": maybe_float(cand["worst_acc"]) - maybe_float(base["worst_acc"]),
            "hard_top2_worst_acc_delta": maybe_float(cand["dispatch_hard_top2_worst_acc"])
            - maybe_float(base["dispatch_hard_top2_worst_acc"]),
            "topk_overflow_delta": maybe_float(cand["capacity_max_topk_overflow_fraction"])
            - maybe_float(base["capacity_max_topk_overflow_fraction"]),
            "mechanism": mechanism,
            "interpretation": interpretation,
            "action": action,
        }

    rows = [
        contrast(
            "expert_identity_alignment",
            "all_weight_average",
            "expert_matched_average",
            "same-name expert tensors are semantically permuted",
            "Most of the toy MoE failure is recovered by matching expert output behavior before averaging.",
            "apply_layerwise_expert_alias_or_matching_before_expert_average",
        ),
        contrast(
            "router_calibration_after_matching",
            "matched_router_frozen_average",
            "matched_router_calibrated_average",
            "expert weights and router dispatch must be co-calibrated",
            "A frozen router is safer than naive router averaging, but held-out router calibration adds measurable soft accuracy.",
            "calibrate_or_distill_router_under_route_overlap_guard",
        ),
        contrast(
            "route_conditioned_output_projection",
            "expert_weight_search_router_calibrated_average",
            "expert_output_projection_router_calibrated_average",
            "expert source weights should explain routed output residuals, not only parameter deltas",
            "Output-space projection gives the best soft-router score, but it is less robust under hard sparse dispatch.",
            "use_projection_when_captured_fraction_is_high",
        ),
        contrast(
            "projection_confidence_blend",
            "unified_output_projection_moe_average",
            "unified_confidence_blended_moe_average",
            "projection is reliable for some experts and over-moves others",
            "Confidence blending trades a small soft loss for better hard top-2 sparse accuracy.",
            "blend_projection_with_search_using_expert_captured_fraction",
        ),
        contrast(
            "capacity_bias_correction",
            "unified_confidence_blended_moe_average",
            "unified_confidence_blended_bias_capacity_average",
            "top-k capacity overflow is a separate failure mode from task loss",
            "Bias-only capacity correction sharply reduces overflow but costs some accuracy; keep it as a capacity gate.",
            "apply_router_bias_delta_when_overflow_is_above_capacity_budget",
        ),
    ]
    dense_like = pd.DataFrame(rows)
    dense_like["projection_captured_fraction_mean"] = float(confidence["projection_captured_fraction"].mean())
    dense_like["projection_captured_fraction_min"] = float(confidence["projection_captured_fraction"].min())
    dense_like["projection_captured_fraction_max"] = float(confidence["projection_captured_fraction"].max())
    dense_like["expert_match_mean_cosine"] = toy.get("expert_match_mean_cosine")
    return dense_like


def intervention_plan(
    dense: pd.DataFrame,
    moe: pd.DataFrame,
    materialization: dict[str, Any],
) -> pd.DataFrame:
    dense_bridge_gain = float(
        dense.loc[dense["contrast"] == "probe_guided_bridge_vs_uniform", "avg_primary_delta"].iloc[0]
    )
    dense_module_delta = float(
        dense.loc[dense["contrast"] == "aggressive_module_guard_vs_global_bridge", "avg_primary_delta"].iloc[0]
    )
    dense_selective_delta = float(
        dense.loc[dense["contrast"] == "selective_norm_guard_vs_global_bridge", "avg_primary_delta"].iloc[0]
    )
    expert_gain = float(moe.loc[moe["contrast"] == "expert_identity_alignment", "soft_worst_acc_delta"].iloc[0])
    router_gain = float(moe.loc[moe["contrast"] == "router_calibration_after_matching", "soft_worst_acc_delta"].iloc[0])
    capacity_delta = float(moe.loc[moe["contrast"] == "capacity_bias_correction", "topk_overflow_delta"].iloc[0])
    qwen_blocker = materialization.get("current_blocking_stage", "")

    return pd.DataFrame(
        [
            {
                "stage": "dense_global_coefficients",
                "trigger": "uniform midpoint underperforms probe-selected bridge",
                "decision": "enabled_for_dense_baseline" if dense_bridge_gain > 0 else "disabled",
                "same_shape_action": "write base + 0.25 * instruct_delta + 1.0 * coder_delta",
                "why": "NLL barrier probe predicted the midpoint ridge and the vLLM eval confirmed a positive average-score delta.",
                "evidence": f"bridge avg primary delta vs uniform = {fmt(dense_bridge_gain)}",
            },
            {
                "stage": "dense_module_freeze_guard",
                "trigger": "module conflict looks high",
                "decision": "rejected_as_default" if dense_module_delta <= 0 else "candidate",
                "same_shape_action": "do not freeze full embedding/norm groups or damp all MLP tensors by default",
                "why": "The broad module action removed useful adaptation even though the conflict probe was real.",
                "evidence": f"module-guard delta vs bridge = {fmt(dense_module_delta)}",
            },
            {
                "stage": "dense_static_tensor_freeze",
                "trigger": "highest-conflict norm tensors",
                "decision": "rejected_as_default" if dense_selective_delta <= 0 else "candidate",
                "same_shape_action": "only use static tensor freezes inside a scored ablation sweep",
                "why": "The targeted high-conflict norm freeze did not improve held-out endpoint behavior.",
                "evidence": f"selective-norm delta vs bridge = {fmt(dense_selective_delta)}",
            },
            {
                "stage": "moe_expert_identity",
                "trigger": "expert output cosine or route behavior shows index mismatch",
                "decision": "required_for_moe" if expert_gain > 0.05 else "optional",
                "same_shape_action": "generate layer-scoped source_tensor_aliases before expert averaging",
                "why": "MoE experts are not exchangeable by tensor name; matching recovers the largest failure component.",
                "evidence": f"expert matching soft worst-acc gain = {fmt(expert_gain)}",
            },
            {
                "stage": "moe_expert_weights",
                "trigger": "per-expert projection captured_fraction is heterogeneous",
                "decision": "enabled_for_moe",
                "same_shape_action": "blend search weights and output-projection weights per expert using captured_fraction",
                "why": "Projection is highly reliable for some experts and unreliable for others, so a single expert-weight rule is too coarse.",
                "evidence": (
                    "captured_fraction min/mean/max = "
                    f"{fmt(moe['projection_captured_fraction_min'].iloc[0])}/"
                    f"{fmt(moe['projection_captured_fraction_mean'].iloc[0])}/"
                    f"{fmt(moe['projection_captured_fraction_max'].iloc[0])}"
                ),
            },
            {
                "stage": "moe_router",
                "trigger": "expert weights changed or route overlap is below guard threshold",
                "decision": "guarded_calibration_required" if router_gain > 0 else "freeze_router",
                "same_shape_action": "freeze, KD-calibrate, or small-lambda calibrate router only after route overlap checks",
                "why": "Router prior and expert weights must be calibrated together; blind router averaging is not a valid default.",
                "evidence": f"matched router calibration soft worst-acc gain = {fmt(router_gain)}",
            },
            {
                "stage": "moe_capacity",
                "trigger": "top-k load overflow exceeds capacity budget",
                "decision": "capacity_gate_not_unconditional",
                "same_shape_action": "write router-bias deltas only when overflow reduction is worth the accuracy cost",
                "why": "Capacity overflow is separate from NLL/accuracy; optimizing only accuracy leaves dispatch infeasible under sparse serving.",
                "evidence": f"confidence-blended bias-capacity overflow delta = {fmt(capacity_delta)}",
            },
            {
                "stage": "real_qwen_moe_materialization",
                "trigger": "all probe-gated MoE components need real Qwen topology and routing inputs",
                "decision": "blocked_until_real_probe" if qwen_blocker else "ready",
                "same_shape_action": "do not emit final Qwen MoE writer command until exact headers and real routing probe exist",
                "why": "Toy/template writer readiness is not evidence that the real Qwen MoE tensors and routes are ready.",
                "evidence": f"current blocking stage = {qwen_blocker or 'none'}",
            },
        ]
    )


def build_report(summary: dict[str, Any], dense: pd.DataFrame, moe: pd.DataFrame, plan: pd.DataFrame) -> str:
    lines = [
        "# Probe-Gated Unified Average Plan",
        "",
        "这份结果不是按场景枚举“哪个算法最好”，而是把 probe 观察到的机制转成同构 checkpoint 的 gate。Dense 和 MoE 共用一个原则：只有当 probe 和 held-out eval 都支持某个 intervention 时，才把它变成默认写出规则。",
        "",
        "## Current Decision",
        "",
        f"- Dense default: `{summary['dense_default_action']}`",
        f"- MoE default: `{summary['moe_default_action']}`",
        f"- Real Qwen MoE blocker: `{summary['real_qwen_moe_blocker']}`",
        f"- Same-shape invariant: `{summary['same_shape_invariant']}`",
        "",
        "## Mechanism Contrasts",
        "",
        "| domain | contrast | deltas | mechanism | action |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for _, row in dense.iterrows():
        dense_delta = (
            f"avg={fmt(row['avg_primary_delta'])}, "
            f"worst={fmt(row['worst_primary_delta'])}, "
            f"probe={fmt(row['probe_delta'])}"
        )
        lines.append(
            f"| Dense | `{row['contrast']}` | {dense_delta} | {row['mechanism']} | `{row['action']}` |"
        )
    for _, row in moe.iterrows():
        moe_delta = (
            f"soft={fmt(row['soft_worst_acc_delta'])}, "
            f"hard_top2={fmt(row['hard_top2_worst_acc_delta'])}, "
            f"overflow={fmt(row['topk_overflow_delta'])}"
        )
        lines.append(
            f"| MoE | `{row['contrast']}` | {moe_delta} | {row['mechanism']} | `{row['action']}` |"
        )
    lines.extend(
        [
            "",
            "## Unified Gate",
            "",
            "| stage | decision | same-shape action | evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in plan.iterrows():
        lines.append(
            f"| `{row['stage']}` | `{row['decision']}` | {row['same_shape_action']} | {row['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{summary['outputs']['dense_mechanism_contrasts']}`",
            f"- `{summary['outputs']['moe_mechanism_contrasts']}`",
            f"- `{summary['outputs']['intervention_plan']}`",
            f"- `{summary['outputs']['summary']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mechanism-level probe-gated average plan for Dense and MoE models.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/probe_gated_unified_average_plan"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense = dense_contrasts()
    moe = moe_contrasts()
    materialization = read_json("results/moe_materialization_pipeline_plan/summary.json")
    plan = intervention_plan(dense, moe, materialization)

    dense_bridge_gain = float(
        dense.loc[dense["contrast"] == "probe_guided_bridge_vs_uniform", "avg_primary_delta"].iloc[0]
    )
    module_guard_delta = float(
        dense.loc[dense["contrast"] == "aggressive_module_guard_vs_global_bridge", "avg_primary_delta"].iloc[0]
    )
    expert_gain = float(moe.loc[moe["contrast"] == "expert_identity_alignment", "soft_worst_acc_delta"].iloc[0])
    capacity_overflow_delta = float(moe.loc[moe["contrast"] == "capacity_bias_correction", "topk_overflow_delta"].iloc[0])

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "plan_ready_real_moe_waiting_for_probe",
        "dense_default_action": "probe_guided_global_bridge_only",
        "dense_bridge_avg_primary_delta_vs_uniform": dense_bridge_gain,
        "dense_module_guard_delta_vs_bridge": module_guard_delta,
        "moe_default_action": "expert_identity_plus_confidence_blended_expert_weights_plus_guarded_router_plus_capacity_gate",
        "moe_expert_identity_soft_worst_acc_gain": expert_gain,
        "moe_capacity_topk_overflow_delta": capacity_overflow_delta,
        "real_qwen_moe_blocker": materialization.get("current_blocking_stage"),
        "same_shape_invariant": "No stage changes tokenizer, model class, tensor names, tensor shapes, router shape, or expert count.",
        "outputs": {
            "dense_mechanism_contrasts": rel(output_dir / "dense_mechanism_contrasts.csv"),
            "moe_mechanism_contrasts": rel(output_dir / "moe_mechanism_contrasts.csv"),
            "intervention_plan": rel(output_dir / "intervention_plan.csv"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }

    dense.to_csv(output_dir / "dense_mechanism_contrasts.csv", index=False)
    moe.to_csv(output_dir / "moe_mechanism_contrasts.csv", index=False)
    plan.to_csv(output_dir / "intervention_plan.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(summary, dense, moe, plan), encoding="utf-8")
    print(f"Wrote probe-gated unified average plan to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
