#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

LITERATURE_SOURCES = [
    {
        "key": "expert_merging",
        "source": "https://arxiv.org/abs/2509.25712",
        "mechanism": "Learns layer/chunk coefficients from unlabeled hidden/logit alignment, supporting calibration-data-driven merging rather than fixed coefficients.",
    },
    {
        "key": "mergeme",
        "source": "https://arxiv.org/abs/2502.00997",
        "mechanism": "Identifies MoE parameter interference and routing as separate problems during expert model merging.",
    },
    {
        "key": "mergemoe",
        "source": "https://arxiv.org/abs/2510.14436",
        "mechanism": "Frames MoE expert merging through expert-output behavior instead of tensor-name averaging alone.",
    },
    {
        "key": "git_rebasin",
        "source": "https://arxiv.org/abs/2209.04836",
        "mechanism": "Permutation/gauge alignment must be handled before weight-space interpolation is meaningful.",
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


def read_json(path: str | Path) -> dict[str, Any]:
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def fnum(value: Any) -> float:
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def result_rows(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    role_by_method = {
        "instruct": "source",
        "coder": "source",
        "linear_merge": "average_baseline",
        "linear_merge_routercal": "router_calibrated_average",
    }
    trainable_policy = {
        "instruct": "none",
        "coder": "none",
        "linear_merge": "none_after_weight_average",
        "linear_merge_routercal": "router_only_gate_weight_update",
    }
    for method, values in results.items():
        rows.append(
            {
                "method": method,
                "role": role_by_method.get(method, "unknown"),
                "general_nll": fnum(values["general"]),
                "code_nll": fnum(values["code"]),
                "avg_nll": fnum(values["avg"]),
                "worst_nll": fnum(values["worst"]),
                "trainable_policy": trainable_policy.get(method, "unknown"),
            }
        )
    return pd.DataFrame(rows)


def build_deltas(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    instruct = results["instruct"]
    coder = results["coder"]
    linear = results["linear_merge"]
    routercal = results["linear_merge_routercal"]
    best_source_general = min(fnum(instruct["general"]), fnum(coder["general"]))
    best_source_code = min(fnum(instruct["code"]), fnum(coder["code"]))
    best_source_worst = min(fnum(instruct["worst"]), fnum(coder["worst"]))
    best_source_avg = min(fnum(instruct["avg"]), fnum(coder["avg"]))
    return pd.DataFrame(
        [
            {
                "metric": "worst_nll_reduction_vs_linear",
                "value": fnum(linear["worst"]) - fnum(routercal["worst"]),
                "interpretation": "positive means router-only calibration improved the averaged MoE",
            },
            {
                "metric": "avg_nll_reduction_vs_linear",
                "value": fnum(linear["avg"]) - fnum(routercal["avg"]),
                "interpretation": "positive means the calibration improved the two-task average NLL",
            },
            {
                "metric": "general_nll_reduction_vs_linear",
                "value": fnum(linear["general"]) - fnum(routercal["general"]),
                "interpretation": "positive means general held-out NLL improved after router calibration",
            },
            {
                "metric": "code_nll_reduction_vs_linear",
                "value": fnum(linear["code"]) - fnum(routercal["code"]),
                "interpretation": "positive means code held-out NLL improved after router calibration",
            },
            {
                "metric": "routercal_general_gap_to_best_source",
                "value": fnum(routercal["general"]) - best_source_general,
                "interpretation": "negative would mean router calibration beats both sources on general NLL",
            },
            {
                "metric": "routercal_code_gap_to_best_source",
                "value": fnum(routercal["code"]) - best_source_code,
                "interpretation": "negative means router calibration beats both sources on code NLL",
            },
            {
                "metric": "routercal_worst_gap_to_best_source",
                "value": fnum(routercal["worst"]) - best_source_worst,
                "interpretation": "negative would justify accepting the router-calibrated average by worst-task NLL alone",
            },
            {
                "metric": "routercal_avg_gap_to_best_source",
                "value": fnum(routercal["avg"]) - best_source_avg,
                "interpretation": "negative would justify accepting the router-calibrated average by average NLL alone",
            },
        ]
    )


def build_report(summary: dict[str, Any], method_metrics: pd.DataFrame, deltas: pd.DataFrame) -> str:
    linear = summary["linear_merge"]
    routercal = summary["router_calibrated"]
    lines = [
        "# Qwen3 MoE Router Calibration NLL Probe",
        "",
        "这个 artifact 固化一个真实 Qwen3-30B-A3B Instruct/Coder MoE probe：先做 50/50 linear merge，再只训练 `mlp.gate.weight` router tensors，experts 和 shared modules 全部冻结。它回答的不是“最终能不能接受这个 checkpoint”，而是“MoE average 的剩余误差是不是主要来自 router dispatch”。",
        "",
        "## Result",
        "",
        f"- Linear merge worst-NLL: `{fmt(linear['worst_nll'])}`；router-calibrated worst-NLL: `{fmt(routercal['worst_nll'])}`；reduction `{fmt(summary['worst_nll_reduction_vs_linear'])}`。",
        f"- Linear merge avg-NLL: `{fmt(linear['avg_nll'])}`；router-calibrated avg-NLL: `{fmt(routercal['avg_nll'])}`；reduction `{fmt(summary['avg_nll_reduction_vs_linear'])}`。",
        f"- Code NLL goes from `{fmt(linear['code_nll'])}` to `{fmt(routercal['code_nll'])}`；gap to best source is `{fmt(summary['routercal_code_gap_to_best_source'])}`。",
        f"- General NLL goes from `{fmt(linear['general_nll'])}` to `{fmt(routercal['general_nll'])}`；gap to best source is `{fmt(summary['routercal_general_gap_to_best_source'])}`。",
        "",
        "Interpretation: router-only training improves the averaged MoE on both probe tasks, and code NLL beats both sources, but worst/avg NLL still does not dominate the best source. So the mechanism is real, while acceptance still needs the downstream vLLM gate.",
        "",
        "## Mechanism",
        "",
        "For an aligned MoE, expert averaging mainly changes the expert functions, but router averaging changes a discrete top-k dispatch boundary. If the two source routers disagree on many tokens, the average router can send a token to a compromise expert set even when the experts themselves remain usable. Training only the router is a direct test of this hypothesis: if NLL drops while experts are frozen, the residual error is dispatch/co-adaptation, not expert geometry.",
        "",
        "## Method Metrics",
        "",
        "| method | role | general NLL | code NLL | avg NLL | worst NLL | trainable policy |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in method_metrics.iterrows():
        lines.append(
            f"| `{row['method']}` | `{row['role']}` | {fmt(row['general_nll'])} | "
            f"{fmt(row['code_nll'])} | {fmt(row['avg_nll'])} | {fmt(row['worst_nll'])} | "
            f"`{row['trainable_policy']}` |"
        )
    lines.extend(["", "## Mechanism Deltas", "", "| metric | value | interpretation |", "| --- | ---: | --- |"])
    for _, row in deltas.iterrows():
        lines.append(f"| `{row['metric']}` | {fmt(row['value'])} | {row['interpretation']} |")
    lines.extend(
        [
            "",
            "## Literature Priors Used",
            "",
            "| key | source | mechanism |",
            "| --- | --- | --- |",
        ]
    )
    for source in LITERATURE_SOURCES:
        lines.append(f"| `{source['key']}` | {source['source']} | {source['mechanism']} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['method_metrics']}`",
            f"- `{summary['outputs']['mechanism_deltas']}`",
            f"- `{summary['outputs']['literature_sources']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_summary = read_json(args.input_summary)
    results = input_summary["results"]
    method_metrics = result_rows(results)
    deltas = build_deltas(results)
    linear_row = method_metrics[method_metrics["method"] == "linear_merge"].iloc[0].to_dict()
    routercal_row = method_metrics[method_metrics["method"] == "linear_merge_routercal"].iloc[0].to_dict()
    delta_lookup = {row["metric"]: fnum(row["value"]) for _, row in deltas.iterrows()}

    status = (
        "router_calibration_improves_linear_merge_but_needs_downstream_gate"
        if delta_lookup["worst_nll_reduction_vs_linear"] > 0
        else "router_calibration_not_supported_by_probe"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "input_summary": rel(args.input_summary),
        "models": input_summary.get("models", {}),
        "steps": input_summary.get("steps"),
        "lr": input_summary.get("lr"),
        "linear_merge": linear_row,
        "router_calibrated": routercal_row,
        "worst_nll_reduction_vs_linear": delta_lookup["worst_nll_reduction_vs_linear"],
        "avg_nll_reduction_vs_linear": delta_lookup["avg_nll_reduction_vs_linear"],
        "general_nll_reduction_vs_linear": delta_lookup["general_nll_reduction_vs_linear"],
        "code_nll_reduction_vs_linear": delta_lookup["code_nll_reduction_vs_linear"],
        "routercal_general_gap_to_best_source": delta_lookup["routercal_general_gap_to_best_source"],
        "routercal_code_gap_to_best_source": delta_lookup["routercal_code_gap_to_best_source"],
        "routercal_worst_gap_to_best_source": delta_lookup["routercal_worst_gap_to_best_source"],
        "routercal_avg_gap_to_best_source": delta_lookup["routercal_avg_gap_to_best_source"],
        "acceptance_decision": (
            "mechanism_supported_but_do_not_accept_without_matched_vllm_eval"
            if delta_lookup["routercal_worst_gap_to_best_source"] >= 0
            else "candidate_can_enter_downstream_gate"
        ),
        "outputs": {},
    }

    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_path = output_dir / "method_metrics.csv"
    deltas_path = output_dir / "mechanism_deltas.csv"
    literature_path = output_dir / "literature_sources.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary["outputs"] = {
        "method_metrics": rel(method_path),
        "mechanism_deltas": rel(deltas_path),
        "literature_sources": rel(literature_path),
        "summary": rel(summary_path),
        "report": rel(report_path),
    }

    method_metrics.to_csv(method_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    literature_path.write_text(
        json.dumps(json_safe(LITERATURE_SOURCES), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, method_metrics, deltas), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a formal Qwen3 MoE router-calibration NLL probe artifact.")
    parser.add_argument("--input-summary", type=Path, default=Path("results/fp_moe_router_calibrate/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen3_moe_router_calibration_nll_probe"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args)
    print(f"Wrote router-calibration NLL probe to {repo_path(args.output_dir).resolve()}")
    print(
        "Status: "
        f"{summary['status']}; worst reduction={summary['worst_nll_reduction_vs_linear']:.4f}; "
        f"code gap to best source={summary['routercal_code_gap_to_best_source']:.4f}"
    )


if __name__ == "__main__":
    main()
