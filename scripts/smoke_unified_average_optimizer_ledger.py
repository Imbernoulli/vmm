#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

import build_unified_average_optimizer as optimizer


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
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def ledger_for(summary: dict[str, Any]) -> pd.DataFrame:
    hypotheses = optimizer.build_mechanism_hypotheses(summary)
    return optimizer.build_evidence_ledger(summary, hypotheses)


def set_moe(summary: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = copy.deepcopy(summary)
    out.setdefault("moe", {}).update(updates)
    return out


def case_specs(base: dict[str, Any]) -> list[dict[str, Any]]:
    unified_method = "qwen3_moe_unified_mechanism_candidate"
    source_method = "source_qwen3_30b_instruct"
    router_method = "smoke_router_calibrated_cap001"
    baseline_method = "qwen3_moe_searched_no_gt065_max_retention_candidate"
    return [
        {
            "case": "awaiting_eval_keeps_provisional",
            "summary": set_moe(
                base,
                qwen3_final_selection_status="awaiting_source_eval",
                qwen3_final_selected_method=None,
                qwen3_eligible_candidates=0,
                qwen3_router_calibration_status="awaiting_baseline_eval",
                qwen3_router_calibration_selected_method=None,
            ),
            "expect": {
                "moe_risk_weighted_expert_caps_preserve_useful_route_mass": "awaiting_downstream_eval",
                "downstream_source_dominance_is_final_gate": "awaiting_downstream_eval",
                "router_calibration_repairs_dispatch_but_is_not_acceptance": "promising_but_unaccepted",
            },
        },
        {
            "case": "unified_candidate_downstream_win",
            "summary": set_moe(
                base,
                qwen3_final_selection_status="select_unified_candidate",
                qwen3_final_selected_method=unified_method,
                qwen3_eligible_candidates=1,
            ),
            "expect": {
                "moe_risk_weighted_expert_caps_preserve_useful_route_mass": "supports_current_action",
                "downstream_source_dominance_is_final_gate": "supports_current_action",
            },
        },
        {
            "case": "source_endpoint_dominates",
            "summary": set_moe(
                base,
                qwen3_final_selection_status="keep_source_endpoint",
                qwen3_final_selected_method=source_method,
                qwen3_eligible_candidates=0,
            ),
            "expect": {
                "moe_risk_weighted_expert_caps_preserve_useful_route_mass": "falsified_by_downstream_eval",
                "downstream_source_dominance_is_final_gate": "supports_source_fallback",
            },
        },
        {
            "case": "router_calibration_selected",
            "summary": set_moe(
                base,
                qwen3_router_calibration_status="selected_router_calibrated_candidate",
                qwen3_router_calibration_selected_method=router_method,
                qwen3_router_calibration_eligible_candidates=1,
            ),
            "expect": {
                "router_calibration_repairs_dispatch_but_is_not_acceptance": "supports_current_action",
            },
        },
        {
            "case": "router_calibration_rejected",
            "summary": set_moe(
                base,
                qwen3_router_calibration_status="keep_frozen_router_baseline",
                qwen3_router_calibration_selected_method=baseline_method,
                qwen3_router_calibration_eligible_candidates=0,
            ),
            "expect": {
                "router_calibration_repairs_dispatch_but_is_not_acceptance": "supports_freeze_router_baseline",
            },
        },
    ]


def build_report(summary: dict[str, Any], matrix: pd.DataFrame) -> str:
    lines = [
        "# Unified Average Optimizer Ledger Smoke",
        "",
        "This smoke matrix verifies that the mechanism evidence ledger changes verdicts when downstream or router-selection evidence changes. It guards against treating structural/NLL probes as final acceptance.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Passed cases: `{summary['passed_case_count']}/{summary['case_count']}`",
        "",
        "| case | hypothesis | expected | actual | passed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['case']}` | `{row['hypothesis_id']}` | `{row['expected_verdict']}` | "
            f"`{row['actual_verdict']}` | `{bool(row['passed'])}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{summary['outputs']['matrix']}`",
            f"- `{summary['outputs']['summary']}`",
            f"- `{summary['outputs']['report']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test unified average optimizer evidence-ledger transitions.")
    parser.add_argument("--summary", type=Path, default=Path("results/unified_average_optimizer/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/unified_average_optimizer_ledger_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = read_json(args.summary)
    if not base:
        raise SystemExit(f"Missing optimizer summary: {args.summary}")
    rows: list[dict[str, Any]] = []
    for spec in case_specs(base):
        ledger = ledger_for(spec["summary"])
        by_id = {str(row["hypothesis_id"]): row for _, row in ledger.iterrows()}
        for hypothesis_id, expected in spec["expect"].items():
            actual = str(by_id[hypothesis_id]["verdict"])
            rows.append(
                {
                    "case": spec["case"],
                    "hypothesis_id": hypothesis_id,
                    "expected_verdict": expected,
                    "actual_verdict": actual,
                    "passed": actual == expected,
                }
            )
    matrix = pd.DataFrame(rows)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "ledger_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    case_count = int(matrix["case"].nunique()) if not matrix.empty else 0
    passed_case_count = int(matrix.groupby("case")["passed"].all().sum()) if not matrix.empty else 0
    summary = {
        "schema_version": 1,
        "status": "passed" if bool(matrix["passed"].all()) else "failed",
        "case_count": case_count,
        "assertion_count": int(len(matrix)),
        "passed_case_count": passed_case_count,
        "failed_case_count": case_count - passed_case_count,
        "passed_assertion_count": int(matrix["passed"].sum()),
        "failed_assertion_count": int((~matrix["passed"]).sum()),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    matrix.to_csv(matrix_path, index=False)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, matrix), encoding="utf-8")
    print(f"Wrote unified average optimizer ledger smoke to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; cases {passed_case_count}/{case_count}")


if __name__ == "__main__":
    main()
