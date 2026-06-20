#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> pd.DataFrame:
    path = repo_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


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


def fnum(value: Any) -> float | None:
    value = clean_value(value)
    return None if value is None else float(value)


def close(left: Any, right: Any, tol: float = 1e-9) -> bool:
    left_value = fnum(left)
    right_value = fnum(right)
    if left_value is None or right_value is None:
        return left_value is right_value
    return abs(left_value - right_value) <= tol


def row_by_mechanism(table: pd.DataFrame, mechanism: str) -> dict[str, Any]:
    rows = table[table["mechanism"].astype(str) == mechanism]
    if rows.empty:
        return {}
    return {str(key): clean_value(value) for key, value in rows.iloc[0].to_dict().items()}


def count_status(table: pd.DataFrame, pattern: str) -> int:
    if table.empty:
        return 0
    return int(table["status"].astype(str).str.contains(pattern).sum())


def build_report(summary: dict[str, Any], matrix: pd.DataFrame) -> str:
    lines = [
        "# Average Trust-Region Bounds Smoke",
        "",
        "This smoke verifies that the trust-region bound table is synchronized with the current Dense/MoE probes, router margin gate, mechanistic cap, and final selector state.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Assertions: `{summary['passed_assertion_count']}/{summary['assertion_count']}`",
        "",
        "| assertion | expected | actual | passed |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in matrix.iterrows():
        lines.append(
            f"| `{row['assertion']}` | `{row['expected']}` | `{row['actual']}` | `{bool(row['passed'])}` |"
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
    parser = argparse.ArgumentParser(description="Smoke-test average trust-region bounds consistency.")
    parser.add_argument(
        "--bounds-dir",
        type=Path,
        default=Path("results/average_trust_region_bounds"),
    )
    parser.add_argument(
        "--final-selector-summary",
        type=Path,
        default=Path("results/qwen3_moe_final_candidate_selection/summary.json"),
    )
    parser.add_argument(
        "--router-margin-summary",
        type=Path,
        default=Path("results/qwen3_moe_router_margin_fragility/summary.json"),
    )
    parser.add_argument(
        "--mechanistic-summary",
        type=Path,
        default=Path("results/qwen3_moe_mechanistic_unified_candidate/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/average_trust_region_bounds_smoke"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds_dir = repo_path(args.bounds_dir)
    bounds_summary = read_json(bounds_dir / "summary.json")
    constraints = read_csv(bounds_dir / "trust_region_constraints.csv")
    final_selector = read_json(args.final_selector_summary)
    router_margin = read_json(args.router_margin_summary)
    mechanistic = read_json(args.mechanistic_summary)
    final_current = final_selector.get("current_selection") or {}
    dense_local = row_by_mechanism(constraints, "local_quadratic_trust_radius")
    dense_path = row_by_mechanism(constraints, "heldout_lambda_frontier")
    router_bound = row_by_mechanism(constraints, "router_topk_margin")
    mechanistic_bound = row_by_mechanism(constraints, "mechanistic_scale_law_cap")
    final_bound = row_by_mechanism(constraints, "final_average_acceptance")
    rows = [
        {
            "assertion": "constraint_count_matches_table",
            "expected": int(len(constraints)),
            "actual": int(bounds_summary.get("constraint_count", -1)),
            "passed": int(bounds_summary.get("constraint_count", -1)) == int(len(constraints)),
        },
        {
            "assertion": "passed_count_matches_status_table",
            "expected": count_status(constraints, "passed|allowed"),
            "actual": int(bounds_summary.get("passed_count", -1)),
            "passed": int(bounds_summary.get("passed_count", -1)) == count_status(constraints, "passed|allowed"),
        },
        {
            "assertion": "rejected_count_matches_status_table",
            "expected": count_status(constraints, "reject|conditional|do_not"),
            "actual": int(bounds_summary.get("rejected_count", -1)),
            "passed": int(bounds_summary.get("rejected_count", -1))
            == count_status(constraints, "reject|conditional|do_not"),
        },
        {
            "assertion": "waiting_count_matches_status_table",
            "expected": count_status(constraints, "awaiting|unaccepted"),
            "actual": int(bounds_summary.get("waiting_count", -1)),
            "passed": int(bounds_summary.get("waiting_count", -1))
            == count_status(constraints, "awaiting|unaccepted"),
        },
        {
            "assertion": "dense_local_bound_rejects_full_task_vector",
            "expected": "bound<1 and candidate_over_bound>1",
            "actual": (
                f"bound={dense_local.get('allowed_value')}, over={dense_local.get('candidate_over_bound')}, "
                f"status={dense_local.get('status')}"
            ),
            "passed": (
                fnum(dense_local.get("allowed_value")) is not None
                and fnum(dense_local.get("allowed_value")) < 1.0
                and fnum(dense_local.get("candidate_over_bound")) is not None
                and fnum(dense_local.get("candidate_over_bound")) > 1.0
                and dense_local.get("status") == "reject_linear_task_vector_average"
            ),
        },
        {
            "assertion": "dense_summary_safe_uniform_lambda_matches_constraint",
            "expected": bounds_summary.get("dense_safe_uniform_lambda"),
            "actual": dense_path.get("allowed_value"),
            "passed": close(bounds_summary.get("dense_safe_uniform_lambda"), dense_path.get("allowed_value")),
        },
        {
            "assertion": "router_safe_lambda_matches_margin_probe",
            "expected": router_margin.get("min_safe_lambda_proxy"),
            "actual": router_bound.get("allowed_value"),
            "passed": close(router_margin.get("min_safe_lambda_proxy"), router_bound.get("allowed_value")),
        },
        {
            "assertion": "router_midpoint_rejected_by_margin_bound",
            "expected": "candidate_over_bound>1",
            "actual": (
                f"candidate={router_bound.get('candidate_value')}, allowed={router_bound.get('allowed_value')}, "
                f"over={router_bound.get('candidate_over_bound')}, status={router_bound.get('status')}"
            ),
            "passed": (
                fnum(router_bound.get("candidate_over_bound")) is not None
                and fnum(router_bound.get("candidate_over_bound")) > 1.0
                and router_bound.get("status") == "reject_direct_router_average"
            ),
        },
        {
            "assertion": "mechanistic_cap_matches_candidate_summary",
            "expected": mechanistic.get("effective_hard_cap") or mechanistic.get("hard_cap"),
            "actual": mechanistic_bound.get("allowed_value"),
            "passed": close(
                mechanistic.get("effective_hard_cap") or mechanistic.get("hard_cap"),
                mechanistic_bound.get("allowed_value"),
            ),
        },
        {
            "assertion": "mechanistic_candidate_passes_cap_and_retention_bound",
            "expected": "cap_passed",
            "actual": (
                f"status={mechanistic_bound.get('status')}, max={mechanistic_bound.get('candidate_value')}, "
                f"cap={mechanistic_bound.get('allowed_value')}, retention={mechanistic.get('selected_nonbase_mass_retention')}"
            ),
            "passed": (
                mechanistic_bound.get("status") == "mechanistic_cap_and_retention_passed"
                and fnum(mechanistic_bound.get("candidate_value")) is not None
                and fnum(mechanistic_bound.get("allowed_value")) is not None
                and fnum(mechanistic_bound.get("candidate_value")) <= fnum(mechanistic_bound.get("allowed_value"))
                and fnum(mechanistic.get("selected_nonbase_mass_retention")) is not None
                and fnum(mechanistic.get("min_retention")) is not None
                and fnum(mechanistic.get("selected_nonbase_mass_retention")) >= fnum(mechanistic.get("min_retention"))
            ),
        },
        {
            "assertion": "final_bound_status_matches_final_selector",
            "expected": final_current.get("status"),
            "actual": final_bound.get("evidence"),
            "passed": (
                final_bound.get("status") == "awaiting_matched_vllm_eval"
                and final_current.get("status") == "awaiting_source_eval"
                and int(final_current.get("eligible_candidate_count", -1)) == 0
            ),
        },
    ]
    matrix = pd.DataFrame(rows)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "trust_region_bounds_smoke_matrix.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    matrix.to_csv(matrix_path, index=False)
    passed = int(matrix["passed"].astype(bool).sum())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed == len(matrix) else "failed",
        "assertion_count": int(len(matrix)),
        "passed_assertion_count": passed,
        "failed_assertion_count": int(len(matrix) - passed),
        "outputs": {
            "matrix": rel(matrix_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_report(summary, matrix), encoding="utf-8")
    print(f"Wrote average trust-region bounds smoke to {output_dir.resolve()}")
    print(f"Status: {summary['status']}; assertions {passed}/{len(matrix)}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
