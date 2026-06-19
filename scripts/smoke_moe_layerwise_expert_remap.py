#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_moe_expert_remap_plan import build_remap


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


def synthetic_match() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer_id": 0,
                "reference_expert": 0,
                "target_expert_before_alignment": 1,
                "output_cosine": 0.99,
            },
            {
                "layer_id": 0,
                "reference_expert": 1,
                "target_expert_before_alignment": 0,
                "output_cosine": 0.95,
            },
            {
                "layer_id": 1,
                "reference_expert": 0,
                "target_expert_before_alignment": 0,
                "output_cosine": 0.92,
            },
            {
                "layer_id": 1,
                "reference_expert": 1,
                "target_expert_before_alignment": 1,
                "output_cosine": 0.70,
            },
        ]
    )


def run_smoke(output_dir: Path, min_cosine: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    match = synthetic_match()
    match_path = output_dir / "expert_match_input.csv"
    match.to_csv(match_path, index=False)

    remap, alias_lines = build_remap(match, source_name="code", min_cosine=min_cosine)
    remap_path = output_dir / "expert_remap.csv"
    alias_path = output_dir / "source_tensor_aliases.txt"
    remap.to_csv(remap_path, index=False)
    alias_path.write_text("\n".join(alias_lines) + "\n", encoding="utf-8")

    usable = remap[remap["status"] == "use_alias"]
    manual = remap[remap["status"] == "needs_manual_review"]
    layer_aware_aliases = [
        line for line in alias_lines if line and not line.startswith("#") and r"layers\." in line
    ]
    expected_alias_count = 3
    checks = [
        ("alias_rule_count", len(usable) == expected_alias_count),
        ("manual_review_count", len(manual) == 1),
        ("all_usable_rules_layer_aware", len(layer_aware_aliases) == expected_alias_count),
        (
            "manual_review_rule_not_emitted",
            not any("layers\\.1\\..*experts\\.1" in line for line in layer_aware_aliases),
        ),
    ]
    check_rows = [{"check": name, "passed": passed} for name, passed in checks]
    pd.DataFrame(check_rows).to_csv(output_dir / "checks.csv", index=False)
    all_passed = all(row["passed"] for row in check_rows)
    summary = {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "input_rows": int(len(match)),
        "alias_rule_count": int(len(usable)),
        "layer_aware_rule_count": int(len(layer_aware_aliases)),
        "manual_review_count": int(len(manual)),
        "min_cosine_threshold": min_cosine,
        "outputs": {
            "checks": rel(output_dir / "checks.csv"),
            "expert_match_input": rel(match_path),
            "expert_remap": rel(remap_path),
            "source_tensor_aliases": rel(alias_path),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# MoE Layer-Wise Expert Remap Smoke",
        "",
        "这个 smoke 验证 `build_moe_expert_remap_plan.py` 在 `expert_match.csv` 带 `layer_id` 时会生成 layer-scoped source tensor alias，而不是把某个 expert id 映射错误地应用到所有层。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Input rows: `{summary['input_rows']}`",
        f"- Alias rules: `{summary['alias_rule_count']}`",
        f"- Layer-aware rules: `{summary['layer_aware_rule_count']}`",
        f"- Manual review rows: `{summary['manual_review_count']}`",
        "",
        "## Files",
        "",
        f"- `{summary['outputs']['checks']}`",
        f"- `{summary['outputs']['expert_remap']}`",
        f"- `{summary['outputs']['source_tensor_aliases']}`",
        f"- `{summary['outputs']['summary']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test layer-wise MoE expert remap alias generation.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_layerwise_expert_remap_smoke"))
    parser.add_argument("--min-cosine", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(repo_path(args.output_dir), args.min_cosine)
    print(f"Wrote MoE layer-wise expert remap smoke to {repo_path(args.output_dir).resolve()}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
