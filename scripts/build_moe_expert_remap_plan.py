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


def build_alias_pattern(reference_expert: int, target_expert: int, layer_id: int | None = None) -> tuple[str, str]:
    if layer_id is None:
        return (
            rf"(^|.*\.)experts\.{reference_expert}\.",
            rf"\1experts.{target_expert}.",
        )
    return (
        rf"(^|.*\.layers\.{layer_id}\..*)experts\.{reference_expert}\.",
        rf"\1experts.{target_expert}.",
    )


def layer_value(item: pd.Series) -> int | None:
    for column in ("layer_id", "layer", "layer_idx"):
        if column in item and not pd.isna(item[column]):
            return int(item[column])
    return None


def build_remap(match: pd.DataFrame, source_name: str, min_cosine: float) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    alias_lines = [
        "# SOURCE::BASE_REGEX::SOURCE_REPLACEMENT",
        "# For each output/base expert index, read the matched source expert tensor.",
    ]
    sort_columns = [column for column in ("layer_id", "layer", "layer_idx", "reference_expert") if column in match.columns]
    for _, item in match.sort_values(sort_columns or ["reference_expert"]).iterrows():
        layer_id = layer_value(item)
        reference_expert = int(item["reference_expert"])
        target_expert = int(item["target_expert_before_alignment"])
        output_cosine = float(item["output_cosine"])
        pattern, replacement = build_alias_pattern(reference_expert, target_expert, layer_id=layer_id)
        status = "use_alias" if output_cosine >= min_cosine else "needs_manual_review"
        rows.append(
            {
                "source": source_name,
                "layer_id": layer_id,
                "output_expert": reference_expert,
                "matched_source_expert": target_expert,
                "output_cosine": output_cosine,
                "status": status,
                "base_regex": pattern,
                "source_replacement": replacement,
                "alias_rule": f"{source_name}::{pattern}::{replacement}",
            }
        )
        if status == "use_alias":
            alias_lines.append(f"{source_name}::{pattern}::{replacement}")
    return pd.DataFrame(rows), alias_lines


def build_writer_command(
    *,
    source_name: str,
    alias_file: Path,
    recommended_method: str | None,
    freeze_router: bool,
) -> str:
    parts = [
        "python scripts/write_same_shape_average_checkpoint.py",
        "--base MOE_BASE_OR_ANCHOR_PATH",
        "--source general=GENERAL_MODEL_PATH",
        f"--source {source_name}={source_name.upper()}_MODEL_PATH",
        "--source-weight general=0.5",
        f"--source-weight {source_name}=0.5",
    ]
    if freeze_router:
        parts.append("--freeze-router")
    parts.extend(
        [
            f"--source-tensor-alias-file {rel(alias_file)}",
            "--output-dir results/checkpoints/moe_expert_matched_candidate",
            "--dry-run",
        ]
    )
    if recommended_method and recommended_method != "expert_matched_average":
        parts.append(f"# recommended_method={recommended_method}")
    return " ".join(parts)


def build_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    remap: pd.DataFrame,
    writer_command: str,
) -> str:
    lines = [
        "# MoE Expert Remap Plan",
        "",
        "这个报告把 expert-output matching 结果转成 same-shape checkpoint writer 可以读取的 source tensor alias 规则。它解决的是 MoE average 中最实际的一步：输出 checkpoint 的 expert index 不变，但从 source checkpoint 读取已经匹配过的 expert tensor。",
        "",
        f"- Source with remap: `{summary['source_name']}`",
        f"- Recommended upstream method: `{summary.get('recommended_method')}`",
        f"- Remap status: `{summary['remap_status']}`",
        f"- Alias rules: `{summary['alias_rule_count']}`",
        f"- Layer-aware rules: `{summary['layer_aware_rule_count']}`",
        f"- Min output cosine: `{summary['min_output_cosine']:.4f}`",
        "",
        "## Expert Mapping",
        "",
        "| layer | output expert | matched source expert | output cosine | status |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in remap.iterrows():
        layer = "" if pd.isna(row.get("layer_id")) else int(row["layer_id"])
        lines.append(
            f"| {layer} | {int(row['output_expert'])} | {int(row['matched_source_expert'])} | "
            f"{float(row['output_cosine']):.4f} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Writer Dry-Run Command",
            "",
            "```bash",
            writer_command,
            "```",
            "",
            "## Interpretation",
            "",
            "- `source_tensor_aliases.txt` 不改变输出 tensor names/shapes，只改变某个 source 在读取 tensor 时用哪个 expert index。",
            "- 这和 `expert_matched_average` 的语义一致：先对齐 expert 功能，再做同构平均。",
            "- 如果某个 match 的 output cosine 低于阈值，报告会标成 `needs_manual_review`，不自动写 alias rule。",
            "",
            "## Files",
            "",
            f"- `{rel(output_dir / 'expert_remap.csv')}`",
            f"- `{rel(output_dir / 'source_tensor_aliases.txt')}`",
            f"- `{rel(output_dir / 'writer_command.txt')}`",
            f"- `{rel(output_dir / 'summary.json')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source tensor alias rules from MoE expert matching output.")
    parser.add_argument("--expert-match", default="results/toy_moe_merge/expert_match.csv")
    parser.add_argument("--method-selection-summary", default="results/toy_moe_method_selection/summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/toy_moe_expert_remap_plan"))
    parser.add_argument("--source-name", default="code")
    parser.add_argument("--min-cosine", type=float, default=0.90)
    parser.add_argument("--no-freeze-router", dest="freeze_router", action="store_false")
    parser.set_defaults(freeze_router=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    match = pd.read_csv(repo_path(args.expert_match))
    remap, alias_lines = build_remap(match, args.source_name, args.min_cosine)
    selection_summary_path = repo_path(args.method_selection_summary)
    selection_summary = read_json(selection_summary_path) if selection_summary_path.exists() else {}
    alias_file = output_dir / "source_tensor_aliases.txt"
    alias_file.write_text("\n".join(alias_lines) + "\n", encoding="utf-8")
    writer_command = build_writer_command(
        source_name=args.source_name,
        alias_file=alias_file,
        recommended_method=selection_summary.get("recommended_method"),
        freeze_router=args.freeze_router,
    )
    (output_dir / "writer_command.txt").write_text(writer_command + "\n", encoding="utf-8")
    remap.to_csv(output_dir / "expert_remap.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_name": args.source_name,
        "recommended_method": selection_summary.get("recommended_method"),
        "recommended_decision": selection_summary.get("recommended_decision"),
        "min_cosine_threshold": args.min_cosine,
        "min_output_cosine": float(remap["output_cosine"].min()) if not remap.empty else None,
        "mean_output_cosine": float(remap["output_cosine"].mean()) if not remap.empty else None,
        "alias_rule_count": int((remap["status"] == "use_alias").sum()),
        "layer_aware_rule_count": int(((remap["status"] == "use_alias") & remap["layer_id"].notna()).sum()),
        "manual_review_count": int((remap["status"] == "needs_manual_review").sum()),
        "remap_status": "ready" if int((remap["status"] == "needs_manual_review").sum()) == 0 else "needs_manual_review",
        "same_shape_constraint": "Output expert indices and tensor names stay unchanged; aliases only remap which source expert tensor is read.",
        "outputs": {
            "expert_remap": rel(output_dir / "expert_remap.csv"),
            "source_tensor_aliases": rel(alias_file),
            "writer_command": rel(output_dir / "writer_command.txt"),
            "summary": rel(output_dir / "summary.json"),
            "report": rel(output_dir / "report.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        build_report(output_dir=output_dir, summary=summary, remap=remap, writer_command=writer_command),
        encoding="utf-8",
    )
    print(f"Wrote MoE expert remap plan to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
