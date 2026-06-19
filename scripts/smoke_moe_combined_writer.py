#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from write_same_shape_average_checkpoint import write_average_checkpoint


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


def make_checkpoint(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 2,
                "model_type": "tiny_combined_moe_writer_smoke",
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "num_hidden_layers": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})


def source_tensors(offset: float) -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.full((2, 2), offset + 40.0),
        "model.int_buffer": torch.tensor([1, 2], dtype=torch.int64),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.full((2, 2), offset + 10.0),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.full((2, 2), offset + 20.0),
        "model.layers.0.router.bias": torch.full((2,), offset + 35.0),
        "model.layers.0.router.weight": torch.full((2, 2), offset + 30.0),
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), offset + 1.0),
    }


def build_tensor_rules(path: Path) -> None:
    lines = [
        "# Shared module rule.",
        ".*self_attn.*::general=0.5,code=0.5",
        "# Expert rules. The code source is aliased before these weights are applied.",
        ".*experts\\.0\\..*::general=0.2,code=0.8",
        ".*experts\\.1\\..*::general=0.7,code=0.3",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_alias_rules(path: Path) -> None:
    lines = [
        "# SOURCE::BASE_REGEX::SOURCE_REPLACEMENT",
        "code::(^|.*\\.)experts\\.0\\.::\\1experts.1.",
        "code::(^|.*\\.)experts\\.1\\.::\\1experts.0.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tensor_add_csv(path: Path) -> None:
    lines = [
        "tensor,index,delta,reason",
        "model.layers.0.router.bias,0,0.125,lift_underused_expert_logit",
        "model.layers.0.router.bias,1,-0.375,reduce_overloaded_expert_logit",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_tensors(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    general = source_tensors(100.0)
    code = source_tensors(200.0)
    expected: dict[str, torch.Tensor] = {}
    for name, base_tensor in base.items():
        if not torch.is_floating_point(base_tensor):
            expected[name] = base_tensor
        elif "self_attn" in name:
            expected[name] = base_tensor + 0.5 * (general[name] - base_tensor) + 0.5 * (code[name] - base_tensor)
        elif "experts.0." in name:
            aliased_code = name.replace("experts.0.", "experts.1.")
            expected[name] = base_tensor + 0.2 * (general[name] - base_tensor) + 0.8 * (code[aliased_code] - base_tensor)
        elif "experts.1." in name:
            aliased_code = name.replace("experts.1.", "experts.0.")
            expected[name] = base_tensor + 0.7 * (general[name] - base_tensor) + 0.3 * (code[aliased_code] - base_tensor)
        elif name == "model.layers.0.router.bias":
            expected[name] = base_tensor + torch.tensor([0.125, -0.375], dtype=base_tensor.dtype)
        else:
            expected[name] = base_tensor
    return expected


def sanitize_manifest(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base"] = "TMP/base"
    manifest["sources"] = {"code": "TMP/code", "general": "TMP/general"}
    manifest["tensor_add_csv"] = ["TMP/router_bias_delta.csv"]
    for source_name, source_summary in manifest.get("source_summaries", {}).items():
        source_summary["root"] = f"TMP/{source_name}"
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moe_combined_writer_") as tmp_raw:
        tmp = Path(tmp_raw)
        base_dir = tmp / "base"
        general_dir = tmp / "general"
        code_dir = tmp / "code"
        written_dir = tmp / "written"
        tensor_rule_file = tmp / "tensor_rules.txt"
        alias_file = tmp / "source_tensor_aliases.txt"
        tensor_add_csv = tmp / "router_bias_delta.csv"

        base = source_tensors(0.0)
        make_checkpoint(base_dir, base)
        make_checkpoint(general_dir, source_tensors(100.0))
        make_checkpoint(code_dir, source_tensors(200.0))
        build_tensor_rules(tensor_rule_file)
        build_alias_rules(alias_file)
        build_tensor_add_csv(tensor_add_csv)

        args = argparse.Namespace(
            allow_missing_source_tensors=False,
            base=str(base_dir),
            copy_metadata=True,
            dry_run=False,
            freeze_regex=[],
            freeze_router=True,
            output_dir=str(written_dir),
            output_dtype="base",
            source=[f"general={general_dir}", f"code={code_dir}"],
            source_tensor_alias=[],
            source_tensor_alias_file=[str(alias_file)],
            source_weight=["general=0.0", "code=0.0"],
            tensor_add_csv=[str(tensor_add_csv)],
            tensor_rule=[],
            tensor_rule_file=[str(tensor_rule_file)],
        )
        manifest = write_average_checkpoint(args)
        actual = load_file(str(written_dir / "model.safetensors"))
        expected = expected_tensors(base)

        rows = []
        all_passed = True
        for name in sorted(expected):
            expected_tensor = expected[name]
            actual_tensor = actual[name]
            if torch.is_floating_point(expected_tensor):
                max_abs_error = float((actual_tensor - expected_tensor).abs().max().item())
                passed = max_abs_error < 1e-6
                expected_mean = float(expected_tensor.mean().item())
                actual_mean = float(actual_tensor.mean().item())
            else:
                passed = torch.equal(actual_tensor, expected_tensor)
                max_abs_error = 0.0 if passed else float("inf")
                expected_mean = None
                actual_mean = None
            all_passed = all_passed and passed
            rows.append(
                {
                    "actual_mean": actual_mean,
                    "expected_mean": expected_mean,
                    "max_abs_error": max_abs_error,
                    "passed": passed,
                    "tensor": name,
                }
            )
        sanitized_manifest = sanitize_manifest(written_dir / "merge_manifest.json", output_dir)

    checks = pd.DataFrame(rows)
    checks.to_csv(output_dir / "tensor_checks.csv", index=False)
    code_summary = sanitized_manifest.get("source_summaries", {}).get("code", {})
    summary = {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "checked_tensors": int(len(checks)),
        "failed_tensors": int((~checks["passed"]).sum()),
        "code_aliased_tensors": int(code_summary.get("aliased_tensors", 0)),
        "tensor_rule_count": len(sanitized_manifest.get("tensor_rules", [])),
        "tensor_alias_rule_count": len(sanitized_manifest.get("tensor_alias_rules", [])),
        "additive_delta_tensors": int(manifest.get("additive_delta_tensors", 0)),
        "additive_delta_values": int(manifest.get("additive_delta_values", 0)),
        "rule_counts": manifest.get("rule_counts", {}),
        "outputs": {
            "merge_manifest": rel(output_dir / "merge_manifest.json"),
            "report": rel(output_dir / "report.md"),
            "summary": rel(output_dir / "summary.json"),
            "tensor_checks": rel(output_dir / "tensor_checks.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# MoE Combined Writer Smoke",
        "",
        "这个 smoke 构造 tiny MoE-like checkpoint，并在一次真实 same-shape writer 调用里同时启用 expert tensor rules、source expert alias remap、freeze-router 和 router-bias additive delta。",
        "数值检查用 swapped code experts 验证 alias 确实先于 expert rule 生效；如果 alias 没有生效，expert 0/1 的 expected mean 会错。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Checked tensors: `{summary['checked_tensors']}`",
        f"- Failed tensors: `{summary['failed_tensors']}`",
        f"- Tensor rules: `{summary['tensor_rule_count']}`",
        f"- Alias rules: `{summary['tensor_alias_rule_count']}`",
        f"- Code aliased tensors: `{summary['code_aliased_tensors']}`",
        f"- Additive deltas: `{summary['additive_delta_values']}` values across `{summary['additive_delta_tensors']}` tensors",
        "",
        "## Files",
        "",
        f"- `{summary['outputs']['tensor_checks']}`",
        f"- `{summary['outputs']['merge_manifest']}`",
        f"- `{summary['outputs']['summary']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test combined MoE expert-rule, alias, and router-bias writer behavior.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_combined_writer_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(repo_path(args.output_dir))
    print(f"Wrote MoE combined writer smoke to {repo_path(args.output_dir).resolve()}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
