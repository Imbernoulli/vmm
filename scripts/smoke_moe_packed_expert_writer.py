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
                "model_type": "tiny_packed_moe_writer_smoke",
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


def packed_matrix(first: float, second: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.stack(
        [
            torch.full(shape, first, dtype=torch.float32),
            torch.full(shape, second, dtype=torch.float32),
        ],
        dim=0,
    )


def source_tensors(offset: float) -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.full((2, 2), offset + 40.0),
        "model.int_buffer": torch.tensor([1, 2], dtype=torch.int64),
        "model.layers.0.mlp.experts.down_proj.weight": packed_matrix(offset + 10.0, offset + 20.0, (2, 2)),
        "model.layers.0.mlp.experts.gate_up_proj": packed_matrix(offset + 30.0, offset + 40.0, (4, 2)),
        "model.layers.0.router.weight": torch.full((2, 2), offset + 50.0),
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), offset + 1.0),
    }


def build_tensor_rules(path: Path) -> None:
    path.write_text(".*self_attn.*::general=0.5,code=0.5\n", encoding="utf-8")


def build_packed_rules(path: Path) -> None:
    lines = [
        "tensor,output_expert,source,source_expert,weight,reason",
        "model.layers.0.mlp.experts.down_proj.weight,0,general,0,0.25,blend_same_general",
        "model.layers.0.mlp.experts.down_proj.weight,0,code,1,0.75,remap_code_expert_1_to_output_0",
        "model.layers.0.mlp.experts.down_proj.weight,1,general,1,1.0,keep_general_expert_1",
        "model.layers.0.mlp.experts.gate_up_proj,0,general,1,0.6,remap_general_expert_1_to_output_0",
        "model.layers.0.mlp.experts.gate_up_proj,0,code,0,0.4,blend_code_expert_0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_tensors(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    general = source_tensors(100.0)
    code = source_tensors(200.0)
    expected: dict[str, torch.Tensor] = {}
    down_name = "model.layers.0.mlp.experts.down_proj.weight"
    gate_name = "model.layers.0.mlp.experts.gate_up_proj"
    for name, base_tensor in base.items():
        if not torch.is_floating_point(base_tensor):
            expected[name] = base_tensor
        elif name == "model.layers.0.self_attn.q_proj.weight":
            expected[name] = base_tensor + 0.5 * (general[name] - base_tensor) + 0.5 * (code[name] - base_tensor)
        elif name == down_name:
            tensor = base_tensor.clone()
            tensor[0] = (
                base_tensor[0]
                + 0.25 * (general[name][0] - base_tensor[0])
                + 0.75 * (code[name][1] - base_tensor[0])
            )
            tensor[1] = base_tensor[1] + 1.0 * (general[name][1] - base_tensor[1])
            expected[name] = tensor
        elif name == gate_name:
            tensor = base_tensor.clone()
            tensor[0] = (
                base_tensor[0]
                + 0.6 * (general[name][1] - base_tensor[0])
                + 0.4 * (code[name][0] - base_tensor[0])
            )
            tensor[1] = base_tensor[1]
            expected[name] = tensor
        else:
            expected[name] = base_tensor
    return expected


def sanitize_manifest(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base"] = "TMP/base"
    manifest["sources"] = {"code": "TMP/code", "general": "TMP/general"}
    manifest["packed_expert_rule_csv"] = ["TMP/packed_expert_rules.csv"]
    for source_name, source_summary in manifest.get("source_summaries", {}).items():
        source_summary["root"] = f"TMP/{source_name}"
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_smoke(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moe_packed_expert_writer_") as tmp_raw:
        tmp = Path(tmp_raw)
        base_dir = tmp / "base"
        general_dir = tmp / "general"
        code_dir = tmp / "code"
        written_dir = tmp / "written"
        tensor_rule_file = tmp / "tensor_rules.txt"
        packed_rule_csv = tmp / "packed_expert_rules.csv"

        base = source_tensors(0.0)
        make_checkpoint(base_dir, base)
        make_checkpoint(general_dir, source_tensors(100.0))
        make_checkpoint(code_dir, source_tensors(200.0))
        build_tensor_rules(tensor_rule_file)
        build_packed_rules(packed_rule_csv)

        args = argparse.Namespace(
            allow_missing_source_tensors=False,
            base=str(base_dir),
            copy_metadata=True,
            dry_run=False,
            freeze_regex=[],
            freeze_router=True,
            output_dir=str(written_dir),
            output_dtype="base",
            packed_expert_rule_csv=[str(packed_rule_csv)],
            source=[f"general={general_dir}", f"code={code_dir}"],
            source_tensor_alias=[],
            source_tensor_alias_file=[],
            source_weight=["general=0.0", "code=0.0"],
            tensor_add_csv=[],
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
    summary = {
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "checked_tensors": int(len(checks)),
        "failed_tensors": int((~checks["passed"]).sum()),
        "packed_expert_rule_tensors": int(manifest.get("packed_expert_rule_tensors", 0)),
        "packed_expert_rule_slices": int(manifest.get("packed_expert_rule_slices", 0)),
        "packed_expert_rule_values": int(manifest.get("packed_expert_rule_values", 0)),
        "packed_expert_slice_rule_summary": sanitized_manifest.get("packed_expert_slice_rule_summary", {}),
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
        "# MoE Packed-Expert Writer Smoke",
        "",
        "这个 smoke 验证 same-shape writer 能处理真实 Qwen MoE 这种 packed expert tensor：输出 tensor 名字和 shape 不变，但第 0 维的 expert slice 可以按 CSV 指定的 source expert slice 与权重写入。",
        "测试里 `down_proj` 和 `gate_up_proj` 都是 `(num_experts, ...)` 形式；code/general 的 source expert index 被故意交叉，用逐张量数值检查验证 remap 和权重确实生效。",
        "",
        f"- Status: `{summary['status']}`",
        f"- Checked tensors: `{summary['checked_tensors']}`",
        f"- Failed tensors: `{summary['failed_tensors']}`",
        f"- Packed rule tensors: `{summary['packed_expert_rule_tensors']}`",
        f"- Packed rule slices: `{summary['packed_expert_rule_slices']}`",
        f"- Packed rule values: `{summary['packed_expert_rule_values']}`",
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
    parser = argparse.ArgumentParser(description="Smoke-test packed expert slice rules in the same-shape writer.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe_packed_expert_writer_smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(repo_path(args.output_dir))
    print(f"Wrote MoE packed-expert writer smoke to {repo_path(args.output_dir).resolve()}")
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
