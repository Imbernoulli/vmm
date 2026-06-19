#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
EXPERT_RE = re.compile(r"(?:^|\.)(?:experts|local_experts)\.(\d+)(?:\.|$)")
PACKED_EXPERT_RE = re.compile(r"(?:^|\.)mlp\.experts\.(?:gate_up_proj|down_proj)(?:\.|$)")


@dataclass(frozen=True)
class TensorHeader:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard: str

    @property
    def numel(self) -> int:
        out = 1
        for dim in self.shape:
            out *= int(dim)
        return int(out)

    @property
    def bytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = repo_path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_model_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = repo_path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    return name.strip(), repo_path(path.strip())


def find_config(path: Path) -> Path | None:
    if path.is_file() and path.name == "config.json":
        return path
    if path.is_dir():
        direct = path / "config.json"
        if direct.exists():
            return direct
        snapshots = sorted(path.glob("snapshots/*/config.json"))
        if snapshots:
            return snapshots[0]
    return None


def load_config(path: Path) -> dict[str, Any] | None:
    config_path = find_config(path)
    if config_path is None:
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def text_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def config_summary(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = text_config(config)
    root = config or {}
    num_experts = cfg.get("num_experts")
    experts_per_tok = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token")
    active_fraction = None
    if isinstance(num_experts, int) and num_experts > 0 and isinstance(experts_per_tok, int):
        active_fraction = experts_per_tok / num_experts
    layer_types = cfg.get("layer_types") if isinstance(cfg.get("layer_types"), list) else []
    return {
        "model_type": root.get("model_type") or cfg.get("model_type"),
        "architectures": root.get("architectures"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "intermediate_size": cfg.get("intermediate_size"),
        "moe_intermediate_size": cfg.get("moe_intermediate_size"),
        "shared_expert_intermediate_size": cfg.get("shared_expert_intermediate_size"),
        "num_experts": num_experts,
        "num_experts_per_tok": experts_per_tok,
        "active_expert_fraction_per_token": active_fraction,
        "router_aux_loss_coef": cfg.get("router_aux_loss_coef"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "layer_type_counts": to_jsonable(dict(pd.Series(layer_types).value_counts())) if layer_types else {},
        "is_moe_config": bool(num_experts or cfg.get("moe_intermediate_size") or "moe" in str(root.get("model_type", "")).lower()),
    }


def discover_safetensors(path: Path) -> tuple[Path, dict[str, Path], bool] | None:
    if path.is_file() and path.name.endswith(".safetensors"):
        headers = tensor_names_in_shard(path)
        return path.parent, {name: path for name in headers}, False
    if not path.exists() or not path.is_dir():
        return None
    direct_index = path / SAFE_WEIGHTS_INDEX_NAME
    direct_single = path / SAFE_WEIGHTS_NAME
    if direct_index.exists():
        return path, read_index(path, direct_index), True
    if direct_single.exists():
        return path, {name: direct_single for name in tensor_names_in_shard(direct_single)}, False
    snapshots = sorted(path.glob(f"snapshots/*/{SAFE_WEIGHTS_INDEX_NAME}"))
    if snapshots:
        root = snapshots[0].parent
        return root, read_index(root, snapshots[0]), True
    singles = sorted(path.glob(f"snapshots/*/{SAFE_WEIGHTS_NAME}"))
    if singles:
        return singles[0].parent, {name: singles[0] for name in tensor_names_in_shard(singles[0])}, False
    return None


def read_index(root: Path, index_path: Path) -> dict[str, Path]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {name: root / shard for name, shard in payload.get("weight_map", {}).items()}


def tensor_names_in_shard(shard: Path) -> list[str]:
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def read_headers(path: Path) -> tuple[Path | None, list[TensorHeader], bool]:
    discovered = discover_safetensors(path)
    if discovered is None:
        return None, [], False
    root, tensor_to_shard, has_index = discovered
    by_shard: dict[Path, list[str]] = {}
    for name, shard in tensor_to_shard.items():
        by_shard.setdefault(shard, []).append(name)
    headers: list[TensorHeader] = []
    for shard, names in tqdm(sorted(by_shard.items()), desc=f"headers {path.name}", leave=False):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor_slice = handle.get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                dtype = SAFETENSORS_DTYPE_MAP.get(dtype_name)
                if dtype is None:
                    raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} for tensor {name!r}")
                headers.append(
                    TensorHeader(
                        name=name,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype=dtype,
                        shard=shard.name,
                    )
                )
    return root, headers, has_index


def tensor_group(name: str) -> str:
    lowered = name.lower()
    if "embed_tokens" in lowered or "wte" in lowered:
        return "embedding"
    if "lm_head" in lowered:
        return "lm_head"
    if re.search(r"(^|\.)(router|gate)(\.|$)", name) and not re.search(r"(gate_proj|shared_expert_gate)", name):
        return "router"
    if "shared_expert" in lowered:
        return "shared_expert"
    if EXPERT_RE.search(name) or PACKED_EXPERT_RE.search(name):
        return "routed_expert"
    if "norm" in lowered:
        return "norm"
    if "self_attn" in lowered or ".attn." in lowered or "attention" in lowered or "linear_attn" in lowered:
        return "attention"
    if "mlp" in lowered or "ffn" in lowered or "feed_forward" in lowered:
        return "dense_mlp"
    return "other"


def layer_id(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def expert_id(name: str) -> int | None:
    match = EXPERT_RE.search(name)
    return int(match.group(1)) if match else None


def packed_expert_count(name: str, shape: tuple[int, ...]) -> int | None:
    if not PACKED_EXPERT_RE.search(name) or not shape:
        return None
    return int(shape[0])


def summarize_headers(headers: list[TensorHeader]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for header in headers:
        rows.append(
            {
                "tensor": header.name,
                "group": tensor_group(header.name),
                "layer": layer_id(header.name),
                "expert": expert_id(header.name),
                "packed_expert_count": packed_expert_count(header.name, header.shape),
                "shape": "x".join(str(dim) for dim in header.shape),
                "dtype": str(header.dtype).replace("torch.", ""),
                "numel": header.numel,
                "bytes": header.bytes,
                "shard": header.shard,
            }
        )
    tensors = pd.DataFrame(rows)
    if tensors.empty:
        return tensors, pd.DataFrame(), pd.DataFrame(), {"weights_available": False}
    group_summary = (
        tensors.groupby("group", as_index=False)
        .agg(tensors=("tensor", "count"), numel=("numel", "sum"), bytes=("bytes", "sum"))
        .sort_values("bytes", ascending=False)
    )
    layer_summary = (
        tensors.dropna(subset=["layer"])
        .groupby(["layer", "group"], as_index=False)
        .agg(tensors=("tensor", "count"), numel=("numel", "sum"), bytes=("bytes", "sum"))
        .sort_values(["layer", "group"])
    )
    routed_rows = tensors[tensors["group"] == "routed_expert"].copy()
    if routed_rows.empty:
        expert_summary = pd.DataFrame(columns=["layer", "expert", "packed_expert_count", "tensors", "numel", "bytes"])
    elif routed_rows["expert"].notna().any():
        expert_summary = (
            routed_rows.groupby(["layer", "expert"], dropna=False, as_index=False)
            .agg(
                packed_expert_count=("packed_expert_count", "max"),
                tensors=("tensor", "count"),
                numel=("numel", "sum"),
                bytes=("bytes", "sum"),
            )
            .sort_values(["layer", "expert"])
        )
    else:
        expert_summary = (
            routed_rows.groupby(["layer"], dropna=False, as_index=False)
            .agg(
                packed_expert_count=("packed_expert_count", "max"),
                tensors=("tensor", "count"),
                numel=("numel", "sum"),
                bytes=("bytes", "sum"),
            )
            .sort_values(["layer"])
        )
        expert_summary["expert"] = "packed"
        expert_summary = expert_summary[["layer", "expert", "packed_expert_count", "tensors", "numel", "bytes"]]
    explicit_experts = tensors["expert"].dropna().nunique() if "expert" in tensors else 0
    packed_experts = tensors["packed_expert_count"].dropna().max() if "packed_expert_count" in tensors else None
    num_experts_with_weights = int(explicit_experts) if explicit_experts else (int(packed_experts) if pd.notna(packed_experts) else 0)
    compact = {
        "weights_available": True,
        "total_tensors": int(len(tensors)),
        "total_numel": int(tensors["numel"].sum()),
        "total_bytes": int(tensors["bytes"].sum()),
        "groups": {
            row["group"]: {
                "tensors": int(row["tensors"]),
                "numel": int(row["numel"]),
                "bytes": int(row["bytes"]),
            }
            for _, row in group_summary.iterrows()
        },
        "num_layers_with_weights": int(tensors["layer"].dropna().nunique()) if "layer" in tensors else 0,
        "num_experts_with_weights": num_experts_with_weights,
        "packed_expert_tensor_count": int(tensors["packed_expert_count"].notna().sum()) if "packed_expert_count" in tensors else 0,
    }
    return tensors, group_summary, expert_summary, compact


def inspect_model(name: str, path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(path)
    cfg_summary = config_summary(config)
    config_path = find_config(path)
    weights_root, headers, has_index = read_headers(path)
    tensor_df, group_df, expert_df, header_summary = summarize_headers(headers)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if not tensor_df.empty:
        tensor_df.to_csv(output_dir / f"{safe_name}_tensors.csv", index=False)
        group_df.to_csv(output_dir / f"{safe_name}_groups.csv", index=False)
        expert_df.to_csv(output_dir / f"{safe_name}_experts.csv", index=False)

    return {
        "name": name,
        "path": str(path),
        "config_path": str(config_path) if config_path else None,
        "weights_root": str(weights_root) if weights_root else None,
        "has_safetensors_index": has_index,
        "config": cfg_summary,
        "headers": header_summary,
        "artifacts": {
            "tensors_csv": rel(output_dir / f"{safe_name}_tensors.csv") if not tensor_df.empty else None,
            "groups_csv": rel(output_dir / f"{safe_name}_groups.csv") if not group_df.empty else None,
            "experts_csv": rel(output_dir / f"{safe_name}_experts.csv") if not expert_df.empty else None,
        },
    }


def compare_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left_idx, left in enumerate(models):
        for right in models[left_idx + 1 :]:
            left_cfg = left["config"]
            right_cfg = right["config"]
            fields = [
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "num_experts",
                "num_experts_per_tok",
                "vocab_size",
            ]
            mismatches = [field for field in fields if left_cfg.get(field) != right_cfg.get(field)]
            rows.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "same_config_for_average": len(mismatches) == 0,
                    "mismatched_fields": ",".join(mismatches),
                    "left_weights_available": left["headers"].get("weights_available", False),
                    "right_weights_available": right["headers"].get("weights_available", False),
                }
            )
    return rows


def build_report(models: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# Checkpoint Topology Inspect",
        "",
        "这个报告只读 config 和 safetensors header，不加载权重内容。它用于 Average 前的第一层同构检查：模型是否是 MoE、专家数/激活专家数是多少、router/expert/shared 参数是否可被分组处理。",
        "",
        "## Models",
        "",
        "| model | model_type | layers | hidden | experts | active/top-k | active fraction | weights | total bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for model in models:
        cfg = model["config"]
        headers = model["headers"]
        lines.append(
            "| {name} | {model_type} | {layers} | {hidden} | {experts} | {topk} | {frac} | {weights} | {bytes} |".format(
                name=model["name"],
                model_type=cfg.get("model_type"),
                layers=cfg.get("num_hidden_layers"),
                hidden=cfg.get("hidden_size"),
                experts=cfg.get("num_experts"),
                topk=cfg.get("num_experts_per_tok"),
                frac=f"{cfg.get('active_expert_fraction_per_token'):.4f}" if cfg.get("active_expert_fraction_per_token") is not None else "n/a",
                weights="yes" if headers.get("weights_available") else "no",
                bytes=headers.get("total_bytes", "n/a"),
            )
        )
    lines.extend(["", "## Average-Relevant Notes", ""])
    for model in models:
        cfg = model["config"]
        lines.append(f"### {model['name']}")
        if cfg.get("is_moe_config"):
            lines.append(
                f"- MoE config: `{cfg.get('num_experts')}` experts, `{cfg.get('num_experts_per_tok')}` active per token; active fraction `{cfg.get('active_expert_fraction_per_token')}`."
            )
            lines.append("- Average implication: router、shared modules、routed experts 必须分组处理；不能把 router 当普通 dense 层同权平均。")
        else:
            lines.append("- Dense config: 没有 MoE expert/router 字段；可走 dense average / task-vector / layer-wise coefficient 路线。")
        if model["headers"].get("weights_available"):
            groups = model["headers"].get("groups", {})
            for group in ("router", "routed_expert", "shared_expert", "attention", "dense_mlp", "embedding", "lm_head"):
                if group in groups:
                    lines.append(f"- `{group}`: tensors `{groups[group]['tensors']}`, bytes `{groups[group]['bytes']}`.")
        else:
            lines.append("- 本地没有 safetensors 权重 shard；当前只完成 config-level topology probe。")
        lines.append("")
    if comparisons:
        lines.extend(["## Pairwise Config Compatibility", "", "| left | right | same config | mismatched fields |", "| --- | --- | --- | --- |"])
        for row in comparisons:
            lines.append(
                f"| {row['left']} | {row['right']} | {row['same_config_for_average']} | {row['mismatched_fields'] or 'none'} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`",
            "- `*_groups.csv` when safetensors headers are available",
            "- `*_experts.csv` when expert tensors are visible in headers",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect checkpoint topology from config and safetensors headers without loading weights.")
    parser.add_argument("--model", action="append", required=True, help="Model spec NAME=PATH or PATH. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/checkpoint_topology_inspect"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [inspect_model(name, path, output_dir) for name, path in (parse_model_spec(raw) for raw in args.model)]
    comparisons = compare_models(models)
    summary = to_jsonable({"models": models, "comparisons": comparisons})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if comparisons:
        pd.DataFrame(comparisons).to_csv(output_dir / "compatibility.csv", index=False)
    (output_dir / "report.md").write_text(build_report(models, comparisons), encoding="utf-8")
    print(f"Wrote checkpoint topology inspect to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
