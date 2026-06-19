#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


ROUTER_TENSOR_RE = re.compile(r"(^|\.)(router|gate)(\.|$)")
ROUTER_EXCLUDE_RE = re.compile(r"(gate_proj|shared_expert_gate)")
COPY_METADATA_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "generation_config.json",
)
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


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard: Path


@dataclass
class WeightIndex:
    root: Path
    tensor_to_file: dict[str, Path]
    tensor_info: dict[str, TensorInfo]
    shard_to_tensors: dict[Path, list[str]]
    has_index: bool


@dataclass(frozen=True)
class TensorRule:
    pattern: re.Pattern[str]
    raw_pattern: str
    weights: dict[str, float]


def discover_safetensors(model_path: str | Path) -> WeightIndex:
    path = Path(model_path)
    if path.is_file() and path.name.endswith(".safetensors"):
        return build_single_file_index(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected a local model path, got: {path}")
    if path.is_file():
        raise FileNotFoundError(f"Expected a safetensors file or model directory, got: {path}")

    index_path = path / SAFE_WEIGHTS_INDEX_NAME
    single_path = path / SAFE_WEIGHTS_NAME
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        tensor_to_file = {name: path / shard for name, shard in weight_map.items()}
        return build_index(path, tensor_to_file, has_index=True)
    if single_path.exists():
        return build_single_file_index(single_path)

    snapshots = sorted((path / "snapshots").glob(f"*/{SAFE_WEIGHTS_INDEX_NAME}"))
    if snapshots:
        snapshot_root = snapshots[0].parent
        payload = json.loads(snapshots[0].read_text(encoding="utf-8"))
        tensor_to_file = {name: snapshot_root / shard for name, shard in payload.get("weight_map", {}).items()}
        return build_index(snapshot_root, tensor_to_file, has_index=True)
    singles = sorted((path / "snapshots").glob(f"*/{SAFE_WEIGHTS_NAME}"))
    if singles:
        return build_single_file_index(singles[0])
    raise FileNotFoundError(f"No safetensors weights found under {path}")


def build_single_file_index(path: Path) -> WeightIndex:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor_to_file = {name: path for name in handle.keys()}
    return build_index(path.parent, tensor_to_file, has_index=False)


def build_index(root: Path, tensor_to_file: dict[str, Path], has_index: bool) -> WeightIndex:
    shard_to_tensors: dict[Path, list[str]] = {}
    tensor_info: dict[str, TensorInfo] = {}
    for name, shard in tensor_to_file.items():
        shard_to_tensors.setdefault(shard, []).append(name)

    for shard, names in shard_to_tensors.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in names:
                tensor_slice = handle.get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                if dtype_name not in SAFETENSORS_DTYPE_MAP:
                    raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} for tensor {name!r}")
                tensor_info[name] = TensorInfo(
                    shape=tuple(tensor_slice.get_shape()),
                    dtype=SAFETENSORS_DTYPE_MAP[dtype_name],
                    shard=shard,
                )
    return WeightIndex(
        root=root,
        tensor_to_file=tensor_to_file,
        tensor_info=tensor_info,
        shard_to_tensors={shard: sorted(names) for shard, names in shard_to_tensors.items()},
        has_index=has_index,
    )


def parse_name_value(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=VALUE, got: {raw}")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing name in: {raw}")
    return name, float(value)


def parse_source(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=MODEL_PATH, got: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing source name in: {raw}")
    return name, path


def parse_weight_map(raw_items: list[str] | None, source_names: list[str]) -> dict[str, float]:
    if not raw_items:
        weight = 1.0 / max(1, len(source_names))
        return {name: weight for name in source_names}
    weights: dict[str, float] = {}
    for item in raw_items:
        name, value = parse_name_value(item)
        if name not in source_names:
            raise ValueError(f"Weight provided for unknown source {name!r}; known sources: {source_names}")
        weights[name] = value
    for name in source_names:
        weights.setdefault(name, 0.0)
    return weights


def parse_rule_weights(raw: str, source_names: list[str]) -> dict[str, float]:
    weights = {name: 0.0 for name in source_names}
    if not raw.strip():
        return weights
    for chunk in raw.split(","):
        name, value = parse_name_value(chunk.strip())
        if name not in source_names:
            raise ValueError(f"Rule references unknown source {name!r}; known sources: {source_names}")
        weights[name] = value
    return weights


def parse_tensor_rules(raw_items: list[str] | None, source_names: list[str]) -> list[TensorRule]:
    rules: list[TensorRule] = []
    for item in raw_items or []:
        if "::" not in item:
            raise ValueError("Tensor rules must use PATTERN::SOURCE=WEIGHT,SOURCE=WEIGHT syntax")
        pattern, raw_weights = item.split("::", 1)
        rules.append(
            TensorRule(
                pattern=re.compile(pattern),
                raw_pattern=pattern,
                weights=parse_rule_weights(raw_weights, source_names),
            )
        )
    return rules


def output_dtype(base_dtype: torch.dtype, requested: str) -> torch.dtype:
    if requested == "base":
        return base_dtype
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def is_floating_dtype(dtype: torch.dtype) -> bool:
    return torch.is_floating_point(torch.empty((), dtype=dtype))


def is_router_tensor(name: str) -> bool:
    return bool(ROUTER_TENSOR_RE.search(name)) and not ROUTER_EXCLUDE_RE.search(name)


def choose_weights(
    name: str,
    default_weights: dict[str, float],
    tensor_rules: list[TensorRule],
    freeze_patterns: list[re.Pattern[str]],
    freeze_router: bool,
) -> tuple[dict[str, float], str]:
    for rule in tensor_rules:
        if rule.pattern.search(name):
            return rule.weights, f"tensor_rule:{rule.raw_pattern}"
    if any(pattern.search(name) for pattern in freeze_patterns):
        return {source: 0.0 for source in default_weights}, "freeze_regex"
    if freeze_router and is_router_tensor(name):
        return {source: 0.0 for source in default_weights}, "freeze_router"
    return default_weights, "default"


def validate_compatible(
    base: WeightIndex,
    sources: dict[str, WeightIndex],
    strict_names: bool,
) -> dict[str, Any]:
    base_names = set(base.tensor_info)
    errors: list[str] = []
    source_summaries: dict[str, Any] = {}
    for source_name, source_index in sources.items():
        source_names = set(source_index.tensor_info)
        missing = sorted(base_names - source_names)
        extra = sorted(source_names - base_names)
        shape_mismatches = []
        for name in sorted(base_names & source_names):
            if base.tensor_info[name].shape != source_index.tensor_info[name].shape:
                shape_mismatches.append(
                    {
                        "tensor": name,
                        "base_shape": base.tensor_info[name].shape,
                        "source_shape": source_index.tensor_info[name].shape,
                    }
                )
        if strict_names and missing:
            errors.append(f"{source_name}: missing {len(missing)} base tensors, first={missing[:3]}")
        if strict_names and extra:
            errors.append(f"{source_name}: has {len(extra)} extra tensors, first={extra[:3]}")
        if shape_mismatches:
            errors.append(f"{source_name}: {len(shape_mismatches)} shape mismatches, first={shape_mismatches[:3]}")
        source_summaries[source_name] = {
            "tensors": len(source_index.tensor_info),
            "missing_tensors": len(missing),
            "extra_tensors": len(extra),
            "shape_mismatches": len(shape_mismatches),
            "root": str(source_index.root),
        }
    if errors:
        raise ValueError("Input checkpoints are not same-shape compatible:\n" + "\n".join(errors))
    return source_summaries


def load_tensors(index: WeightIndex, names: list[str]) -> dict[str, torch.Tensor]:
    by_shard: dict[Path, list[str]] = {}
    for name in names:
        by_shard.setdefault(index.tensor_info[name].shard, []).append(name)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in shard_names:
                out[name] = handle.get_tensor(name)
    return out


def tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * torch.empty((), dtype=dtype).element_size()


def copy_metadata(base_root: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    if not base_root.is_dir():
        return copied
    for pattern in COPY_METADATA_PATTERNS:
        for src in base_root.glob(pattern):
            if not src.is_file() or src.suffix == ".safetensors":
                continue
            dst = output_dir / src.name
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)
            copied.append(src.name)
    return sorted(set(copied))


def write_index_file(
    output_dir: Path,
    weight_map: dict[str, str],
    base: WeightIndex,
    requested_dtype: str,
) -> None:
    total_size = 0
    for name, shard_name in weight_map.items():
        info = base.tensor_info[name]
        dtype = info.dtype if requested_dtype == "base" or not is_floating_dtype(info.dtype) else output_dtype(info.dtype, requested_dtype)
        total_size += tensor_nbytes(info.shape, dtype)
    payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_average_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    source_specs = [parse_source(item) for item in args.source]
    source_names = [name for name, _ in source_specs]
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"Source names must be unique: {source_names}")
    default_weights = parse_weight_map(args.source_weight, source_names)
    tensor_rules = parse_tensor_rules(args.tensor_rule, source_names)
    freeze_patterns = [re.compile(pattern) for pattern in args.freeze_regex]

    base = discover_safetensors(args.base)
    sources = {name: discover_safetensors(path) for name, path in source_specs}
    source_summaries = validate_compatible(base, sources, strict_names=not args.allow_missing_source_tensors)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_counts: dict[str, int] = {}
    floating_tensors = 0
    frozen_tensors = 0
    copied_tensors = 0
    output_weight_map: dict[str, str] = {}
    shard_summaries = []

    for shard_idx, (base_shard, names) in enumerate(tqdm(sorted(base.shard_to_tensors.items()), desc="write shards")):
        if args.dry_run:
            for name in names:
                info = base.tensor_info[name]
                if not is_floating_dtype(info.dtype):
                    copied_tensors += 1
                    continue
                weights, reason = choose_weights(
                    name,
                    default_weights=default_weights,
                    tensor_rules=tensor_rules,
                    freeze_patterns=freeze_patterns,
                    freeze_router=args.freeze_router,
                )
                rule_counts[reason] = rule_counts.get(reason, 0) + 1
                floating_tensors += 1
                if all(abs(value) <= 1e-12 for value in weights.values()):
                    frozen_tensors += 1
            shard_summaries.append({"shard": base_shard.name, "tensors": len(names)})
            continue

        base_values = load_tensors(base, names)
        needed_names = [name for name in names if torch.is_floating_point(base_values[name])]
        source_values = {source_name: load_tensors(index, needed_names) for source_name, index in sources.items()}
        output_tensors: dict[str, torch.Tensor] = {}
        shard_name = base_shard.name if base.has_index else SAFE_WEIGHTS_NAME
        if base.has_index and len(base.shard_to_tensors) > 1:
            shard_name = base_shard.name
        elif len(base.shard_to_tensors) > 1:
            shard_name = f"model-{shard_idx + 1:05d}-of-{len(base.shard_to_tensors):05d}.safetensors"

        for name in names:
            base_tensor = base_values[name]
            output_weight_map[name] = shard_name
            if not torch.is_floating_point(base_tensor):
                output_tensors[name] = base_tensor
                copied_tensors += 1
                continue
            weights, reason = choose_weights(
                name,
                default_weights=default_weights,
                tensor_rules=tensor_rules,
                freeze_patterns=freeze_patterns,
                freeze_router=args.freeze_router,
            )
            rule_counts[reason] = rule_counts.get(reason, 0) + 1
            floating_tensors += 1
            if all(abs(value) <= 1e-12 for value in weights.values()):
                output_tensors[name] = base_tensor.to(dtype=output_dtype(base_tensor.dtype, args.output_dtype))
                frozen_tensors += 1
                continue
            merged = base_tensor.to(torch.float32)
            for source_name, weight in weights.items():
                if abs(weight) <= 1e-12:
                    continue
                source_tensor = source_values[source_name].get(name)
                if source_tensor is None:
                    if args.allow_missing_source_tensors:
                        continue
                    raise KeyError(f"Missing tensor {name!r} in source {source_name!r}")
                merged = merged + float(weight) * (source_tensor.to(torch.float32) - base_tensor.to(torch.float32))
            output_tensors[name] = merged.to(dtype=output_dtype(base_tensor.dtype, args.output_dtype))

        if not args.dry_run:
            save_file(output_tensors, str(output_dir / shard_name), metadata={"format": "pt"})
        shard_summaries.append({"shard": shard_name, "tensors": len(names)})

    copied_metadata = [] if args.dry_run or not args.copy_metadata else copy_metadata(base.root, output_dir)
    if not args.dry_run and len(set(output_weight_map.values())) > 1:
        write_index_file(output_dir, output_weight_map, base, requested_dtype=args.output_dtype)

    manifest = {
        "schema_version": 1,
        "base": str(args.base),
        "sources": {name: path for name, path in source_specs},
        "source_summaries": source_summaries,
        "default_weights": default_weights,
        "tensor_rules": [{"pattern": rule.raw_pattern, "weights": rule.weights} for rule in tensor_rules],
        "freeze_regex": args.freeze_regex,
        "freeze_router": args.freeze_router,
        "output_dtype": args.output_dtype,
        "dry_run": args.dry_run,
        "floating_tensors": floating_tensors,
        "frozen_tensors": frozen_tensors,
        "copied_nonfloating_tensors": copied_tensors,
        "rule_counts": rule_counts,
        "shards": shard_summaries,
        "copied_metadata": copied_metadata,
        "same_shape_constraint": "All output tensors keep base tensor names and shapes; metadata is copied from base when available.",
    }
    (output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a same-shape averaged safetensors checkpoint from compatible local checkpoints.")
    parser.add_argument("--base", required=True, help="Anchor/base model directory or safetensors file.")
    parser.add_argument("--source", action="append", required=True, help="Source spec NAME=MODEL_DIR_OR_SAFETENSORS. Repeatable.")
    parser.add_argument("--source-weight", action="append", default=None, help="Default source delta weight NAME=FLOAT. Defaults to uniform.")
    parser.add_argument(
        "--tensor-rule",
        action="append",
        default=None,
        help="Regex-specific weights: PATTERN::SOURCE=WEIGHT,SOURCE=WEIGHT. First matching rule wins.",
    )
    parser.add_argument("--freeze-regex", action="append", default=[], help="Freeze matching tensors to base weights.")
    parser.add_argument("--freeze-router", action="store_true", help="Freeze router/gate tensors, excluding gate_proj/shared_expert_gate.")
    parser.add_argument("--allow-missing-source-tensors", action="store_true", help="Keep base tensors when a source tensor is missing.")
    parser.add_argument("--output-dtype", choices=["base", "float32", "float16", "bfloat16"], default="base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-copy-metadata", dest="copy_metadata", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write merge_manifest.json without writing weights.")
    parser.set_defaults(copy_metadata=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_average_checkpoint(args)
    print(f"Wrote merge manifest to {Path(args.output_dir).resolve() / 'merge_manifest.json'}")
    if args.dry_run:
        print("Dry run only; no weight shards were written.")
    else:
        print(f"Wrote same-shape checkpoint shards: {len(manifest['shards'])}")


if __name__ == "__main__":
    main()
