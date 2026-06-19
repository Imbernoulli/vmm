#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file


def resolve_safetensors(model_id_or_path: str) -> list[Path]:
    path = Path(model_id_or_path)
    if path.exists():
        if path.is_file() and path.name.endswith(".safetensors"):
            return [path]
        index = path / SAFE_WEIGHTS_INDEX_NAME
        single = path / SAFE_WEIGHTS_NAME
        if index.exists():
            payload = json.loads(index.read_text(encoding="utf-8"))
            return sorted({path / shard for shard in payload["weight_map"].values()})
        if single.exists():
            return [single]
        snapshots = list((path / "snapshots").glob("*/" + SAFE_WEIGHTS_INDEX_NAME))
        if snapshots:
            payload = json.loads(snapshots[0].read_text(encoding="utf-8"))
            return sorted({snapshots[0].parent / shard for shard in payload["weight_map"].values()})
        singles = list((path / "snapshots").glob("*/" + SAFE_WEIGHTS_NAME))
        if singles:
            return [singles[0]]
        raise FileNotFoundError(f"No safetensors weights found under {path}")

    try:
        index_file = cached_file(model_id_or_path, SAFE_WEIGHTS_INDEX_NAME)
        payload = json.loads(Path(index_file).read_text(encoding="utf-8"))
        return sorted({Path(index_file).parent / shard for shard in payload["weight_map"].values()})
    except Exception:
        single_file = cached_file(model_id_or_path, SAFE_WEIGHTS_NAME)
        return [Path(single_file)]


def load_float_state(model_id_or_path: str, max_tensors: int | None = None) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    shards = resolve_safetensors(model_id_or_path)
    for shard in tqdm(shards, desc=f"load {model_id_or_path}", leave=False):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                value = handle.get_tensor(name)
                if torch.is_floating_point(value):
                    state[name] = value.to(torch.float32).cpu()
                    if max_tensors is not None and len(state) >= max_tensors:
                        return state
    return state


def compatible_names(base: dict[str, torch.Tensor], expert: dict[str, torch.Tensor]) -> list[str]:
    return [
        name
        for name, value in base.items()
        if name in expert and value.shape == expert[name].shape and torch.is_floating_point(value)
    ]


def layer_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def summarize_delta(base: dict[str, torch.Tensor], expert: dict[str, torch.Tensor], expert_name: str) -> pd.DataFrame:
    rows = []
    for name in compatible_names(base, expert):
        delta = expert[name] - base[name]
        rows.append(
            {
                "expert": expert_name,
                "tensor": name,
                "group": layer_group(name),
                "numel": int(delta.numel()),
                "delta_norm": float(torch.linalg.norm(delta.reshape(-1))),
                "base_norm": float(torch.linalg.norm(base[name].reshape(-1))),
                "relative_norm": float(torch.linalg.norm(delta.reshape(-1)) / torch.linalg.norm(base[name].reshape(-1)).clamp_min(1e-12)),
                "mean_abs_delta": float(delta.abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def pairwise_conflict(
    base: dict[str, torch.Tensor],
    states: dict[str, dict[str, torch.Tensor]],
) -> pd.DataFrame:
    columns = [
        "left",
        "right",
        "group",
        "numel",
        "cosine",
        "sign_conflict",
        "weighted_conflict",
        "left_norm",
        "right_norm",
    ]
    rows = []
    for left, right in combinations(states.keys(), 2):
        common = sorted(set(compatible_names(base, states[left])).intersection(compatible_names(base, states[right])))
        grouped: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for name in common:
            grouped.setdefault(layer_group(name), []).append((states[left][name] - base[name], states[right][name] - base[name]))
        for group, chunks in grouped.items():
            a = torch.cat([chunk[0].reshape(-1) for chunk in chunks]).to(torch.float64)
            b = torch.cat([chunk[1].reshape(-1) for chunk in chunks]).to(torch.float64)
            denom = torch.linalg.norm(a) * torch.linalg.norm(b)
            cos = float((a @ b) / denom.clamp_min(1e-12))
            active = (a.abs() > 1e-10) & (b.abs() > 1e-10)
            if int(active.sum()) == 0:
                sign_conflict = 0.0
                weighted_conflict = 0.0
            else:
                conflict = torch.sign(a[active]) != torch.sign(b[active])
                sign_conflict = float(conflict.to(torch.float32).mean())
                weights = a[active].abs() * b[active].abs()
                weighted_conflict = float((weights * conflict.to(torch.float64)).sum() / weights.sum().clamp_min(1e-12))
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "group": group,
                    "numel": int(a.numel()),
                    "cosine": cos,
                    "sign_conflict": sign_conflict,
                    "weighted_conflict": weighted_conflict,
                    "left_norm": float(torch.linalg.norm(a)),
                    "right_norm": float(torch.linalg.norm(b)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def plot_pairwise(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    pairs = [f"{row.left} vs {row.right}" for row in df[["left", "right"]].drop_duplicates().itertuples()]
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(len(pairs), 1, figsize=(max(11, len(groups) * 0.28), 3.2 * len(pairs)), squeeze=False, constrained_layout=True)
    for ax, pair in zip(axes.ravel(), pairs, strict=True):
        left, right = pair.split(" vs ")
        sub = df[(df["left"] == left) & (df["right"] == right)].set_index("group").reindex(groups)
        matrix = sub[["cosine", "sign_conflict", "weighted_conflict"]].T.to_numpy(dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(pair)
        ax.set_yticks(range(3), labels=["cosine", "sign conflict", "weighted conflict"])
        ax.set_xticks(range(len(groups)), labels=groups, rotation=90, fontsize=6)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Qwen-compatible task-vector diagnostics without benchmark evaluation.")
    parser.add_argument("--base", required=True, help="Base model id or local path.")
    parser.add_argument("--expert", action="append", required=True, help="Expert spec NAME=MODEL_ID_OR_PATH. Repeat for multiple experts.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen_probe"))
    parser.add_argument("--max-tensors", type=int, default=None, help="Debug option to stop after N floating tensors per model.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_float_state(args.base, max_tensors=args.max_tensors)
    states: dict[str, dict[str, torch.Tensor]] = {}
    summaries = []
    for item in args.expert:
        if "=" not in item:
            raise ValueError("--expert must be NAME=MODEL_ID_OR_PATH")
        name, model = item.split("=", 1)
        state = load_float_state(model, max_tensors=args.max_tensors)
        names = compatible_names(base, state)
        if not names:
            raise ValueError(f"No compatible floating tensors between base and expert {name}")
        states[name] = state
        summaries.append(summarize_delta(base, state, name))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(args.output_dir / "delta_summary.csv", index=False)
    grouped = (
        summary_df.groupby(["expert", "group"], as_index=False)
        .agg(numel=("numel", "sum"), delta_norm=("delta_norm", "sum"), relative_norm=("relative_norm", "mean"), mean_abs_delta=("mean_abs_delta", "mean"))
        .sort_values(["expert", "group"])
    )
    grouped.to_csv(args.output_dir / "delta_summary_by_group.csv", index=False)

    pair_df = pairwise_conflict(base, states)
    pair_df.to_csv(args.output_dir / "pairwise_conflict.csv", index=False)
    plot_pairwise(pair_df, args.output_dir / "pairwise_conflict_heatmap.png")

    manifest = {
        "base": args.base,
        "experts": args.expert,
        "num_base_tensors": len(base),
        "num_experts": len(states),
        "max_tensors": args.max_tensors,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote Qwen/task-vector diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
