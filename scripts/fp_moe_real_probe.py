"""First-principles experiment 3: the expert-gauge problem on REAL MoE LLMs.

Two questions, two modes:

  mode=gauge_selfmerge  (one model)
     A MoE layer is invariant to jointly permuting (experts, router rows).
     We build M' = gauge-permuted copy of a real MoE LLM, verify M' computes the
     SAME logits as M, then show that naive index-wise averaging of M and M'
     (two *functionally identical* models) is catastrophic, while recovering the
     permutation (functional/weight alignment) and averaging restores M exactly.
     => proves "same tensor name" is not a stable expert identity on real LLMs.

  mode=cross_correspondence  (two finetunes + shared base)
     The open empirical question: do two independently fine-tuned MoE LLMs keep
     their experts index-aligned, or do experts drift/permute relative to each
     other?  For each layer we compute the cross-model expert correspondence of
     base-subtracted expert deltas, run Hungarian assignment, and report how
     often identity is optimal and how diagonal the correspondence is.
     => decides whether real MoE merging must solve expert alignment first.

Works on unpacked and packed HF MoEs (OlmoeForCausalLM, Qwen3MoeForCausalLM):
   experts at  {prefix}.experts.{i}.{gate,up,down}_proj.weight
   or packed at {prefix}.experts.{gate_up,down}_proj with expert dimension 0
   router at   {prefix}.gate.weight    (rows index experts)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def log(m):
    print(m, flush=True)


EXPERT_RE = re.compile(r"^(.*)\.experts\.(\d+)\.(.*)$")
PACKED_EXPERT_RE = re.compile(r"^(.*)\.experts\.([^.]*(?:proj|proj\.weight))$")


def _keys_and_shapes(source):
    if isinstance(source, dict):
        return source.keys(), {k: tuple(v.shape) for k, v in source.items() if hasattr(v, "shape")}
    return source, {}


def find_moe_layers(source):
    """Return {prefix: n_experts} and the router key per prefix."""
    keys, shapes = _keys_and_shapes(source)
    experts = defaultdict(set)
    subkeys = defaultdict(set)
    packed = defaultdict(set)
    for k in keys:
        m = EXPERT_RE.match(k)
        if m:
            prefix, idx, sub = m.group(1), int(m.group(2)), m.group(3)
            experts[prefix].add(idx)
            subkeys[prefix].add(sub)
            continue
        m = PACKED_EXPERT_RE.match(k)
        if m:
            prefix, sub = m.group(1), m.group(2)
            # Packed state_dict keys are usually e.g. experts.gate_up_proj
            # with shape [n_experts, ...].  Header-only keys may not expose
            # shape, so n_experts can be filled from router rows later.
            if re.search(r"\b(experts|mlp)\b", prefix):
                packed[prefix].add(sub)
    layers = {}
    for prefix, idxs in experts.items():
        layers[prefix] = {
            "format": "unpacked",
            "n_experts": max(idxs) + 1,
            "subkeys": sorted(subkeys[prefix]),
            "router": prefix + ".gate.weight",
        }
    for prefix, subs in packed.items():
        if prefix in layers:
            continue
        n_experts = None
        for sub in subs:
            shape = shapes.get(f"{prefix}.experts.{sub}")
            if shape:
                n_experts = shape[0]
                break
        router = prefix + ".gate.weight"
        if router in shapes:
            n_experts = shapes[router][0]
        layers[prefix] = {
            "format": "packed",
            "n_experts": n_experts,
            "subkeys": sorted(subs),
            "router": router,
        }
    return layers


# ----------------------------------------------------------------------------------
# safetensors streaming (memory-light)
# ----------------------------------------------------------------------------------
def snapshot_dir(model_id):
    if os.path.isdir(model_id):
        return model_id
    base = Path.home() / ".cache/huggingface/hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    snaps = sorted(base.glob("*"))
    if not snaps:
        raise FileNotFoundError(f"no snapshot for {model_id}")
    return str(snaps[-1])


def build_weight_index(sdir):
    """name -> shard file path (handles single-shard and sharded)."""
    idx = Path(sdir) / "model.safetensors.index.json"
    if idx.exists():
        wm = json.load(open(idx))["weight_map"]
        return {n: str(Path(sdir) / f) for n, f in wm.items()}
    single = Path(sdir) / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as f:
            return {n: str(single) for n in f.keys()}
    raise FileNotFoundError(f"no safetensors in {sdir}")


class ShardReader:
    def __init__(self, index):
        self.index = index
        self._open = {}

    def get(self, name):
        from safetensors import safe_open

        path = self.index[name]
        if path not in self._open:
            self._open[path] = safe_open(path, framework="pt")
        return self._open[path].get_tensor(name)


def get_tensor(source, name):
    if isinstance(source, dict):
        return source[name]
    return source.get(name)


def expert_tensor(source, prefix, info, i, sub):
    if info["format"] == "packed":
        return get_tensor(source, f"{prefix}.experts.{sub}")[i]
    return get_tensor(source, f"{prefix}.experts.{i}.{sub}")


def set_expert_tensor(target, source, prefix, info, dst_i, src_i, sub):
    if info["format"] == "packed":
        name = f"{prefix}.experts.{sub}"
        target[name][dst_i] = source[name][src_i].clone()
    else:
        target[f"{prefix}.experts.{dst_i}.{sub}"] = source[f"{prefix}.experts.{src_i}.{sub}"].clone()


def expert_delta_vec(reader, base_reader, prefix, info, i):
    parts = []
    for sub in info["subkeys"]:
        w = expert_tensor(reader, prefix, info, i, sub).to(torch.float32).reshape(-1)
        if base_reader is not None:
            w = w - expert_tensor(base_reader, prefix, info, i, sub).to(torch.float32).reshape(-1)
        parts.append(w)
    return torch.cat(parts)


def hungarian(C):
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-C.numpy())
        perm = [0] * len(r)
        for i, j in zip(r, c):
            perm[i] = int(j)
        return perm
    except Exception:
        E = C.shape[0]
        used, perm = set(), [0] * E
        for i in range(E):
            order = torch.argsort(C[i], descending=True).tolist()
            for j in order:
                if j not in used:
                    perm[i] = j
                    used.add(j)
                    break
        return perm


# ----------------------------------------------------------------------------------
# mode: cross_correspondence
# ----------------------------------------------------------------------------------
def cross_correspondence(args):
    s1, s2 = snapshot_dir(args.model_a), snapshot_dir(args.model_b)
    r1 = ShardReader(build_weight_index(s1))
    r2 = ShardReader(build_weight_index(s2))
    rb = ShardReader(build_weight_index(snapshot_dir(args.base))) if args.base else None
    layers = find_moe_layers(r1.index.keys())
    log(f"found {len(layers)} MoE layers; base-subtract={rb is not None}")

    # order layers numerically
    def lnum(p):
        m = re.search(r"layers\.(\d+)", p)
        return int(m.group(1)) if m else 0

    prefixes = sorted(layers, key=lnum)
    if args.max_layers:
        prefixes = prefixes[: args.max_layers]

    per_layer = []
    for pi, prefix in enumerate(prefixes):
        info = layers[prefix]
        E = info["n_experts"]
        if E is None:
            E = int(r1.get(info["router"]).shape[0])
            info["n_experts"] = E
        D1 = torch.stack([expert_delta_vec(r1, rb, prefix, info, i) for i in range(E)])
        D2 = torch.stack([expert_delta_vec(r2, rb, prefix, info, i) for i in range(E)])
        D1 = D1 / D1.norm(dim=1, keepdim=True).clamp_min(1e-12)
        D2 = D2 / D2.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = D1 @ D2.t()  # [E,E] cosine
        perm = hungarian(C)
        ident = list(range(E))
        diag = torch.diagonal(C)
        argmax_match = (C.argmax(dim=1) == torch.arange(E)).float().mean().item()
        per_layer.append({
            "prefix": prefix,
            "n_experts": E,
            "identity_optimal": perm == ident,
            "hamming_to_identity": sum(1 for i in range(E) if perm[i] != i),
            "diag_mean_cos": float(diag.mean()),
            "offdiag_mean_cos": float((C.sum() - diag.sum()) / (E * E - E)),
            "argmax_is_identity_frac": argmax_match,
            "matched_mean_cos": float(torch.tensor([C[i, perm[i]] for i in range(E)]).mean()),
        })
        log(f"  [{prefix}] E={E} identity_opt={perm==ident} "
            f"argmax_id_frac={argmax_match:.3f} diag_cos={float(diag.mean()):.3f} "
            f"matched_cos={per_layer[-1]['matched_mean_cos']:.3f} ham={per_layer[-1]['hamming_to_identity']}")

    n = len(per_layer)
    summary = {
        "model_a": args.model_a, "model_b": args.model_b, "base": args.base,
        "n_layers": n,
        "frac_layers_identity_optimal": sum(p["identity_optimal"] for p in per_layer) / max(n, 1),
        "mean_argmax_is_identity_frac": sum(p["argmax_is_identity_frac"] for p in per_layer) / max(n, 1),
        "mean_diag_cos": sum(p["diag_mean_cos"] for p in per_layer) / max(n, 1),
        "mean_offdiag_cos": sum(p["offdiag_mean_cos"] for p in per_layer) / max(n, 1),
        "mean_matched_cos": sum(p["matched_mean_cos"] for p in per_layer) / max(n, 1),
        "per_layer": per_layer,
    }
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cross_correspondence.json").write_text(json.dumps(summary, indent=2))
    log("\n=== CROSS-CORRESPONDENCE SUMMARY ===")
    log(f"frac layers identity-optimal      : {summary['frac_layers_identity_optimal']:.3f}")
    log(f"mean argmax-is-identity fraction  : {summary['mean_argmax_is_identity_frac']:.3f}")
    log(f"mean diagonal cos (expert i<->i)  : {summary['mean_diag_cos']:.3f}")
    log(f"mean off-diagonal cos             : {summary['mean_offdiag_cos']:.3f}")
    log(f"mean matched (Hungarian) cos      : {summary['mean_matched_cos']:.3f}")
    log(f"wrote {outdir/'cross_correspondence.json'}")


# ----------------------------------------------------------------------------------
# mode: gauge_selfmerge
# ----------------------------------------------------------------------------------
def get_probe_batches(tok, n_seq, seqlen):
    # The gauge-selfmerge test only needs deterministic token inputs.  Avoid
    # datasets.load_dataset here because sandboxed runs may not be able to write
    # Hugging Face cache lock files even when the dataset is already cached.
    seed_text = """
    Model averaging is a weight-space operation.  A dense model has one copy of
    each layer, while a mixture-of-experts model has routed expert functions and
    a router whose rows define the expert identity used by the forward pass.
    If expert rows and router rows are permuted together, the MoE computes the
    same function, but same-name parameter averaging no longer respects expert
    identity.  This probe checks that gauge symmetry on a real MoE checkpoint.

    We evaluate negative log likelihood on this deterministic text.  The exact
    content is not important; the comparison is within the same input batch for
    baseline, gauge-permuted, naive same-name average, and aligned average.
    """
    text = "\n".join(seed_text.strip() for _ in range(max(8, n_seq * 4)))
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for i in range(0, max(0, len(ids) - seqlen), seqlen):
        out.append(ids[i : i + seqlen])
        if len(out) >= n_seq:
            break
    if not out:
        out = [ids[:seqlen]]
    return out


@torch.no_grad()
def nll(model, batches, device):
    tot, ntok = 0.0, 0
    lf = torch.nn.CrossEntropyLoss(reduction="sum")
    for ids in batches:
        ids = ids.unsqueeze(0).to(device)
        lg = model(ids).logits[0]
        tot += float(lf(lg[:-1].float(), ids[0, 1:]))
        ntok += ids.shape[1] - 1
    return tot / max(ntok, 1)


def moe_blocks_from_sd(sd):
    """Detect MoE blocks in an in-memory state_dict, handling BOTH the packed
    format (experts.gate_up_proj as a [E,...] 3D tensor) and the unpacked format
    (experts.{i}.{sub}).  Returns {prefix: {packed:[keys], unpacked:{sub:{i:key}},
    E, router}}."""
    blocks = {}
    for k, v in sd.items():
        mp = re.match(r"^(.*)\.experts\.([A-Za-z_]+)$", k)  # packed: no digit
        if mp and v.ndim == 3:
            prefix = mp.group(1)
            b = blocks.setdefault(prefix, {"packed": [], "unpacked": defaultdict(dict), "E": 0})
            b["packed"].append(k)
            b["E"] = max(b["E"], v.shape[0])
            continue
        mu = EXPERT_RE.match(k)  # unpacked: experts.{i}.{sub}
        if mu:
            prefix, idx, sub = mu.group(1), int(mu.group(2)), mu.group(3)
            b = blocks.setdefault(prefix, {"packed": [], "unpacked": defaultdict(dict), "E": 0})
            b["unpacked"][sub][idx] = k
            b["E"] = max(b["E"], idx + 1)
    for prefix, b in blocks.items():
        rk = prefix + ".gate.weight"
        b["router"] = rk if rk in sd else None
    return blocks


def expert_vecs(sd, prefix, b):
    """Per-expert flattened weight vectors [E, D] for matching (packed or unpacked)."""
    E = b["E"]
    rows = [[] for _ in range(E)]
    if b["packed"]:
        for pk in sorted(b["packed"]):
            t = sd[pk].float()
            for i in range(E):
                rows[i].append(t[i].reshape(-1))
    else:
        for sub in sorted(b["unpacked"]):
            for i in range(E):
                rows[i].append(sd[b["unpacked"][sub][i]].float().reshape(-1))
    return torch.stack([torch.cat(r) for r in rows])


def apply_expert_perm(sd_dst, sd_src, prefix, b, perm):
    """new expert i <- src expert perm[i] (packed dim-0 index or per-key swap),
    and permute router rows by perm."""
    E = b["E"]
    if b["packed"]:
        idx = torch.tensor(perm)
        for pk in b["packed"]:
            sd_dst[pk] = sd_src[pk].index_select(0, idx).clone()
    else:
        for sub, idxmap in b["unpacked"].items():
            for i in range(E):
                sd_dst[idxmap[i]] = sd_src[idxmap[perm[i]]].clone()
    if b["router"] is not None:
        sd_dst[b["router"]] = torch.stack([sd_src[b["router"]][perm[i]] for i in range(E)])


def gauge_selfmerge(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sdir = snapshot_dir(args.model_a)
    tok = AutoTokenizer.from_pretrained(sdir, local_files_only=True)
    log(f"loading {args.model_a} ...")
    model = AutoModelForCausalLM.from_pretrained(sdir, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    batches = get_probe_batches(tok, args.n_probe, args.seqlen)

    sd = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    blocks = moe_blocks_from_sd(sd)
    fmt = "packed" if any(b["packed"] for b in blocks.values()) else "unpacked"
    log(f"{len(blocks)} MoE blocks  format={fmt}")
    if not blocks:
        raise RuntimeError("No MoE blocks found in loaded state_dict; inspect state_dict key names before running self-merge.")
    base_nll = nll(model, batches, device)
    log(f"baseline NLL = {base_nll:.4f}")

    # ---- build gauge-permuted copy ----
    g = torch.Generator().manual_seed(args.seed)
    perms = {}
    sd_perm = OrderedDict((k, v.clone()) for k, v in sd.items())
    for prefix, b in blocks.items():
        perm = torch.randperm(b["E"], generator=g).tolist()
        perms[prefix] = perm
        apply_expert_perm(sd_perm, sd, prefix, b, perm)

    model.load_state_dict(sd_perm, strict=True)
    perm_nll = nll(model, batches, device)
    log(f"gauge-permuted copy NLL = {perm_nll:.4f}  (should ~= baseline)")

    # ---- naive index-wise average of M and gauge(M) ----
    sd_avg = OrderedDict((k, ((sd[k].float() + sd_perm[k].float()) / 2).to(sd[k].dtype)) for k in sd)
    model.load_state_dict(sd_avg, strict=True)
    naive_nll = nll(model, batches, device)
    log(f"NAIVE same-name average NLL = {naive_nll:.4f}")

    # ---- recover permutation by weight matching, then aligned average ----
    recovered_ok = 0
    sd_perm_aligned = OrderedDict((k, v.clone()) for k, v in sd_perm.items())
    for prefix, b in blocks.items():
        E = b["E"]
        Dref = expert_vecs(sd, prefix, b)
        Dp = expert_vecs(sd_perm, prefix, b)
        Dref = Dref / Dref.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Dp = Dp / Dp.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Dref @ Dp.t()
        inv = hungarian(C)  # inv[i] = which permuted-expert matches reference-expert i
        apply_expert_perm(sd_perm_aligned, sd_perm, prefix, b, inv)
        recovered_ok += int(inv == [perms[prefix].index(i) for i in range(E)])

    sd_avg2 = OrderedDict((k, ((sd[k].float() + sd_perm_aligned[k].float()) / 2).to(sd[k].dtype)) for k in sd)
    model.load_state_dict(sd_avg2, strict=True)
    aligned_nll = nll(model, batches, device)
    log(f"ALIGNED average NLL = {aligned_nll:.4f}  (should ~= baseline)")
    log(f"blocks with exact permutation recovery: {recovered_ok}/{len(blocks)}")
    layers = blocks

    summary = {
        "model": args.model_a,
        "n_moe_layers": len(layers),
        "moe_format": fmt,
        "num_experts": int(next(iter(blocks.values()))["E"]),
        "num_probe_sequences": int(args.n_probe),
        "sequence_length": int(args.seqlen),
        "baseline_nll": base_nll,
        "gauge_permuted_nll": perm_nll,
        "naive_sameNAME_average_nll": naive_nll,
        "aligned_average_nll": aligned_nll,
        "layers_perm_recovered": recovered_ok,
        "naive_degradation_vs_baseline": naive_nll - base_nll,
        "gauge_degradation_vs_baseline": perm_nll - base_nll,
        "aligned_degradation_vs_baseline": aligned_nll - base_nll,
    }
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "gauge_selfmerge.json").write_text(json.dumps(summary, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "report.md").write_text(build_gauge_report(summary), encoding="utf-8")
    log("\n=== GAUGE SELF-MERGE (real LLM) ===")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"wrote {outdir/'gauge_selfmerge.json'}")


def build_gauge_report(summary):
    return "\n".join(
        [
            "# Real MoE Expert-Gauge Self-Merge Probe",
            "",
            "这个实验在真实 MoE LLM checkpoint 上做一个函数等价反事实：把每一层的 expert slice 和 router row 同步置换，模型函数应保持不变；然后比较同名 average 和恢复 expert 对齐后的 average。",
            "",
            "## Result",
            "",
            f"- Model: `{summary['model']}`",
            f"- MoE layers: `{summary['n_moe_layers']}`; experts/layer: `{summary['num_experts']}`; format: `{summary['moe_format']}`",
            f"- Baseline NLL: `{summary['baseline_nll']:.4f}`",
            f"- Gauge-permuted NLL: `{summary['gauge_permuted_nll']:.4f}`; delta `{summary['gauge_degradation_vs_baseline']:.6f}`",
            f"- Naive same-name average NLL: `{summary['naive_sameNAME_average_nll']:.4f}`; delta `{summary['naive_degradation_vs_baseline']:.4f}`",
            f"- Aligned average NLL: `{summary['aligned_average_nll']:.4f}`; delta `{summary['aligned_degradation_vs_baseline']:.6f}`",
            f"- Exact permutation recovery: `{summary['layers_perm_recovered']}/{summary['n_moe_layers']}` layers",
            "",
            "## Interpretation",
            "",
            "如果 gauge-permuted NLL 与 baseline 一致，而 same-name average 退化，就说明 MoE 的 expert index 不是稳定语义。即使两个 checkpoint 表示同一个函数，只要 expert gauge 不一致，按同名 tensor average 也会破坏模型。恢复 expert 对齐后再 average 是同构 MoE merge 的必要前置步骤。",
            "",
            "## Files",
            "",
            "- `summary.json`",
            "- `gauge_selfmerge.json`",
        ]
    ) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gauge_selfmerge", "cross_correspondence"], required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default="results/fp_moe_real_probe")
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "gauge_selfmerge":
        gauge_selfmerge(args)
    else:
        cross_correspondence(args)


if __name__ == "__main__":
    main()
