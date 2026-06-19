"""First-principles experiment 1: the curvature-displacement law of model merging.

Question (independent of any prior recipe): WHY does linear averaging of two
fine-tuned LLMs degrade each task, and is the degradation predicted by the
loss curvature of each task along the displacement between the two models?

Second-order theory.  For task T with loss L_T minimized (locally) at theta_T,
    L_T(theta) ~= L_T(theta_T) + 1/2 (theta - theta_T)^T H_T (theta - theta_T),
and H_T ~= F_T (the Fisher / Gauss-Newton curvature of the NLL).  For the
uniform average theta_m = 1/2(theta_A + theta_B) we have
    theta_m - theta_A = 1/2 (theta_B - theta_A) = 1/2 d,
so the predicted degradation of task A is
    Delta L_A ~= 1/8 * d^T F_A d ~= 1/8 * sum_k F_A[k] * d[k]^2   (diagonal F).

We test this on the real Qwen2.5-0.5B instruct/coder experts with two tasks:
  - "general"  : wikitext-2 full-sequence NLL  (instruct is the better endpoint)
  - "code"     : HumanEval completion NLL       (coder is the better endpoint)

We then compare merges (uniform, Fisher-weighted, task-arithmetic scales) on the
worst-task and average NLL, and report the per-tensor curvature*displacement
budget that localises *where* interference lives.

Everything here is measured on real weights and real data; no recipes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
LOCAL_QWEN_BASE = Path("/srv/home/bohanlyu/MLS-Bench/vendor/data/models/Qwen2.5-0.5B")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# model / param helpers
# --------------------------------------------------------------------------------------
def default_base_model() -> str:
    return str(LOCAL_QWEN_BASE) if LOCAL_QWEN_BASE.exists() else "Qwen/Qwen2.5-0.5B"


def load_model(name: str, device: str, dtype=torch.float32, local_files_only: bool = True):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, local_files_only=local_files_only)
    model.to(device)
    model.eval()
    return model


def float_param_dict(model) -> "OrderedDict[str, torch.Tensor]":
    """Ordered name->cpu-fp32 tensor for every floating, trainable parameter."""
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in model.named_parameters():
        if torch.is_floating_point(p):
            out[name] = p.detach().to("cpu", torch.float32).clone()
    return out


@torch.no_grad()
def set_param_dict(model, params: "dict[str, torch.Tensor]") -> None:
    device = next(model.parameters()).device
    for name, p in model.named_parameters():
        if name in params:
            p.copy_(params[name].to(device=device, dtype=p.dtype))


def flatten(params: "dict[str, torch.Tensor]", names) -> torch.Tensor:
    return torch.cat([params[n].reshape(-1) for n in names])


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double(); b = b.double()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if float(d) > 0 else 0.0


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def build_wikitext_batches(tokenizer, n_seq: int, seqlen: int):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t and not t.isspace())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    batches = []
    step = seqlen
    for i in range(0, len(ids) - seqlen, step):
        chunk = ids[i : i + seqlen]
        if len(chunk) < seqlen:
            break
        mask = torch.ones(seqlen, dtype=torch.bool)  # score all (full-seq LM)
        batches.append((chunk, mask))
        if len(batches) >= n_seq:
            break
    return batches


def build_humaneval_batches(tokenizer, n_problems: int, max_len: int = 512):
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    batches = []
    for ex in ds:
        prompt = ex["prompt"]
        completion = ex["canonical_solution"]
        p_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        c_ids = tokenizer(completion, return_tensors="pt").input_ids[0]
        ids = torch.cat([p_ids, c_ids])[:max_len]
        mask = torch.zeros(len(ids), dtype=torch.bool)
        # score only completion tokens that survived truncation
        start = min(len(p_ids), len(ids))
        mask[start:] = True
        if mask.sum() < 2:
            continue
        batches.append((ids, mask))
        if len(batches) >= n_problems:
            break
    return batches


def nll_on_batches(model, batches, device) -> float:
    """Mean per-token NLL over masked target positions."""
    total_nll = 0.0
    total_tok = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for ids, mask in batches:
            ids = ids.unsqueeze(0).to(device)
            logits = model(ids).logits[0]  # [L, V]
            # predict token t from position t-1
            tgt = ids[0, 1:]
            lp = logits[:-1]
            m = mask[1:].to(device)
            ce = loss_fn(lp.float(), tgt)  # [L-1]
            total_nll += float((ce * m).sum())
            total_tok += int(m.sum())
    return total_nll / max(total_tok, 1)


def diag_fisher(model, batches, device) -> "OrderedDict[str, torch.Tensor]":
    """Diagonal empirical Fisher = mean over examples of (grad of NLL)^2."""
    fisher = OrderedDict(
        (n, torch.zeros_like(p, device="cpu", dtype=torch.float32))
        for n, p in model.named_parameters()
        if torch.is_floating_point(p)
    )
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    n_ex = 0
    for ids, mask in batches:
        ids_b = ids.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(ids_b).logits[0]
        tgt = ids_b[0, 1:]
        lp = logits[:-1]
        m = mask[1:].to(device)
        ntok = int(m.sum())
        if ntok == 0:
            continue
        ce = loss_fn(lp.float()[m], tgt[m]) / ntok  # mean NLL for this example
        ce.backward()
        for n, p in model.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += (p.grad.detach().to("cpu", torch.float32)) ** 2
        n_ex += 1
    for n in fisher:
        fisher[n] /= max(n_ex, 1)
    model.zero_grad(set_to_none=True)
    return fisher


# --------------------------------------------------------------------------------------
# merges (per-tensor dict space, isomorphic output)
# --------------------------------------------------------------------------------------
def merge_uniform(A, B):
    return OrderedDict((n, 0.5 * (A[n] + B[n])) for n in A)


def merge_taskarith(base, A, B, scale):
    return OrderedDict((n, base[n] + scale * ((A[n] - base[n]) + (B[n] - base[n]))) for n in A)


def merge_fisher(A, B, FA, FB, eps=1e-10):
    out = OrderedDict()
    for n in A:
        fa, fb = FA[n], FB[n]
        denom = fa + fb
        w = (fa * A[n] + fb * B[n]) / denom.clamp_min(eps)
        uni = 0.5 * (A[n] + B[n])
        out[n] = torch.where(denom > eps, w, uni)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=default_base_model())
    ap.add_argument("--expert-a", default="Qwen/Qwen2.5-0.5B-Instruct", help="general expert")
    ap.add_argument("--expert-b", default="Qwen/Qwen2.5-Coder-0.5B-Instruct", help="code expert")
    ap.add_argument("--out", default="results/fp_curvature_law")
    ap.add_argument("--allow-download", action="store_true", help="allow Hugging Face network fetches instead of cache-only loading")
    ap.add_argument("--n-general", type=int, default=64)
    ap.add_argument("--n-code", type=int, default=96)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--n-interp", type=int, default=9)
    ap.add_argument("--fisher-n", type=int, default=48)
    args = ap.parse_args()
    local_files_only = not args.allow_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.expert_a, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("building eval data ...")
    gen_batches = build_wikitext_batches(tok, args.n_general, args.seqlen)
    code_batches = build_humaneval_batches(tok, args.n_code)
    log(f"  general(wikitext) seqs={len(gen_batches)}  code(humaneval) seqs={len(code_batches)}")

    # ---- load experts, capture params ----
    log("loading expert A (general/instruct) ...")
    mA = load_model(args.expert_a, device, local_files_only=local_files_only)
    A = float_param_dict(mA)
    names = list(A.keys())
    log("  computing Fisher_A on general data ...")
    FA = diag_fisher(mA, gen_batches[: args.fisher_n], device)
    del mA
    torch.cuda.empty_cache()

    log("loading expert B (code/coder) ...")
    mB = load_model(args.expert_b, device, local_files_only=local_files_only)
    B = float_param_dict(mB)
    # sanity: identical architecture
    assert list(B.keys()) == names, "param-name mismatch between experts"
    log("  computing Fisher_B on code data ...")
    FB = diag_fisher(mB, code_batches[: args.fisher_n], device)

    base = None
    work = mB  # reuse a loaded module to evaluate arbitrary param dicts
    try:
        log("loading base ...")
        mbase = load_model(args.base, device, local_files_only=local_files_only)
        base = float_param_dict(mbase)
        work = mbase
    except Exception as e:  # base weights may not be cached offline
        log(f"  base unavailable ({type(e).__name__}); proceeding without base "
            f"(skip task-arithmetic + base geometry)")

    # ---- displacement / geometry ----
    d = flatten(B, names) - flatten(A, names)
    geom = {"norm(B-A)": float(d.norm())}
    if base is not None:
        dA = flatten(A, names) - flatten(base, names)
        dB = flatten(B, names) - flatten(base, names)
        geom.update({
            "cos(tauA,tauB)": cosine(dA, dB),
            "norm tauA": float(dA.norm()),
            "norm tauB": float(dB.norm()),
        })
    log(f"geometry: {geom}")

    def eval_both(params, label):
        set_param_dict(work, params)
        g = nll_on_batches(work, gen_batches, device)
        c = nll_on_batches(work, code_batches, device)
        log(f"  [{label}] general_nll={g:.4f} code_nll={c:.4f} worst={max(g,c):.4f}")
        return {"general": g, "code": c, "worst": max(g, c), "avg": 0.5 * (g + c)}

    log("evaluating endpoints ...")
    res = {}
    if base is not None:
        res["base"] = eval_both(base, "base")
    res["expert_A_general"] = eval_both(A, "A=general")
    res["expert_B_code"] = eval_both(B, "B=code")

    # best per-task endpoint NLL (the local minima the theory expands around)
    gA = res["expert_A_general"]["general"]
    cB = res["expert_B_code"]["code"]

    # ---- interpolation path A -> B ----
    log("interpolation path A->B ...")
    path = []
    for i in range(args.n_interp):
        t = i / (args.n_interp - 1)
        pt = OrderedDict((n, (1 - t) * A[n] + t * B[n]) for n in names)
        r = eval_both(pt, f"t={t:.2f}")
        path.append({"t": t, **r})

    # ---- curvature-displacement prediction at the uniform midpoint ----
    log("curvature-displacement law @ midpoint ...")
    # per-tensor budget  1/8 * sum F * d^2
    budget_A = OrderedDict()
    budget_B = OrderedDict()
    for n in names:
        dd = (B[n] - A[n]) ** 2
        budget_A[n] = float((FA[n] * dd).sum()) * 0.125
        budget_B[n] = float((FB[n] * dd).sum()) * 0.125
    pred_dLA = sum(budget_A.values())
    pred_dLB = sum(budget_B.values())

    mid = merge_uniform(A, B)
    mid_res = eval_both(mid, "uniform-mid")
    act_dLA = mid_res["general"] - gA
    act_dLB = mid_res["code"] - cB
    law = {
        "predicted_dL_general": pred_dLA,
        "actual_dL_general": act_dLA,
        "predicted_dL_code": pred_dLB,
        "actual_dL_code": act_dLB,
        "ratio_general": act_dLA / pred_dLA if pred_dLA else None,
        "ratio_code": act_dLB / pred_dLB if pred_dLB else None,
    }
    log(f"  general: predicted {pred_dLA:.4f}  actual {act_dLA:.4f}")
    log(f"  code   : predicted {pred_dLB:.4f}  actual {act_dLB:.4f}")

    # top tensors by combined interference budget
    combined = sorted(names, key=lambda n: budget_A[n] + budget_B[n], reverse=True)
    top_tensors = [
        {"name": n, "budget_general": budget_A[n], "budget_code": budget_B[n]}
        for n in combined[:25]
    ]

    # ---- merge comparison ----
    log("merge comparison ...")
    merges = {"uniform": mid_res}
    merges["fisher"] = eval_both(merge_fisher(A, B, FA, FB), "fisher")
    if base is not None:
        for s in (0.3, 0.5, 0.7):
            merges[f"taskarith_{s}"] = eval_both(merge_taskarith(base, A, B, s), f"taskarith{s}")

    summary = {
        "models": {"base": args.base, "A": args.expert_a, "B": args.expert_b},
        "geometry": geom,
        "endpoints": res,
        "interpolation": path,
        "curvature_law": law,
        "top_interference_tensors": top_tensors,
        "merges": merges,
        "best_endpoint_nll": {"general": gA, "code": cB},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {outdir/'summary.json'}")

    # ---- figure ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [p["t"] for p in path]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(ts, [p["general"] for p in path], "o-", label="general (wikitext) NLL")
        ax.plot(ts, [p["code"] for p in path], "s-", label="code (humaneval) NLL")
        ax.plot(ts, [p["worst"] for p in path], "k--", label="worst-task NLL")
        ax.axvline(0.5, color="gray", ls=":", alpha=0.6)
        ax.set_xlabel("interpolation t  (0=instruct/general, 1=coder/code)")
        ax.set_ylabel("NLL")
        ax.set_title("Interpolation barrier between two experts")
        ax.legend()

        ax = axes[1]
        labs = [t["name"].replace("model.layers.", "L").replace(".weight", "") for t in top_tensors][:15]
        bg = [t["budget_general"] for t in top_tensors][:15]
        bc = [t["budget_code"] for t in top_tensors][:15]
        y = range(len(labs))
        ax.barh(y, bg, alpha=0.7, label="1/8 F_gen d^2")
        ax.barh(y, bc, left=bg, alpha=0.7, label="1/8 F_code d^2")
        ax.set_yticks(list(y))
        ax.set_yticklabels(labs, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("predicted interference budget (NLL units)")
        ax.set_title("Where interference lives (per-tensor)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "curvature_law.png", dpi=130)
        log(f"wrote {outdir/'curvature_law.png'}")
    except Exception as e:  # noqa
        log(f"figure failed: {e}")

    # ---- console verdict ----
    log("\n=== VERDICT ===")
    if "cos(tauA,tauB)" in geom:
        log(f"task-vector cosine(instruct,coder) = {geom['cos(tauA,tauB)']:.3f}")
    log(f"curvature law general ratio(actual/pred) = {law['ratio_general']}")
    log(f"curvature law code    ratio(actual/pred) = {law['ratio_code']}")
    log("worst-task NLL by merge:")
    for k, v in merges.items():
        log(f"   {k:14s} worst={v['worst']:.4f} avg={v['avg']:.4f}")


if __name__ == "__main__":
    main()
