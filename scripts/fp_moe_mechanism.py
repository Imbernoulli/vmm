"""First-principles experiment 2: what *uniquely* breaks when you average MoE models.

Dense merging degradation is a smooth quadratic in the displacement (curvature
law, see fp_curvature_law.py).  A Mixture-of-Experts layer adds structure that
no second-order weight-space expansion can capture:

    y(x) = sum_i  g_i(x) * E_i(x),     g = top-k softmax over router logits W_r x

Three failure modes specific to the *gate*:
  (F1) Expert permutation symmetry.  Expert index i in model 1 and model 2 need
       not be the "same" function -> averaging E_i^1 with E_i^2 averages
       unrelated experts.  (The MoE analogue of Git Re-Basin neuron permutation.)
  (F2) Router / dispatch drift.  top-k is a discrete argmax; averaging routers
       produces dispatch that matches neither parent, so tokens land on experts
       not tuned for them.  This term is *non-smooth* in the weights.
  (F3) Route-conditioned curvature.  Expert i's loss curvature is only defined
       over the tokens routed to i, so curvature weighting of experts must be
       conditioned on routing, not global.

This script builds a fully controlled MoE on a piecewise-regression task where
we KNOW the ground-truth domain structure, then measures each failure mode and
the marginal value of each fix:
    naive linear avg
    + expert alignment (Hungarian match on expert FUNCTION, transferable to LLMs)
    + route-conditioned Fisher weighting of aligned experts
    + router calibration (light re-fit of the linear router to the merged experts)

All numbers come from real forward/backward passes on trained networks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------------------
# ground-truth piecewise task:  domain(x)=argmax_k u_k.x ;  y = teacher_domain(x)
# --------------------------------------------------------------------------------------
class Task:
    def __init__(self, d_in=16, K=4, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.d_in, self.K, self.device = d_in, K, device
        self.U = torch.randn(K, d_in, generator=g)  # gating hyperplanes
        # one random 2-layer tanh teacher per domain
        self.teachers = []
        for k in range(K):
            gk = torch.Generator().manual_seed(seed * 100 + k + 1)
            W1 = torch.randn(d_in, 32, generator=gk) / d_in**0.5
            b1 = torch.randn(32, generator=gk) * 0.1
            W2 = torch.randn(32, 1, generator=gk) / 32**0.5
            self.teachers.append((W1, b1, W2))

    def domain(self, x):
        return (x @ self.U.t()).argmax(dim=1)

    def target(self, x):
        dom = self.domain(x)
        y = torch.zeros(x.shape[0], 1)
        for k in range(self.K):
            m = dom == k
            if m.any():
                W1, b1, W2 = self.teachers[k]
                h = torch.tanh(x[m] @ W1 + b1)
                y[m] = h @ W2
        return y

    def sample(self, n, domain_weights=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        if domain_weights is None:
            x = torch.randn(n, self.d_in, generator=g)
            return x.to(self.device), self.target(x).to(self.device)
        # rejection-sample to hit a target domain distribution
        xs = []
        dw = torch.tensor(domain_weights, dtype=torch.float32)
        dw = dw / dw.sum()
        per = (dw * n).round().long()
        for k in range(self.K):
            need = int(per[k])
            got = []
            tries = 0
            while sum(t.shape[0] for t in got) < need and tries < 200:
                xb = torch.randn(need * 4, self.d_in, generator=g)
                xb = xb[self.domain(xb) == k]
                got.append(xb)
                tries += 1
            xk = torch.cat(got)[:need] if got else torch.empty(0, self.d_in)
            xs.append(xk)
        x = torch.cat(xs)
        idx = torch.randperm(x.shape[0], generator=g)
        x = x[idx]
        return x.to(self.device), self.target(x).to(self.device)


# --------------------------------------------------------------------------------------
# MoE model (top-1) with explicit, nameable parameter blocks
# --------------------------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, d_in=16, d_h=32, E=4, topk=1):
        super().__init__()
        self.router = nn.Linear(d_in, E, bias=False)
        self.experts = nn.ModuleList([Expert(d_in, d_h) for _ in range(E)])
        self.E, self.topk = E, topk

    def forward(self, x, return_route=False):
        logits = self.router(x)  # [N,E]
        if self.topk == 1:
            gate_idx = logits.argmax(dim=1)  # [N]
            probs = F.softmax(logits, dim=1)
            gsel = probs.gather(1, gate_idx[:, None]).squeeze(1)  # [N]
            out = torch.zeros(x.shape[0], 1, device=x.device)
            for i in range(self.E):
                m = gate_idx == i
                if m.any():
                    out[m] = gsel[m, None] * self.experts[i](x[m])
            if return_route:
                return out, gate_idx, logits
            return out
        else:
            topv, topi = logits.topk(self.topk, dim=1)
            w = F.softmax(topv, dim=1)
            out = torch.zeros(x.shape[0], 1, device=x.device)
            for j in range(self.topk):
                idx = topi[:, j]
                for i in range(self.E):
                    m = idx == i
                    if m.any():
                        out[m] += w[m, j, None] * self.experts[i](x[m])
            if return_route:
                return out, topi[:, 0], logits
            return out


def float_params(model):
    return OrderedDict(
        (n, p.detach().cpu().clone()) for n, p in model.named_parameters()
    )


@torch.no_grad()
def set_params(model, params):
    for n, p in model.named_parameters():
        p.copy_(params[n].to(p.device))


def train(model, task, steps, domain_weights, lr=3e-3, seed=0, only_router=False):
    params = [p for n, p in model.named_parameters() if (("router" in n) or not only_router)]
    opt = torch.optim.Adam(params, lr=lr)
    for s in range(steps):
        x, y = task.sample(512, domain_weights=domain_weights, seed=seed * 100000 + s)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def per_domain_loss(model, task, n=2000):
    out = {}
    for k in range(task.K):
        dw = [0.0] * task.K
        dw[k] = 1.0
        x, y = task.sample(n, domain_weights=dw, seed=777 + k)
        pred = model(x)
        out[k] = float(F.mse_loss(pred, y))
    out["worst"] = max(out[k] for k in range(task.K))
    out["avg"] = sum(out[k] for k in range(task.K)) / task.K
    return out


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def expert_correspondence(mA, mB, probe_x):
    """C[i,j] = cosine similarity between expert_i^A and expert_j^B raw outputs
    over a shared probe set.  Identity-optimal => experts aligned; otherwise
    they permuted during fine-tuning."""
    E = mA.E
    OA = torch.stack([mA.experts[i](probe_x).squeeze(1) for i in range(E)])  # [E,N]
    OB = torch.stack([mB.experts[j](probe_x).squeeze(1) for j in range(E)])
    OA = OA - OA.mean(1, keepdim=True)
    OB = OB - OB.mean(1, keepdim=True)
    C = torch.zeros(E, E)
    for i in range(E):
        for j in range(E):
            C[i, j] = F.cosine_similarity(OA[i], OB[j], dim=0)
    return C


def hungarian(C):
    """maximise total similarity; returns perm pi so that B-expert pi[i] <-> A-expert i."""
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(-C.numpy())
        perm = [0] * len(r)
        for i, j in zip(r, c):
            perm[i] = int(j)
        return perm
    except Exception:
        # greedy fallback
        E = C.shape[0]
        used = set()
        perm = [0] * E
        for i in range(E):
            best, bj = -1e9, 0
            for j in range(E):
                if j in used:
                    continue
                if C[i, j] > best:
                    best, bj = float(C[i, j]), j
            perm[i] = bj
            used.add(bj)
        return perm


@torch.no_grad()
def router_agreement(mA, mB, probe_x):
    _, gA, _ = mA(probe_x, return_route=True)
    _, gB, _ = mB(probe_x, return_route=True)
    return float((gA == gB).float().mean())


@torch.no_grad()
def prediction_mse(mA, mB, probe_x):
    return float(F.mse_loss(mA(probe_x), mB(probe_x)))


@torch.no_grad()
def route_load(model, probe_x):
    _, gate_idx, _ = model(probe_x, return_route=True)
    counts = torch.bincount(gate_idx.cpu(), minlength=model.E).float()
    return (counts / counts.sum().clamp_min(1.0)).tolist()


def permute_B(paramsB, perm, E):
    """Relabel model B's experts (and router output rows) by perm: new expert i
    takes old expert perm[i].  Keeps function identical (gauge symmetry)."""
    out = OrderedDict()
    for n, v in paramsB.items():
        out[n] = v.clone()
    # router.weight: [E, d_in] rows index experts
    rw = paramsB["router.weight"].clone()
    out["router.weight"] = torch.stack([rw[perm[i]] for i in range(E)])
    # experts.i.* <- experts.perm[i].*
    for i in range(E):
        src = perm[i]
        for sub in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
            out[f"experts.{i}.{sub}"] = paramsB[f"experts.{src}.{sub}"].clone()
    return out


def parse_perm(value, E):
    if value in {"", "none", "identity"}:
        return list(range(E))
    out = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(out) != E or sorted(out) != list(range(E)):
        raise ValueError(f"--gauge-perm must be a permutation of 0..{E - 1}; got {value!r}")
    return out


def diag_fisher(model, task, domain_weights, n_batches=32, bs=256, seed=0):
    fisher = OrderedDict(
        (n, torch.zeros_like(p)) for n, p in model.named_parameters()
    )
    for s in range(n_batches):
        x, y = task.sample(bs, domain_weights=domain_weights, seed=seed * 999 + s)
        model.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
    for n in fisher:
        fisher[n] = (fisher[n] / n_batches).cpu()
    model.zero_grad(set_to_none=True)
    return fisher


# --------------------------------------------------------------------------------------
# merges
# --------------------------------------------------------------------------------------
def merge_uniform(A, B):
    return OrderedDict((n, 0.5 * (A[n] + B[n])) for n in A)


def merge_fisher(A, B, FA, FB, eps=1e-12):
    out = OrderedDict()
    for n in A:
        denom = FA[n] + FB[n]
        w = (FA[n] * A[n] + FB[n] * B[n]) / denom.clamp_min(eps)
        out[n] = torch.where(denom > eps, w, 0.5 * (A[n] + B[n]))
    return out


def calibrate_router(model, task, steps=200, lr=5e-3, seed=0):
    """Light re-fit of ONLY the linear router to the (frozen) merged experts on a
    small mixed probe set.  Architecture/param-shapes unchanged."""
    for n, p in model.named_parameters():
        p.requires_grad_("router" in n)
    opt = torch.optim.Adam([model.router.weight], lr=lr)
    for s in range(steps):
        x, y = task.sample(512, domain_weights=[1.0] * task.K, seed=424242 + s)
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def write_method_metrics(path, results, K):
    fields = ["method", "worst", "avg"] + [f"domain_{k}" for k in range(K)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, metrics in results.items():
            row = {"method": method, "worst": metrics["worst"], "avg": metrics["avg"]}
            row.update({f"domain_{k}": metrics[k] for k in range(K)})
            writer.writerow(row)


def build_mechanism_rows(results):
    def loss_reduction(before, after):
        return results[before]["worst"] - results[after]["worst"]

    rows = [
        {
            "mechanism": "expert_identity_alignment",
            "baseline": "uniform_same_name",
            "intervention": "uniform_aligned",
            "worst_loss_reduction": loss_reduction("uniform_same_name", "uniform_aligned"),
            "interpretation": "same tensor name is not a stable expert identity after a legal MoE gauge permutation",
        },
        {
            "mechanism": "router_calibration_after_alignment",
            "baseline": "uniform_aligned",
            "intervention": "uniform_aligned_routercal",
            "worst_loss_reduction": loss_reduction("uniform_aligned", "uniform_aligned_routercal"),
            "interpretation": "after experts move, the top-k router boundary must be re-fit to the merged experts",
        },
        {
            "mechanism": "route_conditioned_fisher",
            "baseline": "uniform_aligned",
            "intervention": "fisher_aligned",
            "worst_loss_reduction": loss_reduction("uniform_aligned", "fisher_aligned"),
            "interpretation": "curvature weighting is not automatically safe; it must pass a held-out gate",
        },
        {
            "mechanism": "router_calibration_after_fisher",
            "baseline": "fisher_aligned",
            "intervention": "fisher_aligned_routercal",
            "worst_loss_reduction": loss_reduction("fisher_aligned", "fisher_aligned_routercal"),
            "interpretation": "router calibration can repair some dispatch drift even when expert weighting is imperfect",
        },
        {
            "mechanism": "router_cannot_fix_misaligned_experts",
            "baseline": "uniform_same_name",
            "intervention": "uniform_same_name_routercal",
            "worst_loss_reduction": loss_reduction("uniform_same_name", "uniform_same_name_routercal"),
            "interpretation": "router-only fitting is limited if the averaged experts themselves mix unrelated functions",
        },
    ]
    return rows


def write_mechanism_csv(path, rows):
    fields = ["mechanism", "baseline", "intervention", "worst_loss_reduction", "interpretation"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary, mechanism_rows):
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    best_merge = summary["best_merge_method"]
    best_overall = summary["best_overall_method"]
    lines = [
        "# MoE Average Mechanism Probe",
        "",
        "这个实验只回答一个问题：为什么同构 MoE checkpoint 不能直接按同名 tensor average。我们把 B 模型做了一个函数等价的 expert/router row 置换；B 的输出几乎不变，但同名 expert index 的语义被打乱，因此它能隔离 MoE 特有的平均失败机制。",
        "",
        "## 结果摘要",
        "",
        f"- 函数等价置换后的 B 与原 B probe MSE: `{summary['gauge_equivalence_mse']:.6g}`。",
        f"- same-name uniform worst loss: `{summary['results']['uniform_same_name']['worst']:.4f}`。",
        f"- expert-aligned uniform worst loss: `{summary['results']['uniform_aligned']['worst']:.4f}`。",
        f"- aligned + router calibration worst loss: `{summary['results']['uniform_aligned_routercal']['worst']:.4f}`。",
        f"- aligned + Fisher worst loss: `{summary['results']['fisher_aligned']['worst']:.4f}`。",
        f"- 本次 best merge: `{best_merge}`，worst loss `{summary['results'][best_merge]['worst']:.4f}`。",
        f"- 本次 overall lowest: `{best_overall}`，worst loss `{summary['results'][best_overall]['worst']:.4f}`。",
        "",
        "## 机制结论",
        "",
        "| mechanism | baseline | intervention | worst-loss reduction | implication |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in mechanism_rows:
        lines.append(
            f"| `{row['mechanism']}` | `{row['baseline']}` | `{row['intervention']}` | "
            f"{row['worst_loss_reduction']:.4f} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Unified Rule",
            "",
            "不是固定说 TIES/Fisher/RegMean 哪个永远最好，而是用同一个 gate：先检查模型是否同构；MoE 先做 expert function alignment；expert 权重只在 route-conditioned probe 上降低 held-out loss 时启用；router 只允许在 expert 已对齐之后做小步校准或蒸馏；capacity correction 只在 sparse top-k overflow 超预算时启用。所有步骤保持 tokenizer、模型类、tensor name/shape、expert 数不变。",
            "",
            "## Files",
            "",
            "- `summary.json`",
            "- `method_metrics.csv`",
            "- `mechanism_deltas.csv`",
            "- `moe_mechanism.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_figure(path, summary):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log(f"figure skipped: {exc}")
        return

    methods = [
        "expert_A",
        "expert_B",
        "base",
        "uniform_same_name",
        "uniform_same_name_routercal",
        "uniform_aligned",
        "uniform_aligned_routercal",
        "fisher_aligned",
        "fisher_aligned_routercal",
    ]
    labels = [
        "A",
        "B",
        "base",
        "same-name avg",
        "same-name + router",
        "aligned avg",
        "aligned + router",
        "aligned Fisher",
        "Fisher + router",
    ]
    values = [summary["results"][m]["worst"] for m in methods if m in summary["results"]]
    labels = labels[: len(values)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    bars = ax.bar(range(len(values)), values, color=["#4c78a8", "#4c78a8", "#8c8c8c"] + ["#d95f02"] * 2 + ["#1b9e77"] * 2 + ["#7570b3"] * 2)
    best = min(values)
    for bar, val in zip(bars, values):
        if abs(val - best) < 1e-12:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.6)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("worst-domain MSE (lower is better)")
    ax.set_title("MoE merge interventions")

    ax = axes[1]
    C = torch.tensor(summary["expert_correspondence_matrix"])
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("expert output cosine: A_i vs observed B_j")
    ax.set_xlabel("observed B expert j")
    ax.set_ylabel("A expert i")
    ax.set_xticks(range(C.shape[1]))
    ax.set_yticks(range(C.shape[0]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{float(C[i, j]):.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fp_moe_mechanism")
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--base-steps", type=int, default=1500)
    ap.add_argument("--ft-steps", type=int, default=800)
    ap.add_argument(
        "--gauge-perm",
        default="2,0,3,1",
        help="Function-preserving permutation applied to B's expert/router indices; use 'identity' to disable.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    task = Task(d_in=16, K=args.K, seed=1)

    log("training base MoE on all domains ...")
    base = MoE(16, 32, E=args.E, topk=args.topk)
    train(base, task, args.base_steps, domain_weights=[1.0] * args.K, seed=1)
    base_loss = per_domain_loss(base, task)
    log(f"  base per-domain: {base_loss}")
    base_params = float_params(base)

    # two fine-tunes from the SHARED base: A emphasises domains 0,1 ; B emphasises 2,3
    wA = [4.0, 4.0, 0.25, 0.25][: args.K]
    wB = [0.25, 0.25, 4.0, 4.0][: args.K]
    log("fine-tuning expert A (domains 0,1) and B (domains 2,3) from shared base ...")
    mA = MoE(16, 32, E=args.E, topk=args.topk)
    set_params(mA, base_params)
    train(mA, task, args.ft_steps, domain_weights=wA, seed=2)
    mB = MoE(16, 32, E=args.E, topk=args.topk)
    set_params(mB, base_params)
    train(mB, task, args.ft_steps, domain_weights=wB, seed=3)
    A = float_params(mA)
    B_trained = float_params(mB)
    lossA = per_domain_loss(mA, task)
    lossB = per_domain_loss(mB, task)
    log(f"  A per-domain: {lossA}")
    log(f"  B per-domain: {lossB}")

    probe_x, _ = task.sample(1024, domain_weights=[1.0] * args.K, seed=9)

    natural_C = expert_correspondence(mA, mB, probe_x)
    natural_perm = hungarian(natural_C)

    gauge_perm = parse_perm(args.gauge_perm, args.E)
    B = permute_B(B_trained, gauge_perm, args.E)
    mB_observed = MoE(16, 32, E=args.E, topk=args.topk)
    set_params(mB_observed, B)
    gauge_equivalence_mse = prediction_mse(mB, mB_observed, probe_x)
    lossB_observed = per_domain_loss(mB_observed, task)
    log(f"  applied function-equivalent B gauge permutation: {gauge_perm}")
    log(f"  B vs gauge(B) probe MSE = {gauge_equivalence_mse:.8f}")
    log(f"  gauge(B) per-domain: {lossB_observed}")

    # ---- diagnostics: F1 permutation, F2 router drift ----
    C = expert_correspondence(mA, mB_observed, probe_x)
    perm = hungarian(C)
    identity = list(range(args.E))
    diag_sum = float(sum(C[i, i] for i in range(args.E)))
    perm_sum = float(sum(C[i, perm[i]] for i in range(args.E)))
    agree_raw = router_agreement(mA, mB_observed, probe_x)
    log(f"  expert correspondence diag: {[round(float(C[i,i]),3) for i in range(args.E)]}")
    log(f"  Hungarian perm: {perm}  identity-optimal={perm==identity}")
    log(f"  router agreement(raw) = {agree_raw:.3f}")

    # ---- merges ----
    work = MoE(16, 32, E=args.E, topk=args.topk)

    def ev(params, label):
        set_params(work, params)
        r = per_domain_loss(work, task)
        log(f"  [{label}] worst={r['worst']:.4f} avg={r['avg']:.4f} "
            f"per-domain={[round(r[k],4) for k in range(args.K)]}")
        return r

    results = {}
    results["base"] = base_loss
    results["expert_A"] = lossA
    results["expert_B"] = lossB_observed

    log("\n--- merge ladder ---")
    same_name = merge_uniform(A, B)
    results["uniform_same_name"] = ev(same_name, "uniform_same_name")

    # expert-aligned: permute B to match A's expert functions, then average
    B_aligned = permute_B(B, perm, args.E)
    # measure agreement after alignment
    set_params(work, B_aligned)
    agree_aligned = router_agreement(mA, work, probe_x)
    aligned_uniform = merge_uniform(A, B_aligned)
    results["uniform_aligned"] = ev(aligned_uniform, "uniform+align")

    # router calibration on top of aligned uniform isolates F2 from F1.
    set_params(work, aligned_uniform)
    calibrate_router(work, task)
    results["uniform_aligned_routercal"] = per_domain_loss(work, task)
    uarc = results["uniform_aligned_routercal"]
    log(f"  [uniform+align+routercal] worst={uarc['worst']:.4f} avg={uarc['avg']:.4f} "
        f"per-domain={[round(uarc[k],4) for k in range(args.K)]}")

    # route-conditioned Fisher on aligned params
    FA = diag_fisher(mA, task, wA, seed=11)
    # Fisher for B must be computed on the aligned param layout -> load aligned B
    set_params(work, B_aligned)
    FB = diag_fisher(work, task, wB, seed=12)
    results["fisher_aligned"] = ev(merge_fisher(A, B_aligned, FA, FB), "fisher+align")

    # + router calibration (light re-fit of router to merged experts)
    merged = merge_fisher(A, B_aligned, FA, FB)
    set_params(work, merged)
    calibrate_router(work, task)
    results["fisher_aligned_routercal"] = per_domain_loss(work, task)
    rc = results["fisher_aligned_routercal"]
    log(f"  [fisher+align+routercal] worst={rc['worst']:.4f} avg={rc['avg']:.4f} "
        f"per-domain={[round(rc[k],4) for k in range(args.K)]}")

    # ablation: router calibration on top of same-name uniform cannot fully fix
    # expert semantic mismatch.
    set_params(work, same_name)
    calibrate_router(work, task)
    results["uniform_same_name_routercal"] = per_domain_loss(work, task)

    merge_methods = [k for k in results if k not in {"base", "expert_A", "expert_B"}]
    best_merge_method = min(merge_methods, key=lambda k: results[k]["worst"])
    best_overall_method = min(results, key=lambda k: results[k]["worst"])
    mechanism_rows = build_mechanism_rows(results)

    summary = {
        "args": vars(args),
        "natural_hungarian_perm_before_gauge": natural_perm,
        "natural_identity_optimal_before_gauge": natural_perm == identity,
        "gauge_perm_applied_to_B": gauge_perm,
        "gauge_equivalence_mse": gauge_equivalence_mse,
        "expert_correspondence_matrix": C.tolist(),
        "hungarian_perm": perm,
        "identity_optimal": perm == identity,
        "corr_diag_sum": diag_sum,
        "corr_perm_sum": perm_sum,
        "router_agreement_raw": agree_raw,
        "router_agreement_aligned": agree_aligned,
        "route_load_A": route_load(mA, probe_x),
        "route_load_B_observed": route_load(mB_observed, probe_x),
        "best_merge_method": best_merge_method,
        "best_overall_method": best_overall_method,
        "mechanism_deltas": mechanism_rows,
        "results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_method_metrics(outdir / "method_metrics.csv", results, args.K)
    write_mechanism_csv(outdir / "mechanism_deltas.csv", mechanism_rows)
    (outdir / "report.md").write_text(build_report(summary, mechanism_rows), encoding="utf-8")
    log(f"\nwrote {outdir/'summary.json'}")
    log(f"wrote {outdir/'method_metrics.csv'}")
    log(f"wrote {outdir/'mechanism_deltas.csv'}")
    log(f"wrote {outdir/'report.md'}")
    write_figure(outdir / "moe_mechanism.png", summary)

    log("\n=== VERDICT (worst-domain loss, lower=better) ===")
    order = ["expert_A", "expert_B", "base", "uniform_same_name", "uniform_same_name_routercal",
             "uniform_aligned", "uniform_aligned_routercal", "fisher_aligned", "fisher_aligned_routercal"]
    for k in order:
        if k in results:
            log(f"   {k:28s} worst={results[k]['worst']:.4f} avg={results[k]['avg']:.4f}")


if __name__ == "__main__":
    main()
