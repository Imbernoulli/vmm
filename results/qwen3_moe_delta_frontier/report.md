# Qwen3 MoE Delta Frontier Probe

这个 probe 只读已物化 checkpoint 的 delta audit，不重新加载模型权重。
目的不是替代 vLLM 下游评测，而是回答：当前几版规则到底改变了哪些参数组，下一版算法应该把风险预算放在哪里。

- Status: `delta_frontier_ready`
- Candidates: `8`
- Best delta-safety candidate: `tail_trimmed`
- Trust-region total relative delta norm: `0.249`
- Expert-only total relative delta norm: `0.246`
- Tail-trimmed total relative delta norm: `0.243`
- Searched no-gt-0.65 total relative delta norm: `0.248`
- Layer/chunk total relative delta norm: `0.243`
- Unified mechanism total relative delta norm: `0.248`
- Unified mechanism matches searched no-gt-0.65 delta: `True`
- Trust -> expert-only relative norm reduction: `0.003`
- Expert-only -> tail-trimmed relative norm reduction: `0.003`
- Tail-trimmed -> searched no-gt-0.65 relative norm delta: `0.004`
- Searched no-gt-0.65 -> layer/chunk relative norm reduction: `0.004`
- Tail-trimmed / searched / layer-chunk routed tensors >0.6505: `0` / `0` / `0`
- Expert-only attention changed tensors: `0`
- Tail-trimmed attention changed tensors: `0`
- Layer/chunk attention changed tensors: `0`
- Expert-only router changed tensors: `0`
- Tail-trimmed router changed tensors: `0`
- Layer/chunk router changed tensors: `0`
- Next required gate: `vllm_downstream_eval_trust_region_vs_expert_only_tail_trimmed_vs_searched_cap_law_vs_layer_chunk`

## Candidate Frontier

| candidate | total rel | routed rel | attention rel | router changed | max routed rel | routed >1 | routed >0.75 | routed >0.65 | routed >0.6505 | changed tensors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_guarded` | 0.286 | 0.293 | 0.189 | 0/48 | 1.327 | 182 | 839 | 1156 | 1156 | 10641 |
| `audit_gated` | 0.264 | 0.270 | 0.189 | 0/48 | 0.750 | 0 | 164 | 1146 | 1146 | 10641 |
| `trust_region` | 0.249 | 0.255 | 0.189 | 0/48 | 0.750 | 0 | 14 | 366 | 354 | 10641 |
| `expert_only` | 0.246 | 0.255 | 0.000 | 0/48 | 0.750 | 0 | 14 | 366 | 354 | 10353 |
| `tail_trimmed` | 0.243 | 0.252 | 0.000 | 0/48 | 0.650 | 0 | 0 | 80 | 0 | 10353 |
| `searched_no_gt065` | 0.248 | 0.256 | 0.000 | 0/48 | 0.650 | 0 | 0 | 245 | 0 | 10353 |
| `layer_chunk` | 0.243 | 0.252 | 0.000 | 0/48 | 0.650 | 0 | 0 | 89 | 0 | 10353 |
| `unified_mechanism` | 0.248 | 0.256 | 0.000 | 0/48 | 0.650 | 0 | 0 | 245 | 0 | 10353 |

## Pairwise Reductions

| from | to | total rel reduction | routed rel reduction | attention rel reduction | routed >1 reduction | routed >0.75 reduction | routed >0.65 reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_guarded` | `audit_gated` | 0.022 | 0.023 | 0.000 | 182 | 675 | 10 |
| `audit_gated` | `trust_region` | 0.015 | 0.016 | 0.000 | 0 | 150 | 780 |
| `trust_region` | `expert_only` | 0.003 | 0.000 | 0.189 | 0 | 0 | 0 |
| `expert_only` | `tail_trimmed` | 0.003 | 0.003 | 0.000 | 0 | 14 | 286 |
| `tail_trimmed` | `searched_no_gt065` | -0.004 | -0.005 | 0.000 | 0 | 0 | -165 |
| `searched_no_gt065` | `layer_chunk` | 0.004 | 0.004 | 0.000 | 0 | 0 | 156 |
| `layer_chunk` | `unified_mechanism` | -0.004 | -0.004 | 0.000 | 0 | 0 | -156 |

## Highest Trust-Region Layers

| layer | route rel | trust rel | expert-only rel | route->trust reduction | trust->expert-only reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | 0.354 | 0.310 | 0.308 | 0.043 | 0.002 |
| 17 | 0.347 | 0.291 | 0.288 | 0.056 | 0.004 |
| 12 | 0.340 | 0.288 | 0.284 | 0.052 | 0.004 |
| 23 | 0.336 | 0.284 | 0.281 | 0.052 | 0.003 |
| 20 | 0.340 | 0.280 | 0.278 | 0.059 | 0.003 |
| 26 | 0.343 | 0.279 | 0.277 | 0.063 | 0.002 |
| 24 | 0.326 | 0.277 | 0.275 | 0.049 | 0.003 |
| 8 | 0.313 | 0.277 | 0.273 | 0.036 | 0.003 |
| 25 | 0.316 | 0.273 | 0.270 | 0.043 | 0.003 |
| 15 | 0.310 | 0.270 | 0.267 | 0.039 | 0.004 |
| 10 | 0.302 | 0.269 | 0.265 | 0.033 | 0.003 |
| 0 | 0.290 | 0.266 | 0.263 | 0.023 | 0.003 |

## Interpretation

Trust-region rules control the routed-expert delta tail; expert-only freezes attention without changing routed tail risk. Tail-trimmed then reduces the remaining routed tail while preserving the frozen attention/router contract. The searched no-gt-0.65 candidate tests whether the hand-built route/load/category risk penalties can be replaced by a simpler global expert cap. The layer/chunk candidate then tests whether layer sensitivity coefficients can reduce structural delta further without removing useful Coder specialization. Attention, cap-law complexity, and layer sensitivity should therefore be decided by downstream eval, not by delta safety alone.

实际含义：trust-region/audit-gated 的价值主要是压 routed expert 的高 relative-delta tail；expert-only 只是把 shared attention 从候选里拿掉，几乎不改变 routed expert 风险；tail-trimmed 才继续压剩余 routed tail。searched no-gt-0.65 则把复杂风险 penalty 换成统一 cap，给下一轮 eval 一个更简单的候选。layer/chunk candidate 再把机制 leverage 里的层敏感度转成系数，给下一轮 eval 一个更细粒度的候选。所以 attention 是否保留、risk penalty 是否保留、layer sensitivity 是否有用，都不能靠 delta safety 单独判断，必须靠同任务 vLLM 下游结果决定。

## Files

- `candidate_frontier`: `results/qwen3_moe_delta_frontier/candidate_delta_frontier.csv`
- `group_frontier`: `results/qwen3_moe_delta_frontier/group_delta_frontier.csv`
- `pairwise_reductions`: `results/qwen3_moe_delta_frontier/pairwise_delta_reductions.csv`
- `tail_thresholds`: `results/qwen3_moe_delta_frontier/tail_thresholds.csv`
- `layer_frontier`: `results/qwen3_moe_delta_frontier/layer_delta_frontier.csv`
- `summary`: `results/qwen3_moe_delta_frontier/summary.json`
- `report`: `results/qwen3_moe_delta_frontier/report.md`
