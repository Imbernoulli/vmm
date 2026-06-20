# Qwen3 MoE Router Calibration NLL Probe

这个 artifact 固化一个真实 Qwen3-30B-A3B Instruct/Coder MoE probe：先做 50/50 linear merge，再只训练 `mlp.gate.weight` router tensors，experts 和 shared modules 全部冻结。它回答的不是“最终能不能接受这个 checkpoint”，而是“MoE average 的剩余误差是不是主要来自 router dispatch”。

## Result

- Linear merge worst-NLL: `2.6355`；router-calibrated worst-NLL: `2.4140`；reduction `0.2214`。
- Linear merge avg-NLL: `1.5754`；router-calibrated avg-NLL: `1.4145`；reduction `0.1610`。
- Code NLL goes from `0.5154` to `0.4149`；gap to best source is `-0.0139`。
- General NLL goes from `2.6355` to `2.4140`；gap to best source is `0.1265`。

Interpretation: router-only training improves the averaged MoE on both probe tasks, and code NLL beats both sources, but worst/avg NLL still does not dominate the best source. So the mechanism is real, while acceptance still needs the downstream vLLM gate.

## Mechanism

For an aligned MoE, expert averaging mainly changes the expert functions, but router averaging changes a discrete top-k dispatch boundary. If the two source routers disagree on many tokens, the average router can send a token to a compromise expert set even when the experts themselves remain usable. Training only the router is a direct test of this hypothesis: if NLL drops while experts are frozen, the residual error is dispatch/co-adaptation, not expert geometry.

## Method Metrics

| method | role | general NLL | code NLL | avg NLL | worst NLL | trainable policy |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `instruct` | `source` | 2.2875 | 0.4288 | 1.3582 | 2.2875 | `none` |
| `coder` | `source` | 2.6123 | 0.6111 | 1.6117 | 2.6123 | `none` |
| `linear_merge` | `average_baseline` | 2.6355 | 0.5154 | 1.5754 | 2.6355 | `none_after_weight_average` |
| `linear_merge_routercal` | `router_calibrated_average` | 2.4140 | 0.4149 | 1.4145 | 2.4140 | `router_only_gate_weight_update` |

## Mechanism Deltas

| metric | value | interpretation |
| --- | ---: | --- |
| `worst_nll_reduction_vs_linear` | 0.2214 | positive means router-only calibration improved the averaged MoE |
| `avg_nll_reduction_vs_linear` | 0.1610 | positive means the calibration improved the two-task average NLL |
| `general_nll_reduction_vs_linear` | 0.2214 | positive means general held-out NLL improved after router calibration |
| `code_nll_reduction_vs_linear` | 0.1005 | positive means code held-out NLL improved after router calibration |
| `routercal_general_gap_to_best_source` | 0.1265 | negative would mean router calibration beats both sources on general NLL |
| `routercal_code_gap_to_best_source` | -0.0139 | negative means router calibration beats both sources on code NLL |
| `routercal_worst_gap_to_best_source` | 0.1265 | negative would justify accepting the router-calibrated average by worst-task NLL alone |
| `routercal_avg_gap_to_best_source` | 0.0563 | negative would justify accepting the router-calibrated average by average NLL alone |

## Literature Priors Used

| key | source | mechanism |
| --- | --- | --- |
| `expert_merging` | https://arxiv.org/abs/2509.25712 | Learns layer/chunk coefficients from unlabeled hidden/logit alignment, supporting calibration-data-driven merging rather than fixed coefficients. |
| `mergeme` | https://arxiv.org/abs/2502.00997 | Identifies MoE parameter interference and routing as separate problems during expert model merging. |
| `mergemoe` | https://arxiv.org/abs/2510.14436 | Frames MoE expert merging through expert-output behavior instead of tensor-name averaging alone. |
| `git_rebasin` | https://arxiv.org/abs/2209.04836 | Permutation/gauge alignment must be handled before weight-space interpolation is meaningful. |

## Outputs

- `results/qwen3_moe_router_calibration_nll_probe/method_metrics.csv`
- `results/qwen3_moe_router_calibration_nll_probe/mechanism_deltas.csv`
- `results/qwen3_moe_router_calibration_nll_probe/literature_sources.json`
- `results/qwen3_moe_router_calibration_nll_probe/summary.json`
- `results/qwen3_moe_router_calibration_nll_probe/report.md`
