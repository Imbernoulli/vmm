# Model Averaging Literature and Probe Matrix

这份报告把 Dense model averaging、线性插值、task-vector merging 和 MoE 专用 merging 的论文证据整理成一张工程矩阵。重点不是机械列方法，而是回答：看到哪些指标时，应该使用哪类平均策略；哪些情况下应该先拒绝或校准，而不是直接写 checkpoint。

## 结论

1. Dense 模型可以从 linear average、greedy soup、task arithmetic 开始，但前提是同源、低 barrier、端点能力相近；否则 `0.5/0.5` 只是负 baseline。
2. TIES、DARE、DELLA、STAR 这些方法本质上是在处理 delta 冗余、符号冲突或谱空间冲突；它们需要 delta magnitude、sign conflict、singular spectrum 等 probe 支撑。
3. Fisher、RegMean、AdaMerging、DAM 这类方法把 average 变成重要性加权或 coefficient learning；它们更像 probe-guided average，而不是固定菜谱。
4. MoE 的核心失败模式是 router/expert 共同失配：router 可能 breakdown，expert index 可能不再对应同一功能，专家专长会让全参数同权平均更脆弱。
5. 对 Qwen3-30B-A3B / Qwen3-Coder-30B-A3B 这类同构 MoE，最保守路径是 topology gate -> router gate -> expert matching -> route-frequency tensor rules -> sparse capacity gate -> held-out eval。

## 关键计数

- Sources reviewed: `22`
- Method families: `7`
- Probe groups: `7`
- MoE optimization stages: `7`

## 方法矩阵

| method family | dense use | MoE use | primary probe | recommended action |
| --- | --- | --- | --- | --- |
| Uniform / linear average | Baseline for same-base checkpoints; useful when lambda path has low barrier. | Negative baseline unless router/expert probes prove route stability. | endpoint score; lambda sweep; midpoint barrier; worst-task score | Only materialize if validation grid beats endpoints or is not endpoint-only. |
| Task arithmetic / coefficient search | Search task-vector coefficients on same-anchor deltas. | Apply separately to shared modules, router, and expert groups. | alpha/beta grid; layer cosine; held-in retention | Move from global weights to layer/module/expert-specific weights. |
| Sign / sparsity conflict methods | Use TIES, DARE, DELLA, or STAR when deltas are redundant or sign-conflicting. | Use on shared and expert FFN deltas after expert matching, not on router blindly. | sign conflict; weighted conflict; delta magnitude distribution; singular spectrum | Convert conflict signals into tensor rules and preserve critical groups. |
| Importance / activation-aware average | Use Fisher, RegMean, or RegMean++ when calibration activations are available. | Estimate expert sensitivity with route-conditioned NLL and activation covariance. | diagonal Fisher; activation covariance; NLL sensitivity | Report as structured average with plane residual, not as raw on-plane average. |
| Output-space calibrated average | Fit merge coefficients to calibration outputs when labels are scarce but source logits are available. | Use source logits, route logits, and expert outputs as residual targets for router/expert weighting. | output residual energy; source-logit KL; layer-wise projection residual | Optimize coefficients against output/KD residuals and validate under hard dispatch. |
| Alignment before averaging | Needed when checkpoints are not same initialization or barrier remains high. | Needed when expert indices or feature spaces are permuted. | permutation residual; feature CKA; expert output cosine | Run feature/expert matching before computing any average. |
| Router-aware MoE average | Not applicable. | Freeze or calibrate router; merge shared/expert tensors with separate rules. | route overlap; router entropy; max expert fraction; top-k margin | Keep router frozen, calibrate router, or reject candidate before writing checkpoint. |

## Probe 矩阵

| probe | measures | dense decision | MoE decision | artifact target |
| --- | --- | --- | --- | --- |
| endpoint and held-in retention | Whether each source model's native ability survives the average. | Reject averages that improve mean score while sacrificing one expert. | Use per-domain retention before trusting route-aware rules. | method_metrics.csv; decision_table.csv |
| lambda path and midpoint barrier | Linear connectivity between endpoints or between anchor and source. | Low barrier supports soups/task arithmetic; high barrier triggers alignment or layer-wise coefficients. | Run shared-only, router-frozen, and all-weight paths separately. | path_metrics.csv; qwen path sweep; alpha/beta grids |
| delta cosine and sign conflict | Parameter-level direction agreement and destructive sign disagreement. | Choose TIES/DARE/DELLA/STAR or freeze conflict-heavy groups. | Compute per shared module and per matched expert, never only globally. | delta_summary.csv; interference.csv |
| activation/Fisher sensitivity | Which parameters or linear layers are important on calibration data. | Use Fisher/RegMean/AdaMerging style coefficients. | Compute route-conditioned sensitivity for experts and router. | future activation covariance and Fisher summaries |
| router entropy and route overlap | Whether MoE routing still dispatches tokens to appropriate experts after merging. | Not applicable. | Freeze/calibrate router or reject all-weight average if overlap collapses. | router_summary.csv; route_overlap.csv; router_readiness.csv |
| expert output similarity | Whether expert index e in two checkpoints represents the same function. | Use analogous feature alignment only if initialization differs. | Build expert remap aliases before averaging expert tensors. | expert_match.csv; source_tensor_aliases.txt |
| output residual and source KD | Whether a candidate average can reproduce source logits or expert outputs on calibration inputs. | Use output-space coefficient fitting when parameter probes are ambiguous. | Use route/output KD to calibrate router and expert weights, then recheck sparse dispatch. | router_kd_trace.csv; router_route_kd_trace.csv; unified_moe_trace.csv |

## MoE 优化路线

| stage | question | required probe | accept rule | writer action |
| --- | --- | --- | --- | --- |
| 0_topology_gate | Do all inputs have the same config, tokenizer, router shape, expert count, and tensor names? | config/header inspection | same shape or documented source tensor aliases only | Proceed to dry-run validation. |
| 1_router_gate | Does simple averaging break routing? | router entropy, top-k agreement, route overlap, max expert fraction | No collapse, no large drift, no fragile top-k boundary. | Freeze router by default; allow small router delta only after readiness passes. |
| 2_expert_alignment | Are source expert indices semantically aligned? | expert output cosine, route coactivation, task profile similarity | Matched experts above cosine threshold; manual review for low matches. | Pass source_tensor_aliases.txt to same-shape writer. |
| 3_expert_weighting | Which source should dominate each expert tensor? | route frequency, NLL sensitivity, expert delta conflict | Weights reflect task route mass and do not damage general retention. | Emit tensor_rules.txt with per-expert source weights. |
| 4_shared_module_merge | Can shared attention/norm/MLP be averaged globally? | layer cosine, sign conflict, Fisher/activation sensitivity | Use module-specific weights when conflicts concentrate. | Emit tensor rules for shared modules; freeze risky lm_head/norm if needed. |
| 5_sparse_capacity_gate | Does sparse top-k dispatch overload experts under the deployment capacity factor? | top-k expert counts, capacity ratio, overflow fraction | Capacity-aware score beats route-KD/calibrated baselines or remains on the Pareto frontier. | Increase capacity guard, keep route-KD alternative, or reject deployment candidate. |
| 6_candidate_acceptance | Does the materialized checkpoint beat baselines on held-out tasks? | held-in retention, worst score, format safety, cost | Beat all-weight average and avoid endpoint-only pseudo-success. | Promote candidate only after held-out eval. |

## 对当前仓库的直接影响

- `results/average_decision_report/report.md` 负责把 Dense/Qwen merge grid、barrier 和 conflict probe 变成是否 materialize 的决策。
- `results/moe_routing_readiness/report.md` 是 MoE 的 router gate；没有真实 routing probe 时，MoE route-weight recipe 必须保持 `waiting_for_routing_probe`。
- `results/toy_moe_expert_remap_plan/source_tensor_aliases.txt` 对应 MoE 优化路线里的 expert alignment stage：它不改变输出结构，只改变 source tensor 的读取坐标。
- 下一步真实 Qwen3 MoE 实验应优先补 `Qwen3-30B-A3B-Base`、`Qwen3-30B-A3B`、`Qwen3-Coder-30B-A3B-Instruct` 的 route traces、expert output similarity 和 held-out NLL，而不是先做全权重平均。

## 文件

- `results/model_averaging_literature_review/method_matrix.csv`
- `results/model_averaging_literature_review/probe_matrix.csv`
- `results/model_averaging_literature_review/moe_optimization_matrix.csv`
- `results/model_averaging_literature_review/source_matrix.csv`
- `results/model_averaging_literature_review/summary.json`

## Sources

- Fisher merging (2021): [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832)
- Model soups (2022): [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)
- Task arithmetic (2022): [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)
- Git Re-Basin (2022): [Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)
- ZipIt (2023): [ZipIt! Merging Models from Different Tasks without Training](https://arxiv.org/abs/2305.03053)
- TIES (2023): [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- AdaMerging (2023): [AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/abs/2310.02575)
- DARE (2023): [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)
- DELLA (2024): [DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling](https://arxiv.org/abs/2406.11617)
- Model merging survey (2024): [Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities](https://arxiv.org/abs/2408.07666)
- DAM (2024): [Merging in a Bottle: Differentiable Adaptive Merging and the Path from Averaging to Automation](https://arxiv.org/abs/2410.08371)
- WEMoE (2024): [Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging](https://arxiv.org/abs/2410.21804)
- MergeME (2025): [MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs](https://arxiv.org/abs/2502.00997)
- STAR (2025): [STAR: Spectral Truncation and Rescale for Model Merging](https://arxiv.org/abs/2502.10339)
- Qwen3 (2025): [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- Sub-MoE (2025): [Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266)
- FroM (2025): [FroM: Frobenius Norm-Based Data-Free Adaptive Model Merging](https://arxiv.org/abs/2506.02478)
- RegMean++ (2025): [RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging](https://arxiv.org/abs/2508.03121)
- Merge scaling laws (2025): [Model Merging Scaling Laws in Large Language Models](https://arxiv.org/abs/2509.24244)
- MergeMoE (2025): [MergeMoE: Efficient Compression of MoE Models via Expert Output Merging](https://arxiv.org/abs/2510.14436)
- Output-space projection (2026): [Model Merging by Output-Space Projection](https://arxiv.org/abs/2605.29101)
- HARC (2026): [When Model Merging Breaks Routing: Training-Free Calibration for MoE](https://arxiv.org/abs/2606.03391)
