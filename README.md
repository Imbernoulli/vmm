# Visualizing Model Merging：任务向量空间中的模型合并可视化

> **最新（真实第一性原理实证，2026-06）：** 见 **[第一性原理发现：为什么线性平均失败 & MoE 该怎么合并](FIRST_PRINCIPLES_FINDINGS.md)**（真实权重+真实数据，非 plan/recipe）。核心四点：(1) dense 线性平均失败的本质是 loss **barrier**——二阶曲率定律低估真实退化 30–43×；(2) MoE 有精确的 **expert gauge 对称性**，真实 OLMoE-1B-7B 上 naive 同名平均把 NLL 从 `2.857` 砸到 `9.975`，weight-align 后完美恢复；(3) Qwen3-30B-A3B Instruct vs Coder 在**全部 48 层天然 index-aligned**（首次在真实 LLM 尺度上测量）；(4) **同一对能力 dense 不可合并、MoE 却能平滑合并**（worst-NLL barrier `≈3.0`→`0.11`），因为 routing 把干扰局部化。脚本 `scripts/fp_*.py`，结果 `results/fp_*`，图 `results/fp_figures/`。

这份仓库把 `proposal.md` 里的想法实现成了一个可运行的研究 artifact：从小型图像分类模型开始，逐步扩展到 ViT/pretrained ViT 和 Qwen 系列 LLM，观察模型合并点在任务向量空间中的位置、多个任务 basin 是否重叠，以及合并失败是否和 task-vector interference 有关。

后续 Qwen Dense/MoE 和下游微调模型的实验设计见：[Qwen Dense/MoE 下游微调模型合并实验方案](QWEN_DENSE_MOE_EXPERIMENT_PLAN.md)，结构化目标模型清单见：[Qwen Target Model Registry](results/qwen_target_model_registry/report.md)。Averaging 失败诊断、probe 清单和 MoE route-aware averaging 路线见：[Dense/MoE Model Averaging 的指标、Probe 与优化路线](MODEL_AVERAGING_PROBES_AND_MOE_OPTIMIZATION.md)。当前已有实验的同构 Average 决策汇总见：[Average Decision Report](results/average_decision_report/report.md)，probe-gated unified action plan 见：[Probe-Gated Unified Average Plan](results/probe_gated_unified_average_plan/report.md)，Dense/MoE 文献和 probe 矩阵见：[Model Averaging Literature Review](results/model_averaging_literature_review/report.md)，candidate materialization 选择见：[Average Candidate Recipes](results/average_candidate_recipes/report.md)，probe-guided dense candidate 见：[Probe-Guided Dense Average Candidate](results/probe_guided_dense_average_candidate/report.md)，dense module/norm guard ablation 见：[Qwen Dense Module-Guarded Candidate](results/qwen_dense_module_guarded_candidate/report.md)、[Qwen Dense Norm-Guarded Candidate](results/qwen_dense_norm_guarded_candidate/report.md) 和 [Qwen Dense Selective-Norm Candidate](results/qwen_dense_selective_norm_guarded_candidate/report.md)，dense sparse-method candidate 见：[Qwen Dense Sparse-Method Candidate](results/qwen_dense_sparse_method_candidate/report.md) 和 [Qwen Dense Attention Sparse-Method Candidate](results/qwen_dense_attention_sparse_method_candidate/report.md)，MoE 拓扑检查见：[Checkpoint Topology Inspect](results/checkpoint_topology_inspect/report.md)，Qwen3 MoE unified preflight 见：[Qwen3 MoE Unified Average Preflight](results/moe_unified_preflight_qwen3_30b/report.md)，Qwen3 routing gate 见：[Qwen3 MoE Routing Readiness](results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/report.md)，Qwen3 route-guarded candidate 见：[Qwen3 MoE Route-Guarded Candidate](results/qwen3_moe_unified_route_guarded_candidate/report.md)，Qwen3 materialized delta audit 见：[Materialized Checkpoint Delta Audit](results/qwen3_moe_materialized_delta_audit/report.md)，Qwen3 audit-gated candidate 见：[Qwen3 MoE Audit-Gated Candidate](results/qwen3_moe_audit_gated_candidate/report.md)，Qwen3 trust-region candidate 见：[Qwen3 MoE Trust-Region Candidate](results/qwen3_moe_trust_region_candidate/report.md)，Qwen3 trust-region delta audit 见：[Qwen3 MoE Trust-Region Delta Audit](results/qwen3_moe_trust_region_delta_audit/report.md)，Qwen3 trust-region delta validation 见：[Qwen3 MoE Trust-Region Delta Validation](results/qwen3_moe_trust_region_delta_validation/report.md)，真实 MoE materialization gate 见：[MoE Materialization Pipeline Plan](results/moe_materialization_pipeline_plan/report.md)，MoE 参数组计划见：[MoE Same-Shape Average Plan](results/moe_average_plan/report.md)，MoE routing 风险诊断见：[MoE Routing Readiness](results/moe_routing_readiness/report.md)，MoE route-weight tensor rules 见：[MoE Route-Weight Recipes](results/moe_route_weight_recipes/report.md)，MoE router-bias capacity recipe 见：[MoE Router Bias Plan](results/moe_router_bias_plan/report.md)，confidence-blended router-bias recipe 见：[MoE Confidence-Blended Router Bias Plan](results/moe_confidence_blended_router_bias_plan/report.md)，confidence-blended combined writer recipe 见：[MoE Confidence-Blended Combined Recipe](results/moe_confidence_blended_combined_recipe/report.md)，combined writer 数值 smoke 见：[MoE Combined Writer Smoke](results/moe_combined_writer_smoke/report.md)，layer-wise expert remap smoke 见：[MoE Layer-Wise Expert Remap Smoke](results/moe_layerwise_expert_remap_smoke/report.md)，checkpoint materialization readiness 见：[Checkpoint Materialization Readiness](results/checkpoint_materialization_readiness/report.md)，toy MoE 验证见：[Toy MoE Route-Aware Merge](results/toy_moe_merge/report.md)，toy expert 重排写出计划见：[Toy MoE Expert Remap Plan](results/toy_moe_expert_remap_plan/report.md)，checkpoint 写出器 smoke 见：[Same-Shape Checkpoint Writer Smoke](results/same_shape_writer_smoke/report.md)。

这里说的 Average 不是 ensemble，也不是把 MoE experts 扩成更多分支；最终目标模型必须和输入模型保持同构，能用同一个 config/tokenizer/model class 直接加载。Probe 的作用是决定哪些模型、层、模块或 experts 可以被平均，以及平均系数应该怎么设。

当前最实在的 Qwen3 MoE 结果是：我已经把 route/load probe、category specialization、router fragility 和真实 delta audit 合成一个同构 trust-region checkpoint，并新增一个 [expert-only attention ablation](results/qwen3_moe_expert_only_trust_region_candidate/report.md) 去隔离 shared attention 的作用；随后又物化了 [tail-trimmed expert-only candidate](results/qwen3_moe_tail_trimmed_expert_only_candidate/report.md)，只对剩余 high-tail routed experts 做二次收缩。tail-trimmed 的实际 [delta audit](results/qwen3_moe_tail_trimmed_delta_audit/report.md) 通过：attention `0/288` changed、router `0/48` changed、total relative delta norm `0.243`、routed `>0.75` 为 `0`。现阶段不能提前说更保守就一定更好；真正的算法优劣还要看 Qwen3 Instruct/Coder source、route-guarded、audit-gated、trust-region、expert-only 和 tail-trimmed 在同一套下游任务上的表现。

最新 [Qwen3 MoE Delta Frontier Probe](results/qwen3_moe_delta_frontier/report.md) 把五个已物化 candidate 的真实 safetensors delta 放到同一张表里。结论很明确：route-guarded -> audit-gated 主要消掉 routed expert 的危险大步长（`>1.0` 从 `182` 到 `0`，`>0.75` 从 `839` 到 `164`）；audit-gated -> trust-region 继续把 `>0.75` 从 `164` 压到 `14`；trust-region -> expert-only 不再改善 routed expert tail，只是把 attention delta 从 `0.189` 清零；expert-only -> tail-trimmed 才继续压 routed tail，把 `>0.65` 从 `366` 降到 `80`（剩余都在 `0.6501` rounding slop 内）。因此下一版 unified MoE 规则里，trust-region/tail-trim 是安全机制，attention 是否保留是效用问题，必须交给同任务 vLLM eval 决定。

最新 [Qwen3 MoE Mechanism-Gated vLLM Eval Gate](results/qwen3_moe_mechanism_eval_gate/report.md) 已把这个问题转成可执行评测：两个 source endpoint 加五个 same-shape Qwen3 MoE candidates 都是 `ready_to_host`，并生成 `run_eval_gate.sh` 逐个 `vllm serve -> eval -> stop`。这个 gate 不再问“哪个算法名最好”，而是逐项检验 tail delta cap、route/load trust-region、shared attention ablation、0.65 tail trim 和 endpoint fallback；如果所有 average 候选被 source 支配，selection rule 会返回同构 endpoint/no-average。当前本机 `nvidia-smi` 不可用，所以 selection 状态是 `awaiting_remote_vllm_eval`。

最新 [Qwen3 MoE Trust-Region Cap-Law Search](results/qwen3_moe_trust_region_cap_search/report.md) 又把 trust-region 规则本身做了一次内部优化搜索：在 `5,243` 个真实 expert groups 上搜索 `432` 个可解释 cap law，指标是“降低高 relative-delta routed experts，同时保留 route-mass-weighted Coder contribution”。结果反而提醒我们要简化：当前手写 risk penalties 的 retention 是 `0.9818`，仍有 `129` 个 group 高于 `0.65`；简单 uniform `0.65` cap 的 retention 是 `0.9823`，高于当前规则且 `>0.65` groups 为 `0`。这不是下游性能结论，但说明当前复杂 risk penalty 在 delta-threshold 上不够高效，下一轮 vLLM 应重点比较 expert-only / tail-trimmed / 简化 cap law。

最新 Dense 机制结果：我用真实 Qwen2.5-0.5B Instruct/Coder 权重跑了一个 [Dense Curvature-Displacement Probe](results/fp_curvature_law/report.md)，直接比较 diagonal-Fisher 二阶预测和真实 A->B interpolation loss。uniform midpoint 的 general NLL 从 `3.099` 升到 `5.911`，code NLL 从 `0.515` 升到 `1.747`；但 Fisher 二阶预测的 degradation 只有 `0.0656/0.0462`，真实 degradation 是 `2.812/1.232`，actual/predicted ratio 分别是 `42.86` 和 `26.66`。这说明当前 Qwen instruct/coder midpoint 不是一个局部小凸二次问题，而是明显跨了非局部 barrier；Fisher merge 的 worst NLL `5.249` 虽然比 uniform `5.911` 好，但仍在高 loss 区域。因此 Dense 侧统一规则也不能是“直接 Fisher/直接 0.5 平均”，而要先用 path/eval probe 判断 barrier，再做 coefficient、layer 或 tensor gate。

最新 unified Dense selector 结果：我把 linear average、task arithmetic、sign-elect、magnitude-weighted merge 写进同一个候选族 `base + lambda * combine(delta_A, delta_B)`，再用 held-out worst-task NLL 选择，结果见 [Unified Merge Family Probe](results/fp_merge_compare_dense/report.md)。在 Qwen2.5-0.5B Instruct/Coder/Base 的小样本 probe 上，selector 选中的是 `lambda=0.0`，也就是拒绝吸收两个 task delta，退回 base anchor；test worst NLL 是 `5.183`，略差于 best endpoint `5.151`，但明显好于 linear midpoint `8.948` 和 TIES baseline `9.110`。这不是“已经找到超过所有端点的平均模型”，而是机制上很关键：unified 方法必须允许输出“不合并/少合并”，否则就会被 midpoint ridge 和错误的 sparse conflict rule 拖坏。

最新 Dense 生成式 smoke 结果：我又把同一组 Qwen2.5-0.5B 模型放到一个安全的 exact-answer generation eval 上，结果见 [Generation Exact-Answer Merge Eval](results/fp_gen_eval_dense/report.md)。这个 smoke 不下载数据集，也不执行模型生成的代码，只问 2 个 math exact-answer 和 2 个 code-output exact-answer。linear midpoint 的 avg/worst accuracy 是 `0.000/0.000`，直接崩掉；unified `lambda=0` 是 `0.500/0.000`，和 base/instruct 类似，至少避免了 midpoint 生成退化；coder 是 `0.500/0.500`，在这个极小切片上 worst 最好。这个结果没有证明 unified 已经赢过 endpoint，反而说明 selector 还需要把生成式 held-out gate 纳入选择，否则只靠 NLL 会倾向保守退回 anchor。

最新第一性原理 MoE 机制 probe：我把一个训练好的 B 模型做了函数等价的 expert/router row 置换，B 的输出几乎不变（probe MSE `7.66e-16`），但同名 expert index 的语义被打乱。结果见 [MoE Average Mechanism Probe](results/fp_moe_mechanism/report.md)：same-name uniform average 的 worst-domain MSE 是 `0.5105`；用 expert-output cosine + Hungarian 恢复语义对齐后降到 `0.1252`；再只校准 router 降到 `0.1095`。相反，aligned Fisher 在这个设置里升到 `0.1520`，说明 curvature/Fisher 不是无条件更好，必须经过 held-out gate。这个实验给 unified average 的形式一个机制解释：MoE 必须先对齐 expert identity，再决定 expert 权重，router 只能在 experts 已经对齐后小步校准或蒸馏；不能指望 router 单独修掉错配 experts，也不能机械套 Fisher/TIES。

最新真实 MoE 反事实：我把 [Real MoE Expert-Gauge Self-Merge Probe](results/fp_moe_real_probe/report.md) 跑在 `allenai/OLMoE-1B-7B-0924-Instruct` 上。这个模型是 packed MoE：`16` 层、每层 `64` experts。同步置换 expert slices 和 router rows 后，NLL 基本不变（`4.1678 -> 4.1656`），说明置换确实函数等价；但把原模型和这个等价模型按同名 tensor average，NLL 直接升到 `9.6588`；恢复 expert permutation 后 average 回到 `4.1678`，`16/16` 层精确恢复。这是真实 MoE LLM 上的证据：即使两个 checkpoint 表示同一个函数，只要 expert gauge 不一致，同名 average 也会失败。同目录下的 Qwen3 Instruct/Coder cross-correspondence 显示官方这对模型 `48/48` 层 identity-optimal，mean diagonal cosine `0.183`、off-diagonal `0.00014`；所以这对 Qwen3 可以先用 identity mapping，但 expert-alignment gate 不能删。

最新 MoE selector 结果：我把真实 OLMoE gauge 反事实、Qwen3 Instruct/Coder expert correspondence、Qwen3 真实 route/load probe 和 toy MoE route/capacity selector 合成了 [MoE Probe-Gated Selector](results/moe_probe_gated_selector/report.md)。当前规则是：全局上 `reject_same_name_average_without_alignment`，因为 OLMoE 同名 average 的 NLL degradation 是 `5.491`，aligned degradation 是 `0.000`；对 Qwen3 这对官方同族模型，expert identity gate 通过，因为 identity-optimal layers 是 `1.000`、argmax identity 是 `1.000`、diag/offdiag cosine ratio 约 `1339.5`；同构 preflight 也通过。但真实 routing readiness 显示直接 router average 高风险，selector 决策已变成 `reject_direct_router_average_calibrate_or_freeze`，下一步 blocker 是 `materialized_route_guarded_candidate_vllm_eval`。也就是说，Qwen3 experts 可以先按 identity 对齐，但 router 必须 freeze/小步校准/route-KD 或做 capacity correction，不能直接平均。

最新 Qwen3 MoE unified preflight：我把 [Qwen3 MoE Unified Average Preflight](results/moe_unified_preflight_qwen3_30b/report.md) 跑在本地缓存的 `Qwen3-30B-A3B-Instruct-2507` 和 `Qwen3-Coder-30B-A3B-Instruct` 上，只读 config 和 safetensors header，不加载 30B 权重内容。结果是同构合同通过：`48` 层、每层 `128` experts、top-k `8`、`48` 个 router tensor 全部同形，`18,432` 个 routed expert tensor layout 全部匹配；前面的 expert correspondence 也通过 identity gate。这个结果把 unified 方法的边界收窄了：Qwen3 这对模型不再卡在结构或 expert identity；后续真实 routing probe 已经补上，剩下的关键问题是 route/load 决策是否能在下游 eval 中保住源模型能力。

最新 Qwen3 MoE 真实 routing probe：`results/moe_routing_probe/qwen3_30b_instruct_vs_coder` 已经完成 `12` 个 prompts、`8` 类场景、`48` 个 router 的 route overlap 和 expert load 捕获；[Qwen3 MoE Routing Readiness](results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/report.md) 给出的状态是 `high_risk_calibrate_router_before_merge`。关键数字是：top-k Jaccard mean/min 为 `0.454/0.242`，top1 agreement mean/min 为 `0.413/0.069`；`576` 个 router/prompt slice 中，`493` 个要求 `calibrate_router_before_average`，只有 `31` 个直接通过 small-lambda gate，另有 `6` 个因 load concentration 建议 freeze/check load balance。expert load 侧有 `4,459` 行 high-load expert 需要保护或 source-weight 处理。这个结果解释了为什么 unified MoE 不能只靠 expert identity：expert index 对齐解决的是 gauge，route overlap 和 load 才决定 router 能不能动。

最新 Qwen3 MoE route-guarded candidate：我把上面的真实 route/load probe 转成 [Qwen3 MoE Route-Guarded Candidate](results/qwen3_moe_unified_route_guarded_candidate/report.md)，并已在本地物化成 same-shape checkpoint `results/checkpoints/qwen3_moe_unified_route_guarded_candidate`（57G，本地忽略，不进 git）。这个候选 base 选 Instruct，本地 Coder 只作为 source delta；router 先 `--freeze-router`，shared attention 只给 coder 一个 `0.25` 小步长，expert FFN 按真实来源模型的 route mass 生成 source-route-conditioned tensor rules。结果是 `5,243` 个 expert 规则、`35,432` 行 route mass 被使用、`73,728` 行不属于目标 source/category 的路由被跳过；真实写出 `16` 个 safetensors shard 和 `model.safetensors.index.json`，header 检查显示输出 `18,867` 个 tensor 与 base 完全同名、同 shape、同 dtype。这个 candidate 还不是最终答案；下一步必须在 GPU/vLLM host 上跑下游任务，并和两个 source endpoint 对照。

最新 Qwen3 MoE materialized delta audit：我新增 [Materialized Checkpoint Delta Audit](results/qwen3_moe_materialized_delta_audit/report.md)，直接读取已物化 safetensors，而不是只看 writer manifest。结果是 `passed`：输出 checkpoint 的 `18,867` 个 tensor 同构，实际改变 `10,641` 个 tensor、约 `56.3%` 参数量，总 relative delta norm `0.286`；关键是 `48/48` 个 router tensor 全部未变，embedding/head 和 norm 也未变。改动集中在 `10,353/18,432` 个 routed expert FFN tensor（relative delta norm `0.293`）和全部 `288` 个 attention tensor（relative delta norm `0.189`）。这验证了当前 candidate 的机制确实是“冻结 router、保护结构、只在 source-route-conditioned experts 和小步 attention 上移动”。

最新 Qwen3 MoE audit-gated candidate：基于上面的真实 delta audit，我新增 [Qwen3 MoE Audit-Gated Candidate](results/qwen3_moe_audit_gated_candidate/report.md)，并已物化成 `results/checkpoints/qwen3_moe_audit_gated_candidate`（57G，本地忽略）。它保留 route-frequency expert rule，但对实际 FFN relative delta norm 超过 `0.75` 的 expert 按比例缩小非 base source（Coder）delta weight；`5,243` 个 expert rules 中 `302` 个被缩小，mean effective nonbase weight 从 `0.201` 降到 `0.193`，最大 audited relative delta 原本是 `1.327`，最强缩放系数是 `0.565`。新的 [audit-gated delta audit](results/qwen3_moe_audit_gated_delta_audit/report.md) 证明这个约束已经进入真实 safetensors：总 relative delta norm 从 `0.286` 降到 `0.264`，routed expert FFN relative delta norm 从 `0.293` 降到 `0.270`，routed expert 单 tensor 最大 relative delta 从 `1.327` 降到 `0.750`，`>1.0` 的 routed tensors 从 `182` 降到 `0`；router 仍是 `0/48` 改动。这个候选是更保守的下一版，不替代 vLLM eval；它的作用是把“probe 发现少数 expert 移动过大”转成可物化的规则。

最新 Qwen3 MoE trust-region candidate：我新增 [Qwen3 MoE Trust-Region Candidate](results/qwen3_moe_trust_region_candidate/report.md)，把 route/load readiness、category specialization、router fragility 和 materialized delta audit 合成同一个 expert-level trust region。它不是统一套一个 `0.75` cap，而是对高负载 expert、shared/mixed expert、fragile-router layer、低 route 证据和 category/source mismatch 分别降低 target cap，只缩小非 base source（Coder）delta，router 仍冻结。结果是 `5,243` 个 expert rules 中 `405` 个被缩小，其中 `103` 个不是因为 delta 已超过 `0.75`，而是被 MoE 内部风险信号额外触发；mean effective nonbase weight 从 `0.201` 降到 `0.186`。它已物化成 `results/checkpoints/qwen3_moe_trust_region_candidate`（57G，本地忽略），真实 Qwen3 writer manifest 显示 `18,867` 个 floating tensors，expert/attention/router hits 分别是 `15,729/288/48`。新的 [trust-region delta audit](results/qwen3_moe_trust_region_delta_audit/report.md) 证明约束进入真实 safetensors：总 relative delta norm `0.249`，routed expert FFN relative delta norm `0.255`，routed tensor 最大 relative delta `0.750`，`>1.0` 为 `0`、`>0.75` 为 `14`、`>0.65` 为 `366`；router 仍是 `0/48` 改动。新的 [trust-region delta validation](results/qwen3_moe_trust_region_delta_validation/report.md) 进一步逐 tensor 对齐预测和真实 audit：max abs relative-delta prediction error 只有 `0.000093`，P99 是 `0.000007`，没有 tensor 超过 `0.002` tolerance；那 `14` 个略高于 `0.75` 的 routed tensors 全在 `0.751` rounding slop 内。下一步是 vLLM 下游评测。

最新 MoE 机制结果：toy MoE 上，失败主因不是“平均”这个动作本身，而是 expert 语义 index、router dispatch 和 expert load capacity 一起漂移。旧的 `unified_moe_average` 先用 per-expert source weight search 处理 expert 语义/重要性，再对 router seed 和 capacity loss 系数做 held-out selection sweep；它的 soft worst accuracy 是 `0.785`，hard top-2 worst accuracy 是 `0.690`，max top-k overflow 是 `0.0775`。单独把 expert seed 换成 route-conditioned output-space projection 后，`unified_output_projection_moe_average` 的 soft worst accuracy 升到 `0.795`、overflow 降到 `0.0700`，但 hard top-2 worst accuracy 降到 `0.685`，说明 projection 在 soft 输出空间有用，但会在某些 sparse expert 上过度移动。新的 `unified_confidence_blended_moe_average` 用 output-space projection 的 `captured_fraction` 当 expert-level 置信度：projection 能解释该 expert 输出残差时更信 projection 权重，解释不了时退回 calibration search 权重。结果是 soft worst accuracy `0.790`、hard top-2 worst accuracy `0.690`、max top-k overflow `0.07625`，即在不损失旧 unified hard sparse 分数的情况下提升 soft 分数并略降 overflow。严格 capacity-aware sparse 下，`unified_moe_bias_capacity_average` 仍是当前推荐：hard top-2 worst accuracy `0.6825`、max top-k overflow `0.0475`、`accuracy - overflow = 0.635`。这些机制已经落到 checkpoint recipe：router bias 用 `scripts/build_moe_router_bias_plan.py` 生成 `tensor,index,delta`；旧 unified 和 confidence-blended unified 都有 writer-ready router-bias plan；expert-search、output-projection 和 confidence-blended expert 权重分别写成 [searched expert tensor rules](results/toy_moe_expert_weight_recipes/report.md)、[output-projection expert tensor rules](results/toy_moe_output_projection_recipes/report.md) 和 [confidence-blended expert tensor rules](results/toy_moe_confidence_blended_recipes/report.md)。新的 [combined writer recipe](results/moe_confidence_blended_combined_recipe/report.md) 把 5 条 tensor rule、4 条 expert alias rule 和 4 行 router-bias delta 合成同一个 same-shape writer command；[combined writer smoke](results/moe_combined_writer_smoke/report.md) 已用 swapped experts 数值验证 alias、expert rule、freeze-router 和 bias delta 能在一次写出里同时生效。

这个结果也解释了为什么不能只机械套一个低-overflow router：`unified_router_kd_seed_average` 把 Router-KD router 直接接到 expert-search 权重上后，hard top-2 code accuracy 只有 `0.5975`，说明 router prior 和 expert 权重必须共同校准。output-space projection probe 的 mean captured fraction 是 `0.616`；`expert_output_projection_router_calibrated_average` soft worst accuracy 达到 `0.8075`，是当前 soft-router 最优，但 hard top-2 worst accuracy 只有 `0.6475`。confidence blend 后，expert 0/3 因 captured fraction 高而主要采用 projection 权重，expert 1/2 因 captured fraction 低而更多回退到 search 权重；这就是它能支配旧 unified seed、但不假装支配所有 Pareto 点的原因。

最新真实 vLLM source-vs-merge 结果：本地已把 Qwen2.5-0.5B base、Qwen2.5-0.5B-Instruct、Qwen2.5-Coder-0.5B-Instruct 和 materialized `qwen_0_5b_instruct_coder_uniform_average` 都用 vLLM host，并在 GSM8K、MMLU、safety、HumanEval compile 各 `64` 个样本上跑完同一套下游评测。结果见 [Qwen Source-vs-Merge vLLM Comparison](results/vllm_source_merge_comparison/report.md)：base avg primary `0.375`、instruct `0.227`、coder `0.199`、uniform average `0.180`，uniform average 在 4 个模型里排第 4，低于最佳源模型 `0.195`。逐任务看，它比最佳源模型分别低 GSM8K `0.094`、MMLU `0.125`、safety `0.047`、HumanEval compile `0.609`；safety 的 `0.500` 也不是好现象，因为 safe non-refusal 是 `1.000`，unsafe refusal 是 `0.000`，相当于几乎不拒绝 unsafe prompts。这说明 0.5/0.5 Dense uniform average 不只是 probe 上可疑，真实 endpoint 对照下也被三个源模型同时支配；下一步应做 probe-guided same-shape average，而不是继续盲目平均。

最新 probe-guided dense average 结果：基于 Qwen instruct/coder NLL grid，脚本选择了同构 bridge candidate `qwen_0_5b_probe_guided_bridge_a025_b100`，即 `alpha=0.25,beta=1.0`，不是端点复制，也不是 `0.5/0.5`。NLL probe 上它相对 uniform midpoint 的 worst NLL 降低 `1.921`、avg NLL 降低 `2.551`；写出 checkpoint 后真实 vLLM eval 得到 avg primary `0.203`，比 uniform `0.180` 高 `0.023`，GSM8K 从 `0.000` 到 `0.062`，MMLU 从 `0.219` 到 `0.250`，unsafe refusal 从 `0.000` 到 `0.500`。但它仍低于 best source `0.375`，HumanEval compile 仍是 `0.000`。结论是：global scalar coefficient 已经能验证“避开 midpoint ridge”这个机理，但还不够；下一步要做 layer/module-wise weighting，而不是继续调一个全局数字。

最新 module-wise dense ablation 结果：Qwen instruct/coder 的 task-vector 冲突不是均匀分布的，`norm_anchor` 虽然只有 `43,904` 个参数，但 mean tensor cosine 是 `-0.164`、sign conflict 约 `0.441`，说明 layernorm/最终 norm 的尺度方向在两个微调源之间有明显反向成分。直接把这个 probe 机械地扩展成“冻结 embedding/norm、阻尼 MLP”的 `qwen_0_5b_module_guarded_bridge` 反而变差，真实 vLLM avg primary 从 global bridge 的 `0.203` 降到 `0.160`；只冻结全部 norm 的 `qwen_0_5b_norm_guarded_bridge` 与 global bridge 打平为 `0.203`，但任务分布改变：GSM8K/MMLU 分别低 `0.016/0.031`，safety 高 `0.047`；只冻结 6 个极端 post-attention norm 的 `qwen_0_5b_selective_norm_guarded_bridge` 也没有变好，avg primary 是 `0.191`。结论是：module conflict probe 有诊断价值，但 action 必须窄化并做 ablation；当前证据说明 norm 尺度效应更像全局耦合，不是“找到几个最高冲突 tensor 冻住”就能解决。

最新 dense sparse-method candidate 结果：新的 sparse-coordinate 实验不再问“哪个场景套哪个算法”，而是把机制拆到坐标级。`qwen_0_5b_sparse_method_bridge` 沿用 global bridge 的 `instruct=0.25,coder=1.0`，再从 Qwen instruct/coder 的 tensor conflict probe 中选出 sign conflict `>=0.44`、delta cosine `<=0.16`、参数量 `>=100k` 的 attention/MLP tensor，对这些 tensor 执行 TIES-style coordinate trim/sign-elect/merge。它选中 `99` 个 tensor、覆盖约 `48.73%` 参数，真实 vLLM avg primary 是 `0.156`，比 global bridge 低 `0.047`，说明对 MLP 大范围做 TIES 会破坏能力。收窄后的 `qwen_0_5b_attention_sparse_method_bridge` 只选 attention tensor：`49` 个 tensor、覆盖约 `4.62%` 参数，真实 vLLM avg primary 是 `0.203`，与 global bridge 打平、比 uniform 高 `0.023`；任务分布是 safety 高 `0.062`，GSM8K/MMLU 分别低 `0.047/0.016`。当前机制结论是：global coefficient 负责避开 midpoint ridge，coordinate conflict rule 不能粗暴作用到 MLP 语义容量；attention-only 是更安全的窄 intervention，但还没有支配 global bridge。

## 一屏版结论

如果只想先看结果，可以先读这几条：

1. **Digits 是最干净的正例。** 低 worst-loss 区域沿着两个任务都能接受的 valley 展开，base、linear average、best grid 都在这个 valley 附近；两个 expert endpoint 各自只擅长一个任务，所以在 worst-loss 图上反而是高处。
2. **CIFAR-10 是 naive average 失败例。** linear average 没有落到最好的共同区域，validation grid best 明显更靠近低 worst-loss 区域；这说明 merge coefficient 不能总用 `0.5,0.5`。
3. **Pretrained ViT transfer 是“model soup 直觉”更成立的例子。** shared low-loss basin 更宽，linear average 已经不错，grid best 还能略好一点。
4. **Qwen instruct/coder multi-expert 是最值得警惕的例子。** `alpha=0.5,beta=0.5` 的 linear average 落在高 worst-NLL ridge 上；instruct endpoint / best 比平均好得多，说明 LLM expert merge 不能只做朴素平均。
5. **不是所有有效方法都真的在这个二维 plane 上。** base、expert、linear average、task arithmetic、grid point 是 raw task-vector plane 里的点；RegMean、layer-wise task arithmetic 这类方法可能离开这个 plane。展示图里这类点用 `projected` 标记，表示它们只是投影到 `alpha,beta` 平面上看位置。
6. **这些图看起来有些凸，不是理论假设。** 这只是当前 same-base task-vector slice 和 worst-loss/NLL 指标在选定范围内的形状。Li et al. 那类 loss-landscape 图用的是随机或 filter-normalized 方向，目标是展示局部训练地形；这里的方向是任务语义方向，回答的是 merge 几何，所以图形不必长得一样。
7. **Average 决策现在由 probe 输出驱动。** [Average Decision Report](results/average_decision_report/report.md) 把 merge grid、conflict probe 和可选 MoE routing probe 汇总成同构 checkpoint 的权重建议；当前 Qwen instruct/coder 被标成 `avoid_uniform_average`，建议先做 connectivity/barrier 筛选再重学平均权重。
8. **Dense/MoE 文献矩阵已经转成工程规则。** [Model Averaging Literature Review](results/model_averaging_literature_review/report.md) 整理了 `22` 篇 Dense averaging、task-vector、conflict-aware merge、output-space projection、MoE routing/expert merging 相关来源，并映射成 `7` 类方法、`7` 类 probe 和 `7` 个 MoE 优化 gate。
9. **Qwen 目标模型已经落成 registry。** [Qwen Target Model Registry](results/qwen_target_model_registry/report.md) 现在列出 `17` 个候选条目：dense `12`、MoE `5`；其中包含官方 Qwen 分支、DeepSeek/AM 等第三方下游模型、DianJin/Long1K 论文候选和下游 adapter/finetune 候选池。第一轮建议先跑 `Qwen2.5-7B-Instruct + Coder + Math + DeepSeek-R1-Distill-Qwen-7B`。
10. **MoE 不能把 router/expert 当普通 dense 层平均。** [MoE Same-Shape Average Plan](results/moe_average_plan/report.md) 把 router、shared modules、expert FFN、embedding/lm_head 和 LoRA/adapters 分开：默认先冻结/校准 router，experts 先按 route frequency/output similarity 匹配，再写回同 expert 数的 checkpoint。
11. **同构 checkpoint 写出路径已经打通。** `scripts/write_same_shape_average_checkpoint.py` 会先验证 tensor name/shape，再按 `base + sum_i w_i * (source_i - base)` 写 safetensors；现在支持 `--tensor-method-rule` 对高 sign-conflict tensor 做 TIES-style coordinate trim/sign-elect/merge，支持 `--tensor-add-csv` 写 router-bias scalar delta，也支持 `--packed-expert-rule-csv` 对 Qwen-style packed expert tensor 的第 0 维做 slice-level source expert remap/weight。Qwen2.5-0.5B base/instruct/coder dry-run 已检查 `290` 个 tensor 无缺失、无 shape mismatch；本地还已写出 `qwen_0_5b_instruct_coder_uniform_average` 这个 0.5/0.5 Dense 负 baseline checkpoint，用于验证真实 host/eval 管线。writer smoke 现在覆盖 Dense sparse method、freeze-router、bias additive correction、展开式 expert alias 和 packed expert slice rule。
12. **MoE 大模型先做 header/config probe。** [Checkpoint Topology Inspect](results/checkpoint_topology_inspect/report.md) 已对本地完整 `Qwen3.6-35B-A3B` safetensors 做 header-only 拓扑检查，不加载 67G 权重内容：`qwen3_5_moe`、`40` 层、`256` experts、每 token 激活 `8` 个，active fraction `0.03125`；真实权重里 routed experts 是 packed tensor 形式，`82` 个 `routed_expert` tensors、`66,035,122,176` bytes，router 是 `41` 个 tensors。这说明 expert remap 不能再假设 `experts.17.*` 这种展开式名字；[MoE Packed-Expert Writer Smoke](results/moe_packed_expert_writer_smoke/report.md) 已验证保持 tensor 名字/shape 不变时，能对 packed expert 第一维写入 matched source expert slice。
13. **不是每个 best grid 都值得写 checkpoint。** [Average Candidate Recipes](results/average_candidate_recipes/report.md) 明确把当前 Qwen instruct/coder best grid 标成 `skip_endpoint_only`，因为 `alpha=1,beta=0` 只是端点，不是有价值的 average；`0.5/0.5` uniform average 也被 probe 标成负 baseline。
14. **Qwen3 MoE route-aware average 已落到真实 tensor-rule 文件。** [Qwen3 MoE Route-Guarded Candidate](results/qwen3_moe_unified_route_guarded_candidate/report.md) 用真实 Instruct/Coder route/load probe 生成 `5,243` 个 expert 规则；writer dry-run 已在真实 Qwen3 safetensors 上命中 `15,729` 个 expert FFN tensor、`288` 个 attention tensor，并冻结 `48` 个 router tensor。它不是扩 expert，也不是 ensemble，输出 checkpoint 仍保持原结构。
15. **真实 MoE routing probe CLI 已经跑过 Qwen3。** `scripts/probe_moe_routing.py` 会捕获 MoE router hook，输出 `router_summary.csv`、`expert_load.csv`、可选 `route_overlap.csv`，并额外写 `summary.json` 和 `report.md`；[Qwen3 MoE Routing Readiness](results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/report.md) 已分析真实 `12` prompts / `48` routers，结论是 direct router average 高风险，必须 freeze、校准或 route-KD 后再进入 vLLM eval。
16. **MoE router 先过 readiness gate。** [MoE Routing Readiness](results/moe_routing_readiness/report.md) 会把 `router_summary.csv`、`route_overlap.csv`、`expert_load.csv` 转成 collapse、route drift、top-k 边界脆弱性和 expert load 风险；只有这些风险可控，才考虑开放 router 小 λ 或生成 expert-wise tensor rules。
17. **Toy MoE 已经复现 expert-index mismatch 和 router 漂移风险。** [Toy MoE Route-Aware Merge](results/toy_moe_merge/report.md) 中，直接 all-weight average 的 worst accuracy 是 `0.545`，expert-matched average 是 `0.750`，route-aware expert average 是 `0.750`；`unified_moe_average` 的 soft/hard top-2/max-overflow 是 `0.785/0.690/0.0775`；`unified_output_projection_moe_average` 是 `0.795/0.685/0.0700`；新的 `unified_confidence_blended_moe_average` 是 `0.790/0.690/0.07625`，说明 captured-fraction-gated expert 权重能在不损失 hard top-2 的情况下改善旧 unified。bias-only capacity 修正把 max top-k overflow 降到 `0.0475`，capacity-aware score 达到 `0.635`。
18. **同一个 readiness gate 已能分析多方法 MoE probe。** [Toy MoE Routing Readiness](results/toy_moe_routing_readiness/report.md) 把 toy MoE 的 base、endpoint、all-weight、expert-matched、route-aware 方法分开诊断；其中 `all_weight_average` 的 general slice 触发 `calibrate_router_before_average`，而 expert-matched/route-aware 的 route overlap 接近 `1.0`。
19. **MoE 方法选择已从“读表”变成自动决策。** [Toy MoE Method Selection](results/toy_moe_method_selection/report.md) 把 worst accuracy、routing readiness、hard top-2 dispatch 和 capacity overflow 合在一起：`all_weight_average` 被判为 `reject_routing_breakdown`；soft dispatch 下推荐 `expert_output_projection_router_calibrated_average`；sparse hard top-2 推荐 `unified_confidence_blended_route_kd_seed_average`；严格 capacity-aware sparse 推荐 `unified_moe_bias_capacity_average`，并把 confidence-blended unified、output-projection unified、bias-capacity unified 和 Router-KD 留在 hard top-2 / overflow Pareto frontier。
20. **Expert matching 已能进入 checkpoint materialization。** [Toy MoE Expert Remap Plan](results/toy_moe_expert_remap_plan/report.md) 把 expert-output matching 转成 `source_tensor_aliases.txt`：输出 checkpoint 的 expert index、tensor name 和 shape 不变，只改变某个 source 读取哪个 matched expert tensor；当前 toy 的 4 个 global alias rule 全部 ready，最小 output cosine 为 `0.943`。[MoE Layer-Wise Expert Remap Smoke](results/moe_layerwise_expert_remap_smoke/report.md) 进一步验证真实多层 MoE 可生成 layer-scoped alias，避免某一层的 expert 匹配被错误应用到所有层。
21. **Router-bias capacity correction 已落成 recipe。** [MoE Router Bias Plan](results/moe_router_bias_plan/report.md) 用 per prompt/category 的 worst top-k load 生成 writer-ready `router_bias_deltas.csv`；toy `unified_moe_average` 上 expert 0 的 worst top-k fraction 是 `0.3900`，高于 capacity `0.3125`，因此生成 `-0.0530` 的 bias delta，其余 experts 做中心化补偿。[MoE Confidence-Blended Router Bias Plan](results/moe_confidence_blended_router_bias_plan/report.md) 对 `unified_confidence_blended_moe_average` 生成同结构 bias delta：expert 0 的 worst top-k fraction 是 `0.38875`，delta 为 `-0.05230`；expert 3 load 偏低，delta 为 `+0.04469`。[MoE Confidence-Blended Combined Recipe](results/moe_confidence_blended_combined_recipe/report.md) 已把 confidence-blended expert 权重、expert alias remap 和 router-bias delta 合成一条 writer command。真实 Qwen checkpoint 若没有对应 bias tensor，writer 会在校验阶段报错，而不是改变模型结构。
22. **vLLM 下游评测 harness 已通过 mock 和真实 endpoint。** [vLLM Downstream Eval Contract Smoke](results/vllm_downstream_eval_smoke/smoke_report.md) 验证 HTTP contract；[Materialized Checkpoint vLLM Eval](results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/report.md) 是真实 vLLM-hosted checkpoint 评测；[Qwen Source-vs-Merge vLLM Comparison](results/vllm_source_merge_comparison/report.md) 把三个源模型 endpoint 和 uniform-average checkpoint 放到同一套下游任务里比较。
23. **Dense uniform average 的真实下游表现是负结果。** `qwen_0_5b_instruct_coder_uniform_average` 在每任务 `64` 样本上得到 GSM8K `0.000`、MMLU `0.219`、safety `0.500`、HumanEval compile `0.000`，avg primary `0.180`、worst primary `0.000`；同场 source endpoint 里 base/instruct/coder 的 avg primary 分别是 `0.375/0.227/0.199`，所以 uniform average 被三个源模型同时支配。这和前面 Qwen multi-expert plane 里 `0.5/0.5` 高 NLL ridge 一致。
24. **Probe-guided dense average 已跑真实 eval。** [Probe-Guided Dense Average Candidate](results/probe_guided_dense_average_candidate/report.md) 从 NLL grid 选出 `alpha=0.25,beta=1.0` bridge，写成同构 checkpoint，并用 vLLM 跑完同一套下游任务；它比 uniform avg primary 高 `0.023`，但仍比 best source 低 `0.172`，所以只能作为进入 layer/module-wise average 的证据。
25. **Module-wise guard 是负结果，norm-only/selective-norm 给出机制边界。** [Qwen Dense Module-Guarded Candidate](results/qwen_dense_module_guarded_candidate/report.md) 的 avg primary 是 `0.160`，比 global bridge 低 `0.043`；[Qwen Dense Norm-Guarded Candidate](results/qwen_dense_norm_guarded_candidate/report.md) 的 avg primary 与 global bridge 同为 `0.203`，但 safety 上升、GSM8K/MMLU 下降；[Qwen Dense Selective-Norm Candidate](results/qwen_dense_selective_norm_guarded_candidate/report.md) 只冻结 6 个极端 norm tensor 后 avg primary 是 `0.191`。这个结果说明 probe 的作用是定位机理和指导更窄的 intervention，而不是把高冲突模块一刀切冻结。
26. **Sparse-coordinate candidate 已跑真实 vLLM eval，给出机制边界。** [Qwen Dense Sparse-Method Candidate](results/qwen_dense_sparse_method_candidate/report.md) 对 attention+MLP 的 broad TIES 规则得到 avg primary `0.156`，低于 global bridge `0.047`；[Qwen Dense Attention Sparse-Method Candidate](results/qwen_dense_attention_sparse_method_candidate/report.md) 只动 attention 后 avg primary `0.203`，与 global bridge 打平、比 uniform 高 `0.023`。这说明 sparse coordinate merge 不能粗暴覆盖 MLP，下一步应学习 tensor/density gate，而不是扩大规则范围。
27. **真实 Qwen MoE 已从 gate 推进到五个本地 checkpoint、五份文件级 delta audit、逐 tensor trust-region validation 和两个机制 ablation。** 当前真正进展见 [Checkpoint Materialization Readiness](results/checkpoint_materialization_readiness/report.md)、[Materialized Checkpoint Delta Audit](results/qwen3_moe_materialized_delta_audit/report.md)、[Qwen3 MoE Audit-Gated Candidate](results/qwen3_moe_audit_gated_candidate/report.md)、[audit-gated delta audit](results/qwen3_moe_audit_gated_delta_audit/report.md)、[Qwen3 MoE Trust-Region Candidate](results/qwen3_moe_trust_region_candidate/report.md)、[trust-region delta audit](results/qwen3_moe_trust_region_delta_audit/report.md)、[trust-region delta validation](results/qwen3_moe_trust_region_delta_validation/report.md)、[expert-only attention ablation](results/qwen3_moe_expert_only_trust_region_candidate/report.md)、[expert-only delta audit](results/qwen3_moe_expert_only_delta_audit/report.md)、[tail-trimmed expert-only candidate](results/qwen3_moe_tail_trimmed_expert_only_candidate/report.md) 和 [tail-trimmed delta audit](results/qwen3_moe_tail_trimmed_delta_audit/report.md)：`qwen3_moe_unified_route_guarded_candidate`、`qwen3_moe_audit_gated_candidate`、`qwen3_moe_trust_region_candidate`、`qwen3_moe_expert_only_trust_region_candidate` 和 `qwen3_moe_tail_trimmed_expert_only_candidate` 都已物化成本地 `16` shard / `18,867` tensor 同构 checkpoint。trust-region candidate 把 routing/load/category/delta probe 联合成 expert-level cap，缩小 `405/5,243` 个 expert rules；tail-trimmed 又把 `140` 个剩余 high-tail expert groups 继续缩小到 `0.65` cap 附近。实际 audit 显示 tail-trimmed total relative delta norm 是 `0.243`，routed `>0.75` 为 `0`，attention/router 仍是 `0` changed。下一步是 vLLM 下游评测。
28. **Probe-gated unified average 已把“为什么”转成 action gate。** [Probe-Gated Unified Average Plan](results/probe_gated_unified_average_plan/report.md) 不是按场景静态排名方法，而是从已有 vLLM ablation 和 toy MoE 机制对比里提炼默认动作：Dense 侧只保留 `probe_guided_global_bridge_only`，因为 global bridge 比 uniform avg primary 高 `0.023`，而 aggressive module guard 低 `0.043`；MoE 侧默认组合是 expert identity alignment、confidence-blended expert weights、guarded router calibration 和 capacity gate，其中 expert identity matching 的 soft worst-acc gain 是 `0.205`，capacity bias 的 top-k overflow delta 是 `-0.029`。Qwen3 这对 MoE 的 topology、identity 和 routing probe 已经通过必要 gate，并已落成本地 materialized route-guarded candidate。
29. **Unified selector 的第一个真实 Qwen Dense 结果是“拒绝坏合并”。** [Unified Merge Family Probe](results/fp_merge_compare_dense/report.md) 在同一个候选族里比较 linear、task arithmetic、sign-elect 和 magnitude-weighted variants，held-out 选择 `lambda=0.0`；test worst NLL `5.183`，比 linear midpoint 好 `3.765`，比 TIES baseline 好 `3.927`，但仍比 best endpoint 差 `0.032`。结论是现在不能宣称 Dense average 已经支配 endpoint；正确动作是让 selector 有权退回 anchor/endpoint，再扩展到 layer/module-wise 或真实 vLLM selection。
30. **生成式 smoke 复现了 midpoint 坏掉，但也暴露了 NLL selector 的保守性。** [Generation Exact-Answer Merge Eval](results/fp_gen_eval_dense/report.md) 中，linear midpoint 在 2 个 math + 2 个 code-output exact-answer 题上 avg/worst 都是 `0.000`；unified `lambda=0` 是 `0.500/0.000`，避免了 midpoint 崩坏；coder 是 `0.500/0.500`，说明小样本生成式 gate 会更偏向 endpoint。下一步 selector 必须同时看 NLL 和 generation held-out，而不是只看一个 probe。
31. **MoE selector 现在明确区分 expert identity gate 和 router/load gate。** [MoE Probe-Gated Selector](results/moe_probe_gated_selector/report.md) 输出全局规则 `reject_same_name_average_without_alignment`；Qwen3 Instruct/Coder 通过 expert identity gate，可先用 identity mapping；新的 [Qwen3 MoE Unified Average Preflight](results/moe_unified_preflight_qwen3_30b/report.md) 进一步确认这对模型同构合同通过：`48` 个 router tensor 和 `18,432` 个 routed expert tensor layout 匹配。但真实 [Qwen3 MoE Routing Readiness](results/moe_routing_readiness/qwen3_30b_instruct_vs_coder/report.md) 显示 router 不能直接平均：`493/576` 个 router/prompt slice 要求先校准，top-k Jaccard mean/min 只有 `0.454/0.242`。新的 [Qwen3 MoE Route-Guarded Candidate](results/qwen3_moe_unified_route_guarded_candidate/report.md) 已把这个判断落成冻结 router + source-route-conditioned expert weights，并已写成可由 vLLM 加载的本地同构 checkpoint。下一步不再是写更多 plan，而是在 GPU host 上对两个 source endpoint 和这个 candidate 做同一套下游评测。
32. **Checkpoint materialization readiness 已推进到五个 Qwen3 candidates ready for eval。** [Checkpoint Materialization Readiness](results/checkpoint_materialization_readiness/report.md) 现在显示 `12` 个候选里 `6` 个已 materialize，其中 Qwen3 route-guarded、audit-gated、trust-region、expert-only 和 tail-trimmed candidates 都是 `ready_for_vllm_eval`，0.5B uniform baseline 已完成 vLLM eval；[vLLM Checkpoint Eval Plan](results/vllm_checkpoint_eval_plan/report.md) 显示 `11` 个 eval rows，其中 Qwen3 Instruct source、Qwen3 Coder source、五个已物化 Qwen3 MoE candidates 和 0.5B baseline 都是 `ready_to_host`，另外两个 MoE candidates 还缺 checkpoint，`1` 个 toy 不可加载。

核心对象是：

```text
theta(alpha, beta) = theta_0 + alpha * tau_A + beta * tau_B
tau_i = theta_i - theta_0
```

这里的 `alpha,beta` 是两个任务向量的合并系数。我们在这个平面上评估每个点的 task A / task B loss、accuracy、worst-task 指标，并把常见合并方法投影回同一个空间。

## 怎么读这些图

图里的符号含义如下：

- `alpha`：沿着 expert A task vector 走多远。`alpha=1,beta=0` 通常就是 expert A。
- `beta`：沿着 expert B task vector 走多远。`alpha=0,beta=1` 通常就是 expert B。
- 左侧 2D contour map：同一个 `alpha,beta` 网格的俯视图，颜色和等高线表示 `worst_loss` 或 `worst_nll`。颜色越低、等高线数值越低，表示两个任务中最差的那个也更好。
- 右侧 3D surface：同一份网格数据的 3D 表达。`x/y` 还是 `alpha/beta`，`z` 是 loss 或 NLL。它不是另一个平面，也不是额外的数据。
- 图上的实心点：base、两个 expert、linear average、best grid 等具体 checkpoint 或 merge 点，它们严格位于 raw task-vector plane。
- 图上的空心 `projected` 点：方法本身不在这个 raw plane 上，但可以把它投影回 `alpha,beta` 平面看相对位置。Digits 图里的 RegMean 就是这种情况。
- 2D 和 3D 是同一数据的两种读法：2D 更适合看点的位置和等 loss 线，3D 更适合直观看 basin、ridge、valley 的形状。

## 先回答：为什么这里说 2D，而不是早期论文那种 3D 图？

这其实不是矛盾。

早期 loss-landscape 文章里常见的 3D 图，本质上也是一个二维切片：

```text
x 轴 = 方向 1 的系数
y 轴 = 方向 2 的系数
z 轴 = loss
```

也就是说，3D surface 展示的是“二维参数平面上的标量函数”。本项目里的 merge landscape 也是同一类对象，只是二维平面的两个方向不是随机扰动方向，而是有语义的任务向量 `tau_A` 和 `tau_B`。README 里的展示图现在同时给出 2D contour 和 3D surface：前者负责精确读点和等高线，后者负责看几何形状。

所以更准确的说法是：

- 研究空间是二维任务向量平面 `alpha,beta`；
- 可视化形式可以是 2D heatmap/contour，也可以是 3D surface；
- 3D surface 的 z 轴通常是 `worst_loss` 或 `worst_nll`；
- 2D 图更适合精确比较和交互，3D 图更接近经典 loss-landscape 论文的视觉风格；
- 如果要复现 Li et al. 的 [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) 那类更“崎岖/非凸”的视觉效果，应当另外取随机或 filter-normalized directions；这会回答“某个模型附近 loss landscape 长什么样”，而不是“两个 task vector 的 merge 几何长什么样”。

下面这些展示图左侧是 2D contour map，右侧是同一数据的 3D surface。

![Digits 3D worst-loss surface](results/figures_3d/digits_worst_loss_surface.png)

![CIFAR-10 3D worst-loss surface](results/figures_3d/cifar10_worst_loss_surface.png)

![Pretrained ViT 3D worst-loss surface](results/figures_3d/pretrained_vit_worst_loss_surface.png)

![Qwen multi-expert 3D worst-NLL surface](results/figures_3d/qwen_multi_worst_nll_surface.png)

## 结论摘要

当前 coverage audit：`complete = 60`, `partial = 1`, `missing = 0`；唯一 partial 是 generic target-registry vLLM eval 还没有跑完，但 materialized checkpoint、source-vs-merge 对照、probe-guided dense candidate、dense guard ablation 的真实 vLLM eval、真实 Qwen3 MoE materialized checkpoint，以及 probe-gated unified average plan 都已完成。完整汇总见 `results/summary.md` 和 `results/summary.json`。

主要结论：

1. 合并是否成功，确实可以用“是否落在多个任务共同可接受的 basin 交集附近”来解释。
2. 小模型上，线性平均、task arithmetic、TIES/DARE/Fisher 等 on-plane 方法可以直接放到同一个任务向量平面中比较；RegMean、layer-wise task arithmetic 等 off-plane 方法需要标成投影点或单独报告。
3. 单类 expert surrogate 是一个负结果：十个 digit expert 的多数 pair 很容易 merge，global conflict 指标对 drop 的预测很弱，说明 interference 不能只看全局统计。
4. 独立随机初始化会制造表面上的 interpolation barrier；简单 hidden-unit alignment 后，loss barrier 从 `0.064` 降到 `0.006`。
5. CIFAR-10 和 CIFAR100/ViT 证明这个方法不是只适用于 toy MLP。
6. pretrained ViT-B/16 frozen-backbone transfer 提供了更接近大规模视觉模型的证据：linear average worst accuracy `0.763`，grid best `0.783`。
7. Qwen2.5-1.5B base-to-instruct 路径上，`lambda=0.75` 在多个 slice 上表现稳定，MMLU 小切片达到 `18/24 = 0.750`。
8. Qwen2.5-0.5B instruct+coder multi-expert merge 显示，简单平均会明显退化：linear average avg/worst NLL 为 `5.591 / 9.553`，而 instruct endpoint avg NLL 为 `3.009`。
9. Qwen2.5-0.5B instruct/coder 的 materialized `0.5/0.5` Dense uniform average 已用 vLLM 跑完真实下游评测，avg primary 只有 `0.180`，worst primary 为 `0.000`，因此它只能作为负 baseline。

## 研究覆盖

| 模块 | 状态 | 证据 |
| --- | --- | --- |
| 2D/3D task-vector merge landscape | 完成 | digits、CIFAR、pretrained ViT、Qwen multi-expert 均有网格和图 |
| per-task basin overlay | 完成 | digits basin overlay |
| lambda sweep | 完成 | digits、CIFAR、Qwen path |
| method overlay | 完成 | average、task arithmetic、SLERP、TIES、DARE、TIES+DARE、Fisher、RegMean、layer-wise task arithmetic、grid search |
| interference atlas | 完成 | digits、CIFAR、single-digit pairwise |
| one-class expert surrogate | 完成 | 10 个 single-digit expert，45 个 pair |
| randomness / alignment | 完成 | 独立初始化 MLP 对齐前后 barrier |
| natural image | 完成 | CIFAR-10 vehicle/animal |
| ViT / pretrained ViT | 完成 | CIFAR100 ViT-style 与 ImageNet-pretrained ViT-B/16 frozen-backbone transfer |
| Qwen LLM path | 完成 | Qwen2.5-1.5B base-to-instruct |
| Qwen benchmark slices | 完成 | GSM8K、MMLU、HumanEval NLL、BeaverTails safety/refusal |
| Qwen multi-expert | 完成 | Qwen2.5-0.5B base + instruct + coder |
| Qwen dense sparse-coordinate candidate | 完成 | broad attention+MLP 和 attention-only TIES sparse rule 均已 materialize 并完成 vLLM eval |
| Dense/MoE averaging literature matrix | 完成 | 22 个来源被整理成方法、probe、MoE 优化 gate 和 writer action，并补入 output-space / sparse capacity gate |
| Qwen target model registry | 完成 | 17 个 Qwen 官方/第三方/下游候选映射到场景、评测、probe 和同构拓扑 gate |
| MoE routing probe CLI | 完成 | 真实 MoE router hook probe 已实现，能写 router/expert/overlap CSV 与 summary/report |
| MoE routing probe smoke | 完成 | tiny MoE 本地验证捕获 2 个 router、6 行 route overlap |
| Toy MoE expert remap materialization | 完成 | expert-output matching 已转成 source tensor alias rules，保持输出 checkpoint 同构 |
| dashboard | 完成 | `results/dashboard/index.html` |

## 1. Controlled Digits：最干净的 merge-plane demo

设置：先训练一个共享 MLP base，再从同一个 base fine-tune 两个 expert：digits `0-4` 和 digits `5-9`。这个实验便宜、可控，适合密集评估 `41 x 41` 的 `alpha,beta` 网格。

关键数字：

| 指标 | 数值 |
| --- | ---: |
| linear average worst accuracy | `0.922` |
| layer-wise task arithmetic worst accuracy | `0.928` |
| RegMean linear-layer worst accuracy | `0.939` |
| grid max worst accuracy | `0.961` |
| global task-vector cosine | `0.138` |

3D worst-loss surface：

![Digits 3D worst-loss surface](results/figures_3d/digits_worst_loss_surface.png)

2D merge landscape：

![Digits merge landscape](results/digits_merge/figures/merge_landscape.png)

per-task basin overlay：

![Digits per-task basin overlay](results/digits_merge/figures/per_task_basin_overlay.png)

method overlay：

![Digits method overlay](results/digits_merge/figures/method_overlay.png)

lambda sweep：

![Digits lambda sweep](results/digits_merge/figures/lambda_sweep.png)

interference atlas：

![Digits interference heatmap](results/digits_merge/figures/interference_heatmap.png)

解释：digits 是这个项目最接近“hero figure”的部分。展示图里 base、两个 expert、linear average 和 best grid 是 raw task-vector plane 里的点；RegMean 用空心 `projected` 标记，因为它的 `plane_residual` 较大。好的 on-plane merge 点通常靠近两个任务都能接受的区域，而不是只靠近某一个 expert。RegMean 表现更好，但它主要是 off-plane 改进，不能简单解释成“在这个二维平面上移动到了更好位置”。

## 2. One-Class Expert Surrogate：一个 expert 只学一个类

设置：训练 10 个 single-digit expert，每个 expert 只强化一个 digit class，然后评估全部 45 个 pair 的 merge。

关键数字：

| 指标 | 数值 |
| --- | ---: |
| mean linear worst accuracy | `0.986` |
| 最差 pair | `3/9` |
| weighted conflict vs drop Spearman | `0.165` |

pairwise heatmaps：

![Single-digit pairwise heatmaps](results/digit_pairwise_experts/pairwise_heatmaps.png)

conflict vs merge drop：

![Conflict vs drop](results/digit_pairwise_experts/conflict_vs_drop.png)

layer conflict atlas：

![Layer conflict atlas](results/digit_pairwise_experts/layer_conflict_atlas.png)

解释：这个实验很重要，因为它不是只给正结果。多数 single-digit pair 可以被线性平均保留下来，global sign conflict / weighted conflict 对 drop 的预测并不强。这说明“冲突导致 merge 失败”不是一句简单的全局统计结论，layer-local、task-specific 的结构更重要。

## 3. Randomness / Alignment：随机初始化会制造假 barrier

设置：训练两个相同任务但不同随机初始化的小 MLP，比较 hidden-unit permutation alignment 前后的 interpolation path。

关键数字：

| 指标 | 对齐前 | 对齐后 |
| --- | ---: | ---: |
| midpoint accuracy | `0.944` | `0.971` |
| loss barrier | `0.064` | `0.006` |

![Alignment interpolation](results/alignment_barrier/interpolation_alignment.png)

解释：如果两个模型不是从同一个 base fine-tune 出来的，权重空间里的 barrier 可能只是神经元排列对不上造成的。alignment 后 barrier 大幅下降，支持 proposal 里关于 Git Re-Basin / permutation symmetry 的担忧。

## 4. CIFAR-10 Vehicle / Animal：自然图像上的失败案例

设置：训练一个小 GroupNorm CNN base，再 fine-tune vehicle expert 和 animal expert。

关键数字：

| 指标 | 数值 |
| --- | ---: |
| base worst accuracy | `0.376` |
| linear average worst accuracy | `0.249` |
| validation-grid best worst accuracy | `0.426` |

3D worst-loss surface：

![CIFAR-10 3D worst-loss surface](results/figures_3d/cifar10_worst_loss_surface.png)

2D merge landscape：

![CIFAR merge landscape](results/cifar_merge/figures/merge_landscape.png)

method overlay：

![CIFAR method overlay](results/cifar_merge/figures/method_overlay.png)

lambda sweep：

![CIFAR lambda sweep](results/cifar_merge/figures/lambda_sweep.png)

interference atlas：

![CIFAR interference heatmap](results/cifar_merge/figures/interference_heatmap.png)

解释：CIFAR-10 是更真实的自然图像分类案例，naive average 的 worst accuracy 低于 base，说明不是所有 same-base expert 都能被简单 soup。grid search 找到更好的 worst-task 区域，支持“合并系数应该由 basin overlap 指导”。

## 5. CIFAR100 ViT-Style：从 CNN 扩展到 transformer 视觉模型

设置：训练一个小 patch-transformer 做 CIFAR100 coarse-label 分类，再 fine-tune living/object 两个 expert。

关键数字：

| 指标 | 数值 |
| --- | ---: |
| base worst accuracy | `0.189` |
| linear average worst accuracy | `0.076` |
| best method worst accuracy | `0.197` |
| global task-vector cosine | `-0.176` |

merge landscape：

![CIFAR100 ViT merge landscape](results/cifar100_vit_merge/figures/merge_landscape.png)

method overlay：

![CIFAR100 ViT method overlay](results/cifar100_vit_merge/figures/method_overlay.png)

lambda sweep：

![CIFAR100 ViT lambda sweep](results/cifar100_vit_merge/figures/lambda_sweep.png)

task-vector PCA：

![CIFAR100 ViT PCA](results/cifar100_vit_merge/figures/pca_task_vectors.png)

interference atlas：

![CIFAR100 ViT interference](results/cifar100_vit_merge/figures/interference_heatmap.png)

解释：from-scratch ViT-style 模型的绝对性能不高，但它验证了同一套 task-vector plane、method overlay、lambda sweep、PCA geometry 和 interference atlas 能迁移到 transformer 视觉结构。

## 6. Pretrained ViT Transfer：更接近 model soups / CLIP-ViT 语境

设置：使用 ImageNet-pretrained ViT-B/16 frozen backbone，在 CIFAR100 coarse labels 上训练 base/living/object linear heads，并在 head task-vector plane 上做 merge。

关键数字：

| 指标 | 数值 |
| --- | ---: |
| base worst accuracy | `0.740` |
| linear average worst accuracy | `0.763` |
| validation-grid best worst accuracy | `0.783` |
| global head cosine | `-0.068` |

3D worst-loss surface：

![Pretrained ViT 3D worst-loss surface](results/figures_3d/pretrained_vit_worst_loss_surface.png)

2D merge landscape：

![Pretrained ViT merge landscape](results/pretrained_vit_transfer_merge/figures/merge_landscape.png)

method overlay：

![Pretrained ViT method overlay](results/pretrained_vit_transfer_merge/figures/method_overlay.png)

lambda sweep：

![Pretrained ViT lambda sweep](results/pretrained_vit_transfer_merge/figures/lambda_sweep.png)

head conflict：

![Pretrained ViT head conflict](results/pretrained_vit_transfer_merge/figures/interference_heatmap.png)

解释：这个实验不是 full-backbone fine-tuning，但它补上了 pretrained vision transfer 的证据。frozen backbone 让绝对准确率明显高于 from-scratch ViT，同时 head task-vector 仍然展示出 specialization、forgetting 和 basin intersection。

## 7. Qwen2.5-1.5B Base-to-Instruct Path：LLM 不做密集 2D，先做路径诊断

LLM 上密集网格成本高，所以先评估：

```text
theta(lambda) = base + lambda * (instruct - base)
```

关键数字：

| 指标 | 数值 |
| --- | ---: |
| best average-NLL lambda | `0.75` |
| instruction NLL base -> best | `3.612 -> 1.811` |
| GSM8K best strict | `1/12` at `lambda=0.75` 或 `1.0` |
| GSM8K best loose | `3/12` at `lambda=0.75` |
| MMLU best accuracy | `18/24 = 0.750` at `lambda=0.75` |
| HumanEval best solution NLL | `0.964` at `lambda=1.0` |
| safety/refusal best avg NLL | `2.546` at `lambda=0.75` |

base-to-instruct NLL path：

![Qwen path sweep](results/qwen_path_sweep/qwen_path_sweep.png)

largest base-to-instruct deltas：

![Qwen delta norms](results/qwen_path_sweep/delta_norms.png)

GSM8K slice：

![Qwen GSM8K](results/qwen_gsm8k_slice/gsm8k_exact_match.png)

MMLU slice：

![Qwen MMLU](results/qwen_mmlu_slice/mmlu_accuracy.png)

HumanEval NLL slice：

![Qwen HumanEval](results/qwen_humaneval_nll_slice/humaneval_nll.png)

BeaverTails safety/refusal NLL slice：

![Qwen safety refusal](results/qwen_safety_refusal_slice/safety_refusal_nll.png)

解释：在 Qwen2.5-1.5B base-to-instruct 路径上，`lambda=0.75` 经常是折中点。它不是完整 leaderboard 评估，但作为 path diagnostic 足够说明：LLM 的 task-vector 系数也会存在可调的中间区域。

## 8. Qwen2.5-0.5B Multi-Expert Merge：base + instruct + coder

设置：用 Qwen2.5-0.5B base、Qwen2.5-0.5B-Instruct 和 Qwen2.5-Coder-0.5B-Instruct，评估：

```text
theta(alpha, beta) = base + alpha * instruct_delta + beta * coder_delta
```

关键数字：

| 指标 | 数值 |
| --- | ---: |
| best average-NLL method | instruct_expert `3.009` |
| linear-average avg / worst NLL | `5.591 / 9.553` |
| instruct/coder cosine | `0.140` |
| instruct/coder weighted conflict | `0.386` |

3D worst-NLL surface：

![Qwen multi-expert 3D surface](results/figures_3d/qwen_multi_worst_nll_surface.png)

2D merge grid：

![Qwen multi-expert grid](results/qwen_multi_expert_merge/figures/merge_grid.png)

diagonal path：

![Qwen multi-expert diagonal path](results/qwen_multi_expert_merge/figures/diagonal_path.png)

pairwise conflict：

![Qwen multi-expert conflict](results/qwen_multi_expert_merge/figures/pairwise_conflict.png)

解释：这个实验是最接近 proposal 里“官方 base/instruct/expert model merging”的部分。结果也更接近真实 LLM merge 的风险：instruct 和 coder expert 的简单平均并不会自动得到兼顾 instruction/code/general 的模型，反而会在 worst NLL 上恶化。

## 9. 交互式 Dashboard

已生成静态 dashboard：

```text
results/dashboard/index.html
```

它包含：

- overview metrics；
- 可拖动的 precomputed merge-plane explorer；
- task pair / method / objective 切换；
- raw vs normalized plane 切换；
- alpha / beta / lambda 控件；
- pairwise expert、alignment、CIFAR、ViT、Qwen path、Qwen benchmark slices、Qwen multi-expert tabs。

在本地打开：

```bash
python -m http.server 8000
```

然后访问：

```text
http://localhost:8000/results/dashboard/index.html
```

## 10. 如何复现

最小 digits 实验：

```bash
PYTHONPATH=src python scripts/run_digits_merge.py --output-dir results/digits_merge
```

生成 dashboard：

```bash
PYTHONPATH=src python scripts/build_dashboard.py --output-dir results/dashboard
```

生成 3D surface 图：

```bash
python scripts/build_3d_figures.py --output-dir results/figures_3d
```

重新收集汇总和 artifact manifest：

```bash
PYTHONPATH=src python scripts/collect_results.py
```

更多实验脚本：

```text
scripts/run_alignment_barrier.py
scripts/run_digit_pairwise_experts.py
scripts/run_cifar_merge.py
scripts/run_cifar100_vit_merge.py
scripts/run_pretrained_vit_transfer_merge.py
scripts/run_qwen_path_sweep.py
scripts/run_qwen_gsm8k_slice.py
scripts/run_qwen_mmlu_slice.py
scripts/run_qwen_humaneval_nll_slice.py
scripts/run_qwen_safety_refusal_slice.py
scripts/run_qwen_multi_expert_merge.py
```

## 11. 文件索引

| 文件 | 用途 |
| --- | --- |
| `proposal.md` | 原始研究计划 |
| `RESEARCH_REPORT.md` | 更详细的 artifact map 和结果解释 |
| `PAPER.md` | paper-style writeup |
| `results/summary.md` | 自动生成的覆盖审计和关键指标 |
| `ARTIFACT_MANIFEST.json` | 产物清单和 hash |
| `src/mergeviz/weights.py` | state-dict vectorization、task-vector、plane projection |
| `src/mergeviz/merge_methods.py` | average、task arithmetic、SLERP、TIES、DARE、Fisher 等方法 |
| `scripts/build_dashboard.py` | 生成静态 dashboard |
| `scripts/build_3d_figures.py` | 生成 README 中的 3D surface 图 |

## 12. 局限性

这些结果已经覆盖 proposal 的研究 artifact 目标，但还不是 leaderboard-scale 论文评测：

- pretrained ViT 实验冻结 backbone，只 merge linear heads；
- Qwen benchmark slices 很小，不能替代完整 MMLU/GSM8K/HumanEval/安全评测；
- dashboard 查询的是预计算网格，不在浏览器里实时跑 checkpoint evaluation；
- RegMean 目前主要作为 digits MLP 的 linear-layer diagnostic；
- 3D surface 是可视化形式，不解决高维 loss landscape 只看二维切片的根本限制。

当前最稳健的结论是：模型合并可以被理解为在任务向量子空间中寻找多个任务 basin 的低损失交集；失败通常表现为某些任务方向之间没有足够好的共同区域，或者被 layer-local interference / alignment 问题放大。
