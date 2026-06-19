# Visualizing Model Merging：任务向量空间中的模型合并可视化

这份仓库把 `proposal.md` 里的想法实现成了一个可运行的研究 artifact：从小型图像分类模型开始，逐步扩展到 ViT/pretrained ViT 和 Qwen 系列 LLM，观察模型合并点在任务向量空间中的位置、多个任务 basin 是否重叠，以及合并失败是否和 task-vector interference 有关。

后续 Qwen Dense/MoE 和下游微调模型的实验设计见：[Qwen Dense/MoE 下游微调模型合并实验方案](QWEN_DENSE_MOE_EXPERIMENT_PLAN.md)，结构化目标模型清单见：[Qwen Target Model Registry](results/qwen_target_model_registry/report.md)。Averaging 失败诊断、probe 清单和 MoE route-aware averaging 路线见：[Dense/MoE Model Averaging 的指标、Probe 与优化路线](MODEL_AVERAGING_PROBES_AND_MOE_OPTIMIZATION.md)。当前已有实验的同构 Average 决策汇总见：[Average Decision Report](results/average_decision_report/report.md)，Dense/MoE 文献和 probe 矩阵见：[Model Averaging Literature Review](results/model_averaging_literature_review/report.md)，candidate materialization 选择见：[Average Candidate Recipes](results/average_candidate_recipes/report.md)，MoE 拓扑检查见：[Checkpoint Topology Inspect](results/checkpoint_topology_inspect/report.md)，MoE 参数组计划见：[MoE Same-Shape Average Plan](results/moe_average_plan/report.md)，MoE routing 风险诊断见：[MoE Routing Readiness](results/moe_routing_readiness/report.md)，MoE route-weight tensor rules 见：[MoE Route-Weight Recipes](results/moe_route_weight_recipes/report.md)，MoE router-bias capacity recipe 见：[MoE Router Bias Plan](results/moe_router_bias_plan/report.md)，checkpoint materialization readiness 见：[Checkpoint Materialization Readiness](results/checkpoint_materialization_readiness/report.md)，toy MoE 验证见：[Toy MoE Route-Aware Merge](results/toy_moe_merge/report.md)，toy expert 重排写出计划见：[Toy MoE Expert Remap Plan](results/toy_moe_expert_remap_plan/report.md)，checkpoint 写出器 smoke 见：[Same-Shape Checkpoint Writer Smoke](results/same_shape_writer_smoke/report.md)。

这里说的 Average 不是 ensemble，也不是把 MoE experts 扩成更多分支；最终目标模型必须和输入模型保持同构，能用同一个 config/tokenizer/model class 直接加载。Probe 的作用是决定哪些模型、层、模块或 experts 可以被平均，以及平均系数应该怎么设。

最新 MoE 机制结果：toy MoE 上，失败主因不是“平均”这个动作本身，而是 expert 语义 index、router dispatch 和 expert load capacity 一起漂移。当前统一方法 `unified_moe_average` 先用 per-expert source weight search 处理 expert 语义/重要性，再对 router seed 和 capacity loss 系数做 held-out selection sweep；候选 seed 包括 base router、soft-calibrated router、Router-KD router 和 route-KD router。最终选择 `router_kd_seed`、capacity loss `0.0`，soft worst accuracy 为 `0.785`，hard top-2 worst accuracy 为 `0.690`，高于 route-KD 的 `0.685`；max top-k overflow 为 `0.0775`，也略低于 route-KD 的 `0.07875`。进一步的 `unified_moe_bias_capacity_average` 只训练 router bias，不重学完整 router 几何，把 hard top-2 worst accuracy 保持在 `0.6825`，同时把 max top-k overflow 降到 `0.0475`；`accuracy - overflow` 得到 `0.635`，高于 Router-KD 的 `0.62625`。这说明更有效的 MoE merge 不是单纯加大 capacity penalty，而是先修 expert 语义，再校准 router 几何，最后用 bias-only correction 做全局负载修正。这个机制现在已经能写成 checkpoint recipe：`scripts/build_moe_router_bias_plan.py` 从 `expert_load.csv` 生成 `tensor,index,delta`，`scripts/write_same_shape_average_checkpoint.py --tensor-add-csv` 会在保持同构的前提下把 bias delta 写入已有 tensor。

这个结果也解释了为什么不能只机械套一个低-overflow router：`unified_router_kd_seed_average` 把 Router-KD router 直接接到 expert-search 权重上后，hard top-2 code accuracy 只有 `0.5975`，说明 router prior 和 expert 权重必须共同校准。output-space projection probe 也更新为正向结果：它按 base router route probability 给 expert 输出残差加权，mean captured fraction 是 `0.616`；`expert_output_projection_router_calibrated_average` soft worst accuracy 达到 `0.8075`，是当前 soft-router 最优，但 hard top-2 worst accuracy 只有 `0.6475`，因此还不能替代 sparse dispatch / capacity-aware 目标。

最新真实 vLLM source-vs-merge 结果：本地已把 Qwen2.5-0.5B base、Qwen2.5-0.5B-Instruct、Qwen2.5-Coder-0.5B-Instruct 和 materialized `qwen_0_5b_instruct_coder_uniform_average` 都用 vLLM host，并在 GSM8K、MMLU、safety、HumanEval compile 各 `64` 个样本上跑完同一套下游评测。结果见 [Qwen Source-vs-Merge vLLM Comparison](results/vllm_source_merge_comparison/report.md)：base avg primary `0.375`、instruct `0.227`、coder `0.199`、uniform average `0.180`，uniform average 在 4 个模型里排第 4，低于最佳源模型 `0.195`。逐任务看，它比最佳源模型分别低 GSM8K `0.094`、MMLU `0.125`、safety `0.047`、HumanEval compile `0.609`；safety 的 `0.500` 也不是好现象，因为 safe non-refusal 是 `1.000`，unsafe refusal 是 `0.000`，相当于几乎不拒绝 unsafe prompts。这说明 0.5/0.5 Dense uniform average 不只是 probe 上可疑，真实 endpoint 对照下也被三个源模型同时支配；下一步应做 probe-guided same-shape average，而不是继续盲目平均。

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
11. **同构 checkpoint 写出路径已经打通。** `scripts/write_same_shape_average_checkpoint.py` 会先验证 tensor name/shape，再按 `base + sum_i w_i * (source_i - base)` 写 safetensors；现在也支持 `--tensor-add-csv`，可把离线 probe 得到的 router-bias scalar delta 写入已有 bias tensor。Qwen2.5-0.5B base/instruct/coder dry-run 已检查 `290` 个 tensor 无缺失、无 shape mismatch；本地还已写出 `qwen_0_5b_instruct_coder_uniform_average` 这个 0.5/0.5 Dense 负 baseline checkpoint，用于验证真实 host/eval 管线。MoE writer smoke 还验证了 freeze-router 下的 bias additive correction。
12. **MoE 大模型先做 header/config probe。** [Checkpoint Topology Inspect](results/checkpoint_topology_inspect/report.md) 对本地 Qwen3.5-35B-A3B config 做了不加载权重的拓扑检查：`256` experts、每 token 激活 `8` 个，active fraction `0.03125`；这直接支持 router/expert 分组 average。
13. **不是每个 best grid 都值得写 checkpoint。** [Average Candidate Recipes](results/average_candidate_recipes/report.md) 明确把当前 Qwen instruct/coder best grid 标成 `skip_endpoint_only`，因为 `alpha=1,beta=0` 只是端点，不是有价值的 average；`0.5/0.5` uniform average 也被 probe 标成负 baseline。
14. **MoE route-aware average 已落到 tensor-rule 文件。** [MoE Route-Weight Recipes](results/moe_route_weight_recipes/report.md) 会把 routing probe 的 `expert_load.csv` 转成 `tensor_rules.txt`，由 checkpoint writer 直接读取；当前状态是 `waiting_for_routing_probe`，说明还缺真实 Qwen3 MoE route probe，不应该假装已经有 expert-wise 权重。
15. **真实 MoE routing probe CLI 已经补上。** `scripts/probe_moe_routing.py` 会捕获 MoE router hook，输出 `router_summary.csv`、`expert_load.csv`、可选 `route_overlap.csv`，并额外写 `summary.json` 和 `report.md`；[MoE Routing Probe Smoke](results/moe_routing_probe_smoke/report.md) 已验证 tiny MoE 上能抓到 2 个 router 和 6 行 overlap。下一步是把它跑在 Qwen3-30B-A3B / Qwen3-Coder-30B-A3B 上。
16. **MoE router 先过 readiness gate。** [MoE Routing Readiness](results/moe_routing_readiness/report.md) 会把 `router_summary.csv`、`route_overlap.csv`、`expert_load.csv` 转成 collapse、route drift、top-k 边界脆弱性和 expert load 风险；只有这些风险可控，才考虑开放 router 小 λ 或生成 expert-wise tensor rules。
17. **Toy MoE 已经复现 expert-index mismatch 和 router 漂移风险。** [Toy MoE Route-Aware Merge](results/toy_moe_merge/report.md) 中，直接 all-weight average 的 worst accuracy 是 `0.545`，expert-matched average 是 `0.750`，route-aware expert average 是 `0.750`；进一步的 `unified_moe_average` 在 expert 权重搜索后做 router seed/capacity sweep，soft worst accuracy 达到 `0.785`，hard top-2 worst accuracy 达到 `0.690`；bias-only capacity 修正把 max top-k overflow 从 `0.0775` 降到 `0.0475`，capacity-aware score 达到 `0.635`。
18. **同一个 readiness gate 已能分析多方法 MoE probe。** [Toy MoE Routing Readiness](results/toy_moe_routing_readiness/report.md) 把 toy MoE 的 base、endpoint、all-weight、expert-matched、route-aware 方法分开诊断；其中 `all_weight_average` 的 general slice 触发 `calibrate_router_before_average`，而 expert-matched/route-aware 的 route overlap 接近 `1.0`。
19. **MoE 方法选择已从“读表”变成自动决策。** [Toy MoE Method Selection](results/toy_moe_method_selection/report.md) 把 worst accuracy、routing readiness、hard top-2 dispatch 和 capacity overflow 合在一起：`all_weight_average` 被判为 `reject_routing_breakdown`；soft dispatch 下推荐 `expert_output_projection_router_calibrated_average`；sparse hard top-2 推荐 `unified_moe_average`；严格 capacity-aware sparse 推荐 `unified_moe_bias_capacity_average`，并把 unified、bias-capacity unified、Router-KD 留在 hard top-2 / overflow Pareto frontier。
20. **Expert matching 已能进入 checkpoint materialization。** [Toy MoE Expert Remap Plan](results/toy_moe_expert_remap_plan/report.md) 把 expert-output matching 转成 `source_tensor_aliases.txt`：输出 checkpoint 的 expert index、tensor name 和 shape 不变，只改变某个 source 读取哪个 matched expert tensor；当前 4 个 alias rule 全部 ready，最小 output cosine 为 `0.943`。
21. **Router-bias capacity correction 已落成 recipe。** [MoE Router Bias Plan](results/moe_router_bias_plan/report.md) 用 per prompt/category 的 worst top-k load 生成 writer-ready `router_bias_deltas.csv`；toy `unified_moe_average` 上 expert 0 的 worst top-k fraction 是 `0.3900`，高于 capacity `0.3125`，因此生成 `-0.0530` 的 bias delta，其余 experts 做中心化补偿。真实 Qwen checkpoint 若没有对应 bias tensor，writer 会在校验阶段报错，而不是改变模型结构。
22. **vLLM 下游评测 harness 已通过 mock 和真实 endpoint。** [vLLM Downstream Eval Contract Smoke](results/vllm_downstream_eval_smoke/smoke_report.md) 验证 HTTP contract；[Materialized Checkpoint vLLM Eval](results/vllm_checkpoint_eval/qwen_0_5b_instruct_coder_uniform_average/report.md) 是真实 vLLM-hosted checkpoint 评测；[Qwen Source-vs-Merge vLLM Comparison](results/vllm_source_merge_comparison/report.md) 把三个源模型 endpoint 和 uniform-average checkpoint 放到同一套下游任务里比较。
23. **Dense uniform average 的真实下游表现是负结果。** `qwen_0_5b_instruct_coder_uniform_average` 在每任务 `64` 样本上得到 GSM8K `0.000`、MMLU `0.219`、safety `0.500`、HumanEval compile `0.000`，avg primary `0.180`、worst primary `0.000`；同场 source endpoint 里 base/instruct/coder 的 avg primary 分别是 `0.375/0.227/0.199`，所以 uniform average 被三个源模型同时支配。这和前面 Qwen multi-expert plane 里 `0.5/0.5` 高 NLL ridge 一致。
24. **Checkpoint materialization readiness 已推进到 hosted eval complete。** [Checkpoint Materialization Readiness](results/checkpoint_materialization_readiness/report.md) 现在显示 6 个候选里 `1` 个已 materialize 且 `1` 个完成 vLLM eval，`3` 个仍被 placeholder source path 卡住；[vLLM Checkpoint Eval Plan](results/vllm_checkpoint_eval_plan/report.md) 显示 4 个 checkpoint 候选里 `1` 个 eval complete、`2` 个缺 materialized checkpoint、`1` 个 toy 不可加载。

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

当前 coverage audit：`complete = 37`, `partial = 1`, `missing = 0`；唯一 partial 是 generic target-registry vLLM eval 还没有跑完，但 materialized checkpoint 的真实 vLLM eval 和 source-vs-merge 对照都已完成。完整汇总见 `results/summary.md` 和 `results/summary.json`。

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
