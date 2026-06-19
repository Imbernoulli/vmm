# Visualizing Model Merging：任务向量空间中的模型合并可视化

这份仓库把 `proposal.md` 里的想法实现成了一个可运行的研究 artifact：从小型图像分类模型开始，逐步扩展到 ViT/pretrained ViT 和 Qwen 系列 LLM，观察模型合并点在任务向量空间中的位置、多个任务 basin 是否重叠，以及合并失败是否和 task-vector interference 有关。

后续 Qwen Dense/MoE 和下游微调模型的实验设计见：[Qwen Dense/MoE 下游微调模型合并实验方案](QWEN_DENSE_MOE_EXPERIMENT_PLAN.md)。Averaging 失败诊断、probe 清单和 MoE route-aware averaging 路线见：[Dense/MoE Model Averaging 的指标、Probe 与优化路线](MODEL_AVERAGING_PROBES_AND_MOE_OPTIMIZATION.md)。当前已有实验的同构 Average 决策汇总见：[Average Decision Report](results/average_decision_report/report.md)，MoE 参数组计划见：[MoE Same-Shape Average Plan](results/moe_average_plan/report.md)。

这里说的 Average 不是 ensemble，也不是把 MoE experts 扩成更多分支；最终目标模型必须和输入模型保持同构，能用同一个 config/tokenizer/model class 直接加载。Probe 的作用是决定哪些模型、层、模块或 experts 可以被平均，以及平均系数应该怎么设。

## 一屏版结论

如果只想先看结果，可以先读这几条：

1. **Digits 是最干净的正例。** 低 worst-loss 区域沿着两个任务都能接受的 valley 展开，base、linear average、best grid 都在这个 valley 附近；两个 expert endpoint 各自只擅长一个任务，所以在 worst-loss 图上反而是高处。
2. **CIFAR-10 是 naive average 失败例。** linear average 没有落到最好的共同区域，validation grid best 明显更靠近低 worst-loss 区域；这说明 merge coefficient 不能总用 `0.5,0.5`。
3. **Pretrained ViT transfer 是“model soup 直觉”更成立的例子。** shared low-loss basin 更宽，linear average 已经不错，grid best 还能略好一点。
4. **Qwen instruct/coder multi-expert 是最值得警惕的例子。** `alpha=0.5,beta=0.5` 的 linear average 落在高 worst-NLL ridge 上；instruct endpoint / best 比平均好得多，说明 LLM expert merge 不能只做朴素平均。
5. **不是所有有效方法都真的在这个二维 plane 上。** base、expert、linear average、task arithmetic、grid point 是 raw task-vector plane 里的点；RegMean、layer-wise task arithmetic 这类方法可能离开这个 plane。展示图里这类点用 `projected` 标记，表示它们只是投影到 `alpha,beta` 平面上看位置。
6. **这些图看起来有些凸，不是理论假设。** 这只是当前 same-base task-vector slice 和 worst-loss/NLL 指标在选定范围内的形状。Li et al. 那类 loss-landscape 图用的是随机或 filter-normalized 方向，目标是展示局部训练地形；这里的方向是任务语义方向，回答的是 merge 几何，所以图形不必长得一样。
7. **Average 决策现在由 probe 输出驱动。** [Average Decision Report](results/average_decision_report/report.md) 把 merge grid、conflict probe 和可选 MoE routing probe 汇总成同构 checkpoint 的权重建议；当前 Qwen instruct/coder 被标成 `avoid_uniform_average`，建议先做 connectivity/barrier 筛选再重学平均权重。
8. **MoE 不能把 router/expert 当普通 dense 层平均。** [MoE Same-Shape Average Plan](results/moe_average_plan/report.md) 把 router、shared modules、expert FFN、embedding/lm_head 和 LoRA/adapters 分开：默认先冻结/校准 router，experts 先按 route frequency/output similarity 匹配，再写回同 expert 数的 checkpoint。

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

当前 coverage audit 已完成：`complete = 14`, `partial = 0`, `missing = 0`。完整汇总见 `results/summary.md` 和 `results/summary.json`。

主要结论：

1. 合并是否成功，确实可以用“是否落在多个任务共同可接受的 basin 交集附近”来解释。
2. 小模型上，线性平均、task arithmetic、TIES/DARE/Fisher 等 on-plane 方法可以直接放到同一个任务向量平面中比较；RegMean、layer-wise task arithmetic 等 off-plane 方法需要标成投影点或单独报告。
3. 单类 expert surrogate 是一个负结果：十个 digit expert 的多数 pair 很容易 merge，global conflict 指标对 drop 的预测很弱，说明 interference 不能只看全局统计。
4. 独立随机初始化会制造表面上的 interpolation barrier；简单 hidden-unit alignment 后，loss barrier 从 `0.064` 降到 `0.006`。
5. CIFAR-10 和 CIFAR100/ViT 证明这个方法不是只适用于 toy MLP。
6. pretrained ViT-B/16 frozen-backbone transfer 提供了更接近大规模视觉模型的证据：linear average worst accuracy `0.763`，grid best `0.783`。
7. Qwen2.5-1.5B base-to-instruct 路径上，`lambda=0.75` 在多个 slice 上表现稳定，MMLU 小切片达到 `18/24 = 0.750`。
8. Qwen2.5-0.5B instruct+coder multi-expert merge 显示，简单平均会明显退化：linear average avg/worst NLL 为 `5.591 / 9.553`，而 instruct endpoint avg NLL 为 `3.009`。

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
