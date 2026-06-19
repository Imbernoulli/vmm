# Dense/MoE Model Averaging 的指标、Probe 与优化路线

更新时间：2026-06-18

这份文档补充 [Qwen Dense/MoE 下游微调模型合并实验方案](QWEN_DENSE_MOE_EXPERIMENT_PLAN.md)：前一份文档回答“选哪些模型、评什么、怎么评”，这里回答“为了最终做好 Average，应该探测哪些深层信号、这些信号背后的理论解释是什么、以及如何把 probe 结果转化成更好的平均系数或平均前处理”。更结构化的论文到工程规则映射见 [Model Averaging Literature and Probe Matrix](results/model_averaging_literature_review/report.md)。

这里的目标不是机械套一遍 TIES/DARE/RegMean/MoE routing，也不是把多个模型变成运行时 ensemble。这里的 **Average** 采用更严格定义：最终输出必须仍然是一个和输入模型同构的 checkpoint，可被同一个 model class/config/tokenizer 直接加载；dense 模型不能变成 ensemble，MoE 模型不能改变 layer 数、hidden size、router 形状或 expert 数量。如果输入是同构 LoRA/adapters，最终输出也应该是一个同构 adapter，除非明确把 mixture 当作诊断上界。

目标仍然是产生一个 averaged checkpoint：

```text
theta_avg = sum_i w_i * theta_i
```

或者在 same-anchor task-vector 形式下：

```text
theta_avg = theta_anchor + sum_i w_i * (theta_i - theta_anchor)
```

区别在于，`w_i` 不应该盲目取 `1/n`。我们要用 probe 去估计哪些模型、哪些层、哪些 experts 真的处在可平均的连通区域里，然后让 Average 变成一个被证据约束的优化问题。所有权重最后都要写回原始参数空间，而不是在推理时同时调用多个源模型。

## 1. 先给结论

Dense 模型合并可以先从 averaging / task arithmetic 做起，但不能把 `0.5,0.5` 当默认正确答案。更稳的流程是：

```text
端点评测 -> lambda/grid sweep -> delta/probe -> 方法选择 -> 验证集选超参 -> 测试集报告
```

MoE 模型也仍然可以做 Average，但不能只做全参数同权平均。MoE 的关键状态不是一个 dense delta，而是：

```text
router 如何分配 token
每个 expert 学了什么
下游 fine-tune 改了 router、shared modules、还是某些 experts
```

因此 MoE 合并应优先做 route-aware averaging：

- router 先 frozen 或单独平均，再决定是否用小校准集估计 router 平均权重；
- shared attention / norm / embedding 和 expert FFN 分开处理；
- experts 先按 routing/activation/output 相似度匹配，再平均；
- 冲突大的 expert 不要强行同权平均，要降低其权重或只在匹配子空间平均；
- 下游 LoRA/adapters 也可以先做 adapter-level average；mixture 结构只作为对照或上界，最终要压回同构 adapter 或同构 full checkpoint。

## 2. Average 可以写成一个 probe-guided 优化问题

最朴素的平均是：

```text
theta(w) = theta_anchor + sum_i w_i * tau_i
tau_i = theta_i - theta_anchor
```

其中 `w_i = 1/n` 只是一个无信息先验。真正应该求的是：

```text
min_w  max_t  L_t(theta(w))
subject to w_i >= 0, sum_i w_i = 1
```

如果允许 layer-wise 或 module-wise average，则是：

```text
theta_g(w_g) = theta_anchor,g + sum_i w_g,i * tau_i,g
```

这里 `g` 可以是 layer、attention、MLP、router、某个 MoE expert。它仍然是 Average，只是权重从全局标量变成了结构化平均系数；输出结构不变，只是不同参数组使用不同的平均系数。

### 2.1 二阶近似：probe 如何进入系数优化

对某个任务 `t`，在 anchor 或当前平均点附近做 Taylor 近似：

```text
L_t(theta_anchor + sum_i w_i tau_i)
≈ L_t(theta_anchor)
  + sum_i w_i * <g_t, tau_i>
  + 1/2 * sum_i sum_j w_i w_j * tau_i^T H_t tau_j
```

这些项都可以用 probe 近似：

- `<g_t, tau_i>`：用少量 validation NLL 的 finite difference 估计。
- `tau_i^T H_t tau_j`：用 Hessian-vector product、diagonal Fisher，或者更便宜的 pairwise interpolation barrier 估计。
- `tau_i` 的冲突项：用 cosine、sign conflict、weighted conflict 估计。
- 任务约束：用 `worst_score`、held-in retention、general retention 写成约束。

因此更好的 Average 不是“试一个现成方法”，而是：

```text
min_w  sum_t rho_t * surrogate_loss_t(w)
       + lambda_barrier * w^T B w
       + lambda_conflict * conflict(w)
```

其中 `B_ij` 可以来自 `theta_i <-> theta_j` 的 connectivity barrier，`conflict(w)` 来自符号冲突或专家冲突。这个目标最后输出的仍然是一组平均权重 `w`。

### 2.2 Model connectivity：Average 成功的前提

Model Connectivity / Linear Mode Connectivity 给 Average 一个很直接的判断标准：

```text
B(i, j) = max_lambda L((1-lambda) theta_i + lambda theta_j)
          - max(L(theta_i), L(theta_j))
```

如果 `B(i,j)` 很低，两个模型之间的直线路径大概率在同一个 basin 或近似连通区域里，平均点更可能有效。反过来，如果 midpoint barrier 很高，`0.5,0.5` 平均通常就是在穿越高 loss 区域。

对多个下游 Qwen 分支，可以构造一个 connectivity graph：

```text
node = checkpoint
edge(i,j) = barrier(i,j) < threshold
```

然后：

- 同一连通分量内可以做普通或加权 average；
- 分量之间先做 alignment / bridge / layer-wise average；
- 对 barrier 高的模型，不应该直接加入 soup；
- 如果 barrier 只集中在少数层，则做 layer-wise average，而不是放弃全模型平均。

这也是 probe 的核心作用：不是“看图解释失败”，而是先判断哪些模型、哪些层、哪些 experts 可以被平均。

## 3. Dense averaging 为什么有时有效

### 3.1 同源、同 basin、低曲率

Model soups 的核心经验是：多个从同一个大预训练模型 fine-tune 出来的 checkpoint，经常落在同一个低 error basin 里；权重平均不会增加推理开销，却可能提升准确率、鲁棒性和 OOD 表现。这个条件和本仓库里的 pretrained ViT transfer 结果一致：linear average 已经在一个宽共享 basin 里，grid best 只是再优化一点。

要验证这个条件，不能只看端点分数，要看：

- linear interpolation path 上有没有 loss/NLL barrier；
- average 点的 NLL 是否低于两个端点或接近最好端点；
- Hessian/Fisher 近似曲率是否低；
- 2D task-vector plane 是否存在共享低 loss 区域；
- validation soup 是否能贪心加入多个 checkpoint 而不降分。

### 3.2 同一个 base 的 task vectors 可加

Task Arithmetic 把 fine-tuned checkpoint 减去 base 得到 task vector，并用加法、缩放、取反来编辑行为。这对 Qwen 下游分支很自然：

```text
tau_i = theta_i - theta_anchor
theta_merge = theta_anchor + sum_i lambda_i * tau_i
```

但 task vector 相加只有在方向兼容时才稳。兼容性需要用 cosine、sign conflict、layer-wise conflict、NLL barrier 证明，而不是靠“同一个 Qwen 家族”推断。

## 4. Dense averaging 为什么失败

### 4.1 参数干扰

TIES-Merging 指出的两个主要问题是：小幅冗余参数会稀释有效 delta，不同模型对同一参数的符号方向会冲突。对应 probe 是：

| Probe | 公式/记录 | 解释 |
| --- | --- | --- |
| global cosine | `cos(tau_a, tau_b)` | 总方向是否一致。 |
| layer cosine | 每层或模块的 cosine | 找到冲突集中在哪些层。 |
| sign conflict | `sign(tau_a) != sign(tau_b)` 的比例 | TIES 的直接输入。 |
| weighted conflict | 用 `abs(tau_a * tau_b)` 加权的 sign conflict | 比普通比例更重视大 delta。 |
| delta redundancy | 小幅 delta 占比、top-k delta 累积质量 | DARE/DELLA 是否可能有效。 |

看到高 sign conflict 时，不是说“直接机械套 TIES”。更合理的是把它变成 Average 的约束：

```text
coordinate_weight_k = f(delta_magnitude_k, sign_agreement_k, fisher_k)
theta_avg,k = theta_anchor,k + sum_i w_i,k * tau_i,k
```

TIES/DARE/DELLA 可以看成这类 coordinate-wise average 的启发式实现。我们更关心的是：哪些坐标应该保留平均、哪些坐标应该降权、哪些坐标应该回到 anchor。

### 4.2 能力方向不等价

Qwen 代码、数学、金融、安全、长 CoT 分支不是同一种能力的多个随机种子。它们可能改变：

- 输出格式；
- chat template 依赖；
- 代码块风格；
- refusal policy；
- 思维链长度；
- 工具调用格式；
- tokenizer 高频 token 的概率分布。

所以 `avg_score` 不能代表合并成功。必须看 `worst_score`、held-in retention、format success 和 safety retention。

### 4.3 坐标没对齐

如果模型不是同一个 initialization / same-base fine-tune，直接平均可能只是神经元或特征坐标没对齐。Git Re-Basin、ZipIt、heterogeneous model merging 这类工作说明，合并前可能需要 permutation/feature/layer alignment。对 Qwen 同架构下游模型，这个问题比从零训练模型轻，但遇到不同 hidden size、不同 layer depth、不同 tokenizer 或不同 MoE expert ordering 时仍然存在。

## 5. Probe-guided Average 方法表

| 平均策略 | 适用信号 | 不适用信号 | 需要的 probe |
| --- | --- | --- | --- |
| linear average | same-base、低 barrier、端点同 basin | average NLL 明显高、worst task 掉队 | lambda path、2D plane、avg/worst score |
| greedy soup | 多个同任务/同分布 fine-tune | 专长任务差异很大 | validation gain、OOD gain |
| task arithmetic | task vectors 方向可加 | 大 sign conflict、端点语义冲突 | delta cosine、scale sweep |
| SLERP | 权重范数差异大 | task vectors 本身冲突 | norm ratio、angle |
| TIES | sign conflict 高，小 delta 多 | delta 方向同向且 dense | sign conflict、trim density |
| DARE | delta 冗余高，模型规模较大 | 少数关键 delta 不能丢 | delta magnitude distribution、drop sweep |
| DELLA | 小幅 delta 很多，随机 drop 太粗糙 | magnitude 与重要性弱相关 | magnitude-ranked retention curve |
| Fisher average | 有校准集，参数重要性差异大 | Fisher 噪声大或校准集偏 | diagonal Fisher、NLL calibration |
| RegMean / RegMean++ | linear layer 可用 activation statistics 拟合 | activation 分布拿不到或跨层强耦合 | activation covariance、layer residual |
| AdaMerging / layer-wise λ | 层间冲突差异大 | 无可用 unlabeled/validation samples | layer-wise entropy/NLL、coefficient trace |
| alignment / ZipIt | 不同初始化或异构结构 | same-base 且已低 barrier | feature similarity、permutation residual |

这里把这些方法称为“平均策略”是有意的：它们不应该被当成互斥菜谱，而应该被拆成可解释的平均权重来源。例如 Fisher average 用重要性加权，TIES 用符号一致性加权，RegMean 用 activation covariance 估计线性层的平均解，AdaMerging 用无标签数据学习 layer-wise 平均系数。

## 6. 必做 probe 清单

### 6.1 端点评测 probe

每个 source model 和 merge model 都要记录：

- `held_in_score_i`：第 `i` 个 expert 自己任务上的分数。
- `held_in_retention_i = score_merge_i / score_expert_i`。
- `general_retention = score_merge_general / score_anchor_general`。
- `worst_score = min_i score_i`。
- `avg_score = mean_i score_i`。
- `format_success`：JSON、tool call、答案格式、代码块、CoT 标签。
- `safety_retention`：safe prompt 回答质量、unsafe prompt 拒答质量。

### 6.2 语言模型 NLL probe

NLL 是便宜但很有解释力的预筛：

- general corpus NLL；
- response-only instruction NLL；
- math gold answer NLL；
- code canonical solution NLL；
- refusal target NLL；
- answer-letter margin；
- output KL divergence to each endpoint。

如果 merge 的 benchmark 分数低，但 NLL 没坏，问题可能在 generation / decoding / format；如果 NLL 已经坏，问题更可能是参数合并本身。

### 6.3 几何 probe

- `lambda sweep`：每个 expert 到 anchor 的 1D path。
- `alpha,beta grid`：两个 expert 的 2D task-vector plane。
- `barrier = max_loss(path) - max(loss_endpoint_a, loss_endpoint_b)`。
- `plane residual`：off-plane 方法投影回 task-vector plane 后的残差。
- `local curvature`：Hessian/Fisher 近似或 loss 对 λ 的二阶拟合。
- `method projection`：把 TIES/DARE/RegMean/layer-wise merge 投影到 plane 上看“移动到哪里”和“离 plane 多远”。

### 6.4 表示和功能 probe

- layer hidden-state CKA / cosine；
- endpoint logits KL；
- expert output similarity；
- answer style classifier；
- chain length / language mix；
- refusal over-trigger / under-trigger；
- code unit-test failure type。

这些 probe 用来区分“模型不会了”和“模型会但输出协议坏了”。

## 7. MoE 的特殊性

Sparse MoE 和 Dense 最大的区别是条件计算：每个 token 只激活一小部分 experts。经典 sparsely-gated MoE、Switch Transformer、Qwen3-30B-A3B 这类模型都依赖 router/top-k 专家选择。Qwen3-30B-A3B 模型卡显示它有 128 个 experts、每 token 激活 8 个 experts；因此一个下游 fine-tune 可能主要改变某些专家和 router，而不是均匀改变全模型。

这会带来三个平均风险：

1. **router collapse**：合并后大量 token 被路由到少数 experts。
2. **expert semantic mismatch**：第 17 号 expert 在两个模型里功能不一定相同。
3. **shared/routed knowledge 混淆**：通用语法和领域专长可能分别在 shared modules 与 routed experts 中，平均会互相污染。

DeepSeekMoE 提出 shared experts + routed experts 的思路，正好说明 MoE 里“公共能力”和“专长能力”应该分开处理，而不是统一平均。

## 8. MoE 专用 probe

### 8.1 Router probe

对每个 MoE layer 和每类 prompt 记录：

- top-k expert frequency；
- top-1 expert frequency；
- router entropy；
- top-1/top-2 margin；
- route overlap with source model；
- route overlap after merge；
- load balance coefficient；
- effective number of experts；
- token drop / capacity overflow，如果实现暴露该信息。

解释：

- entropy 很低且 max expert fraction 很高：router collapse。
- route overlap 过低：合并改变了专家分工。
- route overlap 很高但任务分数下降：expert 权重本身被破坏。
- top-k margin 很小：路由不稳定，轻微平均可能改变专家选择。

### 8.2 Expert probe

对每个 expert FFN 记录：

- expert-level delta norm；
- expert-level cosine/sign conflict；
- expert output MSE/KL/cosine；
- 被哪些任务 token 激活；
- shared expert vs routed expert 的差异；
- expert importance：按路由频率、prob mass、NLL sensitivity 估计。

一个非常实用的矩阵是：

```text
rows = experts
cols = tasks
value = route frequency / NLL sensitivity / delta norm
```

这张矩阵能告诉我们哪些 experts 是通用的，哪些是数学/代码/领域专长，哪些几乎不用。

### 8.3 MoE 合并几何

Dense 的 task-vector plane 仍然有用，但需要加两层：

- shared-parameter plane：只合并 attention/norm/shared modules；
- expert-parameter plane：只合并被选中的 expert FFN；
- router-frozen vs router-averaged 对比；
- per-expert plane residual；
- route-conditioned NLL surface：只在数学/代码/领域 prompt 上看对应路由区域。

## 9. MoE 的 probe-guided averaging 路线

### 9.1 最小可行路线

1. 端点评测：`moe_general`、`moe_code`、`moe_downstream`。
2. router probe：先不合并，记录每类 prompt 的 route distribution。
3. all-weight average：作为负 baseline。
4. router-frozen average：router 用通用 anchor，只 merge non-router 权重。
5. shared-only average：只平均 attention/norm/shared FFN，expert FFN frozen。
6. expert-wise average：只平均被高频激活或低冲突的 experts。
7. 小校准集 router weighting：只学习 router 或 router bias 的平均/校准系数。

最终 checkpoint 仍然是一个平均后的同构 MoE；只是不再要求所有参数组使用同一个 `w_i`。换句话说，router 矩阵、expert FFN 数量和每个张量 shape 都不变，改变的是每个张量从哪些源 checkpoint 取多大平均权重。

### 9.2 Expert matching 是平均前的对齐

不要假设 expert index 对齐。对两个 MoE checkpoint，可以用以下相似度做 matching：

```text
score(i, j) =
  a * weight_cosine(expert_i, expert_j)
+ b * output_cosine(expert_i(x), expert_j(x))
+ c * route_coactivation(i, j)
+ d * task_profile_similarity(i, j)
```

然后用 Hungarian matching 或 greedy matching 对齐 experts。匹配后再做 expert-wise average：

```text
expert_avg,e = sum_i w_e,i * expert_i,matched_i(e)
```

这仍然是 Average，只是先解决“第 e 个 expert 在不同 checkpoint 中是否表示同一功能”的坐标问题。

本仓库现在把这一步落到了同构 checkpoint 写出路径里：[Toy MoE Expert Remap Plan](results/toy_moe_expert_remap_plan/report.md) 会把 `expert_match.csv` 转成 `source_tensor_aliases.txt`。alias 规则的语义是：输出 checkpoint 仍写 `experts.e.*`，但某个 source 在读取输入 tensor 时可以改读 `experts.matched(e).*`。因此 expert 数、router shape、tensor name 和 tensor shape 都不变，只是修正了 source expert index 的坐标系。

### 9.3 Route-conditioned static average

对于每个 expert 或每个 MoE layer，可以用 route frequency 和 task sensitivity 估计平均权重：

```text
w_layer,expert,i ∝
  validation_gain_i
  * route_frequency_i(layer, expert, task)
  * low_conflict_i(layer, expert)
```

然后写回静态 checkpoint。注意这里不是运行时动态 ensemble，而是用 routing probe 来估计静态平均权重。

### 9.4 Expert union 是诊断上限，不是最终 Average

如果两个 experts 冲突大但都重要，一个自然对照是暂时保留两个 experts，并让 router 选择：

```text
shared modules: merge
expert set: union(source experts)
router: initialize from anchor, then calibrate routing
```

这类似 WEMoE/MergeME 的思路：把静态参数平均变成输入条件动态选择。代价是参数更多，而且已经不满足“输出结构和输入模型一致”的 Average 定义，所以不能作为本项目最终目标模型。

在本项目里，expert union 只适合作为 upper bound / ablation：如果 expert union 很好，而任何静态 average 都很差，说明问题确实来自“互斥 expert 被压坏”。随后要研究的是如何把 union 的信息蒸馏、投影或压缩回同 expert 数、同 router shape 的 averaged checkpoint，而不是停在 union。

### 9.5 Router calibration

router 是 MoE average 的高杠杆参数。建议顺序是：

1. frozen router baseline；
2. average router baseline；
3. only router bias calibration；
4. only router linear weights calibration；
5. router + selected expert gates calibration。

校准目标可以是：

- validation NLL；
- load balance regularization；
- route overlap regularization；
- held-in retention multi-objective；
- expert entropy lower/upper bound。

### 9.6 LoRA/adapters 的平均

很多下游用户不会 full fine-tune Qwen3-30B-A3B，而是训练 LoRA/adapters。此时直接合并 base 权重不是最自然方案。更合理的是：

- base MoE frozen；
- 先做 LoRA delta 的 average / task arithmetic；
- 用 sign conflict、rank overlap、adapter output similarity 判断哪些 LoRA 可平均；
- 冲突小的 adapters 再做 coordinate-wise average；
- 冲突大的 adapters 用 mixture 作为上限，再研究如何蒸馏回同构平均 adapter。

Mixture of LoRA Experts 的价值在于提供一个“不要平均会怎样”的参照。如果 mixture 明显好于 average，就说明平均前还缺少对齐、筛选、分组或蒸馏；它本身不是最终 Average，除非用户的目标模型本来就是同构 MoE-adapter。

## 10. Probe 到 Average 决策的映射

| 看到的现象 | 优先解释 | 下一步 |
| --- | --- | --- |
| average NLL 高于所有端点 | 参数合并破坏共同 basin | 降低/重学 `w_i`，用 barrier penalty，不直接 `1/n` |
| avg score 高但 worst score 很低 | 某个 expert 能力被牺牲 | 用 min-max 目标选平均权重 |
| general retention 低 | anchor 能力被覆盖 | 降低总 delta 权重，embedding/lm_head/norm 单独平均 |
| code/math retained 但格式坏 | decoding/模板/后训练格式冲突 | format probe 进约束，必要时只平均非格式层 |
| sign conflict 集中在 MLP | 专长参数冲突 | coordinate-wise average 降权冲突坐标 |
| sign conflict 集中在 norm/lm_head | 全局分布漂移 | freeze 或 Fisher-weighted average |
| layer-wise cosine 有正有负 | 层作用不同 | layer-wise average coefficient |
| pairwise barrier 高 | 直线不连通 | 不把该模型直接加入 soup，先 alignment/bridge |
| MoE route entropy 过低 | router collapse | router 不同权平均，加入 load-balance 约束 |
| MoE route overlap 低 | router 语义漂移 | router frozen 或 route-overlap regularized average |
| route overlap 高但分数低 | experts 被平均坏 | expert matching 后再 average |
| high-frequency experts 冲突大 | 专长都挤在少数 experts | expert-wise 权重降冲突，union 只作为上限 |
| low-frequency experts delta 很大 | 可能是领域稀有能力 | 按 task route 保留，不按全局平均剪掉 |

## 11. 和本仓库下一步的衔接

本仓库下一步最该补的不是更多小图，也不是盲目实现一堆合并算法，而是三个能反过来指导 Average 的 probe：

1. **Dense delta/connectivity probe v2**：在现有 `probe_qwen_deltas.py` 基础上输出 layer/module/tokenizer group 的 cosine、sign conflict、relative norm、top-k delta mass，以及 pairwise interpolation barrier。
2. **Checkpoint topology inspect**：新增 `scripts/inspect_checkpoint_topology.py`，只读 config 和 safetensors header，不加载权重；当前 [results/checkpoint_topology_inspect/report.md](results/checkpoint_topology_inspect/report.md) 显示本地 Qwen3.5-35B-A3B config 有 `256` experts、每 token 激活 `8` 个 experts，active fraction `0.03125`。
3. **MoE routing probe**：新增 `scripts/probe_moe_routing.py`，对 Qwen3 MoE 记录 router top-k、entropy、load balance、route overlap，用于估计 router/expert 的平均权重；脚本会写 `router_summary.csv`、`expert_load.csv`、可选 `route_overlap.csv`，以及 `summary.json`/`report.md`，tiny smoke 见 [results/moe_routing_probe_smoke/report.md](results/moe_routing_probe_smoke/report.md)。
4. **Average decision report**：新增 `scripts/build_average_decision_report.py`，把端点评测、delta probe、routing probe、lambda/grid sweep 汇总成一张“平均权重应该怎么设”的自动报告，当前输出在 [results/average_decision_report/report.md](results/average_decision_report/report.md)。
5. **Model averaging literature matrix**：新增 `scripts/build_model_averaging_literature_review.py`，把 Dense/MoE averaging 相关论文整理成方法矩阵、probe 矩阵和 MoE 优化 gate，当前输出在 [results/model_averaging_literature_review/report.md](results/model_averaging_literature_review/report.md)。
6. **Average candidate recipes**：新增 `scripts/build_average_candidate_recipes.py`，把 decision report 转成 materialize/skip/template 三类候选，当前输出在 [results/average_candidate_recipes/report.md](results/average_candidate_recipes/report.md)；它会把 endpoint-only best grid 和已失败的 uniform average 排除在 writer 命令之外。
7. **MoE same-shape average plan**：新增 `scripts/build_moe_average_plan.py`，把 router entropy/load/overlap 和 expert route-frequency 转成参数组级别计划，当前输出在 [results/moe_average_plan/report.md](results/moe_average_plan/report.md)。
8. **MoE routing readiness**：新增 `scripts/analyze_moe_routing_readiness.py`，把 `router_summary.csv`、`route_overlap.csv`、`expert_load.csv` 转成 router collapse、route drift、top-k boundary 和 expert load 风险表，当前模板输出在 [results/moe_routing_readiness/report.md](results/moe_routing_readiness/report.md)。
9. **MoE route-weight recipes**：新增 `scripts/build_moe_route_weight_recipes.py`，把 `expert_load.csv` 里的 per-category route mass 转成 checkpoint writer 可读的 `tensor_rules.txt`，当前模板输出在 [results/moe_route_weight_recipes/report.md](results/moe_route_weight_recipes/report.md)；没有真实 routing probe 时显式标成 `waiting_for_routing_probe`。
10. **Toy MoE route-aware merge**：新增 `scripts/run_toy_moe_merge.py`，用可控 soft-router MoE 验证 expert-index mismatch、expert matching 和 route-frequency expert average，当前输出在 [results/toy_moe_merge/report.md](results/toy_moe_merge/report.md)。
11. **Toy MoE expert remap plan**：新增 `scripts/build_moe_expert_remap_plan.py`，把 expert-output matching 转成 checkpoint writer 可读的 source tensor alias 文件，当前输出在 [results/toy_moe_expert_remap_plan/report.md](results/toy_moe_expert_remap_plan/report.md)。
12. **Same-shape checkpoint writer**：新增 `scripts/write_same_shape_average_checkpoint.py`，按 `theta_out = theta_base + sum_i w_i * (theta_i - theta_base)` 写出 safetensors checkpoint，并支持 `--freeze-router`、`--freeze-regex`、regex 级别权重覆盖、`--tensor-rule-file` 和 `--source-tensor-alias-file`；Qwen2.5-0.5B base/instruct/coder dry-run 见 [results/same_shape_writer_smoke/report.md](results/same_shape_writer_smoke/report.md)。

这个 report 的作用不是替代真实评测，而是把已有证据转换成明确决策：

```text
uniform_average_ok
coefficient_search
structured_average
avoid_uniform_average
```

例如当前 Qwen instruct/coder slice 中，`linear_average` 的 worst NLL 明显高于端点和 best grid，midpoint barrier 为正，因此报告给出 `avoid_uniform_average`，建议先做 connectivity/barrier 筛选和同构平均权重重估，而不是把两个 expert 做 `0.5/0.5`。

MoE plan 的关键规则是：

```text
router: frozen -> route-overlap/load-balance calibration -> small-lambda average
shared attention/norm: validation coefficient or layer-wise coefficient
shared MLP: conflict-aware coordinate average
experts: match by output/route profile, then route-frequency weighted average
LoRA/adapters: mixture only as upper bound, then distill/compress back to same-shape adapter
```

这个规则吸收了 2025-2026 的几条 MoE merging 分析：MergeME 指出 MoE 的 simple unweighted averaging 会遇到 parameter interference 和 routing 问题；Sub-MoE 强调 expert specialization 带来的冲突，需要先按 expert output similarity 聚类/对齐；Expert Merging 强调用 unlabeled calibration 学 layer-wise coefficients；HARC 则直接指出 MoE merging 的一个核心失败模式是 routing breakdown，因此 router 必须单独校准。

checkpoint writer 的边界也很明确：它只负责 materialize 同构平均结果，不替代 probe 和验证集选择。也就是说，`w_i`、router freeze、expert-wise 权重等决策应来自 Average decision report、MoE plan 和 held-out validation；writer 只保证输出 tensor names/shapes 与 base 一致。

source tensor alias 是这个边界里的对齐机制，而不是结构扩展机制。它允许 `code::experts.0.* -> code::experts.1.*` 这类读取重定向，用来实现 expert matching；输出端仍然只写 base checkpoint 里已有的 `experts.0.*` 张量，所以不违反“目标模型结构必须和 input 模型一样”的约束。

candidate recipes 的边界也很重要：当前 Qwen instruct/coder 的 `best_grid_alpha=1,beta=0` 是 endpoint，不应被包装成“成功 average”。这说明当前这两个 0.5B 分支还不是好的平均候选，需要换更代表性的下游模型、加入更多 probe 或做 layer/module-wise average；`0.5/0.5` 只适合作为负 baseline。

route-weight recipes 的边界同样明确：它不会凭空发明 expert-wise 权重。没有真实 MoE `expert_load.csv` 时，只生成 shared attention 规则和 dry-run writer 模板；一旦 `scripts/probe_moe_routing.py` 跑出 routing 数据，它才会按 prompt category 的 route mass 归一化出每个 layer/expert 的 source delta 系数。

routing readiness 是 route-weight recipes 的前置 gate：如果 `route_overlap` 很低、`max_top1_fraction` 过高、`effective_top1_experts / num_experts` 太低，或 expert load 高度集中，就先冻结/校准 router，而不是让 writer 开放 router delta。这个 gate 直接对应 HARC 讨论的 routing breakdown，也对应 Sub-MoE/MergeMoE 对 expert specialization 和 output alignment 的强调。

toy MoE 的作用是把这个逻辑先跑通：在保持同构的前提下，code source 被做了函数等价的 expert permutation。直接 all-weight average 的 worst accuracy 掉到 `0.620`，而 expert-output matching 后达到 `0.800`，route-frequency expert average 达到 `0.790`。这不是 Qwen3 结论，但它证明了后续 Qwen3 MoE 实验必须记录 expert index、router overlap、expert load 和 route mass。

更进一步，[Toy MoE Routing Readiness](results/toy_moe_routing_readiness/report.md) 已经把同一个 readiness gate 应用到多方法 probe 上：`code_endpoint_permuted` 和 `all_weight_average` 被标出低 route overlap / 低 top-1 agreement，`expert_matched_average` 和 `route_aware_expert_average` 则保持接近 base 的 routing overlap。也就是说，readiness gate 不只是“等 Qwen 数据”的模板，它已经能解释 toy MoE 里的具体失败模式。

[Toy MoE Method Selection](results/toy_moe_method_selection/report.md) 则把这一步再往前推：它把 method metrics 和 routing readiness 合成 `baseline_only`、`reject_routing_breakdown`、`reject_underperforms_base`、`candidate_with_router_guard` 等决策。当前 toy MoE 中，`all_weight_average` 被拒绝为 routing breakdown，推荐的是 `expert_matched_average`，但要求保留 router guard。这就是后续 Qwen3 MoE 应该复制的决策闭环。

[Toy MoE Expert Remap Plan](results/toy_moe_expert_remap_plan/report.md) 则把推荐方法推进到 materialization：当前 4 个 output expert 都找到 matched source expert，最小 output cosine 为 `0.943`，生成的 `source_tensor_aliases.txt` 可直接传给 checkpoint writer。它还没有替代真实 Qwen3 MoE probe，但已经证明“先对齐 expert，再写回同构 average checkpoint”这条路径在工程上是可执行的。

第一批可以直接接 [Qwen Dense/MoE 下游微调模型合并实验方案](QWEN_DENSE_MOE_EXPERIMENT_PLAN.md) 里的 Dense 7B：

```text
Qwen2.5-7B-Instruct
Qwen2.5-Coder-7B-Instruct
Qwen2.5-Math-7B-Instruct
DeepSeek-R1-Distill-Qwen-7B
```

MoE 部分则从：

```text
Qwen3-30B-A3B-Base
Qwen3-30B-A3B
Qwen3-Coder-30B-A3B-Instruct
```

开始，不急着全参数 merge，先跑 routing probe。

## 12. 参考资料

- Model Soups: https://arxiv.org/abs/2203.05482
- Task Arithmetic: https://arxiv.org/abs/2212.04089
- Fisher merging: https://arxiv.org/abs/2111.09832
- Git Re-Basin: https://arxiv.org/abs/2209.04836
- ZipIt: https://arxiv.org/abs/2305.03053
- TIES-Merging: https://arxiv.org/abs/2306.01708
- DARE: https://arxiv.org/abs/2311.03099
- DELLA: https://arxiv.org/abs/2406.11617
- AdaMerging: https://arxiv.org/abs/2310.02575
- RegMean++: https://arxiv.org/abs/2508.03121
- What Matters for Model Merging at Scale: https://arxiv.org/abs/2410.03617
- Sparsely-Gated MoE: https://arxiv.org/abs/1701.06538
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Expert Choice Routing: https://arxiv.org/abs/2202.09368
- Sparse Upcycling: https://arxiv.org/abs/2212.05055
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- OLMoE: https://arxiv.org/abs/2409.02060
- Mixture-of-Experts Meets Instruction Tuning: https://arxiv.org/abs/2305.14705
- Mixture of LoRA Experts: https://arxiv.org/abs/2404.13628
- WEMoE: https://arxiv.org/abs/2410.21804
- MergeME: https://arxiv.org/abs/2502.00997
- MergeMoE: https://arxiv.org/abs/2510.14436
- Sub-MoE: https://arxiv.org/abs/2506.23266
- Expert Merging: https://arxiv.org/abs/2509.25712
- HARC / When Model Merging Breaks Routing: https://arxiv.org/abs/2606.03391
- Subspace-Boosted Model Merging: https://arxiv.org/abs/2506.16506
