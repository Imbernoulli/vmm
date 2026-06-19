# Qwen Dense/MoE 下游微调模型合并实验方案

更新时间：2026-06-18

这份方案把后续实验范围从“官方 Base vs Instruct”收紧并升级为：围绕 Qwen 官方底座、官方专长分支、以及下游用户/实验室基于 Qwen 继续 SFT/RL/蒸馏得到的模型，系统研究 Dense 模型和 MoE 模型上的 model averaging、线性插值和 probe-guided averaging。这里的 Average 指最终输出仍然是和输入模型同构的 checkpoint，可被同一个 model class/config/tokenizer 直接加载；probe 不是为了堆指标，而是为了判断哪些模型、层、模块、experts 处在可平均的连通区域里，并据此找到更好的平均系数。

核心判断是：真实的合并对象不应只是 `Qwen-Base -> Qwen-Instruct` 这一条官方路径，而应是多个同源微调分支之间的能力整合。例如一个实验室基于 Qwen2.5-Instruct 做金融推理，另一个团队基于 Qwen2.5-Math 或 Qwen2.5-Base 做长思维链推理，另一个团队做代码或 agentic coding；这些分支是否能合成一个更通用的 checkpoint，才是 model merging 更有价值的问题。

## 0. 已有仓库证据

当前仓库已经有一个最小 Qwen 起点：

- [Qwen2.5-1.5B base-to-instruct path](results/qwen_path_sweep/report.md)：`lambda=0.75` 在小 MMLU/safety slice 上比端点更稳，说明 dense 路径上存在有用中间点。
- [Qwen GSM8K/MMLU/HumanEval/safety slices](results/qwen_gsm8k_slice/report.md)：已有一套小规模 benchmark-slice 框架，可以扩展成真实评测。
- [Qwen2.5-0.5B instruct+coder merge plane](results/qwen_multi_expert_merge/report.md)：`linear_average` 明显退化，说明“两个专长分支直接 0.5/0.5 平均”不是可靠默认方法。
- [Qwen multi-expert 3D surface](results/figures_3d/qwen_multi_worst_nll_surface.png)：已经可以把 Qwen 的 `alpha,beta` 合并系数可视化。

这些结果只证明框架能跑通，不足以回答“下游微调模型怎样合并最好”。下一步应换成更有代表性的 Qwen 分支和更完整的评测。

## 1. 代表性模型和场景

### 第一批：Dense 7B，下游专长分支合并

这是最建议先做的主实验，因为 7B 规模能在单机/少卡上反复跑，且下游模型生态足够丰富。

| 角色 | 候选模型 | 为什么选 |
| --- | --- | --- |
| 通用指令 anchor | `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0，通用指令能力强，模型卡显示大量下游 adapters/finetunes/merges。 |
| 代码 expert | `Qwen/Qwen2.5-Coder-7B-Instruct` | 官方代码专长分支，基于 Qwen2.5 系列，模型卡明确覆盖代码生成、代码推理、代码修复。 |
| 数学 expert | `Qwen/Qwen2.5-Math-7B-Instruct` | 官方数学专长分支，支持中英数学、CoT 和 Tool-Integrated Reasoning。 |
| 推理蒸馏 expert | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | DeepSeek 明确说明这是用 R1 生成样本继续微调的 Qwen 系列 dense 模型；属于典型第三方/下游再训练分支。 |
| 领域 expert | `DianJin-R1-7B` 或同类金融/医疗/法律 Qwen 微调模型 | 用来模拟实验室/行业团队基于 Qwen-Instruct 做领域 SFT/RL 的场景；优先选公开权重、同 tokenizer、同架构的模型。 |

第一批实验目标不是追最大榜单，而是回答：

```text
通用指令 + 代码 + 数学/推理 + 一个领域微调分支，能否合成一个 worst-task 不明显掉队的 dense checkpoint？
```

### 第二批：Dense 32B，验证规模效应

32B 适合作为第二阶段，因为近期很多 reasoning 模型都选择 Qwen2.5-32B 作为底座。

| 角色 | 候选模型 | 为什么选 |
| --- | --- | --- |
| 通用/底座 | `Qwen/Qwen2.5-32B` 或 `Qwen/Qwen2.5-32B-Instruct` | Qwen2.5 官方 dense 中大型基线。 |
| 推理 expert | `a-m-team/AM-Thinking-v1` | 模型卡/论文说明它基于 Qwen2.5-32B-Base，通过 SFT + 两阶段 RL 做 reasoning。 |
| 推理蒸馏 expert | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | DeepSeek R1 distill 系列中性能强、影响力大的 Qwen dense 分支。 |
| 长推理 expert | `Long1K-32B` | 论文说明它基于 Qwen2.5-32B-Instruct 用 Long1K 数据微调，适合检验长 CoT 风格是否能 merge。 |
| 领域 expert | `DianJin-R1-32B` 或同类 | 金融推理类下游训练，适合作为“实验室/行业微调”的代表。 |

这里重点看两个问题：

- 规模变大后，平均/DARE/TIES 是否更容易保留多任务能力。
- reasoning 风格的多个分支是否只是“同方向增强”，还是会在格式、长度、拒答、安全、代码能力上互相干扰。

### 第三批：Qwen3 MoE，route-aware merging

MoE 不能简单照搬 dense 平均。Qwen3-30B-A3B 是最合适的第一批 MoE 对象，因为它总参数 30.5B、激活参数约 3.3B，模型卡显示 128 个 experts、每 token 激活 8 个 experts，规模比 235B-A22B 更现实。

| 角色 | 候选模型 | 为什么选 |
| --- | --- | --- |
| MoE base | `Qwen/Qwen3-30B-A3B-Base` | 官方 Qwen3 MoE 预训练底座；适合作为 delta/reference。 |
| 通用 MoE post-trained | `Qwen/Qwen3-30B-A3B` | 官方 post-trained MoE，支持 thinking / non-thinking 模式，是通用 MoE anchor。 |
| 代码 MoE expert | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 官方 MoE 代码/agentic coding 分支；同为 30B-A3B、128 experts、8 activated experts。 |
| 下游 MoE adapters/finetunes | Hugging Face 模型树里的 `Qwen3-30B-A3B-Base` / `Qwen3-Coder-30B-A3B-Instruct` adapters 和 finetunes | 更接近用户/实验室自己训练 LoRA 或 full fine-tune 的真实场景。 |

MoE 阶段的目标不是先追求“全权重平均”，而是比较三类策略：

1. **Router frozen merge**：冻结 router，只 merge attention/shared modules 和被路由到的 expert FFN 权重。
2. **Expert matching merge**：先按 expert 激活分布、router co-assignment、weight cosine 或 expert output similarity 对齐 experts，再 merge。
3. **Adapter/MoLE 风格对照**：如果下游模型主要是 LoRA/adapters，可以用 mixture-of-LoRA-experts 判断“不要压成一个 delta”的上界；最终仍要压回同构 adapter 或同构 full checkpoint。

在真实 Qwen3 MoE 之前，toy sanity check 已经把这个流程跑通：`results/toy_moe_merge/report.md` 复现 expert-index mismatch，`results/toy_moe_method_selection/report.md` 选择 `expert_matched_average` 并保留 router guard，`results/toy_moe_expert_remap_plan/report.md` 则把 expert matching 转成 `source_tensor_aliases.txt`，用于同构 checkpoint writer。真实 Qwen3 实验要复制这条链路，而不是直接把所有 expert 同权平均。

## 2. 我的评测内容是什么

评测要分成“能力保留”“能力互扰”“合并几何”“MoE 路由”四层。

### 2.1 能力保留

每个源模型的 held-in 能力必须保留，否则合并没有意义。

| 能力 | 推荐数据/指标 | 说明 |
| --- | --- | --- |
| 通用知识 | MMLU、C-Eval、CMMLU 小/全量切片 | 看合并后是否低于通用 instruct anchor。 |
| 数学推理 | GSM8K、MATH-500、AIME slice | 同时报 exact match、gold-answer NLL、format compliance。 |
| 代码 | HumanEval、MBPP、LiveCodeBench slice | NLL 只是预筛；最终要跑 pass@1 或单元测试。 |
| 指令跟随 | IFEval、MT-Bench/AlpacaEval 类 judge | 重点看格式、约束遵守、多轮稳定性。 |
| 安全/拒答 | BeaverTails、AdvBench、安全拒答 NLL | 避免合并把领域/代码能力换成安全能力下降。 |
| 领域任务 | 金融：CFLUE、FinQA、CCC；医疗/法律按模型选择 | 只有加入对应下游 expert 时才进入主指标。 |
| 长上下文/agentic | LongBench、Needle、repo-level coding 或工具调用任务 | 对 Qwen3-Coder/Qwen3-MoE 尤其重要。 |

主表不要只报平均分，必须报：

- `avg_score`：所有任务平均。
- `worst_score`：最差任务。
- `held-in_retention`：每个 expert 自己擅长任务相对源模型的保留率。
- `general_retention`：相对通用 instruct/base 的通用能力保留率。
- `format_success`：答案格式、JSON、tool call、CoT 标签等是否还稳定。
- `cost`：显存、latency、tokens/s、MoE activated params。

### 2.2 能力互扰

只看最终分数很难解释失败。每个合并点都要额外记录：

- response-only NLL：通用、数学、代码、领域、安全 prompt 各一组。
- gold answer margin：正确答案和最强错误答案的 logprob 差。
- refusal over-trigger / under-trigger：安全模型常见问题。
- chain-of-thought length：推理模型合并后是否过长、过短、空 `<think>`、语言混杂。
- code syntax/test failure type：语法错、超时、逻辑错、导入错。
- benchmark prompt sensitivity：zero-shot、CoT、few-shot CoT 是否对合并方法异常敏感。

### 2.3 合并几何

对 dense 和 MoE 都要记录这些 probe：

- delta norm by module/layer：哪些层被哪个 expert 改得最多。
- cosine similarity：task vector 全局和分层 cosine。
- sign conflict / weighted conflict：TIES 类方法需要这个证据。
- interpolation barrier：`lambda in [0,1]` 路径上是否有高 loss ridge。
- 2D/3D task-vector plane：选两个主要 experts 画 `alpha,beta -> worst_score/worst_NLL`。
- plane residual：RegMean、layer-wise merge、router-aware merge 是否离开原始二维平面。
- Fisher/activation covariance：给 Fisher average、RegMean、layer-wise λ 提供依据。

### 2.4 MoE 路由专用 probe

MoE 的关键是“哪些 token 被送到哪些 expert”。如果只看总权重差，会漏掉 MoE 特质。

| Probe | 记录什么 | 为什么重要 |
| --- | --- | --- |
| router top-k distribution | 每层每类 prompt 的 top-8 expert 频率 | 看数学/代码/通用 token 是否使用不同 expert。 |
| load balance / entropy | expert 负载分布、router entropy、top-k margin | 发现 expert collapse 或路由过度集中。 |
| route overlap | 合并前后同一 prompt 的 top-k Jaccard | 判断合并是否破坏原来的专家分工。 |
| expert delta norm | 每个 expert FFN 的 delta norm/cosine/conflict | 找出哪些 expert 是共享能力，哪些是专长能力。 |
| router logit drift | router 权重和 logits 的变化 | 决定 router 应 frozen、平均、还是单独校准。 |
| expert output similarity | 同一 token 上 expert 输出的 CKA/MSE/cosine | 比单纯 weight cosine 更接近功能相似性。 |
| token-level routing trace | 数学、代码、金融 prompt 的路由轨迹 | 给 MoE 合并失败提供可解释证据。 |

## 3. 我该如何评测

### 3.1 基础流程

1. **模型资格检查**：确认 architecture、hidden size、layer count、vocab/tokenizer、chat template 是否可对齐。最终 average 目标必须保持同构；不能对齐的模型只做 LoRA/adapter merge、蒸馏或输出层适配对照，不进入直接 full-weight average 主实验。
2. **端点评测**：所有源模型先跑同一套 evaluation suite，端点不过关的模型不进入合并。
3. **delta/probe 预扫描**：计算每个 expert 相对 anchor 的 delta norm、cosine、sign conflict、layer/module 分布。
4. **一维路径**：先跑 `base/instruct -> expert` 或 `anchor -> downstream expert` 的 `lambda sweep`，观察中间点和 barrier。
5. **二维平面**：选择两个最重要 experts 跑 `alpha,beta grid`，先小切片，后 full eval。
6. **方法对比**：平均、task arithmetic、greedy soup、SLERP、TIES、DARE、DELLA、Fisher、RegMean、layer-wise λ、MoE route-aware variants。
7. **验证集选点**：所有 λ、density、drop rate、layer-wise coefficient 只在 validation slice 上选择。
8. **测试集报告**：最终只在 held-out test slice 上报告一次，并加 bootstrap confidence interval。

### 3.2 Dense merge 方法

先跑这些 baseline：

- `linear_average`：朴素平均，是必须打败的 baseline。
- `task_arithmetic`：`theta = theta_anchor + λ * sum(delta_i)`。
- `greedy_model_soup`：按 validation 分数贪心加入候选 checkpoint。
- `SLERP`：检查向量范数/角度影响。
- `TIES`：trim + sign election + merge，专门处理符号冲突。
- `DARE` / `DELLA`：drop/rescale delta，缓解冗余和干扰。
- `Fisher average`：用小校准集估计重要参数。
- `RegMean` / `layer-wise λ`：适合解释为什么某些层应该少合或不合。

调参最少需要这些轴：

```text
lambda:     -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25
density:    0.2, 0.4, 0.6, 0.8, 1.0
drop_rate:  0.2, 0.5, 0.7, 0.9
layer_gate: attention / mlp / norm / embedding / lm_head 分组开关
```

### 3.3 MoE merge 方法

MoE 合并不应默认“全参数同权平均”。建议把参数分组：

| 参数组 | 默认策略 | 原因 |
| --- | --- | --- |
| embedding / lm_head | 小心平均或冻结通用 anchor | 直接平均容易影响 tokenizer 近邻和输出分布。 |
| attention | task arithmetic / TIES / DARE | attention 是共享表示，通常可作为 dense 部分处理。 |
| shared norms | freeze 或小 λ | norm 漂移会造成全局不稳定。 |
| router | 首轮 frozen；第二轮单独校准 | router 决定 token 到 expert 的分配，盲目平均风险高。 |
| expert FFN | expert matching 后同构合并 | MoE 专长主要在 FFN experts；最终不能改变 expert 数量或张量 shape。 |

需要比较的 MoE 策略：

1. **All-weight average**：作为负/朴素 baseline。
2. **Router-frozen average**：router 用通用或目标 anchor，其余模块合并。
3. **Router-calibrated merge**：合并后只用小校准集训练/校准 router。
4. **Expert-matched merge**：先按 routing/activation/weight similarity 对齐 expert，再逐 expert 合并。
5. **Expert union / sparse branch retention 对照**：冲突大的 expert 暂时保留成额外 experts，通过 router 选择；这只作为诊断上界，不是最终 average checkpoint。
6. **MoLE / adapter MoE 对照**：如果下游是 LoRA，可以把 LoRA 当专家做输入自适应路由，用来判断平均损失来自哪里；最终需要蒸馏或压缩回同构 adapter。
7. **MergeMoE-style expert output merge**：把 expert merging 看成输出空间压缩/拟合，适合把上界方案压回同 expert 数的 MoE。

## 4. 待选目标模型

### 推荐的第一轮真实实验

第一轮不要铺太大，先跑下面 4 个 dense 7B 模型：

```text
anchor: Qwen/Qwen2.5-7B-Instruct
expert_code: Qwen/Qwen2.5-Coder-7B-Instruct
expert_math: Qwen/Qwen2.5-Math-7B-Instruct
expert_reasoning: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
optional_domain: 一个公开 Qwen2.5-7B-Instruct 金融/医疗/法律微调模型
```

输出目标：

```text
Qwen2.5-7B dense merged assistant
= 保留通用指令 + 数学 + 代码 + 推理/领域能力的单 checkpoint
= 架构、tokenizer、hidden size、layer count 与 Qwen2.5-7B 输入模型一致
```

第一轮必须产出：

- 端点模型总表。
- `anchor -> each expert` lambda sweep。
- `math/code`、`code/reasoning`、`general/domain` 三张 2D merge plane。
- merge 方法表：average、task arithmetic、TIES、DARE、DELLA、greedy soup、layer-wise λ。
- 每个方法的 `avg_score`、`worst_score`、`held-in_retention`、`general_retention`。
- delta/probe 报告：layer norm、cosine、sign conflict、NLL barrier。
- Checkpoint topology inspect：只读 config/safetensors header，确认 model_type、hidden size、layer count、expert 数、每 token 激活专家数、router/expert tensor 分组，以及候选模型是否同构。
- Average decision report：把 grid、method、delta conflict、routing probe 汇总成同构 checkpoint 的平均权重建议；输出 `uniform_average_ok`、`coefficient_search`、`structured_average` 或 `avoid_uniform_average`。
- Average candidate recipes：把 decision report 转成 `materialize`、`skip_endpoint_only`、`skip_rejected_by_probe`、`template_waiting_for_routing_probe`，避免把端点或失败的 uniform average 当作最终结果。
- MoE same-shape average plan：把 router entropy/load/overlap 和 expert route frequency 转成 router/shared/expert/adapter 的参数组平均策略；最终仍输出同 expert 数、同 router shape 的 MoE checkpoint。
- MoE routing readiness：用 `scripts/analyze_moe_routing_readiness.py` 把 `router_summary.csv`、`route_overlap.csv`、`expert_load.csv` 转成 router collapse、route drift、top-k boundary、expert load 和 category specialization 风险表；当前模板见 [results/moe_routing_readiness/report.md](results/moe_routing_readiness/report.md)。
- MoE route-weight recipes：用 `scripts/build_moe_route_weight_recipes.py` 把 `scripts/probe_moe_routing.py` 的 `expert_load.csv` 转成 checkpoint writer 可读的 `tensor_rules.txt`；当前模板见 [results/moe_route_weight_recipes/report.md](results/moe_route_weight_recipes/report.md)，prompt pack 见 [prompts/qwen_moe_route_probe_prompts.jsonl](prompts/qwen_moe_route_probe_prompts.jsonl)。
- Toy MoE sanity check：用 [results/toy_moe_merge/report.md](results/toy_moe_merge/report.md)、[results/toy_moe_routing_readiness/report.md](results/toy_moe_routing_readiness/report.md) 和 [results/toy_moe_method_selection/report.md](results/toy_moe_method_selection/report.md) 先在可控同构 MoE 上验证 expert-index mismatch、expert-output matching、route-frequency expert average、多方法 readiness gate 和最终方法选择；真实 Qwen3 MoE 实验必须至少复现同类 probe 表。
- Same-shape checkpoint：用 `scripts/write_same_shape_average_checkpoint.py` 把选好的全局/layer/module/router/expert 权重写成 safetensors，再在 held-out slice 上评测；不能只停在图和 CSV。

### 推荐的第二轮 MoE 实验

第二轮跑 MoE：

```text
moe_base: Qwen/Qwen3-30B-A3B-Base
moe_general: Qwen/Qwen3-30B-A3B
moe_code: Qwen/Qwen3-Coder-30B-A3B-Instruct
moe_downstream: 1-2 个公开 Qwen3-30B-A3B LoRA/adapters 或 finetunes
```

输出目标：

```text
Qwen3-30B-A3B route-aware merged MoE
= 同样激活参数预算下，保留通用/代码/下游专长，并且 router 不 collapse
```

MoE 实验必须多报：

- router top-k distribution by task。
- expert load balance / entropy。
- route overlap before/after merge。
- expert-level delta norm/cosine/sign conflict。
- latency、VRAM、tokens/s、activated params。

## 5. 成功标准

一个合并方法只有满足下面条件才算“真的有价值”：

1. `worst_score` 高于朴素 `linear_average`，最好也高于通用 anchor。
2. 每个 expert 的 held-in task 至少保留源模型能力的 `90%`，除非明确做压缩/折中实验。
3. 通用能力不能明显掉：MMLU/C-Eval/IF-Eval/safety 至少不低于 anchor 的 `95%`。
4. 格式和安全不能坏：JSON/tool call/refusal/CoT 标签要单独通过。
5. MoE 不允许 router collapse：expert entropy、load balance、route overlap 必须在合理范围内。
6. 合并方法要可解释：必须给出 delta conflict、layer/module attribution，MoE 还要给 routing probe。

## 6. 为什么这些模型有代表性

- Qwen2.5 官方博客说明 Qwen2.5 提供通用 LLM、Coder、Math 等系列，并覆盖 0.5B 到 72B 等多个规模；Coder/Math 是典型官方专长分支。
- Qwen3 官方博客说明 Qwen3 同时开源 dense 和 MoE 模型，其中 `Qwen3-30B-A3B` 是 30B 总参数、约 3B 激活参数的小 MoE，适合作为 MoE merge 的现实起点。
- `Qwen2.5-Coder-7B-Instruct` 模型卡显示它基于 Qwen2.5 系列，覆盖 0.5B/1.5B/3B/7B/14B/32B，并强调代码生成、代码推理和代码修复。
- `Qwen2.5-Math-7B-Instruct` 模型卡显示它支持中英文数学、CoT 和 Tool-Integrated Reasoning。
- DeepSeek R1 Distill 模型卡说明 `DeepSeek-R1-Distill-Qwen-7B` 基于 `Qwen2.5-Math-7B`，用 DeepSeek-R1 生成样本微调，是非常典型的第三方 Qwen 下游微调分支。
- AM-Thinking-v1 模型卡/论文说明它基于 Qwen2.5-32B-Base，通过 SFT + 两阶段 RL 做 reasoning，代表实验室级别的 32B dense reasoning 微调。
- DianJin-R1 论文说明其 7B/32B 模型从 Qwen2.5-Instruct 微调，代表金融领域推理微调。
- Long1K 论文说明其 Long1K-32B 从 Qwen2.5-32B-Instruct 微调，适合检验长推理数据对 merge 的影响。

## 7. 参考资料

- Qwen3 官方博客：https://qwenlm.github.io/blog/qwen3/
- Qwen2.5 官方博客：https://qwenlm.github.io/blog/qwen2.5/
- Qwen3-30B-A3B 模型卡：https://huggingface.co/Qwen/Qwen3-30B-A3B
- Qwen3-30B-A3B-Base 模型卡：https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
- Qwen3-Coder-30B-A3B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Qwen2.5-7B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen2.5-Coder-7B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Qwen2.5-Math-7B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct
- DeepSeek-R1-Distill-Qwen-7B 模型卡：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- AM-Thinking-v1 模型卡：https://huggingface.co/a-m-team/AM-Thinking-v1
- Qwen2.5 technical report：https://arxiv.org/abs/2412.15115
- Qwen2.5-Coder technical report：https://arxiv.org/abs/2409.12186
- Qwen2.5-Math technical report：https://arxiv.org/abs/2409.12122
- DeepSeek-R1 technical report：https://arxiv.org/abs/2501.12948
- AM-Thinking-v1：https://arxiv.org/abs/2505.08311
- Long1K：https://arxiv.org/abs/2503.18069
- DianJin-R1：https://arxiv.org/abs/2504.15716
- Model Soups：https://arxiv.org/abs/2203.05482
- Task Arithmetic：https://arxiv.org/abs/2212.04089
- TIES-Merging：https://arxiv.org/abs/2306.01708
- DARE：https://arxiv.org/abs/2311.03099
- DELLA：https://arxiv.org/abs/2406.11617
- What Matters for Model Merging at Scale：https://arxiv.org/abs/2410.03617
- Mixture of LoRA Experts：https://arxiv.org/abs/2404.13628
- WEMoE：https://arxiv.org/abs/2410.21804
- MergeME：https://arxiv.org/abs/2502.00997
- MergeMoE：https://arxiv.org/abs/2510.14436
