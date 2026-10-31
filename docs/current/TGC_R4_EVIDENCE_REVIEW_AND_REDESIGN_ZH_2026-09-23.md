# TGC-R4 证据审查、设计修正与实现路线

日期：2026-09-23。

状态：研究设计审查，不是实验完成报告，也不是已封存的运行协议。本文取代 2026-09-22 版 MB-LRR 的执行建议。本轮复核已有结果、关键代码和相关论文，未运行新训练、未修改已封存实验算法、未晋升新主方法。

## 1. 结论：保留 TGC 方向，撤回上一版的直接执行建议

**上一版设计有合理动机，但不足以直接开跑。** 主要问题不在于模块不够复杂，而在于把尚未证实的失败归因写成了事实，又据此引入 mass replay、局部亲和方向和另一个短训 gate。

本次建议收敛为：

> 以 C0 为可重现初始化，先检验合法 teacher 的目标是否有用；通过真实 student 训练路径优化新合成块，以未修订追加图的旧端关系作为约束参照，再用受控实验判断有限旧 X 修订是否必要。teacher 支持图的采样偏差和传播偏差分别诊断，不把 mass、几何距离或短程 loss 当成历史风险保证。

这仍然是一个完整持续凝聚算法的研究方向，不要求预算、文本、X 和拓扑每一部分都重新发明。但“使用 teacher/student”“关系匹配”“有限 revision”本身都不足以证明算法新颖性或 WWW 长论文贡献。

当前状态必须分开：

| 层次 | 当前判断 |
|---|---|
| 已冻结参照 | C0：低成本确定性合成追加，response off |
| 已完成探索 | R2/R2B/R2B2/R3；候选和 selector 尚未形成稳定优势 |
| 值得保留的信号 | 部分 student response、teacher 修订、语义拓扑在特定条件有收益 |
| 撤回的执行配方 | 默认 mass-balanced replay + 低维亲和位移 + 短训 acceptance |
| 本文新建议 | 新块优先的关系响应凝聚，旧端约束与可选有限修订；尚未实现和验证 |

## 2. 证据范围与不能混用的结论

本轮按预算、合成 X、文本、拓扑、记忆维护、选择器和 TGC 七条实验谱系回看项目报告，并核对 R3 机器汇总和关键实现；外部资料核对了 TAGSAM、STAD、DANCE 的论文及 CaT 的摘要。没有重新训练所有历史实验，也没有声称对全部结果文件做了新一轮独立审计。

解释结果时遵守以下边界：

- v3 以前的 batch、数据访问和旧版本 gate 结论不能直接拼入现行主表；早期“定律”“规模门槛”不是已建立的一般理论。
- 下表大部分差值是对应报告内部的受控比较，不应跨报告求一个统一平均提升。
- 历史 task-slice old/new F1 与现在 seen-prompt pooled old/new F1 不同。
- “未通过预注册门槛”不等于统计上证明无效，更不等于整个方法族无效。
- 新顺序和新 seed 不是新数据集；已用于开发的 report-dev 也不能重新称为未见测试集。
- 审计通过说明列出的访问、预算、哈希、重载和指标检查通过，不证明代码不存在其他错误。

### 2.1 各实验方向实际告诉了我们什么

| 方向与证据 | 关键观测 | 对本轮设计的约束 |
|---|---|---|
| [P1 小图预算](P1_SMALL_GRAPH_RESULTS_ZH_2026-09-15.md) | 固定 15 行存在 keep/new 配额混淆；受控后 dispersion 更有竞争力；adaptive-rate 不稳定 | 固定累计 5% 和 C0 allocation，先不让不可靠 response 控制总预算 |
| [P1-X](P1X_SYNTHETIC_X_COMPLETION_ZH_2026-09-18.md) | weighted-X 最终相对 mean-X 约 +0.02/-0.29 pp；moment-X -1.27/-0.86 pp，依次为 Cora/Citeseer | 简单改变均值几何不够；不是证明均值永远最优 |
| [C1 多原型](C1_MULTI_PROTOTYPE_COMPLETION_ZH_2026-09-19.md) | subcluster-moment 最终 +1.96/-0.83 pp，顺序依赖；同时改变局部配额和来源 | 多模态覆盖有信号，但不能把收益全部归于 X 算子 |
| [P2 固定 token 文本](P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md) | mean、attention、dual、DeepSets 等未形成普适优势 | 注意力不是缺失即有收益的组件；本轮固定文本构造 |
| [P2-A 图条件文本](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md) | graph selection 最终 +3.05/-1.85 pp；Cora 打乱图条件也能更好 | 改变文本有时有效，但真实图条件的独特贡献没有被证明 |
| [P2-B student-guided attention](P2B_STUDENT_GUIDED_ATTENTION_COMPLETION_ZH_2026-09-18.md) | 真实 hypergradient 已实现；最终 +0.18/-0.53 pp，构造更贵 | 不是“以前没有 ML”；梯度存在、proxy 下降均不保证完整 student 获益 |
| [P3-A response](P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md) | 最终 -0.93/+0.85 pp，selector 构造成本约为低成本控制的 400--600 倍 | 不继续默认昂贵候选搜索是有效核心 |
| [P1-B guarded response](P1B_GUARDED_RESPONSE_COMPLETION_ZH_2026-09-18.md) | 最终约 +1.82/+1.62 pp，但成本、接受率与独立确认不足 | 不能写成所有 response 都失败；应拆开候选质量、选择质量与代价 |
| [P3-B 拓扑](P3B_TOPOLOGY_DIAGNOSTIC_COMPLETION_ZH_2026-09-18.md) | 当前跨边相对无跨边最终 +2.97/+2.02 pp；旧旧归一化块仍漂移约 0.30--0.32 | 有用的新旧传播不能一概删除；冻结 edge_index 不等于冻结传播 |
| [P4 maintenance](P4_MEMORY_MANAGEMENT_COMPLETION_ZH_2026-09-18.md) | 简单 bounded refresh 最终约 -0.06/-2.64 pp | 向新数据几何插值不是被遗漏的可靠修复 |
| [v5 retention](V5_RETENTION_EXPERIMENT_COMPLETION_ZH_2026-09-19.md) | core 平均约 +0.30 pp，topology -2.36，full -2.91 | 组件叠加不具有收益可加性；小漂移也不是语义保持保证 |
| [C0-M](C0M_CORE_BUFFER_MAINTENANCE_COMPLETION_ZH_2026-09-22.md) | final -0.34、新类 -0.93 pp；仅 2/6 cell 在 C0-1 pp 以内；anchor 使 memory 约 1.63 倍 | 不再次增加未记账的永久 anchors；旧节点可改，但须验证真实效用 |
| [v6-A](V6A_LOCAL_MARGINAL_COMPLETION_ZH_2026-09-19.md) | local selector 8/8 回退，rank 约 -0.018；recluster 有局部收益 | 候选空间和代理目标都可能有问题，不宜一味收紧 gate |
| [v7-A](V7A_GRADIENT_COMPATIBLE_APPEND_COMPLETION_ZH_2026-09-19.md) | rank 约 0.125；早期很大的多原型收益未延续到最终阶段；6/8 有益候选被几何收益条件排除 | 当时的“相对 C0 梯度距离惩罚”不能代表所有参考梯度匹配；分数结构可能直接排除好候选 |
| [P5](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md) | 小图 1%/5%/10% 非单调；长链和顺序影响明显 | 不用“增加 rows/放宽半径一定更好”解释失败；须有独立递归验证 |
| [联合核](PHASE12_C0_JOINT_KERNEL_COMPLETION_ZH_2026-09-20.md) | 36 条件中 20 个正收益，平均 +0.95 pp，最差 -6.41 pp | 有局部机制价值，但均值不够说明可靠性 |
| [T+degree 强控制](TOPOLOGY_CAUSAL_ABLATION_RESULTS_ZH_2026-09-21.md) | 更新后的共同重跑：BookChild NC +1.55、LP AUC/AP +0.72/+0.98 pp；arXiv NC -0.32、LP 近零 | 大图/跨任务有真实积极信号；不应归纳为所有复杂模块无用 |
| [语义拓扑控制](SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md) | 普通 semantic kNN/local-cycle 接近 T+degree；uniform capacity 不稳定 | 不能将一般语义拓扑收益独占归因于单环/新核 |
| [内容差距归因](CONTENT_GAP_ATTRIBUTION_RESULTS_ZH_2026-09-22.md) | anchor-X、anchor-text、联合 anchor 未稳定晋升；固定 C0 topology 后 k-center 内容并非最终普遍更好 | 不能把均值 X 判为唯一根因；图文共同来源值得保持，但未证明能解决所有退化 |
| [R2](TGC_R2_OPTIMIZATION_RESULTS_ZH_2026-09-22.md) | quota/core/radius 扫描未通过；同 teacher 的 gate/no-preserve 12/12 阶段相同 | 半径 0.05 不是全新修复；先确认保护项真的改变了优化 |
| [R2B](TGC_R2B_STUDENT_REVISION_RESULTS_ZH_2026-09-22.md) | k1 最终 +0.459 pp、guarded 仅 +0.060；旧 loss hinge 768 次均未激活 | 应测约束激活率，而不是仅在公式中放一个 preservation 项 |
| [R2B2](TGC_R2B2_RESPONSE_ALIGNMENT_RESULTS_ZH_2026-09-22.md) | w64-k1 开发结果最终 +0.85 pp、相对 random +1.52，但 gate 全拒绝 | 候选曾有积极信号，应通过独立确认而不是事后选配置 |
| [R3](TGC_R3_SELECTOR_CONFIRMATION_RESULTS_ZH_2026-09-22.md) | 独立条件 candidate -0.64、curve -0.50 pp；teacher 最终 +0.60、s1 -0.64 | 当前同时存在候选质量不足与选择器失配；不是只差一个 gate |

综合判断不是“简单方法有理论最优性”，而是：**C0 是目前可用的低成本强初始化，复杂修正通常优化了一个尚未证明与完整训练效用一致的代理量。** 还存在支持图分布、可编辑变量和实验阶段的错位，值得做针对性的因果诊断。

## 3. R3 的两个可复算诊断

### 3.1 即使完美选择，这一批候选的观察均值上限也不高

从 [cells.csv](../../results/tgc_r3_selector_confirmation_20260922/cells.csv) 读取 `stage=2, arm=candidate`：

| 条件 | 相对 C0 的最终差值，pp |
|---|---:|
| Cora/order19221 | +1.4910 |
| Cora/order19222 | -0.7667 |
| Citeseer/order19221 | -1.2652 |
| Citeseer/order19222 | -2.0147 |
| 四个 cell 等权平均 | -0.6389 |

假定事后知道每个 cell 的六种子均值，并在 C0/candidate 之间完美二选一：

\[
\Delta_{\mathrm{hindsight}}=
\frac{1}{4}\sum_{c=1}^{4}\max(0,\Delta_c)
=0.372759\ \mathrm{pp}.
\]

该值仍低于原来 +0.5 pp 的平均收益目标。因此，**对这组冻结候选和已观察 cell 均值，仅改 selector 不可能达到该目标。**

边界：这是 report 事后分析，不是可部署 selector；不是总体期望或未来顺序的性能上界；不能按每个 report seed 偷选一次再称为算法。四个 cell 的候选配对 95% t 区间都跨零，不能由此断言其真实期望增益必为负。

### 3.2 teacher 确实更少采到旧行，但原因尚未闭环

R3 `_teacher_fit` 在 `old synthetic + current raw` 上均匀无放回抽取 batch 64。正式 compact student 在合成 memory 上均匀抽取 batch 16。按各 [manifest](../../results/tgc_r3_selector_confirmation_20260922/) 的实际 rows 计算：

| 条件 | 旧 synthetic 行 | 新 raw 行 | teacher 旧 anchor 占比 | batch64 期望旧 anchor 数 | compact memory 旧行占比 | 旧行修改 quota |
|---|---:|---:|---:|---:|---:|---:|
| Cora/19221/s1 | 16 | 437 | 3.53% | 2.26 | 42.11% | 1 |
| Cora/19221/s2 | 38 | 1119 | 3.28% | 2.10 | 40.43% | 3 |
| Cora/19222/s1 | 19 | 437 | 4.17% | 2.67 | 46.34% | 1 |
| Cora/19222/s2 | 41 | 1068 | 3.70% | 2.37 | 43.62% | 4 |
| Citeseer/19221/s1 | 43 | 801 | 5.09% | 3.26 | 51.81% | 4 |
| Citeseer/19221/s2 | 83 | 554 | 13.03% | 8.34 | 74.77% | 8 |
| Citeseer/19222/s1 | 40 | 645 | 5.84% | 3.74 | 55.56% | 4 |
| Citeseer/19222/s2 | 72 | 778 | 8.47% | 5.42 | 64.86% | 7 |

这说明支持分布不同，不说明 mass replay 必然正确。表中是对比损失 anchor 的采样比例，**不是旧节点的总梯度贡献比例**：GCN 每次仍在完整支持图上传播，被抽中的新行也会通过邻居依赖旧行。

上一版用当前累计 rows 的 10% 举例得到“1、3、9”，不符合 R3 实际只允许修改上一阶段旧 rows 的定义。上表给出正确 quota。

## 4. 上一版 MB-LRR 的具体不足与修正

### 4.1 mass 表示来源数量，不表示质量

一个 synthetic row 的 `mass` 是它凝聚的来源数，不是其语义可靠度、覆盖完整度或 student utility。

- 直接按 mass 加权，更接近源节点频率分布，但可能削弱小类；当前主指标 Macro-F1 对类等权。
- canonical student 对 synthetic rows 等权，配额已经隐含了类规模与 dispersion 的折中；teacher 再用 mass 可能改变学习目标，而非修复同一目标。
- 在同一 InfoNCE batch 中复制同一旧行，会把相同图文对互相当作负例。重复次数越多，不意味着获得了越多独立历史证据。
- 固定 old/new 为 1:1 或 1:2 也不是由当前协议推导出的最佳比例。

**修正：** 默认不做 mass replay。把 teacher 的采样策略作为独立诊断因素：uniform-support 对比 compact-row-ratio。后者使用 `K_old/(K_old+K_new)`，仅匹配旧新两侧的比例，不匹配侧内全部来源分布，也不声称恢复原始历史分布。所有 batch 内 source ID 必须唯一；若做 mass ablation，使用跨 batch 加权无放回采样或明示 query loss 权重，不复制节点充当不同负例。

### 4.2 支持图的传播偏差不能靠采样解决

代码 `_concat_support` 调用 `append_topology(..., cross_degree=1)`。用于 raw delta 时，连接单元是每一个 raw 新节点；最终 compact 图连接的却是每一个 synthetic 新节点。

于是同一旧节点可能接收到远多于最终 memory 的新邻居。对称归一化 GCN 为：

\[
\widehat A=D^{-1/2}(A+I)D^{-1/2}.
\]

即使旧旧边不变，新跨边增加也会改变旧节点度数，进而改变旧旧归一化权重。改变 mini-batch 的 old/new 比例不会改变这个全图传播算子。

**修正：** 分开检查 sampling 与 support graph context。先不修改正式 student 的图算子，也不默认删除跨边。第 7 节给出一个可验证的“按新簇分摊跨边权重”诊断，不将它未经实验写入主方法。

### 4.3 原局部 basis 不完全有定义，也可能重复已失败的插值

上一版 `x_text - x_i` 只有在两者位于同一个有明确映射的坐标空间时才成立；维数相等不代表可相减。GCN 输入 X 与冻结 MiniLM embedding 通常不是同一空间。

`old_prototype - x_i` 也有问题：若旧 prototype 就是 C0 初始化的簇均值，则方向为零；若需要原始历史簇成员、协方差或重算 prototype，则违反不回读历史 raw 的约束，或需要新增存储。

向当前 delta 的亲和方向移动历史 X，和 P4/v5/C0-M 的几何刷新有明显重叠。限制到两个方向可以减小搜索空间，但不能证明方向对最终 student 有用。更小的 0.05 半径也已在 R2 类扫描中出现。

**修正：** 最小版不引入该 basis，不加“靠近新数据就更好”的奖励。使用真实 student 训练后的关系响应梯度；X 与文本在模型输出的共同空间比较，不直接相减输入向量。

### 4.4 只改旧 X，把更可信的新数据监督放在了次要位置

R3 固定新 block，只编辑少量旧行。可是当前 raw delta 是仍然完整可用的证据，历史仅剩有损 memory。原设计实际上用更强的新数据证据指导更不确定的旧节点移动，而不优化本次新凝聚块。

**修正：** 以 new-only 为主候选，old-only 为旧方向对照，joint 为条件实验。允许有限旧 revision，不把“永久不可改旧节点”当任务定义。若 new-only 胜出、joint 没有增益，应删去旧 revision 的贡献主张，而不是为了方法名称强行保留。

注意：项目已有优化新 X 的 ASM/TAGSAM 适配。因此“改新 X”本身也不是新算法，要在同 teacher、同 inner optimizer、同 horizon 下比较历史关系约束是否有新增价值。

### 4.5 不能再次用相同的短训量证明新候选有效

更严格的短训 gate 可以让结果更接近 C0，却未必得到更好的凝聚图。R3 已显示 mixed loss 改善和最终 F1 的方向可以相反。

**修正：** 第一轮不做候选性能 gate，直接报告固定 candidate、C0、matched-random。结构不合法、非有限值、超出硬约束的回退属于工程安全检查，不是“预测 F1 改善”的 selector。只有候选本身可靠后，再讨论是否值得学习选择器。

### 4.6 原来的失败决策树过度推断

“mass-balanced 不如 uniform”只能否定本次平衡方式；不能说明主要问题就是 support，也不能推出应放弃所有 TGC。低维方向失败不证明所有 X revision 无效。

**修正：** 每个停止条件对应一个被检验的假设；目标无益、梯度无效、长程不迁移、历史约束不可靠分别归档，不把局部反例升级为整个方向的否定。

## 5. 外部 TAG 凝聚方法提供了什么依据

以下为截至本轮核对的直接相关资料，不是穷尽文献综述。

| 方法 | 实际机制 | 可以借鉴 | 不能直接搬用或声称 |
|---|---|---|---|
| [TAGSAM](https://arxiv.org/abs/2606.03839)，[作者代码](https://github.com/SundayVHan/TAGSAM) | chunk 级信息覆盖/冗余选择；按文本来源构造 X；固定文本，双层优化 synthetic X，匹配 teacher/student 在相同真实参考样本上的图文相似响应 | 将学习目标放在关系而非单个 embedding 坐标；通过 student 训练路径优化可重训 memory | 静态可读完整源图；其五路 zero-shot accuracy 与本项目 all-prompt Macro-F1 不可直接比；关系匹配不是我们的独创 |
| [STAD](https://arxiv.org/abs/2607.20477)，[作者代码](https://github.com/laiyurui/STAD-Semi-Supervised-Text-Attributed-Graph-Distillation) | 半监督 GCN/MLP 协同、自训练/伪标签；任务相关聚类，共享 sketch 得到 X 和 A；关键词/LLM 文本 | 同一 assignment 联动特征、文本来源和结构；不能把所有问题归于均值 | 使用标签/伪标签和监督任务目标；原生分数不能当当前共同 student 的直接优势 |
| [DANCE](https://arxiv.org/abs/2601.16519) | 联邦 TAG 场景，选择真实核心节点，模型条件文本证据选择，稀疏结构及周期 refresh | 当前模型相关的证据选择，刷新依据随模型变化 | 其核心节点是实际源节点；本地完整图仍可读；不能直接称为 synthetic-only、无历史 raw 的持续凝聚解法；本轮未验证作者实现 |
| [CaT](https://arxiv.org/abs/2309.09455) | 持续图学习中先凝聚新数据，再在压缩记忆上更新模型，处理新旧表示规模不匹配 | 先区分 raw 新数据与压缩旧记忆造成的训练失衡 | 不是当前 TAG 图文、每阶段 scratch student 协议；不能由此推出按 mass 重放是正确解 |

### 5.1 TAGSAM 与本项目旧 ASM 有何关系

TAGSAM 讨论了 InfoNCE 的 batch 依赖和梯度波动，转向 teacher/student 图文相似关系匹配。若图和文本的共同表示同时作正交旋转 Q，则：

\[
(GQ)(TQ)^\top=GT^\top.
\]

这解释了相似矩阵对共同坐标旋转的不敏感，但不是下游 F1 保证；温度、参考样本、负例集合和 teacher 质量仍会影响结果。

项目 [现有适配代码](../../code/v2_llm_fusion/run_tagsam_graphclip_comparison.py) 已经：固定旧 prefix、优化新 X；从头初始化 inner student；用当前 raw 参考最多 256 个节点做响应匹配。其 inner 是 10 步 SGD 类更新，并存在学习 inner rate 的分支。静态适配另见 [run_tagsam_graphclip_static.py](../../code/v2_llm_fusion/run_tagsam_graphclip_static.py)。

所以本文候选真正需要检验的差异是：

1. 是否必须匹配 canonical Adam/batch 与更完整的候选自身轨迹；
2. 是否需要合法混合参考中的新旧竞争，而非只匹配当前 delta；
3. 对旧端响应施加可测的历史约束是否有新增价值；
4. 允许有限旧修订是否比 new-only 更好。

只把 ASM 改名为 TGC-R4，不构成新方法。

### 5.2 STAD 说明的不是“平均特征必然不行”

STAD 的共享 sketch 仍包含簇内加权均值式凝聚。其优势来源包括任务相关 assignment、协同学习及图文构造的一致性，不是简单避开 mean-X。

本项目 [外部基线报告](EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md) 中，同一 STAD memory 在当前 GraphCLIP 风格 student 与原生风格监督 GCN 上表现差异很大：前者最终约 33.99/28.91，后者约 73.63/67.46。该结果提示目标/接口适配的重要性，不代表两个下游模型处于同一公平主表，更不能写成“STAD 凝聚很差”。

### 5.3 不能通过更换 student 掩盖凝聚问题

本轮最终 student 继续从头训练，同一个 GCN、MiniLM、text projection、InfoNCE、batch 和步数。teacher 可以是构造工具，但不得把 teacher 参数偷偷作为最终 student 初始化。若另做监督分类或官方 GraphCLIP checkpoint，必须单列参考。

## 6. 修正版：新块优先、历史关系约束的凝聚

下面是待实现方案的精确定义，暂称 **TGC-R4 关系响应候选**。不另起一个已经暗示验证成功的版本名。

### 6.1 输入、输出和可访问数据

阶段 t 的合法输入：

\[
M_{t-1}=(X_O,T_O,A_{OO},\mathrm{metadata}),\qquad
\Delta_t=(X_D,T_D,A_D).
\]

其中 O 为旧 synthetic 行，D 为当前合法 raw delta。metadata 可包含 mass、stage、source 构造统计，但不包含可回读历史 raw 图的缓存。

初始化仍为：

\[
B_t=\operatorname{Append}_{C0}
\left(M_{t-1},\operatorname{Construct}_{C0}(\Delta_t)\right).
\]

- rows 使用现行累计 5% 配额与 dispersion 分配，不恢复固定每类 15 行。
- 新 X 初始为 synthetic cluster mean，不挑一个真实节点冒充合成节点。
- 文本沿用共同 concat/re-encode 和既定 token 预算。
- 正式图沿用共同 C0 topology；此轮不同时替换为 T+degree。
- row order、文本 embedding、mass、birth stage 和边在候选比较中固定。

首轮 s0 统一直接输出 C0，各臂共用同一个初始 memory；从 s1 才启用本文学习步骤。这样先隔离增量凝聚问题，不把初始静态凝聚变化混入历史效果。未来若优化 s0，必须作为单独协议给所有相关控制共同使用。

协议仍是 class-incremental，C0 类块/配额构造使用协议提供的类别信息，因此不能宣称整个流水线完全无标签。**teacher、内层 student、外层响应目标和候选选择不接收节点分类标签、report/test 标签或类别 prompt 标签配对。**

构造期 teacher、probe 和当前 assignment 可以临时存在。输出仅为合成 memory 及计入容量的少量元数据；阶段结束后不允许下一阶段重新访问本阶段 raw 支持集或 teacher checkpoint。研究归档可以保留，但执行器访问必须受限并记录。

### 6.2 可优化变量：首先新块，旧块是待验证扩展

\[
X_S= X_{B_t}+\Delta X_N+\Delta X_O.
\]

N 为新 synthetic 行，O 为旧 synthetic 行。

| 臂 | 可优化范围 | 目的 |
|---|---|---|
| new-only | 全部新 synthetic X；旧 X 固定 | 直接利用仍可读的新 raw 数据改善本次凝聚 |
| old-only | 继承 R3 的固定旧 mask/quota；新 X 固定 | 与既有 revision 路线作因果对照 |
| joint | new-only 加同一旧 mask | 检查旧修订是否在新块优化之外有额外价值 |

旧 mask 首轮可复用 R3 的 80% core/10% quota 以减少变量，但这是**比较控制**，不是用户任务要求，更不是已证最优的永久保护规则。第一轮不再用一个新 utility selector 同时改变 mask。

new-only 与 old-only 的自由变量数量不同；它们回答“应该把能力用在哪里”，不是严格排除维度数影响的实验。若出现明显差异，补一个可编辑标量数匹配的 new-only 子集控制，再归因。

### 6.3 teacher、固定 probe 和未修订参照

建立合法支持图 U：旧 synthetic memory 与当前 raw delta。支持图上下文在一组实验内冻结。teacher 只在 U 上用图文训练，不读取历史 raw。

从 U 定义固定参考索引 Q，其中旧 synthetic 行和当前 raw 行分开记录；所有被比较模型在**同一个 U、同一 Q、相同文本集合**上计算响应。

- teacher 目标：冻结 teacher 在 Q 上的图文关系；是否有用先做第 9 节诊断。
- 未修订参照：对于每个 construction seed，将 student 在 B_t 上训练到相同 h，再在 U/Q 上产生旧 query 的响应。它不依赖历史 raw，也不需要永久存 teacher。
- 该参照是“本阶段未修订追加图的响应”，不是历史真实数据最优模型，也不保证 C0 本身没有遗忘。
- train/validation 的 query ID 和 initialization seed 分开；若共用 U 的邻域和文本，仍只能称 query/seed 分离，不能称独立图数据。
- 一个 outer step 内候选、teacher 和参照使用固定 probe ID，避免 loss 改善来自换了负例。

teacher 的 batch、steps、支持采样、温度与构造 seed 必须写入新 manifest。第一轮诊断保留 R3 teacher 配置来隔离因素；切换 canonical teacher schedule 时单列配对控制，不同时宣称改善全来自新 loss。

### 6.4 内层必须真的在候选 memory 上训练

\[
\theta_{s,h}(X_S)=
\operatorname{Adam}^{h}
\left(\theta_{s,0};X_S,A_{B_t},T_{B_t},\xi_s\right).
\]

\(\xi_s\) 是预生成的 minibatch 行序列。模型结构、Adam 更新、batch16、lr0.001、文本投影与 logit scale 的处理均沿用 canonical student。

最小实现先做从初始化开始的 32 步完整可微训练，而不是固定 C0 64 步后只微分一步。32 是成本受控的诊断长度，不是“足以替代正式训练”的结论；外层先用固定 16 次更新，与旧路线保持规模可比。

后续只有在该候选有信号时，才测试更长的 128 步等固定 horizon。32/128 与正式 608--1776 步之间仍有偏差，必须最终从头完整训练检验，不能用 short response 自证。

若显存需要截断：候选自身的前缀必须随候选重算，Adam 一二阶状态、step、RNG、行序列保持一致；detach 后是明确的截断 hypergradient。不能继续使用固定 C0 prefix，却称已对齐候选从头训练。梯度 checkpoint/recompute 可以保留精确梯度，但必须测试 RNG 和状态一致性。

### 6.5 外层匹配关系，不直接最小化 frozen teacher 的输入 loss

在共同 Q 上定义：

\[
R_\theta(i,j)=
\frac{\langle \bar g_\theta(U,i),\bar f_\theta(T_j)\rangle}{\tau_{ref}}.
\]

\(\bar g,\bar f\) 是归一化图/文本输出。\(\tau_{ref}\) 为比较双方共同冻结的参考温度，不能让候选靠单独调温度获得虚假的关系改善；实际取值须在运行协议中固定。正式 student 自己的训练温度机制不因此改变。

在 Q 的同一个候选集合上分别作 row/column softmax。定义 \(D_G(P,Q)\) 为 query 集合 G 上两个方向的平均 KL 散度；两个方向都按 query 数取均值，避免新 raw 数量直接支配外层损失。

新端目标：

\[
L_{new}=\mathbb E_s
D_{N_Q}\left(P_T,P_{\theta_{s,h}(X_S)}\right).
\]

其中 \(N_Q\) 是 probe 中当前 raw 的 query 集合，softmax 的候选集合包含旧、新两侧。这样包含新到旧的竞争，而不只看各自 batch 内的对角匹配。

旧端约束：

\[
D_{old}=\mathbb E_s
D_{O_Q}\left(P_{\theta^B_{s,h}},P_{\theta_{s,h}(X_S)}\right)
\le \epsilon_{old}.
\]

\(\theta^B\) 来自未修订 B_t。旧 query 的归一化候选也包含两侧，因此其交叉概率质量不能在 old-old 子矩阵条件归一化时被掩盖。还应分开记录 old-old 排序、old-new 概率质量和新旧对角 alignment，诊断是哪种关系改变。

外层问题为：

\[
\min_{\Delta X_N,\Delta X_O}L_{new}
\quad\text{s.t.}\quad
D_{old}\le\epsilon_{old},\quad
\text{编辑 mask、位移与累计预算约束}.
\]

这里没有把不同 teacher 的矩阵块拼成一个声称自洽的全局概率矩阵；新目标与旧约束可能冲突，必须报告其实际可行性。初始候选等于 B_t，配对 seed 下旧散度为零；优化后的共同可行性仍不能保证。

实现可采用非负乘子的 primal-dual 优化和 X 投影。\(\epsilon_{old}\)、dual 更新步长和 X 范数上限必须事前冻结，并记录约束激活、乘子轨迹和违反比例；不能训练后按 report F1 选 tolerance。先用 construction-only 的 C0 seed 响应波动检查散度数值尺度，这只是校准启发，不是历史风险标定。无约束臂必须保留；若有无约束完全相同，不宣称 preservation 有贡献。

梯度路径必须为：

```text
synthetic X
    -> student 在 synthetic memory 上的多步训练
    -> 训练后 student 在固定合法 U/Q 上的关系
    -> 新端蒸馏目标和旧端约束
    -> hypergradient 回到 synthetic X
```

U/Q 中的旧 X 必须是修改前的 detached 副本，不能随优化变量一起改变；否则可绕过训练，通过改动评价输入降低 loss。teacher 目标、文本、参照模型全部 stop-gradient。

KL 不是被断言优于 TAGSAM wBCE 的新发现。必须有同参考、同 inner schedule 的现有 ASM/TAGSAM-response 控制，并明确它是适配而非完整官方复现。

### 6.6 有限旧修订与累计漂移

每阶段约束 \(\|\Delta x_i\|/\|x_i\|\le r\) 并不控制长链累计位移。若进入 recursive joint 实验，可在节点出生时保存两个标量：

\[
n_i=\max(\|x_i^{birth}\|,\varepsilon),\qquad \ell_i=0,
\]

每次修订更新：

\[
\ell_i\leftarrow\ell_i+
\|x_i^{new}-x_i^{old}\|/n_i.
\]

约束 \(\ell_i\le r_{total}\) 可由三角不等式得到 X 相对出生点的欧氏位移界，不需要保存完整出生特征。两个 float32 需额外 8 bytes/row，须记账；初始化优化完成后才定义该节点 birth state。

这个界不约束重新训练后的真实旧类 F1，更不控制所有邻居的传播影响。永久 core、旧 X 不动、old loss 不增均不能单独称为风险证书。

### 6.7 阶段伪代码

```text
Input: previous synthetic memory M, current legal delta D
       frozen C0 config, construction seeds, fixed training/probe plans

B, current_assignment = C0_construct_and_append(M, D)
if M is empty:
    return B  # first-stage control, no R4 optimization
U, Q_train, Q_val = make_detached_legal_reference(M, D, B)
teacher = fit_teacher(U, teacher_config)
teacher_targets = freeze_relations(teacher, U, Q_train)
baseline_targets = train_on_B_then_freeze_old_relations(B, U, Q_train)

X = clone(B.X)
for fixed outer_step:
    for construction_seed:
        theta = exact_differentiable_canonical_train(X, B.A, B.T, h)
        response = evaluate_on_fixed_reference(theta, U, Q_train)
    update X using new-relation objective and old-relation constraint
    project editable rows to predeclared bounds
    record constraint activity, gradient, drift and timing

candidate = B with X replaced
audit unchanged text/rows/edges/order + legal access + editing limits
seal candidate and no-label diagnostics before report evaluation
discard online access to raw D, U, teacher and transient assignment
Output: candidate synthetic memory, bounded metadata and audit hashes
```

第一轮 shared-parent 诊断不把 candidate 传到下一阶段；真正递归时必须改为上一阶段实际输出的 M，并另做累计风险检验。最终模型始终从头训练，不继承 teacher/inner student。

## 7. 一个值得单独验证的支持图修正

目的不是再加一个复杂拓扑模块，而是排除“把 raw 新节点数量当作 synthetic 新节点数量”造成的旧度数膨胀。

令新 raw 到 C0 新 synthetic 簇的 assignment 为 \(Q\in\{0,1\}^{n_D\times K_N}\)，每个 raw 行恰属一个非空簇；\(C=\operatorname{diag}(Q^\top\mathbf1)\)。令未归一化 compact 图的旧新权重块为 \(W_{ON}\)。构造 teacher 支持图跨边：

\[
W_{OD}=W_{ON}C^{-1}Q^\top.
\]

即把一个 compact 跨边的总权重均分给其新簇成员。于是：

\[
W_{OD}\mathbf1=W_{ON}\mathbf1.
\]

在旧旧权重块与自环规则相同、采用同一对称归一化时，旧节点总度一致，**支持图与 compact 图的旧旧归一化块一致**。这是该构造可单测的代数性质，不是性能定理。

限制也很明确：

- 新 raw 内部仍使用当前合法真实边，其新端度数和两跳传播不可能与 compact 图完全相同。
- 跨边是合成参考连接，不是发现了旧 raw 与新 raw 的真实边。
- assignment 只来自当前 delta，阶段结束不需要保留原始成员。
- 当前 `norm_adj` 是按 edge_index 的无权实现；不能只保存 `edge_weight` 却让它被忽略。应新增构造期参考归一化 helper，先验证 unit-weight 与原实现等价，再测加权公式；不改正式 student 的共同算子。
- 这和 replay 的 mass 权重不同：这里用当前簇大小分摊已确定的跨边权重，不按历史 mass 改训练频率。

首个诊断用 2x2：uniform-support / compact-row-ratio，分别搭配原 raw-append 支持图 / 上述加权支持图。teacher seed、训练步数、batch 和所有可评价输入保持共同。不能将两因素同时变化的收益全部归于“平衡”。

只有该诊断支持其作用，才把它纳入后续主候选配置；失败只否定这一参考图构造，不否定 teacher 或全部 revision。

## 8. 实现位置、必要检查与理论边界

### 8.1 与现有代码的连接

| 当前文件 | 可复用部分 | 必须新增或避免的行为 |
|---|---|---|
| [tgc_revision_gate.py](../../code/v2_llm_fusion/tgc_revision_gate.py) | 合法 delta、clone/hash、结构审计 | `_concat_support` 的上下文显式化；不再默认 teacher 输入梯度就是最终效用梯度 |
| [tgc_student_revision.py](../../code/v2_llm_fusion/tgc_student_revision.py) | functional GCN、InfoNCE、可微 Adam、数值核对 | 扩展 new/old mask；不能沿用“只许旧行改变”的 invariant 到 new-only |
| [tgc_response_alignment.py](../../code/v2_llm_fusion/tgc_response_alignment.py) | optimizer/RNG/prefix 状态审计 | 固定 C0 prefix 只留旧对照；不能误当候选完整响应 |
| [tgc_selector_confirmation.py](../../code/v2_llm_fusion/tgc_selector_confirmation.py) | seed 分离、封存、精确结果复用 | 第一轮不启动 selector 网格，不用 report 决定接受 |
| [run_student_compatible_append.py](../../code/v2_llm_fusion/run_student_compatible_append.py) | C0 构造、append、旧 topology 保持 | 当前 assignment 仅作为可选构造期返回值，不增加永久 raw 依赖 |
| [run_tagsam_graphclip_comparison.py](../../code/v2_llm_fusion/run_tagsam_graphclip_comparison.py) | 响应匹配适配与官方 loss 校验 | 补同 inner schedule 的 matched ASM 对照；保留原冻结实验代码 |
| [run_gate1_continual.py](../../code/graphclip_native/run_gate1_continual.py) | 共同 student/归一化语义 | 正式训练不为本候选单独改 batch、模型或 weighted operator |

新功能应落在独立 R4 模块/runner，由共享 helper 支持；不得覆盖 R3 代码或输出目录，让已有证据不可重现。

### 8.2 新实验之前必须通过的测试

1. **内层等价：** 同 seed、Adam 状态、batch sequence，functional unroll 与共同 student 在 1/4/32 步数值一致。
2. **hypergradient：** 微型图有限差分方向检查；stop-gradient 必须产生严格 no-op；所有未授权行梯度为零。
3. **固定评价输入：** 候选 X 改变不改变 U/Q、teacher target 或 baseline target 的 hash。
4. **模态空间：** 不允许未映射的 X 与 MiniLM 输入 embedding 直接相减；关系矩阵行列对应一致。
5. **旧约束：** 初始 D_old 接近零；人工违例会触发约束；记录真实训练中是否激活。
6. **sampler：** batch 内无重复 ID；有不足时明确截断/重分配；实际 old/new 比例及负例数入日志。
7. **支持图：** 若测试加权分摊，验证对称、稀疏、无多加自环、old-degree 等式和旧旧归一化块等式。
8. **访问边界：** 构造函数没有历史 raw/report 标签入参；dataset loader 和文件读取也按 allowlist 审计，不能只检查函数签名。
9. **输出边界：** rows、token、text、row order、edges 不变；X 是真实优化的 synthetic 值；状态重载与指标独立重算。
10. **成本边界：** teacher、baseline reference、unroll、outer、临时支持物化、峰值显存与持久 metadata 分别记账。

### 8.3 有理由，不等于有下游保证

InfoNCE 对一个 query 的梯度依赖 \(p_{ij}-\mathbb1[i=j]\)，其中 \(p\) 由当前负例集合、温度与采样组成决定。相同 memory 在不同 batch/horizon 下的梯度不同，解释了代理错位的可能性，但并不证明关系蒸馏一定解决它。

有限 memory 对真实历史数据的表示是有损的。旧 synthetic probe 上的关系保持，最多说明该 probe 上的行为被约束；没有额外代表性假设，就不能推出整个历史分布的风险界。

若能直接保证真实节点对所有类别 prompt 的 logit 扰动小于 epsilon，且原分类 margin 大于 2 epsilon，则对应 argmax 不变。但目前 X 范数界、synthetic KL 界均没有建立到该条件的有效上界。因此论文不能将其称为“无遗忘保证”。

关系蒸馏仍是为图文 student 优化的任务相关目标。增加 LP 等任务验证之前，不应宣称 task-agnostic condensation。

## 9. 建议执行顺序：先检验候选，再检验算法

以下全部为待执行任务。参数起点用于讨论，不是结果后才可以改的“预注册”；运行前必须另存 protocol 和配置 hash。

### R4-0：现有证据复算与冻结

本轮已完成：R3 二选一事后上限、旧新采样比例、关键实现与论文机制核对。

待做：封存新 runner 配置、construction/validation/report seeds 和全新确认顺序；核查顺序实际类分组而非只换 order 名。R3 四个已暴露条件只作为开发诊断。

### R4-1：teacher 目标和支持上下文诊断

- 先完成第 7 节 2x2 诊断，不修改最终 memory。
- 分别记录 teacher 在合法 old/new/mixed probe 的 alignment、relation、检索表现和实际传播量。
- 可单列 teacher 对真实报告节点的分类作为开发诊断，不把它当凝聚结果，不把 report 标签接入构造或在线选择。
- 再测试同架构学生是否能在合法 U 上直接学到冻结 teacher 关系。该项是蒸馏接口 sanity，不是压缩实验，不能混入 memory 性能主表。
- 若 teacher 目标不稳定或直接蒸馏都无信号，先修目标/teacher；不是宣布整个 TGC 不可能。
- 支持图/采样条件先按预声明的诊断规则冻结；不能看 report F1 后逐 cell 选择最优 teacher，再称为一个无标签算法。

### R4-2：最小候选矩阵

固定一个 teacher/support 条件、相同旧端约束和 32-step scratch unroll，首先比较：

| 外层新端目标 | new-only | old-only |
|---|---|---|
| 现有实例级 InfoNCE probe 目标 | 必做 | 必做 |
| teacher 关系目标 | 必做，优先候选 | 必做，revision 对照 |

共同控制：C0；关系 new-only 的 stop-gradient 与逐行等幅 random；同 teacher、同 Adam/horizon 的 delta-only ASM-response。R3 frozen-prefix arm 仅作为路径诊断，不把不同候选空间与不同训练路径的全部差异归因于一个因素。

关系 new-only 有信号后，依次增加：

1. 去掉旧端约束，检查约束是否有效、是否只是额外正则；
2. 加入同 mask 的旧编辑，检查 joint 是否优于 new-only；
3. 原真实 teacher 关系对比打乱对应/未训练 teacher，检查增益是否只是任意扰动；
4. 更长候选自身 horizon，检查短程到完整训练的迁移；
5. 必要时可编辑参数量匹配控制。

不在第一轮同时扫描文本、预算、topology、mass、radius、quota 和 selector。matched-random 需匹配修改行、逐行位移范数、输出预算，不只匹配全局 RMS。

第一轮直接跑共同正式 student 得出 candidate 质量，不因无标签 validation 指标稍差就把全部 candidate 回退为 C0。validation 用于诊断过拟合；若最终需要 checkpoint selection，规则必须预先固定并单列成本，不能由 report F1 选 outer checkpoint。

### R4-3：独立确认与递归

先固定一个候选，不组合逐 cell 最优模块。再在 Cora/Citeseer 新顺序、独立 construction/report seed group 上做完整训练；report 保持至少六个 seeds、共同 C0 与 k-center、同预算 ASM 控制。

可沿用之前的工程推进标准作为讨论起点：最终均值目标 +0.5 pp、至少 3/4 cell 在 C0-1 pp 以内、最差不低于 -2 pp、old 最差下降不超过 2 pp、new 平均不下降、至少不劣于 matched-random。**必须同时报告配对区间、阶段平均与新旧切片，不能把门槛通过说成统计显著或普遍优于。**

只有 shared-parent 确认有信号，才测递归输出：前一阶段真实 candidate 进入下一阶段，记录累计编辑和传播变化。随后再扩 1%/5%/10%、3-stage/base+4；先小图验收，再决定大图与 LP。

### R4-4：成本与贡献决定

原文“构造不超过 C0 的 3 倍”暂撤回。C0 已计时构造仅约 0.193 秒，单看倍数会丢失绝对成本；但也不能因此忽略昂贵优化。

必须报告：

- 端到端构造秒数与纯优化秒数，冷/热启动和缓存条件；
- 构造加一个共同 student 的真实总成本；
- teacher、参照模型、候选优化、任何 selector 的分项成本；
- 持久 memory bytes、峰值 CPU/GPU memory 和临时支持图大小；
- 如讨论重复部署，明确部署次数，给出实际摊销曲线，不把六个实验 report seeds 冒充六次真实部署。

总成本上限须在正式确认前冻结，不能见到新结果再放宽。第一轮先做 correctness/smoke 计时确定可行预算，不运行无限网格。

## 10. 何时能形成完整算法贡献

一个有说服力的完整算法可以是：

```text
类规模/离散度配额
    -> 共享来源簇的 synthetic X + 压缩文本初始化
    -> C0 合成 topology 和 append
    -> 合法当前 teacher 与未修订历史参照
    -> 经共同 student 训练路径的关系响应凝聚
    -> 必要时有界旧修订
    -> 输出同预算可重训 synthetic TAG
```

预算、均值初始化、文本重编码使用成熟构件是合理的。要作为算法贡献，需要证明中间学习步骤确实改善 memory 的可重训效用，而非仅降低 teacher loss。

可争取的贡献与证据要求：

| 潜在贡献 | 需要什么证据 | 当前状态 |
|---|---|---|
| 历史 raw 不可回读下的 teacher/参考构造 | 合法访问审计；sampling/context 因果控制；不能靠保存完整旧模型/图作弊 | 旧路径有访问审计；本文新设计未运行 |
| 针对 scratch student 的新旧关系凝聚 | 比同协议 ASM、InfoNCE、C0、random 更好；学习目标和训练轨迹消融 | 未验证 |
| 有限旧 revision 的必要性 | joint 稳定优于 new-only，递归不累积退化 | 未验证，不能预先列为成立贡献 |
| 支持图跨边尺度一致性 | 数学等式、算子单测、与采样分开的实验收益 | 本文给出待验证构造；只有代数性质可先讨论 |
| 多任务可重训 memory | 同一 memory 在 NC/LP 等共同协议有效；非逐任务单独选最优图 | T+degree 有局部先例，不等于 R4 已有证据 |

若只有 new-only 有效，论文应如实写“历史约束下的增量凝聚”，而不是强讲 revision。若只有某个大图/任务有效，可以限定适用场景，不必要求每个 cell 都 SOTA；但需要强共同基线、稳定的平均收益和机制证据，不能只靠挑出一张好表。

若新方案不优于同预算 ASM，它很可能只是蒸馏方法的持续适配，还不足以支撑强算法新颖性。若候选有收益但成本显著增加，应定位效用/存储权衡，而非计算高效。当前不能承诺该路线能达到 WWW 录用要求。

## 11. 最终建议与本轮交付

1. 保留“teacher/student 学习合成 memory”的方向，但不按原 MB-LRR 配方直接执行。
2. 优先改候选优化对象和目标：新块优先、共同 student 轨迹、合法关系目标；旧修订需要独立证明必要性。
3. 把 teacher 支持图的传播尺度作为一个明确可测问题，和采样比例分开。
4. 不先引入 mass replay、亲和几何 basis、文本/拓扑联调或另一个短程性能 gate。
5. R4 正式 `20260923c` 已完成：支持图 2x2 诊断、完整候选矩阵、六 seed report、重载审计与成本记录均通过；`relation_new` 未通过晋升门槛，`relation_old` 仅保留为不稳定描述性信号。详见 [TGC-R4 正式结果](TGC_R4_RELATIONAL_RESULTS_ZH_2026-09-23.md)。

本轮文档验收：R3 差值与八个阶段采样比例已从 CSV/manifest 复算；R4 `20260923c` 的 792 个槽位、504 次实际训练和四个 audit 已复算；本文、结果报告、旧方案和索引的本地链接均可解析。第 7 节公式另做的双精度小型矩阵算例误差约 `2.22e-16`；正式 lifted support 在 8 个增量 cell 的旧块误差最高约 `5.96e-7`。这证明结构构造正确，但不构成下游性能保证。

**一句话：TGC 值得继续，但需要证明“训练过后的 student 确实更容易从这张合成图学到有用关系”，而不是继续给旧 X 加一个看起来合理的移动规则。**
