# T+degree 之后的研究推进方案

日期：2026-09-21  
状态：研究路线设计文档，不代表下列候选方案已经完成或得到验证。

执行更新（2026-09-21）：[首批拓扑因果拆分](TOPOLOGY_CAUSAL_ABLATION_RESULTS_ZH_2026-09-21.md)和[语义拓扑强控制](SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md)均已完成。前者复用 2,160 个旧 fit；后者新增 1,440 个 fit、80 张 memory，并完成两数据集独立重载审计。补充检查已确认全部 80 张 memory 的跨边策略、40 张 local-cycle memory 的等度序列，并记录块内算子微基准及存储大小；不是端到端成本验收。BookChild 的 local-cycle 五阶段平均 NC/LP 优于 C0，arXiv 的收益主要出现在部分最终或 LP 条件；uniform capacity 方向不稳定，单环必要性也未被证明。两批都沿用上一轮顺序/seed，不能视为新独立确认；P0/P1/CTAG-1 仍不能整体标为完成，具体剩余任务见第 15 节。

## 1. 先给出结论

当前不应把论文主线表述为“提出了一个新的预算、文本编码或特征合成算法”。这些部分目前主要沿用已经验证过的简单构造：按压缩率增长的 memory、类内分配、簇均值 X、中心文本拼接与冻结 MiniLM 重编码。继续在这些组件上叠加模块，较难形成清晰的新颖性。

当前最值得保留的是 **T+degree**：在节点内容、memory 大小和边数不变的条件下，重新设计合成图的追加拓扑，使新块具有受控连通性，并限制新节点对旧块的连接负载。它已经在 BookChild 的节点分类和链路预测上带来一致收益，在 arXiv 上接近持平但存在退化和顺序依赖。因此它是目前最有实验依据的研究锚点，但还不是已经充分证明的文献级算法贡献。

本路线的目标不是证明“环一定优于 C0”，而是回答三个更有价值的问题：

1. 持续图凝聚中，合成拓扑是否会独立影响下游 student 的可重训效用？
2. T+degree 的收益究竟来自连通性、度负载均衡、文本语义边，还是 C0 的拓扑缺陷？
3. 在不强制历史行和历史边永久冻结的情况下，哪一种记忆管理策略在压缩率、历史保持、新类塑性和计算成本之间更合理？

只有这三个问题得到回答，T+degree 才有资格从“一个有效的工程构造”升级为论文主方法。

## 2. 需要立即修正的问题定义

### 2.1 旧行和旧边不可修改不是问题的强制约束

此前实验使用了 append-only、旧合成行保留、旧旧边不改写的条件，但这是当前协议选择，不是用户已经规定的不可违反约束，也不能直接写成持续凝聚问题的定义。

正式论文应区分三种访问条件：

| 条件 | 原始历史数据 | 旧合成行 | 旧旧边 | 研究含义 |
|---|---|---|---|---|
| Append-only | 不可回读 | 保留 | 保留 | 最低访问成本、最严格的追加条件 |
| Row-refresh | 不可回读 | 可重新生成或合并 | 原则上保留 | 研究内容更新与拓扑更新的分离 |
| Memory-rewire | 不可回读 | 保留或少量更新 | 允许受限重连 | 研究拓扑维护成本与效用的折中 |
| Global re-condensation | 可以访问全部历史原始数据 | 可全部改写 | 可全部改写 | 非持续上界/强参考，不属于同一访问条件 |

T+degree 目前只验证了第一行。后续不应继续把 append-only 当作唯一合法形式，而应把它写成一个明确的低访问成本 setting，并用 Row-refresh 或 Memory-rewire 作为放宽约束的对照。这样论文贡献来自“在不同访问预算下如何维护可重训合成记忆”，而不是来自人为规定旧边不能动。

### 2.2 类别增量协议不是唯一的新颖性来源

项目已经使用过 Cora/Citeseer 的多阶段协议以及 arXiv、BookChild 的 base+4 五阶段协议。base+4 应保留为主协议之一，但不能声称“提出 base+4 就是创新”。后续需区分：

- 类别增量：用于与持续图凝聚工作对齐；
- 到达规模不均衡：用于检验预算和跨块连接负载；
- 时间/事件流：用于说明 Web 场景中的数据持续到达；
- 不同访问条件：用于证明方法不是依赖旧边冻结这一人为设定。

主论文不需要把所有协议都塞进主表，但至少应有一个标准 base+4 结果、一个不同类别顺序、一个非均匀到达或时间顺序确认。

## 3. 当前贡献的客观分级

| 候选贡献 | 当前证据 | 当前定位 | 是否足够单独支撑 WWW 长论文 |
|---|---|---|---|
| 持续 TAG 合成问题与访问协议 | 已有多版本协议和审计 | 问题定义/基准基础 | 不够，需要真实场景动机和强基线 |
| T+degree 拓扑追加 | BookChild 正收益，arXiv 近持平 | 最有潜力的算法机制 | 目前不够，需要因果拆分和泛化 |
| 旧块负载与归一化传播变化分析 | 有明确机制假设，尚未完整测量 | 理论/机制解释候选 | 目前不够，需要可验证指标 |
| 跨节点分类与链路预测复用 memory | 已完成初步确认 | 跨任务有效性证据 | 是支持性贡献，不是单独算法创新 |
| 新 budget、文本 encoder、response selector | 多数简单版本未稳定晋升 | 负结果和边界 | 不应继续堆叠为主线 |
| 审计、重载、成本账 | 证据较完整 | 可复现性和工程质量 | 不能替代算法新颖性 |

因此，当前更准确的论文主张是：

> 在持续 TAG 合成中，合成节点内容相同并不意味着合成图对下游 student 等价；追加拓扑和旧块负载会影响可重训效用。我们提出并分析一种低成本的拓扑负载均衡追加算子，并明确其收益边界。

这比声称“提出全新的图凝聚框架”更可信，也比单纯报告一个新协议更有科学问题。

## 4. 总体路线

后续分为五个阶段。阶段之间有晋升门，不满足前一阶段证据时不进入下一阶段的复杂设计。

### P0：重构问题和统一比较底座

**目标：** 把 append-only 从强制问题约束改成一个可比较的访问条件，避免后面所有结论建立在误读协议之上。

**任务：**

1. 在 runner 中显式增加 `access_mode`：`append_only`、`row_refresh`、`memory_rewire`、`global_recondense`。
2. 固定 C0 和 T+degree 的节点内容、rows、mass、文本和总边数；只改变拓扑或记忆管理条件。
3. 统一报告 NC、LP、构造时间、student 训练时间、推理时间、memory 大小和峰值显存。
4. 记录每阶段 old/new/cross/within-new 指标，以及孤立点、度分布、度 Gini、最大度和归一化传播漂移。
5. 把 arXiv/BookChild 的 base+4 作为标准主协议；不把它作为新颖性本身。

**P0 产出：** 一张访问条件主表、一张成本表、一张协议卡。  
**P0 通过条件：** 所有方法使用同一 student、同一 rows/edge budget、同一 order/seed 组，且能够从 artifact 重新计算主要汇总指标。

### P1：T+degree 的因果拆分

**目标：** 判断 T+degree 的收益是否来自两个独立部件，而不是把整个组合当作黑箱。

固定 X、文本、rows、总边数、stage、row order 和 student，只改变端点。至少运行下面 2×2：

| | C0 跨块连接 | capacity-aware 跨块连接 |
|---|---|---|
| C0 块内连接 | A：C0 | B：只替换跨块分配 |
| T+degree 块内连接 | C：只替换块内连边 | D：完整 T+degree |

再加入三个必要控制：

- degree-preserving random：匹配 T+degree 的度序列，但随机打乱端点；
- random ring/path：保留连通环形式，但不使用文本相似度；
- semantic kNN/medoid：在相同边数和最大度限制下使用常规语义邻居。

**测量：**

- 拓扑：孤立节点率、最大度、度 Gini、跨块边比例、块内连通分量；
- 传播：旧节点归一化邻接块的 Frobenius/row-L1 漂移、旧节点聚合权重变化、谱半径/谱差异；
- 任务：NC old/new/all F1、LP old/new/cross/within-new AUC/AP；
- 成本：构造和 student 训练总时间。

**P1 晋升条件：** T+degree 或其一个明确子组件在至少两个数据集、两个顺序或两类任务上优于 C0，并且不被 degree-preserving random、random ring 和语义 kNN 全部复现。若所有拓扑都接近，则主贡献应转为“拓扑对持续 TAG 的影响诊断”，不再宣称特定环构造是核心算法。

### P2：把 T+degree 推广为一般的拓扑追加算子

**目标：** 从“环 + ceil(new/old) 容量”升级为可解释、可调节的 `Topology-Aware Append`，但只在 P1 找到明确因果来源后实施。

候选形式不是再加一个深度模型，而是将新增边选择写成受约束优化：

\[
\min_{E_t^{new}}
 -\lambda_s\,\mathrm{SemSim}(E_t^{new})
 +\lambda_l\,\mathrm{Load}(E_t^{new})
 +\lambda_p\,\mathrm{PropDrift}(E_t^{new})
 +\lambda_c\,\mathrm{CoveragePenalty}(E_t^{new})
\]

约束包括新增边数、连通性或最小度、每个旧节点的接收容量，以及最大新增负载。T+degree 是其中一个固定、低成本的特例；C0 是无负载约束的语义选边特例。

优先验证两个简化版本：

1. **degree-aware capacity：** 容量不再统一使用 `ceil(new_rows / old_rows)`，而根据旧节点当前度、mass 和新旧相似度分配，控制相对负载 `b_i/(d_i+1)`。
2. **soft closure：** 不强制每个新块闭成环；当闭合边语义相似度过低时，允许保持路径、增加一条低权重边或改用匹配边，并记录连通性代价。

不要同时加入 response selector、attention、动态总预算和可训练 kernel。该阶段只回答拓扑负载和闭合策略两个问题。

### P3：访问条件与任务泛化

**目标：** 证明方法解决的是持续合成记忆维护问题，而不是只适用于 append-only 分类。

最小实验矩阵：

| 轴 | 设置 |
|---|---|
| 访问条件 | append-only、row-refresh、memory-rewire、global re-condensation |
| 数据规模 | 1%、5%、10% memory rate |
| 协议 | base+4、另一类别顺序、非均匀阶段或时间顺序 |
| 数据集 | arXiv、BookChild、SportsFit；至少两个正式确认，第三个作外部确认 |
| 下游任务 | 节点分类、链路预测；若资源允许增加文本-节点检索 |
| 参考 | full-raw 同 student、k-center/medoid、静态 TAG 凝聚适配、全历史重凝聚 |

这里的“任务无关”只能分级声明：

- 两个任务都有效：跨任务可重训效用；
- 三个任务都有效：多任务合成记忆；
- 还需在不同 student 或模型族上有效，才可接近模型无关；
- 不能仅凭 NC+LP 两个模型就称 task-agnostic 或 model-agnostic。

### P4：理论和论文证据闭合

**目标：** 提供机制分析，而不是声称拓扑约束可以理论上证明 F1/AUC 提升。

建议推导旧块归一化传播扰动。令旧节点原度为 \(d_i\)，新增跨块边数为 \(b_i\)，加入自环后的归一化邻接为 \(\hat A\)。可以分析：

\[
\frac{\left|\hat A^{new}_{ij}-\hat A^{old}_{ij}\right|}
     {\max(\hat A^{old}_{ij},\epsilon)}
\quad\text{以及}\quad
\frac{b_i}{d_i+1}.
\]

目标是证明：在总新增边数固定时，控制最大相对负载或其均衡程度，可以限制旧块传播系数的最大变化；同时分析新块内部连通性对孤立节点和信息传播路径的影响。该结论只能支持“传播扰动受控”，不能直接推出下游 F1/AUC 上升。

P4 需要补充：

1. 旧节点相对负载分布与 old-class retention 的相关性；
2. 传播漂移与 LP cross/within-new 的相关性；
3. 低相似闭合边与 arXiv 退化的关系；
4. 复杂度：构造时间是否随新增 rows、候选边数和旧 memory 规模可控。

## 5. 论文主线的三种可能结局

### 结局 A：T+degree 通过因果和泛化门

论文主方法命名为 **Topology-Aware Append**，T+degree 作为简单、可复现的实例。主贡献写成：

1. 定义历史不可回读下的持续 TAG 合成记忆维护问题，并明确多种访问条件；
2. 发现合成节点内容固定时，追加拓扑仍会改变 student 的可重训效用；
3. 提出受限拓扑追加算子，联合语义相似性、连通性和新增负载控制；
4. 在节点分类和链路预测上验证收益、成本和适用边界。

这是最有希望的 WWW 长论文路线。

### 结局 B：T+degree 本身不普适，但机制成立

如果 T+degree 只在部分数据集有效，而负载控制、传播漂移与任务差异存在稳定关系，则论文写成：

> 一个关于持续 TAG 合成中拓扑作用和负载扰动的机制研究。

此时不要宣称统一最优算法，而是以机制、边界和可复现实验为核心。主方法保留为一个分析性构造。

### 结局 C：T+degree 不能超过强控制

如果它在因果拆分后被随机度匹配、kNN 或 medoid 全部复现，则不应继续包装 T+degree。可以保留：

- 持续 TAG 的统一基准和协议；
- 简单拓扑修复的负结果；
- 不同任务和访问条件下的系统分析。

这可能形成一篇较完整的 benchmark/analysis paper，但作为 WWW 长论文主方法的风险明显更高，应考虑进一步寻找更强的、由机制诊断驱动的算子。

## 6. 强制停止和晋升规则

### 6.1 可以晋升为主方法的条件

至少满足：

- 在至少三个数据集或两个数据集加一个独立协议上，主任务平均不低于 C0，并有至少两个条件取得有意义正收益；
- 相对于 degree-preserving random、random ring、semantic kNN/medoid 等强结构控制，仍能解释额外收益；
- 新增构造成本不超过 C0 的约 2 倍，student 训练成本基本不增加；
- full-raw retention、旧类保持和新类塑性没有系统性崩溃；
- 访问条件、标签使用、transductive 推理和 test 使用边界在论文中完全透明。

### 6.2 必须停止继续堆模块的情况

出现下列任一情况时，不再加入 attention、response、dynamic budget 等模块：

- 收益只存在于一个数据集或一个类别顺序；
- 随机度匹配控制达到同样效果；
- 高收益主要来自新增边或更多 rows，而非端点选择；
- 方法只改善 LP，但 zero-training X/text cosine 已经解释了大部分结果；
- 额外成本超过收益，且无法在大图上扩展；
- 访问条件变化后收益消失。

## 7. 近期执行顺序

### 第一批：必须先做

1. 统一 `access_mode` 和成本账；
2. 完成 P1 的 2×2 拆分与三个拓扑控制；
3. 计算旧块归一化传播漂移；
4. 在新协议中补 full-raw、k-center/medoid、全历史重凝聚的匹配结果；
5. 重新确定 arXiv 退化来自闭合边、跨块负载，还是 student/LP 训练目标。

### 第二批：P1 有明确信号后再做

1. degree-aware capacity；
2. soft closure；
3. row-refresh 与 memory-rewire；
4. 1%/5%/10% rate 曲线和非均匀到达顺序；
5. SportsFit 或另一个未用于选择的确认数据集。

### 第三批：论文证据和写作

1. 锁定一个主方法和一个冻结 C0；
2. 建立主表、机制表、成本表、失败边界表；
3. 将预算、文本和 response 的失败探索压缩为相关消融/附录；
4. 完成相关工作核验和理论分析；
5. 按 WWW 八页正文限制组织叙事，附录放协议、伪代码、审计和完整 seed 表。

## 8. 当前论文建议表述

目前可以写：

> We study continual text-attributed graph condensation when historical raw data cannot be revisited and the condensed memory must support retraining downstream graph-text models. Our current evidence suggests that, even with identical synthetic node content and memory size, the topology used to append new synthetic blocks changes downstream utility. We therefore investigate topology-aware append operators that control connectivity and incremental load, and characterize when they help or fail across node classification and link prediction.

目前不应写：

- “旧合成节点和旧边不可修改是问题本身的必然约束”；
- “T+degree 普遍优于 C0 或完整图”；
- “方法是无监督的”，因为构造使用类别标签分配预算和分块；
- “方法与任务或模型无关”，当前只有两个下游任务和两个 student 族；
- “提出全新的预算、文本或特征合成方法”；
- “已经达到 SOTA”或“已经证明适合所有持续 TAG”。

## 9. 最终决策

现在可以开始论文写作，但写作和实验应并行：

- 立即写问题定义、T+degree 算法、协议边界、最新结果和失败探索；
- 暂不锁定摘要中的强创新表述；
- 先完成 P0/P1，再决定 T+degree 是否升级为 `Topology-Aware Append` 主方法；
- 如果 P1 只证明拓扑本身重要而未证明 T+degree 特有优势，则把论文定位为机制/系统研究；
- 如果 P1 和 P3 在多个数据集、访问条件和任务上均支持负载均衡算子，再把它提升为算法贡献。

一句话决策：**继续围绕 T+degree，但围绕的是“拓扑追加机制是否成立”这一科学问题，而不是继续把当前的环和容量规则当作最终创新；同时现在就开始写论文骨架，避免在没有明确晋升证据前无限优化。**

## 10. 完整算法目标：从 T+degree 到 CTAG-Append

用户的最终目标不是只提出一个拓扑小修补，而是形成一个完整的持续 TAG 凝聚算法。为此，后续主线可以暂命名为 **CTAG-Append（Continual Text-Attributed Graph Condensation with Content-Topology Co-design）**。名称不是最终论文标题，核心是把持续凝聚拆成四个相互配合的决策：

1. **Content：** 新到数据怎样变成合成节点；
2. **Budget：** 哪些类别和子簇获得多少合成容量；
3. **Topology：** 新节点怎样和新旧 memory 连接；
4. **Maintenance：** 历史 memory 是否需要有限刷新或重连。

T+degree 是第三部分的当前实例，不是完整算法的全部。完整算法还必须有明确的 memory 状态、访问条件、预算规则和下游 student 评价接口。

### 10.1 Memory 状态

在阶段 t 保存：

\[
M_t=(X_t^{syn},T_t^{syn},E_t^{syn},m_t,s_t,q_t),
\]

其中：

- \(X_t^{syn}\)：合成节点属性；
- \(T_t^{syn}\)：合成文本或冻结文本表示；
- \(E_t^{syn}\)：合成拓扑；
- \(m_t\)：每个合成节点覆盖的原始样本质量/数量；
- \(s_t\)：节点出生阶段；
- \(q_t\)：覆盖、稳定性和历史使用统计。

算法只在当前阶段访问 \(M_{t-1}\) 和新到合法数据 \(\Delta_t\)。是否允许修改旧行、旧边由 `access_mode` 决定，而不是写死在问题定义中。

## 11. CTAG-Append 的完整流程

### Step 1：新阶段信息分析

对当前新数据按类别和类内子簇计算：

- 样本规模 \(n_{t,c,k}\)；
- 属性离散度和文本离散度；
- 类内多模态程度；
- 与已有合成 memory 的语义新颖度；
- 当前阶段对旧类/新类覆盖的缺口。

第一版只使用可复现的统计量，不使用大型生成模型。新颖度可以由当前文本/属性空间中到旧 memory 原型的距离估计；若后续要加入 student response，必须作为独立候选模块验证。

### Step 2：动态但受控的预算分配

固定 5% 是可靠基线，但完整算法应允许预算随数据量和信息量变化。候选形式为：

\[
K_t=\min\left(K_{max},\;
K_{t-1}+\max\left(b_{min},
\left\lfloor\rho\Delta N_t(1+\kappa I_t)\right\rfloor\right)\right),
\]

其中 \(I_t\) 是新阶段信息量，综合新颖度、离散度和覆盖缺口。为避免方法只是依靠更多 rows，必须同时报告：

- fixed-rate；
- fixed-total；
- dynamic-rate with the same final budget；
- dynamic-rate with matched student steps。

类别/子簇预算可使用：

\[
w_{t,c,k}=n_{t,c,k}^{\alpha}
(1+\lambda d_{t,c,k})
(1+\mu u_{t,c,k})
(1+\nu h_{t,c,k}),
\]

再用最大余数法分配整数 rows。其中 \(d\) 表示离散度，\(u\) 表示新颖度，\(h\) 表示覆盖缺口。首轮实现只保留规模和离散度，新增项逐个加入，不能一次组合后无法判断来源。

### Step 3：多原型合成节点

当前 mean-X 对单峰簇有效，但对多模态簇可能把不同结构平均掉。完整算法的候选升级是 **multi-prototype synthetic node construction**：

1. 在类内先做 joint X/text 子簇；
2. 每个子簇至少生成一个中心合成节点；
3. 若子簇离散度或多模态分数超过阈值，再生成边界/方向原型；
4. 所有节点都是新的合成向量，不直接把真实节点作为凝聚图节点；
5. 用 source mass 记录每个原型覆盖的真实样本量。

一种可验证的非学习版本是低秩残差原型：

\[
x_{c,k,r}=\operatorname{Norm}
\left(\mu_{c,k}+a_rU_{c,k}\sqrt{\Lambda_{c,k}}\right),
\]

其中 \(\mu\) 是簇均值，\(U\sqrt{\Lambda}\) 是簇内主要变化方向，\(a_r\) 是固定的中心/边界系数。它比简单平均保留更多类内几何，但不会泄漏真实节点身份。是否优于 mean-X 必须用相同 rows 和相同拓扑单独验证。

### Step 4：固定预算下的文本集合构造

当前中心文本拼接/re-encode 是可靠基线，但文本利用仍较简单。完整算法可加入 **graph-conditioned text set construction**：

1. 对候选文本用冻结 MiniLM 得到句向量；
2. 用合成 X 和簇统计量作为 query；
3. 对候选文本计算相关性与多样性分数；
4. 在固定 token/字符预算下选择若干片段；
5. 拼接后重新用冻结 MiniLM 编码，保持文本空间一致。

该机制不应该直接使用可训练 attention 作为默认版本，因为小图上已有 attention 方向没有稳定晋升。首版优先使用确定性 query-key 打分和 diversity penalty，只有在固定 token 的独立实验中稳定优于 concat/mean，才能成为完整算法的一部分。

### Step 5：拓扑感知追加

这是当前 T+degree 的核心升级位置。对新块和旧 memory 的候选边，统一求解：

\[
\max_{E_t^{new}}
\sum_{(i,j)\in E_t^{new}}\operatorname{SemSim}(i,j)
-\lambda_l\sum_i\left(\frac{b_i}{d_i+1}\right)^2
-\lambda_p\operatorname{PropDrift}(E_t^{new})
\]

约束包括总边数、最小连通性、节点最大度、旧节点接收容量和可选的重连预算。

当前 T+degree 对应这个目标的低成本特例：

- 新块内部使用 cap-2 路径后闭环；
- 每个新节点获得跨块连接；
- 旧节点的本阶段接收量受容量限制。

后续更一般的版本包括：

- `degree-aware capacity`：按旧节点当前度、mass 和相对负载分配容量；
- `soft closure`：闭合边过低质量时允许路径或替代边；
- `bounded rewire`：允许只重连少量旧新边，不触碰全部历史拓扑；
- `propagation-aware edge selection`：直接惩罚旧归一化传播漂移。

这样创新点不再是“固定造一个环”，而是**在持续追加下联合控制语义连接、连通覆盖和增量传播负载**。

### Step 6：有限 memory maintenance

完整算法不应预设旧 memory 永远不可修改。定义每阶段维护预算 \(R_t\)：

\[
R_t=\gamma_xK_t+\gamma_e|E_t|.
\]

当 `append_only` 时 \(R_t=0\)；当 `row_refresh` 或 `memory_rewire` 时，只允许在预算内：

- 合并高度相似的旧合成节点；
- 用新旧合成信息更新原型统计量；
- 重连传播负载过高的少量旧节点；
- 保留 birth stage、source mass 和审计轨迹。

刷新只能读取旧 memory 和当前新数据，不能回读旧原始数据。这样可以研究“不可回读”和“不可修改”两个约束分别造成的损失，而不是把二者混为一个人工限制。

### Step 7：下游 student compatibility check

主算法的评价目标不是让合成节点看起来接近原始数据，而是让 memory 支持重新训练下游模型。因此最终选择应至少检查：

- 图文对齐分数；
- 多个 student seed 的稳定性；
- 节点分类 old/new/all F1；
- 链路预测 old/new/cross AUC/AP；
- memory size、构造时间和 student 训练时间。

早期 response selector 的排序一致性不足，不能直接作为默认核心。更稳妥的做法是先用上述统计量和拓扑漂移指标做低成本筛选，再对极少数候选进行短训验证。若短训排序仍不能预测 canonical ranking，就不把 response 选择器写进主方法。

## 12. 目标算法的创新点分层

### 第一层：必须成立的核心创新

**Topology-aware continual append：** 在历史数据不可回读的条件下，将合成内容追加和合成拓扑维护统一建模，并控制新增连接对旧块传播的相对负载。

### 第二层：可以增强论文的算法创新

**Bounded memory maintenance：** 不把旧节点/旧边永久冻结，而是引入显式有限刷新/重连预算，形成 append-only 到 global refresh 的连续访问-效用曲线。

### 第三层：可选增强，不应一开始全部加入

- 信息量驱动的动态预算；
- 多原型 X 合成；
- 图条件文本集合选择；
- 低成本 student-compatible 选择。

第三层只有在单项实验通过后才能纳入最终算法。完整算法不是模块越多越好，而是每个模块都应对应一个明确挑战：预算、类内多模态、文本冗余、拓扑负载或历史更新。

## 13. 推荐版本路线

| 版本 | 内容 | 目的 |
|---|---|---|
| CTAG-0 | 当前固定预算 + mean-X + concat/re-encode + T+degree | 冻结可复现实验锚点 |
| CTAG-1 | CTAG-0 + 2×2 拓扑因果拆分 + degree-aware capacity | 证明拓扑机制是否成立 |
| CTAG-2 | CTAG-1 + multi-prototype X | 解决类内多模态和简单平均问题 |
| CTAG-3 | CTAG-2 + bounded rewire/refresh | 解除不必要的旧边冻结，测访问-效用折中 |
| CTAG-4 | CTAG-3 + 动态预算或文本集合选择，二选一 | 形成完整主方法，不同时加入两个高风险模块 |
| CTAG-final | 通过跨数据集、跨任务和成本门后冻结 | 论文最终版本 |

当前版本号不应直接跳到 CTAG-final。每次只增加一个机制，并用新顺序/新数据集作为确认条件，避免在开发集上逐步调出一个看似优越的组合。

## 14. 最终推进原则

目标可以明确为：**提出一个完整的持续 TAG 凝聚算法，而不是只发表 T+degree。** 但完整算法的科学贡献必须来自一个统一的设计逻辑：

> 根据新阶段信息分配有限合成容量，用多原型和固定预算文本构造表示内容，再通过拓扑负载感知的追加与有限维护，使 memory 在多个阶段和多个下游任务上保持可重训性。

后续不建议重新平均地优化所有模块。优先级应是：

1. 先证明 T+degree 的拓扑机制不是 C0 缺陷修复的偶然结果；
2. 再把它推广成 degree/load-aware 的一般拓扑追加算子；
3. 然后加入 bounded refresh，解除不必要的旧边冻结假设；
4. 最后从预算、X 或文本中只选择一个经过实验证明有增益的组件加入；
5. 用 full-raw、k-center/medoid、静态 TAG 凝聚和全历史重凝聚建立完整比较。

如果这些步骤成功，论文可以提出的是一个真正完整的 **content-budget-topology-maintenance co-design** 算法；如果其中只有拓扑诊断成立，也仍然可以诚实地把它写成机制驱动的持续 TAG 凝聚方法，而不是虚构一个“所有模块都有效”的结果。

## 15. 第一批之后的执行状态

本节根据结果更新优先级，不追改第一批冻结设置。

| 路线项 | 状态 | 下一步 |
|---|---|---|
| P0：共同内容、预算、评价与审计 | 已对齐并补齐存储、块内微基准；完整构造/推理成本及显式 access_mode 尚不完整 | 下一 runner 补齐计时与 provenance，按矩阵计算 completion 数量；保留本批原始错误计数及更正收据 |
| P1：2×2 + 两随机端点控制 | 已完成 2,160 fits；BookChild 有明确条件性收益 | 保留全部正负结果，不事后替换主方法身份 |
| P1：semantic kNN/medoid 强控制 | 四臂已完成 1,440 fits；local-cycle 等度序列已确认，kNN 不等最大度；medoid 未完成；不是独立确认 | 在新顺序/新 report seeds 上复核 semantic local-cycle 与 kNN，补 medoid；保留“无统一最优”的负结果 |
| P1：传播诊断 | 已完成 row-L1、度、负载、连通分量；原列出的全部谱指标未完成 | 不把最大负载下降等同于平均漂移下降 |
| CTAG-1：degree-aware capacity | 未实现；uniform capacity 仅作控制且结果不稳定 | 先做相对负载/传播感知的跨边选择，固定边数并与 nearest 配对比较 |
| soft closure | 未实现 | 固定边预算做质量受限替代，不同时叠加 capacity/refresh |
| 新顺序确认和成本门 | 未完成；本批只完成同条件机制控制 | 登记未用类别顺序和 report seeds；补原始读取、X/文本、拓扑、写盘和 student 训练的端到端计时 |
| CTAG-2/3/4 及强基线主表 | 未完成 | 按前置机制证据分批推进，不同时堆叠 X、文本、预算和维护模块 |

接下来顺序收窄为：**新条件复核 semantic local-cycle/kNN -> 补 medoid与完整成本 -> 相对负载/传播感知跨边与 soft closure 单项开发 -> bounded rewire -> 选择一个内容侧改进 -> 完整基线比较。** T+degree 继续作为研究基础，C0、internal_only 和随机控制保留；尚不宣布 CTAG-final 或 SOTA。
