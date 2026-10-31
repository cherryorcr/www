# Semantic-Recursive：历史原始图不可回读条件下的持续文本属性图凝聚

**Semantic-Conserved Recursive Append：受约束的语义守恒递归记忆追加与可重训性分析**  
论文草稿版本：2026-09-27  
研究状态：Semantic-Recursive 已冻结为论文主方法候选；C0 保留为共同控制、消融基线和回退版本。本文不宣称 SOTA。

## 摘要

文本属性图（Text-Attributed Graph, TAG）凝聚需要同时压缩节点属性、文本语义和图结构。静态凝聚通常可以访问完整原始图并重写全部合成记忆，而持续 TAG 凝聚还必须面对历史原始图不可回读、memory 随数据到达而增长、旧类保持和新类塑性相互冲突等限制。一个几何上接近原始数据的候选节点，不一定能够被重新初始化的图文 student 有效学习；反之，一个结构和语义均不完美的合成记忆，可能因为传播位置和跨阶段状态更适合重训而获得更高下游效用。

本文提出并系统评估 **Semantic-Conserved Recursive Append**，简称 Semantic-Recursive。方法在固定压缩率下对当前增量数据进行类内局部 two-prototype split/merge，使用 Mean-X 和来源可追溯的文本重编码构造新 payload，再通过同类一对一语义匹配将完整 payload 放入当前结构 slot。每个阶段只追加当前新 block，上一阶段的 Semantic-Recursive memory 作为下一阶段 parent；历史 raw graph 不再读取，旧 synthetic prefix 和旧 topology 保持不变。方法不依赖昂贵的 student response selector，也不使用最终 F1 选择候选。

在 Cora/Citeseer 两个小图、两个类别顺序和六个 Student seeds 的冻结比较中，Semantic-Recursive 的 Final Macro-F1/Accuracy 为 **53.02/59.59**。新增 360-fit 外部比较允许各方法自行构造初始 memory 和拓扑：CaT-MMD-TAG 适配为 **54.62/59.96**、k-center 为 **54.18/59.43**、GECC-hard-TAG 适配为 **48.70/52.53**、medoid 为 **47.32/53.11**；SR 尚未稳定超过最强外部对照。此前共享 C0 初始图/拓扑的内容控制表中，SR 相对 C0 的最终 F1/Accuracy 为 **+0.925/+0.364 pp**，相对所测 coreset 均值也较高，但该结论只适用于该控制协议。在 arXiv、BookChild 和 SportsFit 的 `base+4` 实验中，相对 C0 的 12 个增量条件平均提升 **0.858 pp**，9/12 为正，新类 F1 平均提升 **1.764 pp**；独立类别顺序的总体增量平均仅约 **+0.020 pp**。这些证据支持有条件的机制收益，不能表述为顺序不变的普适优势。

本文的核心结论不是 Semantic-Recursive 已经超过所有 TAG 凝聚方法，而是：在历史 raw 不可回读的 continual TAG setting 中，守恒的 payload-slot 追加算子能够改善部分新类的可重训性，并且可以通过 budget、prefix、payload 和访问权限不变量进行审计；与此同时，动态预算、response selector、图条件文本 attention、旧 memory refresh 和拓扑重连等复杂模块存在明显的数据集或顺序依赖。该结果为持续 TAG 凝聚的算法设计和公平评估提供了一个可复现主线，也明确了当前方法的失效边界。

**关键词：** 文本属性图；持续图凝聚；合成记忆；图文对齐；递归追加；语义守恒；新类塑性；可重训性

## 1 引言

图凝聚的目标是在远小于原图的节点和边预算下，构造一份能够支持下游模型训练的合成图。对于 TAG，压缩对象不仅包括节点特征和邻接关系，还包括每个节点的文本描述、图文对齐关系和文本驱动的局部结构。现有静态方法可以在一次性访问完整原图后共同优化合成节点、文本和拓扑；这种能力不能直接迁移到持续场景。

持续 TAG 凝聚研究的问题是：数据按阶段到达后，系统只能保留合成 memory，而不能重新访问过去的完整 raw graph。每一个新阶段都需要在有限预算下加入新信息，同时尽量保持历史信息。这个过程有四个困难。

1. **历史访问限制。** 旧阶段的原始节点、文本和边不能再次参与构造，算法只能依赖上一阶段的 synthetic memory。
2. **预算增长和规模失配。** memory 应随累计数据增长，但固定总预算、固定每类行数和动态 response budget 对不同类别规模并不等价。
3. **新类塑性与历史保持冲突。** 新类需要足够的多模态和传播槽位，旧类又不能被新增 block 的传播扰动或过度压缩。
4. **student 可重训性和几何相似性不一致。** k-center、medoid 等 coreset 方法优化的是空间覆盖，不能保证合成 X、文本和 topology 共同形成适合图文 student 重新学习的训练接口。

本文围绕一个窄而可审计的 operator 展开：在不增加 memory rows、不改变 Student、不回读历史 raw 的前提下，能否把当前新 block 的完整图文 payload 放到更合适的语义传播 slot，并递归地传递到后续阶段？

本文的主要贡献如下。

1. **持续 TAG 凝聚的受限问题定义和审计协议。** 明确历史 raw 不可回读、cumulative fixed-rate memory、递归 parent、report-dev 隔离和 Student 重训边界，定义 budget、prefix、payload conservation 和 access 四类不变量。
2. **Semantic-Conserved Recursive Append。** 提出一种由局部 two-prototype structural proposal 和同类语义守恒 payload assignment 组成的持续 memory operator。assignment 阶段不复制、不删除或拆开已生成的完整 payload，而是在固定 topology 和固定 rows 下进行可追溯的 slot 重排，并将上阶段 semantic memory 递归作为下一阶段 parent。split/merge 对当前新 block 的 proposal 内容变化与 assignment 的守恒范围分开审计。
3. **跨小图和大图的分层验证。** 小图包含共享初始图/拓扑的内容控制表，以及允许自行构造的 SR、CaT-MMD-TAG、GECC-hard-TAG、k-center、medoid 外部表；full raw 和 stateful Student 单列。大图采用 arXiv、BookChild、SportsFit 的 `base+4` 协议，分别报告 continual、offline synthetic、offline raw 和 stateful Student 权限。
4. **正向机制与负结果的联合分析。** 结果显示 Semantic-Recursive 在 canonical/reference order 上有方向性收益，尤其改善部分新类 F1；独立顺序会暴露 trade-off。response selector、动态预算、图条件文本、旧 memory maintenance 和复杂拓扑模块未形成稳定主线。本文把这些结果作为算法边界而不是隐藏的调参失败。

![图 1：Semantic-Recursive 的持续访问边界、预算守恒 refinement 和递归追加流程。红色表示历史 raw 禁止回读，蓝色表示当前可访问输入或 parent，紫色表示完整 payload assignment。](assets/semantic_recursive_pipeline.png)

图 1 将方法最重要的约束画出来：当前阶段只接收新的 raw delta，上一阶段只以 synthetic parent 形式保留；历史 raw 不再回读。two-prototype refinement 和 semantic assignment 都发生在当前新增 block 内，最后才追加到 parent memory。

## 2 相关工作

### 2.1 静态图凝聚

GCond、SFGC、GCondenser、OpenGC 等静态图凝聚方法通常在一次性访问完整训练图后，利用梯度匹配、特征统计、结构约束或可学习的 synthetic node 来构造小图。它们适合回答“在固定任务和完整数据访问下，怎样生成一个高效训练图”，但不直接回答历史 raw 不可回读时如何追加 memory。

### 2.2 文本属性图和 GraphCLIP-style 对齐

TAG 方法需要把节点图结构、节点属性和文本描述映射到共同空间。GraphCLIP-style student 通过图编码器、文本编码器和图文对比目标学习这种对齐，推理时可以将真实图节点表示与类别 prompt 匹配。本文使用的是原始项目中的两层 GCN student 和冻结 MiniLM 文本编码器，而不是官方 12 层 GPS/GraphCLIP 模型。官方 checkpoint 只作为外部模型参考，不作为凝聚性能的同 student 分母。

### 2.3 TAG 凝聚与外部基线

TAGSAM 和 STAD 是本文使用的公开 TAG 方法参照。由于它们的原生 student、监督目标、图结构和数据访问权限与本文 continual InfoNCE student 不完全相同，本文将其放入静态适配层，而不把异构结果直接排序成一个主榜单。random、k-center/FPS、herding 和 k-means medoid 是不依赖模型训练的真实节点 coreset 参考，用来区分空间覆盖与 synthetic TAG condensation 的差异。

已有工作也直接涉及持续或演化图凝聚，因此本文不把“持续图凝聚尚未被研究”作为贡献。CaT（Balanced Continual Graph Learning with Graph Condensation）在每个任务上构造 condensed graph memory，并将 incoming graph 的压缩结果用于后续训练；PUMA（Efficient Continual Graph Learning for Node Classification with Graph Condensation）进一步使用伪标签、重训练、一次传播和 edge-free memory。GECC（Scalable Graph Condensation with Evolving Capabilities）把演化图凝聚转化为 propagated-feature clustering，并在后续阶段继承聚类中心；OpenGC（Graph Condensation for Open-World Graph Learning）研究时间变化和结构感知分布偏移下的凝聚泛化。这些工作确立了 continual/evolving graph condensation 的先例，但它们主要面向数值特征、监督节点分类、动态图快照或累计数据访问，并没有直接给出本文的文本 payload、图文 InfoNCE Student 和受限历史 raw 访问接口。本文的研究空白应准确表述为：**持续文本属性图凝聚在历史 raw 不可回读和图文 Student 重训条件下的 payload 构造与审计问题仍缺少统一方法和同协议证据。**

CCC（Condensation-Concatenation Framework for Dynamic Graph Continual Learning）和 CA-DGCL（Dynamic Graph Continual Learning via Condensation and Attachment）是另外两个相邻的动态图持续学习工作。CCC 将历史 condensed snapshots 送入动态 GNN，提取历史 embedding，并只在结构变化影响区域拼接历史表示；CA-DGCL 跨时间匹配 condensed nodes，构造 node-chain tensor，经 Tucker 分解生成历史节点并连接到当前图。它们说明拓扑变化和历史表示 replay 是持续图学习的重要问题，但保存/更新的是动态模型表示或生成节点，不能直接等同于本文的独立重训 TAG memory。

因此，CaT/PUMA/GECC/CCC/CA-DGCL 是最接近的持续或演化图外部参照，TAGSAM/STAD 是静态 TAG 参照；它们的原论文数字不能直接和本文的 zero-shot Macro-F1 排成一个榜单。当前已实现 `CaT-MMD-TAG adaptation` 和 `GECC-hard-TAG adaptation`，公开它们使用的标签、raw 访问、Student、拓扑和文本附着规则。PUMA/CCC/CA-DGCL 目前仅进入 related work 和 closest external 计划，不填写未经运行的同协议分数。

### 2.4 持续图学习和持续凝聚

持续图学习通常保存模型参数、回放样本或原型，以缓解灾难性遗忘。CaT 和 PUMA 是与本文持续 memory 最接近的外部谱系；GECC 则更接近无梯度、增量聚类式的演化凝聚。本文额外要求输出一份可以独立重训的 synthetic TAG memory，因此不能把保留 student 参数的 replay + delta fine-tune 视为同资源凝聚方法。该 stateful control 仍然有价值，它回答的是“如果系统同时保留模型状态，直接继续训练能达到什么水平”。

## 3 问题定义和统一协议

### 3.1 TAG 和阶段数据

将原始 TAG 写为

\[
G=(V,A,X,T,Y),
\]

其中 (V) 是节点，(A) 是边，(X) 是节点属性，(T) 是文本，(Y) 是类别标签。第 (t) 阶段到达合法增量

\[
\Delta_t=\{(x_i,t_i,y_i):i\in V_t^{new}\}.
\]

算法可以读取当前合法 delta 和上一阶段 synthetic memory (M_{t-1})，不能读取历史阶段的完整 raw nodes、raw text 或 raw edges。当前实现使用类别分块和同类 assignment，因此“历史 raw 不可回读”不等于“构造完全不使用类别信息”；类别/阶段分块权限在协议中显式披露。最终保存的 memory 记为

\[
M_t=(\widetilde X_t,\widetilde T_t,\widetilde E_t,\Pi_t),
\]

其中 (Pi_t) 记录 source mass、stage、depth 和可审计 provenance。

### 3.2 Growing memory budget

主协议使用 cumulative fixed-rate，而不是固定总行数：

\[
K_t=\lfloor \rho N_{\le t}^{pool}\rfloor,\qquad \rho=0.05.
\]

阶段新增行数为

\[
b_t=K_t-K_{t-1}.
\]

因此 memory 会随累计数据增加，同时保持约 5% 的压缩率。预算的类别内分配沿用 C0 的确定性 dispersion allocation，不使用 response-gain 动态扩容。该设计同时控制了公平性和 memory growth：如果性能提升，不能简单归因于方法保留了更多行。

### 3.3 类别顺序和 canonical

本文使用 `canonical/reference order` 表示预先冻结的参考类别到达顺序。它不是数据集固有的唯一协议，也不意味着该顺序更合理。独立 order 用于检验顺序泛化。由于 fixed-rate 下每个顺序的每阶段 raw 样本量可能不同，新增 memory rows 也会不同，因此不同 order 之间不直接比较绝对 F1；主比较在同一 dataset-order cell 内进行 paired comparison。

小图使用 Cora/Citeseer 的三阶段 class-incremental 协议和 `order19611`、`order19612` 两个类别顺序。大图使用 arXiv、BookChild、SportsFit 的 `base+4` 五阶段协议，base fraction 为 0.4。P0 独立顺序使用 seed `49013`，用于暴露跨顺序边界，不参与方法调参。

### 3.4 Student 和评价指标

所有 continual 主表使用同一个 Student：

- 两层 GCN；
- hidden/output dimension 384；
- 冻结 MiniLM 文本编码器，文本 embedding dimension 384；
- 可训练 text projection；
- bidirectional graph-text InfoNCE；
- Adam，batch size 16；
- 小图 Student seeds 150--155，大图 Student seeds 100--105；
- 每个阶段从头训练，不继承上一阶段 Student 参数；
- 大图本轮使用 2048 optimizer steps，小图使用统一 protocol_v3 student steps。

训练目标为

\[
\mathcal L=\frac{1}{2}(\mathcal L_{g\rightarrow t}+\mathcal L_{t\rightarrow g}).
\]

节点分类推理使用真实图结构和类别 prompt 的图文相似度。Macro-F1 是各类别 F1 的平均，Accuracy 是预测正确节点比例。两者同时报告，因为 Macro-F1 对小类更敏感，而 Accuracy 更受大类比例影响。本文的 F1 是节点分类指标，不是凝聚 loss，也不是监督 GCN 的训练准确率。

主报告指标包括：final Macro-F1、Accuracy、stage-average、old-class F1、new-class F1、positive-cell ratio、worst-stage、memory rows、construction time 和 Student time。

## 4 Semantic-Recursive 方法

### 4.1 方法概览

Semantic-Recursive 可以概括为：

> **C0 topology + local two-prototype structural proposal + Mean-X and source-consistent text + same-class semantic conserved payload assignment + recursive append + original Student**

它不是一个新的 GNN，也不是一个需要最终 F1 选择候选的 response selector。其算法贡献集中在 continual memory operator：固定资源下，对当前新增 block 的结构 proposal 和语义 payload placement 进行受约束的递归追加。

### 4.2 Step 0：读取合法输入

第 (t) 阶段只读取：

1. 上一阶段 Semantic-Recursive memory (M_{t-1})；
2. 当前阶段合法 raw delta (Delta_t)；
3. 当前阶段允许使用的节点属性、文本和类别信息。

历史 raw graph 不参与后续构造。当前阶段的类别分块和类别内 matching 依赖合法的类别/阶段元数据；report-dev 和 test 标签只用于最终评价，不用于 proposal、matching 或候选选择。

### 4.3 Step 1：图文联合 source space

对当前 delta 节点 (i)，冻结 MiniLM 得到文本 embedding (e_i)，将图属性归一化后构造联合 source representation：

\[
z_i=[\operatorname{Norm}(x_i);\operatorname{Norm}(e_i)].
\]

根据 C0 的类别级新增 rows 分配，每个新类别形成若干 source slots。source slot 是构造过程中的临时来源分组，不是真实节点，也不会直接作为最终凝聚图节点保存。

### 4.4 Step 2：局部 two-prototype split/merge

固定当前新 block 的 rows 数，在每个新类内做一次局部重组：

1. 计算 source slot 的图文联合离散度；
2. 找到高离散度且样本量足够的 slot，将它拆成两个子簇；
3. 找到联合中心最接近的两个低复杂度 slot，将它们合并；
4. 通过一拆一并保持当前新增 rows 不变。

形式上写为

\[
G_{c,r^\star}\rightarrow G_{c,r_1}\cup G_{c,r_2},
\]

同时将两个相邻 source pieces 合并。该步骤不是增加 memory，也不是让每个类别固定生成相同数量的 prototype；它把固定容量从简单区域转移到图文联合离散度更高的局部区域，以缓解 Mean-X 对类内多模态的过度平均。当前实现的“高离散度拆分”和“最近中心合并”仍是确定性启发式，没有 gain-based 全局最优保证。

### 4.5 Step 3：构造合成属性 X

对最终 source piece (P_j)，合成属性使用 Mean-X：

\[
\widetilde x_j=\frac{1}{|P_j|}\sum_{i\in P_j}x_i.
\]

当前主方法没有启用 P-guided X、student 反向优化 X 或 response-based X。此前的 weighted-X、moment-X、P-guided two-prototype 和 student-compatible revision 已经做过实验，但没有形成跨数据集、跨顺序的稳定收益。Mean-X 因此是当前主线中最简单、可审计的属性算子。

### 4.6 Step 4：构造合成文本

X 和文本必须来自同一个 source piece。对每个 (P_j)：

1. 按联合中心距离选取最多 5 个代表文本；
2. 使用约 320 字符的内容预算进行截断和拼接；
3. 对拼接文本重新输入冻结 MiniLM；
4. 保存原始文本来源和 re-encoded embedding。

\[
\widetilde e_j=E_{MiniLM}(\operatorname{Concat}(\operatorname{TopTexts}(P_j))).
\]

这里的 top-5 是当前已验证主方法的上限，不是所有文本处理方法的理论最优。它的优点是来源可追溯且不引入生成式 hallucination。P2 的 mean pooling、text-only attention、attribute selection、graph-conditioned selection 和 shuffled graph query 没有显示跨数据集的稳定优势，因此没有进入主方法。

### 4.7 Step 5：语义守恒 payload assignment

two-prototype proposal 生成的新 payload 不直接固定写回原 source slot，而是在同一类别内部做一对一语义匹配。对 proposal (p_i) 和目标 slot (s_j)，使用图文联合代价：

\[
C_{ij}=\left\|\operatorname{Norm}([\widetilde x_i;\widetilde e_i])-\operatorname{Norm}([x_j^{slot};e_j^{slot}])\right\|_2^2.
\]

通过 Hungarian matching 求发生重组 payload 到目标 slot 的最小代价排列：

\[
\pi^*=\arg\min_\pi\sum_i C_{i,\pi(i)}.
\]

被移动的是完整 payload，而不是单独的 X 或单独的文本。payload 包括：

- 合成 X；
- re-encoded text embedding 和原始文本来源；
- source mass；
- stage/depth metadata；
- provenance 记录。

因此输出和输入满足 payload multiset 守恒：

\[
\operatorname{Multiset}(P_t^{out})=\operatorname{Multiset}(P_t^{in}).
\]

该约束保证 assignment 本身不会通过复制优质节点、丢弃困难节点或增加隐含 memory 来制造分数提升。它也使 assignment 阶段的变化可以归因于 slot placement，而不是 payload 数量变化。split/merge 之前的 proposal 内容仍可能改变当前新增 block，必须与 assignment 的 payload 守恒分开讨论。

### 4.8 Step 6：递归 append 和固定 topology

stage 0 使用 C0 memory：

\[
M_0^{SR}=M_0^{C0}.
\]

后续阶段使用上一阶段 Semantic-Recursive memory 作为 parent：

\[
M_t^{SR}=\operatorname{Append}(M_{t-1}^{SR},B_t^{SR}).
\]

旧 synthetic prefix 的 X、文本、mass、depth、stage 和旧边保持不变，只追加当前 block。主方法直接复用同阶段 C0 的 `edge_index`：

\[
E_t^{SR}=E_t^{C0}.
\]

这意味着 Semantic-Recursive 当前不改变 edge count、degree、old-old edges 或 C0 的跨块连接规则。固定 topology 是有意的因果控制：T+degree、simple rewiring、message-conditioned edge 和 GraphGPS 已经显示明显的数据集/任务依赖，不能与 payload assignment 同时写入主方法。

### 4.9 Step 7：Student 重训

每个阶段的 synthetic memory 从头训练同一个两层 GCN Student。Student 不继承上一阶段参数，不使用节点分类交叉熵，而使用 graph-text bidirectional InfoNCE。完成训练后，在真实图结构上用类别 prompt 进行 zero-shot style matching，报告 cumulative、old 和 new Macro-F1 以及 Accuracy。

### 4.10 四类不变量

Semantic-Recursive 的实现和审计围绕四个不变量：

1. **Budget invariant：** 每个阶段 rows 等于固定 rate 预算，semantic matching 不改变总行数。
2. **Prefix invariant：** recursive 阶段的历史 synthetic prefix 与上一阶段输出 bitwise 一致。
3. **Payload invariant：** 同一类内 assignment 前后的 payload multiset 完整守恒。
4. **Access invariant：** 构造阶段不读取历史 raw；report/test 标签不参与构造或选择。类别/阶段分块是否可用由具体协议声明，不把它含混写成完全 label-free。

## 5 实验设计和基线层级

### 5.1 Continual memory-only 主表

下列方法具有相同的 Student、memory rate、batch、steps 和 report-dev 规则：

| 方法 | 类型 | 是否读取历史 raw | 是否保留 Student 状态 | 作用 |
|---|---|---:|---:|---|
| C0 | 内部共同控制 | 否 | 否 | 冻结的 dispersion + Mean-X + center text + C0 topology |
| Semantic-Recursive | 本文主方法 | 否 | 否 | two-prototype + semantic conserved payload + recursive parent |
| random real-node | coreset | 否 | 否 | 随机真实节点覆盖参照 |
| k-center / greedy FPS | coreset | 否 | 否 | 几何覆盖参照 |
| herding | coreset | 否 | 否 | 均值方向真实节点参照 |
| k-means medoid | coreset | 否 | 否 | 聚类中心真实节点参照 |

### 5.2 Offline 和 stateful 参考

| 方法 | 历史 raw | 可重写旧 memory | 说明 |
|---|---:|---:|---|
| full-history synthetic | 是 | 是 | 同构造族的离线重凝聚参考 |
| complete raw graph, equal steps | 是 | 是 | 完整 raw rows、相同 Student steps |
| complete raw graph, 8192 steps | 是 | 是 | 完整 raw rows、较长训练参考，不是理论上界 |
| Student delta-only fine-tune | 当前 delta | 不适用 | 保留上一阶段 Student 参数 |
| Student C0-replay + delta fine-tune | C0 memory + 当前 delta | 不适用 | stateful Student 上界式控制，训练成本约 2.9 倍 |

这些参考不能直接与 memory-only continual 方法混合排序。它们的作用是分离“凝聚 memory 的质量”和“允许访问更多 raw 或保留模型参数”带来的收益。

### 5.3 静态 TAG 方法和官方模型参照

TAGSAM、STAD 和官方 GraphCLIP checkpoint 单独放入静态/外部参考层。只有当公开方法使用相同 current-delta access、memory budget、Student 和指标时，才可进入严格 continual 主表。

### 5.4 持续图凝聚外部参照

CaT、PUMA 和 GECC 是本文需要正面讨论的持续/演化图凝聚参照。CaT 以 incoming graph 的分布匹配生成 condensed graph memory；PUMA 在此基础上加入伪标签、重训练、一次传播和 edge-free memory；GECC 将 propagated-feature clustering 与增量中心继承用于 evolving graph condensation。OpenGC 则研究 open-world 分布变化下的时间不变凝聚。它们说明持续图凝聚并非空白领域，但其原生 setting 主要是数值特征、监督节点分类、动态图快照或累计数据访问。

本轮已核对 CaT/GECC 官方代码并完成受限 TAG 机制适配。CaT-MMD-TAG 用每步重新初始化的随机编码器优化类条件表示均值，3000 步 Adam 更新 synthetic X，自环式 memory；GECC-hard-TAG 对传播后的特征做 20 次 hard clustering，输出传播特征中心。二者附着同类至多五来源、320 字符加分隔符的文本并冻结 MiniLM 重编码，下游统一原 GCN/InfoNCE。

这不是官方完整复现：CaT 的监督 TiM 被共同 Student 替代；GECC 原 evolving 代码使用累计 raw 初始化重聚类，本轮仅在 disjoint class-incremental 中继承旧 synthetic 中心，不读取旧 raw，也未复现 fuzzy 调优分支。k-center/medoid 则从 stage0 自行选择节点并构图，不再借用 SR/C0 的初始图或 slot。PUMA 尚未实现。详见[冻结协议](../current/EXTERNAL_CONTINUAL_ADAPTER_PROTOCOL_ZH_2026-09-27.md)。

## 6 结果

### 6.1 小图：冻结 C0 和历史强参考

在早期 canonical 小图协议中，C0 的最终结果为 Cora 49.85、Citeseer 59.17。其它参考如下：

| 方法 | Cora Final Macro-F1 | Citeseer Final Macro-F1 | 访问/训练条件 |
|---|---:|---:|---|
| C0 | 49.85 | 59.17 | 约 5% synthetic memory，同 Student |
| k-center | 46.35 | 55.34 | 真实节点 coreset |
| recluster | 49.70 | 54.24 | 历史/全量重聚类参考 |
| full-history synthetic | 47.21 | 48.28 | 离线重凝聚 |
| complete pool, 8192 steps | 51.31 | 55.60 | 完整合法 raw，长训选择 checkpoint |

该表说明完整 raw 训练不是理论上界，Student 优化路径、文本配对和 topology 都会影响结果。它也说明 C0 不是公开基线，而是后续所有内部机制实验的共同控制。

### 6.2 小图：共享初始图/拓扑的内容控制

以下为 Cora/Citeseer、两个递归类别顺序和六个配对 Student seeds 的四个 dataset-order cells 等权平均。Final F1/Accuracy 是最终阶段指标，Stage-avg 是三个阶段累计指标。coreset 臂使用共同 C0 stage0 和 C0 topology，因此本表是内容控制，不代表各 coreset 的完整独立构造性能；后者见第 6.2.1 节。

| 方法 | Final F1 | Final Acc. | Stage-avg F1 | Stage-avg Acc. | Old F1 | New F1 | 相对 C0 Final F1 | 正收益 cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 internal control | 52.09 | 59.22 | 47.25 | 47.95 | 51.95 | 51.34 | 0.00 | 0/4 |
| **Semantic-Recursive** | **53.02** | **59.59** | **48.35** | **48.73** | **52.32** | **52.76** | **+0.92** | **3/4** |
| Random real-node coreset | 48.22 | 55.74 | 46.59 | 46.68 | 48.39 | 47.34 | -3.87 | 1/4 |
| k-center / greedy FPS | 48.23 | 55.32 | 46.12 | 47.65 | 48.53 | 46.70 | -3.86 | 0/4 |
| Herding real-node coreset | 50.40 | 58.97 | **48.45** | **49.32** | 49.63 | 50.02 | -1.69 | 1/4 |
| K-means medoid coreset | 50.04 | 57.73 | 47.85 | 48.23 | 51.28 | 46.95 | -2.05 | 0/4 |
| Full raw graph scratch | 49.73 | 54.96 | 45.67 | 42.37 | 48.84 | 49.84 | -2.37 | 1/4 |
| Student delta-only fine-tune | 51.26 | 57.57 | 45.59 | 44.92 | 53.57 | 46.72 | -0.83 | 2/4 |
| Student C0-replay + delta fine-tune | 53.03 | 58.76 | 47.19 | 46.03 | **54.73** | 49.33 | **+0.94** | 2/4 |

Semantic-Recursive 相对 C0 的最终 F1 平均提升 0.925 pp，Accuracy 提升 0.364 pp。相对 random、k-center/FPS、herding 和 medoid 的均值差异在此控制协议下均为正。旧汇总合并四个 cell 的 seed pairs 所得 CI 不能当作 24 个独立顺序的泛化区间；新版外部表按固定条件和组级分别报告。逐 cell 结果如下，格式为 `Macro-F1 / Accuracy`：

| Dataset/order | C0 | Semantic-Recursive | Random | k-center/FPS | Herding | Medoid |
|---|---:|---:|---:|---:|---:|---:|
| Citeseer/order19611 | 56.62/62.43 | **57.70/62.33** | 50.67/57.00 | 51.63/56.84 | 49.86/55.38 | 54.05/61.55 |
| Citeseer/order19612 | 54.54/61.91 | 54.01/61.44 | **55.96/59.98** | 54.31/60.24 | 51.76/59.77 | 54.07/62.12 |
| Cora/order19611 | 49.51/55.23 | **50.21/56.40** | 43.28/51.72 | 45.10/49.88 | 46.10/53.94 | 46.62/53.26 |
| Cora/order19612 | 47.70/57.32 | **50.14/58.18** | 42.96/54.24 | 41.89/54.31 | **53.89/66.79** | 45.41/54.00 |

Student C0-replay + delta fine-tune 的最终 F1 与 Semantic-Recursive 接近，但它保留跨阶段模型参数、继续访问当前 raw delta，训练成本约为 memory-only 的 2.9 倍，因此只能作为 stateful control。

### 6.2.1 小图：外部方法独立构造的主要比较

固定 SR 输出，所有外部臂从 stage0 构造自己的 memory；5% fixed-rate、同条件 rows/Student steps、六个配对 seeds。两图各两个既有顺序、三阶段，共 360 次 Student fit，全部 checkpoint 重载误差为 0；SR 与上一轮逐 seed F1/Accuracy 完全一致。此矩阵不是新独立顺序确认。

| 方法 | Final F1 | Final Accuracy | Stage-avg F1 | Stage-avg Accuracy | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Semantic-Recursive** | 53.02 | 59.59 | 48.35 | 48.73 | 52.32 | 52.76 |
| CaT-MMD-TAG 适配 | **54.62** | **59.96** | 50.52 | 47.87 | 53.01 | **55.95** |
| GECC-hard-TAG 适配 | 48.70 | 52.53 | 44.43 | 41.19 | 46.61 | 50.95 |
| k-center 独立构图 | 54.18 | 59.43 | **53.43** | **54.58** | **53.03** | 54.84 |
| medoid 独立构图 | 47.32 | 53.11 | 43.43 | 41.79 | 46.98 | 46.74 |

![图：两图两顺序外部基线主要结果](assets/external_continual_20260927_means.png)

| Dataset/order | SR F1 / Acc. | CaT F1 / Acc. | GECC F1 / Acc. | k-center F1 / Acc. | medoid F1 / Acc. |
|---|---:|---:|---:|---:|---:|
| Cora/19611 | 50.21 / 56.40 | 55.64 / 60.76 | 47.78 / 53.32 | 54.54 / 59.78 | 46.39 / 51.78 |
| Cora/19612 | 50.14 / 58.18 | 50.98 / 60.33 | 45.58 / 50.43 | 50.60 / 54.06 | 46.58 / 55.23 |
| Citeseer/19611 | 57.70 / 62.33 | 56.93 / 59.87 | 48.72 / 50.57 | 51.72 / 57.52 | 47.33 / 51.88 |
| Citeseer/19612 | 54.01 / 61.44 | 54.93 / 58.88 | 52.71 / 55.80 | 59.87 / 66.35 | 48.96 / 53.55 |

![图：SR 减去外部基线的逐条件 F1 差值](assets/external_continual_20260927_cell_deltas.png)

SR 相对 CaT、GECC、k-center、medoid 的 Final F1 差分别为 -1.60、+4.32、-1.17、+5.70 pp，正条件比例分别为 1/4、4/4、1/4、4/4。固定 Cora/19611 条件下，SR 对 CaT 差 -5.43 pp，Student-seed 配对 95% CI 为 [-7.56,-3.30]；Citeseer/19612 对 k-center 差 -5.87 pp，CI [-9.72,-2.02]。它们反映固定 memory 下的种子波动，不是未知顺序泛化保证。两张图共享数据，四条件组级区间亦只作描述。

四条件的最终存储 rows 分别为 Cora 94、Citeseer 111；每枚 seed 的三阶段 Student 时间为 SR 7.85 秒、CaT 7.77 秒、GECC 7.78 秒、k-center 7.84 秒、medoid 7.83 秒。SR 此次复用封存产物，构造时间为 N/A；其余三阶段构造时间为 14.87、0.41、0.03、0.17 秒，均不含共享数据/编码器初始化。coreset 原文 token 存储约为 SR 的 2.8/3.4 倍，当前等 rows 不等于等文本预算；CaT 与 SR 文本规模接近。导出 Student memory 大小不包含构造 shadow/workspace，不能据此声称总计算或持久状态最省。

完整标准差、逐条件 CI、预算轨迹和来源见[外部适配结项报告](../current/EXTERNAL_CONTINUAL_ADAPTER_COMPLETION_ZH_2026-09-27.md)。这批证据需要收紧“优于强基线”的结论，但没有改变 SR 的已有分数。第 6.2 节 k-center 的低分不能代替本节独立构造的强结果，也不能将两表差异全部归因于拓扑，因为 stage0 同时改变。

### 6.3 大图：`base+4` 五层参考

大图统一使用 5% fixed-rate、batch 16、Student seeds 100--105、2048 steps。下表为最终阶段 Macro-F1。`full-history synthetic` 和 `complete raw` 允许回读累计 raw，因此是 offline reference。

| 方法 | arXiv | BookChild | SportsFit |
|---|---:|---:|---:|
| C0 continual append | 18.817 | 12.607 | 25.013 |
| **Semantic-Recursive append** | **19.227** | **13.313** | **28.076** |
| Random real-node coreset | 18.169 | 13.136 | 26.756 |
| K-center coreset | 16.667 | 11.712 | 27.840 |
| Real-node FPS coreset | 18.058 | 13.480 | 26.147 |
| Full-history synthetic re-condensation | 20.544 | 12.407 | 26.467 |
| Full-history synthetic, GPU implementation | 19.692 | 14.425 | 28.357 |
| Complete raw graph, equal steps | 18.842 | 18.018 | 26.769 |
| Complete raw graph, 8192 steps | 20.155 | 17.217 | 29.502 |

Semantic-Recursive 相对 C0 的增量阶段汇总为：

| 数据集 | cumulative 平均变化 | old-class 平均变化 | new-class 平均变化 | 正阶段 | 最差阶段 |
|---|---:|---:|---:|---:|---:|
| arXiv | +0.637 | -0.157 | **+2.971** | 3/4 | -0.421 |
| BookChild | +0.203 | -0.361 | **+2.241** | 3/4 | -1.552 |
| SportsFit | **+1.735** | **+2.158** | +0.080 | 3/4 | -2.381 |
| **合计** | **+0.858** | +0.547 | **+1.764** | **9/12** | -2.381 |

该结果支持两个结论。第一，收益不是由更多 rows、更多 edges、更多 Student steps 或不同 Student seeds 造成，两个 arm 使用成对相同的资源。第二，收益不是无条件稳定：arXiv/BookChild 的正向作用主要体现在 new-class plasticity，SportsFit 的作用更偏向 old-class retention，而负阶段仍然存在。

### 6.4 P0：独立类别顺序和 construction seed

P0 不改变主方法，只检查 canonical 结果是否依赖 construction seed，以及独立类别顺序是否仍然得到正收益。三个大图使用 construction seeds 9001/9002/9003，五阶段 memory hash 完全一致，说明构造过程 bitwise deterministic。独立 order seed 为 49013，结果如下：

| 数据集 | 增量 cumulative 平均 | old-class 平均 | new-class 平均 | 正阶段 | 最差阶段 |
|---|---:|---:|---:|---:|---:|
| arXiv | -0.373 | +0.693 | -2.385 | 1/4 | -0.914 |
| BookChild | +0.402 | -1.139 | +1.327 | 2/4 | -0.760 |
| SportsFit | +0.030 | -0.769 | +1.440 | 2/4 | -2.434 |
| **合计** | **+0.020** | - | - | **5/12** | -2.434 |

因此，本文不把 canonical 的 +0.858 pp 宣称为顺序不变的平均收益。P0 的正确结论是：Semantic-Recursive 的构造是可复现的，但其下游收益依赖当前 delta 的类别组成、固定率新增预算和 C0 topology。

![图 2：Semantic-Recursive 的小图统一结果、大图 canonical 增益、独立顺序边界和 old/new trade-off。所有增量均为同一 dataset-order cell 内相对 C0 的 paired delta。](assets/semantic_recursive_result_summary.png)

图 2 直观显示当前证据的两面性：canonical order 上三类大图均有正的增量平均，且 new-class 在 arXiv 和 BookChild 上提升较明显；但 independent order 的 arXiv 为负、总体接近零，说明当前算子仍依赖到达顺序。

### 6.5 静态 TAG 和外部模型参照

这些结果不进入严格 continual memory-only 主表，只用于说明不同任务和访问条件下的性能位置。

| 方法 | Cora Final F1 | Citeseer Final F1 | 协议边界 |
|---|---:|---:|---|
| TAGSAM-ASM self-loop | 47.73 | 51.09 | 静态适配，self-loop topology |
| TAGSAM-ASM semantic | 52.86 | 42.00 | 静态适配，semantic topology |
| STAD-local-canonical | 33.99 | 28.91 | 本地适配，监督式原生方法不等同当前 Student |
| 官方 GraphCLIP checkpoint | 58.98 | 57.76 | 官方 GPS checkpoint，完整图上下文，纯推理 |

官方 checkpoint 的 Accuracy 为 Cora 67.53%、Citeseer 64.58%。它使用官方 12 层 GPS、完整原始图上下文和冻结预训练权重，不能用来单独归因凝聚算法。它的作用是外部模型参考，不是 Semantic-Recursive 的性能保留率分母。

### 6.6 已完成的负结果和机制消融

下表汇总了进入主方法前做过的主要探索。负结果并不表示对应算子在任何条件下都无效，而是表示在当前统一协议下没有通过跨数据集、跨顺序的晋升门禁。

| 方向 | 代表设置 | 结果/现象 | 对主方法的影响 |
|---|---|---|---|
| 预算 | fixed-rate vs adaptive-rate | Cora 49.16、Citeseer 54.90，方向翻转 | 保留固定压缩率，暂不让 response 决定总预算 |
| response extra budget | response-gain extra | Cora 46.00、Citeseer 53.43 | 选择误差和额外 rows 混淆，未晋升 |
| 图条件文本 | center concat / graph selection / shuffled graph | Cora 48.46/51.51/52.32；Citeseer 54.97/53.12/53.94 | 保留简单 center concat，避免把不稳定 attention 写入主线 |
| student response selector | short-horizon、teacher proxy、guarded gate | ranking 不能稳定预测 full-step F1，成本高 | response 作为诊断，不作为选择器 |
| 拓扑 T+degree | BookChild 分类约 +1.546；arXiv 分类约 -0.662 | 任务和数据集依赖 | 主方法固定 C0 topology，先隔离 payload 作用 |
| 历史 memory maintenance | bounded refresh、C0-M、relation maintenance | C0-M final 约 -0.34 pp，new-class 约 -0.93 pp | 不启用旧 memory refresh |
| semantic permutation | 递归小图平均 +1.640，6/8 正 | 有机制信号但仍有顺序边界 | 发展为守恒 payload assignment |
| role-aware matching | 独立确认平均约 +1.468，最差约 -2.042 | 复杂 role cost 不能稳定预测 Student utility | 不加入 role/message/plasticity 权重 |
| GraphGPS student | stagewise calibration | 没有形成足够的凝聚收益 | 固定原始两层 GCN Student，避免改变任务 |

这些实验共同导出当前主方法的“最小充分版本”：保留简单可靠的 X、文本和 Student，只在固定资源和固定 topology 下引入可审计的 payload-slot 语义守恒，并递归传递该状态。

### 6.6.1 文本处理的组件级负结果

下表固定合成行数、C0 topology、文本预算和 Student 训练协议；每一格为小图最终阶段 Macro-F1。文本预算固定为 64 tokens，不能通过增加文本长度换取结果。这里的目标不是证明某个文本算子永远无效，而是判断它能否作为 Semantic-Recursive 的默认组件。

| 文本算子 | Cora | Citeseer | 结论 |
|---|---:|---:|---|
| Center concat，当前默认 | 50.64 | 56.03 | 稳定、可追溯，保留 |
| Mean pooling | 53.06 | 57.34 | 首轮较高，但独立 seed 下 Cora 阶段均值下降 |
| Distance-weighted pooling | 53.28 | 56.42 | 单图收益不稳定，跨数据集不一致 |
| Set attention | 52.95 | 58.40 | Citeseer 有利，Cora 独立确认下降 |
| Dual prototype | 50.46 | 55.04 | 没有超过 center concat |
| No-fuse | 41.77 | 54.52 | 直接去掉文本融合明显损害 Cora |
| No-text | 4.30 | 5.49 | 文本是当前 zero-shot Student 的必要输入 |

独立 construction seed 的 DeepSets/set encoder 结果为 Cora `51.89 / 51.31 / 50.29`、Citeseer `56.51 / 57.44 / 56.99`。因此，平均池化和 attention 的局部正数不能单独写成主方法收益；当前证据更支持“文本重新编码是必要的”，而不是“某一种可学习文本池化已经被验证为最优”。

### 6.6.2 图条件文本的负结果

P2-A 进一步固定 3 个类别顺序和 6 个 Student seeds，比较文本选择是否真的需要图邻域条件。下表为最终阶段 Macro-F1，所有方法使用相同 rows、同一文本 token 预算和同一 topology。

| 图文处理方式 | Cora | Citeseer | 相对 center concat |
|---|---:|---:|---|
| Center concat | 48.46 | 54.97 | 基准 |
| Text-only attention | 51.25 | 54.08 | `+2.79 / -0.89` |
| Graph selection | 51.51 | 53.12 | `+3.05 / -1.85` |
| Graph pooling | 49.46 | 54.10 | `+1.00 / -0.87` |
| Shuffled graph condition | 52.32 | 53.94 | `+3.86 / -1.03` |

真实 graph selection 比 shuffled graph condition 低 `0.81` 个百分点（Cora）和 `0.82` 个百分点（Citeseer）。这说明当前图条件 query 没有被证明真正利用了邻域语义，故主方法保留简单的 center concat，不把 graph-conditioned attention 写成已成立贡献。

### 6.6.3 合成属性 X 的负结果与目标错位

P1-X 固定聚类、文本和 topology，仅替换合成属性的生成算子。三顺序平均的最终阶段 Macro-F1 如下：

| X 构造方式 | Cora | Citeseer | 相对 Mean-X |
|---|---:|---:|---:|
| Mean-X，当前默认 | 48.40 | 54.63 | 基准 |
| Weighted-X | 48.42 | 54.33 | `+0.02 / -0.29` |
| Moment-X | 47.13 | 53.76 | `-1.27 / -0.86` |

Message-conditioned X 虽然使 `81/81` 个 changed rows 的局部 message error 平均下降约 `29.5%`，但累计 F1 相对 C0 下降 `0.824` 个百分点，old-class F1 上升 `1.476` 个百分点、new-class F1 下降 `3.415` 个百分点；message error 与最终 F1 差的相关约为 `0.121`，与 new-class F1 差的相关约为 `0.151`。Shared-P + bounded-X 的累计变化为 `-0.154` 个百分点，正收益 cell 仅 `5/12`，最差 cell `-3.700` 个百分点。由此可见，局部 message proxy 下降不等价于新类可学习性提升，不能用它单独驱动 X 更新。

### 6.6.4 response selector 与动态预算的负结果

| 机制 | Cora | Citeseer | 额外成本或边界 | 结论 |
|---|---:|---:|---|---|
| Response selector，顺序平均 | `-0.93 pp` | `+0.85 pp` | 约 `259.18 s / 0.46 s` 与 `229.78 s / 0.56 s`，选择器/低成本构造 | 排序跨数据集翻转，约 400 到 600 倍开销 |
| Adaptive-rate | `49.16`，相对 fixed-rate `+1.88 pp` | `54.90`，相对 fixed-rate `-2.85 pp` | 总预算随 response 改变 | 方向不稳定，保留为消融 |
| Fixed-total | 相对 fixed-rate `-4.49 pp` | 相对 fixed-rate `-4.15 pp` | 总 rows 不随数据规模增长 | 不适合作为持续默认协议 |
| Response-gain extra budget | 下降 | 下降 | 同时改变 rows、候选和训练轨迹 | 无法区分选择收益与资源收益 |

这些结果解释了为什么主方法不采用昂贵的多候选 response selector，也不让短训 response 决定总预算。response 仍然可以作为后续诊断信号，但在当前证据下不是可靠的构造目标。

### 6.6.5 拓扑和历史 memory maintenance 的负结果

| 机制 | 数据集/任务 | 主要结果 | 对主方法的处理 |
|---|---|---|---|
| R17 message topology gate | 小图节点分类 | 最好 gate 最终 `+0.247 pp`，stage mean `-0.682 pp`，old/new 为 `+1.995 / -2.362 pp`，最差 cell `-4.919 pp` | 不晋升，固定 C0 topology |
| T+degree | BookChild 节点分类 | 分类约 `+1.546 pp` | 条件性正信号，未并入 Semantic-Recursive |
| T+degree | arXiv 节点分类 | 分类约 `-0.662 pp` | 说明拓扑收益依赖数据集 |
| T+degree | BookChild 链路预测 | AUC/AP 约 `+0.723 / +0.980 pp` | 任务相关的拓扑收益，单独报告 |
| T+degree | arXiv 链路预测 | AUC/AP 约 `-0.011 / -0.050 pp` | 近似持平，不能宣称普适 |
| Bounded refresh | Cora | cumulative/current/old 为 `-0.06 / +1.13 / -0.26 pp` | 旧 memory 修改不稳定 |
| Bounded refresh | Citeseer | cumulative/current/old 为 `-2.64 / -1.95 / -2.72 pp` | 关闭 refresh |
| Role-aware matching | 独立小图 cell | 平均 `+1.468 pp`，7/12 正，最差 `-2.042 pp` | 作为机制消融，不进入默认算子 |

拓扑和 refresh 的结果并不是“这些操作在任何任务都无用”，而是说明它们同时改变消息传播和历史保持，风险高于 payload assignment。Semantic-Recursive 因此选择固定 C0 topology 和不可回读的递归 parent，把主贡献限制在可审计的 payload-slot placement。

## 6.7 其它 TAG 凝聚方法的分层结果

其它方法必须按访问权限和下游任务分层。严格 continual 主表只允许使用当前 delta、上一阶段 synthetic parent、相同 memory rate、相同 Student 和相同训练步数；否则得到的是 offline reference、stateful control 或外部模型参考，不能直接排成同一名次。

### 6.7.1 严格 memory-only continual

C0、SR 和共享初始图/拓扑的 coreset 控制见第 6.2 节；外部 CaT/GECC 机制适配与独立构图 coreset 见第 6.2.1 节。二者都不继承 Student 参数或回读历史 raw，但回答的问题不同。SR 四条件 Final F1 为 53.02，不能因为它高于旧控制表的 coreset，就忽略本轮 CaT 的 54.62 和 k-center 的 54.18。

### 6.7.2 Offline synthetic、完整 raw 与 stateful Student

历史 canonical 数据集级参考：

| 方法 | Cora Final F1 | Citeseer Final F1 | 访问/训练条件 |
|---|---:|---:|---|
| Full-history synthetic re-condensation | 47.21 | 48.28 | 每阶段回读全部历史 raw 并重写 memory |
| Complete raw graph, 8192 steps | 51.31 | 55.60 | 完整 raw、长训和 checkpoint 选择；非理论上界 |

两个小图、两个顺序的四条件等权参照：

| 方法 | Final F1 | Final Accuracy | 访问/训练条件 |
|---|---:|---:|---|
| SR memory-only scratch | 53.02 | 59.59 | 5% memory，从头重训 |
| Complete raw graph scratch | 49.73 | 54.96 | 累计合法 raw，实际每阶段 2048 steps |
| Student delta-only fine-tune | 51.26 | 57.57 | 保留上一阶段 Student，访问当前 delta |
| Student C0-replay + delta fine-tune | 53.03 | 58.76 | 保留 Student、memory replay 和当前 delta |

此处更正旧表：49.73/54.96、51.26/57.57、53.03/58.76 是四条件 F1/Accuracy，并非 Cora/Citeseer 的两个 F1。完整 raw 与 SR 使用同一步数规则，但 raw 达到 2048-step 上限，SR 最终为 Cora 1504、Citeseer 1776 steps，不是严格 equal steps。SR 同条件相对这一 raw 参考的 F1 保留率为 102.08%-109.83%，Accuracy 为 104.09%-116.44%；大于 100% 不代表超过充分优化的 full-data 上界。

这些结果不是用来证明 Semantic-Recursive 在所有条件下超过完整 raw，而是分离三种因素：memory 构造质量、原始数据访问权限和模型状态保留。尤其 replay + delta 的分数接近 Semantic-Recursive，但训练成本约为其 `2.9x`，因此不能当作同资源凝聚方法。

### 6.7.3 静态 TAG 方法和外部模型参考

| 方法 | Cora Final F1 | Citeseer Final F1 | 方法边界 |
|---|---:|---:|---|
| TAGSAM-ASM self-loop | 47.73 | 51.09 | 静态适配，可回读累计 pool，self-loop topology |
| TAGSAM-ASM semantic | 52.86 | 42.00 | 静态适配，可回读累计 pool，semantic topology |
| STAD-local-canonical | 33.99 | 28.91 | 静态 TAG 适配，迁移到本文 InfoNCE Student |
| STAD native supervised sanity | 73.63 | 67.46 | 原生监督 GCN，不与本文 Student 直接比较 |
| 官方 GraphCLIP checkpoint | 58.98 | 57.76 | 官方 GPS checkpoint，完整图上下文，纯推理 |

STAD native supervised sanity 使用的是监督 GCN，因此分数不能解释为 STAD 在本文图文 Student 下失败；它说明下游目标和模型选择会显著改变凝聚图的相对排名。官方 GraphCLIP checkpoint 也不是凝聚方法，它只提供完整图和不同 Student 架构下的外部位置参考。TAGSAM 和 STAD 的结果有助于定位当前方法与公开 TAG 方法的差距，但不能在不同访问权限、不同损失和不同 backbone 下声称严格 SOTA。

### 6.7.4 负结果对方法选择的联合解释

组件实验给出一个一致的选择原则：当某个模块同时改变 payload 内容、传播 topology、memory 总量或 Student 训练轨迹时，局部 proxy 的改善不能直接被解释为凝聚效用改善。文本 attention、message-conditioned X、response budget、旧 memory refresh 和拓扑重连都出现了“局部指标变好但 new-class 或跨顺序 F1 变差”的情况。Semantic-Recursive 只保留能够在固定资源下审计的操作：Mean-X、center text re-encode、C0 topology、semantic conserved assignment 和 recursive append。

## 7 结果解释：为什么当前设计合理

### 7.1 固定 rate 而不是动态 response budget

response-gain extra budget 看起来能够把更多 rows 分给有用的新信息，但前期实验显示短训 ranking、teacher proxy 和最终 full-step F1 并不稳定一致。若同时改变候选、总 rows 和 Student 训练轨迹，就无法判断提升来自更好的构造还是更多资源。因此主协议固定 (\rho=5\%)，先把方法收益放在相同 memory budget 上。

### 7.2 Mean-X 和简单文本不是缺乏设计，而是因果控制

P-guided X、加权 X、text attention 和 graph-conditioned text 都可能在单个数据集上改善，但尚未稳定跨数据集。继续替换 X 和文本会让“payload placement 的收益”和“表示算子收益”同时变化。当前主方法保留 Mean-X、top-5 center text 和 MiniLM re-encode，使 semantic assignment 的变化可以被单独审计。

### 7.3 固定 topology 使贡献更可解释

T+degree 和其它拓扑改动在 BookChild 上有收益，也在 arXiv 上下降；链路预测和节点分类的方向也不完全一致。因而当前方法不声称拓扑创新，而是继承 C0 topology，专门研究在同一传播算子下 payload slot placement 是否能改变 student retrainability。

### 7.4 递归 parent 是持续问题的组成部分

stage-local semantic 每阶段都从 C0 parent 开始，而 semantic-recursive 把上一阶段 semantic memory 继续作为 parent。小图递归确认中，平均 cumulative 为 +1.640 pp，最差 cell 从 stage-local 的约 -2.625 pp 改善到 -0.528 pp。该结果说明 payload 的历史排列会影响下一阶段的传播和新增 block 作用，递归不是实现细节。

### 7.5 守恒 assignment 比任意 role matching 更适合做主线

此前非守恒 role-aware matching 可能复制、覆盖或丢弃 payload，使得结果同时混入 memory 内容变化。当前 assignment 保持 payload multiset、source mass 和 topology 不变，因此即使收益有限，也能清楚说明观察到的是 slot placement 的作用，而不是资源偷换。

## 8 解决了哪些 challenge，哪些仍未解决

### 8.1 已经部分解决的 challenge

1. **历史 raw 不可回读：** 构造审计显示 historical raw reads 为 0，算法只依赖 parent memory 和 current delta。
2. **memory 随数据增长：** fixed-rate budget 使 memory 随累计数据增加，不需要固定每类 15 rows。
3. **图文 payload 一致性：** X、文本、mass 和来源来自同一 source piece，assignment 移动完整 payload。
4. **持续过程可审计：** budget、prefix、payload multiset、topology hash、seed 和 Student protocol 均可检查。
5. **新类可重训性：** canonical 大图中 new-class Macro-F1 平均提高 1.764 pp，说明方法对部分新类塑性确实有帮助。

### 8.2 尚未解决的 challenge

1. **顺序不变性：** P0 独立 order 合计只有 5/12 正阶段，增量平均约 +0.020 pp。
2. **稳定超越强基线：** Semantic-Recursive 在当前大图表中有方向性收益，但没有证明稳定超过完整 raw、强 k-center 或所有静态 TAG 方法。
3. **新类和旧类 trade-off：** arXiv/BookChild 的收益偏新类，SportsFit 偏旧类；最差阶段仍可下降超过 2 pp。
4. **理论保证：** Hungarian semantic matching 是成熟工具，目前没有证明该 cost 必然降低最终 Student loss 或 F1。
5. **公开机制与原生方法差异：** CaT/GECC 小图 TAG 机制适配已完成，尚未完成 PUMA、GECC 原生演化/调优分支和跨大图对照；TAGSAM/STAD 仍属于不同协议的静态参照。

## 9 创新性和论文贡献的客观评价

当前最可信的算法创新不是新的 GNN 或文本 encoder，而是一个面向访问约束的 continual memory operator：

- 在历史 raw 不可回读时定义递归 synthetic parent；
- 在固定 budget 和固定 topology 下构造新 block；
- 对完整 graph-text payload 做同类语义守恒 slot assignment；
- 用不变量审计 memory 是否真的只追加了合法信息。

这一贡献有三个优点。第一，它对应一个明确而实际的 continual TAG challenge，而不是把静态 condensation 结果直接套到持续 setting。第二，它的机制可以被逐步拆分，实验能区分更多 memory、改变 topology、改变 Student 和改变 payload placement。第三，正负结果都能形成可解释研究故事：canonical order 有收益，独立 order 暴露边界，复杂 selector 和动态预算没有稳定晋升。

但创新强度仍然有限：Mean-X、文本拼接、Hungarian matching 和 C0 topology 本身不是新工具；已形式化的最优性只针对固定代价下已修改 payload 的矩形 assignment 子问题，不覆盖残余补全、split/merge 或下游 F1。当前也没有稳定 SOTA 证据，新增外部对照中 SR 的平均 F1 低于 CaT 适配和独立 k-center。方法可继续作为长论文研究主线，但还需要独立外部比较、匹配预算控制和更有说服力的机制证据；不将 internal C0 称为公开 baseline，也不混排 stateful Student 或监督 STAD。

## 10 后续工作和投稿路线

### 10.1 近期必须完成

1. 冻结当前 Semantic-Recursive 和 C0，不再根据独立 order 结果反向调参。
2. 将小图统一表、大图五层表和 P0 独立 order 表合并为论文主结果、附录结果和失败边界三张表。
3. CaT/GECC 小图机制适配已完成；下一步对 SR、CaT 和独立 k-center 做 3-5 个新预注册顺序、构造 seeds 和 matched-budget 控制，PUMA/原生配置及静态 TAG 对照按访问层继续补充。
4. 对所有主表补充 construction time、Student time、memory rows、old/new F1 和 paired confidence interval。
5. 保留 canonical、independent order 和 construction seed 三类结果，不只报告最好的 canonical cell。

### 10.2 只允许一个窄的算法扩展

如果继续做新算法，优先采用有界的 message-aware semantic matching：

\[
C_{ij}=C_{ij}^{semantic}+\lambda C_{ij}^{message},
\]

其中 (lambda) 在实验前固定，不使用最终 F1 选择；拓扑、rows、文本 token budget 和 Student 保持不变；如果局部 message cost 变差，则回退到当前 semantic matching。该扩展只能作为单因素实验，不能同时加入动态预算、old-memory refresh、GraphGPS、文本 attention 和新的 X 优化。

### 10.3 论文定位

推荐标题方向：

> **Continual TAG Condensation under Historical Raw-Data Inaccessibility: Semantic-Conserved Recursive Memory Append and Plasticity Analysis**

推荐主张：

> 在历史 raw 不可回读的持续 TAG 凝聚中，Semantic-Recursive 用固定预算下的语义守恒 payload placement 和递归 parent 保持可审计的 synthetic memory。它在 canonical/reference order 上改善部分新类的 student retrainability，同时通过独立 order 和系统负结果揭示了顺序依赖和 old/new trade-off。

不推荐主张：

- 对所有数据集和顺序稳定超过 C0；
- 优于所有强 coreset 或公开 TAG condensation 方法；
- 达到 universal SOTA；
- semantic matching 理论上最优；
- 官方 GraphCLIP 或监督 STAD 分数与本文 Student 结果可直接横向排序。

## 11 结论

本文整理了一个面向持续 TAG 凝聚的完整主线方法 Semantic-Recursive。它从 current delta 构造图文联合 source slots，在固定 rows 下进行局部 two-prototype split/merge，用 Mean-X 和来源一致的文本重编码生成 payload，再以同类语义守恒 assignment 将完整 payload 放入结构 slot，最后递归追加到上一阶段 synthetic memory。旧 prefix、旧 topology、memory budget 和 Student 训练协议均可以审计。

实验表明，SR 在小图内容控制中相对 C0 提升 0.925 pp，在三个大图 canonical `base+4` 的增量条件平均提升 0.858 pp，新类提升 1.764 pp。新增独立构造外部表中，SR Final F1 53.02，高于 GECC 适配 48.70 与 medoid 47.32，低于 CaT 适配 54.62 与 k-center 54.18；它是有竞争力但未全面领先的合成方法。大图独立顺序相对 C0 也只有接近零的总体收益。必须同时保留这些边界，不能将共享 C0 拓扑的 coreset 控制结论推广到所有外部方法。

因此，Semantic-Recursive 可以作为当前论文的主方法和实验组织中心，论文应采用“受约束 memory operator + student retrainability + 机制正负结果”的定位。投稿前仍需补强公开基线和匹配预算协议，但不建议再把大量未经验证的 selector、dynamic budget、attention、refresh 和 GraphGPS 模块堆叠到主方法中。

## 参考工作和证据文件

正式投稿前需要将以下工作替换为完整 BibTeX 信息：GraphCLIP、TAGSAM、STAD、GCond、SFGC、GCondenser、OpenGC、k-center coreset 及持续图学习相关工作。本稿中的数值和协议证据来自项目内部报告：

- [小图统一基线比较](../current/TGC_SEMANTIC_RECURSIVE_UNIFIED_BASELINE_COMPARISON_ZH_2026-09-26.md)
- [Semantic-Recursive 完整评估](../current/TGC_SEMANTIC_RECURSIVE_COMPLETE_ASSESSMENT_ZH_2026-09-26.md)
- [小图递归确认](../current/TGC_SEMANTIC_RECURSIVE_CONFIRMATION_ZH_2026-09-26.md)
- [大图 base+4 确认](../current/TGC_SEMANTIC_RECURSIVE_LARGE_CONFIRMATION_ZH_2026-09-26.md)
- [大图五层 baseline comparison](../current/TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md)
- [P0 独立顺序与 construction seed](../current/TGC_P0_INDEPENDENT_ORDER_CONFIRMATION_ZH_2026-09-26.md)
- [官方 GraphCLIP checkpoint](../current/OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md)
- [持续图凝聚相关工作与基线推进](../current/CONTINUAL_GRAPH_BASELINE_REVIEW_ZH_2026-09-27.md)
- [Semantic-Recursive 受约束传输实现规范](../current/SEMANTIC_RECURSIVE_CONSTRAINED_TRANSPORT_SPEC_ZH_2026-09-27.md)
- [外部持续凝聚适配冻结协议](../current/EXTERNAL_CONTINUAL_ADAPTER_PROTOCOL_ZH_2026-09-27.md)
- [外部持续凝聚适配 360-fit 结项](../current/EXTERNAL_CONTINUAL_ADAPTER_COMPLETION_ZH_2026-09-27.md)
