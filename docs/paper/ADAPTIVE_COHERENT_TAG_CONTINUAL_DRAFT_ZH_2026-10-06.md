# Adaptive Coherent TAG：历史原始图不可回读条件下的持续图文凝聚

**论文草稿，中文工作版**  
**版本：2026-10-06**  
**当前主方法：Adaptive coherent，即 CaT 初始化的 X-conditioned coherent adaptive slot routing**

## 摘要

持续图凝聚需要在图数据不断到达的情况下维护一个固定或受限增长的合成记忆，但现有图凝聚方法通常假设可以访问完整原始图，或者只处理结构和节点特征而没有显式建模节点文本。对于文本属性图（Text-attributed Graph, TAG），这一假设带来两个额外困难：历史原始图不能回读时，新增合成节点的文本 payload 必须来自当前阶段可访问的数据；同时，同一类别内不同合成节点使用什么文本，会直接影响下游图文模型的类内一致性和新类塑性。

本文研究历史 raw graph 不可回读、旧合成记忆只读、当前阶段增量数据可访问的持续 TAG 凝聚问题。我们以 CaT 的 class-wise MMD 合成特征优化为基础，提出 X-conditioned coherent adaptive slot routing：首先使用 CaT 通过梯度下降生成当前阶段合成特征 X；随后只在当前 delta 中构造真实、可审计的文本候选池；依据每个类别的图文离散度自适应分配 1、2 或 3 个文本 slot；最后使用合成特征 X 对候选文本进行可微的类别内路由，并将 soft assignment 材料化为真实文本。该过程不回读历史 raw graph，不修改旧 memory，也不把不可落地的文本向量凸组合直接存入 memory。

在 Arxiv、SportsFit、WikiCS 和 BookHis 四个大图上，我们完成了 3 个独立类别顺序、3 个 construction seeds 和 6 个 Student seeds 的确认实验，共 36 个配对 cell。Adaptive coherent 在 36 个 cell 中有 32 个高于 CCTP top-5；在 Arxiv、SportsFit 和 WikiCS 上分别取得稳定的平均收益，但在 BookHis 上仍受到新类塑性和长尾分布偏置的限制。与随机 coreset、K-center、Medoid、CaT-MMD-TAG 和 GECC-hard-TAG 的严格协议适配版本相比，方法在多数数据集上表现更好，但尚不能宣称跨所有数据集和所有公开 TAG 方法的普遍 SOTA。本文同时报告方法的协议边界、失败结果和公开方法适配方式，并客观讨论其作为 WWW 长文的贡献基础和仍需补齐的实验。

**关键词：** 持续图凝聚；文本属性图；图文对齐；CaT；payload routing；zero-shot node classification

![方法流程](assets/adaptive_coherent/method_flow.png)

**图 1.** Adaptive coherent 的总体流程。CaT 只负责优化新增合成特征 X；文本分支在当前 delta 中建立真实候选池，按图文离散度分配 slot，再根据 X 路由到最终真实文本。

## 1 引言

### 1.1 研究背景

图凝聚的目标是在较小的合成图或合成记忆上保留原图的训练价值，使下游模型能够在显著降低数据和训练成本的情况下获得接近原始图的性能。静态图凝聚通常可以访问完整原始图，并在一次优化中同时使用全图特征、标签、结构和下游模型响应。然而，真实应用中的图数据往往是连续到达的：新节点和新类别在不同阶段出现，旧数据可能由于隐私、存储或计算约束而不能再次读取。此时，每个阶段只能使用当前增量构造新的 memory block，并将其追加到历史 synthetic prefix。

文本属性图进一步引入了文本 payload 问题。合成节点的特征可以通过梯度优化得到，但文本通常不能简单地对字符串做连续优化。若为每个合成节点独立选择不同的文本，类内文本方差可能被下游对比学习当作负信号；若每个类别只广播同一条文本，则可能丢失类别内的局部语义。因而，持续 TAG 凝聚需要同时处理以下约束：

1. 历史 raw graph 不可回读；
2. 旧 synthetic memory 不应被任意重写；
3. 文本 payload 必须来自当前可访问的数据并可审计；
4. 合成特征、文本表示和下游图文 Student 之间需要保持语义相容；
5. 方法应当能够在不同规模和不同文本离散度的数据集之间迁移。

### 1.2 现有方法的不足

CaT 为持续图凝聚提供了一个重要基础：它使用类别条件的分布匹配目标优化合成特征，并可以在增量阶段生成新的合成节点。但 CaT 本身不处理 TAG 文本 payload，也不解决同一类别内多个合成节点如何使用不同或共享文本的问题。直接把 CCTP 或简单的 top-k 文本拼接附着到 CaT X 上，会出现两个问题：第一，同一类别的文本选择粒度与 X 的合成行粒度不一致；第二，逐行文本变化可能被图文对比学习放大，导致类内语义分裂。

我们之前还尝试过 Student-response、旧 memory revision、X/P/A 联合梯度优化和拓扑重连等方向。这些实验说明，直接用下游 Student 的短期 loss 反向选择文本或边，容易受到训练轨迹、候选离散化和长 horizon 不稳定的影响；而一次性同时修改 X、文本、拓扑和旧 memory，则难以判断收益来源。因此本文将当前主线收敛为一个较小但可解释的核心机制：保留 CaT 对 X 的稳定优化，只在真实文本 payload 的类别内分配粒度和路由方式上做受限改进。

### 1.3 本文贡献

本文的贡献可以概括为四点：

1. **问题定义。** 提出历史 raw graph 不可回读、旧 synthetic prefix 不修改、当前 delta 可访问的持续 TAG 凝聚协议，并明确 memory、访问和材料化不变量。
2. **方法设计。** 提出 X-conditioned coherent adaptive slot routing：使用当前 delta 的图文离散度决定类别级 slot 数量，再使用 CaT 合成特征 X 对真实文本候选做可微路由和硬材料化。
3. **协议化验证。** 在多个小图和大图上统一 Student、memory budget、类别到达协议、历史 raw 访问边界和评估规则，报告 Macro-F1、Accuracy、Old/New F1、阶段平均性能、最差阶段和构造成本。
4. **失效边界。** 系统报告 fixed slot、anchor、类共享文本、逐行文本、多原型、CCTP、coreset 和 GECC 适配版本的结果，说明方法收益主要出现在哪些数据集，BookHis 上为什么仍然失败，以及为什么不能把当前结果包装成无条件 SOTA。

## 2 相关工作

### 2.1 图凝聚与持续图学习

传统图凝聚方法通常在完整图上优化合成节点特征、标签和拓扑，使合成图能够近似原图或让 Student 在合成图上训练后获得较好性能。持续图学习方法则关注类别、节点或任务逐步到达时的旧知识保持与新知识学习。CaT（*CaT: Balanced Continual Graph Learning with Graph Condensation*）将图凝聚引入持续场景，其核心思想是为每个增量阶段生成新的合成记忆，并通过类别级分布匹配保持持续学习能力。

本文不重新发明 CaT 的 X 优化目标，而是在其输出的合成特征上增加 TAG 文本 payload 的一致性和路由机制。这样可以将性能差异归因到文本处理，而不是把 CaT 的图凝聚、Student 和文本模块同时改变。

### 2.2 TAG 凝聚与图文对齐

静态 TAG 凝聚方法通常需要同时处理节点特征、图结构和文本属性，常见路线包括使用文本编码器生成节点表示、通过图文对齐损失约束合成特征，或者通过文本片段、摘要和局部语义候选建立合成节点的文本 payload。TAGSAM、GraphCLIP 及相关方法可以作为静态 TAG 的参考，但其原始协议通常使用完整静态图、固定训练/验证划分或监督 Student，不能直接等价于本文的 current-delta-only 持续协议。

本文采用严格适配而不是直接移植官方结果：所有方法都必须只读取当前 delta，使用相同 5% memory 预算、相同 Prompt zero-shot Student，并将旧 memory 作为 frozen prefix。这样做牺牲了与原论文官方数字直接比较的便利，但避免把协议差异误认为算法收益。

### 2.3 Coreset 和结构无关方法

K-center/FPS、Medoid 和 Random real coreset 是重要的低成本参照。它们不生成连续合成特征，而是在当前 delta 中选择真实节点，随节点保留真实文本。这些方法可以检验 Adaptive coherent 的收益是否只是因为选择了更有代表性的真实节点，而不是因为文本 routing 本身。

## 3 问题定义与持续 TAG 协议

### 3.1 持续 TAG 数据流

在阶段 t，原始图增量表示为：

\[
\Delta G_t=(\Delta V_t,\Delta E_t, X_t^{raw}, T_t^{raw}, Y_t^{raw}).
\]

其中 `X` 为节点特征，`T` 为原始文本，`Y` 为类别标签。历史阶段的 raw graph 不可回读；系统只保留已材料化的 synthetic memory：

\[
M_{t-1}=(X_{t-1}^{syn}, T_{t-1}^{syn}, E_{t-1}^{syn}, Y_{t-1}^{syn}).
\]

本阶段构造器输出一个新增 block `B_t`，最终 memory 为：

\[
M_t=M_{t-1}\oplus B_t.
\]

当前主方法使用 append-only frozen-prefix 规则：`M_{t-1}` 不被修改，`B_t` 只来自 `Delta G_t` 和 CaT 当前阶段计算结果。

### 3.2 访问和预算约束

本文采用以下统一设置：

| 协议项 | 设置 | 作用 |
|---|---|---|
| 历史 raw graph | 不可访问 | 防止每阶段重新读取旧完整图 |
| 当前 delta | 可访问 | 允许构造当前阶段 X 和文本候选 |
| 旧 synthetic memory | 只读 prefix | 保持 continual append-only 语义 |
| memory | 5% cumulative rate | 所有方法使用相同预算 |
| 类别到达 | base+4 或三阶段增量 | 评估多阶段持续适应 |
| 文本预算 | 每个 payload 最多 320 字符 | 控制存储和编码成本 |
| Student | Prompt zero-shot | 测试时不训练分类器 head |
| 构造选择 | 不使用 report/test 标签和最终 F1 | 防止测试泄漏 |

### 3.3 下游 Student

当前实验使用冻结的 Prompt zero-shot Student。Student 为 edge-aware 两层 GCN、图特征投影和文本投影组成的图文模型，训练阶段使用双向 InfoNCE；测试时不训练 learned classifier head，而是将图节点表示与类别文本 prompt 比较，选择相似度最高的类别。

这一设计有两个重要含义：第一，文本 payload 不是展示用字符串，而是直接参与 Student 的图文对齐；第二，Macro-F1 和 Accuracy 必须同时报告，因为 BookHis 等长尾数据集上 Accuracy 可能因多数类偏置上升，而 New F1 反而下降。

## 4 CaT 基础与 Adaptive coherent 方法

### 4.1 CaT 继承的内容

本文继承 CaT 的核心内容如下：

1. **类别条件的合成行分配。** 根据当前阶段的类别标签和 memory budget，为每个类别分配需要生成的合成行数。
2. **类别级 MMD/CGM 分布匹配。** 在当前 delta 中计算真实特征的类别分布，在合成特征上优化对应的类别分布差异。
3. **梯度优化合成特征 X。** `X` 是可学习参数，通过随机特征映射和类别级 MMD 目标更新，而不是通过随机抽取真实节点替代。
4. **阶段级追加。** 每个阶段只生成新增 block，并与历史 synthetic memory 组合供 Student 训练。
5. **预算控制。** 通过固定比例或固定行数限制合成 memory 大小。

本文没有声称重新提出 CaT 的 MMD 目标，也没有将 Student 的训练 loss 反传到 CaT 的 X 优化中。当前主方法的创新集中在文本 payload 的候选构造、slot 粒度和 X-conditioned routing。

### 4.2 CaT 特征优化

令 `R_t` 为当前阶段需要生成的合成行，`y_j` 为合成行 j 的类别。CaT 首先优化：

\[
L_{CaT}(X_t^{syn})=
\sum_{c\in\mathcal C_t}
\operatorname{MMD}\left(
\phi(X_t^{raw,c}),
\phi(X_t^{syn,c})
\right),
\]

其中 `phi` 是随机特征映射，`c` 是当前阶段类别。实现中使用固定的随机投影/随机特征权重对真实特征和合成特征进行变换，再按类别计算均值匹配损失。对 `X_t^{syn}` 执行梯度下降，得到 CaT 输出：

\[
X_t^{(0)}=\operatorname{CaT}(\Delta G_t, m_t).
\]

这是本文方法与 CaT 的接口：后续文本模块以 `X_t^(0)` 作为 query/row representation，而不是重新优化一个完全不同的特征空间。

### 4.3 文本 payload 接口约束

文本模块只处理当前阶段 delta 中可访问的真实文本，并将最终字符串、来源节点 ID 和冻结 MiniLM embedding 保持一一对应。主方法不生成新的自然语言，不读取历史 raw，也不把无法落地的 embedding 凸组合直接写入 memory。候选池的完整构造、文本预算和来源审计规则统一放在第 4.5 节，避免在多个小节重复定义同一对象。

### 4.4 slot policy 概览

主方法不为所有类别固定使用相同数量的文本。它先根据当前 delta 内每个类别的图文离散度决定 slot 数，再在该类别的真实候选池中选择对应数量的代表文本。离散度定义、相对分位数规则、候选数量上限和大图中的实际分配统一在第 4.5.2 节给出；本节只说明该 policy 的作用：让内部较集中类别保持低 payload 开销，让图文异质性较高的类别保留多个语义支撑点。

### 4.5 有界候选选择

当前最终大图主方法在 runner 中使用 `candidate_mode=coherent_raw_medoid` 和 `adaptive_slots=True`。本节给出文本候选从 raw delta 到最终 slot bank 的完整流程。这里的 “medoid” 不是再次运行完整的 k-medoids，而是指：从当前类别的真实文本记录中选出少量兼容且互补的代表候选。必须区分两个集合：候选池是当前类别所有可访问的 raw 文本，slot bank 是经过压缩后供合成行路由使用的 1--3 条真实文本。

#### 4.5.1 主方法的候选来源

对阶段 `t` 的类别 `c`，先取当前 raw delta 中所有满足 `y_i=c` 的节点：

\[
\mathcal R_{t,c}=\{i\in\Delta V_t:y_i=c\}.
\]

每个节点 `i` 提供一个真实候选记录：

\[
r_i=(s_i,\, id_i,\, x_i^{raw},\, e_i^{text}),
\]

其中 `s_i` 是原始文本字符串，`id_i` 是来源节点 ID，`x_i^{raw}` 是该节点的原始特征，`e_i^{text}` 是冻结 MiniLM 对最终字符串的编码。候选只来自当前 delta，不来自旧 synthetic memory、历史 raw graph、report/test 节点，也不由语言模型生成。主方法不会把多个无关节点的文本拼接成一个新字符串。

文本先做空白归一化和边界感知截断：不超过 320 字符时原样保留；超过时优先在句末标点处截断，其次在空格处截断，最后才进行硬字符截断。之后对截断后的最终字符串重新编码，因此保存的字符串、embedding 和来源 ID 是一致的。

#### 4.5.2 自适应决定候选数量

对每个当前类别计算图文离散度：

\[
d_c=\frac{1}{2}\left(1-\frac{1}{|\mathcal R_{t,c}|}
\sum_{i\in\mathcal R_{t,c}}\cos(x_i,\bar x_c)\right)
+\frac{1}{2}\left(1-\frac{1}{|\mathcal R_{t,c}|}
\sum_{i\in\mathcal R_{t,c}}\cos(e_i,\bar e_c)\right).
\]

然后只在当前阶段出现的类别之间对 `d_c` 做相对排序：

| 当前阶段类别离散度分位 | 候选 slot 数 |
|---|---:|
| `< 50%` | 1 |
| `50%--80%` | 2 |
| `>= 80%` | 3 |

slot 数还会被当前类别的合成行数和可用 raw 候选数截断。因此这不是“每类固定三个文本”，而是低离散度类别用一个候选，高离散度类别才允许多个候选。

#### 4.5.3 从 raw 候选中选出 slot bank

对每个候选文本 `k`，先用它的来源 raw 特征作为图原型 `g_k`，计算它与该类别所有 CaT 合成行的平均图相似度、与类别文本中心的文本相似度，并计算 prompt margin penalty。实现中的 seed 代价为：

\[
b_k=1-\frac{1}{|R_c|}\sum_{j\in R_c}\cos(X_j,g_k)
+0.50\left(1-\cos(e_k,\bar e_c)\right)
+0.15\,p_k,
\]

其中 `p_k` 是该真实文本相对类别 prompt 的 margin penalty。首先选取 `b_k` 最小的候选作为兼容性 seed。

随后不再枚举文本组合，而是逐个加入与已选候选差异较大的记录。标准 coherent 分支的补充分数为：

\[
s(k)=0.50\,d_{text}(k,S)
+0.35\,d_{graph}(k,S)
-0.15\,b_k,
\]

其中 `S` 是当前已选候选集合，`d_text` 和 `d_graph` 分别是到已选文本 embedding 和图原型的最小 cosine distance。这样既保留一个最符合当前 CaT `X` 的候选，也避免所有 slot 只是同一条或高度重复的文本。

主方法最终只保留 `1--3` 条真实文本记录，复杂度是对当前类别节点的线性扫描乘以很小的 slot 数，不是组合枚举。也就是说，大类可能有很多 raw 文本，但不会产生指数级的文本组合。

#### 4.5.4 与标准 CCTP/chunk 消融的区别

论文中还报告的 nearest、TAGSAM-style chunk、graph-conditioned 和 graph-text aligned 属于另一条标准文本适配路径。该路径先在当前类别建立有界 support pool：按 CCTP anchor、与 CaT `X` 的图特征相似度、与类别文本中心的文本相似度依次合并并去重，最多保留 `support_pool_size=32` 个 raw 来源；再把来源文本切成最多 120 字符的 chunk，按文本、图相似度、prompt 和冗余惩罚组成最多 320 字符的真实 payload。

因此：`32` 是标准 chunk/CCTP 消融的 support pool 上限，不是 Adaptive coherent 主方法的候选数量；主方法的最终 slot bank 上限是自适应得到的 `1--3`。两者都不访问历史 raw，也不生成不可审计的连续文本向量。

### 4.6 X-conditioned differentiable routing

给定类别 c 的合成行集合 `R_c` 和候选 slot 集合 `P_c`，对每一行 j 维护 assignment logits `z_j`：

\[
\pi_{j}=\operatorname{softmax}(z_j).
\]

主方法实际使用的行到 slot 代价为：

\[
C_{j,k}=
1-\cos(X_j,g_k)
+w_t\left(1-\cos(e_k,\bar e_{P_c})\right)
+\lambda_p p_k,
\]

其中 `g_k` 是候选文本来源的 raw 图特征，`\bar e_{P_c}` 是当前候选 bank 的文本中心，`p_k` 是 prompt margin penalty。当前实现的 `w_t=0.50`，`\lambda_p=0.15`。

初始 assignment 为：

\[
\pi^{(0)}_{j}=\operatorname{softmax}(-C_j/\tau),
\]

其中 `\tau=0.15`。随后只优化 assignment logits，不更新 CaT 的 `X`、文本编码器或 raw 文本。

路由阶段实际优化：

\[
L_{route}=
\frac{1}{|R_c|}\sum_{j,k}\pi_{j,k}C_{j,k}
+\lambda_{load}L_{load}
 +0.01L_{ent}
+\lambda_{anchor}D_{KL}(\pi_j\|\pi_j^{init}).
\]

其中：

- 第一项让每个合成行选择与自身 X 更相容的文本；
- `L_load=KL(u_{use}\|u)`，其中 `u_{use}=\frac{1}{|R_c|}\sum_j\pi_j` 是实际 slot 使用分布，`u` 是均匀 slot 分布，用于抑制所有行塌缩到同一个 slot；
- `L_ent=\frac{1}{|R_c||P_c|}\sum_{j,k}\pi_{j,k}\log\pi_{j,k}`。由于该项本身为负熵，正系数会轻微鼓励 assignment 保留不确定性，而不是过早变成 one-hot；
- anchor KL 防止短迭代路由偏离初始图文兼容结果过远。当前默认 `routing_steps=30`，优化器是 Adam，学习率为 `0.05`。

需要明确：这里的“可微”指 assignment logits 的局部梯度优化，并不是 Student 下游 loss 反向穿过硬文本字符串更新 X。当前 `X` 在 CaT 完成后被冻结；文本候选来自真实字符串；最终 memory 使用 hard materialization。

### 4.7 硬材料化与阶段追加

路由结束后，对每个合成行取最大概率 slot：

\[
k_j^*=\arg\max_k\pi_{j,k}.
\]

最终保存：

\[
B_t=\{(X_j^{(0)},T_{k_j^*},y_j)\}_{j\in R_t}.
\]

其中 `T_{k_j*}` 是当前 raw delta 中的真实文本字符串经过 320 字符预算处理后的结果。随后将 `B_t` 追加到旧 memory。当前版本的审计不变量是：

| 不变量 | 当前实现 |
|---|---|
| 历史 raw reads | 0 |
| old memory updated | False |
| X updated after CaT | False |
| topology updated | False |
| text generated by LLM | False |
| hard materialization | True |
| current-delta-only | True |

### 4.8 方法复杂度与实现位置

设当前 delta 节点数为 `n`，合成行数为 `m`，slot 上限为 `s<=3`，文本编码维度为 `d`。当前 delta 中的每种文本只编码一次；slot 选择和 row-to-slot routing 都只在有限 slot 上进行，默认每个类别执行 30 步 Adam。因而额外成本主要来自当前 delta 文本编码和小规模 assignment 优化，不随历史 raw 图累积，也不枚举文本组合。CCTP/chunk 分支的 `support_pool_size=32` 仅用于其自身消融，具体区别见第 4.5.4 节。

关键实现文件为：

- `code/v2_llm_fusion/cat_multimodal_minimal.py`：CaT-style X、文本支持池和 payload 基础工具；
- `code/v2_llm_fusion/x_conditioned_payload_routing.py`：候选 slot、adaptive slot、routing 和 hard materialization；
- `code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py`：大图统一 runner；
- `code/v2_llm_fusion/run_prompt_zero_shot_small_20261006.py`：小图 Prompt Student runner。

## 5 实验设置

### 5.1 数据集和协议

| 规模 | 数据集 | 阶段设置 | 顺序/seed确认 | 主要目的 |
|---|---|---|---|---|
| 小图 | Cora | 三阶段 | 3 个类别顺序；当前 construction seed 主结果已完成 | 检查文本 routing 是否破坏 CaT |
| 小图 | Citeseer | 三阶段 | 3 个类别顺序；当前 construction seed 主结果已完成 | 检查文本 routing 的顺序泛化 |
| 大图 | Arxiv | base+4 | 3 顺序 × 3 construction seeds × 6 Student seeds | 图文离散度和规模迁移 |
| 大图 | SportsFit | base+4 | 3 顺序 × 3 construction seeds × 6 Student seeds | 文本结构较丰富的数据 |
| 大图 | WikiCS | base+4 | 3 顺序 × 3 construction seeds × 6 Student seeds | 图文语义泛化 |
| 大图 | BookHis | base+4 | 3 顺序 × 3 construction seeds × 6 Student seeds | 长链、长尾和失败边界 |

所有方法使用相同 memory budget、Student seeds、report/test 规则和阶段输出。大图共完成 36 个配对 cell。小图主表使用三种类别顺序的等权汇总；额外 construction seed 复现仍应在论文定稿前补齐。

### 5.2 评价指标

主指标为 Final Macro-F1 和 Accuracy；辅助指标包括 Stage-average F1、Old F1、New F1、Worst-stage F1、construction time 和 Student time。对于持续任务，Final Macro-F1 不能单独使用：Old F1 衡量历史类别保持，New F1 衡量增量类别塑性，二者共同决定方法是否只是偏向多数旧类。

## 6 基线及协议适配

### 6.1 基线分层原则

当前实验中的“baseline”必须同时说明原论文任务、本文做的适配以及数值表中的角色。普通图持续凝聚加文本适配、静态 TAG 凝聚改成持续协议，以及本文内部文本组件消融不是同一类比较。

#### 6.1.1 普通图持续/演化凝聚的 TAG 适配

这一组方法原本不处理文本属性。本文保留其特征凝聚或聚类机制，再为输出的合成节点附加当前 delta 中合法的真实文本。

| 方法名 | 原始任务与核心机制 | 本文的 TAG 持续适配 | 当前数值角色 |
|---|---|---|---|
| **Adaptive coherent** | 本文主方法 | CaT X + 当前 delta 真实文本 + 自适应 1/2/3 slot + X-conditioned routing | 候选主方法 |
| **CaT-MMD-TAG + CCTP top-5** | 持续普通图凝聚；按类别用 MMD/CGM 优化合成特征 | 每阶段只用 current delta 优化 X；每个合成行从当前 delta 选择合法真实文本；旧 memory frozen-prefix | 严格外部基线 |
| **GECC-hard-TAG** | 演化/持续普通图凝聚；hard-hop/Lloyd 类原型聚类 | 在 current delta 上运行 hard clustering，再附加当前阶段真实 TAG 文本 | 严格适配外部基线 |
| **PUMA-inspired** | 持续普通图凝聚；pseudo-label、propagation、edge-free memory | 仅在 Cora/Citeseer 做小图兼容 adapter，并统一 Student、预算和访问边界 | 小图适配，未进入大图主表 |

`CaT-MMD-TAG` 和 `GECC-hard-TAG` 不是官方论文数字的原样复现，而是“普通图持续/演化凝聚机制 + TAG 文本附着”的严格协议适配版本。

#### 6.1.2 静态 TAG 凝聚的持续协议适配

这一组方法原本直接在静态 TAG 上工作，通常可以访问完整图、静态训练划分或自己的 teacher/Student。到目前为止，还没有一个官方静态 TAG 方法被完整改造成可进入本文 strict 主表的端到端持续版本。

| 方法名 | 原始角色 | 当前处理 | 是否进入严格数值主表 |
|---|---|---|---|
| TAGSAM | 静态 TAG 凝聚 | 只借鉴文本 chunk/payload 思想，没有直接移植官方静态流程 | 否，reference-only |
| STAD | 静态 TAG teacher/distillation 方向 | 尚未完成 current-delta-only、历史访问审计和统一 Prompt Student 适配 | 否，reference/adapted-only |
| GraphCLIP | 静态图文表示/对齐参考 | 依赖静态图文训练和不同图编码器 | 否，reference-only |

因此，`CaT + TAGSAM-style chunk` 不是 TAGSAM 的持续复现，而只是 CaT 固定 X 后的文本 payload 组件控制。真正的“静态 TAG -> 持续 TAG”外部 baseline 仍需单独实现和验证。

#### 6.1.3 普通静态图凝聚参考

GCond、SFGC、GCondenser 等属于普通静态图凝聚方法，不是 TAG 方法，也没有完成本文的持续 TAG 适配。它们通常需要完整历史 raw graph 或监督 Student，目前只作为 related-work/reference-only。

### 6.2 真实节点、学习控制与内部消融

| 方法名 | 作用 | 本文适配 | 数值角色 |
|---|---|---|---|
| Random real coreset | 随机保留真实节点 | current delta-only，保留真实文本，同预算同 Student | 严格外部控制 |
| K-center/FPS | 几何覆盖代表点 | 在当前 delta 图/文联合空间采样真实节点 | 严格外部控制 |
| Medoid coreset | 选择联合空间代表节点 | 当前 delta 内选择 medoid，文本随节点保留 | 严格外部控制 |
| Direct append / continual Student | 不做合成凝聚的持续学习控制 | 直接追加当前阶段数据并训练同一 Student，不输出 synthetic memory | 学习控制，不是凝聚基线 |
| C0 / X+P | 本项目已有方法控制 | 保留原始合成 memory 接口，用于测量 Adaptive coherent 的新增收益 | 内部 prior control |

以下方法共享 CaT 的 class-wise MMD X，只改变文本 payload 的材料化方式：nearest、chunk payload、graph-conditioned、graph-text aligned、CCTP top-3/top-5 和 class text-center。它们回答的是“CaT 的 X 对文本附着规则是否敏感”，属于 CaT 文本组件消融，不应和 GECC、K-center 或 Adaptive coherent 混合成外部方法排名。`class text-center` 让同类合成行共享一个 payload，只作为分析控制。

### 6.3 严格主表的最终组成

论文主表建议只放：`Adaptive coherent`、`CaT-MMD-TAG + CCTP top-5`、`GECC-hard-TAG`、`K-center/FPS`、`Medoid` 和 `Random real coreset`，并把 `C0/X+P` 与 `Direct append` 标为控制项。`PUMA-inspired` 单独放小图附表；TAGSAM、STAD、GraphCLIP、GCond、SFGC 和 GCondenser 放 reference/adapted 表，不与 strict 主表共用排名。

主表回答的是：在相同持续 TAG 访问边界、memory budget、Student 和评估协议下，Adaptive coherent 是否优于普通持续凝聚方法的 TAG 适配版本和强 coreset 控制；它不声称已经完成所有静态 TAG 方法的持续化复现。

本轮统一汇总和图表见：

- `results/expanded_baseline_comparison_20261006/summary.md`；
- `results/expanded_baseline_comparison_20261006/all_cells.csv`；
- `results/expanded_baseline_comparison_20261006/pooled_methods.csv`；
- `results/expanded_baseline_comparison_20261006/final_f1_strong_by_dataset.png`；
- `results/expanded_baseline_comparison_20261006/accuracy_strong_by_dataset.png`；
- `results/expanded_baseline_comparison_20261006/final_f1_cat_text_ablation_by_dataset.png`；
- `results/expanded_baseline_comparison_20261006/accuracy_cat_text_ablation_by_dataset.png`。

强基线表和组件消融表必须分开呈现。前者用于支持主方法相对于外部参照的结论，后者用于解释文本 payload 机制和 CaT-TAG 适配的失败边界。

### 6.4 为什么不能直接引用官方 SOTA 数字

静态 TAGSAM、GraphCLIP、GCond、SFGC 和 GCondenser 通常可以读取完整训练图，或者使用监督分类器/不同 Student。若把其官方数字直接放入本文主表，会同时改变历史 raw 访问、Student、memory budget 和任务定义。因此本文只在严格适配表中比较可复现的机制，官方结果作为 related work/reference-only，不声称实现了原论文端到端复现。

## 7 实验结果

### 7.1 独立顺序和 construction seed 的大图确认

下表为每个数据集 3 个类别顺序 × 3 个 construction seeds 的 9 个 cell 等权均值；括号内为 cell 间标准差。每个 cell 使用 6 个相同 Student seeds。

| 数据集 | 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 |
|---|---|---:|---:|---:|---:|---:|
| Arxiv | **Adaptive coherent** | **21.93 +/- 1.42** | **41.41 +/- 3.02** | 22.27 | 20.03 | 29.95 |
| Arxiv | Fixed coherent | 22.39 +/- 1.28 | 40.27 +/- 3.00 | 22.82 | 19.97 | 30.23 |
| Arxiv | CCTP top-5 | 18.93 +/- 0.89 | 29.86 +/- 2.94 | 19.06 | 18.20 | 30.95 |
| Arxiv | K-center/FPS | 17.50 +/- 0.49 | 26.98 +/- 1.85 | 17.17 | 19.37 | 29.74 |
| Arxiv | Medoid | 18.19 +/- 1.03 | 26.98 +/- 3.37 | 18.01 | 19.26 | 30.55 |
| SportsFit | **Adaptive coherent** | **47.31 +/- 1.95** | **64.51 +/- 6.64** | 46.66 | 50.93 | 52.83 |
| SportsFit | Fixed coherent | 48.34 +/- 1.95 | 65.06 +/- 5.79 | 47.79 | 51.37 | 54.57 |
| SportsFit | CCTP top-5 | 25.02 +/- 2.43 | 24.49 +/- 4.66 | 24.76 | 26.46 | 37.86 |
| SportsFit | K-center/FPS | 29.54 +/- 1.01 | 34.44 +/- 2.86 | 28.74 | 33.93 | 44.77 |
| SportsFit | Medoid | 29.04 +/- 0.93 | 34.70 +/- 3.00 | 29.20 | 28.16 | 43.48 |
| WikiCS | **Adaptive coherent** | **73.06 +/- 1.60** | **77.07 +/- 2.46** | 73.29 | 70.95 | **81.64** |
| WikiCS | Fixed coherent | 68.99 +/- 1.46 | 74.51 +/- 1.83 | 69.54 | 64.02 | 77.54 |
| WikiCS | CCTP top-5 | 50.44 +/- 2.28 | 54.03 +/- 3.17 | 51.24 | 43.25 | 67.57 |
| WikiCS | K-center/FPS | 50.29 +/- 2.70 | 53.41 +/- 3.88 | 50.49 | 48.51 | 70.70 |
| WikiCS | Medoid | 55.01 +/- 2.24 | 59.29 +/- 4.22 | 55.56 | 50.07 | 71.31 |
| BookHis | **Adaptive coherent** | **14.71 +/- 2.67** | **48.83 +/- 19.48** | 14.53 | 16.69 | 17.71 |
| BookHis | Fixed coherent | 14.99 +/- 3.10 | 47.24 +/- 19.64 | 15.30 | 11.58 | 17.91 |
| BookHis | CCTP top-5 | 14.74 +/- 2.94 | 26.74 +/- 11.22 | 13.92 | 23.69 | 39.49 |
| BookHis | K-center/FPS | **18.08 +/- 3.03** | 35.68 +/- 10.67 | **17.12** | **28.69** | **44.46** |
| BookHis | Medoid | 17.33 +/- 1.96 | 32.12 +/- 7.61 | 16.46 | 26.98 | 42.87 |

![大图结果](assets/adaptive_coherent/large_results.png)

**图 2.** 四个大图上独立顺序和 construction seed 的 Final Macro-F1。Adaptive coherent 在 Arxiv、SportsFit 和 WikiCS 上明显优于 CCTP；BookHis 上不优于 K-center/Medoid。

相对同一 cell 的 CCTP top-5，Adaptive coherent 的配对结果为：

| 数据集 | Final F1 差值 | 正收益 cell | 最差差值 |
|---|---:|---:|---:|
| Arxiv | +3.00 pp | 9/9 | +0.79 pp |
| SportsFit | +22.29 pp | 9/9 | +16.49 pp |
| WikiCS | +22.62 pp | 9/9 | +18.78 pp |
| BookHis | -0.03 pp | 5/9 | -8.07 pp |
| 四数据集合计 | +11.97 pp 等权 | 32/36 | -8.07 pp |

这组结果支持三个结论。第一，Arxiv、SportsFit 和 WikiCS 的正向信号不是单一顺序或单一 construction seed 造成的。第二，Adaptive slot 并不总是优于 fixed slot：WikiCS 上自适应规则明显有效，但 Arxiv 和 SportsFit 上 fixed coherent 略高。第三，BookHis 暴露出当前方法的主要边界：整体 Accuracy 较高，但 New F1 和阶段平均性能并不稳定，不能只用 Accuracy 宣称成功。

### 7.2 小图比较

小图使用三种类别顺序的等权均值，Student 和 memory 协议保持一致。当前小图 construction seed 的补充复现仍需在最终论文前完整归档，但三顺序结果已经可以作为方法不破坏 CaT 的初步证据。

| 数据集 | 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 |
|---|---|---:|---:|---:|---:|---:|
| Cora | **X-conditioned coherent routing** | **84.23** | **85.75** | **84.00** | **84.53** | **89.69** |
| Cora | Formal prompt-margin | 83.80 | 85.47 | 83.39 | 84.36 | 89.39 |
| Cora | CCTP/CaT nearest payload | 83.74 | 85.49 | 83.35 | 84.27 | 89.15 |
| Citeseer | X-conditioned coherent routing | 69.72 | 73.86 | 69.70 | 69.76 | 78.56 |
| Citeseer | **Formal prompt-margin** | **70.19** | **73.96** | **70.09** | **70.38** | **79.26** |
| Citeseer | CCTP/CaT nearest payload | 69.63 | 73.63 | 69.44 | 70.02 | 78.75 |

小图结果说明 X-conditioned routing 至少不会系统性破坏 CaT，但提升幅度很小，且 Citeseer 上 formal prompt-margin 略优。因此大图收益不能简单外推为所有规模都显著领先。

### 7.3 公开基线的 canonical 适配比较

下表为固定类别顺序/固定 construction seed 下的 6 Student seeds 结果，目的是展示方法与更多外部机制的对应关系。它与上面的 9-cell 独立确认表不同，不能混合做统计显著性结论。

| 数据集 | Adaptive | CCTP top-5 | GECC-hard-TAG | K-center | Medoid | Random real |
|---|---:|---:|---:|---:|---:|---:|
| Arxiv F1 / Acc | 25.59 / 50.70 | 24.27 / 48.39 | 20.23 / 31.43 | 19.43 / 30.69 | 18.06 / 24.71 | 17.81 / 24.66 |
| SportsFit F1 / Acc | 47.47 / 68.14 | 23.99 / 43.02 | 26.44 / 26.95 | 29.42 / 32.01 | 27.00 / 27.78 | 27.44 / 30.55 |
| WikiCS F1 / Acc | 73.87 / 77.82 | 54.30 / 58.55 | 51.11 / 53.22 | 50.24 / 53.19 | 54.97 / 59.03 | 54.03 / 58.22 |
| BookHis F1 / Acc | 17.67 / 60.18 | 17.35 / 30.03 | 19.25 / 34.48 | 20.04 / 38.01 | 17.01 / 30.21 | 18.68 / 33.59 |

补充控制项（只在已有 Arxiv/SportsFit strict control 结果覆盖的数据集报告）：

| 数据集 | Direct append F1 / Acc | C0 / X+P F1 / Acc | Adaptive coherent F1 / Acc |
|---|---:|---:|---:|
| Arxiv | 18.82 / 25.49 | 19.39 / 29.14 | 25.59 / 50.70 |
| SportsFit | 26.89 / 32.71 | 32.07 / 40.66 | 47.47 / 68.14 |

完整逐数据集、包含标准差、Old/New F1、最差阶段和构造成本的版本见 `results/expanded_baseline_comparison_20261006/summary.md`；其中 `baseline_adaptation_table.md` 逐项列出每个外部方法为满足 current-delta-only/frozen-prefix 所做的改动。

这张表显示：

- WikiCS 上 Adaptive coherent 相对最强 Medoid 仍有约 18.90 pp 的 F1 优势；
- SportsFit 上 Adaptive coherent 也明显高于这些适配基线；
- BookHis 上 Adaptive coherent 的 Accuracy 很高，但 F1 低于 K-center 和 GECC，New F1 只有 1.23；
- 因此 pooled 平均不能替代逐数据集结论，也不能把 BookHis 的多数类偏置隐藏起来。

#### 7.3.1 四个大图的严格主表

下面的表只放协议匹配的强基线和控制方法。单元格为 `Final Macro-F1 / Accuracy`，均来自相同 `base+4`、5% cumulative memory、6 个 Student seeds 和 frozen-prefix 规则。`C0/X+P` 与 `Direct append` 只在已有严格控制结果覆盖的 Arxiv、SportsFit 上报告，因此其余数据集使用 `—`，不把缺失结果当成零分。

| 数据集 | **Adaptive coherent** | CaT-MMD-TAG + CCTP top-5 | GECC-hard-TAG | K-center/FPS | Medoid | Random real | C0 / X+P | Direct append |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arxiv | **25.59 / 50.70** | 24.27 / 48.39 | 20.23 / 31.43 | 19.43 / 30.69 | 18.06 / 24.71 | 17.81 / 24.66 | 19.39 / 29.14 | 18.82 / 25.49 |
| SportsFit | **47.47 / 68.14** | 23.99 / 43.02 | 26.44 / 26.95 | 29.42 / 32.01 | 27.00 / 27.78 | 27.44 / 30.55 | 32.07 / 40.66 | 26.89 / 32.71 |
| WikiCS | **73.87 / 77.82** | 54.30 / 58.55 | 51.11 / 53.22 | 50.24 / 53.19 | 54.97 / 59.03 | 54.03 / 58.22 | — | — |
| BookHis | 17.67 / **60.18** | 17.35 / 30.03 | **19.25 / 34.48** | **20.04 / 38.01** | 17.01 / 30.21 | 18.68 / 33.59 | — | — |

主方法在 Arxiv、SportsFit 和 WikiCS 的 Final Macro-F1 均高于严格适配的 CaT、GECC 和 coreset 参照；BookHis 则不是 F1 最优，只在 Accuracy 上出现明显升高。这个差异必须作为长尾新类塑性风险报告，而不能用 Accuracy 单独包装成全面优势。

四个大图的等权汇总如下。该汇总只作为整体趋势，不替代逐数据集结果，因为四个数据集的类别数、文本质量和长尾程度不同。

| 方法 | 覆盖数据集 | Final Macro-F1 | Accuracy | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|
| **Adaptive coherent** | Arxiv, SportsFit, WikiCS, BookHis | **41.15** | **64.21** | 43.59 | 24.70 |
| CaT-MMD-TAG + CCTP top-5 | 四个大图 | 29.98 | 45.00 | 31.55 | 18.38 |
| K-center/FPS | 四个大图 | 29.78 | 38.47 | 31.41 | 18.04 |
| GECC-hard-TAG | 四个大图 | 29.26 | 36.52 | 30.71 | 18.59 |
| Medoid | 四个大图 | 29.26 | 35.43 | 30.89 | 16.80 |
| Random real | 四个大图 | 29.49 | 36.75 | 30.75 | 20.57 |

这个 pooled 结果显示主方法整体有较大正向差异，但主要贡献来自 SportsFit 和 WikiCS；论文结论仍以逐数据集的强基线表和 Old/New F1 为准。

![严格协议强基线 Final Macro-F1](assets/adaptive_coherent/final_f1_strong_by_dataset.png)

**图 3.** 四个大图上 Adaptive coherent 与协议匹配强基线的 Final Macro-F1。橙色为主方法，误差线为 6 个 Student seeds 的标准差。

![严格协议强基线 Accuracy](assets/adaptive_coherent/accuracy_strong_by_dataset.png)

**图 4.** 四个大图上的 Accuracy。BookHis 中主方法的 Accuracy 明显高于其它方法，但这与其低 New F1 同时出现，说明多数类偏置不能被忽略。

#### 7.3.2 Old/New F1 诊断

| 数据集 | 方法 | Old F1 | New F1 | 解释 |
|---|---|---:|---:|---|
| Arxiv | Adaptive coherent | 28.59 | 8.59 | 旧类保持和新类性能均高于 CaT+CCTP 的对应结果 |
| Arxiv | CaT+CCTP top-5 | 27.54 | 5.76 | 作为最直接的文本凝聚参照 |
| SportsFit | Adaptive coherent | 51.73 | 24.03 | 总体 F1 和 Accuracy 均明显领先，New F1 也略高于 C0 |
| SportsFit | C0/X+P | 33.78 | 22.71 | 旧类保持较弱，说明主方法收益不只是旧类偏置 |
| WikiCS | Adaptive coherent | 74.86 | 64.94 | 旧类和新类同时改善，是最完整的正向数据集 |
| WikiCS | Medoid | 57.43 | 32.79 | 强 coreset 仍明显落后于主方法 |
| BookHis | Adaptive coherent | 19.17 | 1.23 | 新类塑性失败，不能只依据 60.18 Accuracy 判断成功 |
| BookHis | K-center/FPS | 21.08 | 8.59 | BookHis 上更好的外部 F1 参照 |

![Old/New F1 trade-off](assets/adaptive_coherent/old_new_tradeoff.png)

**图 5.** Old F1 与 New F1 的 trade-off。BookHis 的主方法点位显示出明显的新类塑性异常。

### 7.4 CaT 文本组件消融

下面的实验固定 CaT 的 `X` 优化、memory 行数、Student 和阶段协议，只改变文本 payload 的构造方式。因此它们不是新的外部强基线，而是用于解释 CaT-TAG 适配中“文本怎么附着”这一组件的消融。完整四数据集结果已经写入 `results/expanded_baseline_comparison_20261006/summary.md` 和 `all_cells.csv`。下表给出每个组件的 `Final Macro-F1 / Accuracy`，完整的 Old/New F1、阶段平均值和构造时间见汇总文件。

需要区分 construction seed：强基线 frozen panel 使用 `270927`，本轮 CaT 文本组件消融使用独立的 `314159`。两者的阶段协议、memory budget、历史 raw 访问边界和 Student seeds 相同，但 construction seed 不同。因此下表用于机制诊断和跨数据集趋势分析，不把它写成与强基线逐 cell 配对的显著性结论。

| 数据集 | nearest | TAGSAM-style chunk | graph-conditioned | graph-text aligned | class text-center | CCTP top-3 |
|---|---:|---:|---:|---:|---:|---:|
| Arxiv | 19.52 / 30.54 | 19.96 / 27.39 | 19.91 / 30.26 | 19.01 / 28.56 | 23.63 / 48.21 | 20.13 / 30.02 |
| SportsFit | 30.58 / 27.59 | 27.32 / 28.35 | 28.89 / 27.75 | 28.65 / 25.10 | 21.73 / 35.01 | 30.65 / 37.30 |
| WikiCS | 51.24 / 54.83 | 38.82 / 40.31 | 38.27 / 42.32 | 25.14 / 26.54 | 65.37 / 69.39 | 50.86 / 55.90 |
| BookHis | 17.70 / 32.13 | 14.57 / 26.27 | 15.96 / 31.10 | 18.22 / 32.19 | 7.75 / 14.26 | 15.53 / 23.63 |

![CaT 文本组件 Final Macro-F1](assets/adaptive_coherent/final_f1_cat_text_ablation_by_dataset.png)

**图 6.** 固定 CaT 的 X、memory 和 Student，只改变文本 payload 处理方式时的 Final Macro-F1。该图是组件消融，不是六个独立外部强基线的排名。

![CaT 文本组件 Accuracy](assets/adaptive_coherent/accuracy_cat_text_ablation_by_dataset.png)

**图 7.** CaT 文本组件的 Accuracy。class text-center 在 WikiCS/部分数据集上 Accuracy 较高，但它让同一类别所有合成行共享一条文本，必须单独解释，不能替代 node-specific payload 的主方法。

![Adaptive coherent relative delta heatmap](assets/adaptive_delta_heatmap.png)

**图 8.** Adaptive coherent 相对于强基线的 Final-F1 差值热图。绿色表示主方法更高，红色表示主方法更低；BookHis 的负向单元与 New F1 下降相互印证。

这一层实验说明：CaT 的 X 优化本身不能推出唯一正确的文本材料化规则；文本粒度、来源多样性和类内重复会显著改变 Prompt Student 的结果。但 `class text-center` 的高分不能直接当成强基线优势，因为它让同一类别所有合成行共享一个 payload，可能利用类别共享文本带来的先验，并不等价于保留节点级 TAG 信息。

### 7.5 内部消融和负结果

当前方法是多轮实验收敛出来的结果，而不是一次性把多个启发式模块叠加。重要对照包括：

| 版本 | 改变内容 | 主要观察 |
|---|---|---|
| CaT/CCTP | CaT X + 类别内 top-5 真实文本附着 | 稳定、简单，是最重要 paired baseline |
| Fixed coherent | 固定 slot 数 | Arxiv/SportsFit略高，WikiCS低于 adaptive |
| Adaptive coherent | 图文离散度决定 1/2/3 slot | 大图 3/4 数据集有稳定收益 |
| Adaptive + CCTP anchor | 候选必须优于 CCTP 才替换 | 可避免部分灾难性退化，但总体收益更保守 |
| 逐行 chunk payload | 每个合成行独立拼接文本 chunk | 小图明显退化，说明类内文本方差过大 |
| 无约束 coherent routing | 直接接受 X-conditioned candidate | BookHis 可降到接近随机，说明局部代价不等价于长期 Student 价值 |
| 双原型类共享 | 每类简单增加第二个文本原型 | Cora/Citeseer 均下降，说明无约束多原型会重新引入类内分裂 |
| formal prompt-margin | 每类一个 prompt-aware payload | 小图较稳，但不是多 payload 主方法，跨大图不稳定 |

这些负结果的重要意义是：当前 Student 并不是“文本越多越好”。在固定 memory 下，同一类别内的 payload 方差、文本来源和 X 的粒度必须同时受控。

## 8 讨论

### 8.1 主要创新点

当前方法的创新性应分为“问题创新”和“算法创新”两层评价。

**问题层面：** 历史 raw graph 不可回读条件下的持续 TAG 凝聚，和静态 TAG condensation、普通持续图学习都不同。它要求在每个阶段只使用当前 raw delta 构造图文 memory，并处理旧 memory 访问边界、文本来源合法性、预算和新旧阶段组合。这个问题设定本身具有明确应用价值，且目前公开工作中针对这一完整协议的系统研究仍较少。

**算法层面：** Adaptive coherent 不是重新发明 CaT 的合成特征目标，而是对 CaT 的文本附着阶段做结构化改进。其相对简单 CCTP 的新增机制是：

1. 用当前类别的图文离散度决定 payload slot 粒度，而不是固定一个类别一个文本或固定三个文本；
2. 候选来自当前 raw delta 的真实材料化文本，不使用不可审计的连续文本向量作为最终 payload；
3. 用合成特征 X 对类别内候选做可微 assignment，使不同合成行可以根据自身 X 使用不同真实文本；
4. 用硬材料化、来源 ID、字符 span 和访问审计保证构造结果可复现；
5. 通过对逐行文本、双原型、anchor 和 BookHis 失败的系统实验，说明类内文本方差是持续 TAG Student 中的关键瓶颈。

但必须客观指出：当前版本没有对合成拓扑做创新，也没有 Student-response outer-loop 或旧 memory revision。它更准确的定位是“CaT-based continual TAG condensation 的图文 payload routing 算法”，而不是一个完整解决 X、文本、拓扑和旧记忆联合优化的最终框架。

### 8.2 当前结果是否足够发 WWW 长文

当前结果已经具备长文候选的基础，但还不足以稳妥地宣称强贡献或普遍 SOTA。

**有利因素：**

- 问题设定清晰，历史 raw 不可回读具有现实约束；
- 方法流程可实现、可审计，并且与 CaT 有清楚的接口；
- 四个大图中三个数据集对 CCTP 有稳定收益，36 个配对 cell 中 32 个为正；
- 公开基线适配、F1/Accuracy、Old/New F1、成本和失败结果已经开始形成完整实验故事；
- 方法不是简单把某个数据集的测试 F1 用于调参，slot 规则依赖当前 delta 的相对离散度。

**主要风险：**

- BookHis 上 F1 和 New F1 失败，当前方法并非跨数据集稳定优于 coreset；
- 小图提升有限，说明方法收益更依赖大图文本和类别结构；
- 外部 CaT/GECC 等结果是严格协议下的机制适配，而非官方代码原样复现；
- 当前没有拓扑学习，完整 TAG condensation 的贡献边界需要在论文标题和摘要中说清楚；
- 当前 routing 的可微优化只作用于 assignment logits，并没有把下游 Student loss 反传到 X 或文本生成器；
- 还缺少第二个下游任务、跨数据集 zero-shot、正式 paired confidence interval 和更完整的小图 construction seed 复现。

因此，当前最客观的判断是：**作为 WWW 长文候选，贡献基础中等偏上；作为已经足够稳妥的接收版本，还需要补齐泛化、统计和失败边界实验。** 不能仅凭 WikiCS/SportsFit 的高分宣称 SOTA。

### 8.3 后续最重要的改进

后续不建议再次同时修改 X、文本、拓扑、Student 和旧 memory。优先级如下：

1. **补齐统计证据。** 完成小图 construction seeds `314159`、`271828`；计算正式 95% paired CI、positive-cell ratio 和 worst cell。
2. **诊断 BookHis。** 报告每阶段每类 New F1、class-wise recall、预测类别分布、slot coverage、文本重复率和类别 raw 文本离散度，判断失败来自 payload collapse、类别不平衡还是 Student prompt 偏置。
3. **统一主方法命名。** 小图 fixed coherent 与大图 adaptive coherent 应形式化为同一个 `X-conditioned coherent slot routing` 框架，adaptive slot 作为由离散度驱动的 slot policy；如果实现仍然不同，不能在论文中混叫一个方法。
4. **补一个受限拓扑组件。** 只在当前 delta 和新增 block 内增加可审计的 block-internal topology，保持 edge budget 和 old prefix 不变；先做 topology-free、CCTP topology、X-conditioned topology 三臂，避免把拓扑和文本同时改到无法归因。
5. **增加第二下游任务。** 优先做 link prediction 或 edge classification，但必须重新定义 synthetic edge、负采样和 historical raw 访问边界。若资源有限，至少增加 prompt zero-shot 的跨顺序和跨数据集测试。
6. **补充外部公开方法说明。** 明确哪些是 strict adapted baseline，哪些只是 reference-only；不要把 static TAGSAM、GraphCLIP 或监督 Student 的官方数字直接放入主榜。
7. **报告成本。** 构造时间、文本编码调用、GPU memory、Student 时间和与 full raw/reference 的比例必须进入主表或附录。

### 8.4 当前是否应继续优化

可以继续优化，但应保持“单组件、可归因、可回退”的原则。最值得尝试的不是继续增加 selector，而是针对 BookHis 做一个预注册的 class-floor 或 new-class protected routing：每个新类至少获得一个合法 slot 和最低 payload coverage，旧类的 routing 只能在固定预算内调整。该改动应只作用于 routing policy，不修改 CaT X、Student 和拓扑；如果不能在两个以上数据集平均改善 New F1，就将其作为失败边界而不是主方法。

## 9 结论

本文提出并验证了 Adaptive coherent，一种在历史原始图不可回读的持续 TAG 凝聚协议下运行的 X-conditioned coherent adaptive slot routing 方法。它保留 CaT 的类别级 MMD 特征凝聚和阶段追加机制，在当前阶段真实文本中建立有限候选池，根据图文离散度分配 slot，并用合成特征 X 对真实文本 payload 做类别内路由和硬材料化。

实验表明，Adaptive coherent 在 Arxiv、SportsFit 和 WikiCS 上相对 CCTP top-5 有稳定收益，在四个大图的 36 个配对 cell 中有 32 个正向结果；但 BookHis 上仍存在明显的新类塑性问题，说明当前方法尚未解决长尾持续 TAG 的全部挑战。因而，本文不主张“全面 SOTA”，而主张一个更准确的结论：在严格 current-delta-only、frozen-prefix 和固定 memory 协议下，图文 payload 的类别内粒度与 X-conditioned routing 是影响持续 TAG 凝聚性能的重要因素。

## 参考文献（工作版）

1. CaT: Balanced Continual Graph Learning with Graph Condensation. arXiv:2309.09455.
2. GECC: Scalable and Structure-free Graph Condensation with Evolving Capabilities.
3. PUMA: Efficient Continual Graph Learning for Node Classification with Graph Condensation.
4. TAGSAM: static text-attributed graph condensation reference implementation.
5. GraphCLIP: graph-text representation alignment and TAG condensation reference.
6. Reimers, N. and Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP, 2019.
7. Gonzalez, T. F. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science, 1985.

正式投稿前需要补齐上述工作的标准 BibTeX、发表 venue、版本号和官方实现链接，并在 related work 中准确区分 static TAG condensation、continual graph condensation、continual learning control 和本文 strict continual TAG protocol。

## 附录 A：复现和审计文件

| 内容 | 文件 |
|---|---|
| 主方法实现 | `code/v2_llm_fusion/x_conditioned_payload_routing.py` |
| CaT-style 多模态基础 | `code/v2_llm_fusion/cat_multimodal_minimal.py` |
| 大图统一 runner | `code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py` |
| 小图 Prompt Student runner | `code/v2_llm_fusion/run_prompt_zero_shot_small_20261006.py` |
| 独立顺序/seed 汇总 | `docs/current/ADAPTIVE_COHERENT_ORDER_SEED_RESULTS_ZH_2026-10-06.md` |
| 基线适配说明 | `docs/current/EXTERNAL_BASELINE_METHOD_NOTES_20261006.md` |
| 外部基线对比 | `docs/current/EXTERNAL_BASELINE_COMPARISON_WIKICS_BOOKHIS_20261006.md` |
| 大图协议 | `docs/current/ADAPTIVE_COHERENT_ORDER_SEED_PROTOCOL_ZH_2026-10-06.md` |
| 小图 F1/Accuracy | `docs/current/MULTI_DATASET_F1_ACCURACY_COMPARISON_20261006.md` |
| 四数据集强基线与文本消融汇总 | `results/expanded_baseline_comparison_20261006/summary.md` |
| 逐 cell CSV | `results/expanded_baseline_comparison_20261006/all_cells.csv` |
| 基线适配卡片 | `results/expanded_baseline_comparison_20261006/baseline_adaptation_table.md` |
| 强基线 F1/Accuracy 图 | `docs/paper/assets/adaptive_coherent/final_f1_strong_by_dataset.png`；`accuracy_strong_by_dataset.png` |
| 文本消融 F1/Accuracy 图 | `docs/paper/assets/adaptive_coherent/final_f1_cat_text_ablation_by_dataset.png`；`accuracy_cat_text_ablation_by_dataset.png` |
| Old/New trade-off 与差值热图 | `docs/paper/assets/adaptive_coherent/old_new_tradeoff.png`；`docs/paper/assets/adaptive_delta_heatmap.png` |

## 附录 B：当前结论边界

- 不把 Adaptive coherent 描述为所有数据集 SOTA；
- 不把 CaT/GECC 适配结果描述为官方端到端复现；
- 不把 Prompt zero-shot Student 的结果描述为监督分类 Student 结果；
- 不把 edge-free/self-loop-only 实验描述为完整拓扑凝聚；
- 不把 9-cell 确认结果和 canonical 单顺序结果混合计算；
- BookHis 的 Accuracy 与 New F1 必须同时展示；
- 在补齐正式 paired confidence interval、BookHis 类别级诊断和至少一个额外泛化任务前，主方法仍应称为“当前候选主方法”，不能写成无条件的全面 SOTA。
