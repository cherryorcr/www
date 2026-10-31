# Unified-Q：历史 raw 不可回读条件下的持续文本属性图凝聚

**论文草稿版本：2026-10-09**  
**证据状态：sealed test 已完成；主方法定位为 In-loop CaT-Q 候选，不宣称普遍 SOTA。**

## 摘要

文本属性图（Text-Attributed Graph，TAG）凝聚不仅要压缩节点特征和图结构，还要保留可以被下游图文模型重新利用的文本信息。在持续场景中，系统按阶段接收新节点；阶段结束后历史 raw graph 不可回读，只能访问当前增量和已经保存的 synthetic memory。这个访问约束使静态 TAG 凝聚方法不能直接作为持续算法使用：方法必须决定新数据如何进入固定压缩率或固定阶段预算的 memory、旧 synthetic rows 是否保持、文本如何和合成特征对应，以及构造出的 memory 是否能迁移到不同的下游 Student。

本文整理并验证 Unified-Q，一种以 class-masked differentiable assignment matrix (Q) 为共同变量的持续 TAG 凝聚方法。(Q) 同时决定合成特征的加权池化、真实文本 payload 的软路径和最终硬材料化来源；构造阶段使用 in-loop gradient matching、one-hop propagation matching 和有界的特征 residual，避免先独立完成 CaT 再把文本作为事后附加物。最终保存的文本来自当前增量中的真实文本，并与训练前向中的 hard straight-through 路径保持一致。当前冻结版本采用 edge-free memory，旧 synthetic prefix 不更新；拓扑 revision、EM 和 joint-distribution extension 不属于本轮主方法。

在 WikiCS、SportsFit 和 BookHis 上，我们使用 3 个类别顺序、3 个 construction seeds、3 种 Prompt Student，并在 `original_id % 10 == 2` 的 sealed test 上复用冻结 memory。共完成 27 个构造 cells、4 个 memory arms、324 个 readout rows 和 972 次 Student 训练。结果显示，Unified-Q 在 BookHis 上相对 CaT 适配平均提升 9.03--22.23 个百分点，在部分 WikiCS/SportsFit 架构上与强基线接近或略优；但它没有稳定超过 GECC，也不能声称跨数据集 SOTA。本文因此将贡献定位为一个可审计的持续 TAG 设定、一个共享 (Q) 的图文联合构造机制，以及对跨数据集、跨 Student 正负边界的系统分析，而不是“所有条件下都最优”的方法。

**关键词：** 文本属性图；持续图凝聚；历史 raw 不可回读；可微 assignment；gradient matching；图文对齐；zero-shot prompt readout

## 1 引言

图凝聚通常通过少量 synthetic nodes 和 edges 重新训练下游模型。对于 TAG，memory 中还要保存文本或文本 embedding，因此压缩对象是一个联合接口：

\[
M_t=(X_t,T_t,A_t,Y_t).
\]

静态凝聚可以反复访问完整原图，最后一次性优化所有 synthetic variables。持续 TAG 凝聚则不同：在阶段 (t) 只能访问当前增量 (Delta_t) 和之前保存的 synthetic memory (M_{t-1})，不能再次读取历史 raw nodes、raw texts 或 raw edges。该约束带来四个核心问题：

1. 新旧信息如何在固定 memory budget 中分配；
2. 一个 synthetic slot 应该吸收哪些当前 raw nodes；
3. 合成特征和文本 payload 如何保持一致；
4. memory 是否只对一个 Student 有效，还是能够迁移到不同图编码器。

早期版本分别尝试了静态 CaT-style feature condensation、简单文本拼接、semantic-recursive append 和拓扑 revision。它们提供了有用的负结果：文本事后附着可能与合成 (X) 不一致，固定 class-shared payload 会造成严重文本重复，动态 topology 和 response selector 会引入较强的协议依赖。因此当前主线不再把多个启发式模块叠加，而是把“节点分配、特征池化、文本选择和 Student-compatible objective”收束到一个共享的 (Q) 中。

本文的贡献边界如下：

1. **问题设定。** 明确历史 raw 不可回读、当前 delta-only、冻结 synthetic prefix 和 cumulative fixed-rate memory 的持续 TAG 凝聚协议，并通过 sealed split 审计数据访问和 memory hash。
2. **共享 assignment。** 用 class-masked Sinkhorn (Q) 把当前 raw nodes 运输到新 synthetic slots，(Q) 同时影响 (X)、文本和构造代理目标，而不是先完成 CaT 后再附加文本。
3. **可材料化图文路径。** 训练时保留 soft text route 的梯度，前向使用对应 hard source text 的 straight-through 表示，最终保存同一规则选出的真实文本并重新编码。
4. **跨 Student 证据。** 在 Prompt-GCN、Prompt-SGC 和 Prompt-GraphSAGE 上进行相同 memory 的 sealed readout，报告 Macro-F1、Accuracy、Old/New F1 和负结果。

本文不主张普遍超过 GECC，不将 CaT、GECC 或 PUMA 的协议适配结果称为官方 native reproduction，也不把当前 edge-free 版本称为完整的拓扑凝聚算法。

## 2 相关方法与比较身份

### 2.1 CaT 与本文的关系

CaT-style baseline 通过传播后的特征统计或 MMD 类目标优化 synthetic feature。本文借鉴的是它的 class-wise distribution matching 思想和以合成 (X) 作为 Student 输入的路线，但没有把 CaT 作为预训练初始化后再独立优化文本。Unified-Q 从当前阶段的 (Q_0) 开始，并在同一个 in-loop objective 中更新 (Q) 与特征 residual。

### 2.2 Gradient matching 与 Prompt Student

本文构造代理使用一个轻量 GraphTextPromptStudent，计算真实参考集合和合成 memory 对同一模型参数产生的梯度，再匹配梯度方向和尺度。最终 readout 使用 Prompt-GCN、Prompt-SGC 或 Prompt-GraphSAGE。Student 训练时使用 memory labels 计算 pair loss、prompt cross-entropy 和 margin loss；预测时没有训练分类 head，而是将图表示和类别 prompt 表示做 cosine/logit 匹配。因此更准确的表述是 **prompt-based zero-shot readout**，不是完全无监督的训练过程，也不是严格 inductive zero-shot。

### 2.3 基线身份

| 方法 | 类型 | 当前实验中的适配 | 论文中的证据等级 |
|---|---|---|---|
| In-loop CaT-Q | 本文方法 | 共享 (Q)、in-loop gradient matching、propagation matching、bounded (X) residual、真实文本 hard materialization；edge-free、旧 prefix 冻结 | 主方法候选 |
| Q-medoid | 内部控制 | 只把 (Q)-max 文本来源替换为同类 Q-weighted medoid，其余构造不变 | 文本材料化消融 |
| CaT + nearest5 | CaT 机制适配 | current-delta-only、冻结旧 prefix、同 memory budget；CaT-style (X) condensation 后按 graph-nearest 选择 5 条文本 | 严格机制适配基线，非官方复现 |
| GECC-hard + TAG | GECC 机制适配 | hard hop-weight propagation、Lloyd-style clustering、current-delta-only、nearest5 TAG text | 严格机制适配基线，非官方复现 |
| PUMA-inspired TAG | 外部机制适配 | pseudo-label/confidence/one-step propagation 与统一 TAG Student、预算和访问边界对齐 | 补充适配结果 |
| k-center/FPS、medoid | 真实节点 coreset | 合法 current pool 内选择真实节点，使用同一 Student 评估 | 简单强参照 |
| TAGSAM、STAD | 静态或补充 TAG 方法 | 保留原方法机制并单独标记访问协议、Student 和 order 差异 | supplementary/reference |
| GCond、SFGC、GCondenser、官方 GraphCLIP checkpoint | reference-only | 仅核验论文/代码机制或官方模型推理，不伪造同协议持续结果 | 不进入严格主表 |

这里的“严格机制适配”表示相同 memory rows、文本编码器、Student 训练步数、类别顺序和评估划分；它不表示复现了原论文的全部数据访问、Student 或 native training pipeline。

## 3 持续 TAG 协议

### 3.1 访问约束

阶段 (t) 的合法输入为：

\[
\Delta_t=(X_t^{raw},T_t^{raw},Y_t^{raw},A_t^{raw}),
\qquad M_{t-1}.
\]

构造器不能读取历史 raw graph。当前 sealed 主方法实际采用 `use_boundary=False` 和 edge-free synthetic memory，因此旧 memory 只以 frozen prefix 的 (X,T,Y) 形式进入 Student-compatible proxy；旧 rows 不通过 `old_update_radius` 更新。

### 3.2 Budget 与类别分配

协议的预算原则使用 cumulative fixed-rate：

\[
K_t=\left\lfloor 0.05N_t^{pool}\right\rfloor,
\qquad
m_t=K_t-|M_{t-1}|.
\]

新 slots 按当前 delta 的类别比例分配，每个当前类别在预算可行时至少得到一个 slot。这个分配不是固定“每类三个”或“每类五个”，也不是根据 sealed test 调参。

sealed readout 不重新改变预算，而是直接复用已冻结构造 cell 的阶段预算。三张大图最终 rows 分别为 WikiCS 410、SportsFit 6057 和 BookHis 1454；因此 sealed test 比较的是同预算 memory，而不是重新搜索 budget。

### 3.3 评价

报告最终 Macro-F1、Accuracy、Old F1、New F1。Macro-F1 按当前已见类别等权，Accuracy 是 report/test 节点上的整体正确率。构造和方法选择不读取 sealed test；本轮 sealed test 使用 `original_id % 10 == 2`，冻结 memory 直接复用并做 archive hash 校验。

## 4 方法：Edge-free In-loop CaT-Q

### 4.1 方法总览

当前阶段的步骤是：

1. 统计当前 delta 的类别规模，决定新增 slots 及其标签；
2. 对当前真实文本做统一清理和最多 320 字符截断，再用冻结 MiniLM 编码；
3. 用当前节点特征、一跳消息和文本构造 class-aware slot prototypes；
4. 构造 class-masked 初始 assignment logits，并用 Sinkhorn 得到 (Q_0)；
5. 在 (Q) 和 (X) residual 的同一优化循环中计算真实侧/合成侧 Student proxy loss 及梯度匹配；
6. 用 (Q) 加权得到 synthetic (X)，用 straight-through route 连接 soft text 和 hard source text；
7. 施加 propagation matching、(Q) anchor、(X) trust region 和旧 prefix anchor；
8. 最后按同一 (Q)-max 规则选择当前真实文本，重新编码并保存 memory。

当前 sealed 主版本没有合成边：(A_t^{syn}=\varnothing)。因此它验证的是图文 memory 构造和跨 Student 迁移，而不是 topology revision 的最终效果。

### 4.2 Slot prototype 与初始 (Q_0)

对 raw feature、文本 embedding 和当前一跳消息分别归一化。`diverse_message_text` 初始化在每个类别内部，用 max-min 方式选择分散的代表行。第一个代表接近类中心，后续代表最大化与已选代表的最小距离。该过程只需要当前类内的向量和，不遍历全图的所有 raw pairs。

对于当前 raw row (i) 和 synthetic slot (j)，初始相似度为：

\[
s_{ij}=1.0\,\cos(x_i,p^x_j)
 +0.75\,\cos(h_i,p^h_j)
 +0.35\,\cos(e_i,p^t_j),
\]

其中 (h_i) 是当前 raw graph 的一跳归一化消息，(e_i) 是 MiniLM 文本 embedding。不同类别之间的 logits 直接屏蔽；这不是把其他类别设置成“很小但仍可用”的候选，而是 class mask 后在每个类别块内部单独做 transport。

### 4.3 Class-masked Sinkhorn assignment

对类别 (c) 的 (n_c) 个 raw rows 和 (m_c) 个 slots，(Q_c) 满足：

\[
Q_c 1=1,
\qquad
Q_c^\top 1=\frac{n_c}{m_c}1,
\qquad
Q_{ij}=0\quad (y_i\ne y_j).
\]

第一条保证每个 raw row 的质量被完整分配；第二条保证同一类别的 slot 获得相同总质量；第三条避免跨类别泄漏。Sinkhorn 在 log-space 中迭代，并对大矩阵使用 checkpointing 降低 activation memory。(Q) 不是最终文本索引，而是将 raw information 运输到 synthetic slots 的可微权重矩阵。

### 4.4 (Q) 生成 synthetic (X)

令 (m_j=\sum_iQ_{ij})，当前主方法的 pooled feature 为：

\[
\bar x_j(Q)=\frac{\sum_iQ_{ij}x_i}{m_j}.
\]

为了保留有限的 proxy gradient 修正，引入 residual (r_j)：

\[
x_j^{syn}=\bar x_j(Q)+r_j.
\]

每轮将 residual 投影到：

\[
\|r_j\|_2\leq 0.05\|\bar x_j(Q)\|_2.
\]

所以 (Q) 决定主要内容，residual 只能做受限修正，不会把 CaT-style feature geometry 完全改写。

### 4.5 文本处理与硬一致性

soft text route 为：

\[
\tilde e_j(Q)=\operatorname{norm}\left(
\frac{\sum_iQ_{ij}e_i}{m_j}
\right).
\]

但是最终 memory 不能保存一个无法解释的任意 embedding。主方法因此采用 straight-through hard path：前向使用 (Q) 最大来源的真实文本 embedding (e_{i^*})，反向使用

\[
e_j^{forward}=e_{i^*}+(\tilde e_j-\mathrm{sg}(\tilde e_j)).
\]

这样，Student proxy 看到的内容是某个真实文本，梯度仍能经过 (	ilde e_j) 回到 (Q)。最终也使用相同的 (i^*=\arg\max_iQ_{ij}) 选择真实文本字符串，按相同 320 字符规则处理并重新用 MiniLM 编码。审计要求 hard forward embedding 与最终保存 embedding 的最大误差小于 (10^{-4})。

`Q-medoid` 是只替换这一硬选择规则的内部控制，不能和主方法混称；sealed 结果中它与主方法接近，说明当前收益不能只归因于 hard text materialization。

### 4.6 Student-compatible gradient matching

构造 proxy 采用同一个轻量 GraphTextPromptStudent。对真实参考集合和 synthetic memory 分别计算：

\[
L_{proxy}=0.5L_{pair}+L_{prompt}+0.1L_{margin},
\]

其中 (L_{pair}) 是 symmetric multi-positive graph-text InfoNCE，(L_{prompt}) 是对类别 prompt 的 cross-entropy，(L_{margin}) 是正确类别与最强错误 prompt 的 margin loss。

令 (g_r) 和 (g_s) 为真实侧和 synthetic 侧对 proxy Student 参数的梯度，梯度匹配距离为：

\[
L_{GM}=1-\cos(g_r,g_s)
 +0.25\left|\log\|g_r\|-
 \log\|g_s\|\right|.
\]

当前主版本设置 (lambda_{GM}=1)。因此这里确实使用了梯度优化，但梯度主要更新构造变量 (Q) 的 logits 和 (r_j)，不是在最终评估时把 memory 当作可训练 Student 参数。

### 4.7 CaT-style distribution matching、propagation matching 与完整损失

主方法还启用了 CaT-style 的非线性类别分布匹配。对当前 raw 一跳消息使用固定随机投影 bank：

\[
\phi_b(h)=\operatorname{norm}(\operatorname{ReLU}(hW_b)).
\]

令 (R_c^b) 是当前 raw 类别 (c) 的均值，(S_c^b) 是 synthetic side 对应类别的均值，则：

\[
L_{DM}=\frac{1}{B}\sum_b\sum_c
\left\|S_c^b-R_c^b\right\|_2^2.
\]

sealed 主方法的 (lambda_DM=1)。这部分是对 CaT 分布匹配思想的 in-loop 适配，不是先优化完 CaT 再单独改文本。

真实当前 delta 的一跳消息作为目标。由于 sealed 主方法的 synthetic memory 是 edge-free，synthetic side 没有非自环传播；实现中的归一化邻接仍保留 self-loop，因此新 slot 的 synthetic message 主要退化为自身特征，而不是历史邻居聚合。propagation matching 明确惩罚“没有合成拓扑时无法恢复的一跳传播”。令 (h_j^{target}) 是 (Q) 池化后的目标消息，(h_j^{syn}) 是 edge-free synthetic side 的 self-loop message：

\[
L_{prop}=\frac{1}{m}\sum_j
\|h_j^{target}-h_j^{syn}\|_2^2,
\qquad h_j^{syn}\approx x_j^{syn}.
\]

主版本的总目标为：

\[
L=
L_{GM}+L_{DM}+0.5L_{prop}
 +0.1L_Q+0.1L_X+0.1L_{old}.
\]

其中 (L_Q=\|Q-Q_0\|_F^2)，(L_X) 是按 pooled feature 归一化的 residual penalty，(L_{old}) 在旧 prefix 允许更新的配置中约束旧 residual；sealed 主方法的 `old_update_radius=0`，因此旧 rows 实际冻结，(L_{old}) 为零。

本轮没有打开 joint pair-distribution MMD、EM alternating inference 或 topology loss。它们是后续候选或负结果，不应在当前主方法名称中隐藏。这里的 (L_DM) 是主方法的一部分，而 joint distribution matching 是单独的诊断扩展。

### 4.8 伪代码

```text
for stage t:
    read current delta and frozen synthetic prefix only
    allocate new slot labels by current class mass
    encode current texts with frozen MiniLM
    build class-masked initial logits from X, one-hop message and text
    Q0 = masked_sinkhorn(initial_logits)
    initialize Q_logits = log(Q0), residual = 0
    for optimization step:
        Q = masked_sinkhorn(Q_logits)
        X_new = Q.T @ raw_X / column_mass + residual
        text_soft = normalize(Q.T @ raw_text / column_mass)
        text_forward = hard_Q_source + ST(text_soft)
        compute real/synthetic proxy gradients
        loss = gradient_match + distribution_match + propagation + Q_anchor + X_anchor
        update Q_logits and residual
        project residual to the trust region
    materialize top-1 real text source selected by final Q
    append new rows; keep old rows unchanged
```

## 5 实验设计与审计

### 5.1 数据、顺序和构造种子

| 数据集 | 类别顺序 seeds | construction seeds | 最终 memory rows |
|---|---|---|---:|
| WikiCS | 9201, 9202, 9203 | 271108, 271109, 271110 | 410 |
| SportsFit | 9201, 9202, 9203 | 271108, 271109, 271110 | 6057 |
| BookHis | 9202, 9203, 9205 | 271108, 271109, 271110 | 1454 |

每个数据集有 9 个 construction cells，三个数据集合计 27 个 cells。每份冻结 memory 由 Prompt-GCN、Prompt-SGC、Prompt-GraphSAGE 读取，每个 Student 使用 seeds 150/151/152、2048 steps 和 batch size 24。

### 5.2 Sealed test

构造阶段不读取 sealed test，也不使用 sealed test 选择方法或参数。sealed mask 为 `original_id % 10 == 2`。每个 cell 的四个 memory arm 先做 archive hash 校验，再以相同 Student 协议评估。324 个 readout rows 每行是一个 dataset/order/construction-seed/arm/Student 组合；每个 row 内含 3 个 Student seeds，因此总计 972 次 Student fit。

### 5.3 代码和结果入口

- 主构造器：[unified_q_multimodal.py](../../code/v2_llm_fusion/unified_q_multimodal.py)
- 构造实验 runner：[run_unified_q_inloop_cat_20261007.py](../../code/v2_llm_fusion/run_unified_q_inloop_cat_20261007.py)
- Prompt Student：[graphtext_prompt_student.py](../../code/v2_llm_fusion/graphtext_prompt_student.py)
- SGC Student：[graphtext_sgc_transfer.py](../../code/v2_llm_fusion/graphtext_sgc_transfer.py)
- GraphSAGE Student：[graphtext_sage_transfer.py](../../code/v2_llm_fusion/graphtext_sage_transfer.py)
- Sealed evaluator：[evaluate_uq_sealed_test_20261009.py](../../code/v2_llm_fusion/evaluate_uq_sealed_test_20261009.py)
- 外部基线适配：[external_continual_mechanisms.py](../../code/v2_llm_fusion/external_continual_mechanisms.py)

## 6 主要结果

下面所有数值均为 sealed test 上的 9 个 construction cells 均值；单位为百分比。括号中为同一方法的 construction-cell 标准差。表中的 CaT 和 GECC 是机制适配版本，不是原论文官方分数。

### 6.1 WikiCS

| Student | 方法 | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Prompt-GCN | In-loop CaT-Q | 78.52 ± 1.42 | 80.88 ± 1.61 |
| Prompt-GCN | Q-medoid | 78.72 ± 0.84 | 81.16 ± 1.18 |
| Prompt-GCN | CaT + nearest5 | 78.60 ± 0.67 | 81.86 ± 0.55 |
| Prompt-GCN | GECC-hard + TAG | 77.91 ± 0.41 | 80.71 ± 0.74 |
| Prompt-SGC | In-loop CaT-Q | 76.52 ± 0.81 | 79.70 ± 0.93 |
| Prompt-SGC | Q-medoid | 76.94 ± 0.60 | 80.00 ± 0.71 |
| Prompt-SGC | CaT + nearest5 | 71.12 ± 2.41 | 74.60 ± 2.38 |
| Prompt-SGC | GECC-hard + TAG | 76.99 ± 0.84 | 79.74 ± 0.79 |
| Prompt-GraphSAGE | In-loop CaT-Q | 78.03 ± 2.07 | 80.09 ± 2.29 |
| Prompt-GraphSAGE | Q-medoid | 78.44 ± 2.05 | 80.53 ± 2.17 |
| Prompt-GraphSAGE | CaT + nearest5 | 79.67 ± 1.17 | 82.23 ± 1.04 |
| Prompt-GraphSAGE | GECC-hard + TAG | 76.61 ± 0.98 | 79.06 ± 1.49 |

### 6.2 SportsFit

| Student | 方法 | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Prompt-GCN | In-loop CaT-Q | 77.94 ± 2.77 | 83.76 ± 2.17 |
| Prompt-GCN | Q-medoid | 77.90 ± 2.07 | 83.46 ± 2.18 |
| Prompt-GCN | CaT + nearest5 | 76.07 ± 3.52 | 84.52 ± 3.00 |
| Prompt-GCN | GECC-hard + TAG | 77.70 ± 2.00 | 84.93 ± 2.16 |
| Prompt-SGC | In-loop CaT-Q | 73.34 ± 0.97 | 80.40 ± 1.54 |
| Prompt-SGC | Q-medoid | 73.65 ± 0.84 | 80.81 ± 1.35 |
| Prompt-SGC | CaT + nearest5 | 56.36 ± 4.65 | 60.04 ± 7.40 |
| Prompt-SGC | GECC-hard + TAG | 75.01 ± 2.54 | 81.96 ± 2.87 |
| Prompt-GraphSAGE | In-loop CaT-Q | 75.32 ± 2.10 | 79.76 ± 1.82 |
| Prompt-GraphSAGE | Q-medoid | 75.64 ± 2.90 | 80.35 ± 2.82 |
| Prompt-GraphSAGE | CaT + nearest5 | 77.55 ± 3.03 | 84.52 ± 2.56 |
| Prompt-GraphSAGE | GECC-hard + TAG | 75.00 ± 2.18 | 81.52 ± 3.51 |

### 6.3 BookHis

| Student | 方法 | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Prompt-GCN | In-loop CaT-Q | 39.91 ± 3.74 | 73.23 ± 4.42 |
| Prompt-GCN | Q-medoid | 38.79 ± 3.53 | 72.62 ± 5.05 |
| Prompt-GCN | CaT + nearest5 | 17.68 ± 2.87 | 53.98 ± 19.14 |
| Prompt-GCN | GECC-hard + TAG | 38.31 ± 2.66 | 71.47 ± 6.86 |
| Prompt-SGC | In-loop CaT-Q | 34.74 ± 3.00 | 68.86 ± 5.18 |
| Prompt-SGC | Q-medoid | 35.02 ± 3.97 | 68.97 ± 6.36 |
| Prompt-SGC | CaT + nearest5 | 13.82 ± 2.10 | 50.65 ± 14.92 |
| Prompt-SGC | GECC-hard + TAG | 35.74 ± 2.48 | 70.78 ± 4.26 |
| Prompt-GraphSAGE | In-loop CaT-Q | 38.81 ± 3.99 | 70.89 ± 6.54 |
| Prompt-GraphSAGE | Q-medoid | 37.40 ± 4.34 | 69.66 ± 7.33 |
| Prompt-GraphSAGE | CaT + nearest5 | 29.78 ± 3.32 | 62.88 ± 14.31 |
| Prompt-GraphSAGE | GECC-hard + TAG | 36.96 ± 2.58 | 68.85 ± 8.04 |

### 6.4 配对差值

正值表示 In-loop CaT-Q 优于对应基线，单位为 Macro-F1 percentage points。

| 数据集 / Student | 相对 CaT | 相对 GECC |
|---|---:|---:|
| WikiCS / Prompt-GCN | -0.08 | +0.61 |
| WikiCS / Prompt-SGC | +5.40 | -0.47 |
| WikiCS / Prompt-GraphSAGE | -1.64 | +1.42 |
| SportsFit / Prompt-GCN | +1.86 | +0.24 |
| SportsFit / Prompt-SGC | +16.97 | -1.67 |
| SportsFit / Prompt-GraphSAGE | -2.24 | +0.32 |
| BookHis / Prompt-GCN | +22.23 | +1.60 |
| BookHis / Prompt-SGC | +20.92 | -1.01 |
| BookHis / Prompt-GraphSAGE | +9.03 | +1.85 |

### 6.5 可视化

![Sealed test Macro-F1](../../results/uq_text_path_sealed_test_20261009_final/final_f1.png)

![Sealed test Accuracy](../../results/uq_text_path_sealed_test_20261009_final/accuracy.png)

## 7 结果解释与机制归因

### 7.1 当前方法真正带来的收益

最清楚的收益不是“所有图都超过所有 baseline”，而是：

1. **在长尾和大规模 BookHis 上改善明显。** 相对 CaT nearest5，三个 Student 的 F1 都提升，且 9 个 construction cells 全部为正；相对 GECC，GCN 和 GraphSAGE 为正，SGC 略低。
2. **在部分 Student 上更能承受 CaT 文本适配失配。** SportsFit/SGC 和 BookHis/GCN、SGC 的 CaT nearest5 退化明显，而共享 (Q) 的主方法没有同步塌缩。
3. **跨 Student 的 memory 复用是可测的。** 同一份 synthetic memory 被三个不同图传播算子读取，主方法不是只绑定一个下游 GCN checkpoint。
4. **协议和数据访问边界被实际审计。** 这一点是 setting 贡献，不是性能 SOTA 贡献：sealed test 未参与构造，archive hash、mask hash 和 memory row budget 均通过检查。

### 7.2 当前方法没有证明什么

Q-medoid 与 In-loop CaT-Q 很接近，因此不能把全部收益写成“q_max 文本选择带来的提升”。GECC 在多个条件下优于主方法，说明当前 GM proxy 和 text route 还没有形成稳定的跨架构优势。当前版本没有 topology，因此不能宣称解决了新旧边保持；旧 memory 也没有更新，只有 frozen-prefix append。

另外，构造时仍使用标签完成阶段划分、类别 mask、slot 分配和 prompt proxy loss。预测时没有训练分类头，但不能把整个 pipeline 称为严格 label-free zero-shot。论文应使用“prompt-based zero-shot readout”或“GraphCLIP-style prompt evaluation”。

### 7.3 负结果的研究含义

- **CaT + nearest5 在 BookHis 失败。** 这首先说明不同构造器生成的 (X) 和简单 nearest text policy 之间存在数据分布交互，不能把文本附着当作无关后处理。
- **Q-medoid 没有稳定超过主方法。** 单独改善 hard source selection 不足以保证下游提升，说明 Student readout 对类内文本覆盖和图文匹配的要求不等价于 embedding 代表性。
- **GECC 的竞争力很强。** 它说明基线适配并非天然弱，主方法若要进一步提升，必须针对当前 Unified-Q 的目标和 Student 失配提出一个单一、可证伪的修正，而不是继续堆叠 EM、topology、attention 和动态预算。
- **此前的 shared/independent Q、no-GM 和 joint-distribution 控制没有给出稳定通用收益。** 这些结果应作为消融和边界，不应从论文中删除。

## 8 贡献与新颖性客观评价

### 8.1 可以成立的贡献

本文可以较稳妥地声称：

1. 提出了一个明确的历史 raw 不可回读的持续 TAG 凝聚协议；
2. 将 class-constrained (Q) 作为特征池化、文本 routing 和 Student-compatible construction objective 的共同隐变量；
3. 通过 hard-consistent materialization 让训练中的文本对象和最终保存的真实文本一致；
4. 对不同 Student、类别顺序、construction seed 和 sealed test 做了系统实验，并报告了方法失效条件。

### 8.2 不能夸大的地方

从算法新颖性看，Sinkhorn assignment、gradient matching、CaT-style distribution matching、straight-through hard selection 和 prompt contrastive Student 都有先例或直接借鉴来源。当前版本的 novel 点主要是它们在“历史 raw 不可回读的持续 TAG”设定下的统一组合，而不是单个基础算子的新理论。

当前证据不足以支持：

- 稳定 SOTA；
- 普遍超过 GECC；
- 文本 routing 对所有数据集都有独立正贡献；
- 完整 topology-aware continual condensation；
- 对旧 memory 的联合刷新或新旧边协同优化。

因此，投稿 WWW 长文的可行性取决于论文定位：若定位为“新的持续 TAG 问题、可审计协议、统一构造框架和跨架构边界分析”，有明确研究价值；若定位为“新算法普遍显著超过强基线”，当前证据不够。

## 9 后续工作与冻结规则

当前 sealed test 不再用于调参。后续若继续优化，必须新开预注册的顺序和 sealed split，并且一次只改一个机制：

1. 先检查 GM proxy 与实际 multi-positive Student 的失配，尤其是类内逐行对应关系是否被 full-candidate group loss 消除了；
2. 如果确实是目标失配，再做一个类内图文联合分布匹配版本，并保留现有主表；
3. 只有在新的 split 上得到稳定收益后，才把 topology 或旧 memory bounded refresh 加入主方法；
4. EM 只能作为 Q 的交替推断解释，不能因为形式上存在潜变量就直接称为完整 EM；
5. 新旧边和 topology 需要单独的 node-classification 与 link-prediction 任务验证，不能从 edge-free 结果外推。

在论文写作上，现阶段可以开始正式写作，主方法应写成“暂定最终方法”并明确边界。最终冻结条件应包括：独立顺序平均不退化、至少一个大图上相对强 baseline 有稳定收益、成本可接受，以及组件归因不依赖单一 Student。若这些条件不满足，仍可以投稿 setting/analysis 取向的论文，但不应把 Unified-Q 宣称为通用 SOTA condensation method。

## 10 复现与审计文件

- [sealed 结论](../current/UNIFIED_Q_SEALED_TEST_CONCLUSION_ZH_2026-10-09.md)
- [扩展确认报告](../current/UNIFIED_Q_EXTENDED_CONFIRMATION_ZH_2026-10-09.md)
- [外部 baseline fidelity audit](../current/EXTERNAL_BASELINE_FIDELITY_AUDIT_ZH_2026-10-09.md)
- [sealed 汇总表](../../results/uq_text_path_sealed_test_20261009_final/summary.md)
- [sealed JSON](../../results/uq_text_path_sealed_test_20261009_final/summary.json)
- [sealed audit](../../results/uq_text_path_sealed_test_20261009_final/audit.json)

论文中的核心实现位置：

- `code/v2_llm_fusion/unified_q_multimodal.py`
- `code/v2_llm_fusion/run_unified_q_inloop_cat_20261007.py`
- `code/v2_llm_fusion/graphtext_prompt_student.py`
- `code/v2_llm_fusion/evaluate_uq_sealed_test_20261009.py`
- `code/v2_llm_fusion/external_continual_mechanisms.py`

## 结论

Unified-Q 目前已经从“有多个组件的启发式尝试”推进到一个可以被清楚定义、审计和复现的持续 TAG 凝聚候选方法。它确实使用了 (Q) 和 gradient matching 将特征、文本和 Student-compatible objective 放入同一构造循环，也在 BookHis 和部分 Student 条件下产生了明显收益。但它还没有证明跨数据集、跨架构优于 GECC，也没有解决 topology 和旧记忆联合更新问题。最诚实、也最有说服力的论文叙述应是：提出并系统研究一个新的历史受限持续 TAG 凝聚问题，给出 Unified-Q 统一构造，并展示其收益、适用条件和失败边界。
