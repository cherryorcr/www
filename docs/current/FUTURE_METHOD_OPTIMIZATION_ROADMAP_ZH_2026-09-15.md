> 2026-09-17 状态说明：本文保留了不同时期的方法解释和原始计划，后半的待办状态部分已过时。当前版本关系、P0-P5 完成度和后续优先级统一见 [v4 状态与后续优化路线](V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md)。P2 已收尾但新文本机制未晋升；部分 P3/P5 已提前实施，不能按本文旧复选框判断当前进度。

## 先给结论

当前真正可以作为“方法主线”的不是计划书中所有模块的简单叠加，而是：

> **fixed-rate memory + dispersion allocation + synthetic-only append + frozen-old topology + student-response calibration**

也就是 v4 safe mainline。  
复杂度分配、adaptive-rate、response-gain 扩容、DeepSets 文本编码、历史 memory refresh 目前仍是候选或后续方向，不能全部写成已验证优势。

## 1. 任务设定

方法处理的是 **图文联合的持续图凝聚**：

\[
D_0 \rightarrow D_1 \rightarrow D_2 \rightarrow \cdots
\]

每个阶段只接收当前增量数据 \(\Delta_t\)。阶段结束后，历史原始图不能再次访问，只允许保留上一阶段的 synthetic memory。

当前小图协议：

- Cora：`[0,1] → [2,3] → [4,5,6]`
- Citeseer：`[0,1] → [2,3] → [4,5]`
- 主 memory rate：5%
- student batch：16
- student seeds：100--105
- construction pool 排除 dev、report-dev 和 test
- test 只在最终确认阶段读取

当前 v4 实现中，selector 使用独立 seeds `300/301/302`，避免用同一批 student seed 既选方法又报告结果。

## 2. 完整算法流程

### Step 1：图文预处理

每个源节点有：

- 图特征 \(x_i\)
- 原始文本 \(s_i\)
- 类别标签只用于定义 class-incremental 阶段，不用于 student 训练

文本经过冻结 MiniLM：

\[
t_i=\operatorname{MiniLM}(s_i)\in \mathbb{R}^{384}
\]

然后形成联合表示：

\[
h_i=[\operatorname{Normalize}(x_i);t_i]
\]

图特征和文本 embedding 同时参与凝聚空间构造。

### Step 2：计算 memory 总预算

主协议使用固定压缩率：

\[
K_t=\left\lfloor \rho N_t^{pool}\right\rfloor
\]

其中：

- \(\rho=0.05\)
- \(N_t^{pool}\) 是截至当前阶段累计 construction-pool 节点数
- 因此 memory 会随着新数据到来而增长

这比固定总 memory 更符合持续学习场景，也能避免“后续数据到来但 memory 完全不增长”。

### Step 3：按类别分配新 memory

当前稳定的 dispersion allocation 使用：

\[
w_c=\sqrt{n_c}(1+d_c)
\]

其中：

- \(n_c\)：当前类别源节点数量
- \(d_c\)：类别内部图文联合离散度
- \(\sqrt{n_c}\)：让大类获得更多表示，但避免线性垄断预算
- \(1+d_c\)：对内部多模态、结构复杂的类别增加容量

最终使用 largest-remainder 将固定新增预算分配给各类别，并尽量保证每个新类至少获得一行。

旧版“每个类别固定 15 rows”仍保留为 legacy control，但不是当前 v4 默认策略。

### Step 4：构造 synthetic nodes

每个类别或子簇内部进行 graph-text joint clustering。

对于一个簇 \(C_j\)，合成节点特征为：

\[
x_j^{syn}=\frac{1}{|C_j|}\sum_{i\in C_j}x_i
\]

注意：

> 真实节点只作为凝聚来源，不会直接成为凝聚图节点。最终 memory 中的节点全部是 synthetic nodes。

每个 synthetic node 还保存：

- `mass`：该节点代表的真实节点数量
- `stage`：产生阶段
- `depth`：历史文本融合深度
- 文本表示
- topology 信息
- allocation 和覆盖统计

### Step 5：文本构造

当前 v4 默认文本方案是 `joint_g5_c320`：

1. 在 joint graph-text 空间找到簇中心；
2. 选择距离中心最近的最多 5 个源文本；
3. 每个文本截断；
4. 拼接成一个文本；
5. 再用冻结 MiniLM 重新编码。

形式上：

\[
T_j=\operatorname{MiniLM}
\left(
s_{j_1}\Vert s_{j_2}\Vert\cdots\Vert s_{j_g}
\right)
\]

重新编码很重要，因为它允许 MiniLM 重新理解文本组合，而不是简单平均原始 embedding。

P2 中还测试了：

- diversity-aware concat
- mean pooling
- distance-weighted pooling
- set attention pooling
- dual prototype
- DeepSets
- no-fuse、no-text 等控制组

但目前没有候选在跨数据集、独立 seed 和阶段平均指标上稳定胜过 center concat，因此没有改写 v4 默认方案。

### Step 6：构造 synthetic topology

默认采用 `frozen-old topology`：

- 保留旧 memory 中仍被保留节点之间的 old-old edges；
- 为当前新增节点构造 new-new edges；
- 按文本相似度添加受控 old-new edges；
- 默认 `cross_degree=1`。

其核心约束是：

\[
E_t^{old-old}=E_{t-1}^{old-old}
\]

也就是说，旧节点如果没有被 memory budget 淘汰，其原有拓扑关系不被重建。

这里的 frozen-old 并不意味着旧节点永远不会被删除。预算不足时，旧节点仍可能根据 `mass` 被裁剪；但保留下来的旧节点，其特征、文本和 old-old topology 不会主动更新。

### Step 7：Student-Compatible Append

持续更新公式为：

\[
M_t=
\operatorname{Append}
\left(
\operatorname{Shrink}(M_{t-1}),
\operatorname{Construct}(\Delta_t)
\right)
\]

每一阶段：

1. 根据 \(K_t\) 计算旧 memory 保留量；
2. 根据新类别预算构造 delta synthetic block；
3. 按 `mass` 保留旧节点；
4. 追加新 synthetic nodes；
5. 保留或扩展 topology；
6. 不重新读取历史原始节点。

这样保证方法满足真正的 continual constraint。

### Step 8：Student response selection

v4 不仅依赖几何距离或聚类误差，而是检查：

> 这个 synthetic memory 能否重新训练出好的 GraphCLIP student？

对候选 memory 进行短程 student 重训练，使用：

- 多个 selection seeds
- 多个 batch
- 多个 training horizons，例如 \(0.5\times,1\times,2\times\) steps

响应分数大致为：

\[
J=
F1_{\text{cumulative}}
+\lambda_o F1_{\text{old}}
\]

并计算风险下界：

\[
LCB=J-z\cdot Uncertainty
\]

只有当候选满足：

- LCB gain 非负；
- 最差 response horizon 不下降；
- 旧类保持不明显恶化；

才允许接受 row transfer。否则自动退回 dispersion baseline。

因此 v4 的 response selector 是一个“有安全回退机制”的校准器，而不是强行替代稳定基线。

## 3. 下游 Student 模型

当前 student 是一个 GraphCLIP-style 双塔模型。

### 图编码器

两层 GCN：

\[
z_g=
\operatorname{Normalize}
\left(
GCN_2(
\operatorname{ReLU}(GCN_1(X,A))
)
\right)
\]

其中：

- \(A\)：synthetic graph 的归一化邻接矩阵
- hidden dimension：384
- output dimension：384

### 文本编码器

MiniLM 保持冻结，输出经过一个可训练线性投影：

\[
z_t=\operatorname{Normalize}(W_t t)
\]

### 训练损失

使用双向 InfoNCE：

\[
\mathcal{L}
=
\frac{1}{2}
\left[
CE(Sz_gz_t^\top,I)
+
CE(Sz_tz_g^\top,I)
\right]
\]

其中 \(S\) 是可学习 temperature/logit scale。

student 只在 synthetic memory 上训练，不使用真实节点标签。

## 4. 为什么选择这个 Student

### 两层 GCN

选择两层 GCN 的原因：

- 能利用 synthetic topology；
- 与真实图推理时的消息传播形式一致；
- 参数量和训练成本较低；
- 避免深层 GNN 过平滑；
- 方便在多个候选 memory 上重复训练。

### 384 维表示

MiniLM 输出为 384 维，因此图塔和文本塔统一到 384 维：

- 减少额外瓶颈；
- 保持 GraphCLIP 文本空间兼容；
- 让不同文本构造方法可以公平比较。

### Student 从头训练

如果继承上一阶段 student 参数，memory 的真实质量会被掩盖。每一阶段从头训练可以直接回答：

> 当前 memory 是否包含足够的图文结构信息，使一个新 student 重新学习出来？

### batch=16 与 scaled steps

训练步数为：

\[
\operatorname{steps}
=
\max(50,\min(2048,16\times \text{synthetic rows}))
\]

这样 memory 越大，student 获得的训练曝光量相应增加，同时设置上限控制成本。

## 5. 图和文本分别怎样使用

### 图的作用

图特征用于：

- 构造 synthetic node feature；
- 与文本共同形成 joint clustering 空间；
- 计算类别内部 dispersion；
- 训练 GCN 图编码器；
- 最终在真实图 adjacency 上评估节点预测。

### 文本的作用

源文本用于：

- MiniLM embedding；
- graph-text joint clustering；
- 文本代表选择；
- synthetic node text 构造；
- synthetic topology 构造；
- 训练 student 的文本目标。

类别 prompt 例如：

```text
a research paper about neural networks
```

也由同一个 MiniLM 编码，并用于 zero-shot 分类。

需要注意：当前历史主指标是“节点只取 seen classes，但预测时仍允许所有 prompt 竞争”的 open-vocabulary 口径。P2 已补充 seen-prompt-only 诊断，因此正式论文中必须明确 prompt scope，不能直接把历史 early-stage 数值称为标准 closed-set F1。

## 6. 主要解决的 challenge

### 1. 历史原图不可访问

通过 synthetic-only append，使后续阶段只依赖：

\[
M_{t-1}+\Delta_t
\]

不需要重新读取历史原图。

### 2. 极高压缩率下的信息保留

通过固定压缩率、mass、joint clustering 和文本再编码，在少量 synthetic nodes 中保留图文信息。

### 3. 类别规模不平衡

通过 \(\sqrt{n_c}\) 分配，避免大类完全垄断 memory，也比固定 15 rows 更适应数据规模。

### 4. 类内多模态

通过 dispersion 和 joint-space clustering，为结构复杂类别提供更多表示容量。

### 5. 图文对应关系漂移

文本和图特征共同聚类，合成后重新编码文本，减少单独压缩图和文本导致的错配。

### 6. 持续学习中的 topology drift

frozen-old topology 保留旧节点之间的结构，避免每次追加新节点都重建整个 synthetic graph。

### 7. selector 与最终 student 不一致

使用多 horizon、多 batch、独立 selection seeds 和 LCB，降低“短训看起来有效、完整训练反而退化”的问题。

### 8. 实验协议不公平

统一了：

- 数据划分；
- memory rate；
- batch；
- student steps；
- seeds；
- topology；
- test sealing；
- artifact hash；
- synthetic-only 约束。

## 7. 当前效果

v4 小图结果：

| 数据集   | dispersion baseline | v4 mainline |
| -------- | ------------------: | ----------: |
| Cora     |               49.85 |       49.85 |
| Citeseer |               59.17 |       60.77 |

解释：

- Cora：response transfer 没有通过最差 horizon 和风险门禁，因此自动回退到 dispersion；
- Citeseer：一次固定预算内的 row transfer 通过门禁，最终提升约 1.59 个百分点；
- 目前只能说明 v4 的安全回退机制有效，不能宣称 response selector 已经跨数据集稳定优于 dispersion。

P2 文本实验也没有找到稳定替代方案：

- mean pooling 在部分独立确认中有正增益；
- set pooling 对 Citeseer 较好，但对 Cora 不稳定；
- weighted pooling 增益较小；
- DeepSets 没有表现出稳定优势；
- diversity concat 没有跨数据集稳定提升。

因此当前默认仍是简单的 center concat。

## 8. 当前没有启用的模块

以下模块已经实现或有实验分支，但暂时不属于默认主线：

- adaptive-rate budget
- response-gain extra budget
- subcluster complexity-v2
- bounded historical refresh
- full history re-clustering
- DeepSets text encoder
- path beam search
- global topology rebuild

它们的共同问题是：目前还没有同时满足跨数据集、独立 seed、阶段平均、旧类保持和完整训练确认。

当前最准确的方法定位是：

> **一个 synthetic-only 的持续图文凝聚方法，以 dispersion 作为稳定 memory 分配先验，以 frozen-old topology 保持历史结构，并用 student response 对固定预算内的新增分配进行风险控制。**

相关代码和方案说明：

- [v4 方法与优化方案](D:/Aprojects/continue_tag_condensation/docs/current/V4_TARGET_METHOD_AND_OPTIMIZATION_PLAN_ZH_2026-09-15.md)
- [v4 小图验收结果](D:/Aprojects/continue_tag_condensation/docs/current/V4_MAINLINE_SMALL_RESULTS_ZH_2026-09-15.md)
- [统一协议](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/protocol_v3.py)
- [Student-Compatible Append 主 runner](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_student_compatible_append.py)
- [P2 文本压缩验收](D:/Aprojects/continue_tag_condensation/docs/current/P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md)

# GraphCLIP continual v4：后续方法优化总体路线与实验优先级

日期：2026-09-15  
适用项目：GraphCLIP continual TAG condensation  
当前基线：unified v3 + frozen-old Student-Compatible Append

本文是后续研究与实验执行报告。它区分已经实现的 v3 基线、待验证的 v4 机制和仅用于诊断的对照，不把尚未实验验证的设想写成结论。

## 1. 总体目标

将当前方法从“固定配方的持续追加”推进为：

> 在持续数据流和增长式压缩预算下，构造全由合成节点组成的图文记忆；根据新信息的有效性动态分配 memory、文本和拓扑容量；使用 student response 校准追加、刷新和压缩决策。

最终方法需要同时回答四个问题：

1. 数据增加时，memory 是否应该按固定压缩率增长，还是根据新信息量自适应增长？
2. 每个阶段和每个类别应保留多少个合成节点？
3. 多条原始文本如何压缩成一个合成节点的语义表示？
4. 历史合成节点何时保留、刷新、拆分、合并或淘汰？

## 2. 必须保持的设计原则

### 2.1 凝聚图只使用合成节点

真实节点不能直接成为最终凝聚图的节点。真实数据只允许用于：

- 计算合成节点的属性均值或稳健 prototype；
- 选择代表性文本并形成文本 prototype；
- 计算簇质量、覆盖度和难度；
- 构造 real-node baseline 作为对照。

最终 memory 的每一行都是一个 synthetic node，至少包含：

```text
synthetic_x
synthetic_text / text prototype
text embedding(s)
mass
depth
stage
synthetic edge_index
quality / coverage metadata
```

`random_real`、`k_center`、`herding` 和 `kmeans_medoid` 只能作为“真实节点记忆”对照，不能混入主方法定义。

### 2.2 所有预算必须显式记录

每个结果必须同时记录：

- 原始节点数和累计节点数；
- synthetic rows 数；
- 实际压缩率；
- 每阶段新增 rows；
- 每个类别或子簇的 rows；
- 文本 token/字符预算；
- student training steps 和 batch；
- selector 训练总成本。

否则不能判断提升来自算法，还是仅仅来自 memory 更大、文本更长或训练更久。

## 3. v4 方法的总体结构

第 (t) 阶段的推荐形式为：

\[
K_t = \operatorname{Budget}(N_t, U_t, K_{t-1})
\]

\[
\Delta_t \xrightarrow{\text{adaptive allocation}}
\{G_{t,1},\ldots,G_{t,k_t}\}
\xrightarrow{\text{text compression}}
\Delta_t^{syn}
\]

\[
M_t = \operatorname{ManageMemory}(M_{t-1},\Delta_t^{syn},K_t)
\]

其中：

- (N_t)：截至当前阶段的原始数据量；
- (U_t)：新到信息的有用程度；
- (K_t)：当前允许的 synthetic memory 大小；
- (Delta_t^{syn})：当前阶段生成的合成节点块；
- `ManageMemory`：追加、保留、刷新、拆分、合并和淘汰。

推荐的完整流程：

1. 读取当前 delta，排除 test 来源和 test 标签。
2. 计算新信息量、分布离散度、多模态程度和结构覆盖缺口。
3. 计算本阶段总 memory 预算 (K_t)。
4. 在 delta 内部进行无标签联合聚类，形成多个合成子簇。
5. 为每个子簇生成 synthetic feature、文本集合表示和局部结构统计。
6. 构造 frozen-old、controlled-update 等拓扑候选。
7. 用短程 student response curve 评估候选。
8. 以新类塑性、旧类保持、稳定性和结构覆盖为多目标选择候选。
9. 在总预算约束下追加新 synthetic nodes，并受控更新低质量旧节点。
10. 用独立 canonical seeds 完整训练和评估。

## 4. 优化方向一：信息量自适应的增长预算

### 4.1 基础预算协议

当前 v3 的 `--rate` 实际使用：

\[
K_t=\lfloor \rho N_t^{pool}\rfloor
\]

它已经是随累计数据增长的 fixed-rate budget，不是固定总预算。后续应明确区分三种协议：

#### Fixed-rate

\[
K_t=\lfloor \rho N_t\rfloor
\]

用于主公平比较，控制压缩率不变。

#### Fixed-total

\[
K_t=K_{max}
\]

用于 memory 受限压力测试，不作为默认 continual 机制。

#### Hybrid

\[
K_t=\max(\lfloor\rho N_t\rfloor,K_{t-1}+b_{min})
\]

保证小 delta 到来时也有最低新增容量。

### 4.2 信息量自适应预算

进一步令压缩率随新信息量变化：

\[
\rho_t=\operatorname{clip}
\left(\rho_0[1+\eta(U_t-\bar U)],
\rho_{min},\rho_{max}\right)
\]

\[
K_t=\max\left(K_{t-1}+b_{min},\lfloor\rho_tN_t\rfloor\right)
\]

其中 (U_t) 初期必须使用不依赖标签的指标：

\[
U_t = a\,novelty_t+b\,dispersion_t+c\,multimodality_t
       +d\,coverageGap_t+e\,selfSupervisedGain_t
\]

可用指标包括：

- 新旧 joint embedding 分布距离；
- delta 内部离散度；
- 类内或簇内多模态程度；
- 新数据对旧 prototype 的覆盖缺口；
- student 的自监督 loss 或图文对齐改善；
- 新 synthetic candidate 的 response gain。

最后一项如果使用 dev F1，必须标为“有监督 selector 版本”，不能和 label-free 构造混称。

### 4.3 推荐实现顺序

先做两阶段分配，避免一开始出现预算和 selector 的循环依赖：

1. 用固定基础压缩率生成初始候选。
2. 计算 (U_t) 和每个子簇的边际收益。
3. 只把额外预算分配给边际收益仍为正的子簇。
4. 达到 (K_t) 或边际收益低于阈值后停止。

这比直接让一个复杂模型同时预测 rows 和文本更容易审计。

### 4.4 必做实验

比较：

- fixed-total；
- fixed-rate：0.1%、0.5%、1%、5%；
- hybrid；
- adaptive-rate；
- adaptive-rate with response gain。

每种协议都要画：

- cumulative F1 vs. memory rows；
- cumulative F1 vs. compression rate；
- memory rows vs. stage；
- wall time vs. stage；
- old retention vs. new plasticity。

## 5. 优化方向二：按规模和复杂度分配 synthetic rows

### 5.1 不再固定每类 15 rows

固定 15 只能作为 control。主方法应先计算每个类别或无标签子簇的权重：

\[
w_{t,c}=n_{t,c}^{\alpha}
(\lambda_d d_{t,c})
(\lambda_m m_{t,c})
(\lambda_g g_{t,c})
\]

再进行 largest-remainder 分配：

\[
b_{t,c}=\operatorname{Allocate}(B_t,w_{t,c})
\]

其中：

- (n_{t,c})：规模；
- (d_{t,c})：图文联合离散度；
- (m_{t,c})：多模态程度；
- (g_{t,c})：结构覆盖缺口。

第一版推荐：

\[
w_{t,c}=\sqrt{n_{t,c}}(1+\lambda d_{t,c})
\]

它能增加大类容量，又不会让最大类线性吞噬预算。

### 5.2 类别内部必须细分为多个合成子簇

对一个类别或一个无标签分布块执行：

\[
\Delta_{t,c}\rightarrow
\{G_{t,c,1},G_{t,c,2},\ldots,G_{t,c,k}\}
\]

每个 (G_{t,c,j}) 生成一个 synthetic node，而不是保留真实 medoid。每个 synthetic node 可保存：

- 簇内 feature 的稳健均值或截尾均值；
- 多个文本代表的集合表示；
- `mass` 和簇半径；
- 图结构密度和邻居语义统计；
- 质量分数和覆盖分数。

这样得到的是多个合成 prototype 节点，而不是以真实节点代替凝聚节点。

### 5.3 两种使用场景

#### Class-incremental 分析场景

可以使用已知类别进行预算统计和阶段分配，用于回答不同类别规模对持续分类的影响。但要明确这是 protocol-level label use，不是 label-free construction。

#### 通用下游任务场景

不使用类别标签，先在 joint graph-text space 中进行多模态聚类，再用子簇复杂度分配 synthetic rows。此版本更适合：

- 链路预测；
- 相似度检索；
- 社区发现；
- 异常检测；
- 图迁移和跨任务评估。

### 5.4 必做实验

同一总预算下比较：

1. equal / fixed rows；
2. linear by class size；
3. sqrt by class size；
4. sqrt + dispersion；
5. subcluster complexity allocation；
6. response-based marginal allocation。

## 6. 优化方向三：文本从拼接升级为固定预算的集合压缩

### 6.1 当前结论

当前 `g3/g5/g8 + character budget` 主要是文本使用量控制，不是已被证明的语义层级。已有证据支持“融合后重新编码”通常优于直接平均，但没有证明固定 `g=5, chars=320` 普遍最优。

### 6.2 推荐的文本候选

#### Candidate A：中心文本拼接

选择靠近簇中心的代表文本，拼接后重新用 MiniLM 编码。作为可解释 control。

#### Candidate B：diversity-aware extractive selection

在固定 token budget 下，交替优化代表性和多样性：

\[
score(i)=\lambda sim(i,center)
-(1-\lambda)\max_{j\in S}sim(i,j)
\]

它选择的是文本证据，不生成新文本，因此可避免生成式摘要的 hallucination。

#### Candidate C：dual prototype

一个 synthetic node 内同时保存：

- center prototype：表示最典型语义；
- diversity prototype：表示类内或簇内差异。

最终可输出一个主 embedding 和一个辅助 embedding，或在 student 中使用双头对齐。

#### Candidate D：真正的 set encoder

保留若干文本 embedding：

\[
E_c=\{e_{c,1},...,e_{c,k}\}
\]

使用 DeepSets、attention pooling 或 Set Transformer 得到：

\[
e_c^{set}=f_\theta(E_c)
\]

该 synthetic node 仍然只有一个图节点，只是文本侧保留集合统计，不是把真实节点加入凝聚图。

### 6.3 公平比较条件

所有文本方案必须固定：

- synthetic rows 数；
- token/character 总预算；
- text encoder；
- graph features；
- student steps 和 batch；
- random seeds。

否则更长的拼接文本会天然获得额外信息，无法判断编码器是否真正更好。

## 7. 优化方向四：历史 synthetic memory 管理

### 7.1 需要从“冻结或全量重聚类”升级

当前两端方案是：

- `frozen_old`：历史行和 old-old topology 保留；
- `full_history/recluster`：历史被重新凝聚或重写。

后续采用受控管理器：

\[
M_t=\operatorname{ManageMemory}(M_{t-1},\Delta_t^{syn})
\]

每个旧 synthetic node 具有质量状态：

- student response contribution；
- mass；
- redundancy；
- old retention contribution；
- new coverage overlap；
- topology centrality。

### 7.2 管理动作

#### Keep

历史质量高、覆盖独特的节点原样保留。

#### Bounded refresh

只刷新低质量且与新信息相近的少量旧节点：

\[
x_i'=(1-\alpha)x_i+\alpha x_{new}
\]

文本 prototype 同步更新，但保留 refresh history 和质量变化。

#### Split

当一个旧 synthetic node 的簇半径或文本多模态程度过高时，将其拆成两个或多个合成节点。

#### Merge

当两个旧节点高度冗余且覆盖相近时，合并为一个新的 synthetic node。

#### Evict

只有在新增信息的边际收益高于旧节点保持价值时才淘汰；不能只按低 mass 淘汰。

### 7.3 拓扑管理

拓扑应拆成独立因素：

- old-old edges；
- new-new edges；
- old-new cross edges；
- global rebuild。

主方法先保留 frozen-old control，再验证受控 cross-edge update 和低质量节点刷新。所有拓扑方案要匹配边数或报告边数差异。

## 8. 优化方向五：SCA 选择器升级

SCA 仍是重要主线，但需要从“多个手工配方选一个”升级为可解释的多目标选择器。

### 8.1 选择目标

对候选 (q) 同时评估：

\[
J(q)=
\bigl(
F1_{new}(q),
Retention_{old}(q),
Stability(q),
Coverage(q),
-Cost(q)
\bigr)
\]

优先筛选 Pareto front，再使用固定的风险效用函数选最终方案。

### 8.2 必须保留的稳健性机制

- 多 horizon response curve；
- step × batch ranking consistency；
- selection seeds 与 canonical seeds 分离；
- 多组互不重叠的 selector seed；
- full-step terminal confirmation；
- 路径 beam 作为稳定性诊断；
- 报告候选路径频率、top-1 agreement 和 full-step regret。

大 runner 的简化版和小图增强版必须统一，否则不能把两组结果放在同一条方法主线上。

## 9. 优化方向六：持续阶段协议

### 9.1 推荐正式协议

- 主表：base + 4 incremental sessions；
- 快速消融：3-stage；
- 稳健性：5 个类别顺序；
- 长期遗忘：8 或 10 stages；
- 每阶段固定类别数量、样本量记录和 memory rule。

三阶段仍可保留，但应标记为快速协议，不能作为唯一持续学习证据。

### 9.2 类别顺序

至少比较：

1. canonical order；
2. frequency-balanced order；
3. 3 个随机 order；
4. 语义相近类别连续到达；
5. 语义差异类别交替到达。

主要指标增加：

- current F1；
- old F1；
- cumulative F1；
- average forgetting；
- backward transfer；
- forward transfer；
- memory growth；
- selector path stability。

## 10. 官方 GraphCLIP 对齐任务

该任务必须作为单独的 student validity workstream，不与 memory 构造改动同时进行。

### 10.1 三个锚点

1. 官方预训练 GraphCLIP 在官方输入协议上的结果；
2. 官方 GraphCLIP 结构迁移到 unified v3 数据、prompt、dev split 的结果；
3. 当前项目 GraphCLIP-style GCN student 在相同输入和训练预算下的结果。

### 10.2 必须控制的变量

- graph tower 结构；
- text tower 和 checkpoint；
- prompt 模板；
- graph preprocessing；
- full graph 或 ego graph；
- student 是否从头训练；
- text encoder 是否冻结；
- batch、steps、seeds；
- 评估类别空间和 dev/test mask。

只有完成这组对照后，才能量化当前 student 与官方 GraphCLIP 的差异，也才能知道差异来自 student 还是 synthetic memory。

## 11. 总体实验优先级

### P0：协议与可审计基础设施（已完成）

目标：先确保所有后续结果可比较。

- [x] 显式支持 fixed-total、fixed-rate、hybrid、adaptive-rate；
- [x] 支持可配置阶段数、base-incremental 和多种确定性类别划分；
- [x] 支持 equal、linear、sqrt、sqrt+dispersion rows allocation；
- [x] manifest 记录真实 synthetic rows、压缩率、文本预算、训练 runs 和 optimizer steps；
- [x] 保持 synthetic-only 主方法约束；
- [x] 分离 selector-dev、报告-dev 和封存 test；
- [x] 固定数据、prompt、代码和阶段 manifest hash。

**验收条件：** 同一命令和 manifest 可以完整重建每个结果；所有 arm 的 student 与预算定义一致。

**P0 验收记录（2026-09-15）：**

- `protocol_v3.py` 新增预算解析、adaptive-rate 和四路 disjoint masks；
- `run_unified_v3_continual.py` 新增预算模式、阶段模式、报告集、rows allocation 和 provenance 字段；
- construction text cache 改为 pool-only，selector-dev/report-dev/test 不进入 synthetic construction；
- 默认主方法仍只保存 synthetic nodes，real-node arms 仅作为对照；
- 本地通过 Python compile、预算/mask 纯函数、rows allocation、阶段生成和 synthetic block 检查；
- CLI smoke test 已显示全部 P0 开关；
- 尚未将 adaptive-rate、base+4 或非 legacy allocation 的结果写入性能主表，这些属于 P1 实验；
- P0 不代表性能提升，只代表协议和可审计基础设施就绪。

### P1：预算增长与 rows allocation

目标：验证“按信息量增长、按复杂度分配”是否必要。

第一轮：Cora、Citeseer；第二轮：BookChild、SportsFit、arXiv。

- [ ] 压缩率 sweep（历史 fixed-rate 已有，但 5% allocation 混淆已发现；matched-quota 的多 rate 待补）；
- [x] Cora/Citeseer 5% matched-quota 下固定 15 vs equal/linear/sqrt/dispersion；
- [x] Cora/Citeseer fixed-total、hybrid(min_increment=5)、adaptive-rate 小图验证；
- [x] 类内多子簇 synthetic node 的最小实现和小图首轮验证；
- [x] response-based marginal allocation 的最小实现和小图首轮验证；
- [x] response-gain extra budget 的最小实现和小图首轮验证；
- [ ] memory rows、F1、forgetting、覆盖度和训练成本曲线。

**验收条件：** 新策略必须在 matched memory 或 matched compression rate 下提升，而不能只依赖更多 rows。

**小图阶段结论（2026-09-15）：**

- 原始 `legacy_15` 结果存在新旧 rows quota 混淆，不能作为纯 allocation 结论；
- matched-quota 下 `dispersion` 在 Cora/Citeseer 5% 均为最佳候选，分别为 49.85 和 59.17；
- fixed-total 在两个小图上都低于 fixed-rate；
- adaptive-rate 在 Cora 提升、Citeseer 下降，暂时只能作为数据集相关消融；
- `hybrid(min_increment=5)` 未改变预算，实验强度不足；
- 类内子簇复杂度分配可运行，但 Cora/Citeseer 均未超过 dispersion；
- response-marginal 在 Cora 有效、Citeseer 下降；response-gain extra 在两者都未改善 canonical F1；
- 因此 P1 的小图第一道门禁通过，但 P1 总体验收尚未完成，暂不上大图主表。

**response 机制首轮消融（2026-09-15）：**

| 方案 | Cora | Citeseer |
|---|---:|---:|
| fixed-rate + dispersion | 49.85 | 59.17 |
| complexity-only | 49.38 | 53.44 |
| response-marginal | 52.40 | 57.29 |
| complexity + response-marginal | 48.64 | 55.29 |
| response-gain extra | 46.00 | 53.43 |

这说明 response selector 有研究价值，但当前 ranking 与 canonical student 不稳定；复杂度先验没有改善 selector，response-gain 也产生了 canonical regret。全模块合并暂不运行，先修正 selector 稳定性。

### P1 小图门禁（当前后续顺序）

1. 在 matched-quota 下补做 `rate=0.01` 和 `rate=0.10`，固定 `dispersion`、`sqrt` 和 k-center 对照；
2. 增加一个独立类别顺序或 `base+4` 小图确认；
3. 将 hybrid 的 `min_increment` 设为确实改变阶段预算的值，并记录额外 memory 成本；
4. 汇总 cumulative/current/old F1、rows、训练 steps、训练时间和阶段分配；
5. 只有当 `dispersion` 在至少两个 rate 和一个额外阶段/顺序设置下仍不劣于基线，才进入大图。

若门禁失败：保留 fixed-rate + 最简单稳定 allocation 作为主线，删除“信息自适应预算”作为方法贡献。若门禁通过：再把 `dispersion` 带到 BookChild、SportsFit、arXiv，adaptive-rate 仍只作为消融。

### P1 response 机制的后续门禁

当前三种新机制只完成了最小实现和首轮小图实验，不能直接合并为最终方法。继续研究前必须增加：

1. 独立 selection seed group；
2. full-step terminal confirmation；
3. selector-dev response 与 canonical F1 的 regret；
4. response gain 的置信区间和训练成本；
5. 与 dispersion baseline 的 matched rows 对照。

若 response selector 在独立 seed 和 full-step confirmation 下仍然跨数据集不稳定，则将其保留为负结果和诊断机制，不进入默认主线。

### P2：文本压缩

目标：判断收益来自文本数量，还是来自文本构造机制。

- [ ] center concat；
- [ ] diversity-aware extractive；
- [ ] dual prototype；
- [ ] set encoder；
- [ ] no-text / no-fuse / no-reencode controls。

**验收条件：** 固定总 token budget 下，在至少多个数据集上稳定改善，并能解释失败数据集。

### P3：SCA 统一与选择稳定性

目标：把 student response 从实验性 selector 变成可复现机制。

- [ ] 合并大 runner 与增强版 selector；
- [ ] 多组 selection seed；
- [ ] ranking consistency；
- [ ] retention-aware Pareto；
- [ ] terminal confirmation；
- [ ] beam 只作为风险诊断和可选成本档位。

**验收条件：** 报告路径频率和 regret 后，不能依靠挑选最好路径；selector 在独立 seed group 上保持合理 agreement。

### P4：历史 memory 管理

目标：解决 frozen-old 的适应不足和 recluster 的历史漂移。

- [ ] bounded refresh；
- [ ] importance-based eviction；
- [ ] synthetic split/merge；
- [ ] cross-edge update；
- [ ] memory manager 的成本和稳定性分析。

**验收条件：** 在相同 memory growth 下同时改善 old retention 和 new plasticity，或明确展示 Pareto trade-off。

### P5：多阶段、官方 student 和外部基线

目标：确认方法不是三阶段或单一 student 的特例。

- [ ] base+4 主协议；
- [ ] 5/8 stage 长期遗忘；
- [ ] 多类别顺序；
- [ ] 官方 GraphCLIP 三锚点对齐；
- [ ] 统一 continual SOTA/相邻方法复现或严格差异说明。

**验收条件：** 结果可在至少多个数据集、多个阶段协议和多个 seed 下复现，才讨论论文主张范围。

## 12. 推荐的最终 v4 主线（候选，不是已冻结方法）

在全部实验之前，不直接把所有模块合并。推荐最终候选主线为：

1. fixed-rate 作为公平基线，adaptive-rate 作为方法变量；
2. delta-only 无标签联合聚类；
3. 每个 cluster 生成一个 synthetic node，绝不直接使用真实节点作为凝聚图节点；
4. 先用 matched-quota 比较 `dispersion` 与 `sqrt`；当前只把 `dispersion` 记为小图候选，不提前冻结为通用规则；
5. 用 diversity-aware text set compression 生成文本 prototype；
6. frozen-old topology 作为默认稳定控制；
7. 只对低质量旧 synthetic nodes 进行 bounded refresh；
8. SCA 用 response curve、retention、ranking consistency 和 cost 做选择，但 response-gain budget 尚未通过验证；
9. 通过 independent selector seeds 和 canonical seeds 分离验证；
10. 先以 3-stage 小图完成门禁，再以 base+4 incremental 作为大图主表，5/8-stage 作为长期记忆实验。

## 13. 论文贡献候选

在实验完成前只能作为候选表述：

1. **Auditable continual TAG benchmark protocol**：统一预算、阶段、student、数据访问、拓扑和评估口径。当前这是证据最充分的贡献候选。
2. **Synthetic-only complexity-aware allocation**：在不把真实节点混入凝聚图的前提下，按子簇规模、离散度和多模态程度分配合成节点容量；目前只有小图 5% matched-quota 证据。
3. **Information-adaptive continual condensation protocol**：暂列为待验证假设；adaptive-rate 当前跨数据集方向不稳定，不能作为已成立贡献。
4. **Student-response calibrated text compression**：在固定 token budget 下，用 student 可重训性评价文本集合压缩方式，尚未开始 P2 验证。
5. **Controlled historical memory management**：在 frozen-old、bounded refresh、split/merge 和淘汰之间分析保持与塑性的权衡，尚未开始 P4 验证。

最终是否保留某个贡献，取决于 P1-P5 的独立实验是否显示稳定收益；不能仅凭方法复杂度宣称创新。

## 14. 当前执行结论

P0 已完成，P1 已完成小图第一轮但尚未总体验收。下一步先完成 P1 小图门禁；门禁通过后才运行大图。P2 文本压缩必须在固定 rows、固定预算和固定 allocation 下单独进行；P3 SCA、P4 memory manager 和 P5 外部基线都不能与尚未冻结的 P1 配额规则同时改变。

当前论文不应宣称全面 SOTA。更稳妥的主线是：

> 以 synthetic-only continual memory 为约束，先建立可审计的 matched-quota 持续协议，再研究复杂度感知分配、文本压缩和历史记忆保持与新类塑性的机制权衡。
