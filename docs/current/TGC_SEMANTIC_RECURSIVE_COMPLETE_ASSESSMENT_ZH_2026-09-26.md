# Semantic-Recursive Continual TAG Condensation：完整方法、实验与论文评估

日期：2026-09-26  
状态：当前论文主方法候选已冻结；C0 保留为冻结共同控制。  
主方法名称：**Semantic-Conserved Recursive Append**（本文简称 semantic-recursive）。

## 1. 结论先行

当前可以把 semantic-recursive 固定为论文主方法候选，但不能把它表述为已经稳定超过所有基线或已经达到 SOTA。

大图 `base+4` 五阶段实验已经完成。统一协议下，semantic-recursive 相对配对 C0 的结果为：

| 指标 | 结果 |
|---|---:|
| 增量 stage cells | 12 |
| 正收益 cells | 9/12 |
| cumulative Macro-F1 平均变化 | **+0.858 pp** |
| new-class Macro-F1 平均变化 | **+1.764 pp** |
| old-class Macro-F1 平均变化 | +0.547 pp |
| 最差 cell | -2.381 pp |

这说明该方法已经有跨大图的方向性正向机制信号，尤其对部分新类塑性有帮助；但收益仍然依赖数据集和类别到达阶段。因此当前最准确的论文定位是：

> 在历史原始图不可回读的 continual TAG condensation 场景下，提出一种具有预算、前缀和 payload 守恒约束的语义递归追加算子，并系统研究其对新类塑性、历史保持和下游 student retrainability 的影响。

两张小图的递归统一基线比较随后完成。Cora/Citeseer 的两个递归顺序、9 个方法臂和 6 个配对 Student seeds 共 108 条 stage 记录显示：Semantic-Recursive 相对 C0 的最终 F1 平均为 **+0.925 pp**，Accuracy 平均为 **+0.364 pp**，4 个 dataset/order cell 中 3 个为正；相对 random、k-center/FPS、herding 和 medoid 的 paired F1 差异均为正，且 coreset 对照的置信区间不包含 0。直接 Student replay + delta fine-tune 的最终 F1 与 Semantic-Recursive 接近，但保留跨阶段模型参数、训练成本约为 2.9 倍，因此作为 stateful control 单列。完整表和协议边界见 [小图统一基线比较报告](TGC_SEMANTIC_RECURSIVE_UNIFIED_BASELINE_COMPARISON_ZH_2026-09-26.md)。

当前最重要的边界：

- **semantic-recursive**：论文主方法候选；
- **C0**：冻结共同控制、消融基线和回退版本；
- `k-center`、完整 raw graph、full-history synthetic：参考或离线对照；
- T+degree、role-aware、response selector、GraphGPS、文本 attention：历史探索或条件性候选，不属于当前主方法。

## 2. 问题定义与研究目标

### 2.1 Continual TAG condensation 的访问约束

数据以阶段形式到达。第 (t) 阶段可访问当前增量数据：

\[
\Delta_t=\{(x_i, e_i, y_i):i\in\text{current pool}\}.
\]

算法可以访问上一阶段合成 memory (M_{t-1})，但不能再次读取历史完整 raw graph。目标是在有限 memory 下形成可重训的 synthetic TAG，并支持后续类别到达。

这与静态 TAG condensation 不同：静态方法可以一次性看到完整图并重写全部合成节点；当前问题要求构造器在历史数据不可回读条件下继续追加。

### 2.2 Growing memory budget

当前主协议使用固定压缩率，而不是固定总行数：

\[
K_t=\lfloor \rho N_{\leq t}\rfloor,\qquad \rho=0.05,
\]

第 (t) 阶段新增行数为：

\[
b_t=K_t-K_{t-1}.
\]

因此 memory 随累计数据增长，同时保持约 5% 的压缩率。预算由 C0 的确定性 dispersion allocation 分配到新类别，不使用 response-gain 动态扩容。

## 3. 当前主方法的完整流程

当前方法可以概括为：

> **冻结 C0 topology + 局部 two-prototype structural proposal + class-local semantic conserved payload assignment + recursive append + original Student**

### Step 0：读取合法输入

第 (t) 阶段只读取：

1. 上一阶段 synthetic parent (M_{t-1})；
2. 当前阶段合法 raw delta (Delta_t)；
3. 当前阶段允许使用的节点属性、文本、边和类别信息。

历史 raw graph 不参与后续 block 构造。report/test 标签不参与候选选择。

### Step 1：构造图文联合 source space

对当前 delta 中的节点构造联合表示：

\[
z_i=[\operatorname{Norm}(x_i);e_i],
\]

其中 (x_i) 是节点属性，(e_i) 是冻结 MiniLM 文本 embedding。类别内根据 C0 的 rows allocation 形成 source slots：

\[
G_{c,1},G_{c,2},\ldots,G_{c,m_c}.
\]

source slot 是当前新数据的临时来源分组，不是将真实节点直接放入最终凝聚图。

### Step 2：局部 two-prototype split/merge

当前方法不完全重聚类所有新节点，而是在每个新类内做局部固定预算重组：

1. 计算每个 source slot 的图文联合离散度；
2. 找到离散度最高且样本数足够的 slot；
3. 将该 slot 拆为两个子簇；
4. 在其它 slot 中找联合中心最近的一对；
5. 将该近邻 pair 合并；
6. 通过“一拆一并”保持当前新增 rows 不变。

形式上：

\[
G_{c,r^\star}\rightarrow G_{c,r_1}\cup G_{c,r_2},
\]

同时：

\[
G_{c,p},G_{c,q}\rightarrow G_{c,p}\cup G_{c,q}.
\]

它的作用不是增加 memory，而是把固定容量从低复杂度区域转移到高离散度区域，缓解 Mean-X 对类内多模态的过度平均。

### Step 3：生成合成节点属性 X

对最终 source piece (G)，当前主方法使用确定性的 Mean-X：

\[
\tilde{x}_G=\frac{1}{|G|}\sum_{i\in G}x_i.
\]

重要说明：当前 semantic-recursive 不是 P-guided X 优化，也不是 student 反向优化 X。此前 P-weighted X、multi-prototype X 和 student-compatible X 都做过实验，但没有形成足够稳定的跨数据集收益，因此主方法保留简单、可审计的 Mean-X。

### Step 4：生成合成文本

每个 source piece 使用与 X 相同的来源成员：

1. 按联合中心距离选择最多 5 个代表文本；
2. 每个来源按约 320 字符预算截断；
3. 将代表文本拼接；
4. 使用冻结 MiniLM 对拼接后的文本重新编码。

\[
\tilde e_G=E_{\text{MiniLM}}(\operatorname{Concat}(\text{top-5 source texts})).
\]

这样做的关键是图属性和文本来自同一 source piece。它不是生成式摘要，文本来源可以追溯到当前合法 delta。

### Step 5：语义守恒 payload assignment

two-prototype structural proposal 产生若干新 rows 后，方法不简单地把 proposal 固定写回原 slot，而是在同一类别内部进行一对一匹配。

对 proposal (p_i) 和目标 slot (s_j) 构造图文联合代价：

\[
C_{ij}=\left\|
\operatorname{Norm}([\tilde x_i;\tilde e_i])-
\operatorname{Norm}([x_j^{slot};e_j^{slot}])
\right\|_2^2.
\]

通过 Hungarian matching 求：

\[
\pi^*=\arg\min_\pi\sum_i C_{i,\pi(i)}.
\]

被移动的是完整 payload，而不是单独的 X 或单独的文本。守恒单元包括：

- 合成 X；
- text embedding；
- 原始文本来源；
- mass；
- depth；
- stage metadata。

因此它满足：

\[
\operatorname{Multiset}(P_t^{out})=
\operatorname{Multiset}(P_t^{in}).
\]

它不复制、不删除、不重新平均已经生成的 structural payload，只在同类 slots 内进行确定性语义重排。

### Step 6：递归追加

stage 0 使用 C0：

\[
M_0^{SR}=M_0^{C0}.
\]

后续阶段使用上一阶段 semantic memory 作为 parent：

\[
M_t^{SR}=\operatorname{Append}(M_{t-1}^{SR},B_t^{SR}).
\]

其中 (B_t^{SR}) 只由当前 delta 构造。旧 synthetic prefix 保持 bitwise 不变，只有当前 block 被追加。

### Step 7：固定拓扑

当前主方法直接复用同阶段 C0 的 `edge_index`：

\[
E_t^{SR}=E_t^{C0}.
\]

因此 semantic-recursive 不改变：

- rows 数；
- edge 数；
- old-old edges；
- 当前 C0 的跨块连接规则；
- Student 的传播算子定义。

这一设计是有意的：此前拓扑、文本、X、selector 同时变化时无法判断收益来源。当前版本先固定拓扑，只检验“局部结构内容重组 + payload slot assignment + recursive parent”这条机制。

### Step 8：Student 重训和评价

所有主表方法使用同一个 Student：

- 两层 GCN；
- hidden/output dimension 384；
- 冻结 MiniLM；
- 可训练 text projection；
- 双向 graph-text InfoNCE；
- Adam，学习率约 `1e-3`；
- batch size 16；
- Student seeds 100--105；
- 每阶段从头训练，不继承上一阶段 Student；
- 大图本轮使用 2048 optimizer steps。

训练目标为：

\[
\mathcal L=\frac12(\mathcal L_{g\rightarrow t}+\mathcal L_{t\rightarrow g}).
\]

推理时使用真实图结构与类别 prompt 的图文相似度进行节点分类。Macro-F1 是节点分类的宏平均 F1，不是凝聚 loss，也不是监督 GCN 的分类训练结果。

## 4. 当前方法满足的四个不变量

当前算法可以形式化为四个 continual invariants。

### 4.1 Budget invariant

\[
|M_t|=K_t=\lfloor \rho N_{\leq t}\rfloor.
\]

算法不通过额外 rows 换取提升。

### 4.2 Prefix invariant

\[
M_{t-1}^{SR}\text{ 是 }M_t^{SR}\text{ 的不可改写前缀}.
\]

### 4.3 Payload conservation invariant

新增 block 内 structural payload 的 multiset 不因 slot assignment 而改变，只改变同类 slot 的归属。

### 4.4 Access invariant

\[
\operatorname{RawAccess}(t)\subseteq\Delta_t.
\]

构造阶段历史 raw reads=0；真实完整图只在下游评估中使用，不回流到凝聚器。

## 5. 已完成的主要实验与结果

### 5.1 冻结 C0 小图参考

Cora/Citeseer canonical、约 5% memory 的历史基线：

| 方法 | Cora 最终 F1 | Citeseer 最终 F1 | 解释 |
|---|---:|---:|---|
| C0 | 49.85 | 59.17 | 当前共同控制 |
| k-center | 46.35 | 55.34 | 强真实节点 coreset 参考 |
| recluster | 49.70 | 54.24 | 全部/历史重聚类参考 |
| full-history synthetic | 47.21 | 48.28 | 离线重凝聚参考 |
| 完整 pool、8192 steps | 51.31 | 55.60 | 完整 raw 训练参考 |

C0 在小图上不是理论上界，但在有限 memory 和统一 Student 条件下具有较强的任务效用。完整 pool 也不自动优于 synthetic memory，说明下游结果同时受训练步数、图结构、文本配对和 student 稳定性影响。

### 5.2 预算分配实验：P1

比较过 fixed-total、fixed-rate、hybrid、adaptive-rate、legacy fixed rows、dispersion allocation 和 response-based allocation。

代表性小图结果：

| 预算策略 | Cora | Citeseer | 结论 |
|---|---:|---:|---|
| fixed-rate | 47.28 | 57.75 | 保留为默认公平协议 |
| adaptive-rate | 49.16 | 54.90 | 方向跨数据集翻转 |
| response-gain extra | 46.00 | 53.43 | canonical regret，未晋升 |

固定 15 rows 还存在 keep/new 配额混淆。受控实验后，dispersion 在部分 cell 更有竞争力，但没有证明通用最优。因此当前方法固定 cumulative rate 和 C0 dispersion allocation，暂不让不稳定的 response signal 决定总预算。

### 5.3 X 构造和 coreset 实验

比较过 Mean-X、weighted-X、moment-X、P-guided two-prototype、k-center/medoid 和真实节点 FPS。

关键发现：

1. k-center 在个别阶段可以很强，但最终阶段并不稳定超过 C0；
2. 固定 C0 topology 后，k-center 的内容优势仍在部分阶段存在，说明差距不只是拓扑造成；
3. Mean-X 附近的简单加权并没有稳定缩小差距；
4. P-guided two-prototype 的开发集正收益在独立顺序上明显回落，new-class 也不稳定；
5. 当前最可靠的改进不是继续扫描 X 权重，而是把 X 与文本放在同一 source piece 中，再通过守恒 assignment 处理 slot 角色。

### 5.4 文本处理与 P2 实验

当前 C0 文本是 `top-5 + 320 chars + MiniLM re-encode`。比较过：

- concat + re-encode；
- mean pooling；
- text-only attention；
- text selection；
- attribute selection；
- graph-conditioned selection；
- graph pooling；
- shuffled graph query；
- student-guided attention。

P2-A 结果：

| 数据集 | center concat | graph selection | shuffled graph |
|---|---:|---:|---:|
| Cora final | 48.46 | 51.51 | 52.32 |
| Citeseer final | 54.97 | 53.12 | 53.94 |

graph selection 相对 center concat 在 Cora 为 `+3.05 pp`，但 Citeseer 为 `-1.85 pp`；真实 graph query 还低于 shuffled query。因此图条件文本接口没有通过跨数据集晋升门禁。

结论不是“所有 attention 都无效”，而是：当前邻域均值 query + MMR 选片无法稳定把图结构转化为更好的文本选择。主方法继续使用简单、可追溯的 center concat。

### 5.5 Response selector、TGC-R 和 student-compatible 优化

探索过：

- teacher proxy loss；
- short-horizon student response；
- response curve；
- response-gain extra budget；
- P-guided X；
- R2B/R2B2 student-compatible revision；
- guarded selector。

主要负结果：

1. 短训阶段的 ranking 不能稳定预测 canonical full-step ranking；
2. teacher loss、local message loss 和最终 all-class Macro-F1 并不等价；
3. selector 可以过滤明显坏候选，但经常召回不到真正的好候选；
4. response selector 成本显著高于低成本构造；
5. R2B/R2B2 存在局部正信号，但 gate 和长 horizon 不稳定；
6. response-gain extra 在 Cora/Citeseer 都没有形成稳定收益。

因此主方法不再使用 student response 选择候选，也不使用 response signal 控制预算。Student response 被保留为诊断工具，而不是当前算法的核心选择器。

### 5.6 拓扑实验

比较过：

- C0 topology；
- T+degree；
- simple cross-edge rewiring；
- topology-safe / message-conditioned edges；
- role-aware topology；
- GraphGPS student calibration；
- link prediction 下的 topology controls。

T+degree 的代表性结果：

| 数据集/任务 | 结果 |
|---|---:|
| BookChild 节点分类 | 约 +1.546 pp |
| BookChild link prediction AUC/AP | 约 +0.723 / +0.980 pp |
| arXiv 节点分类 | 约 -0.662 pp |
| arXiv link prediction AUC/AP | 约 -0.011 / -0.050 pp |

这说明拓扑机制具有任务和数据集依赖。GraphGPS 也没有在当前凝聚协议中形成足够收益，因此当前 semantic-recursive 固定 C0 topology，先隔离内容/slot assignment 的因果作用。

### 5.7 历史 memory maintenance

比较过 bounded refresh、old-row revision、C0-M core-buffer maintenance 和 relation maintenance。

代表性结果：

- 简单 bounded refresh 最终约 `-0.06/-2.64 pp` 的数据集差异；
- C0-M final 约 `-0.34 pp`，new-class 约 `-0.93 pp`；
- maintenance 相对 recursive 平均约 `-1.00 pp`，且构造成本增加。

结论是：仅根据 redundancy/novelty 或短训 relation loss 修改旧行，不能稳定解决历史保持和新类塑性。当前主方法保留 prefix，不引入旧 memory refresh。

### 5.8 Semantic permutation 与 role-aware 探索

semantic permutation 的核心启发是：改变内容 payload 所处的传播 slot，可能比继续修改 X 本身更直接地影响 GCN message passing。

小图递归确认：

| 范围 | cumulative 平均 | new 平均 | 正 cell | 最差 |
|---|---:|---:|---:|---:|
| Cora/Citeseer 8 cells | +1.640 | +2.672 | 6/8 | -0.528 |

独立顺序 confirmation 中，semantic permutation 平均约 `+1.878 pp`、8/12 正，但最差 cell 约 `-3.630 pp`，Citeseer new-class 仍不稳定。role-aware conserved assignment 独立确认平均约 `+1.468 pp`、7/12 正，最差约 `-2.042 pp`，也未通过严格晋升门禁。

这些结果说明：

- payload-slot placement 可能是真实机制；
- 守恒审计比旧的非守恒 role-match 更可靠；
- 但 semantic cost 还不能稳定预测最终 student utility；
- 不应仅凭开发集最好 cell 把 role-aware 权重或 response cost 写进主方法。

semantic-recursive 主方法采用更保守的版本：two-prototype structural proposal + 简单 semantic conserved assignment + recursive parent，不加入 role/message/plasticity 权重。

## 6. 最新大图五层比较

统一协议：三个大图、`base+4`、5% fixed-rate、batch 16、Student seeds 100--105、2048 steps。所有方法完成五阶段，协议审计 `all_pass=True`。

### 6.1 最终阶段 Macro-F1

| 数据集 | C0 | semantic-recursive | k-center | full-history synthetic | raw equal | raw 8192 |
|---|---:|---:|---:|---:|---:|---:|
| arXiv | 18.817 | **19.227** | 16.667 | 20.544 | 18.842 | 20.155 |
| BookChild | 12.607 | **13.313** | 11.712 | 12.407 | 18.018 | 17.217 |
| SportsFit | 25.013 | **28.076** | 27.840 | 26.467 | 26.769 | 29.502 |

raw equal/raw 8192 允许回读累计 raw graph，属于离线参考，不是合法 continual constructor。它们用于回答“完整真实图直接训练时性能位置在哪里”，不能作为 continual 主方法直接比较。

### 6.2 semantic-recursive 相对 C0

| 数据集 | 增量平均 | old 平均 | new 平均 | 正阶段 | 最差阶段 |
|---|---:|---:|---:|---:|---:|
| arXiv | +0.637 | -0.157 | **+2.971** | 3/4 | -0.421 |
| BookChild | +0.203 | -0.361 | **+2.241** | 3/4 | -1.552 |
| SportsFit | **+1.735** | **+2.158** | +0.080 | 3/4 | -2.381 |
| 合计 | **+0.858** | **+0.547** | **+1.764** | 9/12 | -2.381 |

结果不能解释为无条件优势，但支持三点：

1. 机制不只在小图有效；
2. arXiv/BookChild 的收益主要表现为新类塑性；
3. SportsFit 的收益更多来自旧类保持和阶段交互。

## 7. 为什么当前设计是合理的

当前设计不是把所有可能模块叠加，而是由负结果逐步约束出来的最小充分版本。

### 7.1 固定预算而非 response 动态预算

adaptive-rate 在 Cora 正、Citeseer 负；response-gain extra 在两个小图都没有稳定 canonical 收益。若继续让不稳定 response 控制 rows，会同时放大：

- 候选评价误差；
- 总 memory 差异；
- Student steps 差异；
- 选择成本。

因此 fixed-rate 是公平性和可解释性更强的默认协议。

### 7.2 保留 Mean-X 和简单文本

P-guided X、attention、graph-conditioned text 都有局部正例，但没有跨数据集稳定优势。复杂接口会把文本、X、student 和 topology 的误差叠加，反而无法判断机制来源。

当前先把增益来源限制在 source grouping、局部 split/merge 和 semantic payload placement，避免把不稳定的文本/特征学习器误认为凝聚算法本身的收益。

### 7.3 固定 C0 topology

拓扑改动会同时改变：

- message passing；
- degree normalization；
- old block 的有效传播；
- link prediction 与 node classification 的目标。

T+degree 的结果已经证明拓扑收益数据集依赖。因此当前固定 C0 topology，不是认为拓扑不重要，而是先避免拓扑混淆，并让 semantic assignment 的作用可审计。

### 7.4 递归 parent 而不是每阶段重新从 C0 开始

小图递归确认中，semantic recursive 的最差 cell 从 stage-local 的约 `-2.625 pp` 改善到约 `-0.528 pp`，说明上一阶段 synthetic payload 的状态会影响下一阶段追加。递归 parent 因此是 continual 语义的一部分，而不是实现细节。

### 7.5 守恒 assignment 而不是任意 role matching

此前非守恒 role-match 可能覆盖、复制或丢失 source payload，导致正收益无法解释。当前方法显式保存 payload multiset、source coverage、old prefix 和 topology hash，因此实验结果至少可以确定来自合法的 slot rearrangement，而不是 memory 内容数量混淆。

## 8. 当前方法真正解决的 challenge

### Challenge 1：历史 raw graph 不可回读

方法只用当前 delta 构造新 block，满足 continual access constraint；历史 raw reads=0 已通过审计。

### Challenge 2：memory 随数据增长但不能任意重写

固定压缩率使 (K_t) 随数据增长；recursive append 维护历史 prefix，避免每阶段全局重凝聚。

### Challenge 3：新类塑性与历史保持的冲突

two-prototype split/merge 将容量集中到高离散度区域；semantic slot assignment 尝试让新内容进入更适合的传播槽位。大图结果显示 new-class 平均 +1.764 pp，但仍有最差负 cell，因此是“部分缓解”而非完全解决。

### Challenge 4：图文 payload 的一致性

X 和文本来自同一 source piece，并在 assignment 时作为完整单元移动，避免只更新 X 或只更新文本导致图文错配。

### Challenge 5：持续凝聚结果的可审计性

预算、prefix、payload multiset、source provenance、topology hash 和 Student protocol 都可以单独检查。此前大量负结果说明，没有这种审计很容易把 batch、训练步数、配额或历史访问差异误当成算法提升。

## 9. 创新性与贡献的客观评价

### 9.1 创新点

当前最可信的算法创新点是一个 continual memory operator，而不是新的 GNN 或新的文本 encoder：

> **在历史 raw graph 不可回读的条件下，将局部复杂度保持的 split/merge、图文 payload 守恒匹配和递归 append 统一为一个可审计的 synthetic memory 更新算子。**

可以具体拆成：

1. **Continual setting contribution**：明确历史 raw inaccessible、growing memory、class-incremental TAG condensation 的协议和权限边界；
2. **Algorithm contribution**：固定 budget 下的局部 split/merge 与同类 semantic slot assignment；
3. **Invariant contribution**：budget、prefix、payload conservation、raw access 四类不变量；
4. **Empirical contribution**：展示 content、text、topology、response 和 maintenance 模块的正负边界，并在三个大图上给出方向性收益；
5. **Mechanistic contribution**：说明 synthetic payload 所处的传播 slot 可能比单纯的 embedding 几何距离更关键。

### 9.2 创新性强度

客观判断：当前算法新颖性属于**中等**，不是“强到仅凭方法就足以保证 WWW 长文”的程度。

主要不足：

- two-prototype split/merge 仍是启发式；
- Hungarian semantic matching 本身是成熟工具；
- topology 仍继承 C0；
- Mean-X 和 concat text 都不是新的表示学习方法；
- 尚无定理证明 semantic matching 必然改善 Student；
- 大图独立类别顺序和 construction seed 仍需补；
- 同协议强静态 TAG condensation 的大图适配仍需补；
- 尚不能宣称稳定优于 C0、k-center 或 TAGSAM。

因此，单独把方法写成“我们提出一种更好的凝聚算法”会偏弱；把它写成“continual TAG condensation 的受约束 memory operator + 机制分析 + 系统负结果”更可信。

### 9.3 是否解决了有用 challenge

是，但解决程度需要精确表述：

| Challenge | 当前判断 |
|---|---|
| 历史 raw 不可回读 | **真正解决并通过审计** |
| growing memory | **实现并验证** |
| old prefix/预算守恒 | **实现并验证** |
| payload 图文一致性 | **部分解决** |
| 新类塑性 | **平均改善，但未稳定解决** |
| 任务无关凝聚 | **尚未证明** |
| 稳定优于强 coreset | **尚未证明** |
| 稳定 SOTA | **不能声称** |

## 10. 能否作为 CCFA/WWW 长论文

### 当前可能性

当前版本已经具备一篇长文的研究故事：

- 问题场景清楚且区别于静态凝聚；
- 访问约束和协议可以审计；
- 有简单可靠基线 C0；
- 有 semantic-recursive 的大图正向信号；
- 有大量系统负结果，可以解释为什么复杂模块没有进入主方法；
- 有完整 raw、coreset、full-history synthetic 等参考层级。

但若按传统“新算法稳定优于强基线”的标准，当前还不够强。主要风险是：

1. 平均提升只有 +0.858 pp，最差 cell -2.381 pp；
2. 只有 9/12 大图增量 cell 为正；
3. static TAGSAM/STAD 尚未完成严格同协议的大图主表；
4. 独立类别顺序和 construction seed 仍缺；
5. 算法中的两个核心操作仍带启发式色彩；
6. 任务泛化尚未被 link prediction 等多任务结果充分证明。

### 论文定位建议

更合理的长文标题方向是：

> Continual TAG Condensation under Historical Raw-Data Inaccessibility: Semantic-Conserved Recursive Memory Append and Plasticity Analysis

主张应是：

- 提出一个合法、可审计、递归的 continual condensation operator；
- 平均改善新类塑性和部分大图下游效用；
- 系统揭示预算、X、文本、拓扑和 response selector 的失效边界。

不要主张：

- universal SOTA；
- 对所有数据集稳定超过 C0；
- semantic matching 理论最优；
- 已经证明任务无关。

## 11. 还可以优化什么

### 优先级 P0：确认，不改主方法

1. 大图独立类别顺序；
2. construction seed；
3. semantic-recursive 与 stage-local parent 消融；
4. 逐阶段 old/new F1、memory rows、构造时间和 Student 时间；
5. 静态 TAGSAM 的严格同协议适配；
6. 强 coreset 与完整 raw reference 的统一主表。

这一步决定当前收益是否能从“方向性信号”升级为“可泛化机制”。

### 优先级 P1：受限 message-aware semantic matching

如果要继续做算法改进，建议只做一个窄版本：

\[
C_{ij}=C_{ij}^{semantic}+\lambda C_{ij}^{message}.
\]

其中 (C^{message}) 使用 C0 固定拓扑下的局部 GCN message compatibility，而不是最终 F1。必须满足：

- 不改变 C0 topology；
- 不修改旧 prefix；
- 不访问历史 raw；
- 固定 (lambda)，不按 report F1 选择；
- 失败时回退到 semantic matching；
- 与 semantic-recursive 进行配对比较；
- 记录 new/old F1、最差阶段、移动行数和额外成本。

它的研究意义是把“语义匹配”进一步连接到真正的传播机制，但不能预先假设一定有效。

### 优先级 P2：新类塑性的确定性增强

不建议继续扩大 response selector。更低风险的方向是：

- 对新类内高离散度 source piece 设置最小多原型覆盖；
- 只允许同类内部的 message-compatible slot assignment；
- 引入 class-floor，避免新类的所有 rows 被合并到低复杂度区域；
- 将 proposal mass 和传播 role 作为约束，而不是作为最终 F1 选择器。

这些机制应先在固定单阶段上做因果诊断，再进入长链。

### 优先级 P3：文本和拓扑

文本方向只保留一个有界候选：固定 token budget 的 center-anchored attention。必须同时比较 center concat、text attention、graph-conditioned attention 和 shuffled graph。当前 P2-A 负结果不能被忽略。

拓扑方向先做单阶段 P3-B：固定 X、文本、rows 和 edge count，只改变跨块传播方向或边权；记录归一化旧块漂移、跨边 message norm、old/new F1 和 link prediction。只有两个数据集和独立顺序都通过，才考虑大图。

### 不建议当前继续的方向

- response-gain 动态预算；
- 复杂 adaptive-rate；
- 无界 old-memory refresh；
- 多候选 student selector；
- 同时修改 X、文本、拓扑和 Student；
- GraphGPS 作为新的 Student 主线；
- 只在最好类别顺序上继续调参。

## 12. 最终建议

现在可以冻结 semantic-recursive 为论文主方法候选，冻结 C0 为共同控制，并开始论文框架写作。但在提交前必须补齐：

1. 独立顺序/seed confirmation；
2. 静态 TAG condensation 强基线；
3. 不变量和复杂度的形式化；
4. 至少一个针对新类塑性的受限机制实验；
5. 完整成本和失败案例分析。

当前最诚实、也最有机会形成 CCFA/WWW 长文的贡献不是“所有模块都有效”，而是：

> 一个在严格 continual 访问约束下可运行、可审计、可递归的 TAG condensation memory operator，以及对其在塑性、保持、拓扑和 student response 之间关系的系统实证分析。

## 13. 证据文件

- 当前主方法运行器：[run_tgc_semantic_recursive_large_20260926.py](../../code/v2_llm_fusion/run_tgc_semantic_recursive_large_20260926.py)
- 大图 semantic-recursive 报告：[TGC_SEMANTIC_RECURSIVE_LARGE_CONFIRMATION_ZH_2026-09-26.md](TGC_SEMANTIC_RECURSIVE_LARGE_CONFIRMATION_ZH_2026-09-26.md)
- 大图五层比较：[TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md](TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md)
- 大图结果汇总：[tgc_large_baseline_comparison_20260926.json](../../results/tgc_large_baseline_comparison_20260926.json)
- 大图协议审计：[tgc_large_baseline_protocol_audit_20260926.json](../../results/tgc_large_baseline_protocol_audit_20260926.json)
- 小图递归确认：[TGC_SEMANTIC_RECURSIVE_CONFIRMATION_ZH_2026-09-26.md](TGC_SEMANTIC_RECURSIVE_CONFIRMATION_ZH_2026-09-26.md)
- 当前历史探索总览：[CURRENT_PROGRESS_AND_BEST_METHOD_SUMMARY_ZH_2026-09-25.md](CURRENT_PROGRESS_AND_BEST_METHOD_SUMMARY_ZH_2026-09-25.md)
- P1 预算结果：[P1_SMALL_GRAPH_RESULTS_ZH_2026-09-15.md](P1_SMALL_GRAPH_RESULTS_ZH_2026-09-15.md)
- P2-A 文本结果：[P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md)
- P3-A response 结果：[P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md](P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md)
- P4 memory maintenance：[C0M_CORE_BUFFER_MAINTENANCE_COMPLETION_ZH_2026-09-22.md](C0M_CORE_BUFFER_MAINTENANCE_COMPLETION_ZH_2026-09-22.md)
- T+degree 对比：[DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md](DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md)

## 14. P0 独立顺序与 construction seed 验证更新（2026-09-26）

P0 已完成，完整结果见：[TGC_P0_INDEPENDENT_ORDER_CONFIRMATION_ZH_2026-09-26.md](TGC_P0_INDEPENDENT_ORDER_CONFIRMATION_ZH_2026-09-26.md)。本轮不改变 semantic-recursive、C0 或 Student 协议。

### 14.1 Seed 确定性

arXiv、BookChild、SportsFit 分别使用 construction seeds `9001/9002/9003` 重新构造。三个数据集的五阶段 memory hash 均 bitwise 一致；没有 Student 训练、没有 F1 选择、历史 raw reads=0。因此 canonical 结果不是 construction seed 偶然性造成的，semantic-recursive 的构造过程可以作为确定性、可审计算子报告。

### 14.2 独立类别顺序

独立顺序 seed 为 `49013`，采用同一 `base+4`、5% fixed-rate、batch=16、steps=2048 和 Student seeds=100--105。12 个增量 stage cells 的结果为：

| 数据集 | cumulative 增量平均 | new-class 平均 | 正收益 | 最差 cell |
|---|---:|---:|---:|---:|
| arXiv | -0.373 pp | -2.385 pp | 1/4 | -0.914 pp |
| BookChild | +0.402 pp | +1.327 pp | 2/4 | -0.760 pp |
| SportsFit | +0.030 pp | +1.440 pp | 2/4 | -2.434 pp |
| 合计 | **+0.020 pp** | -- | **5/12** | **-2.434 pp** |

这一步没有复现 canonical 顺序上的 `+0.858 pp、9/12 正 cells`。因此主方法可以固定为论文候选和机制研究对象，但不能继续写成跨类别顺序稳定提升。当前最准确的表述是：semantic-recursive 在部分数据集和顺序上能改善新类可重训性，同时暴露出明显的顺序依赖和 old/new trade-off；C0 继续保留为共同控制和稳定回退版本。

P0 的最终状态是“seed invariance 通过、independent-order generalization 未通过”。后续不应根据这一个独立顺序反向调参；应先完成强 baseline 主表和成本审计，再决定是否只加入一个有界 message-aware matching 或新类 class-floor / 多原型覆盖诊断。
