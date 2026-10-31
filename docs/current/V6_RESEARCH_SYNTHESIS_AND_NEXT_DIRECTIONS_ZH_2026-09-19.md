# v3-v6 全部实验归纳与后续方法方向

日期：2026-09-19。

> **后续状态（v7-A 已结项）**：本文第 5--9 节提出的 gradient-compatible append 仅是 v6 阶段的待验证假设。v7-A 已在 Cora/Citeseer 的 8 个小图反事实 cell、144 次正式 fit 和独立审计中验证：selector 平均 Spearman `0.125`、selected gain `0.00 pp`，未通过晋升门禁；候选局部正收益不能拼接为主方法。当前应以 [v7-A 结项报告](V7A_GRADIENT_COMPATIBLE_APPEND_COMPLETION_ZH_2026-09-19.md)和 [v7-A 冻结协议](V7A_GRADIENT_COMPATIBLE_APPEND_FROZEN_PROTOCOL_ZH_2026-09-19.md)为准，保留 C0，不启动 v7-B。

## 1. 总判断

当前最可靠的运行版本仍是 C0：

`5% fixed-rate growing budget + dispersion allocation + Mean-X + joint_g5_c320 concat/re-encode + frozen-old memory/topology + current text-nearest cross-edge + response off`。

从 v3 到 v6-A，没有一个新增组件在多数据集、多顺序和独立 seed 下同时满足性能、稳定性和成本门禁。这不是实验没有价值，而是说明当前问题的瓶颈不是某个孤立模块，而是三个层次之间没有对齐：

1. **合成节点表示**：Mean-X 将类内多模态压成一个均值；k-center/coreset 的覆盖优势又不能直接转化为合成节点；
2. **训练动力学**：短训 response、固定传播漂移和单个 student 的排序不能稳定预测从头训练后的最终效果；
3. **持续 memory 管理**：旧 raw graph 不可回读、旧 synthetic 行不可随意改写、类别规模和分布不断变化，导致固定预算和局部修补都容易失配。

因此，后续不应再把已有 response、attention 或 frozen topology 重新命名为新方法。更有潜力的主线是：

> **Evolving Synthetic Memory + Gradient-Compatible Append + Joint Text-Attribute Kernel Preservation**

中文可称为“可演化合成记忆与训练动力学兼容追加”。这只是待验证的研究假设，不是已经成立的算法或 SOTA 声明。

## 2. 全部实验的证据地图

|方向|原始假设|实际结果|当前决策|
|---|---|---|---|
|v3/SCA response curve|多 horizon、多 batch 的 student response 能可靠选择新增 block|早期 canonical 阶段有效，但后续出现 ranking 失配、seed shift 和 terminal dynamics 失配|保留为诊断，不作为默认选择器|
|frozen-old topology|冻结旧行/旧旧边可以减少 append 后历史传播漂移|确实能控制指定静态传播漂移；但 v5 证明静态漂移小不等于 scratch student 的真实 F1 保持|保留冻结思想，不再单独宣称收益|
|P1 budget/response allocation|固定总 rows 下，response 可以找到更好的类间配额|Cora/Citeseer 平均有正有负，接受率低，构造约慢 75--130 倍|不晋升，作为昂贵消融|
|P1-X Weighted/Moment-X|Mean-X 的方差收缩是主要瓶颈|Weighted-X 近零收益，Moment-X 下降，未缩小 k-center 差距|Mean-X 仍为默认；改进 X 需多原型而不是换均值权重|
|P2 token/set/text compression|固定 token 预算下，pooling、DeepSets、dual prototype 能超过 concat|严格重跑后没有稳定优势；mean/DeepSets 不能稳定改善|center concat 保留|
|P2-A graph-conditioned text|真实图邻域 query 能选出更有用的文本|Cora 有局部收益，Citeseer 下降；真实 graph query 甚至低于 shuffled query|不晋升|
|P2-B student-guided attention|用 student 反向梯度选择文本比语义规则更贴合训练|收益接近零且增加明显构造成本|不晋升|
|P3 topology|跨边或更复杂拓扑可以修复持续塑性|删掉当前跨边会下降，说明跨边有用；random/quality/漂移约束不稳定|保留 current text-nearest cross-edge，不继续堆手工规则|
|P4 refresh/core-buffer|有限刷新旧行可以修复历史过时|Cora 接近零，Citeseer 下降；低 mass 不等于应刷新|默认 frozen-old|
|P5 rate/order|动态 rate 或更长 chain 能证明持续方法更优|rate 非单调，顺序敏感，长链回落，大图有数据集依赖|暂不扩展大图主方法|
|v5 retention|永久 core/buffer + 传播漂移界兼顾保持与塑性|core 有局部正信号，但 topology/full 明显退化，未过总门禁|v5 不晋升|
|v6-A local marginal|局部 merge/split 比整类重新聚类更适合边际配额转移|local 在 8/8 cell 回退 C0；recluster 在 5/8 cell 有正收益，但排序仍不稳定|v6-B 不启动；recluster 作为新假设候选|

## 3. 失败结果共同指向的根因

### 3.1 静态输入保持不等于 student 保持

v5 的传播漂移约束控制的是固定 student 参数下的输入算子变化，但每个阶段 student 都从头训练。真正需要控制的是训练动力学：

\[
\left\|\nabla_\theta L(\theta;M_t)-\nabla_\theta L(\theta;M_{t-1})\right\|
\]

而不是只有：

\[
\left\|S_t-S_{t-1}\right\| \quad\text{或}\quad
\left\|S_tX_t-S_{t-1}X_{t-1}\right\|.
\]

如果 student 被重置，输入算子的小漂移不能保证参数轨迹、分类 margin 或真实图 F1 小漂移。这解释了为什么 topology/full 可以通过传播审计，却在最终分类上退化。

### 3.2 response 的问题不是“候选太少”，而是排序对象错位

response selector 同时受到以下误差影响：

- 短训练与完整训练的 ranking shift；
- synthetic-memory 训练与 real-graph 推理的 support shift；
- selection seed 与 report seed 的分布差异；
- candidate construction、probe training 和正式 student 成本不对称。

v6-A 的 local family 将 candidate 改得很小，但排序仍然不稳定，说明仅仅把动作变成“一行边际修改”无法修复这个问题。recluster 的正收益说明候选空间可能比 local 操作更重要。

### 3.3 Mean-X 的问题是覆盖不足，而不是简单的均值公式

Weighted-X 和 Moment-X 没有稳定改善，说明单纯改变均值权重或加入主方向并不能解决：

- 类内多模态；
- 决策边界附近的少数模式；
- 文本、属性和拓扑共同决定的可训练性；
- 合成图节点数有限时的模式覆盖。

更合理的方向是每个类别维护多个**合成原型**，并让每个原型保留质量、离散度、文本覆盖和图结构统计，而不是依赖一个全类均值。

### 3.4 文本不是独立模块，文本、X 和拓扑存在耦合

P2 的 attention 失败并不证明文本无用，而是说明“先固定 X/拓扑，再让 attention 单独选文本”可能不是正确因果顺序。TAGSAM 的静态设计把 text selection 和 attribute similarity matching 放在同一压缩过程里；当前项目将它们拆成多个独立模块后，文本选择没有得到稳定增益。

### 3.5 固定预算规则过于粗糙，但 response 不是合适的第一步

固定每类 rows、固定 5% 或单纯按样本数分配都忽略了新信息的复杂度；但直接让昂贵 student response 决定预算又会放大选择噪声和成本。预算应先由可审计的覆盖/新颖度统计决定，再用低成本训练动力学指标做有限校验。

## 4. 相关论文能提供什么启发

### 4.1 经典 gradient/trajectory matching

- [GCond: Graph Condensation for Graph Neural Networks](https://arxiv.org/abs/2110.07580)：通过模拟 GNN training trajectory，同时凝聚节点特征与结构。它说明“训练动力学”比静态特征距离更接近 condensation 目标，但也带来高成本和模型依赖。
- [CTRL: Two Trades is not Baffled](https://arxiv.org/abs/2402.04924)：指出仅匹配梯度方向会累积误差，强调更合理的初始化和 trajectory matching。它支持当前项目把重点从局部结构修补转向训练动力学兼容性。
- [Lossless Graph Condensation via Expanding Window Matching](https://arxiv.org/abs/2402.05011)：使用更丰富的 expert supervision 和 expanding-window trajectory matching，说明单一短 horizon 不足以代表完整训练过程。它支持“多 fidelity、但只对少量候选做完整匹配”，而不是对所有候选都跑 response。

### 4.2 低成本、模型无关和覆盖型方法

- [Bonsai: Gradient-free Graph Condensation](https://arxiv.org/abs/2410.17579)：以 computation tree coverage 为基本对象，强调 gradient-free、model-agnostic 和线性成本。它直接对应当前 Mean-X/k-center 困境：真正需要覆盖的是 GNN 的局部计算树，而不只是节点 feature 中心。
- [Structure-free Graph Condensation](https://arxiv.org/abs/2306.02664)：把拓扑信息编码进 graph-free synthetic data，减少显式结构优化。它提示可以把一部分拓扑关系转为合成属性或 kernel，而不是继续手工设计 cross-edge。
- [SimGC: Simple Graph Condensation](https://arxiv.org/abs/2403.14951)：使用更简单的 input-to-prediction alignment，减少复杂外部参数并提升效率。它支持先寻找低成本 utility surrogate，再考虑昂贵 bilevel/response。

### 4.3 持续、开放世界和演化凝聚

- [CaT: Balanced Continual Graph Learning with Graph Condensation](https://arxiv.org/abs/2309.09455)：每轮先凝聚 incoming graph，再将 condensed memory 用于训练，重点解决历史 memory 太小导致的分布失衡。它提示当前项目应明确区分“持续 condensation”与“每阶段 scratch student 评估”两个目标。
- [Graph Condensation for Open-World Graph Learning](https://arxiv.org/abs/2405.17003)：把结构感知的 distribution shift 和时间不变模式纳入 condensation。它支持为新阶段 memory 保存 novelty/shift 统计，而不是只记录 rows 和 mass。
- [Scalable Graph Condensation with Evolving Capabilities (GECC)](https://arxiv.org/abs/2502.17614)：使用 class-wise evolving clustering，并继承旧 condensation 结果作为新阶段 clustering centroids。这与当前 v6 的 recluster 正信号最接近，是后续最值得直接对照的持续凝聚思路。

### 4.4 TAG 与评测方法

- [TAGSAM: Text-attributed Graph Condensation via Text Selection and Attribute Matching](https://arxiv.org/abs/2606.03839)：以 subgraph text selection 和 attribute similarity matching 为核心。对本项目的启发不是再做一个 attention，而是联合保持文本/属性相似矩阵，并在 append 时保留旧块矩阵、只求解新增 block。
- [GC-Bench](https://arxiv.org/abs/2407.00615) 与 [GCondenser](https://arxiv.org/abs/2405.14246)：强调 effectiveness、transferability 和 complexity 的统一评价。当前项目已有成本和审计基础，后续应补跨 student/模型迁移，而不是只报告单一 GCN 的 Macro-F1。
- [Graph Condensation: A Survey](https://arxiv.org/abs/2401.11720)：将 GC 按 effectiveness、generalization、efficiency、fairness、robustness 组织。当前论文可以把 continual TAG 的历史保持、顺序鲁棒性和成本作为独立评价轴。

上述论文说明的共同趋势是：更强的方法正在从“优化一个 synthetic graph”转向“覆盖 computation behavior、处理 distribution shift、降低 model dependence 和报告真实成本”。

## 5. 最值得尝试的新核心：Gradient-Compatible Append

建议不要把下一版定义成更多 response curve，而是定义一个低成本的**训练动力学兼容追加**。

### 5.1 候选 memory

在每个阶段只构造少量候选：

1. C0 dispersion + Mean-X；
2. 受影响类别完整 recluster；
3. 多原型 barycentric-X；
4. 可选的 TAGSAM-style text/attribute kernel candidate。

不再枚举所有 donor/recipient，也不把 local 三槽位操作作为主要候选。

### 5.2 低成本动力学分数

令 \(M_{t-1}\) 是旧 synthetic memory，\(M_t\) 是追加候选。在相同 student 初始化 \(\theta_0\) 和相同 batch 下，计算 0/1/2 个训练 step 的梯度 sketch：

\[
D_{dyn}(M_t,M_{t-1})
=
\frac{1}{K}\sum_{k=0}^{K}
\left(1-
\cos\left(g_k(M_t),g_k(M_{t-1})\right)\right),
\]

其中 `g_k` 可以是 student 参数梯度的随机投影或分层范数，而不是完整 state。对新类再计算覆盖/边界收益：

\[
U_{new}=
\text{coverage}+
\text{class-margin}+
\text{text/attribute coverage}.
\]

候选分数为：

\[
Q(M_t)=U_{new}
-\lambda D_{dyn}
-\mu\,\text{cost}.
\]

这不是监督分类 CE；它仍然服务于图文 InfoNCE student，只把“保持”从静态拓扑漂移换成训练梯度兼容性。

### 5.3 为什么它比当前 response 更合理

- 它直接约束 scratch student 的训练动力学，而不是假设静态传播漂移能推出 F1 保持；
- 只跑 0/1/2 steps，成本远低于 2048-step response curve；
- 可在 C0、recluster、多原型之间做统一排序；
- 可加入旧 memory 与新 block 的梯度分解，显式测量保持/塑性冲突；
- 与 GCond、CTRL、trajectory matching 的理论方向一致，但不要求对全部候选执行完整 bilevel 优化。

## 6. 第二个核心：多原型合成节点，而不是换一种均值

### 6.1 合成节点约束

仍然不把真实节点直接作为凝聚图节点。源样本只用于初始化和离线审计，最终节点必须是合成的。

对于类别/子簇 (c)，先用 k-means++ 或 medoid 只做初始化，再生成合成中心：

\[
X_{c,j}=\mu_{c,j}+U_{c,j}a_{c,j},
\]

其中 \(\mu\) 是合成 barycenter，\(U\) 是类内低秩方向，\(a\) 取受约束的内部分位点。每个 synthetic row 额外保存：mass、离散度、文本覆盖、邻域度和低秩不确定性；不保存原始节点作为部署 memory。

### 6.2 与当前结果的关系

P1-X 已经说明 Weighted-X/Moment-X 不够。下一步不是继续调权重，而是比较：

1. Mean-X；
2. 每类多 prototype barycenter；
3. barycenter + residual basis；
4. 只在 synthetic 节点上增加 virtual views 的版本。

所有版本固定 rows、文本、拓扑和 student；先不加入 response selector，避免无法归因。

## 7. 第三个核心：联合 text/attribute kernel，而不是独立 attention

定义 synthetic rows 的联合相似矩阵：

\[
K=\alpha K_X+\beta K_T+\gamma K_G,
\]

其中 (K_X) 是合成属性相似度，(K_T) 是文本 embedding 相似度，(K_G) 是局部结构/邻域相似度。

持续追加时：

- 旧块 (K_{old,old}) 保持不变；
- 新块 (K_{new,new}) 由当前合法 pool 构造；
- 只求解 (K_{old,new}) 的稀疏、degree-constrained cross block；
- 文本选择直接使用 TAGSAM-style 的代表性 + 多样性/互信息规则；
- 暂不使用 student 反向 attention。

这样可以把文本和拓扑放回同一个可审计对象中，避免 P2 attention “文本单独变好但 graph-text correspondence 变差”。

## 8. 第四个核心：novelty-gated memory 和动态预算

动态预算仍值得研究，但必须建立在可解释的新颖度上，而不是直接把 F1 gain 映射成 rows。

对每个新阶段计算：

- 类内 dispersion；
- 与旧 synthetic prototypes 的最小距离；
- 新旧 text/attribute kernel shift；
- 新阶段 computation-tree coverage 缺口；
- 新类/旧类的 mass 与多模态程度。

再使用：

\[
K_t=\max\left(K_{t-1}+b_{min},\lfloor \rho_tN_t\rfloor\right),
\]

其中 \(\rho_t\) 由 novelty 分段映射得到，并设置上下限。预算需要同时报告：

- F1 vs rows；
- F1 vs compression ratio；
- old/new F1 vs stage；
- construction + selection + student 总成本；
- memory bytes 和 edge bytes。

第一版只做离散三档 novelty budget，不做可训练 policy，也不使用完整 student response。

## 9. 新的分阶段路线

旧 P1-P5 已经混入历史实验，不建议继续沿用编号。建议从 v7 重新编号：

### v7-A：动力学兼容性小图验证，最高优先级

固定 C0、recluster、多原型三个候选；加入 0/1/2-step gradient sketch；比较它与 geometry、full report oracle 的 rank correlation、regret、成本和 old/new gain。

通过条件：

- independent report Spearman >= 0.3；
- 选择后的平均 gain >= 0.5 pp；
- 成本 <= C0 的 2 倍；
- 两个数据集的新顺序均不负；
- 不使用 test，不扩大候选到几十个。

### v7-B：多原型 synthetic X

只改 X，固定 allocation/text/topology/student。重点测 mode coverage、class-margin、k-center gap 和下游跨 student 迁移。

### v7-C：联合 text/attribute kernel

采用 TAGSAM-style text selection + attribute similarity matching；与当前 concat、P2 mean/DeepSets、P2-A/P2-B 负结果同表比较。

### v7-D：novelty-gated budget 与受限 memory update

先做离散 rate，不做 response-gain policy；只有在 v7-A/B/C 至少一个模块成立后，才加入预算增长或 old-row refresh。

### v7-E：强基线、跨 student 和大图

在小图上先补齐 TAGSAM、GCond/trajectory、Bonsai/selection、CaT/GECC 思路可比的协议层；统一 student、预算、condensation cost、storage 和 transferability。只有小图门禁通过后再做 arXiv/BookChild/SportsFit base+4。

## 10. 哪些方向暂时停止

暂时不推荐：

- 对每个候选继续跑完整 response curve；
- 再做一轮普通 attention、DeepSets 或 student-guided attention；
- 继续增加 random/quality/frozen topology 手工规则；
- 用低 mass 规则刷新旧 memory；
- 未做 matched quota 时直接宣称动态预算优越；
- 用某个 report oracle 候选拼接出新的持续主表；
- 继续在 Cora/Citeseer 上无止境搜索超参数。

## 11. 论文定位建议

当前有两条诚实路线：

### 路线 A：方法论文

以“训练动力学兼容的演化 synthetic memory”为核心，只保留一个新机制：gradient-compatible append；多原型和 kernel preservation 作为必要的构造组件。必须先通过 v7-A 的小图门禁，再谈主方法。

### 路线 B：机制/benchmark 论文

如果 v7-A 仍失败，则把贡献定位为：

- 严格的 continual TAG condensation 协议；
- append-only synthetic memory 的可审计实现；
- response、文本、X、拓扑、refresh、局部编辑的系统正/负结果；
- 预算、顺序、student transfer 和真实总成本的评价框架。

在当前证据下，路线 B 比强行声称一个简单模块具有普适新颖性更稳妥；路线 A 只在新的核心机制通过独立验证后启用。
