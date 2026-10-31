# Semantic-Recursive 方法审查与下一轮推进计划

日期：2026-09-26  
定位：对当前主方法的客观审查、方法解释补强和 WWW/CCFA 投稿前路线设计。

## 1. 结论先行

当前 Semantic-Recursive 可以作为论文主线和冻结实验基线，但还不应被描述成已经完成理论闭环的最终算法。

当前证据是：

| 证据 | 结果 | 含义 |
|---|---:|---|
| Cora/Citeseer 小图统一比较 | 相对 C0 Final Macro-F1 `+0.925 pp`，Accuracy `+0.364 pp`，3/4 cell 为正 | 小图上有方向性收益 |
| arXiv/BookChild/SportsFit canonical `base+4` | 相对 C0 增量平均 `+0.858 pp`，new-class `+1.764 pp`，9/12 正 | 大图上存在可复现机制信号 |
| P0 construction seed | 3 个 seed 的五阶段 memory hash 完全一致 | 构造确定性通过 |
| P0 independent order | 增量平均约 `+0.020 pp`，5/12 正，最差 `-2.434 pp` | 跨顺序泛化未通过 |

因此当前最准确的定位是：

> 一个针对历史 raw 不可回读 continual TAG condensation 的、具有正向机制信号但仍有明显顺序依赖的语义守恒递归 memory operator。

它不是当前已经稳定超过所有强基线的 SOTA 算法。方法贡献有价值，但要提高 WWW/CCFA 的说服力，关键不是再增加更多组件，而是把局部 split/merge 和 payload assignment 从启发式规则提升为有明确目标、约束和可验证收益的算法。

## 2. 当前方法真正存在的解释缺口

### 2.1 为什么一定要拆

当前规则是：在类内找到联合图文离散度高的 source slot，将其拆成两个子簇。直觉是高离散度通常意味着类内多模态，Mean-X 会把多个模式平均成一个难以学习的 prototype。

但“高离散度”只是一个启发式相关量，不是充分条件。当前没有证明：

- 拆分后的两个子簇确实比原 slot 更能降低图文重构误差；
- 拆分不会产生两个过小、过噪的 synthetic nodes；
- 拆分后的 payload 在 C0 topology 中有足够不同的传播作用；
- 拆分收益能预测最终 Student 的 new-class F1。

所以论文不能只写“高离散度所以拆”，需要把拆分定义成一个可计算的边际收益最大化问题。

### 2.2 为什么一定要合

当前为了保持新增 rows 不变，在拆出一个 slot 后把另一个 pair 合并。这个“一拆一并”本质上是 budget-preserving 操作，避免方法通过增加 memory 获得收益。

问题在于：如果合并对象只是“中心最近的两个 slot”，它并不保证合并代价最小，也不保证被合并区域对当前 Student 不重要。论文需要说明合并是一个受约束的最小损失操作，而不是为凑 rows 的实现细节。

### 2.3 固定 rows 的意义是什么

固定新增 rows 有一个明确的实验意义：它把比较条件固定在同一 memory budget，使收益不能解释为“方法保留了更多数据”。这对因果归因和公平 baseline 很重要。

但固定 rows 不是自然定律，也不是性能最优的证明。它同时带来一个局限：拆分和合并必须互相抵消，可能阻止本来应该增加容量的高复杂度类别获得更多 rows。

正确表述应该是：

> 固定 rows 是用于隔离结构/语义算子影响的预算守恒约束；它不是宣称固定 rows 在所有数据集上最优。预算增长策略应在单独实验中研究。

### 2.4 Semantic assignment 为什么可能有效

当前方法把完整 `(X, text, mass, provenance)` 作为 payload，在同类 slot 内做一对一语义匹配。其合理性是：X 和文本必须共同移动，避免只调 X 或只调文本造成图文错配；同时 payload multiset 不变，便于审计。

但当前 matching cost 主要是图文联合空间距离，没有直接描述 GCN message compatibility。因此它只能作为“语义角色对齐”的 proxy，不能证明一定改善下游 Student。P0 的顺序依赖说明这一 proxy 对 topology 和类别到达顺序仍然敏感。

## 3. 是否应该保留 C0，还是论文中不提 C0

### 3.1 不建议隐藏 C0

C0 不是公开文献 baseline，但它是项目内部冻结的共同控制。论文中应该保留它，原因有三点：

1. Semantic-Recursive 的增益必须有同资源、同 topology、同 Student 的消融对照；
2. C0 是失败时的回退版本，能帮助解释复杂模块是否真的必要；
3. 隐藏 C0 会让读者无法判断收益来自 Semantic-Recursive，还是来自另一份未说明的实现变化。

### 3.2 正确写法

不要把 C0 写成公开强基线，也不要把它放在“现有方法排行榜”中。建议命名为：

- `C0 internal control`；
- `frozen C0 constructor`；
- `C0 ablation / fallback`。

公开方法放在另一张表：TAGSAM、STAD、GraphCLIP、可复现的静态或 continual TAG 方法。stateful Student 和监督 STAD 单独列为不同权限参考。

### 3.3 是否需要重新设计主方法

建议采用“保留主线、重写一个核心算子”的路线：

- 当前 Semantic-Recursive 作为 `SR-base`，不删除，不改历史结果；
- 新版本只替换当前没有充分解释的 split/merge 规则；
- 不同时加入 response selector、dynamic budget、old-memory refresh、GraphGPS 和 text attention；
- 新版本如果没有通过预注册门槛，论文仍以 SR-base + 机制负结果为主线。

这比重新从头设计一个完全不同的方法更稳，也能保留当前已有的小图和大图证据。

## 4. 建议的新核心算子：Budget-Preserving Adaptive Refinement

暂定名称：**Budget-Preserving Adaptive Semantic Refinement (BASR)**。它不是立即替换主方法的版本，而是下一轮最小改动候选。

### 4.1 目标

给定一个类别内的 source slots \(\{P_j\}\) 和固定新增 rows (K)，选择一次 split 和一次 merge，使类内的图文联合失真下降，同时满足：

- rows 数不变；
- 每个 piece 的最小 mass 不低于阈值；
- X 和文本仍来自同一 piece；
- 不读取历史 raw；
- 不使用最终 F1 选择操作。

### 4.2 Split gain

对候选 piece (P)，先计算其图文联合失真：

\[
D(P)=\frac{1}{|P|}\sum_{i\in P}\|z_i-\bar z_P\|_2^2.
\]

对一个确定性二分 (P\rightarrow(P_a,P_b))，定义 split gain：

\[
G_{split}(P)=D(P)-\frac{|P_a|D(P_a)+|P_b|D(P_b)}{|P|}-\lambda_{min}R_{min}.
\]

其中 (R_{min}) 是子簇过小惩罚。只有当 (G_{split}>0) 且两个子簇都满足最小 mass 时，piece 才有资格被拆。

这个定义把“为什么拆”改成了：

> 拆分必须带来可测量的图文联合失真下降，并且不能牺牲最小覆盖。

### 4.3 Merge cost

对同类候选 (P_a,P_b)，定义合并代价：

\[
C_{merge}(P_a,P_b)=D(P_a\cup P_b)-\frac{|P_a|D(P_a)+|P_b|D(P_b)}{|P_a|+|P_b|}+\lambda_{role}R_{role}.
\]

合并对象不是简单取中心最近 pair，而是在满足同类、最小 mass、来源不重复和 role 约束后，选择代价最小的 pair。

### 4.4 固定预算下的联合选择

在所有可行的一拆一并组合中，选择：

\[
(P^\star,Q^\star)=\arg\max_{P,Q}\left[G_{split}(P)-\mu C_{merge}(Q)\right].
\]

如果没有组合使目标严格为正，则回退到 C0 的原始 source slots，不进行 split/merge。这样“拆”和“合”都有相同的预算守恒目标，不再是两个孤立启发式动作。

### 4.5 与当前 Semantic-Recursive 的关系

| 项目 | SR-base | BASR 候选 |
|---|---|---|
| memory rate | 5% fixed-rate | 不变 |
| 新增 rows | C0 allocation | 不变 |
| X | Mean-X | 不变 |
| text | top-5 / 约 320 chars / re-encode | 不变 |
| topology | C0 topology | 不变 |
| parent | recursive semantic memory | 不变 |
| assignment | semantic conserved payload | 不变 |
| split rule | 高离散度 slot | 最大正 split gain |
| merge rule | 最近 pair | 最小 merge cost |
| response selector | 关闭 | 关闭 |
| fallback | 无明确 objective fallback | 无正收益时回退 SR-base/C0 |

BASR 的价值在于增强可解释性和可证伪性，而不是保证一定提高 F1。只有通过独立顺序和多 seed 验收后，才考虑把它升级为主方法。

## 5. 下一轮实验设计

### P0：当前方法解释审计，不改变主方法

在已有 Cora/Citeseer 和三个大图 memory 上保存每个新类、每个 slot 的：

- (D(P))；
- split gain；
- merge cost；
- source mass；
- text diversity；
- topology degree/slot role；
- semantic assignment move count。

目标是回答：当前实际被拆和被合的 slot 是否真的具有较高失真和较低合并代价。若相关性接近 0，说明现有规则应从主方法中降级为经验性对照。

### P1：拆分/合并单因素消融

在 Cora/Citeseer、至少 3 个类别顺序、6 个 Student seeds 上比较：

1. C0；
2. SR-base；
3. split-only；
4. merge-only；
5. random split + matched merge；
6. BASR objective split + objective merge。

所有版本固定 rows、topology、text、Student 和 seeds。目标是区分：

- 收益来自拆；
- 收益来自合；
- 收益来自简单 payload permutation；
- 还是只来自某次 slot 排列。

### P2：matched-budget 顺序稳健性

固定每阶段 raw rows 和 memory rows，只改变类别排列。至少使用 3 个、最好 5 个独立 order，报告：

- Final Macro-F1 和 Accuracy；
- old/new F1；
- stage average；
- worst stage；
- positive-cell ratio；
- paired 95% CI；
- split/merge 触发率；
- fallback rate。

预注册晋升条件：

- 至少 2/3 大图或 3/4 小图平均为正；
- 平均 Final F1 提升至少 `+0.5 pp`；
- 最差 cell 不低于 `-1.5 pp`；
- new-class 平均不下降；
- Student 时间不超过 SR-base 的 1.2 倍；
- 不允许按 report F1 回选参数。

### P3：大图确认和强基线

若 P1/P2 通过，在 arXiv、BookChild、SportsFit 上重跑 `base+4`，并在同一主表放入：

- SR-base；
- BASR；
- C0；
- random/k-center/FPS；
- full-history synthetic；
- complete raw equal-step 和 long-step；
- 可严格适配的公开 TAG condensation baseline。

### P4：论文写作和决定

只有在 P2/P3 通过后，才把 BASR 作为最终主方法。若 BASR 不通过：

- SR-base 保留为主线；
- BASR 作为解释性失败/边界实验；
- 论文定位改为“continual TAG memory operator + protocol and mechanism analysis”；
- 不继续无止境增加模块。

## 6. 可视化建议

论文至少增加以下四类图，代替只堆表格：

1. **持续访问和递归 memory 流程图。** 展示 `raw delta_t -> source slots -> proposal -> payload assignment -> append(M_{t-1}) -> Student`，用颜色区分可访问的 current raw、可访问的 synthetic parent 和禁止回读的 historical raw。
2. **split/merge 示意图。** 在二维 joint space 中展示一个高离散度 piece 被拆成两个模式、两个低代价 piece 被合并，并标出 rows 守恒。该图必须标注为 schematic，不把示意点当真实结果。
3. **canonical vs independent order 结果图。** 横轴为数据集，分别画 SR-base 相对 C0 的 canonical 增量平均和 independent order 增量平均，并用 old/new 分组柱状图显示 trade-off。
4. **机制失败边界热图。** 行是 budget、text、response、topology、maintenance、semantic assignment，列是 Cora、Citeseer、arXiv、BookChild、SportsFit，颜色为相对 C0 的 paired delta。该图能直观说明为什么当前主方法只保留最小充分组件。

图中所有结果必须使用同协议 paired delta；不同访问权限的方法不能放在同一颜色编码中冒充同一比较层。

## 7 WWW/CCFA 可能性的客观评价

### 7.1 当前版本

当前版本具备：

- 一个明确且相对少被系统研究的 continual TAG condensation 问题；
- 完整的访问权限和协议审计；
- 小图和大图的正向机制信号；
- 递归 parent、payload conservation 和 order boundary 的可解释故事；
- 大量负结果，能够说明复杂模块为何没有进入主方法。

但当前版本仍有明显短板：

- split/merge 的目标函数没有被正式定义；
- 固定 rows 主要是公平性约束，不是算法原则；
- independent order 没有稳定收益；
- 相对强公开 TAG baseline 的严格 continual 对比仍不足；
- 当前算法创新强度中等，更多依赖新 setting、审计协议和系统实验，而不是新的基础算子。

因此，当前稿件作为 WWW/CCFA 长文候选是有理由的，但作为“稳定更强的新凝聚算法”投稿，风险偏高。更准确地说：选题和实验故事达到投稿门槛附近，算法闭环和强基线证据还没有达到稳妥接收水平。

### 7.2 改完之后的投稿门槛

如果 BASR 或等价的目标驱动 refinement 能够满足以下条件，论文竞争力会明显提高：

1. 至少两个大图、多个独立 order 上平均正收益；
2. 相对 C0 和强 coreset 在 matched-budget 下不劣，最好平均提升 `0.5-1.0 pp` 以上；
3. 最差 cell 不出现严重退化；
4. split/merge 触发由可报告的 gain/cost 决定，而不是事后手调；
5. 消融能清楚证明 split、merge、payload assignment 和 recursion 各自作用；
6. construction cost 可控，Student retraining cost 不增加；
7. 至少一个公开 TAG 方法可以在相同 continual 访问权限下复现或明确解释无法适配的原因。

满足这些条件后，投稿主张可以从“一个有正向信号的研究候选”提升为：

> 一个具有明确预算守恒优化目标、可审计递归状态和跨数据集顺序验证的持续 TAG 凝聚算法。

如果这些条件不满足，仍然可以投稿，但应采用更保守的“新问题定义 + 可靠协议 + 机制分析 + 负结果”定位，而不是包装成普遍优越的 SOTA 方法。

## 8 最终建议

1. **不要删除 C0。** 保留 C0 作为内部共同控制、回退版本和因果消融。
2. **不要继续堆 response、attention、refresh 和 GraphGPS。** 先解决 split/merge 的目标解释。
3. **保留 SR-base，新增 BASR 作为单一窄改动。** 成功则升级，失败则把失败结果写入论文边界。
4. **固定 rows 先作为公平性约束，另做 rate/预算实验。** 不要把固定 rows 直接宣称成最优 memory law。
5. **补图表和机制可视化。** 重点展示访问权限、递归 parent、payload 守恒、canonical/independent order 差异和负结果热图。
6. **论文现在可以继续写，但投稿前应完成 P0-P3。** 当前版本适合写框架和方法，最终主方法名、主表和强结论应等目标驱动 refinement 的验证后冻结。

