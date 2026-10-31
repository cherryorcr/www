# 当前持续 TAG 凝聚方法综合评估与后续推进方向

日期：2026-09-19  
适用范围：冻结 v4、P0/P1-X/P1-B/P2-A/P2-B/P3-A/P3-B/C1 小图验收，以及已有 v3 多数据集结果  
文档性质：综合判断与 P4/P5 执行指导，不覆盖历史实验结果

## 0.1 C1 结果纳入后的综合判断

C1 类内多原型与复杂度分配实验已完成。它在 Cora 三顺序平均上相对 C0 提升 `+1.14` final F1、`+3.89` 阶段平均 F1，Moment-X 子候选达到 `+1.96/+6.26`；但 Citeseer 三个候选 final F1 均低于 C0 `0.65--1.07`，阶段平均低 `5.19--7.04`。顺序级结果也显示 C1 的收益主要集中在 Cora `order1702`，而在 Cora `order1701` 和 Citeseer canonical 上退化。

因此当前综合评价不变：C1 是值得保留的机制分析和负结果，不是默认主方法。由于候选同时改变了子簇预算和 X 算子，后续若拆解必须使用独立顺序做“固定分配换 X”和“固定 X 换分配”两项小实验；在此之前，C0 继续作为 P5 的唯一共同控制。

## 1. 结论先行

当前最稳妥、可以作为论文主线和后续实验共同控制的版本是：

```text
5% fixed-rate growing memory
+ dispersion allocation
+ 类内 graph-text joint clustering
+ Mean-X synthetic feature
+ joint_g5_c320 center-concat text
+ frozen old rows / old-old edges
+ current text-nearest cross-edge, cross-degree=1
+ response selector off
```

这里的 `current default` 不是小图上分数最高的所有配置，而是同时满足以下条件的最低风险版本：

- 两个小图上可以复现；
- 构造成本低；
- 历史真实数据不可回读和旧 synthetic 记录可审计；
- 没有把一个不稳定的短训 proxy 写入主算法；
- 每个组件的作用可以单独归因。

需要同时记录一个容易混淆的事实：`response_guarded` 在三顺序平均上高于 dispersion，Cora/Citeseer 最终 F1 分别为 `50.23/56.24`，而 dispersion 为 `48.40/54.63`。但它的构造成本约高 `75--130x`，接受率只有 `4/9` 和 `2/9`，且 matched random transfer 在部分顺序更高。因此它是“有潜力的固定预算候选”，不是当前主方法。

当前研究结论不是“复杂方法全部无效”，而是：

> 已测试的第一代局部算子没有稳定改善“从头训练 student 后的持续任务效用”；唯一较清晰的正机制是新旧跨边整体有用。下一代方法必须联合建模 synthetic X、文本和局部拓扑，而不是继续分别微调均值、文本选择或短训 selector。

## 2. 当前方法的完整定义

### 2.1 每阶段凝聚流程

在阶段 `t`，输入只有合法的当前增量 `Delta_t`、上一阶段 synthetic memory `M_(t-1)` 以及已经保存的统计量。历史原始节点、历史原始文本和历史原始邻接不能重新读取。

1. **确定当前预算。** 使用随累计合法数据规模增长的 `5% fixed-rate` 预算，新增 rows 为当前累计预算减去旧 memory rows。默认不裁剪旧 rows。
2. **按类别分配预算。** 使用类别规模和类内离散度得到 `dispersion allocation`。它是低成本先验，不是通过 test F1 学到的分配器。
3. **类内联合聚类。** 在当前 delta 的 graph-text joint space 中形成若干 cluster，保证属性、文本和类别来源可追溯。
4. **生成合成节点。** 每个 cluster 的属性使用 Mean-X，即成员 `X` 的均值。合成节点不是直接复制真实节点。
5. **构造文本。** 每个合成 cluster 选最多 5 篇代表文本，在约 320 字符预算内拼接，再使用冻结 MiniLM 重新编码，得到 384 维文本表示。
6. **追加拓扑。** 旧 rows 和旧旧边保持不写入；新块内部建立受控边，并按当前文本最近邻规则连接少量旧节点，默认 cross-degree 为 1。
7. **形成新的 synthetic memory。** `M_t` 是旧 synthetic memory 与当前新合成块的 append 结果。当前 response selector 不参与默认主流程。
8. **重训 student。** 每个阶段从头训练两层 GCN，hidden/output 为 384，冻结 MiniLM 表示，使用双向 graph-text InfoNCE，batch 为 16，student seeds 为 `100..105`，步数遵循 v3/v4 的统一规则。
9. **零样本评估。** student 用真实评估图结构和类别 prompt 做匹配，报告 cumulative/stage Macro-F1、old/new F1、forgetting 和成本；student 不使用节点分类标签训练。

### 2.2 当前的优化目标与实际目标并不相同

真正想优化的是合成 memory 经过 student 重训后的效用：

\[
U(M)=\mathbb{E}_{s}[F1(\operatorname{Train}(S,M;s))]
-\lambda C(M)-\mu R_{old}(M),
\]

其中 `C` 表示构造和训练成本，`R_old` 表示旧类保持风险。当前许多算子优化的却是局部 proxy：

| 组件 | 实际优化的 proxy | 与最终 student 效用的距离 |
|---|---|---|
| Mean-X | cluster 内平方重构误差 | 不保证多模态和决策边界保持 |
| Weighted/Moment-X | 加权中心、均值/方差匹配 | 仍只改变一个属性向量，未恢复局部图结构 |
| 文本 attention/selection | 文本相似度、片段互补性或摘要质量 | 不保证双向 InfoNCE 的长期可重训性 |
| 跨边 endpoint | 文本/X 局部相似度 | 会改变 GCN 度归一化和旧节点传播 |
| response selector | 短 horizon student 曲线 | 短训排序与 canonical 排名不稳定 |
| 动态预算 | 规模、离散度或局部 marginal gain | 不能自动反映一行对 old/new F1 的真实贡献 |

所以目前出现“算子有合理直觉，但完整 student 没提升”的结果是可以解释的，并不矛盾。

## 3. 已完成组件的证据等级

### 3.1 预算分配

`dispersion` 是当前最可靠的低成本分配控制。它不能被称为全局最优，但比固定总预算和未经控制的历史方案更适合作为持续协议基线。

固定预算 guarded response transfer 的三顺序平均结果如下：

| 数据集 | dispersion | response_guarded | 平均增益 | 接受率 | 构造成本结论 |
|---|---:|---:|---:|---:|---|
| Cora | 48.40 | 50.23 | +1.82 | 4/9 | 约 130x 构造开销 |
| Citeseer | 54.63 | 56.24 | +1.62 | 2/9 | 约 75x 构造开销 |

它说明 response 可能找到更好的类间配额，但不能证明 response 本身造成了全部收益。Citeseer order1702 中 matched random transfer 高于 response，且大量阶段回退到 dispersion。因此：

- `dispersion` 作为默认；
- `response_guarded` 作为候选消融；
- `adaptive-rate` 和 response-gain extra budget 暂停；
- P4/P5 先不把不稳定 response 写入总预算控制。

### 3.2 合成 X

三顺序平均最终 F1：

| 数据集 | Mean-X | Weighted-X | Moment-X | k-center | kmeans-medoid |
|---|---:|---:|---:|---:|---:|
| Cora | 48.40 | 48.42 | 47.13 | 49.40 | 54.24 |
| Citeseer | 54.63 | 54.33 | 53.76 | 57.73 | 51.08 |

结论：

- Weighted-X 与 Mean-X 几乎相同，不能称为改进；
- Moment-X 在两个数据集下降；
- 换一个均值附近的手工算子没有缩小与 k-center/medoid 的稳定差距；
- 强 coreset 的优势可能来自真实节点属性、真实文本对应关系和非线性局部结构，而不仅是均值的方差收缩。

因此 P1-X 第一代应收尾，不再扫描权重温度或 Moment 系数。下一代应转向“多原型 + 局部拓扑”的联合表示。

### 3.3 文本处理

当前稳定控制仍是 `joint_g5_c320 center concat`。P2-A 的图条件选片结果：

| 对照 | Cora final | Citeseer final | 判定 |
|---|---:|---:|---|
| center concat | 48.46 | 54.97 | 默认控制 |
| graph selection | 51.51 | 53.12 | 跨数据集不稳定 |
| shuffled graph | 52.32 | 53.94 | 真实 graph query 未胜出 |
| text attention | 51.25 | 54.08 | 不能稳定晋升 |

图条件选片相对 center 在 Cora `+3.05`，在 Citeseer `-1.85`；相对 shuffled graph 反而在 Cora/Citeseer 分别 `-0.81/-0.82`。P2-B 中 student-guided attention 产生了非零梯度，但完整 student 没有稳定收益，并额外增加约 18--19 秒三阶段构造成本。

因此文本方向当前最可靠的结论是“冻结 MiniLM 后重新编码有价值”，而不是“拼接已经被证明最优”。后续 attention 只能作为第二代候选，并必须与同 token concat、text-only、attribute-only、shuffled graph 和成本账一起验证。

### 3.4 拓扑结构

这是目前唯一较清晰的正机制证据。只改变跨边，三顺序平均结果为：

| 数据集 | none | current | random | quality |
|---|---:|---:|---:|---:|
| Cora | 45.43 | 48.40 | 50.16 | 48.05 |
| Citeseer | 52.60 | 54.62 | 54.69 | 55.75 |

相对无跨边，current 在 Cora/Citeseer 分别提升 `+2.97/+2.02`。但 random 和 quality 在不同顺序上翻转，不能替换 current。

机制上，跨边同时带来两件事：

1. 给新块提供历史语义或结构信息，改善新旧交互；
2. 改变 `D^(-1/2)(A+I)D^(-1/2)`，即便旧旧边没有改写，也会改变旧节点归一化传播。

所以“frozen old edge”不等于“old propagation frozen”。P4/P5 必须把旧传播漂移、跨边消息量和 old/new F1 一起记录。

### 3.5 Student response

旧 response-on 的三顺序平均变化为 Cora `-0.93`、Citeseer `+0.85`，selector 成本约为低成本构造的 `400--600x`。guarded 版本收缩了搜索范围后平均变好，但成本和因果归因仍未通过。

当前应把 response 定义为：

- 离线诊断工具；
- 少量候选的 confirmation；
- 未来 P5 风险感知选择的研究变量。

不能把它当成默认在线预算管理器，也不能用 response 结果事后改写 frozen v4。

## 4. 为什么小图可能没有显示复杂方法优势

小图确实可能压低复杂机制的收益，但这只能作为待验证假设，不能作为失败后的解释性借口。限制包括：

- 类别数和增量阶段数较少；
- 每类样本及类内多模态有限；
- 5% memory 下每类可用 rows 很少；
- 长期漂移和历史冗余不足；
- response 候选空间太小，额外成本却已经很高；
- Cora/Citeseer 的局部拓扑规模不足以体现复杂跨块管理。

小图已经能够可靠否定以下强表述：

- Weighted/Moment-X 普遍优于 Mean-X；
- graph-conditioned text selection 普遍优于 center 或 shuffled control；
- 短训 response ranking 能稳定预测 canonical ranking；
- response-on 具有一次使用成本优势。

因此正确做法是：先冻结当前默认方法，再用 P4/P5 验证“复杂度是否随图规模和持续长度增加而有价值”，而不是直接在小图上堆叠全部模块。

## 5. 哪些组合值得继续做

不建议把所有失败组件直接组合。建议按机制互补性安排四个组合，其中只有第一个可立即进入下一轮：

| 组合 | 内容 | 当前地位 | 要验证的问题 |
|---|---|---|---|
| C0 | dispersion + Mean-X + center concat + current cross-edge | 当前主控制 | 作为所有 P4/P5 的共同基线 |
| C1 | 类内复杂度分配 + 多原型 synthetic X + center concat + current cross-edge | 最有希望的算法候选 | 多模态和局部结构是否比单一均值更重要 |
| C2 | C0 + bounded refresh 或 elastic reservoir | P4 候选 | frozen-old 是否在长链上过时 |
| C3 | C0 + 有界跨边/单向跨块传播 | P3-B 后置候选 | 跨边收益是否来自有益消息，还是来自旧传播扰动 |

`response_guarded` 不作为 C1/C2/C3 的必选组件；它只做小规模离线 confirmation。文本 attention 也不与 C1 同时首测，否则即使有增益也无法归因。

### 5.1 C1 的理论设计

Mean-X 只给一个 cluster 一个向量，无法同时表示多个峰值、尾部方向和不同局部结构。下一代构造应把一个 cluster 拆成 `k_j` 个 synthetic prototypes：

\[
G_j \rightarrow \{(x_{j,r}, t_{j,r}, e_{j,r})\}_{r=1}^{k_j},
\qquad \sum_j k_j=K_t.
\]

每个 prototype 是合成节点，不是直接使用真实节点。它应同时保存：

- 属性原型 `x`；
- 文本原型 `t`；
- 局部结构角色或邻域统计 `e`；
- 来源 cluster、质量和覆盖统计。

初版不引入可训练 attention，先使用确定性类内分簇、medoid 初始化的 synthetic prototype 和固定拓扑模板。这样可以判断收益是否来自“多原型覆盖”，而不是来自新模型容量。

### 5.2 C2 的理论设计

把旧 memory 拆成：

- `protected core`：历史覆盖和旧类关键原型，不改写；
- `elastic reservoir`：冗余、低稳定性或长期未被支持的旧 rows，允许小比例替换。

每阶段最多刷新旧 rows 的 `10%`，后续再测试 `20%`。刷新只能使用旧 synthetic 统计和当前合法 delta，不能回读历史 raw。若刷新不满足条件，自动回退 append-only。

### 5.3 C3 的理论设计

先固定跨边端点，只改传播算子：

- current 双向跨边；
- old-to-new 单向传播；
- new-to-old 单向传播作为风险对照；
- 有界跨边权重 `0.25/0.5/1.0`；
- block-normalized adjacency 作为单独 operator 对照。

每个 operator 都要同时报告旧块归一化漂移、跨边消息量、旧表征漂移、old/new F1 和成本。不能只看最终 Macro-F1。

## 6. P4：历史 synthetic memory 管理

P4 不是简单地把 frozen-old 改成“全部可更新”。推荐四步：

### P4-A：re-score-only 控制

只更新每个 synthetic row 的质量、冗余、支持阶段和覆盖统计，不改 `X/text/edges`。目标是确认统计管理本身不会造成性能变化，并为后续刷新提供合法输入。

### P4-B：bounded refresh

在固定总预算下，只允许 elastic reservoir 中少量 rows 被当前 delta 的证据替换或修正。每次刷新记录 `birth_stage`、`last_supported_stage`、原因、替换比例和边变化。不得刷新全部旧类，也不能使用历史 raw。

### P4-C：局部 merge/split

只有当 C1 多原型表示通过后才做。merge 只合并高度冗余的同类 synthetic prototypes；split 只针对高 dispersion、低覆盖的 prototype。X、文本和边必须作为一个局部对象共同操作。

### P4-D：importance eviction

最后才做重要性驱逐。优先驱逐冗余、低覆盖、长期未支持且不会破坏少数类覆盖的 rows。它必须有 append-only fallback，不能为了保持固定预算而强行删除历史核心。

### P4 验收门槛

至少在两个数据集上满足：

- cumulative F1 或 old retention 有稳定提升；
- refresh ratio 不超过预设上限；
- edge churn 和 old propagation drift 可控；
- 不明显损伤 new F1；
- memory rows、bytes、构造时间和 student 时间单独报告。

若 P4 失败，冻结旧 memory 不是“错误实现”，而应作为稳定性/可审计性更强的负结果。

## 7. P5：长链、预算率与外推验证

P5 的目标不是为了再次寻找一个小图最高分，而是判断当前方法是否具有跨规模和跨持续长度的有效性。

### P5-A：memory rate curve

在不改变其它设置的前提下比较 `1%/5%/10%`，报告：

- final/stage cumulative Macro-F1；
- old/new F1 和 forgetting；
- rows、bytes、边数；
- 构造时间、student 时间和峰值显存；
- F1 vs rows、F1 vs compression rate、memory size vs stage。

### P5-B：更长持续协议

- Cora/Citeseer：保留可比较的三阶段，并补独立类别顺序；不要强行制造空阶段；
- arXiv、BookChild、SportsFit：主表使用 `base + 4 incremental`，3-stage 做快速消融，5-stage 或更长链作为稳健性实验；
- 同一数据集固定类别块、到达样本量、预算增长规则、student steps 和 seed policy。

### P5-C：五层方法对照

在相同 student、预算和指标下分开报告：

1. 当前持续 append；
2. 全历史 raw 重凝聚；
3. 旧 synthetic 与新 delta 联合重凝聚；
4. 强 coreset/真实节点参照，如 k-center、kmeans-medoid；
5. 完整 raw pool 训练参考及监督模型上限。

已有 TAGSAM/STAD 结果继续作为资格审计和外部参考，按照此前决定不再为了拼主表而补强不匹配的下游模型或协议。重点是区分持续凝聚约束的效果，而不是把不同 student 目标强行排序。

### P5-D：独立确认规则

P5 运行前冻结顺序、selection seed、预算和主指标。开发阶段用过的 Cora order1702/Citeseer order1702 只能作为已知失败诊断条件；新主结论必须使用未参与调参的顺序或 seed group。所有候选先在小图通过，再决定是否进入大图。

### P5 验收门槛

P5 不要求当前方法在所有数据集都超过 k-center。它要回答：

- 简单 Mean-X/dispersion 是否在长链仍可复现；
- 跨边的正机制是否随图规模增强；
- C1 多原型或 C2 bounded refresh 是否只在更长链上显现；
- 性能差距是否能用更低 raw 访问和更小 memory 解释；
- 结论是否不依赖单一顺序、单一 rate 或单一 student seed。

## 8. 推荐执行顺序

1. 冻结 C0，建立 P4/P5 的统一 manifest 和成本字段。
2. 先做 P4-A re-score-only，确认历史管理的访问边界和审计链。
3. 做 P4-B bounded refresh `10%`，只测 C0 + refresh，暂不叠加文本 attention/response。
4. 若 P4-B 有信号，再做 C1 多原型；若没有，优先完成 P5 外推，不继续堆 manager。
5. 做 P3-B-2 的有界跨边/单向传播单阶段诊断，再决定是否进入完整链。
6. 完成 P5 rate curve、base+4、多顺序和五层基线对照。
7. 只有通过独立确认的单一组件，才建立 `v5`；否则保留为 v4 的消融和负结果。

## 9. 最终论文定位

当前最诚实、证据最充分的定位是：

> 在历史原始数据不可回读的持续 TAG 凝聚场景中，本文提出并审计了一套合成图文追加协议，系统分析预算、合成属性、文本压缩、跨阶段拓扑和 student-response 选择的作用边界。实验表明，新旧跨边提供了稳定的机制性帮助，但第一代文本、均值替代和短训 response 算子没有跨数据集稳定晋升；因此当前方法的价值主要在于可审计的持续构造、保持-塑性分析和负结果边界，而不是暂时宣称 SOTA。

如果 P4 或 P5 证明多原型、bounded refresh 或受控传播算子在独立长链上稳定改善，主线可以升级为：

> topology-aware multi-prototype synthetic memory with bounded continual updates.

在此之前，不应把所有候选组合成一个事后命名的新版本。

## 10. 关联报告

- [持续 TAG 凝聚后续方法推进指导](NEXT_METHOD_EXECUTION_GUIDANCE_ZH_2026-09-18.md)
- [P1-X 合成节点特征算子实验报告](P1X_SYNTHETIC_X_COMPLETION_ZH_2026-09-18.md)
- [P1-B 固定预算 Guarded Response 验收报告](P1B_GUARDED_RESPONSE_COMPLETION_ZH_2026-09-18.md)
- [P2-A 图条件文本处理完成报告](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md)
- [P2-B Student 引导 Attention 小图实验报告](P2B_STUDENT_GUIDED_ATTENTION_COMPLETION_ZH_2026-09-18.md)
- [P3-A response 选择可靠性与成本验收报告](P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md)
- [P3-B 拓扑诊断完成报告](P3B_TOPOLOGY_DIAGNOSTIC_COMPLETION_ZH_2026-09-18.md)
- [v4 版本、P0-P5 状态与后续优化路线](V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md)

## 11. P4/P5 总推进计划

### 11.1 当前状态

- P4 最低闭环已完成：rescore-only 作为不改写内容的空操作控制通过 hash 审计，bounded-refresh 10% 在 Cora/Citeseer 未形成稳定收益，已按 frozen-old 更稳健的负结果收尾；不追加 20% refresh、merge/split 或 eviction。
- C3 单阶段传播诊断已完成：old-to-new 单向传播仅在 Cora 提升、Citeseer 退化；new-to-old 单向传播两数据集均退化，当前双向 cross-edge 继续保留。
- P5 已完成 C0 的小图 1%/5%/10% rate curve、5-stage/base+4 长链、一组独立 frequency 顺序，以及 arXiv/BookChild/SportsFit 的大图 base+4 首轮；结果显示 5% 是当前较稳工作点，但 rate 非单调，长链最后阶段出现新类塑性回落，canonical 与独立顺序的最终 F1 差异为 Cora +1.76、Citeseer -6.38 个百分点，大图最终累计 F1 为 arXiv 18.82、BookChild 13.22、SportsFit 27.90。更多独立顺序和统一五层对照仍未完成。
- C0 继续作为所有后续实验的共同控制，不因某个候选的局部高分而改写。

### 11.2 分批执行

| 批次 | 任务 | 是否依赖前一批 | 通过后进入 |
|---|---|---|---|
| Batch 0 | 冻结 C0 manifest、rows/edge/text/student hash 和成本字段 | 无 | 所有实验 |
| Batch 1 | P4-A re-score-only；P4-B bounded refresh 10% | Batch 0 | P4-C 或负结果收尾 |
| Batch 2 | C3 有界跨边、old-to-new/new-to-old、block normalization 单阶段诊断 | Batch 0，可与 Batch 1 并行 | 通过的传播算子进入完整链 |
| Batch 3 | C1 类内复杂度分配与多原型 synthetic X | Batch 0；不与 Batch 1 混合 | 独立确认或停止 |
| Batch 4 | P5 小图 rate curve、独立顺序和更长可行阶段链 | C0 已冻结；候选可稍后补测 | 大图候选筛选 |
| Batch 5 | P5 大图 base+4、3-stage 快速消融、5-stage 稳健性 | Batch 4 的协议通过 | 五层主表 |
| Batch 6 | 持续 append、全历史重凝聚、联合重凝聚、强 coreset、完整 raw 参考和成本总账 | Batch 5 的 protocol freeze | 论文主表与最终版本决定 |

### 11.3 P4 的完成判定

P4 的最低闭环是 `re-score-only -> bounded-refresh-10%`。如果 bounded refresh 在两个数据集上没有提高 cumulative F1 或 old retention，或者引起明显的 new F1、传播漂移、edge churn 或成本退化，则 P4 以“frozen-old 更稳健”的负结果收尾，不继续无条件做 merge/split/eviction。

只有在 10% refresh 出现稳定信号后，才做 20% refresh；只有在 C1 多原型已成立后，才做局部 merge/split。importance eviction 放在最后，并始终保留 append-only fallback。

### 11.4 P5 的完成判定

P5 不要求当前方法在每个数据集上超过 k-center。P5 完成需要回答：

1. C0 在 `1%/5%/10%` memory rate 下是否保持可解释的性能-成本曲线；
2. C0 和通过小图门禁的候选能否在 arXiv、BookChild、SportsFit 的 base+4 协议中复现；
3. 结果是否不依赖单一类别顺序、单一 selection seed 或单一 student seed；
4. 持续追加相对全历史重凝聚和强 coreset 的差距是否能被访问约束、memory bytes 和训练成本解释；
5. 是否有一个单独组件通过独立确认，足以建立 v5，而不是把多个未验证组件事后打包。

### 11.5 可以并行与不能并行的内容

P5 的 C0 基线 rate curve 可以在 P4 运行期间并行，以节省时间；但 P5 的候选主表必须等 P4/C1/C3 的小图门禁结束后再锁定。P4 refresh、C1 多原型和 C3 传播算子不能第一次就组合，否则无法区分历史管理、节点内容和消息传递的收益。

因此最终顺序为：

```text
C0 protocol freeze
-> P4 minimal memory test
-> C3 single-stage topology test
-> C1 multi-prototype test
-> P5 small rate/order validation
-> P5 large base+4 and five-level comparison
-> final v5 decision or v4 negative-result closure
```
