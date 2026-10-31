# Unified-Q-EF 优先级实验核对与补实验计划

日期：2026-10-15

本文档根据优先级清单、冻结结果和 2026-10-15 完成的控制、跨架构和统计核验整理。远程服务器上的原始结果位于：

`/home/pc/research/TAG_condensation/results/`

冻结主表没有被覆盖。

## 结论先行

当前最重要的实验缺口不是继续增加方法数量，而是把已有证据分层并完成必要的审计：

1. 已完成冻结 memory 换一个下游图编码器的迁移核验；
2. 已把“构造时使用图结构”和“最终保存文本如何材料化”分开归因；
3. 已核对 full-history 的 Student/预算来源，但确认它与当前 sealed exact-5% 主表不是同一 order/memory 调度，因此只能作为有条件的 retention reference。

组件结果已经说明：`DM` 是当前最稳定的性能贡献，`GM` 尚未显示出稳定的独立收益，共享 `Q` 目前更适合作为统一支持接口和资源设计，而不是作为已经被证明能提升精度的模块。

## 已经完成的证据

### 1. 外部基线和稳定性

`uq_baseline_expansion_20261013` 已完成五个数据集的扩展面板，核心三种方法在 ArXiv、SportsFit、BookHis 上完成 3 个类别顺序乘 3 个构造种子的九单元确认。每个 cell 内平均 Student seeds 150/151/152。

主方法相对 GECC-hard 的九单元结果：

| 数据集 | Unified-Q-EF F1 | GECC-hard F1 | 差值 | 正收益 cell | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| ArXiv | 52.91 | 52.68 | +0.22 | 4/9 | [-0.34, +0.79] |
| SportsFit | 80.08 | 80.57 | -0.49 | 2/9 | [-1.50, +0.52] |
| BookHis | 48.81 | 45.34 | +3.47 | 9/9 | [+2.20, +4.74] |

主方法相对 CaT-CCTP 的九单元 F1 差值为 ArXiv `+17.93`、SportsFit `+5.58`、BookHis `+19.66`，均为 9/9 正收益。这个结果可以支撑“在当前历史受限 TAG 协议下有效”，但不能写成跨数据集稳定超过最强 GECC 的结论。

### 2. 组件消融

`uq_component_ablations_20261014` 已覆盖三个大图、三个构造种子和统一 Prompt Student：

| 对照 | ArXiv | SportsFit | BookHis | 可支持的结论 |
|---|---:|---:|---:|---|
| Unified-Q-EF - no-GM | -0.24 pp | -0.45 pp | +0.06 pp | GM 当前没有稳定独立收益 |
| Unified-Q-EF - no-DM | +1.91 pp | +14.74 pp | +1.30 pp | DM 是最稳定的性能必要项 |
| Unified-Q-EF - fixed-Q | +12.76 pp | +21.67 pp | +8.91 pp | 学习分配比固定分配重要 |
| shared-Q - independent-Q | +0.04 pp | -0.37 pp | -1.92 pp | 共享 Q 没有稳定精度优势 |

因此正文应把 `GM` 写成兼容性目标或设计组件，不能写成已经被证明的主要增益来源；共享 `Q` 的可辩护价值应放在支持关系统一、参数/显存权衡和可审计接口上。

### 3. 文本存储控制

2026-10-15 完成 `uq_priority_controls_20261014/payload` 的 9 个 cell。控制只替换最终保存的文本，保持 `X`、标签、预算和 Student 不变：

| 数据集 | class-shared 相对主方法 | random-5 相对主方法 | 解释 |
|---|---:|---:|---|
| ArXiv | -0.01 pp | +0.05 pp | 文本替换影响很小 |
| SportsFit | -0.01 pp | +1.08 pp | random-5 有一定偶然收益 |
| BookHis | -0.31 pp | +1.07 pp | 文本选择对结果有数据集依赖 |

这组实验只能说明最终文本材料化的影响，不能单独证明构造阶段文本反馈的因果效果。SportsFit 和 BookHis 的历史 primary archive 不完整，报告中已标记 source replay 和 `primary_hash_match` 限制；同一 replay 内的固定-X 对比仍可用于存储控制，但不应称为严格 byte-identical primary reproduction。

### 4. 图结构必要性控制

2026-10-15 完成 `uq_priority_controls_20261014/no_graph` 的 9 个 cell。只在构造阶段删除当前增量边，Student 评估仍使用统一的完整原始 transductive graph：

| 数据集 | no-graph F1 | 相对主方法 | 正收益 cell | 95% paired CI |
|---|---:|---:|---:|---:|
| ArXiv | 53.51 | +0.55 pp | 2/3 | [-1.81, +2.90] |
| SportsFit | 79.27 | -0.75 pp | 0/3 | [-2.33, +0.84] |
| BookHis | 48.83 | +0.30 pp | 2/3 | [-2.55, +3.16] |

该结果说明 SportsFit 上构造阶段的图消息有较稳定的方向性作用，但在 ArXiv 和 BookHis 上尚不能形成显著的统一结论。论文可以说输入使用当前增量图结构，并报告该机制诊断；不能声称图消息在所有数据集上都是必要的。

### 5. 跨架构 Student：已完成

`uq_cross_arch_20261015_arxiv_v2` 已完成并通过本地审计：54 条记录，覆盖 3 方法 × 2 Student × 9 个 order/construction cells。每个 GraphSAGE 行读取与 GCN 完全相同的 memory、archive hash、sealed mask 和 report mask；构造器没有重新运行。

| Student | Unified-Q-EF F1 | CaT-CCTP F1 | GECC-hard F1 |
|---|---:|---:|---:|
| Prompt-GCN | 52.91 ± 0.73 | 34.98 ± 1.79 | 52.68 ± 0.45 |
| Prompt-GraphSAGE | 53.93 ± 0.60 | 50.24 ± 0.53 | 52.60 ± 0.93 |

在同一 ArXiv memory 上，Unified-Q-EF 的 GraphSAGE−GCN F1 平均为 `+1.02` pp，探索性 crossed bootstrap CI 为 `[+0.27,+1.60]`，8/9 cells 为正。相对 GECC-hard，GCN 为 `+0.22` pp、4/9 正收益；GraphSAGE 为 `+1.34` pp、9/9 正收益。相对 CaT-CCTP，GCN/GraphSAGE 分别为 `+17.93/+3.69` pp。

该结果支持“同一压缩 memory 可以迁移到另一类消息传递 Student”，但不能写成对所有 GNN 架构都无关。当前 GraphSAGE 是 PyG 两层 `SAGEConv(mean)`，显式加入 self-loop 邻域并保留 root transform；它不是 loop-free 的 canonical GraphSAGE，论文中应如实披露。

### 6. Full-history：已完成协议核对，但不是严格匹配主表

`full_raw_prompt_reference_20261011` 的 Student 配置与当前方法一致（GraphTextPromptStudent、冻结 MiniLM、2048 steps、batch 24、Student seeds 150/151/152、prompt-only 读出、edge-free reference），但它使用另一组 canonical class groups 和预算：ArXiv 为 `6773/118539 = 5.71%`，而当前 `order=9101` exact-5% 主表为 `5927/118539 = 5.00%`。full-history 还允许重读累计 raw 并重写旧支持，当前主方法则 current-delta-only、frozen-prefix。

因此之前的 retention 数值可以保留为**matched edge-free full-history reference 的历史快照**，但不能写成当前 sealed exact-5% 的严格 full-history 上界，也不能与当前主表直接形成同一个 paired test。主文应放协议限定和一张补充表；摘要只在明确限定 `historical edge-free reference` 时使用。

### 7. 构造阶段文本反馈：当前 exact-5% 最小对照已完成

`uq_exact5_no_node_text_20261015` 完成 9/9 个 cell（3 个数据集 × 3 个 construction seeds，order 9101）。该对照只在构造阶段将 `text_weight=0` 和 payload pair loss 权重置零，保留 learned Q、DM、propagation、residual、frozen prefix、硬真实文本材料化和同一 Student；因此它可以区分“文本进入 Q/构造”与“最后保存一段文本”。

| 数据集 | feature-only Q + post-hoc text 相对主方法 F1 | 95% n=3 paired CI | Accuracy delta | 结论 |
|---|---:|---:|---:|---|
| ArXiv | +0.36 pp | [-0.18,+0.89] | +0.20 pp | 未显示显著差异 |
| SportsFit | -2.38 pp | [-4.25,-0.51] | -6.25 pp | 文本进入构造有明确方向性收益 |
| BookHis | -0.66 pp | [-3.68,+2.37] | -0.86 pp | 方向偏正但不稳定 |

这组结果比 class-shared/random-5 更接近真正的构造因果对照。论文可以说：文本反馈在 SportsFit 上显著帮助了历史受限构造，但不是所有数据集都稳定必要；不能写成普遍的文本收益定律。主方法保留文本路径是有设计和局部证据支持的，但贡献表述应依赖数据集，不应夸大成所有数据集的 SOTA 来源。

## 仍需补的实验

当前不再需要为这条因果路径继续扩大主实验矩阵。若论文正文需要更强的跨顺序文本反馈结论，可把本轮 3 个 construction seed 扩展到另外两个 order；这属于加固证据，不是主方法能否成立的阻塞项。

## 不建议继续补的内容

- 不建议现在加入 EM、复杂 attention、旧 memory refresh 和可学习拓扑；它们会同时改变多个因素，无法解释当前结果。
- 不建议为了增加表格行数再加入更多弱适配 baseline。五数据集扩展表已经足够，关键是保留 official / mechanism-adapted / inspired reference 的身份标签。
- 不建议把 no-GM 的负结果隐藏。它正好支持方法简化：GM 可作为可选兼容性正则或附录组件，而主方法贡献集中在 learned shared support、DM 和文本接口。

## 建议的最终论文证据结构

1. 主表：Unified-Q-EF、CaT-CCTP、GECC-hard，三大图九单元均值、标准差、paired CI、最差 cell。
2. 扩展表：Cora、Citeseer、WikiCS 以及更广 baseline，但注明单 order 或不同预算。
3. 压缩效用：5% rate sensitivity、full-history matched reference、构造时间和显存。
4. 归因表：no-DM、fixed-Q、shared/independent-Q、no-GM、no-graph、文本材料化。
5. 跨架构表：同一 memory 的 Prompt GCN 与 Prompt GraphSAGE；当前已完成 ArXiv sealed transfer，其他大图暂不补造 archive。

当前可以继续写论文，但核心表述应是：Unified-Q-EF 提供历史访问受限 TAG 凝聚中的统一可学习支持接口，并通过分布匹配构造可读 memory；整体性能在数据集间有差异，shared-Q 和 GM 的独立精度收益尚未被证明。这样比声称所有组件都带来 SOTA 更符合现有证据。

## 结果文件

- [优先级控制汇总（远程）](/home/pc/research/TAG_condensation/results/uq_priority_controls_20261014/summary.md)
- [组件消融汇总（远程）](/home/pc/research/TAG_condensation/results/uq_component_ablations_20261014/report_utc_20261013_201811_267213/summary.md)
- [五数据集基线扩展汇总（远程）](/home/pc/research/TAG_condensation/results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/summary.md)
- [跨架构审计汇总（本地）](D:/Aprojects/continue_tag_condensation/results/uq_cross_arch_20261015_arxiv_v2/aggregate/summary.md)
- [跨架构审计汇总（远程）](/home/pc/research/TAG_condensation/results/uq_cross_arch_20261015_arxiv_v2/aggregate/summary.md)
- [构造期文本反馈对照（本地）](D:/Aprojects/continue_tag_condensation/results/uq_exact5_no_node_text_20261015/summary.md)
- [Full-history 兼容性核对](D:/Aprojects/continue_tag_condensation/docs/current/full_history_compatibility_audit_20261015_zh.md)
