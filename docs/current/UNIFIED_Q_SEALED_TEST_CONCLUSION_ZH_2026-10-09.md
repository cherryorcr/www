# Unified-Q sealed test 与外部 baseline 核对结论

## 完成内容

本轮没有重新构造 memory。评估器直接读取已经冻结的 27 个构造 cell，每个 cell 的四个 memory arm 都通过 archive hash 校验，再在 `original_id % 10 == 2` 的 sealed test 上重新训练和评估 Prompt-GCN、Prompt-SGC、Prompt-GraphSAGE。

- 3 个数据集：WikiCS、SportsFit、BookHis；
- 27 个类别顺序 × construction seed cell；
- 4 个 memory arm：Q-medoid、In-loop CaT-Q、CaT + nearest5、GECC-hard + TAG；
- 3 个 Student，每个 Student 3 个 seed；
- 共 324 个 sealed readout；
- 构造阶段没有读取 sealed test，也没有用 test 选择方法或参数。

## Sealed test 的主要结果

| 数据集 | Student | In-loop CaT-Q Macro-F1 | CaT Macro-F1 | GECC Macro-F1 | 主方法相对 CaT | 主方法相对 GECC |
|---|---|---:|---:|---:|---:|---:|
| WikiCS | Prompt-GCN | 78.52 | 78.60 | 77.91 | -0.08 | +0.61 |
| WikiCS | Prompt-SGC | 76.52 | 71.12 | 76.99 | +5.40 | -0.47 |
| WikiCS | Prompt-GraphSAGE | 78.03 | 79.67 | 76.61 | -1.64 | +1.42 |
| SportsFit | Prompt-GCN | 77.94 | 76.07 | 77.70 | +1.86 | +0.24 |
| SportsFit | Prompt-SGC | 73.34 | 56.36 | 75.01 | +16.97 | -1.67 |
| SportsFit | Prompt-GraphSAGE | 75.32 | 77.55 | 75.00 | -2.24 | +0.32 |
| BookHis | Prompt-GCN | 39.91 | 17.68 | 38.31 | +22.23 | +1.60 |
| BookHis | Prompt-SGC | 34.74 | 13.82 | 35.74 | +20.92 | -1.01 |
| BookHis | Prompt-GraphSAGE | 38.81 | 29.78 | 36.96 | +9.03 | +1.85 |

这些均为 9 个 construction cells 的均值；完整标准差、Accuracy、Old/New F1 和逐 cell 配对差值见 [sealed 汇总表](../../results/uq_text_path_sealed_test_20261009_v3/summary.md)。

## 应如何解释

第一，sealed test 没有推翻主方法。In-loop CaT-Q 在 BookHis 上相对 CaT 的优势很稳定，9/9 个 cell 在三个 Student 上都为正；相对 GECC 的优势较小，GCN 和 GraphSAGE 为正，SGC 略低。SportsFit 和 WikiCS 的结果说明收益依赖图分布和 Student 架构：主方法通常与强基线相近，但不能声称跨数据集、跨架构普遍领先。

第二，Q-medoid 与主方法非常接近。它是同一 Unified-Q 构造中只改变硬文本材料化的内部控制，因此当前结果不能把主要收益单独归因于硬文本选择。更稳妥的表述是：Unified-Q 的联合 Q/DM/GM 构造在特定数据集和架构上有效，而文本材料化策略仍未被证明是唯一关键因素。

第三，CaT + nearest5 在 BookHis 上严重退化，而在 WikiCS/SportsFit 的部分 Student 上表现正常。这说明 CaT 适配的文本附着和数据分布存在明显交互；BookHis 的提升不能简单写成“所有数据集都超过 CaT”。同时，CaT、GECC 在本项目中都是机制适配，不是官方 native reproduction。

## 外部 baseline 的证据等级

### 可进入严格适配主表

- CaT-MMD-TAG：保留 CaT 的 MMD 特征凝聚机制，改为 current-delta-only、冻结 synthetic prefix、固定 memory budget，并附加 TAG 文本；
- GECC-hard-TAG：保留 GECC hard hop-weight/Lloyd 机制，采用相同 continual 输入、预算、Student 和评估协议；
- PUMA-inspired TAG adaptation：保留 pseudo-label、置信度筛选、一次传播和 edge-free memory，但数据、Student 和预算均做了明确适配；
- k-center/FPS、medoid：作为合法 current-pool coreset 参照。

### 只能作为补充或 reference-only

- TAGSAM、STAD：已有 Student seed/order 或访问协议差异，放 adapted/reference 表；
- GCond、SFGC、GCondenser：当前只有官方代码/机制核验，没有同一 TAG continual Student 结果；
- GraphCLIP 官方 checkpoint：完整原图上的模型参考，不是 continual condensation constructor。

完整逐卡审计见 [外部 baseline fidelity audit](EXTERNAL_BASELINE_FIDELITY_AUDIT_ZH_2026-10-09.md)。本轮没有重新在线拉取 GitHub commit，固定 commit 和卡片信息沿用既有 handoff package，因此不新增官方复现声明。

## 对论文结论的影响

当前证据可以支撑：

- 历史 raw 不可回读的 TAG continual condensation setting；
- 一个将 class-constrained Q、分布/梯度匹配、受限特征残差和真实文本 routing 放进同一构造流程的候选方法；
- 在 BookHis 以及部分 WikiCS/SportsFit Student 条件下的稳定收益；
- 跨 Student、跨数据集的完整 sealed 泛化分析，以及 CaT/GECC 适配基线的协议边界。

当前证据不能支撑：

- 普遍 SOTA；
- 对所有 Student 或所有数据集稳定超过 GECC；
- 官方 CaT/GECC/PUMA native reproduction；
- 文本硬材料化或共享 Q 单独带来全部收益。

因此后续不建议根据 sealed test 再调参。论文可以开始写方法、协议、正负结果和限制；若要继续提高贡献强度，应另开预注册的单一机制改动实验，并在新的 sealed split 上重新确认，不能回到当前 sealed test 上迭代优化。
