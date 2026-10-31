# 外部基线协议审计与主方法建议

日期：2026-10-04

## 1. 检查范围

本次检查了：

- `D:\Aprojects\tag_baseline\EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004`；
- 当前冻结的大图公开基线结果；
- XSP、XSP + ST-Sinkhorn 及 XSP 变体结果。

外部交付包包含 13 张 baseline card、严格比较汇总、TAGSAM/STAD 补充结果、reference-only 方法说明和协议文档。最终合并结果共 74 条，已输出到：

- [主表 CSV](../../results/master_protocol_table_20261004/master_protocol_table_20261004.csv)
- [主表 Markdown](../../results/master_protocol_table_20261004/master_protocol_table_20261004.md)
- [主表元数据](../../results/master_protocol_table_20261004/master_protocol_table_20261004.json)

## 2. 协议是否一致

结论：**核心访问约束和 memory rate 基本一致，但完整实验协议并不一致，不能把所有数字直接放入同一个无标签排行榜。**

| 维度 | 外部严格基线 | 当前 XSP 大图表 | 是否一致 |
|---|---|---|---|
| 数据集 | Cora、Citeseer | Arxiv、BookChild、SportsFit | 否 |
| 阶段协议 | 3-stage，最终 stage=2 | base+4，最终 stage=4 | 否 |
| Student seeds | 150--155 | 100--105 | 否 |
| memory | 5% cumulative | 5% cumulative | 是 |
| 构造访问 | 当前 delta + synthetic parent，不读历史 raw | 当前 delta，不读历史 raw | 是 |
| Student 家族 | 两层 GCN、冻结 MiniLM、InfoNCE、每阶段 scratch | 同一 Student 家族 | 是 |
| construction seed | 270927；PUMA 为 44 | 314159 | 否 |
| report/test label 构造 | 禁止 | 禁止 | 是 |

因此当前结果应采用两个主比较面板：

1. **小图严格外部基线面板**：评价 CaT、GECC、k-center、medoid、PUMA 等外部方法之间的协议表现；当前 XSP 尚未在这套 `Cora/Citeseer + seeds 150--155` 协议下运行。
2. **大图冻结面板**：评价 XSP、XSP-ST 与 random、k-center、medoid、CCTP、GECC 在同一 `base+4 + seeds 100--105` 协议下的表现；这是当前判断 XSP 的主要证据。

TAGSAM/STAD 的结果使用 seeds 100--105，且 order 字段不完整，保留为 adapted/reference 补充结果。GraphCLIP、GCond、SFGC、GCondenser 没有共同 continual Student 结果，不能进入数值主表。

## 3. 大图冻结面板的最终结果

所有方法均使用同一大图协议。数值为 final stage 的百分数。

| 方法 | Arxiv F1 | BookChild F1 | SportsFit F1 | 三图平均 F1 | 三图平均 Accuracy | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random real-node | 17.36 | 12.69 | 26.14 | 18.73 | 21.80 | 19.92 | 11.75 |
| K-center | 19.43 | 12.49 | 29.42 | 20.45 | 24.87 | 22.04 | 11.21 |
| Medoid | 17.92 | 14.45 | **30.17** | 20.84 | 25.01 | 22.33 | **12.12** |
| CaT + CCTP top-5 | 19.36 | 13.66 | 30.15 | 21.06 | 24.64 | 22.61 | 11.92 |
| GECC-hard-TAG | 19.96 | 14.88 | 25.09 | 19.98 | 23.78 | 21.54 | 10.81 |
| **XSP** | **21.05** | **15.05** | 28.68 | **21.60** | **27.12** | **23.31** | 11.47 |
| XSP + ST-Sinkhorn | 20.81 | 12.13 | 28.98 | 20.64 | 25.86 | 22.13 | 11.92 |

### 3.1 XSP 的直接结论

- Arxiv：XSP 比本数据集最强常规基线 GECC 高 `+1.09 pp`。
- BookChild：XSP 比 GECC 高 `+0.17 pp`。
- SportsFit：XSP 比 Medoid 低 `-1.48 pp`，不是三图全胜。
- 三图平均：XSP 比 Medoid 高 `+0.76 pp`，比 CCTP 高 `+0.54 pp`。
- Accuracy 平均比 Medoid 高 `+2.11 pp`，Old F1 高 `+0.98 pp`。
- New F1 平均比 Medoid 低 `-0.65 pp`，说明收益主要来自整体和旧类保持，而不是稳定的新类塑性。

所以 XSP 的准确表述是：

> 在当前大图 base+4、固定 5% memory、不可回读历史 raw 的适配协议下，XSP 在三图平均上小幅优于常规 coreset 和 CCTP 参照，并在 Arxiv/BookChild 获得正向提升；其新类塑性和跨数据集稳定性仍有限。

不能表述为“普遍 SOTA”或“所有数据集稳定超过最强基线”。

### 3.2 XSP 与 ST-Sinkhorn

在最新冻结大图重跑中：

- XSP 平均 Final F1：`21.60`；
- XSP-ST 平均 Final F1：`20.64`；
- ST 相对 XSP：`-0.96 pp`；
- Arxiv：`-0.24 pp`；
- BookChild：`-2.92 pp`；
- SportsFit：`+0.30 pp`。

因此 **ST-Sinkhorn 目前不能晋升为主方法核心**。它可以保留为可微 assignment 的消融和候选组件，但最新冻结结果不支持把它写成稳定增益。

此前的 XSP variant sweep 中，ST 版本 pooled F1 高于当时的 XSP；但该 sweep 使用不同的冻结运行快照和 construction seed，且与本轮公开基线重跑的 memory SHA 不一致。它可以作为探索性正向信号，不能覆盖最新冻结复现的结论。

## 4. 小图外部严格表说明了什么

外部包的严格小图面板使用 `Cora/Citeseer + order19611--19615 + Student seeds 150--155`。不含 semantic-recursive 的聚合结果如下：

| 方法 | 有效 cell | Final F1 | Accuracy | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|
| CaT-MMD-TAG | 10 | **55.35** | **60.94** | **53.23** | **58.29** |
| k-center/FPS | 10 | 51.26 | 56.61 | 49.33 | 53.98 |
| medoid | 7 | 49.67 | 55.36 | 47.87 | 51.96 |
| GECC-hard-TAG | 7 | 48.51 | 52.68 | 45.41 | 52.84 |
| PUMA-inspired | 4 | 14.09 | 15.63 | 15.14 | 12.38 |

这张表不能直接证明 XSP 在小图上优于 CaT，因为 XSP 尚未在同一小图严格协议中运行。它只说明：外部基线之间的差距并不小，CaT 是一个必须认真对待的强参照，不能只用大图结果替代小图严格比较。

## 5. class-shared prompt-margin 为什么不作为主方法

SportsFit 上 `class-shared prompt-margin` 达到 `57.01`，但该版本的审计字段明确显示：

- 每个类别只生成一个 materialized payload；
- 同类所有 synthetic rows 广播相同 payload；
- 选择过程显式使用类别 prompt margin；
- Arxiv 为 `27.33`，BookChild 仅为 `3.45`。

这更像是一个类别级文本原型控制，而不是 XSP 的多 slot 信息保留机制。它的高分具有明显数据集依赖性，不能作为公平的外部强基线，也不能用来证明 XSP 的普遍性能。

## 6. 最终主方法建议

### 建议冻结

当前最稳妥的主方法是：

> **CaT initialization + X-conditioned Slot Partition (XSP)**

具体保留：

1. CaT 生成合成特征 `X`；
2. 当前 delta 内按类别处理；
3. 依据每个合成 `X` 槽位建立有限 payload assignment；
4. 每个 slot 材料化真实文本来源，不把一个类别压成单一文本；
5. 只追加新 memory，不修改旧 memory；
6. 使用固定 memory、当前 delta-only 和 no-historical-raw 访问约束。

### 不建议现在晋升

- XSP + ST-Sinkhorn：最新冻结大图结果不稳定；
- class-shared prompt-margin：类别共享文本，跨数据集异常；
- coherence anchor、hard consistency、full 组合：成本高，当前只能作为诊断变体；
- GraphGPS/GraphCLIP Student、动态 selector、response gate：会改变比较对象或扩大成本，暂不放进主线。

## 7. 论文前仍需补的关键实验

主方法结构已经可以冻结，但论文结论还需要补两项：

1. **严格小图桥接**：在 Cora/Citeseer、`order19611/19612`、Student seeds 150--155 下运行 XSP 和 XSP-ST，与外部包中的 CaT、k-center、medoid、GECC 做同协议比较。这样才能回答 XSP 是否不仅在大图有效。
2. **真正独立 construction seed**：固定代码快照、数据 hash、CUDA/线程配置和 payload seed，重新确认 XSP 的 memory SHA 与 F1。已有两个相同 construction seed 的运行产生不同 memory SHA，不能直接合并为独立 seed 证据。

此外建议报告：new-class F1、old-class F1、slot coverage、unique text ratio、构造时间和 Student 时间。XSP 的主要短板是 New F1 没有稳定提升，必须在论文中如实说明。

## 8. 客观评价

当前结果支持 XSP 成为**论文主方法候选**，理由是：

- 它在合法持续访问约束下有清楚的多 slot 文本分配机制；
- 在同一大图协议中平均超过 random、k-center、medoid、CCTP 和 GECC；
- Arxiv 和 BookChild 上有正向结果；
- 不依赖历史 raw graph，且文本来源可审计。

但当前还不足以宣称强 SOTA，主要限制是：

- SportsFit 上落后强 coreset；
- New F1 平均没有提升；
- 大图和外部严格小图协议尚未完全桥接；
- XSP 构造成本高于 k-center/medoid；
- XSP-ST 的增益没有在最新冻结重跑中复现。

最终建议是：**以 XSP 作为主方法，XSP-ST 作为消融，不再继续扩展结构；先完成严格小图桥接和独立 seed 确认，再冻结论文主表。**

