# 严格统一协议下的大图基线对比报告

## 1. 本轮目的

本轮重新运行三张大图上的统一基线矩阵，避免把旧版协议、不同 Student、不同 memory 口径或不同文本缓存混在同一张表中。比较对象是当前候选方法：

> CaT 的 class-wise MMD 特征凝聚 + CCTP top-5 X 初始化 + class-shared real payload 文本材料化。

本轮不把早期的 Semantic-Recursive 作为对照。它偏启发式，且不属于本轮以 CaT/Student 和严格构造边界为核心的比较线。

## 2. 统一协议

| 项目 | 固定设置 |
|---|---|
| 数据集 | arXiv、BookChild、SportsFit |
| 增量协议 | base+4，共 5 个阶段 |
| memory | 累计节点数的 5%，按冻结 manifest 分配 |
| 到达数据 | 每阶段只读取当前 delta；历史 raw graph 不回读 |
| Student | 同一个 edge-aware 两层 GCN + 文本投影 + 双向图文 InfoNCE |
| 文本编码器 | 冻结 MiniLM |
| Student seeds | 100--105，共 6 个配对 seed |
| 评价 | 每个阶段对所有已见 report-dev 节点累计计算；不重复累计节点 |
| 构造禁用信息 | report/test 标签不参与构造或选择 |
| construction seed | 270927；class-shared payload 使用 seed+76 |

所有分数为百分比。`Final F1` 是最后阶段累计 Macro-F1，`Stage avg F1` 是最后阶段各新到达类别块的 F1 平均；二者不是同一个统计量。

## 3. 方法分类

| 方法 | 本轮身份 | 具体含义 |
|---|---|---|
| Append-edge | 内部 sealed control | 读取已封存的 append-edge memory，用于检查当前实验链路 |
| X+P | 内部 sealed control | 读取已封存的 X+P memory，用于内部比较；不是公开基线 |
| Random real-node | 适配的 real-node coreset | 从当前 delta 选真实节点，并用统一 Student 拓扑规则材料化 |
| K-center | 适配的 real-node coreset | 在当前 delta 上做 k-center 选择，再用统一拓扑规则材料化 |
| Medoid | 适配的 real-node coreset | 在当前 delta 上做 medoid 选择，再用统一拓扑规则材料化 |
| CaT-TAG CCTP top-5 | TAG 机制适配基线 | CaT class-wise MMD 生成 X，使用 CCTP top-5 文本；本轮不是官方端到端复现 |
| GECC-TAG hard | TAG 机制适配基线 | GECC hard 机制适配到相同 memory/Student 协议；不是官方端到端复现 |
| CaT-TAG class-shared real payload | 当前候选主方法 | CaT/CCTP 生成 X，再把当前 delta 中真实文本 chunk 按类别共享 payload 规则材料化 |

因此，论文中应把前三类和后两类写清楚：这是一组**同协议适配基线**，不是声称所有方法都已经得到官方实现级复现。

## 4. 当前候选方法到底改了什么

每个阶段的候选方法执行以下步骤：

1. 读取当前类别增量 `delta`，不访问历史原始图。
2. 由 CaT 的 class-wise MMD 构造本阶段新增合成特征 `X`。
3. 使用 CCTP top-5 形成 X 的初始文本来源与类别分配。
4. 对当前 delta 的真实文本进行 chunk 化，并由图特征、文本相似度、类别 prompt、对比项和冗余惩罚进行选择。
5. 每个合成节点最终保存真实可编码的文本 payload，而不是只保存无法对应回原文的软文本向量。
6. 将新 block 追加到 parent memory；本轮候选方法不修改旧 synthetic rows、不修改旧拓扑、不重新访问历史 raw graph。
7. 用同一个 Student、同一训练步数和同一 Student seed 集合评估。

本轮 class-shared payload 的实质贡献不是改变 Student，而是改变**CaT 生成的 X 如何获得真实、可审计、按类别共享的文本内容**。因此可以观察到相同 X 凝聚器下，不同文本材料化策略会造成很大的下游差异。

## 5. 三张大图的最终结果

### 5.1 arXiv

| 方法 | Final F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | 构造秒 |
|---|---:|---:|---:|---:|---:|---:|
| Random real-node | 17.81 | 24.66 | 31.14 | 19.61 | 7.60 | 23.07 |
| K-center | 19.43 | 30.69 | 29.37 | 21.47 | 7.90 | 28.04 |
| Medoid | 18.06 | 24.71 | 29.68 | 19.91 | 7.53 | 40.24 |
| CaT-TAG CCTP top-5 | 24.27 | 48.39 | 24.44 | 27.54 | 5.76 | 152.18 |
| GECC-TAG hard | 20.23 | 31.43 | 30.51 | 22.44 | 7.66 | 186.95 |
| **CaT-TAG class-shared real payload** | **24.79** | 42.56 | **27.54** | 27.48 | **9.56** | 148.49 |

候选方法在 arXiv 的最终 F1 比 CCTP 高 `+0.52`，比所有适配基线都高；但 Accuracy 比 CCTP 低 `5.84` 个百分点，说明收益主要体现在类别平衡和 Macro-F1，而不是所有节点的总正确率。

### 5.2 BookChild

| 方法 | Final F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | 构造秒 |
|---|---:|---:|---:|---:|---:|---:|
| Random real-node | 12.37 | 12.64 | 36.07 | 13.08 | 7.46 | 5.07 |
| K-center | 12.49 | 11.93 | 31.19 | 13.24 | 7.25 | 6.65 |
| Medoid | 12.98 | 13.70 | 35.09 | 13.51 | 9.28 | 9.02 |
| CaT-TAG CCTP top-5 | 4.28 | 31.97 | 4.25 | 4.84 | 0.35 | 77.15 |
| GECC-TAG hard | **14.28** | 14.06 | **37.46** | **15.15** | **8.20** | 93.06 |
| **CaT-TAG class-shared real payload** | 5.46 | 21.07 | 7.82 | 5.80 | 3.10 | 77.53 |

BookChild 是当前候选方法的明确失败条件：最终 F1 低于 GECC、medoid、k-center 和 random real-node。它比 CCTP 高 `+1.18`，但不能因此宣称在 BookChild 上有效。该结果说明 class-shared 文本 payload 的优势依赖于图文结构和类别文本可分性，不能简单外推到所有大图。

### 5.3 SportsFit

| 方法 | Final F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | 构造秒 |
|---|---:|---:|---:|---:|---:|---:|
| Random real-node | 27.44 | 30.55 | 54.11 | 28.86 | 19.59 | 18.82 |
| K-center | 29.42 | 32.01 | 53.95 | 31.42 | 18.46 | 24.72 |
| Medoid | 27.00 | 27.78 | 54.33 | 28.27 | 20.01 | 42.79 |
| CaT-TAG CCTP top-5 | 23.99 | 43.02 | 28.88 | 24.30 | 22.27 | 399.68 |
| GECC-TAG hard | 26.44 | 26.95 | 46.43 | 27.72 | 19.44 | 679.36 |
| **CaT-TAG class-shared real payload** | **46.83** | **63.44** | **59.75** | **49.78** | **30.56** | 408.30 |

SportsFit 上的收益很强：相对 CCTP 为 `+22.84` Final F1、`+20.43` Accuracy、`+30.87` Stage-average F1；相对最强 real-node coreset k-center 仍高 `+17.40` Final F1。该结果是本轮最有力的正向证据，但仍需独立类别顺序确认，不能只用一个顺序作为普遍结论。

## 6. 三图等权平均

| 方法 | Final F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | 最差 stage | 构造秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random real-node | 19.21 | 22.62 | 40.44 | 20.52 | 11.55 | 19.08 | 15.65 |
| K-center | 20.45 | 24.87 | 38.17 | 22.04 | 11.21 | 20.43 | 19.80 |
| Medoid | 19.35 | 22.06 | 39.70 | 20.57 | 12.27 | 19.35 | 30.68 |
| CaT-TAG CCTP top-5 | 17.51 | 41.13 | 19.19 | 18.89 | 9.46 | 17.22 | 209.67 |
| GECC-TAG hard | 20.31 | 24.14 | 38.13 | 21.77 | 11.76 | 20.31 | 319.79 |
| **CaT-TAG class-shared real payload** | **25.69** | **42.36** | **31.70** | **27.69** | **14.41** | **24.07** | 211.44 |

相对 CCTP，候选方法的三图平均变化为：

| 指标 | 候选方法 - CCTP | 候选胜出数据集 |
|---|---:|---:|
| Final F1 | **+8.18** | 3/3 |
| Accuracy | +1.23 | 1/3 |
| Stage avg F1 | **+12.51** | 3/3 |
| Old F1 | +8.80 | 2/3 |
| New F1 | **+4.95** | 3/3 |

相对每个适配基线的 Final F1 平均差值如下：

| 基线 | 平均 Final F1 差值 | 候选胜出数据集 |
|---|---:|---:|
| Random real-node | +6.49 | 2/3 |
| K-center | +5.24 | 2/3 |
| Medoid | +6.35 | 2/3 |
| GECC-TAG hard | +5.38 | 2/3 |
| CCTP top-5 | **+8.18** | 3/3 |

如果每个数据集只取 Final F1 最强的适配基线，候选方法在 arXiv 和 SportsFit 胜出，在 BookChild 落后；因此是 `2/3` 数据集胜出，而不是稳定的全数据集 SOTA。

## 7. 成本和访问约束审计

本轮共检查 `120` 个阶段记录，其中候选方法占 `15` 个阶段：

| 审计项目 | 结果 |
|---|---:|
| historical raw reads 非零 | 0 |
| 使用 report/test 标签构造 | 0 |
| 候选 current-delta-only 失败 | 0 |
| 候选修改旧 memory | 0 |
| 候选修改拓扑 | 0 |
| 候选代表行构造失败 | 0 |

候选方法的平均 Student 成本约 `24.64 s/seed`，与 CCTP 的 `24.52 s/seed` 基本相同；额外成本主要在构造端。候选构造平均约 `211.44 s`，与 CCTP 的 `209.67 s` 接近，明显低于 GECC 的 `319.79 s`。因此当前 payload 机制没有把 Student 训练成本显著放大，但构造器仍然比 k-center/real-node coreset 贵。

## 8. 这轮结果说明什么

1. **class-shared real payload 有真实正向信号。** 它不是只在内部 C0 上提升；在同一协议下相对 CCTP、real-node coreset 和 GECC-TAG 适配版本，三图平均 Final F1 都更高。
2. **收益主要来自文本材料化与类别共享，而不是 Student 改造。** Student、步数、评估和 memory budget 都保持一致，候选只替换 CaT/CCTP 的文本 payload 构造方式。
3. **收益具有明显数据集依赖性。** SportsFit 的图文语义和当前 payload 规则匹配得很好；BookChild 上则出现类别/文本材料化失配，候选明显不如 GECC 和 real-node controls。
4. **Macro-F1 改善不等于 Accuracy 普遍改善。** arXiv 和 BookChild 上候选的 Accuracy 低于 CCTP，说明需要同时报告 Macro-F1、Accuracy、Old F1 和 New F1，不能只挑最有利指标。
5. **当前结果足以把它保留为论文主方法候选，但不足以写成普遍 SOTA。** 更准确的论文表述应是“在严格历史 raw 不可回读协议下，真实 class-shared payload 能在部分大图上显著改善持续 TAG 凝聚的下游适应，且不增加 Student 训练成本；其失效边界与数据集图文匹配有关”。

## 9. 下一步

1. 在 BookChild 补一个独立 construction seed，并至少补一个独立类别顺序；先确认失败是否稳定。
2. 对三图补独立类别顺序或至少对 SportsFit/BookChild 补一组预注册顺序，避免把单一顺序当作泛化结论。
3. 对候选 payload 做最小机制消融：真实 class-shared chunk、普通 top-5 concat、text-only selection、去掉 graph/prompt/redundancy 项；固定 X 和 Student。
4. 如果需要继续优化，优先针对 BookChild 的文本覆盖与类别内多样性，不重新改 Student、不同时改 X 和拓扑。
5. 论文主表中把 `append-edge`/`X+P` 标为内部控制，把 CaT/GECC 标为 adapted baseline；只有拿到官方可复现实现时，才把它们升级为“公开方法原始复现”。

## 10. 可复现文件

- [统一报告](D:/Aprojects/continue_tag_condensation/results/public_baselines_large_corrected_summary_20261003/summary.md)
- [统一 JSON](D:/Aprojects/continue_tag_condensation/results/public_baselines_large_corrected_summary_20261003/summary.json)
- [逐数据集结果](D:/Aprojects/continue_tag_condensation/results/public_baselines_large_corrected_20261003_remote)
- [统一汇总脚本](D:/Aprojects/continue_tag_condensation/tools/summarize_public_baselines_large_corrected_20261003.py)
- [运行脚本](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py)
