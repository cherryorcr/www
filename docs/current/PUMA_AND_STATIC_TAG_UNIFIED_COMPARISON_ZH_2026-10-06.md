# PUMA 与静态 TAG 统一比较实验

## 1. 实验目的

本轮实验补充两类参照：

1. 将 PUMA 的核心机制适配到当前 TAG 持续凝聚协议；
2. 将静态 TAG full-history 凝聚直接应用到大图，再使用相同的 Prompt zero-shot Student 评估。

两类结果都与已有的 Adaptive coherent、CaT-MMD-TAG、GECC-hard、K-center、Medoid 和 Random coreset 统一汇总。结果文件位于：

- [统一明细表](../../results/puma_static_unified_comparison_20261006/unified_cells.csv)
- [严格持续表](../../results/puma_static_unified_comparison_20261006/strict_continual_cells.csv)
- [离线参考表](../../results/puma_static_unified_comparison_20261006/offline_reference_cells.csv)
- [中文汇总](../../results/puma_static_unified_comparison_20261006/summary.md)

## 2. 统一协议

- 数据集：Arxiv、SportsFit、WikiCS、BookHis；
- 阶段：固定 base+4 类别顺序；
- memory：5% 累积预算；
- Student：同一个 Prompt zero-shot Student，Student seeds 为 100--105；
- 文本编码器：同一个冻结 MiniLM；
- 指标：Final Macro-F1、Accuracy、Old F1、New F1、Stage-average F1、Worst-stage F1；
- 构造成本：仅计 synthetic memory construction，Student 训练时间单列；
- 所有测试标签只用于评估，不参与 memory construction。

严格持续方法只读取当前阶段 raw delta 和已有 synthetic prefix，旧 prefix 不回读 raw、不重写。静态 TAG 方法每个阶段读取全部已见 raw 并重建 memory，因此是 offline/reference，不作为严格持续排名。

## 3. 方法及 TAG 文本附着适配

| 方法 | 原始机制/角色 | TAG 文本附着适配 | 访问约束 | 拓扑 |
|---|---|---|---|---|
| Adaptive coherent | CaT class-wise MMD 优化 X，并使用 X-conditioned coherent routing | 从当前 delta 构造合法文本池；根据合成 X 的离散度自适应使用 1/2/3 个 slot，按图相似度分配文本并 hard materialize | current delta only，frozen prefix | Student self-loop |
| CaT-MMD-TAG + CCTP top-5 | CaT 分布匹配 | 每个 synthetic row 从当前 delta 取 graph-nearest top-5 文本来源，在 320 字符约束内重新编码 | current delta only，frozen prefix | Student self-loop |
| GECC-hard-TAG adapted | GECC hard-hop/Lloyd 代表点 | 在 current delta 选择代表点，文本随代表点保留 | current delta only，frozen prefix | Student self-loop |
| K-center/FPS | 真实节点覆盖 coreset | 在 current delta 联合图文空间做 farthest-point sampling，文本随真实节点保留 | current delta only，frozen prefix | Student self-loop |
| Medoid coreset | 真实节点代表点 | 在 current delta 联合空间选择 medoid，保留对应原始文本 | current delta only，frozen prefix | Student self-loop |
| Random real coreset | 真实节点随机控制 | 从 current delta 随机选择真实节点及其对应文本 | current delta only，frozen prefix | Student self-loop |
| PUMA-inspired TAG | one-hop propagation、confidence-filtered pseudo-label teacher、CaT matching | 在当前 delta 上传播一次；用图特征和冻结文本训练线性 pseudo-label teacher；仅保留高置信候选；按 teacher/pseudo label 在 current delta 合法文本中取 top-5，并重新编码 | current delta only，frozen prefix | edge-free，Student self-loop |
| Static-TAG full-history | full-history joint graph-text condensation | 对所有已见 raw 做 joint graph-text clustering，融合 cluster 文本，再构造 text-kNN 拓扑 | full seen raw，允许旧 memory 重写 | text-kNN |

PUMA 结果是 PUMA-inspired TAG adaptation，不是官方 PUMA 原生复现；静态 TAG 结果是本项目统一的 scalable joint graph-text constructor，不应写成官方 TAGSAM/STAD 原样复现。

## 4. 结果摘要

四个大图等权平均如下：

| 方法 | 协议 | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | 构造时间/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Adaptive coherent | strict continual | 41.15 | 64.21 | 43.59 | 24.70 | 45.36 | 113.1 |
| PUMA-inspired TAG | strict continual | 28.01 | 34.77 | 29.42 | 18.06 | 47.45 | 84.3 |
| CaT-MMD-TAG + CCTP top-5 | strict continual | 29.98 | 45.00 | 31.55 | 18.38 | 40.49 | 152.9 |
| GECC-hard-TAG adapted | strict continual | 29.26 | 36.52 | 30.71 | 18.59 | 47.88 | 232.0 |
| K-center/FPS | strict continual | 29.78 | 38.47 | 31.41 | 18.04 | 47.80 | 13.9 |
| Medoid coreset | strict continual | 29.26 | 35.43 | 30.89 | 16.80 | 47.98 | 22.5 |
| Random real coreset | strict continual | 29.49 | 36.75 | 30.75 | 20.57 | 48.81 | 10.9 |
| Static-TAG full-history | offline reference | 30.23 | 38.61 | 31.61 | 20.28 | 48.66 | 122.0 |

主要观察：

- PUMA 在当前统一 Student 上没有超过 CaT 或 Adaptive coherent；它的 stage-average F1 较高，但最终累积分类受旧/新类别边界和文本附着影响，Final F1 偏低；
- 静态 full-history TAG 在 WikiCS 上达到 56.35 F1，在 SportsFit 上达到 27.32 F1，但没有稳定超过严格持续方法；它的主要价值是提供“允许回读历史 raw 时”的离线参考，而不是证明持续算法优越；
- Adaptive coherent 的四图平均 Final F1 和 Accuracy 仍明显最高，但其结果来自独立的文本 routing 机制，不能把 PUMA 或静态 TAG 的低分解释为实现错误；
- BookHis 上所有方法的 New F1 都较低，Accuracy 也明显受类别分布影响，应以 Macro-F1 和 New F1 为主解释；
- PUMA 的构造时间低于 Adaptive coherent 和 CaT，但不是当前性能最优方案。

## 5. 可视化

- [严格持续 Final F1](../../results/puma_static_unified_comparison_20261006/figures/final_f1_by_dataset.png)
- [严格持续 Accuracy](../../results/puma_static_unified_comparison_20261006/figures/accuracy_by_dataset.png)
- [包含静态 reference 的 Final F1](../../results/puma_static_unified_comparison_20261006/figures/final_f1_with_static_reference.png)
- [Old/New F1 trade-off](../../results/puma_static_unified_comparison_20261006/figures/old_new_tradeoff.png)
- [构造成本](../../results/puma_static_unified_comparison_20261006/figures/construction_cost.png)

## 6. 论文中应如何表述

PUMA 应作为“外部机制适配基线”或“机制参照”报告，不能声称是官方 PUMA 复现。静态 TAG 应放在单独的 offline/reference 小节，并明确其 full-history 访问和旧 memory 重写能力。严格持续主表仍以相同访问约束下的 Adaptive coherent、CaT、GECC 和 coreset 为主。
