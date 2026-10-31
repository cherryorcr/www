# Bounded Student-Compatible X+P：论文证据冻结说明

日期：2026-10-01  
状态：**实验结构冻结，论文结论仍需作者最终审核**

## 1. 当前论文主方法

论文主方法暂定为 **Bounded Student-Compatible X+P**。它保留当前 C0 的 edge-aware 拓扑和持续访问边界，只对当前阶段新合成块执行 Student-compatible outer optimization：

- `X`：新合成节点特征，在有限半径和几何约束下更新；
- `P`：由 Sinkhorn 约束的 payload transport，用来重新分配当前新块内部的语义质量；
- `A`：有限拓扑 revision 不进入默认主方法，只作为可选扩展 `X+P+A` 报告；
- 历史 raw 不回读；历史 parent memory 只作为已保存合成记忆参与递归构造；
- Student 仍是原始两层 GCN、可训练文本投影和双向图文 InfoNCE，不更换为 GraphGPS 或监督 GCN。

## 2. 统一实验协议

### 小图严格协议

- 数据集：Cora、Citeseer；
- 类别顺序：主表 `order19611/order19612`，补充验证 `order19613/order19614`；
- 三阶段，5% cumulative memory；
- Student seeds：150–155；
- 每阶段 scratch 训练，batch 16，steps 为 `max(50,min(2048,16*rows))`；
- report-dev 节点用于评估，不参与构造或选择；
- Macro-F1 与 Accuracy 同时报告。

### 大图严格协议

- 数据集：arXiv、BookChild、SportsFit；
- `base+4`，五阶段；
- 5% cumulative memory；
- Student seeds 100–105；
- 历史 raw 不回读；
- 公开方法为当前 Student/continual 接口下的适配版本，不称为官方端到端复现。

## 3. 结果定位

### 大图主表

三图等权汇总如下：

| 方法 | Final F1 | Accuracy | 阶段平均 F1 |
|---|---:|---:|---:|
| Bounded X+P | 18.22 | 24.76 | 22.15 |
| Bounded X+P+A | 19.20 | 27.30 | 22.25 |
| k-center | 18.18 | 23.99 | 22.34 |
| GECC adapted | 17.59 | 22.47 | 21.29 |
| medoid | 17.40 | 21.85 | 21.26 |
| CaT adapted | 17.38 | 23.03 | 20.66 |
| random-real | 17.26 | 21.89 | 20.13 |
| append-edge | 16.24 | 20.44 | 21.31 |

X+P 相对普通 append-edge 的三图平均 Final F1 提升约 `+1.98 pp`。X+P+A 的平均 Final F1 更高，但拓扑 revision 的收益具有数据集依赖性，不能直接晋升为默认主方法。

### 小图三顺序补充

| 数据集 | X+P Final F1 | CaT | GECC | k-center | medoid |
|---|---:|---:|---:|---:|---:|
| Cora | 47.49 | 53.55 | 47.50 | 49.71 | 48.88 |
| Citeseer | 54.60 | 57.32 | 50.09 | 54.65 | 50.01 |

三顺序结果不支持普遍 SOTA，但显示：

- Cora：X+P 与 GECC 基本持平，低于 CaT、k-center、medoid；
- Citeseer：X+P 高于 GECC 和 medoid，接近 k-center，低于 CaT；
- X+P 的效果不是单个顺序或单个 construction seed 造成的，但仍有明显数据集/顺序依赖。

### 小图四顺序 pooled（冻结补充）

将 `order19611`–`order19614` 的八个数据集-顺序 cell 统一汇总后：

| 数据集 | 方法 | Final F1 | Accuracy | Stage-average F1 |
|---|---|---:|---:|---:|
| Cora | Bounded X+P | 47.05 ± 1.70 | 55.27 ± 1.28 | 42.66 |
| Cora | CaT-MMD-TAG adapted | 53.82 ± 2.01 | 61.02 ± 0.60 | 47.90 |
| Cora | GECC-hard-TAG adapted | 47.33 ± 1.51 | 52.29 ± 1.39 | 45.45 |
| Cora | k-center/FPS | 50.30 ± 4.51 | 55.60 ± 6.44 | 46.14 |
| Cora | medoid | 50.42 ± 4.57 | 57.18 ± 4.47 | 44.28 |
| Citeseer | Bounded X+P | 52.91 ± 3.57 | 58.45 ± 3.67 | 49.15 |
| Citeseer | CaT-MMD-TAG adapted | 56.75 ± 2.41 | 60.45 ± 2.48 | 54.19 |
| Citeseer | GECC-hard-TAG adapted | 49.78 ± 1.95 | 52.81 ± 2.28 | 44.77 |
| Citeseer | k-center/FPS | 53.76 ± 4.11 | 59.63 ± 4.71 | 52.68 |
| Citeseer | medoid | 50.42 ± 2.84 | 55.46 ± 3.32 | 46.59 |

配对 Final F1 差值显示：X+P 相对 CaT 在 `0/8` 个 cell 为正，相对 k-center/FPS 在 `3/8` 个 cell 为正，相对 medoid 在 `3/8` 个 cell 为正，相对 GECC 在 `5/8` 个 cell 为正。因而最终定位应是“有 Student 反馈的持续内容/载荷更新，具有方向性收益和互补性”，而不是普遍超过强 coreset 或静态适配方法。order19614 是当前参数冻结后的额外顺序，可用于压力测试；order19613 的早期探索暴露已在报告中单独标注。

## 4. 当前可写入论文的贡献

建议将贡献写成以下三点，而不是写成“超过所有凝聚基线”：

1. **问题贡献**：形式化历史 raw 不可回读条件下的持续 TAG 凝聚，并明确 memory、文本、拓扑和 Student 训练之间的访问边界。
2. **算法贡献**：提出一个受限 Student-compatible outer operator，在不重建历史原图的情况下，用可微 `X` 更新和有界 `P` transport 改善新阶段合成记忆的可重训性。
3. **实证贡献**：在小图和三类大图上系统比较 coreset、静态 TAG 机制适配、普通 append 和 X+P，显示大图上的方向性收益、小图上的适用边界以及 X+P+A 拓扑扩展的非普适性。

## 5. 不能宣称的内容

- 不能写成稳定 SOTA；
- 不能把 CaT/GECC 适配结果称为官方论文复现；
- 不能把 C0、SR 或 T+degree 的历史结果混入 X+P 主方法成绩；
- 不能把 order19613 称为完全盲测顺序，它曾在早期拓扑实验中出现；
- 不能把 X+P+A 的局部高分写成拓扑 revision 普遍有效；
- 不能声称总计算效率优于 coreset 或完整图训练，当前 outer construction 有额外成本。

## 6. 负结果应保留

论文应保留以下负结果作为方法边界，而不是删除：

- Cora 小图上 X+P 低于 CaT、k-center 和 medoid；
- Citeseer 某些顺序上 construction seed 改变会造成下降；
- 简单文本 attention、动态 response selector、无界 old-memory refresh 和 GraphGPS Student 没有形成稳定主方法收益；
- X+P+A 的拓扑 revision 在不同数据集/顺序上的方向不一致；
- 更强静态凝聚方法可以在部分条件下超过 X+P。

这些结果支持一个清晰判断：Student-compatible 优化提供了持续凝聚中的新信息更新机制，但其优化目标与最终零样本分类 F1 并不总是同向。

## 7. 证据入口

- [最终证据与基线矩阵](../current/BOUNDED_XP_FINAL_EVIDENCE_AND_BASELINE_MATRIX_ZH_2026-10-01.md)
- [三顺序补充与成本账](../current/BOUNDED_XP_THREE_ORDER_SUPPLEMENT_AND_COST_ZH_2026-10-01.md)
- [四顺序 pooled 汇总](../current/BOUNDED_XP_FOUR_ORDER_POOLED_SUMMARY_ZH_2026-10-01.md)
- [四顺序 Final F1 可视化](../../results/bounded_xp_final_evidence_20261001/bounded_xp_four_order_final_f1.png)
- [order19613 配对基线比较](../current/BOUNDED_XP_ORDER19613_PAIRED_BASELINE_COMPARISON_ZH_2026-10-01.md)
- [order19613 artifact 审计](../../results/bounded_xp_final_evidence_20261001/order19613_artifact_audit.json)
- [最终证据矩阵 JSON](../../results/bounded_xp_final_evidence_20261001/evidence_matrix.json)

## 8. 投稿前剩余工作

1. 将当前主表、三顺序补充和成本表转换为论文图表，并统一小数位与误差定义。
2. 将四顺序 pooled 结果作为当前小图稳定性主表；不再根据这些结果为 X+P 重新调参。
3. 对公开方法补齐代码版本、适配改动和访问权限说明。
4. 写出 `X/P` outer objective、访问不变量、复杂度和失败条件的正式定义。
5. 最终论文中把“方向性收益、适用边界、负结果和成本”放在同一证据链中，不再新增未预注册模块。

当前最稳妥的论文定位是：

> 在历史原始图不可回读的持续 TAG 凝聚协议下，Bounded Student-Compatible X+P 通过受限 outer optimization 更新新合成内容和 payload 分配，在固定 memory budget 下相对普通追加获得方向性收益；该收益在大图上更明显，在小图上具有数据集和类别顺序依赖。

