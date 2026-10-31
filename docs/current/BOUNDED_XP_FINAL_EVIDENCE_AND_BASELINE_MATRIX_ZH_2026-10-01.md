# Bounded Student-Compatible X+P：最终证据与基线矩阵

日期：2026-10-01

> 当前主方法候选固定为 **Bounded Student-Compatible X+P**。A（有限拓扑 revision）作为可选扩展单独报告，不把 X+P+A 的数据集依赖收益写成无条件主方法收益。

## 1. 读表方式

本报告把结果分成三层。大图 strict continual 表直接使用每个阶段的 `results.json`，保证 append、X+P/X+P+A 和 coreset 的 Student、边表示和评估路径可核对。CaT、GECC 是机制适配，TAGSAM 是 loss/组件适配，均单独标注。小图现在同时有两顺序严格 X+P 与 X+P+A，但仍按数据集和顺序解释，不能把它们概括成普遍排名。

## 2. 大图 strict continual 主表

协议：arXiv、BookChild、SportsFit；`base+4` 五阶段；5% cumulative memory；历史 raw 不可回读；原始 edge-aware 两层 GCN、冻结 MiniLM、可训练文本投影、双向图文 InfoNCE；Student seeds 100–105；每阶段从头训练；报告 cumulative Macro-F1、Accuracy、Old F1、New F1。

| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 | 真实构造秒 |
|---|---|---|---|---|---|---|---|
| arxiv | append-edge | 17.06 | 25.34 | 22.43 | 24.60 | 18.82 | — |
| arxiv | Bounded X+P | 18.21 | 29.80 | 22.81 | 26.53 | 20.01 | 10.69 |
| arxiv | Bounded X+P+A | 17.68 | 29.51 | 22.07 | 25.58 | 19.39 | 37.80 |
| arxiv | random-real | 15.95 | 22.85 | 20.53 | 22.89 | 17.77 | — |
| arxiv | k-center | 18.11 | 31.25 | 21.25 | 26.52 | 19.43 | — |
| arxiv | medoid | 16.16 | 24.99 | 20.91 | 23.01 | 18.29 | — |
| arxiv | CaT-MMD-TAG adapted | 19.11 | 32.90 | 23.88 | 28.12 | 20.57 | — |
| arxiv | GECC-hard-TAG adapted | 17.53 | 27.69 | 22.85 | 25.77 | 18.96 | — |
| bookchild | append-edge | 10.82 | 10.85 | 14.58 | 15.04 | 12.46 | — |
| bookchild | Bounded X+P | 11.12 | 13.42 | 13.88 | 15.38 | 12.86 | 6.44 |
| bookchild | Bounded X+P+A | 11.26 | 12.26 | 13.92 | 15.79 | 12.95 | 14.40 |
| bookchild | random-real | 10.58 | 11.68 | 13.62 | 14.55 | 12.33 | — |
| bookchild | k-center | 10.71 | 11.08 | 13.86 | 14.45 | 12.49 | — |
| bookchild | medoid | 11.91 | 12.77 | 13.98 | 16.42 | 13.78 | — |
| bookchild | CaT-MMD-TAG adapted | 11.91 | 11.99 | 13.98 | 16.75 | 13.57 | — |
| bookchild | GECC-hard-TAG adapted | 13.29 | 14.60 | 16.16 | 18.37 | 15.35 | — |
| sportsfit | append-edge | 20.83 | 25.14 | 26.92 | 26.34 | 24.76 | — |
| sportsfit | Bounded X+P | 25.32 | 31.05 | 29.76 | 32.47 | 29.28 | 14.04 |
| sportsfit | Bounded X+P+A | 28.65 | 40.13 | 30.77 | 36.86 | 32.80 | 50.46 |
| sportsfit | random-real | 25.25 | 31.13 | 26.24 | 31.85 | 29.04 | — |
| sportsfit | k-center | 25.73 | 29.65 | 31.91 | 33.03 | 29.42 | — |
| sportsfit | medoid | 24.13 | 27.80 | 28.88 | 31.20 | 28.58 | — |
| sportsfit | CaT-MMD-TAG adapted | 21.13 | 24.20 | 24.12 | 26.39 | 24.59 | — |
| sportsfit | GECC-hard-TAG adapted | 21.93 | 25.13 | 24.87 | 28.79 | 25.35 | — |

其中 X+P/X+P+A 的真实构造秒来自 outer construction audit；public runner 中的 sealed-memory 加载秒不作为主方法成本。append-edge 也是封存 artifact，因此没有把加载时间当成构造时间。

### 三张大图等权汇总

| 方法 | Final F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 | 相对 append Final F1 |
|---|---|---|---|---|---|---|
| Bounded X+P | 18.22 | 24.76 | 22.15 | 24.79 | 20.72 | 1.98 |
| Bounded X+P+A | 19.20 | 27.30 | 22.25 | 26.08 | 21.71 | 2.96 |
| CaT-MMD-TAG adapted | 17.38 | 23.03 | 20.66 | 23.76 | 19.58 | 1.15 |
| GECC-hard-TAG adapted | 17.59 | 22.47 | 21.29 | 24.31 | 19.89 | 1.35 |
| append-edge | 16.24 | 20.44 | 21.31 | 21.99 | 18.68 | 0.00 |
| k-center | 18.18 | 23.99 | 22.34 | 24.67 | 20.45 | 1.95 |
| medoid | 17.40 | 21.85 | 21.26 | 23.54 | 20.22 | 1.16 |
| random-real | 17.26 | 21.89 | 20.13 | 23.10 | 19.71 | 1.02 |

按这批 stage-level artifacts，X+P 相对 append-edge 的最终 F1 方向应按逐图结果计算；不能把阶段平均高低混同为最终阶段优劣。X+P+A 只作为拓扑扩展，A 在 arXiv、BookChild、SportsFit 的差异方向不同。

## 3. 小图多顺序结果

当前 `xpa_joint_full_20260929` 在 Cora/Citeseer 上覆盖 order19611、order19612 和三阶段，方法是 X+P+A；严格 X+P 结果在 `xp_joint_full_20261001` 生成后自动并入。公开 TAG 适配器覆盖相同数据集和顺序，但属于 mechanism-adapted，不能称为官方端到端结果。

| 数据集 | 方法 | 顺序数 | Final F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 |
|---|---|---|---|---|---|---|---|
| citeseer | Bounded X+P | 2.00 | 55.32 | 61.02 | 53.18 | 57.67 | 50.63 |
| citeseer | Bounded X+P+A | 2.00 | 56.85 | 63.09 | 54.67 | 59.02 | 52.52 |
| citeseer | CaT-MMD-TAG adapted | 2.00 | 55.93 | 59.38 | 54.93 | 58.39 | 51.02 |
| citeseer | GECC-hard-TAG adapted | 2.00 | 50.72 | 53.19 | 47.32 | 52.27 | 47.61 |
| citeseer | k-center | 2.00 | 55.80 | 61.94 | 58.97 | 58.44 | 50.51 |
| citeseer | medoid | 2.00 | 48.14 | 52.72 | 49.46 | 51.19 | 42.06 |
| cora | Bounded X+P | 2.00 | 47.53 | 55.75 | 40.05 | 44.64 | 51.38 |
| cora | Bounded X+P+A | 2.00 | 47.87 | 55.57 | 40.19 | 43.29 | 53.98 |
| cora | CaT-MMD-TAG adapted | 2.00 | 53.31 | 60.55 | 46.12 | 47.63 | 60.88 |
| cora | GECC-hard-TAG adapted | 2.00 | 46.68 | 51.88 | 41.53 | 40.96 | 54.30 |
| cora | k-center | 2.00 | 52.57 | 56.92 | 47.89 | 47.63 | 59.17 |
| cora | medoid | 2.00 | 46.49 | 53.51 | 37.41 | 42.78 | 51.43 |

小图同时区分 X+P 与 X+P+A。当前 X+P 已经完成两顺序严格评估：Cora 上低于 k-center/CaT，Citeseer 上接近但略低；X+P+A 仅在部分顺序和数据集改善，因此不能用它替代 X+P 的主方法结论。

### 小图 construction seed 稳定性

固定数据集、类别顺序、memory budget、X/P 配置、Student seeds 和 report 规则，只将 support-plan 的 `plan_seed` 从 9320 改为 9321。该实验不产生新的主表排名，而是检查合成构造对 plan seed 的敏感性。

| 数据集 | 顺序 | 9320 Final F1 | 9321 Final F1 | Final F1 差值 | 9320 Accuracy | 9321 Accuracy | 9320 New F1 | 9321 New F1 |
|---|---|---|---|---|---|---|---|---|
| cora | order19611 | 49.31 | 51.29 | 1.98 | 56.77 | 57.01 | 46.35 | 51.84 |
| cora | order19612 | 45.75 | 48.45 | 2.70 | 54.74 | 57.93 | 56.42 | 58.34 |
| citeseer | order19611 | 55.94 | 56.13 | 0.20 | 60.50 | 61.13 | 43.68 | 42.77 |
| citeseer | order19612 | 54.71 | 53.08 | -1.63 | 61.55 | 59.56 | 57.58 | 55.29 |

四个 cell 的平均 Final F1 变化为 `0.810 pp`，正向 cell 为 `3/4`。Cora 两个顺序均为正向，Citeseer 一个顺序轻微正向、一个顺序下降；因此 X+P 的 construction seed 影响可观但没有推翻大图方向性收益。
这项结果支持报告 construction-seed 方差，而不是只展示单个 plan seed；它也说明后续应优先补独立类别顺序和 paired baseline，不再针对某个 seed 继续调参。

## 4. 额外顺序：order19613 同协议基线比较

order19613 单独跑了 X+P、C0 anchor、CaT-MMD-TAG、GECC-hard-TAG、k-center 和 medoid，固定三阶段、5% memory、6 个 Student seeds。该实验作为额外顺序验证，不替换原两顺序主表；CaT/GECC 仍标记为机制适配。

| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 |
|---|---|---:|---:|---:|
| cora | Bounded X+P | 47.42 ± 2.11 | 55.78 ± 2.00 | 45.20 |
| cora | C0 append-edge | 49.18 ± 4.31 | 57.13 ± 3.18 | 44.97 |
| cora | CaT-MMD-TAG adapted | 54.03 ± 1.24 | 61.69 ± 1.19 | 47.24 |
| cora | GECC-hard-TAG adapted | 49.15 ± 0.82 | 53.38 ± 0.79 | 50.69 |
| cora | k-center native | 43.98 ± 2.66 | 47.17 ± 4.20 | 38.94 |
| cora | medoid native | 53.67 ± 1.90 | 60.70 ± 1.26 | 48.45 |
| citeseer | Bounded X+P | 53.16 ± 2.47 | 58.46 ± 1.97 | 47.80 |
| citeseer | C0 append-edge | 52.20 ± 1.20 | 57.58 ± 1.70 | 47.66 |
| citeseer | CaT-MMD-TAG adapted | 60.09 ± 1.32 | 64.11 ± 1.52 | 54.07 |
| citeseer | GECC-hard-TAG adapted | 48.84 ± 3.90 | 53.24 ± 3.66 | 41.51 |
| citeseer | k-center native | 52.37 ± 1.47 | 59.09 ± 1.55 | 49.52 |
| citeseer | medoid native | 53.73 ± 2.49 | 57.31 ± 1.44 | 43.58 |

该顺序中，Cora 的 X+P Final F1 为 47.42，低于 CaT/medoid、高于 GECC/k-center；Citeseer 的 X+P Final F1 为 53.16，低于 CaT/k-center/medoid、高于 GECC。因此它支持方向性收益和方法互补性，不支持普遍 SOTA。
详细报告：`D:/Aprojects/continue_tag_condensation/docs/current/BOUNDED_XP_ORDER19613_PAIRED_BASELINE_COMPARISON_ZH_2026-10-01.md`。
artifact 审计：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/order19613_artifact_audit.json`。

## 5. 三顺序补充与成本账

三顺序补充将 order19611、order19612、order19613 的 X+P 与 CaT/GECC/k-center/medoid 统一汇总；它不替换两顺序主表，因为 order19613 曾参与早期拓扑实验。

| 数据集 | 方法 | 三顺序 Final F1 | 跨顺序标准差 | Accuracy | 阶段平均 F1 |
|---|---|---:|---:|---:|---:|
| citeseer | Bounded X+P | 54.60 | 1.39 | 60.17 | 51.39 |
| citeseer | CaT-MMD-TAG adapted | 57.32 | 2.61 | 60.95 | 54.64 |
| citeseer | GECC-hard-TAG adapted | 50.09 | 2.27 | 53.20 | 45.39 |
| citeseer | k-center | 54.65 | 4.53 | 60.99 | 55.82 |
| citeseer | medoid | 50.01 | 3.32 | 54.25 | 47.50 |
| cora | Bounded X+P | 47.49 | 1.78 | 55.76 | 41.77 |
| cora | CaT-MMD-TAG adapted | 53.55 | 2.37 | 60.93 | 46.49 |
| cora | GECC-hard-TAG adapted | 47.50 | 1.80 | 52.38 | 44.59 |
| cora | k-center | 49.71 | 5.34 | 53.67 | 44.91 |
| cora | medoid | 48.88 | 4.15 | 55.90 | 41.09 |

三顺序结果显示：Cora 上 X+P 与 GECC 基本持平但低于 CaT、k-center 和 medoid；Citeseer 上高于 GECC/medoid、接近 k-center 但低于 CaT。成本账同时分开报告 outer 构造时间和 Student 训练时间。
详细文档：`D:/Aprojects/continue_tag_condensation/docs/current/BOUNDED_XP_THREE_ORDER_SUPPLEMENT_AND_COST_ZH_2026-10-01.md`。

## 6. TAGSAM 适配参考

| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 | 状态 |
|---|---|---|---|---|---|
| cora | asm_rowcol_semantic | 42.30 | 53.26 | 47.47 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| cora | init_selfloop | 49.81 | 59.41 | 63.31 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| cora | tagsam_selfloop | 39.81 | 48.34 | 64.83 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| cora | tagsam_semantic | 32.84 | 42.07 | 39.86 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| citeseer | asm_rowcol_semantic | 46.42 | 47.44 | 45.84 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| citeseer | init_selfloop | 56.14 | 61.02 | 63.21 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| citeseer | tagsam_selfloop | 47.56 | 50.00 | 51.74 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |
| citeseer | tagsam_semantic | 30.49 | 31.56 | 40.51 | TAGSAM mechanism adapter; not official end-to-end TAGSAM |

这些结果使用官方 TAGSAM 组件/损失的适配实现，但 Student 和 continual current-delta 接口仍是本项目协议，因此不能与官方论文数字直接比较，也不与 strict edge-aware 主表直接混排。

## 7. 当前可支持的结论

1. X+P 是当前最适合继续作为主方法的版本：它通过 Student-compatible outer optimization 调整新合成内容，并用有界 P 做 payload transport，保留 C0 的 edge-aware 拓扑。
2. X+P+A 的拓扑 revision 不是普适增益，应作为受限扩展或消融。
3. k-center 是必须保留的强 coreset 参照；CaT/GECC 适配方法在部分大图阶段更强，因此论文不能宣称普遍 SOTA。
4. 小图严格 X+P 两顺序矩阵已经补齐，但它在 Cora 上明显落后于 k-center/CaT，在 Citeseer 上接近但略低；construction seed 9321 的附加复现显示 4 个 cell 中 3 个 Final F1 正向，平均变化 +0.81 pp，但仍存在顺序敏感性。order19613 的额外同协议比较进一步显示 X+P 不是普遍 SOTA，而是对不同图和强基线呈互补关系。
5. SR 不进入当前主竞争表；它保留在历史/启发式负结果附录。

## 8. 证据限制与下一步

先冻结 X+P 的算法结构，不再用当前结果继续扫参数；construction seed 稳定性、order19613 同顺序基线比较、三顺序汇总和成本账均已完成。下一步应转为论文证据冻结与方法边界说明，而不是继续针对某个顺序调参。论文中应把大图上的方向性收益、小图上的负结果、三顺序互补性和外部适配边界同时呈现，解释为什么内容/载荷 outer optimization 对图规模和传播结构敏感。不要把旧版 `summary.json` 中与 stage-level `results.json` 不一致的汇总值用于主表。

## 7. 产物

- JSON：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/evidence_matrix.json`
- 大图逐数据集 CSV：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/large_strict_rows.csv`
- 大图汇总 CSV：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/large_pooled.csv`
- 小图汇总 CSV：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/small_pooled.csv`
- 小图 X+P artifact 审计：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/small_xp_artifact_audit.json`
- 小图 construction seed 稳定性：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/small_xp_plan_seed_stability.json`
- order19613 额外顺序比较：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/order19613_xp_baseline_comparison.json`
- order19613 artifact 审计：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/order19613_artifact_audit.json`
- 三顺序补充与成本账：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/three_order_supplement_and_cost.json`

## 8. 自动审计提示

- D:\Aprojects\continue_tag_condensation\tmp_remote_results\public_baselines_large_20261001_gpu0_r4\summary.json overall/x_plus_p disagrees with stage-level results.json; final matrix uses stage-level artifacts.
- D:\Aprojects\continue_tag_condensation\tmp_remote_results\public_baselines_large_20261001_gpu0_r4\summary.json overall/x_plus_p disagrees with stage-level results.json; final matrix uses stage-level artifacts.
- D:\Aprojects\continue_tag_condensation\tmp_remote_results\public_baselines_large_20261001_gpu1_r4\summary.json overall/x_plus_p disagrees with stage-level results.json; final matrix uses stage-level artifacts.
