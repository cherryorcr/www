# Bounded X+P：四个独立类别顺序 pooled 汇总

> 本报告只汇总已经完成的冻结结果，不重新训练、不根据新结果调参。order19611–19613 来自三顺序补充证据，order19614 是额外的盲顺序比较。

## 1. 统一协议

- 数据集：Cora、Citeseer。
- 类别顺序：`order19611`、`order19612`、`order19613`、`order19614`；每个顺序三阶段。
- memory：同一 5% cumulative budget。
- Student：原始两层 GCN、可训练文本投影、双向图文 InfoNCE；每个 cell 使用相同 report seeds。
- X+P：`plan_seed=9320`、inner steps=8、outer steps=4、outer lr=0.02，沿用 C0 edge-aware topology；不启用 A 拓扑 revision。
- CaT/GECC：本项目 continual Student 接口下的机制适配，不是官方端到端复现；k-center/FPS 与 medoid 是选择型控制方法。
- SR、X+P+A、C0 不放入这张公开基线 pooled 主表。C0 仍作为内部 anchor 单独记录。

## 2. 分数据集、四顺序汇总

数值为四个顺序的均值 ± 跨顺序样本标准差；`Final F1` 和 `Accuracy` 是最终阶段，`Stage-average F1` 是三阶段等权平均。

| 数据集 | 方法 | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|
| cora | Bounded X+P | 47.05 ± 1.70 | 55.27 ± 1.28 | 42.66 | 44.26 | 50.78 |
| cora | CaT-MMD-TAG adapted | 53.82 ± 2.01 | 61.02 ± 0.60 | 47.90 | 49.23 | 59.95 |
| cora | GECC-hard-TAG adapted | 47.33 ± 1.51 | 52.29 ± 1.39 | 45.45 | 42.42 | 53.87 |
| cora | k-center/FPS | 50.30 ± 4.51 | 55.60 ± 6.44 | 46.14 | 44.76 | 57.67 |
| cora | medoid | 50.42 ± 4.57 | 57.18 ± 4.47 | 44.28 | 46.01 | 56.30 |
| citeseer | Bounded X+P | 52.91 ± 3.57 | 58.45 ± 3.67 | 49.15 | 52.05 | 54.64 |
| citeseer | CaT-MMD-TAG adapted | 56.75 ± 2.41 | 60.45 ± 2.48 | 54.19 | 54.63 | 61.00 |
| citeseer | GECC-hard-TAG adapted | 49.78 ± 1.95 | 52.81 ± 2.28 | 44.77 | 48.24 | 52.86 |
| citeseer | k-center/FPS | 53.76 ± 4.11 | 59.63 ± 4.71 | 52.68 | 50.94 | 59.39 |
| citeseer | medoid | 50.42 ± 2.84 | 55.46 ± 3.32 | 46.59 | 48.95 | 53.36 |

## 3. 每个顺序的 Final F1

| 数据集 | 顺序 | Bounded X+P | CaT-MMD-TAG | GECC-hard-TAG | k-center/FPS | medoid |
|---|---|---:|---:|---:|---:|---:|
| cora | order19611 | 49.31 | 55.64 | 47.78 | 54.54 | 46.39 |
| cora | order19612 | 45.75 | 50.98 | 45.58 | 50.60 | 46.58 |
| cora | order19613 | 47.42 | 54.03 | 49.15 | 43.98 | 53.67 |
| cora | order19614 | 45.73 | 54.64 | 46.80 | 52.06 | 55.03 |
| citeseer | order19611 | 55.94 | 56.93 | 48.72 | 51.72 | 47.33 |
| citeseer | order19612 | 54.71 | 54.93 | 52.71 | 59.87 | 48.96 |
| citeseer | order19613 | 53.16 | 60.09 | 48.84 | 52.37 | 53.73 |
| citeseer | order19614 | 47.84 | 55.06 | 48.86 | 51.07 | 51.67 |

## 4. X+P 相对基线的配对差值

正值表示 X+P 的最终阶段 F1 更高。`positive cells` 是同一数据集的四个顺序中 X+P 胜出的数量。

| 数据集 | 基线 | 平均 ΔFinal F1 | 正向 cell | 最差 Δ | 最好 Δ | 平均 ΔAccuracy |
|---|---|---:|---:|---:|---:|---:|
| cora | CaT-MMD-TAG adapted |  -6.77 | 0/4 |  -8.91 |  -5.23 |  -5.75 |
| cora | GECC-hard-TAG adapted |  -0.27 | 2/4 |  -1.73 |   1.53 |   2.98 |
| cora | k-center/FPS |  -3.24 | 1/4 |  -6.32 |   3.44 |  -0.32 |
| cora | medoid |  -3.37 | 1/4 |  -9.30 |   2.91 |  -1.91 |
| citeseer | CaT-MMD-TAG adapted |  -3.84 | 0/4 |  -7.22 |  -0.22 |  -2.00 |
| citeseer | GECC-hard-TAG adapted |   3.13 | 3/4 |  -1.01 |   7.21 |   5.64 |
| citeseer | k-center/FPS |  -0.84 | 2/4 |  -5.16 |   4.22 |  -1.18 |
| citeseer | medoid |   2.49 | 2/4 |  -3.83 |   8.60 |   2.99 |

### 全部数据集与顺序合并

| 基线 | 平均 ΔFinal F1 | 正向 cell | 最差 Δ | 最好 Δ | 平均 ΔAccuracy |
|---|---:|---:|---:|---:|---:|
| CaT-MMD-TAG adapted | -5.30 | 0/8 | -8.91 | -0.22 | -3.87 |
| GECC-hard-TAG adapted | 1.43 | 5/8 | -1.73 | 7.21 | 4.31 |
| k-center/FPS | -2.04 | 3/8 | -6.32 | 4.22 | -0.75 |
| medoid | -0.44 | 3/8 | -9.30 | 8.60 | 0.54 |

## 5. C0 anchor（单独记录）

order19614 上的 C0 只用于检查 outer optimization 的增量，不能代替四顺序公开基线 pooled：

| 数据集 | X+P Final F1 | C0 Final F1 | X+P - C0 |
|---|---:|---:|---:|
| cora | 45.73 | 46.00 | -0.27 |
| citeseer | 47.84 | 51.35 | -3.51 |

## 6. 结果解释

1. 四顺序 pooled 后，X+P 的表现应被表述为具有方向性收益和方法互补性，而不是普遍超过所有强基线。
2. CaT、k-center/FPS、medoid 在部分图或顺序上更强；这说明 X+P 的 Student-compatible outer optimization 还受到传播结构、类别顺序和 support plan 的影响。
3. order19614 是没有参与当前参数开发的额外顺序，应作为泛化压力测试；order19613 曾出现在早期拓扑实验中，不能称为完全盲测。
4. construction seed 9321 的独立小实验已经显示 X+P 存在可见的 construction variance，因此 pooled 汇总不能只报告 canonical order 或单个 seed。
5. 该汇总冻结的是证据口径，不自动把 X+P 晋升为普遍 SOTA；论文应同时报告大图正向信号、小图负结果和适配基线边界。

## 7. 文件与限制

- 每 cell 原始汇总：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/four_order_cells.csv`
- 分数据集 pooled：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/four_order_pooled.csv`
- 配对逐 cell 差值：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/four_order_pairwise_deltas.csv`
- 配对 pooled 差值：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/four_order_delta_summary.csv`
- Final F1 顺序曲线：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/bounded_xp_four_order_final_f1.png`
- 本报告没有重新训练，也没有使用最终 F1 选择配置。
- CaT/GECC 仍是机制适配，不能与官方论文数字直接比较。
