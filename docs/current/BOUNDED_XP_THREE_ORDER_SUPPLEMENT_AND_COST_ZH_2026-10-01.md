# Bounded X+P：三顺序补充证据与成本账

> 本文档把 `order19611/19612/19613` 合并为补充稳定性证据。它不替换原两顺序主表，也不把 X+P+A 混入三顺序公开基线比较。

## 1. 三顺序汇总

| 数据集 | 方法 | Final F1 | 跨顺序标准差 | Accuracy | 阶段平均 F1 | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| citeseer | Bounded X+P | 54.60 | 1.39 | 60.17 | 51.39 | 53.36 | 57.09 |
| citeseer | CaT-MMD-TAG adapted | 57.32 | 2.61 | 60.95 | 54.64 | 56.85 | 58.26 |
| citeseer | GECC-hard-TAG adapted | 50.09 | 2.27 | 53.20 | 45.39 | 49.40 | 51.47 |
| citeseer | k-center | 54.65 | 4.53 | 60.99 | 55.82 | 53.72 | 56.53 |
| citeseer | medoid | 50.01 | 3.32 | 54.25 | 47.50 | 50.45 | 49.12 |
| cora | Bounded X+P | 47.49 | 1.78 | 55.76 | 41.77 | 43.32 | 53.05 |
| cora | CaT-MMD-TAG adapted | 53.55 | 2.37 | 60.93 | 46.49 | 45.91 | 63.73 |
| cora | GECC-hard-TAG adapted | 47.50 | 1.80 | 52.38 | 44.59 | 41.26 | 55.82 |
| cora | k-center | 49.71 | 5.34 | 53.67 | 44.91 | 41.88 | 60.15 |
| cora | medoid | 48.88 | 4.15 | 55.90 | 41.09 | 43.75 | 55.72 |

## 2. X+P 相对公开适配基线

| 数据集 | 基线 | X+P - 基线 Final F1 | 正向 cell | 最差 cell |
|---|---|---:|---:|---:|
| cora | CaT-MMD-TAG adapted | -6.06 | 0/3 | -6.61 |
| cora | GECC-hard-TAG adapted | -0.01 | 2/3 | -1.73 |
| cora | k-center | -2.22 | 1/3 | -5.23 |
| cora | medoid | -1.39 | 1/3 | -6.24 |
| citeseer | CaT-MMD-TAG adapted | -2.72 | 0/3 | -6.93 |
| citeseer | GECC-hard-TAG adapted | +4.51 | 3/3 | +2.01 |
| citeseer | k-center | -0.05 | 2/3 | -5.16 |
| citeseer | medoid | +4.60 | 2/3 | -0.57 |

## 3. 小图 order19613 成本

| 数据集 | 方法 | 构造秒 | Student 秒 |
|---|---|---:|---:|
| cora | Bounded X+P | 1.69 | 39.92 |
| cora | CaT-MMD-TAG adapted | 14.58 | 37.62 |
| cora | GECC-hard-TAG adapted | 0.46 | 38.80 |
| cora | k-center native | 0.04 | 37.67 |
| cora | medoid native | 0.16 | 37.02 |
| citeseer | Bounded X+P | 1.73 | 43.31 |
| citeseer | CaT-MMD-TAG adapted | 13.25 | 40.30 |
| citeseer | GECC-hard-TAG adapted | 0.42 | 40.07 |
| citeseer | k-center native | 0.03 | 40.42 |
| citeseer | medoid native | 0.20 | 40.55 |

## 4. 大图成本汇总

| 范围 | 方法 | 平均构造秒 | 平均 Student 秒 |
|---|---|---:|---:|
| arXiv/BookChild/SportsFit base+4 pooled | Bounded X+P | 10.39 | 154.72 |
| arXiv/BookChild/SportsFit base+4 pooled | Bounded X+P+A | 34.22 | 147.53 |
| arXiv/BookChild/SportsFit base+4 pooled | CaT-MMD-TAG adapted | N/A | 150.74 |
| arXiv/BookChild/SportsFit base+4 pooled | GECC-hard-TAG adapted | N/A | 150.99 |
| arXiv/BookChild/SportsFit base+4 pooled | k-center | N/A | 151.62 |
| arXiv/BookChild/SportsFit base+4 pooled | medoid | N/A | 151.10 |

## 5. 论文解释

三顺序结果不支持把 X+P 写成普遍 SOTA：Cora 上它与 GECC 基本持平但低于 CaT、k-center 和 medoid，Citeseer 上高于 GECC/medoid、接近 k-center 但低于 CaT。更准确的表述是：X+P 在固定 continual 访问约束下提供了有 Student 训练反馈的内容与 payload 优化，并在部分数据集/顺序上带来收益，但该收益依赖传播结构和图规模。
成本上，X+P 的额外构造时间明显高于简单 coreset，但小图 order19613 的成本仍是秒级；大图上应把 outer construction 和 Student retraining 分开报告，不能只比较最终 F1。

JSON：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/three_order_supplement_and_cost.json`
