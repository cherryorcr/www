# XSP 小图三共同顺序外部比较（2026-10-04）

本报告把 XSP 与已完成外部适配的 baseline 放在三个共同类别顺序 `order19611/19612/19613` 上比较。这里的“三次划分”是三次预注册类别到达顺序；每个顺序内部仍是三阶段 class-incremental stream，不是三次重新访问历史 raw graph 的 offline 重凝聚。

## 1. 冻结协议

- 数据集：Cora、Citeseer。
- 每个顺序：三阶段类别增量，5% cumulative fixed-rate memory。
- 构造器：只接收当前 `delta_t` 和上一阶段 synthetic parent；历史 raw graph、raw text、raw edge 不可回读。
- Student：原两层 GCN、冻结 MiniLM、text projection、双向 InfoNCE；每阶段从头训练；Student seeds `150--155`。
- 评估：完整真实图的 report-dev 节点，最终阶段 Macro-F1、Accuracy、Old F1、New F1。
- CaT、GECC、k-center、medoid 均为保留原机制后的 TAG/Student 适配，不宣称官方 native reproduction。

## 2. 最终阶段逐顺序结果

| Dataset | Order | Method | Final F1 | Accuracy | Old F1 | New F1 |
|---|---|---|---:|---:|---:|---:|
| citeseer | order19611 | CaT-MMD-TAG | 56.93 | 59.87 | 64.28 | 42.25 |
| citeseer | order19611 | GECC-hard-TAG | 48.72 | 50.57 | 52.26 | 41.65 |
| citeseer | order19611 | XSP | 54.34 | 58.15 | 60.90 | 41.24 |
| citeseer | order19611 | XSP + ST-Sinkhorn | 53.66 | 57.68 | 60.22 | 40.52 |
| citeseer | order19611 | k-center/FPS | 51.72 | 57.52 | 60.38 | 34.40 |
| citeseer | order19611 | medoid | 47.33 | 51.88 | 54.77 | 32.47 |
| citeseer | order19612 | CaT-MMD-TAG | 54.93 | 58.88 | 52.50 | 59.78 |
| citeseer | order19612 | GECC-hard-TAG | 52.71 | 55.80 | 52.28 | 53.57 |
| citeseer | order19612 | XSP | 58.02 | 63.58 | 53.56 | 66.95 |
| citeseer | order19612 | XSP + ST-Sinkhorn | 57.02 | 63.01 | 52.46 | 66.15 |
| citeseer | order19612 | k-center/FPS | 59.87 | 66.35 | 56.50 | 66.63 |
| citeseer | order19612 | medoid | 48.96 | 53.55 | 47.61 | 51.65 |
| citeseer | order19613 | CaT-MMD-TAG | 60.09 | 64.11 | 53.76 | 72.75 |
| citeseer | order19613 | GECC-hard-TAG | 48.84 | 53.24 | 43.66 | 59.21 |
| citeseer | order19613 | XSP | 52.82 | 57.84 | 44.89 | 68.67 |
| citeseer | order19613 | XSP + ST-Sinkhorn | 51.69 | 57.16 | 43.71 | 67.65 |
| citeseer | order19613 | k-center/FPS | 52.37 | 59.09 | 44.28 | 68.55 |
| citeseer | order19613 | medoid | 53.73 | 57.31 | 48.97 | 63.24 |
| cora | order19611 | CaT-MMD-TAG | 55.64 | 60.76 | 56.70 | 54.24 |
| cora | order19611 | GECC-hard-TAG | 47.78 | 53.32 | 47.30 | 48.41 |
| cora | order19611 | XSP | 52.23 | 61.13 | 51.31 | 53.46 |
| cora | order19611 | XSP + ST-Sinkhorn | 50.38 | 58.30 | 51.37 | 49.06 |
| cora | order19611 | k-center/FPS | 54.54 | 59.78 | 56.64 | 51.74 |
| cora | order19611 | medoid | 46.39 | 51.78 | 50.89 | 40.39 |
| cora | order19612 | CaT-MMD-TAG | 50.98 | 60.33 | 38.56 | 67.53 |
| cora | order19612 | GECC-hard-TAG | 45.58 | 50.43 | 34.62 | 60.19 |
| cora | order19612 | XSP | 53.91 | 60.82 | 40.98 | 71.15 |
| cora | order19612 | XSP + ST-Sinkhorn | 54.39 | 60.39 | 41.26 | 71.90 |
| cora | order19612 | k-center/FPS | 50.60 | 54.06 | 38.61 | 66.59 |
| cora | order19612 | medoid | 46.58 | 55.23 | 34.67 | 62.47 |
| cora | order19613 | CaT-MMD-TAG | 54.03 | 61.69 | 42.48 | 69.43 |
| cora | order19613 | GECC-hard-TAG | 49.15 | 53.38 | 41.86 | 58.88 |
| cora | order19613 | XSP | 50.41 | 56.89 | 36.88 | 68.46 |
| cora | order19613 | XSP + ST-Sinkhorn | 48.70 | 54.06 | 35.81 | 65.88 |
| cora | order19613 | k-center/FPS | 43.98 | 47.17 | 30.38 | 62.12 |
| cora | order19613 | medoid | 53.67 | 60.70 | 45.70 | 64.29 |

## 3. 按数据集三顺序汇总

| Dataset | Method | N | Final F1 mean ± std | Accuracy | Old F1 | New F1 | Worst-stage F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| citeseer | CaT-MMD-TAG | 3 | 57.32 ± 2.61 | 60.95 | 56.85 | 58.26 | 48.06 |
| citeseer | GECC-hard-TAG | 3 | 50.09 ± 2.27 | 53.20 | 49.40 | 51.47 | 36.70 |
| citeseer | XSP | 3 | 55.06 ± 2.68 | 59.86 | 53.11 | 58.96 | 48.78 |
| citeseer | XSP + ST-Sinkhorn | 3 | 54.12 ± 2.70 | 59.28 | 52.13 | 58.11 | 47.61 |
| citeseer | k-center/FPS | 3 | 54.65 ± 4.53 | 60.99 | 53.72 | 56.53 | 52.84 |
| citeseer | medoid | 3 | 50.01 ± 3.32 | 54.25 | 50.45 | 49.12 | 38.45 |
| cora | CaT-MMD-TAG | 3 | 53.55 ± 2.37 | 60.93 | 45.91 | 63.73 | 39.62 |
| cora | GECC-hard-TAG | 3 | 47.50 ± 1.80 | 52.38 | 41.26 | 55.82 | 42.07 |
| cora | XSP | 3 | 52.18 ± 1.75 | 59.61 | 43.06 | 64.36 | 42.58 |
| cora | XSP + ST-Sinkhorn | 3 | 51.16 ± 2.93 | 57.59 | 42.82 | 62.28 | 44.56 |
| cora | k-center/FPS | 3 | 49.71 ± 5.34 | 53.67 | 41.88 | 60.15 | 30.30 |
| cora | medoid | 3 | 48.88 ± 4.15 | 55.90 | 43.75 | 55.72 | 30.24 |

## 4. 两个数据集六个共同 cell 的总体汇总

| Method | N | Final F1 mean ± std | Accuracy | Old F1 | New F1 | Construction s |
|---|---:|---:|---:|---:|---:|---:|
| CaT-MMD-TAG | 6 | 55.43 ± 3.04 | 60.94 | 51.38 | 61.00 | 4.92 |
| GECC-hard-TAG | 6 | 48.80 ± 2.32 | 52.79 | 45.33 | 53.65 | 0.13 |
| XSP | 6 | 53.62 ± 2.56 | 59.74 | 48.09 | 61.66 | 5.30 |
| XSP + ST-Sinkhorn | 6 | 52.64 ± 2.99 | 58.43 | 47.47 | 60.19 | 5.57 |
| k-center/FPS | 6 | 52.18 ± 5.19 | 57.33 | 47.80 | 58.34 | 0.01 |
| medoid | 6 | 49.44 ± 3.42 | 55.08 | 47.10 | 52.42 | 0.05 |

## 5. XSP 的配对差值

| Candidate | Baseline | Common cells | Mean ΔF1 | Positive cells | Mean ΔAccuracy | Mean ΔOld F1 | Mean ΔNew F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| XSP | CaT-MMD-TAG | 6 | -1.81 | 2/6 | -1.20 | -3.29 | 0.66 |
| XSP | GECC-hard-TAG | 6 | 4.83 | 6/6 | 6.94 | 2.75 | 8.01 |
| XSP | k-center/FPS | 6 | 1.44 | 4/6 | 2.41 | 0.29 | 3.32 |
| XSP | medoid | 6 | 4.18 | 4/6 | 4.66 | 0.98 | 9.24 |
| XSP + ST-Sinkhorn | CaT-MMD-TAG | 6 | -2.79 | 2/6 | -2.51 | -3.91 | -0.80 |
| XSP + ST-Sinkhorn | GECC-hard-TAG | 6 | 3.84 | 5/6 | 5.64 | 2.14 | 6.55 |
| XSP + ST-Sinkhorn | k-center/FPS | 6 | 0.46 | 3/6 | 1.10 | -0.32 | 1.86 |
| XSP + ST-Sinkhorn | medoid | 6 | 3.20 | 4/6 | 3.36 | 0.37 | 7.78 |

## 6. PUMA、TAGSAM 与 STAD 的处理边界

PUMA 当前每个 dataset/order 有多个适配变体，不能把这些行简单平均后作为一个 baseline 排名；它们应在单独的 variant 表中报告。TAGSAM 与 STAD 的现有结果使用 Student seeds `100--105`，且顺序信息不完整，因此保留为 adapted/reference evidence，不混入本三顺序 strict ranking。

## 7. 结论

1. XSP 已经完成与外部机制适配 baseline 的同协议小图比较；它不是只在自己的 runner 中得到结果。
2. XSP 相对 GECC、k-center 和 medoid 有稳定的竞争力，但相对 CaT 仍不是稳定正收益；因此论文中不能写成全面超过 CaT。
3. XSP + ST-Sinkhorn 在这三个共同顺序上没有显示出比普通 XSP 更可靠的收益，暂时作为消融组件，而不是主方法升级。
4. 三顺序表适合做第一版外部比较；正式论文主表仍应补充更多独立顺序或明确将覆盖不足的方法放入 supplementary/reference 层。

逐 cell、汇总和 paired delta 文件位于 `D:\Aprojects\continue_tag_condensation\results\xsp_small_external_three_orders_20261004`。
