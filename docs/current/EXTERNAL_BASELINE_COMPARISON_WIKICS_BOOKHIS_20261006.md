# WikiCS 与 BookHis 外部基线对比

本表只合并通过协议审计的结果：`base+4`、5% cumulative memory、6 个相同 Student seeds、current-delta construction、frozen prefix、不能读取历史 raw graph。CaT/GECC 是机制适配版，不冒充官方端到端复现。

## 1. Dataset-level results

| Dataset | Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wikics | Random real coreset | strict adapted baseline | 54.03 +/- 5.30 | 58.22 +/- 5.04 | 67.76 | 55.26 | 42.95 | 54.03 | 0.1 | 23.0 |
| wikics | K-center/FPS | strict adapted baseline | 50.24 +/- 4.39 | 53.19 +/- 4.20 | 70.84 | 51.69 | 37.20 | 43.12 | 0.2 | 22.9 |
| wikics | Medoid coreset | strict adapted baseline | 54.97 +/- 3.15 | 59.03 +/- 4.47 | 71.06 | 57.43 | 32.79 | 54.97 | 4.0 | 23.1 |
| wikics | GECC-hard-TAG | strict adapted baseline | 51.11 +/- 3.11 | 53.22 +/- 3.45 | 73.21 | 52.40 | 39.51 | 48.59 | 2.2 | 23.2 |
| wikics | CaT-MMD-TAG (CCTP top-5) | strict adapted baseline | 54.30 +/- 4.11 | 58.55 +/- 5.75 | 71.63 | 56.05 | 38.58 | 51.48 | 22.6 | 23.1 |
| wikics | Adaptive coherent | candidate method | 73.87 +/- 1.82 | 77.82 +/- 0.93 | 81.67 | 74.86 | 64.94 | 73.87 | 26.4 | 22.4 |
| wikics | Adaptive coherent + anchor | internal control | 69.76 +/- 1.77 | 74.17 +/- 1.80 | 79.48 | 70.90 | 59.47 | 67.23 | 25.2 | 22.8 |
| bookhis | Random real coreset | strict adapted baseline | 18.68 +/- 1.41 | 33.59 +/- 4.91 | 42.22 | 19.27 | 12.14 | 18.68 | 1.7 | 22.5 |
| bookhis | K-center/FPS | strict adapted baseline | 20.04 +/- 1.20 | 38.01 +/- 4.41 | 37.05 | 21.08 | 8.59 | 17.79 | 2.5 | 22.5 |
| bookhis | Medoid coreset | strict adapted baseline | 17.01 +/- 2.37 | 30.21 +/- 9.78 | 36.86 | 17.93 | 6.87 | 17.01 | 3.2 | 22.5 |
| bookhis | GECC-hard-TAG | strict adapted baseline | 19.25 +/- 1.24 | 34.48 +/- 5.08 | 41.39 | 20.30 | 7.78 | 19.25 | 59.7 | 22.3 |
| bookhis | CaT-MMD-TAG (CCTP top-5) | strict adapted baseline | 17.35 +/- 0.80 | 30.03 +/- 5.57 | 37.02 | 18.30 | 6.91 | 17.35 | 37.3 | 22.4 |
| bookhis | Adaptive coherent | candidate method | 17.67 +/- 3.99 | 60.18 +/- 2.94 | 13.21 | 19.17 | 1.23 | 17.67 | 55.3 | 25.6 |
| bookhis | Adaptive coherent + anchor | internal control | 16.73 +/- 1.77 | 30.67 +/- 7.57 | 32.56 | 17.52 | 8.07 | 16.73 | 46.1 | 25.6 |

## 2. Equal-weight pooled results

| Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random real coreset | strict adapted baseline | 36.35 | 45.90 | 54.99 | 37.27 | 27.54 | 36.35 | 0.9 | 22.8 |
| K-center/FPS | strict adapted baseline | 35.14 | 45.60 | 53.95 | 36.38 | 22.89 | 30.45 | 1.3 | 22.7 |
| Medoid coreset | strict adapted baseline | 35.99 | 44.62 | 53.96 | 37.68 | 19.83 | 35.99 | 3.6 | 22.8 |
| GECC-hard-TAG | strict adapted baseline | 35.18 | 43.85 | 57.30 | 36.35 | 23.64 | 33.92 | 30.9 | 22.8 |
| CaT-MMD-TAG (CCTP top-5) | strict adapted baseline | 35.83 | 44.29 | 54.33 | 37.17 | 22.75 | 34.42 | 30.0 | 22.8 |
| Adaptive coherent | candidate method | 45.77 | 69.00 | 47.44 | 47.01 | 33.09 | 45.77 | 40.9 | 24.0 |
| Adaptive coherent + anchor | internal control | 43.24 | 52.42 | 56.02 | 44.21 | 33.77 | 41.98 | 35.6 | 24.2 |

## 3. Adaptive coherent 相对外部基线

| Baseline | Mean Final-F1 delta | WikiCS | BookHis | Positive datasets |
|---|---:|---:|---:|---:|
| Random real coreset | +9.42 | +19.84 | -1.01 | 1/2 |
| K-center/FPS | +10.63 | +23.63 | -2.36 | 1/2 |
| Medoid coreset | +9.78 | +18.90 | +0.67 | 2/2 |
| GECC-hard-TAG | +10.59 | +22.76 | -1.58 | 1/2 |
| CaT-MMD-TAG (CCTP top-5) | +9.94 | +19.57 | +0.32 | 2/2 |

## 4. Coverage and paper-use notes

- PUMA-inspired baseline 当前已有 Cora/Citeseer 的严格适配结果，但 runner 只覆盖这两张小图，因此不虚构 WikiCS/BookHis 数值。
- TAGSAM、GCond、SFGC、GCondenser、STAD 的可用实现与当前 continual TAG Student/访问约束不完全一致，单独列为 adapted/reference-only，不放进这张严格主表。
- 这张表回答的是：在同一协议下，Adaptive coherent 是否优于合理的外部机制适配基线；不是宣称官方论文方法被原样复现。

## 5. 结果解释

- WikiCS：Adaptive coherent 的 Final Macro-F1 为 73.87，超过本表最强外部基线 Medoid 约 18.90 pp、CaT 约 19.57 pp；anchor 版本反而低 4.11 pp，说明收益来自无 anchor 的 routing 行为，而不是简单的 CCTP 约束。
- BookHis：Adaptive coherent 的 Final Macro-F1 为 17.67，仅比本轮 CaT 高 0.32 pp，并低于 K-center 2.36 pp；Accuracy 虽为 60.18，但 New F1 只有 1.23、stage-average F1 只有 13.21，说明长尾新类上存在明显的预测塌缩或类别偏置，不能据此宣称跨数据集稳定 SOTA。
- 因此 pooled 平均值 45.77 不能替代 dataset-level 结论。下一步应优先审计 BookHis 的新类 payload/预测分布，再决定是否把 Adaptive coherent 冻结为主方法。

## 6. Protocol audit

| Dataset | Stage protocol | Memory rate | Student seeds | Historical raw reads | Test labels for construction |
|---|---|---:|---|---:|---:|
| wikics | base+4 | 5.00% | [100, 101, 102, 103, 104, 105] | 0 | False |
| bookhis | base+4 | 5.00% | [100, 101, 102, 103, 104, 105] | 0 | False |
