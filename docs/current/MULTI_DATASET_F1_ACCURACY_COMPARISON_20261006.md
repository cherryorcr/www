# 四个大图上的 F1 与 Accuracy 跨数据集比较

四个数据集均使用 `base+4`、5% cumulative memory、6 个 Student seeds、同一 edge-aware 2-layer GCN + text projection + bidirectional InfoNCE Student。每个数据集的类别顺序和预算按其冻结 manifest 执行，因此跨数据集比较用于观察泛化，不把不同数据集的绝对难度混为同一任务。

## 1. Dataset-level comparison

| Dataset | Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv | Random real coreset | strict adapted baseline | 17.81 +/- 1.33 | 24.66 +/- 2.58 | 31.14 | 19.61 | 7.60 | 17.81 | 23.1 | 24.3 |
| arxiv | K-center/FPS | strict adapted baseline | 19.43 +/- 1.04 | 30.69 +/- 3.57 | 29.37 | 21.47 | 7.90 | 19.36 | 28.0 | 24.3 |
| arxiv | Medoid coreset | strict adapted baseline | 18.06 +/- 1.68 | 24.71 +/- 3.23 | 29.68 | 19.91 | 7.53 | 18.06 | 40.2 | 24.3 |
| arxiv | GECC-hard-TAG | strict adapted baseline | 20.23 +/- 1.09 | 31.43 +/- 4.63 | 30.51 | 22.44 | 7.66 | 20.23 | 187.0 | 24.4 |
| arxiv | CaT-MMD-TAG (CCTP top-5) | strict adapted baseline | 24.27 +/- 0.88 | 48.39 +/- 1.48 | 24.44 | 27.54 | 5.76 | 24.27 | 152.2 | 24.5 |
| arxiv | Adaptive coherent | candidate method | 25.59 +/- 1.73 | 50.70 +/- 1.96 | 28.55 | 28.59 | 8.59 | 25.59 | 136.6 | 26.9 |
| arxiv | Adaptive coherent + anchor | internal control | 22.00 +/- 1.50 | 35.56 +/- 3.82 | 34.03 | 24.47 | 8.03 | 22.00 | 132.9 | 26.5 |
| sportsfit | Random real coreset | strict adapted baseline | 27.44 +/- 7.48 | 30.55 +/- 11.65 | 54.11 | 28.86 | 19.59 | 27.44 | 18.8 | 26.0 |
| sportsfit | K-center/FPS | strict adapted baseline | 29.42 +/- 3.89 | 32.01 +/- 5.85 | 53.95 | 31.42 | 18.46 | 29.42 | 24.7 | 26.2 |
| sportsfit | Medoid coreset | strict adapted baseline | 27.00 +/- 4.53 | 27.78 +/- 6.30 | 54.33 | 28.27 | 20.01 | 27.00 | 42.8 | 26.1 |
| sportsfit | GECC-hard-TAG | strict adapted baseline | 26.44 +/- 2.18 | 26.95 +/- 4.79 | 46.43 | 27.72 | 19.44 | 26.44 | 679.4 | 25.6 |
| sportsfit | CaT-MMD-TAG (CCTP top-5) | strict adapted baseline | 23.99 +/- 5.78 | 43.02 +/- 14.67 | 28.88 | 24.30 | 22.27 | 23.99 | 399.7 | 25.8 |
| sportsfit | Adaptive coherent | candidate method | 47.47 +/- 3.06 | 68.14 +/- 4.12 | 58.02 | 51.73 | 24.03 | 47.47 | 234.0 | 24.8 |
| sportsfit | Adaptive coherent + anchor | internal control | 36.11 +/- 2.86 | 33.46 +/- 6.35 | 54.59 | 37.50 | 28.46 | 33.11 | 228.8 | 25.0 |
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

## 2. Equal-weight pooled comparison

| Method | Category | Final Macro-F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | Worst stage | Build s | Student s/seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random real coreset | strict adapted baseline | 29.49 | 36.75 | 48.81 | 30.75 | 20.57 | 29.49 | 10.9 | 23.9 |
| K-center/FPS | strict adapted baseline | 29.78 | 38.47 | 47.80 | 31.41 | 18.04 | 27.42 | 13.9 | 24.0 |
| Medoid coreset | strict adapted baseline | 29.26 | 35.43 | 47.98 | 30.89 | 16.80 | 29.26 | 22.5 | 24.0 |
| GECC-hard-TAG | strict adapted baseline | 29.26 | 36.52 | 47.88 | 30.71 | 18.59 | 28.63 | 232.0 | 23.9 |
| CaT-MMD-TAG (CCTP top-5) | strict adapted baseline | 29.98 | 45.00 | 40.49 | 31.55 | 18.38 | 29.27 | 152.9 | 24.0 |
| Adaptive coherent | candidate method | 41.15 | 64.21 | 45.36 | 43.59 | 24.70 | 41.15 | 113.1 | 24.9 |
| Adaptive coherent + anchor | internal control | 36.15 | 43.46 | 50.16 | 37.60 | 26.01 | 34.77 | 108.3 | 25.0 |

## 3. Adaptive coherent 相对最强外部基线

| Dataset | Adaptive F1 | Best external F1 method | Best external F1 | Delta F1 | Adaptive Accuracy | Best external Accuracy method | Best external Accuracy | Delta Accuracy |
|---|---:|---|---:|---:|---:|---|---:|---:|
| arxiv | 25.59 | CaT-MMD-TAG (CCTP top-5) | 24.27 | +1.32 | 50.70 | CaT-MMD-TAG (CCTP top-5) | 48.39 | +2.30 |
| sportsfit | 47.47 | K-center/FPS | 29.42 | +18.04 | 68.14 | CaT-MMD-TAG (CCTP top-5) | 43.02 | +25.13 |
| wikics | 73.87 | Medoid coreset | 54.97 | +18.90 | 77.82 | Medoid coreset | 59.03 | +18.79 |
| bookhis | 17.67 | K-center/FPS | 20.04 | -2.36 | 60.18 | K-center/FPS | 38.01 | +22.17 |

## 4. Interpretation

- WikiCS、Arxiv、SportsFit 上 Adaptive coherent 的 Final Macro-F1 明显高于本轮外部基线，且 Accuracy 同步提升。
- BookHis 上 Adaptive coherent 的 F1 仍低于 K-center，虽然 Accuracy 很高，但 New F1 很低，说明长尾新类存在类别偏置；不能只看 pooled 平均值。
- Anchor 版本作为内部控制单独报告，不把它当作外部基线；它用于判断 payload anchor 是否真正改善泛化。
- Cora/Citeseer 使用三阶段小图协议，另行报告，不与这四个大图的 base+4 数值做 pooled 混合；详见 `SMALL_DATASET_F1_ACCURACY_COMPARISON_20261006.md`。

## 5. Protocol audit

| Dataset | Stage protocol | Memory rate | Student seeds | Historical raw reads | Test labels used for construction |
|---|---|---:|---|---:|---:|
| arxiv | base+4 | 5.00% | [100, 101, 102, 103, 104, 105] | 0 | False |
| sportsfit | base+4 | 5.00% | [100, 101, 102, 103, 104, 105] | 0 | False |
| wikics | base+4 | 5.00% | [100, 101, 102, 103, 104, 105] | 0 | False |
| bookhis | base+4 | 5.00% | [100, 101, 102, 103, 104, 105] | 0 | False |
