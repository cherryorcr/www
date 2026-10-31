# 按数据集整理的 F1 与 Accuracy 对比表

每个数据集单独列出全部方法，第一行固定为当前主方法。大图使用 `base+4`、5% memory、6 个 Student seeds；Cora/Citeseer 使用独立的三阶段小图协议，不做跨协议 pooled。

## arxiv

主方法固定放第一行。数值为该数据集 base+4 协议、6 个 Student seeds 的 Final Macro-F1 与 Accuracy 均值；其余指标用于观察旧类/新类权衡。

| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive coherent (主方法) | 25.59 +/- 1.73 | 50.70 +/- 1.96 | 28.59 | 8.59 | 28.55 | 25.59 | 136.6 |
| Adaptive coherent + anchor | 22.00 +/- 1.50 | 35.56 +/- 3.82 | 24.47 | 8.03 | 34.03 | 22.00 | 132.9 |
| CaT-MMD-TAG (CCTP top-5) | 24.27 +/- 0.88 | 48.39 +/- 1.48 | 27.54 | 5.76 | 24.44 | 24.27 | 152.2 |
| GECC-hard-TAG | 20.23 +/- 1.09 | 31.43 +/- 4.63 | 22.44 | 7.66 | 30.51 | 20.23 | 187.0 |
| K-center/FPS | 19.43 +/- 1.04 | 30.69 +/- 3.57 | 21.47 | 7.90 | 29.37 | 19.36 | 28.0 |
| Medoid coreset | 18.06 +/- 1.68 | 24.71 +/- 3.23 | 19.91 | 7.53 | 29.68 | 18.06 | 40.2 |
| Random real coreset | 17.81 +/- 1.33 | 24.66 +/- 2.58 | 19.61 | 7.60 | 31.14 | 17.81 | 23.1 |

## sportsfit

主方法固定放第一行。数值为该数据集 base+4 协议、6 个 Student seeds 的 Final Macro-F1 与 Accuracy 均值；其余指标用于观察旧类/新类权衡。

| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive coherent (主方法) | 47.47 +/- 3.06 | 68.14 +/- 4.12 | 51.73 | 24.03 | 58.02 | 47.47 | 234.0 |
| Adaptive coherent + anchor | 36.11 +/- 2.86 | 33.46 +/- 6.35 | 37.50 | 28.46 | 54.59 | 33.11 | 228.8 |
| CaT-MMD-TAG (CCTP top-5) | 23.99 +/- 5.78 | 43.02 +/- 14.67 | 24.30 | 22.27 | 28.88 | 23.99 | 399.7 |
| GECC-hard-TAG | 26.44 +/- 2.18 | 26.95 +/- 4.79 | 27.72 | 19.44 | 46.43 | 26.44 | 679.4 |
| K-center/FPS | 29.42 +/- 3.89 | 32.01 +/- 5.85 | 31.42 | 18.46 | 53.95 | 29.42 | 24.7 |
| Medoid coreset | 27.00 +/- 4.53 | 27.78 +/- 6.30 | 28.27 | 20.01 | 54.33 | 27.00 | 42.8 |
| Random real coreset | 27.44 +/- 7.48 | 30.55 +/- 11.65 | 28.86 | 19.59 | 54.11 | 27.44 | 18.8 |

## wikics

主方法固定放第一行。数值为该数据集 base+4 协议、6 个 Student seeds 的 Final Macro-F1 与 Accuracy 均值；其余指标用于观察旧类/新类权衡。

| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive coherent (主方法) | 73.87 +/- 1.82 | 77.82 +/- 0.93 | 74.86 | 64.94 | 81.67 | 73.87 | 26.4 |
| Adaptive coherent + anchor | 69.76 +/- 1.77 | 74.17 +/- 1.80 | 70.90 | 59.47 | 79.48 | 67.23 | 25.2 |
| CaT-MMD-TAG (CCTP top-5) | 54.30 +/- 4.11 | 58.55 +/- 5.75 | 56.05 | 38.58 | 71.63 | 51.48 | 22.6 |
| GECC-hard-TAG | 51.11 +/- 3.11 | 53.22 +/- 3.45 | 52.40 | 39.51 | 73.21 | 48.59 | 2.2 |
| K-center/FPS | 50.24 +/- 4.39 | 53.19 +/- 4.20 | 51.69 | 37.20 | 70.84 | 43.12 | 0.2 |
| Medoid coreset | 54.97 +/- 3.15 | 59.03 +/- 4.47 | 57.43 | 32.79 | 71.06 | 54.97 | 4.0 |
| Random real coreset | 54.03 +/- 5.30 | 58.22 +/- 5.04 | 55.26 | 42.95 | 67.76 | 54.03 | 0.1 |

## bookhis

主方法固定放第一行。数值为该数据集 base+4 协议、6 个 Student seeds 的 Final Macro-F1 与 Accuracy 均值；其余指标用于观察旧类/新类权衡。

| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage F1 | Build s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive coherent (主方法) | 17.67 +/- 3.99 | 60.18 +/- 2.94 | 19.17 | 1.23 | 13.21 | 17.67 | 55.3 |
| Adaptive coherent + anchor | 16.73 +/- 1.77 | 30.67 +/- 7.57 | 17.52 | 8.07 | 32.56 | 16.73 | 46.1 |
| CaT-MMD-TAG (CCTP top-5) | 17.35 +/- 0.80 | 30.03 +/- 5.57 | 18.30 | 6.91 | 37.02 | 17.35 | 37.3 |
| GECC-hard-TAG | 19.25 +/- 1.24 | 34.48 +/- 5.08 | 20.30 | 7.78 | 41.39 | 19.25 | 59.7 |
| K-center/FPS | 20.04 +/- 1.20 | 38.01 +/- 4.41 | 21.08 | 8.59 | 37.05 | 17.79 | 2.5 |
| Medoid coreset | 17.01 +/- 2.37 | 30.21 +/- 9.78 | 17.93 | 6.87 | 36.86 | 17.01 | 3.2 |
| Random real coreset | 18.68 +/- 1.41 | 33.59 +/- 4.91 | 19.27 | 12.14 | 42.22 | 18.68 | 1.7 |

## cora

小图数值为 `order19611/19612/19613` 三个顺序的等权均值，采用三阶段协议；主线版本固定放第一行。

| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| X-conditioned coherent routing (小图主线) | 84.23 | 85.75 | 84.00 | 84.53 | 89.69 | 84.60 |
| Formal prompt-margin assignment | 83.80 | 85.47 | 83.39 | 84.36 | 89.39 | 84.38 |
| CCTP/CaT nearest payload | 83.74 | 85.49 | 83.35 | 84.27 | 89.15 | 84.13 |

## citeseer

小图数值为 `order19611/19612/19613` 三个顺序的等权均值，采用三阶段协议；主线版本固定放第一行。

| Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| X-conditioned coherent routing (小图主线) | 69.72 | 73.86 | 69.70 | 69.76 | 78.56 | 72.88 |
| Formal prompt-margin assignment | 70.19 | 73.96 | 70.09 | 70.38 | 79.26 | 73.05 |
| CCTP/CaT nearest payload | 69.63 | 73.63 | 69.44 | 70.02 | 78.75 | 72.73 |

## 阅读说明

- 大图主方法为 Adaptive coherent；Cora/Citeseer 的第一行是固定 slot 的 X-conditioned coherent routing，不能把两者名称和协议混写。
- BookHis 的主方法 Accuracy 较高但 New F1 很低，需结合 New F1 和 Macro-F1 判断，不能只按 Accuracy 排名。
- CaT/GECC/coreset 均为当前 continual TAG 访问约束下的机制适配基线，不等同于官方端到端复现。
