# XSP 小图严格外部基线比较（2026-10-04）

本轮把 XSP 和 XSP + ST-Sinkhorn 放到外部 baseline 的小图协议中运行。协议固定为 Cora/Citeseer、三阶段、5% cumulative memory、current delta + synthetic parent、历史 raw 不可回读、Student seeds 150--155。最终评估在完整真实图的 report-dev 节点上进行 prompt-based node classification。

## 1. 最终 stage 的逐顺序比较

| Dataset | Order | Method | Final F1 | Accuracy | Old F1 | New F1 | Memory rows | Construction s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| citeseer | order19611 | CaT-MMD-TAG | 56.93 | 59.87 | 64.28 | 42.25 | 111.00 | 4.75 |
| citeseer | order19611 | GECC-hard-TAG | 48.72 | 50.57 | 52.26 | 41.65 | 111.00 | 0.11 |
| citeseer | order19611 | PUMA-inspired | 16.90 | 20.22 | 19.79 | 11.13 | 111.00 | NA |
| citeseer | order19611 | PUMA-inspired | 12.49 | 14.52 | 13.04 | 11.40 | 111.00 | NA |
| citeseer | order19611 | PUMA-inspired | 21.68 | 25.34 | 27.72 | 9.59 | 111.00 | NA |
| citeseer | order19611 | XSP | 54.34 | 58.15 | 60.90 | 41.24 | 111.00 | 4.69 |
| citeseer | order19611 | XSP + ST-Sinkhorn | 53.66 | 57.68 | 60.22 | 40.52 | 111.00 | 4.93 |
| citeseer | order19611 | k-center/FPS | 51.72 | 57.52 | 60.38 | 34.40 | 111.00 | 0.01 |
| citeseer | order19611 | medoid | 47.33 | 51.88 | 54.77 | 32.47 | 111.00 | 0.04 |
| citeseer | order19612 | CaT-MMD-TAG | 54.93 | 58.88 | 52.50 | 59.78 | 111.00 | 4.54 |
| citeseer | order19612 | GECC-hard-TAG | 52.71 | 55.80 | 52.28 | 53.57 | 111.00 | 0.13 |
| citeseer | order19612 | PUMA-inspired | 12.53 | 14.89 | 11.56 | 14.45 | 111.00 | NA |
| citeseer | order19612 | PUMA-inspired | 21.63 | 25.34 | 24.48 | 15.93 | 111.00 | NA |
| citeseer | order19612 | PUMA-inspired | 17.08 | 19.02 | 16.31 | 18.62 | 111.00 | NA |
| citeseer | order19612 | XSP | 58.02 | 63.58 | 53.56 | 66.95 | 111.00 | 5.14 |
| citeseer | order19612 | XSP + ST-Sinkhorn | 57.02 | 63.01 | 52.46 | 66.15 | 111.00 | 5.34 |
| citeseer | order19612 | k-center/FPS | 59.87 | 66.35 | 56.50 | 66.63 | 111.00 | 0.01 |
| citeseer | order19612 | medoid | 48.96 | 53.55 | 47.61 | 51.65 | 111.00 | 0.05 |
| citeseer | order19613 | CaT-MMD-TAG | 60.09 | 64.11 | 53.76 | 72.75 | 111.00 | 4.09 |
| citeseer | order19613 | GECC-hard-TAG | 48.84 | 53.24 | 43.66 | 59.21 | 111.00 | 0.14 |
| citeseer | order19613 | XSP | 52.82 | 57.84 | 44.89 | 68.67 | 111.00 | 5.20 |
| citeseer | order19613 | XSP + ST-Sinkhorn | 51.69 | 57.16 | 43.71 | 67.65 | 111.00 | 4.93 |
| citeseer | order19613 | k-center/FPS | 52.37 | 59.09 | 44.28 | 68.55 | 111.00 | 0.01 |
| citeseer | order19613 | medoid | 53.73 | 57.31 | 48.97 | 63.24 | 111.00 | 0.08 |
| citeseer | order19614 | CaT-MMD-TAG | 56.20 | 59.40 | 49.24 | 70.11 | 111.00 | 4.22 |
| citeseer | order19614 | XSP | 54.98 | 58.41 | 47.19 | 70.56 | 111.00 | 4.66 |
| citeseer | order19614 | XSP + ST-Sinkhorn | 55.61 | 58.05 | 45.87 | 75.11 | 111.00 | 5.34 |
| citeseer | order19614 | k-center/FPS | 51.19 | 58.10 | 42.65 | 68.26 | 111.00 | 0.01 |
| citeseer | order19615 | CaT-MMD-TAG | 57.10 | 60.71 | 63.74 | 43.81 | 111.00 | 4.67 |
| citeseer | order19615 | XSP | 54.46 | 58.46 | 61.39 | 40.60 | 111.00 | 4.47 |
| citeseer | order19615 | XSP + ST-Sinkhorn | 53.25 | 57.47 | 59.88 | 39.98 | 111.00 | 4.90 |
| citeseer | order19615 | k-center/FPS | 53.10 | 58.78 | 62.23 | 34.86 | 111.00 | 0.01 |
| cora | order19611 | CaT-MMD-TAG | 55.64 | 60.76 | 56.70 | 54.24 | 94.00 | 5.45 |
| cora | order19611 | GECC-hard-TAG | 47.78 | 53.32 | 47.30 | 48.41 | 94.00 | 0.13 |
| cora | order19611 | PUMA-inspired | 17.41 | 17.40 | 17.11 | 17.80 | 94.00 | NA |
| cora | order19611 | PUMA-inspired | 10.06 | 11.01 | 6.50 | 14.80 | 94.00 | NA |
| cora | order19611 | PUMA-inspired | 9.10 | 9.10 | 8.08 | 10.45 | 94.00 | NA |
| cora | order19611 | XSP | 52.23 | 61.13 | 51.31 | 53.46 | 94.00 | 5.59 |
| cora | order19611 | XSP + ST-Sinkhorn | 50.38 | 58.30 | 51.37 | 49.06 | 94.00 | 5.94 |
| cora | order19611 | k-center/FPS | 54.54 | 59.78 | 56.64 | 51.74 | 94.00 | 0.00 |
| cora | order19611 | medoid | 46.39 | 51.78 | 50.89 | 40.39 | 94.00 | 0.04 |
| cora | order19612 | CaT-MMD-TAG | 50.98 | 60.33 | 38.56 | 67.53 | 94.00 | 5.52 |
| cora | order19612 | GECC-hard-TAG | 45.58 | 50.43 | 34.62 | 60.19 | 94.00 | 0.13 |
| cora | order19612 | PUMA-inspired | 9.54 | 10.02 | 12.08 | 6.15 | 94.00 | NA |
| cora | order19612 | PUMA-inspired | 13.89 | 15.07 | 18.32 | 7.98 | 94.00 | NA |
| cora | order19612 | PUMA-inspired | 17.86 | 18.70 | 9.75 | 28.67 | 94.00 | NA |
| cora | order19612 | XSP | 53.91 | 60.82 | 40.98 | 71.15 | 94.00 | 5.69 |
| cora | order19612 | XSP + ST-Sinkhorn | 54.39 | 60.39 | 41.26 | 71.90 | 94.00 | 6.27 |
| cora | order19612 | k-center/FPS | 50.60 | 54.06 | 38.61 | 66.59 | 94.00 | 0.00 |
| cora | order19612 | medoid | 46.58 | 55.23 | 34.67 | 62.47 | 94.00 | 0.04 |
| cora | order19613 | CaT-MMD-TAG | 54.03 | 61.69 | 42.48 | 69.43 | 94.00 | 5.20 |
| cora | order19613 | GECC-hard-TAG | 49.15 | 53.38 | 41.86 | 58.88 | 94.00 | 0.16 |
| cora | order19613 | XSP | 50.41 | 56.89 | 36.88 | 68.46 | 94.00 | 5.49 |
| cora | order19613 | XSP + ST-Sinkhorn | 48.70 | 54.06 | 35.81 | 65.88 | 94.00 | 6.03 |
| cora | order19613 | k-center/FPS | 43.98 | 47.17 | 30.38 | 62.12 | 94.00 | 0.01 |
| cora | order19613 | medoid | 53.67 | 60.70 | 45.70 | 64.29 | 94.00 | 0.07 |
| cora | order19614 | CaT-MMD-TAG | 54.64 | 61.32 | 59.18 | 48.59 | 94.00 | 5.12 |
| cora | order19614 | GECC-hard-TAG | 46.80 | 52.03 | 45.88 | 48.02 | 94.00 | 0.14 |
| cora | order19614 | XSP | 56.29 | 63.53 | 56.96 | 55.39 | 94.00 | 5.57 |
| cora | order19614 | XSP + ST-Sinkhorn | 56.94 | 65.13 | 55.99 | 58.20 | 94.00 | 6.32 |
| cora | order19614 | k-center/FPS | 53.32 | 61.38 | 59.31 | 45.33 | 94.00 | 0.01 |
| cora | order19614 | medoid | 51.06 | 57.07 | 52.46 | 49.19 | 94.00 | 0.05 |
| cora | order19615 | CaT-MMD-TAG | 52.98 | 62.30 | 51.88 | 54.45 | 94.00 | 5.53 |
| cora | order19615 | XSP | 57.05 | 65.38 | 60.87 | 51.96 | 94.00 | 5.98 |
| cora | order19615 | XSP + ST-Sinkhorn | 54.65 | 62.92 | 58.00 | 50.19 | 94.00 | 6.46 |
| cora | order19615 | k-center/FPS | 41.90 | 43.91 | 42.35 | 41.30 | 94.00 | 0.01 |

## 2. 按数据集和可用顺序汇总

| Dataset | Method | Orders | N | Final F1 mean ± std | Accuracy | Old F1 | New F1 | Worst-stage F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| citeseer | CaT-MMD-TAG | order19611,order19612,order19613,order19614,order19615 | 5 | 57.05 ± 1.91 | 60.60 | 56.70 | 57.74 | 49.42 |
| cora | CaT-MMD-TAG | order19611,order19612,order19613,order19614,order19615 | 5 | 53.65 ± 1.78 | 61.28 | 49.76 | 58.85 | 40.41 |
| citeseer | GECC-hard-TAG | order19611,order19612,order19613 | 3 | 50.09 ± 2.27 | 53.20 | 49.40 | 51.47 | 36.70 |
| cora | GECC-hard-TAG | order19611,order19612,order19613,order19614 | 4 | 47.33 ± 1.51 | 52.29 | 42.42 | 53.87 | 43.25 |
| citeseer | PUMA-inspired | order19611,order19611,order19611,order19612,order19612,order19612 | 6 | 17.05 ± 4.09 | 19.89 | 18.82 | 13.52 | 12.51 |
| cora | PUMA-inspired | order19611,order19611,order19611,order19612,order19612,order19612 | 6 | 12.97 ± 3.99 | 13.55 | 11.97 | 14.31 | 8.16 |
| citeseer | XSP | order19611,order19612,order19613,order19614,order19615 | 5 | 54.92 ± 1.91 | 59.29 | 53.58 | 57.61 | 48.55 |
| cora | XSP | order19611,order19612,order19613,order19614,order19615 | 5 | 53.98 ± 2.76 | 61.55 | 49.40 | 60.08 | 45.62 |
| citeseer | XSP + ST-Sinkhorn | order19611,order19612,order19613,order19614,order19615 | 5 | 54.25 ± 2.09 | 58.67 | 52.43 | 57.88 | 47.49 |
| cora | XSP + ST-Sinkhorn | order19611,order19612,order19613,order19614,order19615 | 5 | 53.01 ± 3.37 | 60.16 | 48.49 | 59.05 | 44.96 |
| citeseer | k-center/FPS | order19611,order19612,order19613,order19614,order19615 | 5 | 53.65 ± 3.55 | 59.97 | 53.21 | 54.54 | 52.56 |
| cora | k-center/FPS | order19611,order19612,order19613,order19614,order19615 | 5 | 48.87 ± 5.64 | 53.26 | 45.46 | 53.42 | 37.23 |
| citeseer | medoid | order19611,order19612,order19613 | 3 | 50.01 ± 3.32 | 54.25 | 50.45 | 49.12 | 38.45 |
| cora | medoid | order19611,order19612,order19613,order19614 | 4 | 49.43 ± 3.56 | 56.20 | 45.93 | 54.09 | 30.78 |

## 3. XSP 相对外部 baseline 的配对差值

| Candidate | Baseline | Common cells | Mean ΔF1 | Positive cells | Mean ΔAccuracy | Mean ΔOld F1 | Mean ΔNew F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| XSP | CaT-MMD-TAG | 10 | -0.90 | 4/10 | -0.52 | -1.74 | 0.55 |
| XSP | GECC-hard-TAG | 7 | 5.49 | 7/7 | 7.60 | 3.94 | 7.92 |
| XSP | k-center/FPS | 10 | 3.19 | 8/10 | 3.81 | 2.16 | 4.87 |
| XSP | medoid | 7 | 4.33 | 5/7 | 4.92 | 1.48 | 8.80 |
| XSP | PUMA-inspired | 4 | 38.20 | 4/4 | 42.88 | 36.22 | 41.37 |
| XSP + ST-Sinkhorn | CaT-MMD-TAG | 10 | -1.72 | 4/10 | -1.52 | -2.77 | 0.17 |
| XSP + ST-Sinkhorn | GECC-hard-TAG | 7 | 4.74 | 6/7 | 6.71 | 3.28 | 7.06 |
| XSP + ST-Sinkhorn | k-center/FPS | 10 | 2.37 | 7/10 | 2.80 | 1.12 | 4.49 |
| XSP + ST-Sinkhorn | medoid | 7 | 3.58 | 5/7 | 4.03 | 0.82 | 7.95 |
| XSP + ST-Sinkhorn | PUMA-inspired | 4 | 37.44 | 4/4 | 41.81 | 35.86 | 40.08 |

## 4. 协议与解释边界

- XSP 和 XSP + ST 使用同一 CaT X 初始化、同一文本 payload 接口、同一 edge-free synthetic memory 约束；ST 只改变 Sinkhorn 反向传播路径。
- CaT、GECC、PUMA 是 mechanism-adapted；k-center/medoid 是同协议 real-node coreset，不是 synthetic condensation；都不是各自论文的 native end-to-end 数字。
- baseline 覆盖并不完全相同：CaT/k-center 覆盖五个顺序，GECC/medoid/SR 覆盖较少顺序，PUMA 只覆盖已有两个顺序。因此主判断优先使用 common-cell paired delta，不用不平衡的全覆盖均值制造排名。
- Semantic-Recursive 不放入当前主方法排名；它是此前启发式递归版本，保留在外部包中作为历史补充。

## 5. 当前结论

本轮首先回答 XSP 是否能在小图严格外部协议下运行并与公开方法适配比较。XSP + ST-Sinkhorn 是否升为主方法，应由其相对 CaT 和 k-center 的五顺序 paired delta、最差 cell 和新类 F1 一起决定；不能只依据此前大图 BookChild 的单点改善。

原始逐 cell 结果和审计文件位于 `D:\Aprojects\continue_tag_condensation\results\xsp_small_external_strict_20261004_summary`。
