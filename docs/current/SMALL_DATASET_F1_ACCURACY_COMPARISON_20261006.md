# Cora/Citeseer 小图多顺序 F1 与 Accuracy 比较

本表单独采用小图三阶段协议：5% memory、`order19611/19612/19613`、6 个 Student seeds、current-delta-only、edge-free memory + Student self-loop。它不与大图 `base+4` 结果做 pooled 混合。

## 1. Per-order final results

| Dataset | Order | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cora | order19611 | CCTP/CaT nearest payload | 83.51 +/- 0.33 | 85.30 +/- 0.49 | 85.57 | 80.76 | 87.78 | 82.69 |
| cora | order19611 | Formal prompt-margin assignment | 83.88 +/- 1.22 | 85.55 +/- 1.03 | 86.18 | 80.83 | 88.26 | 83.51 |
| cora | order19611 | X-conditioned coherent routing | 84.08 +/- 0.93 | 85.85 +/- 0.76 | 86.59 | 80.75 | 88.43 | 83.82 |
| cora | order19612 | CCTP/CaT nearest payload | 83.24 +/- 1.28 | 85.24 +/- 1.04 | 81.80 | 85.16 | 88.58 | 84.90 |
| cora | order19612 | Formal prompt-margin assignment | 83.30 +/- 1.10 | 85.24 +/- 0.84 | 81.93 | 85.12 | 88.73 | 84.86 |
| cora | order19612 | X-conditioned coherent routing | 84.37 +/- 1.82 | 85.92 +/- 1.39 | 83.33 | 85.76 | 89.61 | 85.54 |
| cora | order19613 | CCTP/CaT nearest payload | 84.49 +/- 0.90 | 85.92 +/- 0.85 | 82.69 | 86.88 | 91.10 | 84.80 |
| cora | order19613 | Formal prompt-margin assignment | 84.23 +/- 1.00 | 85.61 +/- 0.57 | 82.05 | 87.15 | 91.18 | 84.77 |
| cora | order19613 | X-conditioned coherent routing | 84.24 +/- 1.19 | 85.49 +/- 1.27 | 82.10 | 87.09 | 91.04 | 84.43 |
| citeseer | order19611 | CCTP/CaT nearest payload | 70.09 +/- 1.13 | 74.24 +/- 0.70 | 76.86 | 56.54 | 78.17 | 73.81 |
| citeseer | order19611 | Formal prompt-margin assignment | 70.64 +/- 0.89 | 74.71 +/- 0.83 | 77.32 | 57.28 | 78.85 | 74.26 |
| citeseer | order19611 | X-conditioned coherent routing | 69.83 +/- 1.00 | 74.56 +/- 1.06 | 77.34 | 54.81 | 77.66 | 74.03 |
| citeseer | order19612 | CCTP/CaT nearest payload | 68.82 +/- 1.44 | 72.31 +/- 1.33 | 67.04 | 72.37 | 79.40 | 71.79 |
| citeseer | order19612 | Formal prompt-margin assignment | 69.50 +/- 1.54 | 72.52 +/- 1.38 | 67.77 | 72.95 | 79.77 | 72.02 |
| citeseer | order19612 | X-conditioned coherent routing | 69.52 +/- 1.18 | 72.78 +/- 1.20 | 67.61 | 73.35 | 79.67 | 72.31 |
| citeseer | order19613 | CCTP/CaT nearest payload | 70.00 +/- 1.67 | 74.35 +/- 0.98 | 64.42 | 81.16 | 78.68 | 72.59 |
| citeseer | order19613 | Formal prompt-margin assignment | 70.42 +/- 1.35 | 74.66 +/- 0.94 | 65.17 | 80.92 | 79.15 | 72.86 |
| citeseer | order19613 | X-conditioned coherent routing | 69.80 +/- 1.76 | 74.24 +/- 1.23 | 64.15 | 81.12 | 78.35 | 72.29 |

## 2. Equal-weight mean over the three orders

| Dataset | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| cora | CCTP/CaT nearest payload | 83.74 | 85.49 | 83.35 | 84.27 | 89.15 | 84.13 |
| cora | Formal prompt-margin assignment | 83.80 | 85.47 | 83.39 | 84.36 | 89.39 | 84.38 |
| cora | X-conditioned coherent routing | 84.23 | 85.75 | 84.00 | 84.53 | 89.69 | 84.60 |
| citeseer | CCTP/CaT nearest payload | 69.63 | 73.63 | 69.44 | 70.02 | 78.75 | 72.73 |
| citeseer | Formal prompt-margin assignment | 70.19 | 73.96 | 70.09 | 70.38 | 79.26 | 73.05 |
| citeseer | X-conditioned coherent routing | 69.72 | 73.86 | 69.70 | 69.76 | 78.56 | 72.88 |

## 3. Interpretation

- 这张表用于确认小图上的顺序稳定性和 F1/Accuracy 是否一致，不把单个 order 的最好值当作结论。
- `X-conditioned coherent routing` 是小图固定 slot 版本；大图主表中的 `Adaptive coherent` 使用自适应 slot，名称和协议应区分。
- Cora/Citeseer 的公开 CaT、GECC、coreset 适配结果另有 baseline card 和 external adapter 结果；本表先聚焦当前 routing 线的三种构造对照。
