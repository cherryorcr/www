# Bounded X+P：order19614 独立顺序基线比较

> `order19614` 是本轮新增的独立类别顺序。所有方法使用同一三阶段、5% cumulative memory、原始 Student、6 个 report seeds 和同一 report-dev 规则。

## 1. 结果总表

| 数据集 | 方法 | Final Macro-F1 | Final Accuracy | 阶段平均 F1 | 阶段平均 Accuracy | Old F1 | New F1 | 最差阶段 F1 | rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | Bounded X+P | 45.73 ± 2.11 | 53.81 ± 1.67 | 45.31 | 42.43 | 47.07 | 43.95 | 41.43 | 94 |
| cora | C0 append-edge anchor | 46.00 ± 3.30 | 54.74 ± 2.82 | 46.01 | 43.37 | 48.66 | 42.45 | 43.23 | 94 |
| cora | k-center/FPS | 52.06 ± 1.26 | 61.38 ± 2.59 | 49.86 | 47.31 | 53.42 | 50.24 | 48.74 | 94 |
| cora | medoid | 55.03 ± 2.32 | 61.01 ± 3.36 | 53.84 | 50.14 | 52.78 | 58.03 | 48.78 | 94 |
| cora | CaT-MMD-TAG adapted | 54.64 ± 3.00 | 61.32 ± 2.65 | 52.11 | 46.32 | 59.18 | 48.59 | 45.61 | 94 |
| cora | GECC-hard-TAG adapted | 46.80 ± 1.86 | 52.03 ± 1.31 | 48.05 | 42.43 | 45.88 | 48.02 | 46.80 | 94 |
| cora | k-center native append | 53.32 ± 1.78 | 61.38 ± 1.56 | 63.97 | 61.81 | 59.31 | 45.33 | 53.32 | 94 |
| cora | medoid native append | 51.06 ± 5.32 | 57.07 ± 4.70 | 46.48 | 42.91 | 52.46 | 49.19 | 32.38 | 94 |
| citeseer | Bounded X+P | 47.84 ± 3.49 | 53.29 ± 1.89 | 42.45 | 41.49 | 48.14 | 47.26 | 34.46 | 111 |
| citeseer | C0 append-edge anchor | 51.35 ± 2.29 | 56.64 ± 2.36 | 43.80 | 42.01 | 48.89 | 56.27 | 34.46 | 111 |
| citeseer | k-center/FPS | 51.07 ± 2.16 | 55.54 ± 1.96 | 43.27 | 41.59 | 42.62 | 67.96 | 34.46 | 111 |
| citeseer | medoid | 51.67 ± 5.09 | 59.09 ± 4.70 | 43.86 | 43.81 | 44.46 | 66.08 | 34.46 | 111 |
| citeseer | CaT-MMD-TAG adapted | 55.06 ± 2.00 | 58.93 ± 2.32 | 52.85 | 48.46 | 47.97 | 69.23 | 46.88 | 111 |
| citeseer | GECC-hard-TAG adapted | 48.86 ± 2.25 | 51.62 ± 2.37 | 42.90 | 38.70 | 44.77 | 57.02 | 39.29 | 111 |
| citeseer | k-center native append | 51.19 ± 2.96 | 58.10 ± 2.41 | 56.56 | 57.45 | 42.65 | 68.26 | 51.19 | 111 |
| citeseer | medoid native append | 48.08 ± 2.40 | 53.97 ± 2.46 | 50.87 | 50.10 | 43.54 | 57.14 | 45.95 | 111 |

## 2. X+P 相对差值

> 正值表示 X+P 更高；这里是同一类别顺序内的描述性配对差值，不代表跨数据集显著性检验。

| 数据集 | 对比方法 | X+P - Final F1 | X+P - Accuracy | X+P - 阶段平均 F1 |
|---|---|---:|---:|---:|
| cora | C0 append-edge anchor | -0.27 | -0.92 | -0.69 |
| cora | k-center/FPS | -6.32 | -7.56 | -4.55 |
| cora | medoid | -9.30 | -7.20 | -8.53 |
| cora | CaT-MMD-TAG adapted | -8.91 | -7.50 | -6.80 |
| cora | GECC-hard-TAG adapted | -1.07 | +1.78 | -2.74 |
| cora | k-center native append | -7.59 | -7.56 | -18.65 |
| cora | medoid native append | -5.33 | -3.26 | -1.17 |
| citeseer | C0 append-edge anchor | -3.51 | -3.34 | -1.36 |
| citeseer | k-center/FPS | -3.22 | -2.25 | -0.82 |
| citeseer | medoid | -3.83 | -5.80 | -1.41 |
| citeseer | CaT-MMD-TAG adapted | -7.22 | -5.64 | -10.40 |
| citeseer | GECC-hard-TAG adapted | -1.01 | +1.67 | -0.46 |
| citeseer | k-center native append | -3.35 | -4.81 | -14.12 |
| citeseer | medoid native append | -0.23 | -0.68 | -8.42 |

## 3. 结果解释

- 该顺序不是用来证明 X+P 稳定 SOTA，而是用来检查开发顺序之外的泛化。若 X+P 在该顺序落后，结果应原样保留，不能事后调参。
- C0 只作为内部 append-edge anchor，用于测量 outer X/P 优化的增量；论文中应把它与公开或适配基线分开标记。
- CaT/GECC 采用当前 TAG Student 和 continual delta 接口的机制适配，不是官方端到端复现；k-center/medoid 是同协议的 real-node controls。
- `order19614` 的 source audit 显示 construction 没有读取历史 raw、没有使用 report/test labels，且每阶段 old prefix 和 old-old topology 保持不变。

## 4. 成本与限制

### cora
- Bounded X+P: construction 1.59s; Student wall time 44.18s。
- C0 append-edge anchor: construction N/A; Student wall time 41.01s。
- k-center/FPS: construction N/A; Student wall time 40.04s。
- medoid: construction N/A; Student wall time 40.13s。
- CaT-MMD-TAG adapted: construction 14.79s; Student wall time 40.93s。
- GECC-hard-TAG adapted: construction 0.50s; Student wall time 40.26s。
- k-center native append: construction 0.04s; Student wall time 40.21s。
- medoid native append: construction 0.16s; Student wall time 40.01s。
### citeseer
- Bounded X+P: construction 1.62s; Student wall time 49.19s。
- C0 append-edge anchor: construction N/A; Student wall time 46.84s。
- k-center/FPS: construction N/A; Student wall time 45.98s。
- medoid: construction N/A; Student wall time 47.32s。
- CaT-MMD-TAG adapted: construction 13.63s; Student wall time 46.87s。
- GECC-hard-TAG adapted: construction 0.44s; Student wall time 45.69s。
- k-center native append: construction 0.03s; Student wall time 45.90s。
- medoid native append: construction 0.25s; Student wall time 45.99s。

## 5. 证据边界

- 这份表是新增独立顺序的补充验证，不替换原有两顺序主表。
- 该表不包含 Semantic-Recursive；当前主线评估聚焦有 Student outer optimization 的 Bounded X+P。
- 不应据此宣称普遍优于所有 coreset 或 TAG 凝聚方法；正确结论是方法在不同顺序和图上具有互补性，且独立顺序暴露出 k-center 的强竞争性。

原始汇总 JSON：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/order19614_xp_baseline_comparison.json`
