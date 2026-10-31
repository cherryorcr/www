# Adaptive coherent 在 WikiCS 与 BookHis 上的三方法对比

本报告记录同一份 `base+4`、5% fixed-rate cumulative memory、pool-only construction、`report_dev_mod=3`/`test_mod=2` 协议下的正式结果。每个方法使用相同的 edge-aware 2-layer GCN + text projection + bidirectional InfoNCE Student，Student seeds 为 `100..105`，construction seed 为 `270927`。

## 方法

- **CCTP top-5**：CaT-style class-wise MMD 特征凝聚，按每个合成行选择图文兼容的 5 个当前 delta 文本来源。
- **Adaptive coherent**：在 CaT 特征初始化上进行 X-conditioned coherent adaptive slot routing；slot 数量由当前类别图文离散度自适应决定，保留 320 字符文本预算。
- **Adaptive coherent + CCTP anchor**：在 adaptive coherent 上增加 CCTP anchor 约束，用于限制 payload 偏离原始 CCTP 候选。

三者在本 runner 中都属于 edge-free mechanism adaptation，Student 侧使用 self-loop-only synthetic graph；因此这里验证的是特征与文本 payload 机制，不是合成拓扑的完整比较。

## WikiCS

| Method | Final Macro-F1 (mean +/- std) | Final Accuracy (mean +/- std) | Stage avg F1 | Old F1 | New F1 | Worst stage F1 | Construction (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCTP top-5 | 52.96 +/- 3.51 | 56.98 +/- 4.70 | 73.31 | 54.54 | 38.80 | 49.49 | 21.8 |
| Adaptive coherent | 73.87 +/- 1.82 | 77.82 +/- 0.93 | 81.67 | 74.86 | 64.94 | 73.87 | 26.4 |
| Adaptive coherent + CCTP anchor | 69.76 +/- 1.77 | 74.17 +/- 1.80 | 79.48 | 70.90 | 59.47 | 67.23 | 25.2 |

逐阶段最终 checkpoint 指标（每个 cell 已对 6 个 Student seeds 取均值）：

| Method | Stage F1 (0 -> 4) | Stage Accuracy (0 -> 4) |
|---|---|---|
| CCTP top-5 | 49.49, 61.69, 79.50, 81.84, 67.49 | 38.52, 42.50, 67.04, 69.40, 51.36 |
| Adaptive coherent | 85.45, 74.29, 86.19, 90.20, 77.84 | 82.45, 63.06, 71.91, 82.24, 63.95 |
| Adaptive coherent + CCTP anchor | 76.74, 65.94, 86.36, 88.58, 78.08 | 76.85, 52.08, 71.54, 79.51, 64.29 |

## BookHis

| Method | Final Macro-F1 (mean +/- std) | Final Accuracy (mean +/- std) | Stage avg F1 | Old F1 | New F1 | Worst stage F1 | Construction (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCTP top-5 | 17.38 +/- 1.21 | 33.00 +/- 6.09 | 37.62 | 18.41 | 6.08 | 17.38 | 41.2 |
| Adaptive coherent | 17.67 +/- 3.99 | 60.18 +/- 2.94 | 13.21 | 19.17 | 1.23 | 17.67 | 55.3 |
| Adaptive coherent + CCTP anchor | 16.73 +/- 1.77 | 30.67 +/- 7.57 | 32.56 | 17.52 | 8.07 | 16.73 | 46.1 |

逐阶段最终 checkpoint 指标（每个 cell 已对 6 个 Student seeds 取均值）：

| Method | Stage F1 (0 -> 4) | Stage Accuracy (0 -> 4) |
|---|---|---|
| CCTP top-5 | 31.44, 41.95, 25.86, 37.11, 40.22 | 33.12, 37.40, 10.83, 62.35, 25.64 |
| Adaptive coherent | 36.21, 31.24, 3.99, 3.41, 1.23 | 61.25, 19.96, 3.75, 3.70, 0.64 |
| Adaptive coherent + CCTP anchor | 30.82, 45.83, 21.54, 32.62, 31.79 | 29.07, 38.76, 8.33, 49.38, 19.87 |

## 结果解读

- WikiCS：Adaptive coherent 的 Final Macro-F1 为 `73.87`，比 CCTP top-5 高 `20.90` pp；Accuracy 高 `20.84` pp。
- BookHis：Adaptive coherent 的 Final Macro-F1 为 `17.67`，仅比 CCTP top-5 高 `0.29` pp，但 Accuracy 高 `27.18` pp，同时 New F1 只有 `1.23`，说明长尾类别的新类塑性仍然不足，Accuracy 的提升不能等价理解为均衡分类能力提升。
- CCTP anchor 在两个数据集上都没有优于无 anchor 的 adaptive coherent；它可以作为约束消融，但当前不应升级为主方法。
- WikiCS 的正向结果较强且跨 5 个阶段保持；BookHis 的结果说明该机制对大规模、长尾类别流仍需专门的新类保护或类级配额机制。

## 协议限制

本轮每个数据集只有一个冻结类别顺序和一个 construction seed，虽然每个 cell 使用 6 个 Student seeds，但尚未证明顺序泛化。公开基线在此处是机制适配版本，不是所有论文方法的官方端到端复现。后续应在不改变本轮主结果的前提下补 3 个独立顺序，并单独报告 New/Old F1。

## 原始结果

- WikiCS: `results/public_baselines_adaptive_wikics_20261006/`
- BookHis: `results/public_baselines_adaptive_bookhis_20261006/`
- Protocol manifests: `results/public_baselines_adaptive_wikics_20261006/wikics_base4/protocol.json` 和 `results/public_baselines_adaptive_bookhis_20261006/bookhis_base4/protocol.json`
