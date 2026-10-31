# Adaptive coherent 独立顺序与 construction seed 结果

日期：2026-10-06

## 1. 完成情况

- 大图：Arxiv、SportsFit、WikiCS、BookHis；
- 每个数据集：3 个独立类别顺序 × 3 个 construction seeds = 9 个 cell；
- 每个 cell：6 个 Student seeds；
- 总计：36 个大图 cell，全部完成；
- 小图：Cora、Citeseer，3 个类别顺序、6 个 Student seeds；当前已完成 construction seed `270927` 的完整消融；
- 所有实验使用 current-delta-only、frozen-prefix、5% memory；大图与小图使用的 Student 不同，不能合并视为同一个 Student 协议；
- 历史 raw graph 读取次数为 0。

### Student 记录更正（2026-10-08）

原文“所有实验使用同一个 prompt zero-shot Student”不准确。大图原始 `summary.json` 的 `protocol.student` 记录为 `same original edge-aware 2-layer GCN + text projection + bidirectional InfoNCE`，batch 为 16，Student seeds 为 100--105，steps 为 `max(50,min(2048,16*memory_rows))`。小图使用单独的 Prompt Student runner。

可核查记录：`results/adaptive_order_seed_remote_20261006/adaptive_order_wikics_o9101_c270927_20261006/summary.json`；大图入口为 `code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py`。

本文件保留历史数值，其大图方法间的同协议比较不因这项文档更正而失效。但不能与 2026-10-08 的 In-loop CaT-Q 表直接计算跨版本增益：后者使用多正样本 InfoNCE、prompt CE 和 prompt margin 的 Prompt Student，且 batch、seeds 和 memory 轨迹也有变化。目前尚未在当前统一协议下完成 Adaptive coherent 与 In-loop CaT-Q 的直接比较；不能据此认定 Adaptive coherent 的优势已经消失，也不能把分数变化全部归因于 Student。

## 2. 大图结果

数值是 9 个顺序/构造 cell 的均值；括号内为 cell 间标准差。

| Dataset | Method | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 |
|---|---|---:|---:|---:|---:|---:|
| Arxiv | **Adaptive coherent** | **21.93 +/- 1.42** | **41.41 +/- 3.02** | **22.27** | **20.03** | 29.95 |
| Arxiv | Fixed coherent | 22.39 +/- 1.28 | 40.27 +/- 3.00 | 22.82 | 19.97 | 30.23 |
| Arxiv | CCTP top-5 | 18.93 +/- 0.89 | 29.86 +/- 2.94 | 19.06 | 18.20 | 30.95 |
| Arxiv | K-center | 17.50 +/- 0.49 | 26.98 +/- 1.85 | 17.17 | 19.37 | 29.74 |
| Arxiv | Medoid | 18.19 +/- 1.03 | 26.98 +/- 3.37 | 18.01 | 19.26 | 30.55 |
| SportsFit | **Adaptive coherent** | **47.31 +/- 1.95** | **64.51 +/- 6.64** | **46.66** | **50.93** | 52.83 |
| SportsFit | Fixed coherent | 48.34 +/- 1.95 | 65.06 +/- 5.79 | 47.79 | 51.37 | 54.57 |
| SportsFit | CCTP top-5 | 25.02 +/- 2.43 | 24.49 +/- 4.66 | 24.76 | 26.46 | 37.86 |
| SportsFit | K-center | 29.54 +/- 1.01 | 34.44 +/- 2.86 | 28.74 | 33.93 | 44.77 |
| SportsFit | Medoid | 29.04 +/- 0.93 | 34.70 +/- 3.00 | 29.20 | 28.16 | 43.48 |
| WikiCS | **Adaptive coherent** | **73.06 +/- 1.60** | **77.07 +/- 2.46** | **73.29** | **70.95** | **81.64** |
| WikiCS | Fixed coherent | 68.99 +/- 1.46 | 74.51 +/- 1.83 | 69.54 | 64.02 | 77.54 |
| WikiCS | CCTP top-5 | 50.44 +/- 2.28 | 54.03 +/- 3.17 | 51.24 | 43.25 | 67.57 |
| WikiCS | K-center | 50.29 +/- 2.70 | 53.41 +/- 3.88 | 50.49 | 48.51 | 70.70 |
| WikiCS | Medoid | 55.01 +/- 2.24 | 59.29 +/- 4.22 | 55.56 | 50.07 | 71.31 |
| BookHis | **Adaptive coherent** | **14.71 +/- 2.67** | **48.83 +/- 19.48** | 14.53 | 16.69 | 17.71 |
| BookHis | Fixed coherent | 14.99 +/- 3.10 | 47.24 +/- 19.64 | 15.30 | 11.58 | 17.91 |
| BookHis | CCTP top-5 | 14.74 +/- 2.94 | 26.74 +/- 11.22 | 13.92 | 23.69 | 39.49 |
| BookHis | K-center | **18.08 +/- 3.03** | 35.68 +/- 10.67 | **17.12** | **28.69** | **44.46** |
| BookHis | Medoid | 17.33 +/- 1.96 | 32.12 +/- 7.61 | 16.46 | 26.98 | 42.87 |

## 3. Paired结果

相对同一 cell 的 CCTP top-5：

| Dataset | Adaptive - CCTP Final F1 | Positive cells | Worst delta |
|---|---:|---:|---:|
| Arxiv | +3.00 pp | 9/9 | +0.79 pp |
| SportsFit | +22.29 pp | 9/9 | +16.49 pp |
| WikiCS | +22.62 pp | 9/9 | +18.78 pp |
| BookHis | -0.03 pp | 5/9 | -8.07 pp |
| Equal-dataset mean | +11.97 pp | 32/36 | -8.07 pp |

这说明主方法的正向结果不是只来自一个 construction seed：Arxiv、SportsFit、WikiCS 三个数据集的 9 个配对 cell 全部超过 CCTP。BookHis 则没有稳定收益。

相对简单 coreset，Adaptive coherent 在 Arxiv、SportsFit 和 WikiCS 明显更好，但在 BookHis 低于 K-center 和 Medoid。这一结论比 pooled 平均更重要。

## 4. 小图结果

当前为 construction seed `270927`，三个类别顺序的等权均值。

| Dataset | Method | Final F1 | Accuracy | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|
| Cora | **Adaptive coherent** | **83.89** | **85.63** | **83.44** | **84.47** |
| Cora | Fixed coherent | 83.76 | 85.49 | 83.19 | 84.51 |
| Cora | Formal prompt-margin | 83.93 | 85.51 | 83.60 | 84.37 |
| Cora | CaT nearest | 83.65 | 85.51 | 83.10 | 84.39 |
| Citeseer | **Adaptive coherent** | 69.58 | 73.89 | 69.60 | 69.54 |
| Citeseer | Fixed coherent | 69.58 | 73.88 | 69.52 | 69.69 |
| Citeseer | Formal prompt-margin | **70.19** | **73.96** | **70.09** | **70.38** |
| Citeseer | CaT nearest | 69.63 | 73.63 | 69.44 | 70.02 |

小图结果说明 Adaptive coherent 至少没有破坏 CaT，但它的增益很小：Cora 相对 CaT 约 `+0.24 pp`，Citeseer 约 `-0.05 pp`。因此论文不能声称它在所有规模和数据集上都显著领先。

## 5. 结论

1. Adaptive coherent 已经有足够证据作为当前主方法候选：大图 4 个数据集中的 3 个上稳定优于 CCTP，且相对 CCTP 的总正 cell 比例为 `32/36`。
2. 目前最强、最一致的信号来自“coherent payload + X-conditioned routing”，而不是简单 anchor；anchor 在多个数据集上会牺牲总体 F1。
3. Adaptive slot 并非所有数据集都优于 fixed slot：WikiCS 上明显有益，但 Arxiv/SportsFit 略低于 fixed coherent。因此 adaptive slot 应作为主方法的规模泛化版本，但需要在论文中报告固定 slot 消融，不能把 adaptive 规则说成普遍最优。
4. BookHis 仍是核心风险：总体 F1 不稳定，且 K-center/Medoid 更强。这里应继续做新类容量、类别预测分布和 slot coverage 诊断，而不是只看 Accuracy。
5. 当前结果支持开始写论文，但最终主张应是“在受限持续 TAG 协议下的稳定综合改善”，而不是“跨所有数据集 SOTA”。

## 6. 结果文件

- 大图 cell 汇总：[cells.csv](../../results/adaptive_order_seed_summary_20261006/cells.csv)
- 大图统计：[summary.json](../../results/adaptive_order_seed_summary_20261006/summary.json)
- 大图报告：[summary.md](../../results/adaptive_order_seed_summary_20261006/summary.md)
- 小图结果：[summary.json](../../results/adaptive_small_seed270927_20261006_remote/summary.json)
