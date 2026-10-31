# XSP 四臂归因实验报告
> 目的：在完全相同的 CaT 初始化、类别顺序、memory budget 和 Student 协议下，区分 XSP 的文本分配作用与二次更新 X 的作用。
## 1. 实验问题与四个版本
本轮不是新的主方法搜索，而是对 XSP 性能变化做因果归因。每个阶段只运行一次 CaT，并把同一个 CaT synthetic X 作为四个版本的共同锚点。
| 版本 | X | 文本 | 归因含义 |
|---|---|---|---|
| CaT-nearest | CaT X | CaT graph-nearest top-5 | 共同基线 |
| Fixed-X XSP | 固定 CaT X | XSP slot-conditioned text | 只改变文本分配 |
| XSP-X + CaT-text | XSP 更新 X | CaT 文本 | 只观察二次 X 更新的净影响（X 的训练目标仍来自 XSP） |
| Full XSP | XSP 更新 X | XSP slot-conditioned text | 当前完整 XSP |

## 2. 协议控制
- 数据集：Cora、Citeseer；顺序：`order19611`、`order19612`、`order19613`；三阶段、5% cumulative memory。
- 每阶段只向构造器提供当前 delta 和 synthetic parent；`historical_raw_reads=0`。旧 memory 不修改；拓扑为 edge-free，Student 侧补 self-loop。
- CaT：class-wise MMD，3000 steps；XSP：30 steps、5% trust radius；Student seeds：150--155，原两层 GCN、frozen MiniLM、双向 InfoNCE。
- 评估：full real graph 的 report-dev 节点分类；报告 Final Macro-F1、Accuracy、Old F1、New F1。
- 每个 cell 的构造哈希、parent prefix、checkpoint reload 审计均通过。

## 3. 最终阶段逐 cell 结果
| Dataset | Order | 方法 | Final F1 | Accuracy | Old F1 | New F1 | ΔF1 vs CaT | ΔAcc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | order19611 | CaT-nearest | 57.278 | 60.658 | 64.484 | 42.865 | 0.000 | 0.000 |
| citeseer | order19611 | Fixed-X XSP | 55.482 | 58.986 | 62.231 | 41.984 | -1.795 | -1.672 |
| citeseer | order19611 | Full XSP | 54.345 | 58.150 | 60.895 | 41.244 | -2.933 | -2.508 |
| citeseer | order19611 | XSP-X + CaT-text | 55.789 | 58.516 | 62.488 | 42.390 | -1.488 | -2.142 |
| citeseer | order19612 | CaT-nearest | 51.036 | 55.068 | 46.609 | 59.890 | 0.000 | 0.000 |
| citeseer | order19612 | Fixed-X XSP | 54.010 | 60.345 | 47.776 | 66.478 | 2.974 | 5.277 |
| citeseer | order19612 | Full XSP | 58.023 | 63.584 | 53.560 | 66.951 | 6.988 | 8.516 |
| citeseer | order19612 | XSP-X + CaT-text | 55.321 | 59.457 | 52.375 | 61.213 | 4.285 | 4.389 |
| citeseer | order19613 | CaT-nearest | 58.351 | 63.114 | 51.298 | 72.458 | 0.000 | 0.000 |
| citeseer | order19613 | Fixed-X XSP | 53.998 | 60.293 | 45.274 | 71.448 | -4.353 | -2.821 |
| citeseer | order19613 | Full XSP | 52.816 | 57.837 | 44.888 | 68.671 | -5.536 | -5.277 |
| citeseer | order19613 | XSP-X + CaT-text | 58.003 | 62.957 | 51.020 | 71.967 | -0.349 | -0.157 |
| cora | order19611 | CaT-nearest | 53.813 | 58.549 | 56.821 | 49.804 | 0.000 | 0.000 |
| cora | order19611 | Fixed-X XSP | 54.067 | 61.193 | 53.725 | 54.524 | 0.254 | 2.645 |
| cora | order19611 | Full XSP | 51.088 | 59.902 | 50.090 | 52.419 | -2.726 | 1.353 |
| cora | order19611 | XSP-X + CaT-text | 53.355 | 58.303 | 56.347 | 49.367 | -0.458 | -0.246 |
| cora | order19612 | CaT-nearest | 52.081 | 61.870 | 39.504 | 68.850 | 0.000 | 0.000 |
| cora | order19612 | Fixed-X XSP | 56.483 | 62.116 | 45.719 | 70.835 | 4.402 | 0.246 |
| cora | order19612 | Full XSP | 54.438 | 60.886 | 42.376 | 70.520 | 2.357 | -0.984 |
| cora | order19612 | XSP-X + CaT-text | 51.962 | 61.624 | 39.639 | 68.392 | -0.119 | -0.246 |
| cora | order19613 | CaT-nearest | 53.109 | 61.439 | 40.926 | 69.352 | 0.000 | 0.000 |
| cora | order19613 | Fixed-X XSP | 51.141 | 57.626 | 41.893 | 63.471 | -1.968 | -3.813 |
| cora | order19613 | Full XSP | 50.692 | 57.196 | 37.542 | 68.225 | -2.417 | -4.244 |
| cora | order19613 | XSP-X + CaT-text | 53.283 | 61.685 | 40.947 | 69.733 | 0.175 | 0.246 |

## 4. 跨六个最终 cell 的 pooled 结果
| 方法 | Final F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 | 阶段平均 Accuracy | 正向 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| CaT-nearest | 54.278 | 60.116 | 49.940 | 60.536 | 49.769 | 47.278 | reference |
| Fixed-X XSP | 54.197 | 60.093 | 49.436 | 61.457 | 49.710 | 48.018 | 3/6 |
| XSP-X + CaT-text | 54.619 | 60.424 | 50.469 | 60.510 | 49.741 | 47.267 | 2/6 |
| Full XSP | 53.567 | 59.592 | 48.225 | 61.338 | 48.459 | 47.889 | 2/6 |

相对 CaT-nearest 的 pooled 最终阶段变化：
- **Fixed-X XSP**：Final F1 -0.081 pp，Accuracy -0.023 pp，Old F1 -0.504 pp，New F1 0.920 pp。
- **XSP-X + CaT-text**：Final F1 0.341 pp，Accuracy 0.307 pp，Old F1 0.529 pp，New F1 -0.026 pp。
- **Full XSP**：Final F1 -0.711 pp，Accuracy -0.524 pp，Old F1 -1.715 pp，New F1 0.802 pp。

## 5. 归因结论
1. **文本 slot 分配本身不是主要退化源，但也没有形成稳定总体收益。** Fixed-X XSP 的 pooled Final F1 与 CaT 几乎持平（-0.081 pp），New F1 提升 0.920 pp，但 Old F1 下降 -0.504 pp；说明 slot 文本更偏向新类塑性，不能直接当作通用提升。
2. **当前二次更新 X 的目标存在旧类风险。** XSP-X + CaT-text 的 pooled Final F1 为 0.341 pp，略有正向，但只有 2/6 个最终 cell 为正，说明 X 更新不是稳定改进；它至少没有证明当前 XSP 文本材料化是唯一问题。
3. **Full XSP 的联合目标发生冲突。** Full XSP 的 pooled Final F1 为 -0.711 pp，Old F1 为 -1.715 pp，而 New F1 为 0.802 pp。当前 `XSP loss + slot text materialization` 更像是在用旧类保持换新类塑性，且跨顺序不稳定。
4. **这轮不能把 XSP 晋升为 CaT 的稳定替代方法。** 更合理的判断是：XSP 已证明“X-conditioned payload partition”有可测的新类方向信号，但当前 X 更新目标、文本硬材料化和旧类保持之间没有形成稳定闭环。

## 6. 后续推进建议
- 暂不继续堆叠 selector、attention 或拓扑模块；先把 XSP 的 X 更新拆成更可控的 trust-region / old-class retention 约束，并在相同六个 cell 上复验。
- 若目标是保留新类塑性，可把 Fixed-X XSP 作为文本分配控制臂；若目标是提升总体 F1，优先研究 `XSP-X + CaT-text` 的 X 更新为何只在部分顺序有效。
- 这轮使用 edge-free memory，因此只能归因 X 与文本，不能证明完整 TAG 拓扑凝聚；后续拓扑实验必须另设共同拓扑基线。
- 代码：`code/v2_llm_fusion/run_xsp_attribution_small_20261005.py`；XSP 修正：`code/v2_llm_fusion/x_conditioned_payload_partition.py`。原始 JSON 与 cell 审计保存在对应结果目录。
