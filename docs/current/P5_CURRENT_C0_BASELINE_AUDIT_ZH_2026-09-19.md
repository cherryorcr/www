# 当前 C0 与五层基线可比性审计

日期：2026-09-19

## 1. 为什么需要单独审计

此前五层报告中的 `v4` 使用了 v4 response 分配；本轮 P5 冻结的共同控制是 `C0=response off`。两者采用相同的文本、X、拓扑构造规则和 student，但类别配额和相应 memory artifact 可能不同，因此不能直接把旧表的 `v4` 行当作当前 C0 行。

本报告不新增 TAGSAM/STAD 运行，只做口径拆分：

- 当前 C0：使用本轮 `p5_rate_curve_small_20260919` 的 5% canonical 结果；
- 其它基线：复用已通过 artifact audit 的 `five_level_extension_2026_09_17` 与 `small_comparison_audited_r2_2026_09_16`；
- 所有数字仍是 report-dev Macro-F1，六个 student seeds，test 未读取。

## 2. 当前 C0 数字

| 数据集 | C0 final F1 | C0 stage average | C0 old task-slice F1 | C0 new task-slice F1 | rows |
|---|---:|---:|---:|---:|---:|
| Cora | 49.85 | 53.08 | 75.05 | 40.91 | 94 |
| Citeseer | 59.17 | 64.29 | 76.67 | 55.48 | 111 |

**指标审计补充：** 上表 old/new 为任务切片指标，不可直接与历史五层 pooled old/new 比较。C0 与历史 dispersion 的 6 对 memory 完全一致、36 个逐 seed 累计分数一致；同构造归档参照的 pooled old/new 为 Cora `65.44/29.06`、Citeseer `66.46/44.61`。这不是对 P5 原始预测的复算，详见 [指标口径审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。

C0 相对完整 raw 长训所选检查点的最终性能保留率为 `97.16%/106.42%`，阶段平均保留率为 `88.71%/105.43%`，均使用未舍入值。

旧五层表中的 v4 为：Cora `49.85/53.08`，Citeseer `60.77/64.82`。因此 v4 response 相对当前 C0 的最终差异为 Cora `+0.00`、Citeseer `+1.59` 个百分点；P5 后续不再把这 1.59 个百分点归因给“基础 append 构造”。

## 3. 与已审计基线的最终 F1 对照

下面的 `C0 - baseline` 为正时表示当前 C0 更高。完整数据监督参考另行解释，不与零样本主方法直接排名。

### 3.1 Cora

| 方法 | final F1 | C0 - baseline |
|---|---:|---:|
| 当前 C0：持续 append，response off | **49.85** | - |
| 全历史启发式重凝聚 | 47.21 | **+2.64** |
| 旧 synthetic + 当前 delta 重聚类 | 49.70 | **+0.15** |
| 完整 raw，等步数 | 47.93 | **+1.92** |
| 完整 raw，8192 步 selector 最佳 | 51.31 | -1.46 |
| 静态 TAGSAM self-loop 适配 | 47.73 | **+2.12** |
| 静态 TAGSAM semantic 适配 | 52.86 | -3.01 |

### 3.2 Citeseer

| 方法 | final F1 | C0 - baseline |
|---|---:|---:|
| 当前 C0：持续 append，response off | **59.17** | - |
| 全历史启发式重凝聚 | 48.28 | **+10.89** |
| 旧 synthetic + 当前 delta 重聚类 | 54.24 | **+4.93** |
| 完整 raw，等步数 | 49.89 | **+9.28** |
| 完整 raw，8192 步 selector 最佳 | 55.60 | **+3.57** |
| 静态 TAGSAM self-loop 适配 | 51.09 | **+8.09** |
| 静态 TAGSAM semantic 适配 | 42.00 | **+17.17** |

## 4. 可以支持的判断

1. **当前方法是有效的，但不是普遍最优。** Cora 上它接近 recluster，低于 raw-best 和 TAGSAM semantic；Citeseer 上它明显高于已有全历史、recluster、raw 参考和两种 TAGSAM 适配。
2. **持续约束并不是唯一解释。** Cora 的 raw-best/TAGSAM semantic 均值仍然更高，提示 C0 还有构造或 student plasticity 空间；Citeseer 的优势只在该 canonical 条件内成立，不能据此证明跨顺序鲁棒性。
3. **v4 response 的增益不能混入 C0。** Citeseer 的旧 v4 高分只说明 response 版本在该 canonical 顺序上有额外收益，不改变 P5 对 `response off` 的冻结控制。
4. **不能把静态 TAGSAM 适配称为官方 TAGSAM SOTA。** 这些数字是响应机制适配，且静态方法可回读历史 raw；它们用于机制参照，不用于宣称官方方法被超过。
5. **监督 GCN 只能作为任务上限参考。** Cora/Citeseer 的监督参考分别约 `84.10/69.26`，它改变了训练目标和标签权限，不能当作凝聚算法之间的公平排名。

## 5. 与 P5 大图结果的关系

本表是 canonical 三阶段小图五层基线审计；本轮大图使用 `base+4`，两者不直接混合。大图报告回答的是长链可运行性、F1 轨迹和成本，而不是重新声称 C0 超过所有静态基线。

当前论文主线建议写成：

> 在同一 GraphCLIP student 下，当前 C0 在 Citeseer 上相对全历史重凝聚、recluster、完整 raw 等步数参考和静态 TAGSAM 适配均有明显优势；在 Cora 上与 recluster 接近但低于部分静态/长训参照。由此说明方法具有实际效用和清晰的适用边界，而不是普遍 SOTA。

## 6. 后续决策

- 不重跑 TAGSAM/STAD 以追求更高数字；当前对比已经足以支持“有效但有边界”的定位。
- 继续把 C0 作为 P5 长链和大图的共同控制。
- 如需方法创新，下一步应选择一个组件做独立确认，而不是把旧 v4 response、C1、attention 和 C0 事后拼接成新版本。
- 统一五层主表需要在表头明确 `v4(response on)` 与 `C0(response off)`，并把三阶段/五阶段分开。

## 7. 证据入口

- 当前 C0：`results/p5_rate_curve_small_20260919/cora_r005/results.json`、`citeseer_r005/results.json`
- 五层审计汇总：`results/five_level_summary_2026_09_17/summary.csv`
- 五层验收：`results/five_level_summary_2026_09_17/acceptance.json`
- P5 总报告：[P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)
