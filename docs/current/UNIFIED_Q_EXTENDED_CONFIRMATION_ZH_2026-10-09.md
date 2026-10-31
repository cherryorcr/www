# Unified-Q 3×3 独立确认实验报告

## 1. 实验完成情况

本轮完成冻结协议下的完整矩阵：

- WikiCS、SportsFit、BookHis；
- 每个数据集 3 个可行类别顺序；
- 每个顺序 3 个 construction seeds；
- 共 27 个构造 cell；
- 每个 cell 比较 `uq_qmedoid`、`uq_dm_edgefree`、`cat_cctp`、`gecc_hard` 四个方法臂；
- 每个 memory 使用 Prompt-GCN、Prompt-SGC 和 Prompt-GraphSAGE 三种下游 Student；
- Student seeds 为 150/151/152，2048 steps，batch size 24；
- 评价为 final memory 的 report-dev，不是 sealed test。

27 个构造 cell 和 27 个 GraphSAGE readout 均完成，所有 exit code 为 0。GCN/SGC/GraphSAGE 使用同一 cell 的同一份 memory，memory hash 审计通过。构造阶段采用当前 delta 加自身冻结 synthetic prefix，旧 synthetic prefix 不更新；memory 为 edge-free。

## 2. 方法与基线

| 名称 | 类型 | 适配方式 |
|---|---|---|
| In-loop CaT-Q | 当前主方法 | class-constrained learned Q、in-loop distribution/gradient matching、bounded X residual、Q-conditioned real-text routing |
| Q-medoid | 内部单因素控制 | 仅将 Q-max 硬文本材料化替换为同类 Q-weighted medoid，其他部分不变 |
| CaT + nearest5 | 外部论文机制适配 | CaT-style X condensation，当前 delta-only、冻结旧前缀，附加当前 delta 的 graph-nearest5 文本 |
| GECC-hard + TAG | 外部论文机制适配 | GECC hard clustering 与 TAG propagation weights，当前 delta-only、冻结旧前缀，附加 graph-nearest5 文本 |

CaT 和 GECC 是机制适配，不是官方完整流水线复现；不能把本表写成原论文的官方结果。所有方法使用相同 memory row budget、320 字符上限、Student 训练步数和评估划分，但实际文本 token/字节信息量并不严格相同。

## 3. Final Macro-F1 / Accuracy

下面数值为 9 个构造 cell 的均值 ± cell 标准差。Student seeds 先在每个 cell 内平均，再跨顺序和 construction seed 汇总。

### WikiCS

| Student | 方法 | Final Macro-F1 | Accuracy |
|---|---|---:|---:|
| Prompt-GCN | Q-medoid | 80.43 ± 1.04 | 82.61 ± 1.23 |
| Prompt-GCN | In-loop CaT-Q | 80.38 ± 1.16 | 82.46 ± 1.62 |
| Prompt-GCN | CaT + nearest5 | 80.71 ± 0.44 | 83.49 ± 0.37 |
| Prompt-GCN | GECC-hard + TAG | 80.09 ± 0.89 | 82.74 ± 0.93 |
| Prompt-SGC | Q-medoid | 78.34 ± 0.46 | 81.52 ± 0.31 |
| Prompt-SGC | In-loop CaT-Q | 78.38 ± 0.96 | 81.36 ± 0.86 |
| Prompt-SGC | CaT + nearest5 | 72.20 ± 2.32 | 75.53 ± 2.12 |
| Prompt-SGC | GECC-hard + TAG | 78.69 ± 0.43 | 81.24 ± 0.48 |
| Prompt-GraphSAGE | Q-medoid | 79.42 ± 1.53 | 81.51 ± 1.62 |
| Prompt-GraphSAGE | In-loop CaT-Q | 79.66 ± 1.72 | 81.63 ± 1.98 |
| Prompt-GraphSAGE | CaT + nearest5 | 80.91 ± 0.88 | 83.36 ± 0.78 |
| Prompt-GraphSAGE | GECC-hard + TAG | 78.74 ± 1.48 | 80.80 ± 1.63 |

### SportsFit

| Student | 方法 | Final Macro-F1 | Accuracy |
|---|---|---:|---:|
| Prompt-GCN | Q-medoid | 77.86 ± 2.03 | 83.76 ± 2.09 |
| Prompt-GCN | In-loop CaT-Q | 77.88 ± 2.68 | 84.07 ± 2.08 |
| Prompt-GCN | CaT + nearest5 | 76.00 ± 3.30 | 84.73 ± 3.11 |
| Prompt-GCN | GECC-hard + TAG | 77.61 ± 2.03 | 85.17 ± 2.09 |
| Prompt-SGC | Q-medoid | 72.66 ± 1.00 | 80.99 ± 1.45 |
| Prompt-SGC | In-loop CaT-Q | 72.20 ± 0.82 | 80.56 ± 1.59 |
| Prompt-SGC | CaT + nearest5 | 55.12 ± 4.10 | 60.07 ± 7.49 |
| Prompt-SGC | GECC-hard + TAG | 74.55 ± 2.46 | 82.13 ± 2.93 |
| Prompt-GraphSAGE | Q-medoid | 75.29 ± 2.81 | 80.56 ± 2.66 |
| Prompt-GraphSAGE | In-loop CaT-Q | 75.27 ± 2.06 | 80.00 ± 1.79 |
| Prompt-GraphSAGE | CaT + nearest5 | 77.39 ± 2.97 | 84.83 ± 2.67 |
| Prompt-GraphSAGE | GECC-hard + TAG | 74.91 ± 1.93 | 81.86 ± 3.44 |

### BookHis

| Student | 方法 | Final Macro-F1 | Accuracy |
|---|---|---:|---:|
| Prompt-GCN | Q-medoid | 37.63 ± 3.27 | 73.34 ± 5.02 |
| Prompt-GCN | In-loop CaT-Q | 38.49 ± 3.14 | 73.88 ± 4.41 |
| Prompt-GCN | CaT + nearest5 | 17.73 ± 3.45 | 54.25 ± 20.05 |
| Prompt-GCN | GECC-hard + TAG | 37.78 ± 3.41 | 71.99 ± 6.92 |
| Prompt-SGC | Q-medoid | 34.11 ± 3.94 | 69.77 ± 6.34 |
| Prompt-SGC | In-loop CaT-Q | 33.96 ± 3.80 | 69.72 ± 5.43 |
| Prompt-SGC | CaT + nearest5 | 15.68 ± 2.03 | 50.78 ± 15.82 |
| Prompt-SGC | GECC-hard + TAG | 35.30 ± 3.37 | 71.38 ± 4.97 |
| Prompt-GraphSAGE | Q-medoid | 35.58 ± 4.57 | 70.17 ± 7.13 |
| Prompt-GraphSAGE | In-loop CaT-Q | 36.83 ± 4.22 | 71.23 ± 6.48 |
| Prompt-GraphSAGE | CaT + nearest5 | 29.24 ± 4.05 | 63.26 ± 14.92 |
| Prompt-GraphSAGE | GECC-hard + TAG | 35.50 ± 4.32 | 69.45 ± 8.08 |

## 4. 主方法相对外部基线

正值表示 In-loop CaT-Q 更好。差值按同一数据集、类别顺序和 construction seed 配对；区间为 3×3 crossed bootstrap，仅作探索性描述。

| 数据集 / Student | 相对 CaT F1 | 相对 GECC F1 | 相对 GECC Accuracy |
|---|---:|---:|---:|
| WikiCS / Prompt-GCN | -0.33 | +0.29 | -0.28 |
| WikiCS / Prompt-SGC | +6.18 | -0.32 | +0.12 |
| WikiCS / Prompt-GraphSAGE | -1.26 | +0.92 | +0.83 |
| SportsFit / Prompt-GCN | +1.88 | +0.27 | -1.10 |
| SportsFit / Prompt-SGC | +17.07 | -2.36 | -1.57 |
| SportsFit / Prompt-GraphSAGE | -2.12 | +0.36 | -1.86 |
| BookHis / Prompt-GCN | +20.76 | +0.71 | +1.89 |
| BookHis / Prompt-SGC | +18.28 | -1.33 | -1.66 |
| BookHis / Prompt-GraphSAGE | +7.60 | +1.34 | +1.79 |

## 5. 结果解释

1. **协议完成度已经足够做稳定性判断。** 主方法的结果不是由单一顺序或单一 construction seed 支撑，27 个构造 cell 和三种 Student 均完成。
2. **主方法相对 CaT 的优势主要在 BookHis 和 Prompt-SGC 上。** 这说明当前 Q/DM/GM 与文本 routing 对部分长尾或架构条件有效，但不是所有数据集和 Student 都一致有效。
3. **相对 GECC 没有稳定优势。** 主方法在 WikiCS 的 GraphSAGE、BookHis 的 GCN/GraphSAGE 上略高，但在 SGC 的三个数据集上低于 GECC；因此不能写成“稳定超过 GECC”。
4. **Q-medoid 不是升级后的主方法。** 它与原主方法非常接近，说明硬文本材料化规则不是当前收益的唯一来源，也不支持仅凭 medoid 宣称文本模块已经被证明必要。
5. **CaT 的失败不能简单解释为 CaT 本身弱。** 当前 CaT 是在 TAG continual、current-delta-only、冻结旧前缀和 Prompt Student 协议下的适配；它在 Prompt-GCN/GraphSAGE 的部分数据集仍然有竞争力，而在 SGC/BookHis 上明显退化，说明基线适配和 Student 架构交互很强。

## 6. 客观论文边界

当前证据可以支持：

- 一个受历史 raw 不可回读约束的 TAG continual condensation setting；
- 一个将 class-constrained Q、in-loop distribution/gradient matching、bounded X refinement 和 real-text routing 放入同一构造过程的候选方法；
- 在部分数据集、顺序和 Student 上的增益，以及对 CaT/GECC 适配的系统比较；
- 对文本硬材料化、跨架构迁移和负结果的机制分析。

当前证据不能支持：

- 稳定 SOTA；
- 对所有 Student 的普遍优越性；
- Q 共享、文本 routing 或 GM 中每个组件都已被单独证明不可替代；
- 官方 CaT/GECC 复现结果；
- sealed test 上的最终结论。

下一步应冻结这轮结果，补充封存 test 和更高保真的外部基线核对；不要继续根据 report-dev 结果堆叠新模块。若后续要优化，优先针对已观测到的瓶颈做单一改动并重新做小规模确认，而不是同时加入拓扑、EM 和新的 Student loss。

## 7. 产物

- [完整结构化汇总](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/summary.json)
- [完整表格](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/summary.md)
- [协议审计](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/audit.json)
- [Macro-F1 可视化](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/final_f1.png)
- [Accuracy 可视化](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/accuracy.png)
- [主方法相对 CaT 差值](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/delta_vs_cat.png)
- [主方法相对 GECC 差值](D:/Aprojects/continue_tag_condensation/results/uq_text_path_extended_20261009/delta_vs_gecc.png)
- [15 个新增确认 cell 备份](D:/Aprojects/continue_tag_condensation/results/uq_confirmation_extended_backup_20261009)
- [27 个 GraphSAGE readout 备份](D:/Aprojects/continue_tag_condensation/results/uq_sage_extended_backup_20261009)
