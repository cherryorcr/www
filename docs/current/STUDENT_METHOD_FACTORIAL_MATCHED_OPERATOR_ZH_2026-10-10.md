# Student 与构造方法二因素实验：统一推理算子版本

日期：2026-10-10。本文档是最终阶段 report-dev 结果，不是 sealed test。

## 实验目的

本实验用于回答一个关键归因问题：此前更换 Student 后，CaT、GECC、Adaptive coherent 和 Unified-Q 的结果都发生变化，这些变化究竟来自 Student，还是来自凝聚方法本身。

本轮固定类别顺序、memory budget、构造 seed、Student seeds、训练步数、batch、评估 mask 和推理算子，只改变构造方法与 Student 家族。Legacy 与 Prompt Student 最终预测都使用同一个 `data[real_adj]`，因此 Legacy 不再沿用旧版的原始 `edge_index` 邻接重建。

## 协议

| 项目 | 设置 |
|---|---|
| 数据集 | WikiCS、SportsFit、BookHis |
| 类别顺序 | 9101、9102、9103 |
| 构造方法 | Unified-Q edge-free、GECC-hard TAG adapter、CaT-CCTP adapter、Adaptive coherent attribution |
| Student | Legacy row-wise InfoNCE；Prompt multi-positive graph-text Student |
| Student seeds | 150、151、152；每个 cell 内取均值 |
| 每阶段训练 | 2048 steps，batch 24 |
| 评估 | 最终阶段 Macro-F1、Accuracy、Old F1、New F1；report-dev |
| 构造访问 | current delta + 自身 frozen synthetic prefix；不回读历史 raw |
| 推理 | full transductive graph 的统一 `real_adj` |
| 拓扑 | 四种方法均为 edge-free memory，Student 侧使用自环算子 |
| 统计单位 | 3 个配对类别顺序；区间为 df=2 的描述性 t 区间 |

Prompt Student 的 CaT、Adaptive coherent、Unified-Q memory 使用封存结果并通过 27/27 个 final memory hash 校验。GECC 是 current-delta-only 的机制适配器，不是官方 GECC 完整复现；为避免 GPU 稀疏传播导致 KMeans 输入出现微小数值差异，最终 GECC 采用 CPU propagated features，并用 WikiCS 9101 的重复探针验证 hash 一致。

## Final Macro-F1

| 数据集 | Unified-Q Prompt / Legacy | GECC-hard Prompt / Legacy | CaT-CCTP Prompt / Legacy | Adaptive Prompt / Legacy |
|---|---:|---:|---:|---:|
| WikiCS | 79.86 / 51.18 | 79.08 / 46.24 | 80.35 / 50.07 | 80.47 / 72.63 |
| SportsFit | 78.60 / 26.43 | 77.32 / 26.65 | 73.75 / 26.37 | 74.54 / 46.27 |
| BookHis | 41.97 / 16.96 | 41.23 / 16.15 | 24.83 / 17.80 | 23.47 / 19.11 |

## Final Accuracy

| 数据集 | Unified-Q Prompt / Legacy | GECC-hard Prompt / Legacy | CaT-CCTP Prompt / Legacy | Adaptive Prompt / Legacy |
|---|---:|---:|---:|---:|
| WikiCS | 82.85 / 55.30 | 82.22 / 48.88 | 83.44 / 55.96 | 83.61 / 77.60 |
| SportsFit | 83.93 / 31.94 | 85.52 / 27.25 | 81.94 / 27.35 | 82.98 / 62.40 |
| BookHis | 80.34 / 24.71 | 79.97 / 26.76 | 75.45 / 34.56 | 74.61 / 61.87 |

## 同一 Student 内的构造差异

Unified-Q 相对 CaT-CCTP：

| 数据集 | Student | Δ Final Macro-F1 | 95% order t 区间 | 正收益顺序 | Δ Accuracy |
|---|---|---:|---:|---:|---:|
| WikiCS | Prompt | -0.49 | [-2.34, 1.35] | 1/3 | -0.59 |
| WikiCS | Legacy | +1.11 | [-11.97, 14.20] | 1/3 | -0.66 |
| SportsFit | Prompt | +4.85 | [-1.09, 10.79] | 3/3 | +1.99 |
| SportsFit | Legacy | +0.07 | [-6.88, 7.01] | 1/3 | +4.58 |
| BookHis | Prompt | +17.14 | [8.32, 25.97] | 3/3 | +4.89 |
| BookHis | Legacy | -0.84 | [-2.55, 0.87] | 0/3 | -9.85 |

Unified-Q 相对固定 CPU GECC-hard 的 Final Macro-F1 均值差异为：

| 数据集 | Prompt Student | Legacy Student |
|---|---:|---:|
| WikiCS | +0.78 | +4.94 |
| SportsFit | +1.28 | -0.22 |
| BookHis | +0.74 | +0.81 |

这些数字表明 Unified-Q 与 GECC 处于相近水平，不能写成跨数据集、跨 Student 的稳定超越。

## 归因结论

1. Prompt Student 的共同提升是真实的 Student 主效应，不能归因给 Unified-Q。Prompt 相对 Legacy 的 F1 提升在 Unified-Q 上为 WikiCS `+28.68`、SportsFit `+52.17`、BookHis `+25.01` 个百分点；其它构造方法也有类似变化。
2. Unified-Q 的方法收益主要出现在 Prompt Student 下的 SportsFit 和 BookHis，在 WikiCS 上略低于 CaT 和 Adaptive；Legacy 下没有稳定收益。
3. 共享 Q、in-loop distribution matching 和软/硬文本材料化的一致性仍然具有方法创新价值，但当前实验不能单独证明共享 Q 带来普遍的准确率提升。
4. Adaptive coherent 是本实验中的 fixed-X text attribution 内部控制，不应当作为公开外部 baseline，也不应把其 Legacy 高分解释为 Unified-Q 失效；它说明文本分配对 Student 训练协议高度敏感。
5. GECC-hard 是统一 current-delta-only 访问边界下的适配版本，不是官方完整 GECC 复现；论文中应单独标注适配规则、CPU 确定性处理和限制。
6. 本轮不是严格 unseen-class zero-shot，也没有验证跨架构 Student、链路预测、跨数据集迁移或合成拓扑收益。

## 论文表述建议

当前最稳妥的主张不是“Unified-Q 全面 SOTA”，而是：

> 在历史 raw 不可回读的持续 TAG 凝聚协议下，Unified-Q 将图特征凝聚、共享软分配和 distribution matching 放在同一阶段内优化；在图文 Prompt Student 下，它在 SportsFit 和 BookHis 上相对 CaT-CCTP 有明显收益，并与固定协议的 GECC 适配器保持相近性能。Student×构造交互实验进一步表明，持续 TAG 凝聚方法的收益不能脱离下游图文 Student 单独解释。

这支持将 Unified-Q 作为“面向图文持续 TAG 凝聚的可微构造方法候选”，但还不足以支持“普遍 SOTA”或“对任意下游 Student 都有效”的结论。

## 结果与图表

- [完整汇总](../../results/student_method_factorial_20261010/final_matched_operator/summary.md)
- [逐 cell CSV](../../results/student_method_factorial_20261010/final_matched_operator/cell_results.csv)
- [方法均值 CSV](../../results/student_method_factorial_20261010/final_matched_operator/method_means.csv)
- [同 Student 内差值](../../results/student_method_factorial_20261010/final_matched_operator/method_deltas_vs_cat.csv)
- [构造方法交互项](../../results/student_method_factorial_20261010/final_matched_operator/constructor_interactions.csv)
- [审计结果](../../results/student_method_factorial_20261010/matched_cpu_gecc_audit.json)
- [Final Macro-F1 图](../../results/student_method_factorial_20261010/final_matched_operator/figures/final_f1_student_method.png)
- [Final Accuracy 图](../../results/student_method_factorial_20261010/final_matched_operator/figures/accuracy_student_method.png)
- [Student×构造交互图](../../results/student_method_factorial_20261010/final_matched_operator/figures/student_constructor_interaction_f1.png)
