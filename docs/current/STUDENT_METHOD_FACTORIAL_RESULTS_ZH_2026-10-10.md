# Student 与构造方法二因素实验结果

日期：2026-10-10。

## 实验目的

本实验用于解释“更换 Student 后 CaT、GECC、Unified-Q 等方法的分数共同升高”的原因。固定类别顺序、预算、current-delta-only 访问边界、训练步数、batch、Student seeds、report-dev 规则和推理算子，只改变构造方法与 Student 家族。

统计单位是三个类别顺序（9101、9102、9103）；每个顺序内的三个 Student seeds 先取均值，不把 Student seed 当成额外的独立顺序。

## 协议

| 项目 | 设置 |
|---|---|
| 数据集 | WikiCS、SportsFit、BookHis |
| 构造方法 | Unified-Q edge-free、GECC-hard TAG adapter、CaT-CCTP adapter、Adaptive coherent attribution |
| Student | Legacy row-wise InfoNCE；Prompt multi-positive graph-text Student |
| Student seeds | 150、151、152 |
| 每阶段训练 | 2048 steps，batch 24 |
| 评估 | Final-stage Macro-F1、Accuracy、Old F1、New F1；report-dev |
| 构造访问 | 当前 delta + 自身 frozen synthetic prefix；不回读历史 raw |
| 拓扑 | edge-free memory；Student 使用统一的全图归一化推理算子 |

Prompt Student 的 CaT、Adaptive coherent 和 Unified-Q 结果复用了已封存的 memory，并逐项核对 memory hash；GECC 使用相同 current protocol 在内存中重建后评估，没有保存额外大文件。Adaptive coherent 在此处是 fixed-X text attribution，不等同于历史完整 Adaptive coherent 构造。

## Final Macro-F1

| 数据集 | Unified-Q Prompt / Legacy | GECC Prompt / Legacy | CaT-CCTP Prompt / Legacy | Adaptive Prompt / Legacy |
|---|---:|---:|---:|---:|
| WikiCS | 79.86 / 51.12 | 79.47 / 47.19 | 80.35 / 50.09 | 80.47 / 72.67 |
| SportsFit | 78.60 / 26.14 | 78.31 / 26.63 | 73.75 / 25.89 | 74.54 / 45.40 |
| BookHis | 41.97 / 16.45 | 40.59 / 16.67 | 24.83 / 17.19 | 23.47 / 19.10 |

数值为三个类别顺序的均值；详细 order 均值、标准差和每个 seed 结果见 [summary.md](../../results/student_method_factorial_20261010/summary.md)。

## Final Accuracy

| 数据集 | Unified-Q Prompt / Legacy | GECC Prompt / Legacy | CaT-CCTP Prompt / Legacy | Adaptive Prompt / Legacy |
|---|---:|---:|---:|---:|
| WikiCS | 82.85 / 55.25 | 82.37 / 49.97 | 83.44 / 55.95 | 83.61 / 77.59 |
| SportsFit | 83.93 / 31.14 | 85.93 / 27.72 | 81.94 / 26.63 | 82.98 / 62.08 |
| BookHis | 80.34 / 24.19 | 79.25 / 31.86 | 75.45 / 33.83 | 74.61 / 61.31 |

## 关键归因

### Student 主效应

Prompt Student 相对 Legacy Student 的 Final Macro-F1 差异为：

| 数据集 | Unified-Q | GECC | CaT-CCTP | Adaptive |
|---|---:|---:|---:|---:|
| WikiCS | +28.74 | +32.28 | +30.26 | +7.80 |
| SportsFit | +52.46 | +51.68 | +47.86 | +29.14 |
| BookHis | +25.51 | +23.91 | +7.63 | +4.37 |

这说明分数整体抬升的主要来源确实包含显著的 Student 主效应：Prompt Student 使用同类多正样本图文 InfoNCE、prompt CE 和 prompt margin，且预测时不训练分类头；Legacy 使用原始 row-wise InfoNCE。两者不是同一个 Student 的简单实现替换，因此绝对 F1 不能跨 Student 直接排名。

### Unified-Q 相对 CaT-CCTP

在同一 Student 内配对比较：

| 数据集 | Prompt Student | Legacy Student |
|---|---:|---:|
| WikiCS | -0.49 F1，+1/3 顺序 | +1.03 F1，+1/3 顺序 |
| SportsFit | +4.85 F1，+3/3 顺序 | +0.25 F1，+1/3 顺序 |
| BookHis | +17.14 F1，+3/3 顺序 | -0.74 F1，+0/3 顺序 |

因此 Unified-Q 的方法收益不是一个与 Student 无关的常数：Prompt Student 下，尤其在 SportsFit 和 BookHis 上有清晰收益；Legacy Student 下收益很小或为负。这是明显的 Student×构造交互，也是此前“所有基线在 Unified-Q 后都一起变高”的主要解释之一。

### 与 GECC 的关系

Prompt Student 下 Unified-Q 相对 GECC 的 Final F1 差异为 WikiCS `+0.39`、SportsFit `+0.29`、BookHis `+1.38`；Legacy Student 下分别为 `+3.93`、`-0.49`、`-0.22`。当前证据不能写成 Unified-Q 稳定超过 GECC，只能写成：在同一 current continual protocol 下，Unified-Q 与 GECC 处于相近水平，且差异依赖数据集和 Student。

## 方法解释上的结论

1. Prompt Student 的共同提升不能归因于凝聚方法；它是 Student 训练目标、同类正样本处理、prompt 监督和推理算子共同变化的结果。
2. Unified-Q 的当前实证优势集中在 Prompt Student 下的 SportsFit 和 BookHis，而不是跨 Student、跨数据集的稳定优势。
3. WikiCS 上 CaT-CCTP 和 Adaptive coherent 的 Prompt F1 略高于 Unified-Q；因此不能把 Unified-Q 叙述成全面 SOTA。
4. 本实验支持把 Unified-Q 定位为“面向图文持续凝聚的可微共享分配与 distribution-matching 方法候选”，但还不足以证明共享 Q 本身带来普遍准确率收益。
5. 该实验不是严格 unseen-class zero-shot：构造和 Student 训练阶段使用了已见类别标签；Prompt Student 只是预测阶段不拟合新的分类头，并使用类别 prompt。
6. 本实验也没有验证跨架构 Student 泛化、链路预测、跨数据集迁移或合成拓扑收益。

## 结果文件与图

- [完整汇总](../../results/student_method_factorial_20261010/summary.md)
- [逐 cell CSV](../../results/student_method_factorial_20261010/cell_results.csv)
- [方法均值 CSV](../../results/student_method_factorial_20261010/method_means.csv)
- [同 Student 内相对 CaT 的差值](../../results/student_method_factorial_20261010/method_deltas_vs_cat.csv)
- [Student 主效应](../../results/student_method_factorial_20261010/student_effects.csv)
- [Unified-Q 交互项](../../results/student_method_factorial_20261010/unified_q_interactions.csv)
- [memory hash 审计](../../results/student_method_factorial_20261010/audit.json)
- [Final Macro-F1 图](../../results/student_method_factorial_20261010/figures/final_f1_student_method.png)
- [Final Accuracy 图](../../results/student_method_factorial_20261010/figures/accuracy_student_method.png)
- [Student×方法交互图](../../results/student_method_factorial_20261010/figures/student_constructor_interaction_f1.png)
