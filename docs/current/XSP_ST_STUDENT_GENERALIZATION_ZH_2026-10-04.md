# XSP-Sinkhorn Student 泛化验证报告

状态：冻结 memory 的两种 Student 泛化验证已完成。

## 1. 实验问题

ST-Sinkhorn 是否只对原始两层 GCN Student 有效，还是对不同容量的下游图文 Student 仍有作用？

本轮不重新构造 memory。对同一 construction seed `314159` 生成的 Current XSP 与 ST-Sinkhorn memory，分别训练：

1. `canonical_2layer`：原始两层 GCN + text projection + 双向 InfoNCE；
2. `shallow_1layer`：单层 GCN + text projection + 双向 InfoNCE。

两种 Student 都使用相同的 batch、steps、Student seeds 和 report-dev 评估规则。这样比较的是“同一凝聚 memory 对不同下游 Student 的迁移”，不是重新调参后的新方法。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Arxiv、BookChild、SportsFit |
| 阶段 | base+4 |
| memory | cumulative fixed-rate 5% |
| construction seed | `314159`，memory 已冻结 |
| Student seeds | `100--105`，6 个 paired seeds |
| batch / steps | `16` / `max(50, min(2048, 16 * memory_rows))` |
| text encoder | 冻结 MiniLM，使用与构造时一致的 text cache |
| 构造阶段 raw 访问 | 0；本轮只读取已保存 memory 和评估数据 |
| 结果文件 | `results/xsp_student_generalization_20261004_gpu0/`、`gpu1/` |

## 3. Final F1 与 Accuracy

以下为每个数据集最后阶段、6 个 Student seeds 的均值。差值为 `ST-Sinkhorn - Current XSP`。

| 数据集 | Student | Current F1 | ST F1 | ΔF1 | Current Acc | ST Acc | ΔAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| Arxiv | 两层 GCN | 21.39 | 21.85 | **+0.46** | 33.74 | 35.56 | **+1.82** |
| BookChild | 两层 GCN | 13.40 | 13.93 | **+0.53** | 13.61 | 14.40 | **+0.79** |
| SportsFit | 两层 GCN | 29.54 | 29.62 | **+0.08** | 34.09 | 33.85 | -0.25 |
| Arxiv | 单层 GCN | 20.05 | 20.53 | **+0.48** | 29.13 | 30.32 | **+1.19** |
| BookChild | 单层 GCN | 14.73 | 15.78 | **+1.05** | 16.35 | 17.59 | **+1.24** |
| SportsFit | 单层 GCN | 27.40 | 30.61 | **+3.21** | 30.75 | 35.28 | **+4.53** |

## 4. Old/New F1

| 数据集 | Student | ΔOld F1 | ΔNew F1 |
|---|---|---:|---:|
| Arxiv | 两层 GCN | +0.35 | **+1.11** |
| BookChild | 两层 GCN | +0.39 | **+1.57** |
| SportsFit | 两层 GCN | -0.01 | **+0.56** |
| Arxiv | 单层 GCN | +0.59 | -0.17 |
| BookChild | 单层 GCN | +0.96 | **+1.72** |
| SportsFit | 单层 GCN | **+3.28** | **+2.80** |

等权三图平均：

| Student | 平均 ΔF1 | 平均 ΔAccuracy | 平均 ΔOld F1 | 平均 ΔNew F1 |
|---|---:|---:|---:|---:|
| 两层 GCN | +0.36 | +0.79 | +0.24 | +1.08 |
| 单层 GCN | +1.58 | +2.32 | +1.61 | +1.45 |

## 5. 结果说明

### 5.1 方法收益不依赖两层 GCN

Final F1 在两种 Student、三个数据集的 `6/6` 个 cell 都为正。单层 Student 的提升更大，说明 ST-Sinkhorn 并不是只在原始两层 GCN 的参数化下有效。

这里的含义是有限的：它证明了对 Student 结构有一定迁移性，但还没有证明对所有 GNN、所有文本编码器或所有下游任务都泛化。

### 5.2 主要改善了整体可学习性，并有新类正向信号

New F1 在 `5/6` 个 cell 上提升，平均提升约 `+1.08 pp`（两层）和 `+1.45 pp`（单层）。因此 ST-Sinkhorn 不再只是“通过牺牲新类换旧类”的机制；它可以在多数设置下改善新增类别的识别。

### 5.3 SportsFit 的两层 Accuracy 小幅下降

SportsFit 两层 Student 的 F1 `+0.08 pp`，Accuracy `-0.25 pp`。这说明 macro-F1 与 accuracy 对类别不平衡和错误分布的敏感性不同，不能只报告一个指标。单层 Student 在该数据集上两项均明显提升，因此该 cell 更适合被解释为轻微指标 trade-off，而不是方法完全失败。

### 5.4 BookChild 的结论被修正

在独立 seed 的原始两层 Student 确认中，BookChild Final F1 曾为负；冻结同一 memory 换用统一泛化 runner 后，两层 Student 的配对差为 `+0.53 pp`，单层 Student 为 `+1.05 pp`。这说明 BookChild 的结果对 Student 训练随机性和评估实现很敏感，正式论文必须使用同一 runner、同一代码快照重跑 Current 和所有外部基线，不能混合历史结果。

## 6. 主方法冻结判断

本轮支持将以下版本冻结为下一阶段论文主方法：

```text
CaT X initialization
-> current-delta-only X-conditioned graph/text cost
-> class-restricted balanced entropic Sinkhorn
-> straight-through Sinkhorn assignment
-> slot-conditioned real text materialization
-> append-only edge-free synthetic memory
```

冻结的是结构，不是所有论文结论。当前仍需要：

- 同一冻结代码快照下重跑公开/适配基线；
- 至少 3 个独立类别顺序或 construction seed 的稳定性确认；
- 统一报告构造成本、Student 成本和 memory 审计；
- 最好补充第三种轻量 Student 或一个下游任务，作为泛化增强证据。

不建议再加入 anchor、hard consistency、动态预算、复杂 selector 或拓扑 revision 来追求单个数据集分数。它们会改变主方法定义，也会让当前 `6/6` 的配对结论无法归因。

## 7. 文件

- 代码：[evaluate_xsp_student_generalization_20261004.py](../../code/v2_llm_fusion/evaluate_xsp_student_generalization_20261004.py)
- GPU0 结果：`results/xsp_student_generalization_20261004_gpu0/`
- GPU1 结果：`results/xsp_student_generalization_20261004_gpu1/`
- 独立 seed 确认：[XSP_ST_SINKHORN_INDEPENDENT_CONFIRMATION_ZH_2026-10-04.md](XSP_ST_SINKHORN_INDEPENDENT_CONFIRMATION_ZH_2026-10-04.md)
