# Unified-Q-EF 实验推进状态（2026-10-13）

本记录只汇总当前实验状态，不改写已冻结结果，也不用于根据 sealed test 选择方法。

## 已完成

### Q 归因与文本诊断

- shared Q / independent Q；
- learned Q / fixed Q / literal random-Q；
- no-GM、no-DM、no-residual、no-node-text；
- class-shared、random-5、Q-top1、Q-medoid、Q-coverage；
- 硬文本与软文本的一致性诊断；
- slot coverage、assignment entropy、unique text ratio、soft/hard cosine gap、Q/residual 梯度和成本。

当前证据不支持“shared Q 稳定提高分类精度”。它支持的是：shared Q 统一特征和文本来源，并降低分配参数、构造时间和峰值显存；分类优势必须按数据集和 Student 分开报告。

审计与可视化：

- `results/unified_q_overnight_20261012/q_attribution_audit.md`
- `results/unified_q_overnight_20261012/q_attribution_matrix.csv`
- `results/unified_q_overnight_20261012/q_attribution_coverage.png`
- `results/unified_q_overnight_20261012/q_attribution_coverage.pdf`

### 跨架构 Student

冻结同一 memory，使用 Prompt-GraphSAGE 完成 27 个构造单元。该实验显示：

- BookHis 上 Unified-Q 相对 CaT 的 GraphSAGE Final F1 为 `+7.60 pp`，相对 GECC 为 `+1.34 pp`；
- SportsFit 上相对 CaT 为 `-2.12 pp`，相对 GECC 为 `+0.36 pp`；
- WikiCS 上相对 CaT 为 `-1.26 pp`，相对 GECC 为 `+0.92 pp`。

因此跨架构迁移是有证据的，但不是所有数据集都优于基线。完整报告：

`results/uq_text_path_extended_20261009/summary.md`

### 构造与顺序稳定性

主方法、CaT、GECC 的 3 个类别顺序 × 3 个 construction seed 已完成正式矩阵；每个 cell 内 Student seeds `150/151/152` 取均值，再按 cell 计算均值、标准差、配对区间、positive-cell ratio 和 worst cell。

当前 sealed 汇总快照 `results/uq_final_evidence_report_20261013_004216/` 显示：

- ArXiv：9/9 cells 完成；Unified-Q-EF `51.93 F1 / 71.43 Acc`，GECC-hard `51.29 / 71.21`；
- BookHis：9/9 完成；Unified-Q-EF `47.16 / 82.22`，GECC-hard `44.51 / 82.28`；
- SportsFit：8/9 完成，最后一个 cell 仍在运行；当前 Unified-Q-EF `79.33 / 85.46`，GECC-hard `80.35 / 89.28`，这不是最终 SportsFit 表。

正式配对结果应使用同一 order/construction cell，不能用不同完成数量的行均值直接相减。当前已完成部分表明：BookHis 对 GECC 的 F1 增益较稳定，ArXiv 接近持平，SportsFit 的 F1/Accuracy 仍不支持普遍优势。

## 正在重跑的外部基线

旧 TAGSAM/STAD 结果不是简单纳入主表：

- TAGSAM 失败原因是 frozen-prefix 已保留全部旧行时仍调用不存在的 `shrink_by_mass`；已改为只有在确实需要缩减时才调用，并通过回归测试；
- STAD 失败原因是官方内部划分对每个类别强制要求至少 3 个节点；当前仅 BookHis stage 2 的一个类别有 2 个节点；已采用显式的 `classwise-10pct-sparse-v1` 适配，并记录 train/validation/pseudo masks 的互斥性和无外部节点使用；
- TAGSAM BookHis 修正版已经完成并退出码为 0；
- STAD BookHis 修正版正在运行；
- 原始旧 baseline queue 仍在执行 GECC-soft SportsFit，不能把其未完成结果当作完整基线。

修正版代码、回归测试和重跑队列位于：

- `experiments/uq_native_retry_20261013/code/`
- `tools/test_uq_native_retry_20261013.py`
- `tools/audit_uq_sparse_class_split_20261013.py`

## 论文口径

当前可安全表述：Unified-Q-EF 是在 current-delta-only、frozen synthetic prefix、累计 5% 合法构造池和统一 Prompt Student 协议下，将类别约束的可微 Q 分配、in-loop distribution matching、受限特征 residual 与真实文本路由放入同一构造接口的 edge-free continual TAG 方法。它在 BookHis 等长尾大图上对机制适配 GECC 有稳定 F1 信号，并完成了跨 Student 和跨顺序诊断。

当前不能表述：稳定 SOTA、普遍超过 GECC、shared Q 已被证明是分类精度主因、或 TAGSAM/STAD 已经是官方原协议复现。

## 待完成

1. SportsFit 最后一个正式确认 cell；
2. STAD BookHis 修正版及其审计；
3. 等待旧 native queue 结束后，继续其他 TAGSAM/STAD 重跑，或在资源允许时单独启动剩余 retry；
4. 重新采集和汇总快照，生成最终统一表与基线适配说明。

