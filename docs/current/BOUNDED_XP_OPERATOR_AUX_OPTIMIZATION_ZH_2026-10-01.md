# Bounded X+P 后续优化实验报告

日期：2026-10-01  
范围：Cora、Citeseer；三阶段；5% cumulative memory；原始两层 GCN + 可训练 text projection + 双向图文 InfoNCE Student。

## 1. 本轮目的

本轮不是继续扫描大量超参数，而是针对当前 Bounded Student-Compatible X+P 的两个明确疑点做机制实验：

1. outer objective 是否应当使用与 Student 相同的候选合成图传播算子；
2. P 的 payload transport 是否应当禁止不同新类别之间的混合。

所有实验都保持以下条件不变：

- 当前类别到来顺序和当前 delta 不变；
- 5% cumulative memory budget 不变；
- Student、report seeds、训练步数和 report-dev 规则不变；
- 历史 raw graph 不回读；
- 不使用 report/test labels 构造 memory；
- 不修改旧 synthetic prefix、旧文本和旧旧边；
- 新输出使用独立目录，不覆盖此前结果。

## 2. 实现改动

### 2.1 Operator-aligned objective

原 X+P 的 outer loss 在 candidate memory 上执行 Student inner unroll，但 outer response 主要在当前 raw support graph 上测量。新增的 operator-aligned 版本在 candidate synthetic graph 上计算 Student propagation，并使用当前 delta 构造的文本目标。

实现位置：

- [direct_student_outer.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/direct_student_outer.py)
- [run_bounded_xpa_unified_20260929.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_bounded_xpa_unified_20260929.py)

第一版 class objective 暴露出一个目标函数错误：同一新类别的多个 synthetic slots 使用相同类别文本均值，却被 instance-level InfoNCE 当作不同正类，因此同类 slots 互相成为负样本。修正后采用 class-level many-to-one objective：

- old rows：保留 instance-level 图文对齐；
- new rows：slot graph representation 对齐到类别文本 prototype；
- 反向项：先对同类 synthetic rows 求 graph prototype，再与类别文本 prototype 做 class-level 对齐；
- prompt margin 仍然只使用当前到达类别的 construction-time class id。

### 2.2 Class-constrained P

P 原本是完整 Sinkhorn transport。新增版本将目标 synthetic slot 和当前 raw delta 按 construction-time class id 分块，在每个类别内部独立 Sinkhorn：

\[
P=\operatorname{blockSinkhorn}(P^{(c)}).
\]

实现还记录跨类 transport mass，并验证其为零。这个约束不是最终标签选择，而是当前 delta 已知类别的构造元数据约束。

### 2.3 Auxiliary operator objective

直接用 operator-aligned objective 替换原 support-response loss 过于激进。因此新增候选版本只把修正后的 class-level operator loss 作为辅助项：

\[
L_{outer}=L_{support}+0.25L_{operator-class}.
\]

X、P、memory budget、Student 和访问边界均不变；辅助权重 `0.25` 在开发 cell 之后固定，不根据 `order19614` 调整。

## 3. 已完成实验结果

### 3.1 最终阶段结果：四种机制诊断

表中数值为 Final Macro-F1，括号内为相对原 X+P 的百分点变化。

| 数据集/顺序 | 原 X+P | operator-aligned 初版 | class-constrained P | 两者联合 | 修正后的 class-level operator |
|---|---:|---:|---:|---:|---:|
| Cora/order19611 | 49.31 | 47.81 (-1.50) | 46.44 (-2.86) | 49.00 (-0.30) | **50.36 (+1.06)** |
| Cora/order19612 | 45.75 | 45.43 (-0.32) | **48.86 (+3.11)** | 45.12 (-0.63) | 45.16 (-0.59) |
| Citeseer/order19611 | 55.94 | **56.23 (+0.30)** | 56.87 (+0.94) | 55.10 (-0.83) | 56.68 (+0.74) |
| Citeseer/order19612 | 54.71 | 54.14 (-0.57) | 53.96 (-0.75) | 52.82 (-1.89) | 52.81 (-1.90) |

结论：

- operator alignment 的思路本身有正向信号，但初版 class-level 目标错误会导致性能下降；
- 修正后在 Cora/order19611 和 Citeseer/order19611 有提升，但 order19612 仍下降；
- hard class-constrained P 没有稳定收益，说明 slot class map 不是可靠的硬 provenance；
- 两个机制直接联合会把两个不稳定因素叠加，不能作为主方法。

### 3.2 Auxiliary operator 结果

辅助版本保留原 X+P support objective，只加入固定 `0.25` 的 class-level operator loss。

| 数据集/顺序 | X+P Final F1 | Auxiliary Final F1 | 差值 | X+P Accuracy | Auxiliary Accuracy | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| Cora/order19611 | 49.31 | **50.90** | **+1.59** | 56.77 | **57.81** | +1.05 |
| Cora/order19612 | 45.75 | **46.69** | **+0.94** | 54.74 | **55.72** | +0.98 |
| Citeseer/order19611 | 55.94 | **55.96** | **+0.03** | 60.50 | **60.97** | +0.47 |
| Citeseer/order19612 | 54.71 | 52.55 | **-2.16** | 61.55 | 59.04 | -2.51 |
| Cora/order19614 | 45.73 | **46.25** | **+0.51** | 53.81 | **53.94** | +0.12 |
| Citeseer/order19614 | 47.84 | **49.71** | **+1.87** | 53.29 | **54.81** | +1.52 |

六个可比 cell 的汇总：

- 正向 cell：`5/6`；
- 平均 Final F1 提升：`+0.46 pp`；
- `order19614` 是未参与辅助权重选择的独立顺序；两个数据集均为正；
- `order19614` construction plan seed `9320/9321/9322`：Cora 三次均高于 X+P，均值 `+0.32 pp`；Citeseer 三次均高于 X+P，均值 `+3.16 pp`。

construction seed 结果：

| 数据集 | X+P 基线 | Aux seed 9320 | Aux seed 9321 | Aux seed 9322 | Aux 均值 | 均值差值 |
|---|---:|---:|---:|---:|---:|---:|
| Cora/order19614 | 45.73 | 46.25 | 45.83 | 46.09 | 46.05 | +0.32 |
| Citeseer/order19614 | 47.84 | 49.71 | 49.93 | 53.38 | 51.00 | +3.16 |

## 4. 与公开方法适配基线的关系

在已有统一 continual TAG 协议的共同 cell 上，Auxiliary 版本相对其它方法的平均差值为：

| 对比方法 | Auxiliary 相对该方法的平均 Final F1 差值 | 说明 |
|---|---:|---|
| CaT-MMD-TAG adapted | -4.35 pp | 仍明显落后，CaT 是当前较强参照 |
| GECC-hard-TAG adapted | +1.94 pp | 平均高于适配版 GECC |
| k-center/FPS | -2.97 pp | 仍落后强 coreset |
| medoid | +1.02 pp | 平均略高 |

这些 external 方法是统一 continual 协议下的机制适配，不是所有方法的官方端到端复现；k-center/medoid 是 real-node coreset，不是 synthetic condensation。因此正确表述仍然是“在部分统一 cell 上优于若干适配基线”，不能写成普遍 SOTA。

## 5. 本轮说明了什么

### 5.1 真正的问题不是简单的“有没有梯度”

X 和 P 的 outer gradient 已经存在；性能取决于梯度所优化的 target 是否与最终 Student 的训练和评价语义一致。初版 operator-aligned 失败的核心原因是 class-level duplicate positives 被错误地当成 instance-level negatives，而不是“梯度方法没有用”。

### 5.2 P 不适合直接硬切类别

synthetic slot 的 class id 来自 budget allocation 和 mass matching，不是真实 raw provenance。硬性禁止跨类 transport 会删除可能有用的共享语义上下文，因此 class-constrained P 单独没有稳定收益。当前不应把它放进主方法。

### 5.3 保留原目标、加入小权重辅助项更合理

原 X+P 的 support-response objective 在部分顺序上已经有效。辅助项只提供候选图上的 class-level consistency，避免破坏原有 response 信号。现有 6 个 cell 和独立 construction seed 已显示稳定正信号，但最差 cell 仍然存在，所以仍需更大范围确认。

## 6. 当前候选主方法

暂定为：

> **Bounded Student-Compatible X+P with Class-Level Operator Auxiliary Loss**

完整流程：

1. 从当前阶段 C0 memory 和 parent synthetic memory 组成 candidate prefix；
2. 只读取当前 raw delta，形成 legal support；
3. 用 class-aware slot allocation 为新增 synthetic rows 分配 construction-time class metadata；
4. 初始化可编辑的新 rows `X` 和可微 payload transport `P`；
5. 用 candidate graph 做 Student inner unroll；
6. 主 outer objective 使用原 support-response InfoNCE；
7. 额外计算 candidate graph 上的 class-level operator consistency；
8. 以固定 `0.25` 加到 outer loss；
9. 通过 validation support checkpoint 选择 outer state；
10. materialize 新 synthetic rows，保持旧 prefix、旧文本和旧旧边不变；
11. 保存 payload、class metadata、memory hash、raw access audit 和 construction cost。

## 7. 下一步建议

当前不建议继续加入 hard class-constrained P、A topology revision、attention、GraphGPS 或大型 response selector。下一步应按以下顺序：

1. 在第三个独立类别顺序上复现 Auxiliary 版本，优先使用已有合法 C0 source；
2. 在至少一个大图上直接迁移固定 `0.25` 版本，不重新扫描权重；
3. 补充 construction time、Student time、GPU memory、outer steps 和 raw access cost；
4. 若大图平均不劣于强 coreset 且最差 cell 不严重下降，再冻结为论文主方法；
5. 若大图不成立，保留 X+P 为主线，把 auxiliary operator 作为机制消融和负结果。

## 8. 代码与结果

- [direct_student_outer.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/direct_student_outer.py)
- [run_bounded_xpa_unified_20260929.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_bounded_xpa_unified_20260929.py)
- [基础 X+P 结果](D:/Aprojects/continue_tag_condensation/results/xp_joint_full_20261001_cora)
- [Auxiliary Cora 结果](D:/Aprojects/continue_tag_condensation/tmp_remote_results/xp_operator_aux025_20261001_cora)
- [Auxiliary Citeseer 结果](D:/Aprojects/continue_tag_condensation/tmp_remote_results/xp_operator_aux025_20261001_citeseer)
- [Auxiliary order19614 Cora 结果](D:/Aprojects/continue_tag_condensation/tmp_remote_results/xp_operator_aux025_order19614_cora_20261001)
- [Auxiliary order19614 Citeseer 结果](D:/Aprojects/continue_tag_condensation/tmp_remote_results/xp_operator_aux025_order19614_citeseer_20261001)
- [单元测试](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/test_direct_student_outer.py)

本轮单元测试：`35 passed`。
