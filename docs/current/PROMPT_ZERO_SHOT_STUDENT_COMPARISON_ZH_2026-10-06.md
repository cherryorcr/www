# Prompt Zero-Shot GraphText Student 统一重评报告

## 1. 目的

本轮冻结一个通用 GraphText Student，重新评价所有凝聚臂。新 Student 没有可训练分类器头；推理时只计算真实节点图表示与类别 prompt 文本表示的相似度。旧 Student 的历史结果不与本表混合。

## 2. 固定协议

- 数据集：Cora、Citeseer。
- 顺序：`order19611`、`order19612`、`order19613`。
- 三阶段类别增量；每阶段只向构造器提供 current delta 和已冻结 synthetic parent。
- memory rate：5%；所有方法同一预算、同一 source delta、同一 report-dev 规则。
- 凝聚 memory 使用 edge-free 表示，Student 只补 self-loop；因此这张表比较的是 X/text payload，而不是合成跨节点拓扑。
- Student seeds：150--155；每个 cell 6 个 seed。
- construction seed：270927；CaT 3000 steps；XSP 30 steps。
- MiniLM 编码器冻结；训练只更新 GCN、text projection 和 logit scale；推理无 classifier head。

## 3. Student 形式

图分支是两层对称归一化 GCN，文本分支是冻结 MiniLM embedding 加线性投影。训练损失为：

`L = 0.5 L_multi-positive + 1.0 L_prompt-CE + 0.1 L_prompt-margin`。

其中同类 memory 行被视为多正样本，避免同类重复 payload 在旧的行级 InfoNCE 中互相成为负样本；测试时类别原型始终来自 prompt。
这里的 zero-shot 指 prompt-only 分类头：推理阶段没有 learned linear classifier；它不是尚未见过类别的严格跨数据集 unseen-class zero-shot，当前类别 prompt 参与统一训练协议。

## 4. Pooled 结果

以下是六个 `(dataset, order)` 终点 cell 的平均；`Order std` 是三个顺序之间的标准差，`Worst F1` 是最差顺序。

方法臂含义：`fixed_x_xsp` = 固定 CaT X、只使用 XSP 文本；`xsp_x_cattext` = 更新 X、保留 CaT 文本；`full_xsp` = 同时更新 X 和文本。

| Arm | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat_nearest | 76.64 | 7.69 | 79.51 | 76.32 | 77.14 | 68.82 | 6 |
| formal_prompt_margin | 77.09 | 7.57 | 79.77 | 76.88 | 77.39 | 69.50 | 6 |
| gecc_hard_tag | 74.54 | 8.30 | 78.09 | 73.00 | 77.01 | 65.90 | 6 |
| random_real | 69.24 | 7.16 | 74.76 | 67.81 | 71.32 | 60.26 | 6 |
| k_center | 60.09 | 7.44 | 65.36 | 56.92 | 64.30 | 49.39 | 6 |
| kmeans_medoid | 71.45 | 8.87 | 75.15 | 71.41 | 71.29 | 62.91 | 6 |
| fixed_x_xsp | 76.19 | 7.92 | 79.18 | 75.82 | 76.75 | 68.52 | 6 |
| xsp_x_cattext | 76.94 | 7.65 | 79.76 | 76.67 | 77.34 | 69.00 | 6 |
| xsp_st | 76.40 | 7.92 | 79.43 | 76.07 | 76.95 | 68.61 | 6 |
| full_xsp | 76.68 | 7.67 | 79.56 | 76.24 | 77.33 | 69.06 | 6 |

## 5. 解释

`cat_nearest` 的六 cell pooled Final F1 为 76.64。formal prompt-margin 为 77.09，`xsp_x_cattext`（只更新 X）为 76.94，`fixed_x_xsp`（只更新文本）为 76.19，Full XSP 为 76.68。

因此新 Student 没有掩盖 Full XSP 的表现：Full XSP 被原样重跑，并与“只更新 X”和“只更新文本”的两个归因臂分开报告。当前证据更支持 prompt-margin payload assignment 和受限 X refinement，而不是直接把未经约束的 Full XSP 定为最终主方法；slot-conditioned 文本本身在这一统一 Student 下没有形成稳定 pooled 增益。

real-node coreset 的结果显著低于 synthetic arms，说明在该 5% continual memory 协议下，单纯保留真实节点不能替代 class-wise synthetic X/text payload；但 k-center 的极端顺序退化也提示其对类别顺序和小预算很敏感。

## 6. 审计结论

6 个 cell 的 construction prefix、memory hash 和 checkpoint reload audit 全部通过；reload 最大指标误差为 0。新 Student 的结果可以作为独立主表，但不能和旧 Student 历史数值直接横向比较。

## 7. 后续建议

1. 先把 `xsp_x_cattext`、`xsp_st` 和 `formal_prompt_margin` 作为候选主方法做独立 construction seed 确认；不再用 Full XSP 的单次收益替代稳定性证据。
2. 保持 prompt Student 不变，单独补合成拓扑版本；否则当前结果只能证明 payload/X 处理，不能证明完整 TAG 拓扑凝聚。
3. 若要让 XSP 成为完整主方法，应给二次 X 更新加 old-class retention/trust-region 约束，并用同一 prompt Student 重新验证。
