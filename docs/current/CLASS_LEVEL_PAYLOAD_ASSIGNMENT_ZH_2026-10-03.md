# 类级文本 Payload Assignment 形式化实验报告

> **2026-10-03 修复状态：** 本文早期第 5 节和第 8.5 节使用过错误的 CCTP 文本映射（把 `cat_mmd_tag_cctp_top5` 当成 `text_center`），因此其中的旧 CCTP 和旧 formal prompt-margin 数值只保留为历史诊断，不能作为论文主表。修复后的正式结果见 [Formal Prompt-Margin Assignment 修复与统一基线比较](D:/Aprojects/continue_tag_condensation/results/formal_prompt_margin_nearest_20261003/comparison.md)。

## 1. 实验动机

当前 `class-shared real payload` 已经相对 CCTP 和多个适配基线产生了稳定正向信号，但原实现中每个类别主要由一个 X-space representative 的 `tagsam_chunk` 候选决定。这个规则能够工作，却难以回答：

- 为什么这个候选比同类的其它真实文本候选更好；
- 为什么代表行可以代表该类别的全部合成行；
- 文本语义、X 的图空间一致性和类内覆盖之间如何权衡。

因此本轮不重新设计 CaT 的 X，也不修改 Student、拓扑或历史 memory，而是把文本选择改成一个显式的类级 payload assignment。

## 2. 形式化目标

对当前阶段类别 `c`，令：

- `R_c`：memory 中属于类别 `c` 的全部合成行；
- `X_j`：第 `j` 行的 CaT 合成特征；
- `p`：一个最终可以真实材料化并编码的文本 payload；
- `G_p`：构成 payload 的真实源节点图特征均值；
- `T_p`：payload 的冻结 MiniLM embedding；
- `C_c`：当前 delta 中类别 `c` 的文本中心。

候选 payload 通过以下目标选择：

\[
J_c(p) = d_{graph}(p,c) + \lambda_t d_{text}(p,c)
       + \gamma v_{coverage}(p,c),
\]

其中：

\[
d_{graph}(p,c) = 1 - \frac{1}{|R_c|}
\sum_{j\in R_c}\cos(X_j,G_p),
\]

\[
d_{text}(p,c) = 1 - \cos(T_p,C_c),
\]

\[
v_{coverage}(p,c) = \operatorname{Var}_{j\in R_c}
\left[\cos(X_j,G_p)\right].
\]

本轮固定：

\[
\lambda_t=0.50, \qquad \gamma=0.25.
\]

解释如下：

- 第一项要求文本 payload 的真实来源与该类别全部合成 X 尽可能一致；
- 第二项要求 payload 保持类别文本语义，而不是只追求图相似度；
- 第三项惩罚只适合代表行、但不能覆盖同类其它合成行的候选。

这不是对最终 F1 的 selector，也没有使用 report/test 标签；它只使用当前 delta、当前合成 X 和冻结文本 embedding。

## 3. 候选池如何产生

候选空间不做组合穷举。对每个类别先选一个 X-space representative，仅用于建立有界候选池；然后用该 representative 的当前 delta 支持池生成以下最多五类真实 payload：

1. `cctp`：当前 CCTP top-5 文本；
2. `text_mmr`：文本中心和对比语义驱动的 chunk payload；
3. `tagsam_chunk`：覆盖支持文本并惩罚冗余的 chunk payload；
4. `graph_role`：图相似度、传播 role 和文本对比项共同选择的 payload；
5. `anchor_mix`：保留 CCTP anchor，再补充真实 chunk。

每个候选都是真实字符串，最终用冻结 MiniLM 重新编码，并保存源节点 ID 和 chunk provenance。候选生成复杂度随类别数和 bounded support pool 增长，不随所有文本组合数指数增长。

关键区别在于：representative 只负责产生候选池，最终 `J_c(p)` 会在该类别的全部 `R_c` 上计算，然后只选一个 payload 广播给同类合成行。

## 4. 实现位置

核心实现：

- [cat_multimodal_minimal.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/cat_multimodal_minimal.py:814) 中的 `attach_class_assigned_chunk_payload`；
- [run_public_continual_baselines_large_20261001.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py:349) 中的 `cat_mmd_tag_class_assigned` arm；
- [summarize_class_assignment_large_20261003.py](D:/Aprojects/continue_tag_condensation/tools/summarize_class_assignment_large_20261003.py) 负责协议、审计和 paired 汇总。

构造审计会保存：每个类别的候选 objective、graph cost、text cost、coverage variance、选中的 mode、源节点 ID 和文本 chunk provenance。

## 5. 三图严格统一协议结果

协议为：base+4、累计 5% memory、相同 edge-aware 两层 GCN Student、冻结 MiniLM、Student seeds 100--105、当前 delta-only、无历史 raw 回读、无 report/test 标签。

### 5.1 最终结果

| 数据集 | 方法 | Final F1 | Accuracy | Stage avg F1 | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|
| arXiv | CCTP top-5 | 24.31 | 48.15 | 24.85 | 27.58 | 5.81 |
| arXiv | 启发式 class-shared | 23.57 | 43.89 | 25.43 | 26.18 | 8.74 |
| arXiv | **formal class-assigned** | **25.77** | **49.58** | **28.36** | **29.30** | 5.76 |
| BookChild | CCTP top-5 | 4.16 | 31.80 | 4.06 | 4.71 | 0.33 |
| BookChild | 启发式 class-shared | **6.06** | 19.02 | **8.15** | **6.36** | **4.01** |
| BookChild | formal class-assigned | 2.80 | 4.85 | 6.51 | 2.90 | 2.15 |
| SportsFit | CCTP top-5 | 22.08 | 36.21 | 28.09 | 22.07 | 22.12 |
| SportsFit | 启发式 class-shared | 45.74 | 58.90 | 60.51 | 49.19 | 26.77 |
| SportsFit | **formal class-assigned** | **53.79** | **71.83** | **66.01** | **57.80** | **31.74** |

### 5.2 三图等权平均

| 方法 | Final F1 | Accuracy | Stage avg F1 | Old F1 | New F1 | 最差 stage |
|---|---:|---:|---:|---:|---:|---:|
| CCTP top-5 | 16.85 | 38.72 | 19.00 | 18.12 | 9.42 | 16.46 |
| 启发式 class-shared | 25.12 | 40.60 | 31.36 | 27.24 | 13.17 | 24.23 |
| **formal class-assigned** | **27.46** | **42.09** | **33.63** | **30.00** | **13.22** | **26.63** |

### 5.3 Paired 差值

formal assignment 相对启发式 class-shared：

| 指标 | 平均差值 | 正向数据集 |
|---|---:|---:|
| Final F1 | **+2.33** | 2/3 |
| Accuracy | +1.48 | 2/3 |
| Stage avg F1 | **+2.26** | 2/3 |
| Old F1 | **+2.76** | 2/3 |
| New F1 | +0.05 | 1/3 |

formal assignment 相对 CCTP：

| 指标 | 平均差值 | 正向数据集 |
|---|---:|---:|
| Final F1 | **+10.60** | 2/3 |
| Accuracy | +3.36 | 2/3 |
| Stage avg F1 | **+14.63** | 3/3 |
| Old F1 | **+11.88** | 2/3 |
| New F1 | **+3.80** | 2/3 |

## 6. 选择行为和成本

formal assignment 在 15 个阶段中累计完成了 77 个类别级选择。选中的候选 mode 为：

| mode | 次数 |
|---|---:|
| `anchor_mix` | 26 |
| `cctp` | 25 |
| `text_mmr` | 24 |
| `tagsam_chunk` | 1 |
| `graph_role` | 1 |

这说明新目标没有简单复现原来固定的 `tagsam_chunk`；它主要在 CCTP anchor、文本 MMR 和 anchor-mix 之间做了类级选择。

平均成本：

| 方法 | 构造秒 | Student 秒/seed |
|---|---:|---:|
| CCTP top-5 | 195.65 | 24.37 |
| 启发式 class-shared | 184.22 | 24.56 |
| formal class-assigned | 197.45 | 24.48 |

formal assignment 没有明显增加 Student 训练成本；构造成本与 CCTP 接近。

## 7. 结果分析

### 7.1 形式化确实有价值

相对旧启发式方法，formal assignment 在 arXiv 和 SportsFit 上同时提升 Final F1、Stage-average F1 和 Old F1；三图平均 Final F1 提升 `+2.33 pp`。这说明原来的 class-shared 思路不是只能靠经验规则，类级 assignment 目标能够产生额外收益。

### 7.2 它还没有解决新类塑性

formal assignment 相对启发式方法的平均 New F1 只有 `+0.05 pp`，而且只有 1/3 数据集正向。formal objective 主要改善了旧类保持和阶段平均表现，没有稳定改善新类塑性。

### 7.3 BookChild 暴露了当前目标的不足

BookChild 上 formal assignment 从启发式的 `6.06` 降到 `2.80`。当前最可能的原因不是公式无法计算，而是：

1. 单个 class-shared payload 压缩掉了 BookChild 类内文本差异；
2. `G_p` 使用 payload 源节点图特征均值，不能完整表达 source text 与合成 X 的跨模态对应关系；
3. `d_text` 对齐类别文本中心，容易选择通用但不具有边界区分性的文本；
4. 候选池主要由一个 representative 生成，目标虽然评估全部 `R_c`，但可能从一开始就没有看到适合类内其它区域的候选；
5. coverage variance 只描述相似度方差，没有直接约束新类与相邻旧类的 margin。

因此，形式化 assignment 是有效组件，但还不能直接称为对所有数据集稳定有效的最终方法。

## 8. 候选覆盖与多 payload 后续实验

为判断 BookChild 的失败究竟来自候选池覆盖不足，还是来自“每个类别只能共享一个 payload”的假设，在不改变 CaT 的 `X`、Student、拓扑、memory rows 和文本预算的前提下，又完成了两个最小改动实验：

1. **双 anchor assignment**：每个类别同时使用 X-space 中心行和类内最远行建立候选池，但最终仍只选择一个 payload；
2. **高离散度 two-payload assignment**：先计算当前阶段各类别合成 `X` 到类中心的离散度，只有离散度不低于阶段中位数的类别允许两个 payload；从有界候选池中选择一对，使每个合成行获得图兼容性更高的 payload。低离散度类别仍保持一个 payload。

这两个实验都不访问历史 raw，不使用 report/test 标签，不修改旧 memory 或拓扑，也不增加 memory rows；two-payload 只改变同一批合成行看到的真实文本内容。

### 8.1 三图结果

| 数据集 | Formal single | Formal two-anchor | Formal two-payload | 启发式 class-shared |
|---|---:|---:|---:|---:|
| arXiv Final F1 | 25.77 | **26.69** | 25.25 | 23.57 |
| BookChild Final F1 | 2.80 | 4.17 | 3.88 | **6.06** |
| SportsFit Final F1 | **53.79** | 51.75 | 51.08 | 45.74 |
| 三图等权平均 | **27.46** | 27.54 | 26.74 | 25.12 |

| 方法 | 三图平均 Accuracy | 三图平均 Stage F1 | 三图平均 Old F1 | 三图平均 New F1 | 平均构造秒 |
|---|---:|---:|---:|---:|---:|
| Formal single | **42.09** | **33.63** | 30.00 | **13.22** | 197.4 |
| Formal two-anchor | 37.99 | 32.88 | **30.46** | 11.13 | 211.4 |
| Formal two-payload | 40.09 | 32.13 | 29.34 | 12.17 | 214.4 |

### 8.2 结论

- 双 anchor 相对 single 的 Final F1 平均只增加 `+0.08 pp`，但 New F1 下降 `-2.08 pp`，Accuracy 下降 `-4.10 pp`，且 SportsFit 下降 `-2.04 pp`。候选覆盖不足不是唯一瓶颈。
- two-payload 在 BookChild 比 single 提高 `+1.08 pp`，但仍比启发式 class-shared 低 `-2.18 pp`；在 arXiv 和 SportsFit 分别低 `-0.53` 和 `-2.71 pp`。简单按 X 离散度拆分 payload 不能成为主方法。
- 因此，当前最可靠的版本仍是 **formal single class-assigned**。双 anchor 和 two-payload 应保留为负结果和机制诊断，不晋升为主方法。
- 这轮结果把问题定位得更具体：BookChild 的失效不只是“一个代表行覆盖不足”，更可能是 `G_p` 的源图均值、类别文本中心距离和下游图文 InfoNCE 目标之间不一致。继续增加 payload 数量不会自动修复这种跨模态错配。

### 8.3 访问与实现审计

双 anchor 和 two-payload 的 15 个阶段记录均满足：历史 raw 读取为 `0`、report/test 标签使用为 `0`、旧 memory 更新为 `False`、拓扑更新为 `False`。Student 训练步数和 memory rows 与 single assignment 完全一致；额外成本只来自候选池扩展和候选对评分。

### 8.4 固定 prompt-margin objective

在确认“增加 payload 数量”不是充分修复后，又完成了一个只修改 assignment objective 的版本。它在原目标上增加：

\[
L_{margin}(p)=\max\left(0,m-s(T_p,q_c)+\max_{c'\ne c}s(T_p,q_{c'})\right),
\]

其中 `q_c` 是类别 prompt 的冻结 MiniLM 表示，`m=0.05`，权重固定为 `0.15`。最终候选选择目标变为：

\[
J'_c(p)=J_c(p)+0.15L_{margin}(p).
\]

它不改变 `X`、候选池、文本长度、Student 或 memory rows，只要求真实 payload 在当前类别 prompt 与最相似的其它类别 prompt 之间保留边界。

### 8.5 三图结果（construction seed 270927）

| 数据集 | Formal single | Prompt-margin assignment | 差值 |
|---|---:|---:|---:|
| arXiv Final F1 | 25.77 | **26.66** | **+0.89** |
| BookChild Final F1 | 2.80 | 3.52 | +0.71 |
| SportsFit Final F1 | 53.79 | **54.05** | **+0.26** |
| 三图等权平均 | 27.46 | **28.08** | **+0.62** |

Prompt-margin 相对 formal single 的三图平均变化为：Final F1 `+0.62 pp`、Stage-average F1 `+1.63 pp`、Old F1 `+1.09 pp`，但 New F1 `-2.08 pp`，Accuracy `-2.21 pp`。因此它是目前最有希望的 objective 改进，但还不是已经证明普遍改善的新类塑性方法。

它相对启发式 class-shared 的平均 Final F1 提升 `+2.95 pp`，在 arXiv 和 SportsFit 上分别提升 `+3.09` 和 `+8.31 pp`，但 BookChild 仍低 `-2.55 pp`。这说明 prompt 边界约束能改善部分图文匹配，却不能消除 BookChild 的数据集失配。

独立 construction seed `270928` 的 paired confirmation 已启动，但因远端根分区耗尽而在保存中间 memory 时中止，不能把它当作完成的独立确认结果。已有 seed 的结果只能作为候选正向证据，不能写成稳定提升。

## 9. 后续推进顺序

建议保留 formal single assignment，不把双 anchor 或 two-payload 直接并入主线；下一轮按以下顺序做：

1. **独立 seed 复现**：优先补 BookChild 和 SportsFit，固定公式与权重，不根据最终 F1 选择参数。
2. **不再继续扩展 payload 数量**：two-anchor 和 two-payload 已经回答了覆盖假设，继续增加候选数量的预期收益很低。
3. **先完成独立 seed/顺序确认**：清理远端磁盘后，优先恢复 `270928` 的 paired single vs prompt-margin；不得根据当前单 seed 结果冻结。
4. **若独立确认仍为正**：再做一个不调权重的 message-compatibility 对照，验证 prompt margin 的收益是否来自真正的图文消息一致性，而非 prompt-only 偏置。
5. **若独立确认不稳定**：保留 formal single 作为主方法候选，prompt-margin 作为可解释的正向/负向边界实验，不再继续堆叠 payload 模块。
6. **最终冻结标准**：至少两个独立顺序/seed 组上候选相对启发式和 CCTP 均不劣，再把它晋升为论文主方法中的正式 operator。

## 10. 结果文件

- [正式汇总](D:/Aprojects/continue_tag_condensation/results/class_assignment_large_summary_20261003/summary.md)
- [正式 JSON](D:/Aprojects/continue_tag_condensation/results/class_assignment_large_summary_20261003/summary.json)
- [完整逐阶段结果](D:/Aprojects/continue_tag_condensation/results/class_assignment_large_summary_20261003/stage_rows.csv)
- [Formal 与基线差值](D:/Aprojects/continue_tag_condensation/results/class_assignment_large_summary_20261003/formal_vs_baselines.csv)
- [双 anchor 与 two-payload 统一汇总](D:/Aprojects/continue_tag_condensation/results/class_assignment_payload_variants_summary_20261003/summary.md)
- [全部 payload 变体统一汇总](D:/Aprojects/continue_tag_condensation/results/class_assignment_payload_all_variants_summary_20261003/summary.md)
- [双 anchor 汇总](D:/Aprojects/continue_tag_condensation/results/class_assignment_two_anchor_summary_20261003/summary.md)
- [two-payload 实现](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/cat_multimodal_minimal.py:814)
- [统一汇总脚本](D:/Aprojects/continue_tag_condensation/tools/summarize_class_assignment_large_20261003.py)

## 11. 修复版正式主表

修复内容是：CCTP top-5 和 formal assignment 均采用逐合成行的 graph-nearest 真实文本；`text_center` 仅作为显式 class-shared 控制。修复版仍使用 base+4、累计 5% memory、六个 Student seeds、current-delta-only 和相同的 edge-aware GCN Student。

| 方法 | arXiv Final F1 | BookChild Final F1 | SportsFit Final F1 | 三图平均 F1 |
|---|---:|---:|---:|---:|
| Append-edge | 18.82 ± 1.30 | 12.60 ± 2.04 | 26.89 ± 2.20 | 19.44 |
| X+P | 19.39 ± 1.10 | 13.30 ± 1.54 | 32.07 ± 3.51 | 21.59 |
| Random real-node coreset | 17.81 ± 1.33 | 12.37 ± 1.79 | 27.44 ± 7.48 | 19.21 |
| K-center real-node coreset | 19.43 ± 1.04 | 12.49 ± 2.80 | 29.42 ± 3.89 | 20.45 |
| Medoid real-node coreset | 18.06 ± 1.68 | 12.98 ± 1.43 | 27.00 ± 4.53 | 19.35 |
| GECC-TAG hard | 20.23 ± 1.09 | 14.28 ± 0.96 | 26.44 ± 2.18 | 20.31 |
| CaT-TAG CCTP top-5（修复后 nearest） | 19.15 ± 1.55 | 13.33 ± 1.51 | 27.57 ± 2.76 | 20.02 |
| CaT-TAG class-shared real payload | 24.79 ± 1.03 | 5.46 ± 1.05 | 46.83 ± 4.44 | 25.69 |
| **Formal prompt-margin assignment（修复后）** | **25.19 ± 0.93** | **3.87 ± 0.51** | **55.58 ± 4.94** | **28.21** |

修复版候选在 arXiv 和 SportsFit 上超过本表中的基线，但在 BookChild 上低于所有主要基线。三图平均最高不能掩盖这个失败，因此当前结论应是“修复后的强候选”，而不是“稳定优于所有方法”。完整的 Accuracy、Stage-average F1、Old/New F1、构造成本和修复前后对照见上述正式比较报告。
