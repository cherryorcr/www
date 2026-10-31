# Direct Student Outer：Slot Representation 与 Payload-Mass 聚合实验报告

日期：2026-09-29  
状态：已完成四个小图 cell 的构造、下游 Student 评估和实现审计。

## 1. 本轮目标

本轮继续沿着 `x-provenance` 的最小修改路线推进，回答两个问题：

1. 类内 provenance slot 是否还需要用自己的 Student 图表示对齐 raw subgroup 表示；
2. 当前 slot 外层梯度是否因为等权平均而违背 synthetic payload mass。

所有实验均只更新新增 synthetic rows 的 `X`。不修改旧 rows、文本 embedding、文本内容、拓扑、mass、预算或下游 Student。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | canonical、order19102 |
| cell 数量 | 4 |
| old rows | Cora canonical 25；Cora order19102 35；Citeseer canonical 38；Citeseer order19102 40 |
| Student | canonical 两层 GCN + 可训练 text projection + 双向图文 InfoNCE |
| outer construction | Student inner unroll + outer Adam |
| inner / outer steps | 2 / 2，另有一组 inner=8 的 horizon 对照 |
| outer learning rate | 0.01 |
| old/new support weight | 0.25 / 1.0 |
| X trust-region | old 0.05；new 0.10 |
| construction seeds | train 9300；validation 9310；plan 9320 |
| downstream Student seeds | 150--155 |
| 历史 raw 访问 | 0；只使用 previous synthetic memory + current raw delta support |
| report/test labels | 未用于构造或 checkpoint 选择 |

所有 candidate 都与同一 cell 的 C0 control 配对评估，表中差值单位为百分点。

## 3. 方法变化

### 3.1 `x-slot-rep`

在已有 provenance slot 路由上增加：

\[
L_{slot-rep}=1-\cos\left(h_{slot},\operatorname{sg}(\bar h_{raw,slot})\right).
\]

其中 `h_slot` 是 candidate synthetic graph 上该 slot 的 Student 图表示，`h_raw,slot` 是同一 provenance slot 分配到的 current-delta raw subgroup 表示，`sg` 表示 stop-gradient。总目标为：

\[
L=L_{slot\text{-}InfoNCE}+0.25L_{slot-rep}+L_{anchor}.
\]

该版本的意图是减少整类中心过粗的问题，但没有改变拓扑和 Student 评价协议。

### 3.2 `x-provenance + inner=8`

只把 Student inner unroll 从 2 步延长到 8 步，验证短 horizon 是否是主要瓶颈。其它设置完全不变。

### 3.3 `x-provenance + slot-mass-weighted`

当前 provenance 外层训练把每个 slot 的 loss 等权平均。由于 slot mass 差异很大，这会使小 slot 被放大。新版本使用：

\[
L_{outer}=\frac{\sum_j m_jL_j}{\sum_jm_j},
\]

其中 `m_j` 是冻结的 synthetic slot mass。它只改变 slot-routed outer loss 的聚合，不改变 provenance assignment 和 candidate materialization。

## 4. 主结果

### 4.1 四 cell 平均

| 方法 | Final Macro-F1 | Accuracy | New F1 | Old F1 | Final F1 正向 cell |
|---|---:|---:|---:|---:|---:|
| `x-provenance r=0.10` | -0.028 | -0.638 | **+0.721** | -0.925 | 3/4 |
| `x-slot-rep r=0.10` | +0.026 | -0.616 | -1.161 | +0.624 | 2/4 |
| `x-provenance inner=8` | **+0.297** | **+0.379** | -0.798 | **+1.207** | 3/4 |
| `x-provenance mass-weighted` | +0.057 | +0.327 | -0.420 | +0.749 | 2/4 |

结论：没有一个版本同时在四个 cell 上稳定改善 Final F1 和 New F1。`inner=8` 的平均 Final F1 最高，但主要来自旧类保持，不能解释为解决了新类塑性。

### 4.2 `x-slot-rep` 逐 cell

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.770 | -0.493 | -2.690 | +1.111 |
| Cora / order19102 | +0.904 | -1.333 | +1.470 | -1.240 |
| Citeseer / canonical | +0.407 | +0.339 | -1.149 | +1.671 |
| Citeseer / order19102 | -0.438 | -0.976 | -2.276 | +0.952 |

它只有一个 cell 的 New F1 为正。额外的 slot representation alignment 没有转化为稳定的新类 plasticity。

### 4.3 `x-provenance inner=8` 逐 cell

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.329 | +0.493 | -1.636 | +1.080 |
| Cora / order19102 | +0.150 | -0.083 | -0.298 | -0.257 |
| Citeseer / canonical | +0.361 | +0.339 | -1.978 | +2.930 |
| Citeseer / order19102 | +1.004 | +0.767 | +0.722 | +1.076 |

延长 unroll 改善了 3/4 cell 的 Final F1，但只有 1/4 cell 的 New F1 为正。因此它更像 retention 方向，而不是塑性解决方案。

### 4.4 `x-provenance mass-weighted` 逐 cell

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.492 | +0.411 | -0.609 | +0.702 |
| Cora / order19102 | +0.186 | +0.250 | -0.935 | +0.597 |
| Citeseer / canonical | -0.616 | -0.745 | -1.990 | +0.872 |
| Citeseer / order19102 | +1.149 | +1.395 | +1.852 | +0.825 |

质量加权让 outer update 更符合 payload conservation，并使 Old F1 在四个 cell 都为正，但 New F1 仍只有 1/4 cell 为正。

## 5. 结果解释

### 5.1 不是单纯的 slot 粒度问题

`x-provenance` 已经把类内响应拆到 slot；`x-slot-rep` 再增加 raw subgroup representation alignment 后，New F1 反而从 `+0.721` 降到 `-1.161`。这说明问题不是“只要把 slot 划得更细就能解决”。

更可能的原因是：

- raw subgroup representation 使用 support graph 的传播算子；
- candidate synthetic slot 使用 synthetic graph 的传播算子；
- slot alignment 会把 X 推向一个短期 raw response target，但最终评价使用完整真实图传播和独立 Student；
- 额外 alignment 与跨 slot 的 InfoNCE negatives 发生竞争。

### 5.2 inner horizon 不是唯一瓶颈

inner=8 带来较好的 Old F1 和 Accuracy，但 New F1 仍为负。因而“把 outer unroll 再做长”不能单独解决 candidate support objective 与最终下游 classification 的错位。

### 5.3 payload-mass 加权是合理修正，但不是性能答案

等权 slot aggregation 确实存在解释问题：mass=2 的 slot 和 mass=57 的 slot 对 outer gradient 贡献相同。mass-weighted 修正后旧类保持更稳定，说明聚合方式影响 retention；但新类仍未稳定改善，说明 plasticity 瓶颈还在 X 与固定拓扑、真实传播过程之间的关系上。

### 5.4 trajectory response 也不能直接当 selector

已有 trajectory diagnostic 在不同 cell、不同 checkpoint 上的 response/F1 相关性方向不稳定。部分 cell 在较长 horizon 上出现正相关，但 pooled 结果没有稳定的跨 cell 预测关系。因此不能直接用 response loss 选择 candidate 或控制预算。

## 6. 当前决策

本轮不晋升以下版本为主方法：

- `x-slot-rep`：没有稳定 New F1 收益；
- `x-provenance inner=8`：收益主要来自 Old F1，且构造成本增加；
- `x-provenance mass-weighted`：payload 解释更好，但没有解决 plasticity。

目前可以保留的结论是：

1. provenance routing 是有解释力的 X-only 诊断组件；
2. X 外层梯度确实能改变 synthetic memory，但短期 support loss 不能可靠预测最终下游收益；
3. old/new trade-off 不是简单调 radius、slot representation 或 mass aggregation 就能稳定解决；
4. 当前主方法仍应保持已有冻结版本，不能因为单个 Cora 或 Citeseer cell 的正差值切换主线。

## 7. 后续推进建议

下一步不再继续叠加 X-only proxy loss。优先级应为：

1. 保留 `x-provenance` 和 mass-weighted 结果作为负结果与机制分析；
2. 对候选 X 使用与最终 Student 一致的真实 full-graph propagation 做 read-only response diagnosis；
3. 若仍不能预测最终 F1，停止 response selector 路线；
4. 重新研究有界的 synthetic topology / cross-edge transport，使更新后的 X 与传播结构一致；
5. 只有在独立类别顺序上出现稳定 New F1 和 Final F1 正收益后，才考虑将该组件并入完整主方法。

不建议现在继续扫描更多 radius、更多 loss weight 或更多短 horizon 组合。这些实验会增加局部 cell 选择风险，不能解决已经识别出的传播算子错位。

## 8. 产物

- 实现：[direct_student_outer.py](../../code/v2_llm_fusion/direct_student_outer.py)
- CLI：[run_direct_student_outer.py](../../code/v2_llm_fusion/run_direct_student_outer.py)
- provenance 映射：[prepare_provenance_x_support.py](../../code/v2_llm_fusion/prepare_provenance_x_support.py)
- `x-slot-rep` 构造：`results/direct_student_outer_20260928/slot_rep_x_v1b_*`
- `x-slot-rep` 评估：`results/direct_student_outer_20260928/eval_slot_rep_x_v1b_*.json`
- inner=8 构造：`results/direct_student_outer_20260928/provenance_x_long8_v1_*`
- inner=8 评估：`results/direct_student_outer_20260928/eval_provenance_x_long8_v1_*.json`
- mass-weighted 构造：`results/direct_student_outer_20260928/provenance_x_mass_v1_*`
- mass-weighted 评估：`results/direct_student_outer_20260928/eval_provenance_x_mass_v1_*.json`

