# Provenance-aware X 外层优化实验报告

日期：2026-09-29  
状态：四个小图 cell 已完成 `r=0.05` 与 `r=0.10` 两组构造、配对评估和审计。

## 1. 本轮问题

上一轮 `x-class` 只按类别路由梯度：同一类别内的所有 synthetic rows 仍接收同一个类级响应。本轮继续只更新新 synthetic rows 的 `X`，但在每个类别内部建立 construction-time provenance slot，将当前 raw delta 分配到 synthetic slot，并让每个 slot 只接收自己的 Student 外层响应。

本轮保持不变：

- Student：canonical 两层 GCN + 可训练 text projection + 双向图文 InfoNCE；
- `P`、文本 embedding、拓扑 `A`、mass、depth、stage、总 rows；
- 旧 synthetic rows；
- inner Student 2 步、outer Adam 2 步、outer learning rate `0.01`；
- old/new support loss 权重 `0.25/1.0`；
- 下游 Student seeds `150--155`；
- 历史完整 raw graph 不回读；
- 只使用 construction-time 的当前 delta support，不使用 report/test 标签。

## 2. Provenance slot 如何构造

对每个新类，使用当前 delta raw 行与新 synthetic slot 的联合图文余弦相似度：

```text
joint(raw)  = normalize([normalize(X_raw), normalize(T_raw)])
joint(slot) = normalize([normalize(X_slot), normalize(T_slot)])
```

随后按 descending cosine 做容量约束分配。每个 synthetic slot 的容量由已有 `mass` 给出，所有 raw 行必须恰好分配一次，因此满足：

```text
sum(slot mass) = current raw delta rows
```

这是一种 construction-time 的责任分配，不是声称恢复了历史 raw graph 中真实的来源关系。它不访问历史完整 raw graph，也不修改 `P`、文本或拓扑。

每个 slot 使用独立的 `x_provenance_relative` 变量。外层训练时，对 slot `j` 只计算属于 slot `j` 的 raw support target；其它封存 support 行只作为 contrastive negatives。旧 rows 保持固定。由于极小 slot 可能没有独立的平方 InfoNCE batch，checkpoint 选择使用全局封存 validation objective。

## 3. `r=0.05` 结果

所有差值均为 candidate 相对同 cell `control/C0` 的百分点差值。

| 数据集 / 顺序 | Final F1 | Accuracy | New F1 | Old F1 | 正向 Student seed 数 |
|---|---:|---:|---:|---:|---:|
| Cora / canonical | -2.096 | -1.067 | -3.689 | +0.996 | 2/6 |
| Cora / order19102 | **+3.559** | **+1.917** | **+4.592** | +0.825 | 5/6 |
| Citeseer / canonical | -0.695 | -0.068 | -3.140 | **+2.520** | 2/6 |
| Citeseer / order19102 | -0.419 | -0.837 | -0.520 | -1.051 | 2/6 |
| **四 cell 平均** | **+0.087** | **-0.014** | **-0.687** | **+0.822** | - |

逐 seed Final F1 差值：

- Cora/canonical：`-0.221, +1.546, +0.555, -6.365, +0.408, -8.500`；
- Cora/order19102：`-4.814, +8.173, +0.846, +7.489, +2.119, +7.540`；
- Citeseer/canonical：`+0.523, -0.126, -4.202, -0.087, +1.462, -1.742`；
- Citeseer/order19102：`+2.910, +0.679, -4.500, -1.691, +2.173, -2.072`。

注：报告主结论使用 cell 均值；逐 seed 列只用于显示方差。原始逐 seed 数据保存在对应 `eval_*.json` 中。

## 4. `r=0.10` 结果

| 数据集 / 顺序 | Final F1 | Accuracy | New F1 | Old F1 | 正向 Student seed 数 |
|---|---:|---:|---:|---:|---:|
| Cora / canonical | **+0.668** | **+0.246** | **+1.076** | +0.426 | 4/6 |
| Cora / order19102 | **+0.635** | -0.500 | +0.877 | -0.003 | 2/6 |
| Citeseer / canonical | -1.926 | -2.507 | -1.684 | -1.223 | 1/6 |
| Citeseer / order19102 | +0.511 | +0.209 | **+2.614** | -2.901 | 4/6 |
| **四 cell 平均** | **-0.028** | **-0.638** | **+0.721** | **-0.925** | - |

逐 seed Final F1 差值：

- Cora/canonical：`+2.447, -1.291, -0.523, -3.904, +6.293, +0.985`；
- Cora/order19102：`-2.318, -4.637, -1.029, +8.507, -2.186, +5.474`；
- Citeseer/canonical：`-2.855, -1.675, -3.228, -2.392, +0.466, -1.874`；
- Citeseer/order19102：`+5.964, -2.468, -5.752, -0.576, +4.160, +1.741`。

## 5. 与前两种 X 外层版本比较

| 方法 | 四 cell 平均 Final F1 | 四 cell 平均 New F1 | 四 cell 平均 Old F1 | 结论 |
|---|---:|---:|---:|---|
| X-only | -0.931 | -1.479 | -0.744 | 梯度路径真实存在，但类间梯度混合且 proxy 错位 |
| X-class | -0.326 | -1.302 | +0.717 | 类级路由改善旧类保持，但 canonical 仍退化 |
| X-provenance, `r=0.05` | +0.087 | -0.687 | +0.822 | 更偏保持，最终平均仅接近零 |
| X-provenance, `r=0.10` | -0.028 | **+0.721** | -0.925 | 更偏塑性，旧类损失明显 |

这些数字不能解释为 `r=0.05` 或 `r=0.10` 已经优于 C0。`r=0.05` 的平均 Final F1 被 Cora/order19102 的单个强正 cell 拉高，但只有 1/4 cell 为正；`r=0.10` 的 New F1 平均为正，却以 Old F1 下降换取，且 Citeseer/canonical 明显退化。

## 6. 结果说明

### 6.1 类内 provenance 路由确实解决了一个局部问题

相对 `x-class`，provenance 版本进一步把同一类内部的 support 响应拆到 synthetic slot。两组半径都显示：类内拆分可以在某些顺序上恢复新类响应，尤其 `r=0.10` 的四 cell 平均 New F1 为 `+0.721 pp`，高于 `x-class` 的 `-1.302 pp`。

### 6.2 但它没有解决 outer proxy 与最终 Student 的错位

外层优化仍然是：

```text
synthetic candidate -> 2-step Student unroll -> support graph-text InfoNCE
```

最终评价仍然是：

```text
独立 Student -> 真实 full graph propagation -> class-prompt classification
```

因此 slot 路由只能减少梯度抵消，不能保证短 horizon support InfoNCE 的下降会转化为最终分类收益。

### 6.3 半径体现的是塑性-保持权衡，不是单纯的超参数最优

- `r=0.05`：Old F1 `+0.822 pp`，New F1 `-0.687 pp`；
- `r=0.10`：New F1 `+0.721 pp`，Old F1 `-0.925 pp`。

这说明更大的 X trust region 能释放新类塑性，但会改变新节点在归一化图传播中的消息，进而损害旧类。问题已经从“类内梯度太粗”推进到“如何在 Student 响应层面约束 old/new trade-off”。

## 7. 当前决策

`x-provenance` 暂不晋升为主方法，也不直接进入 `P/A` 联合优化。它是目前最有解释力的 X-only 机制候选：

1. 只修改 `X`，变量边界清楚；
2. 有可审计的 raw-to-slot 容量守恒；
3. 不访问历史 raw graph；
4. 相比 `x-class`，确实改善了部分新类塑性；
5. 但跨数据集、跨顺序和 old/new trade-off 仍不稳定。

## 8. 下一步：有界旧响应约束

下一轮不再扫描更多 radius，而是在 `x-provenance` 上增加一个 construction-time 的旧类响应约束，仍只更新新 synthetic `X`：

\[
L = L_{new}^{slot}
  + \lambda_{old} L_{old}
  + \lambda_{barrier}\max(0, L_{old}-L_{old}^{base}-\epsilon)^2
  + \lambda_x\|\Delta X\|^2.
\]

其中：

- `L_old^base` 来自同一 sealed support、同一 Student train seed 下的 C0 reference；
- `epsilon` 是预先固定的允许代理损失变化，不使用最终 F1 选择；
- `L_new^slot` 仍按 provenance slot 路由；
- `P`、文本、拓扑和下游 Student 不变；
- 若约束失败，保留 C0 candidate，不做无界 revision。

这一步的目的不是让 proxy loss 本身成为论文指标，而是检验：显式限制旧类响应退化，能否在 `r=0.10` 的塑性收益和 `r=0.05` 的旧类保持之间得到更稳的折中。

## 9. 后续受控优化结果

在 `x-provenance` 基础上又完成了三组单因素实验。它们都使用相同的四个 cell、Student seeds `150--155`、`r=0.10`、2-step inner/2-step outer 和相同 C0 control。

### 9.1 有界旧响应 barrier

形式为：

```text
L = L_new + 0.25 L_old
  + 1.0 * relu(L_old - L_old(C0) - 0.02)^2
```

结果：

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -1.072 | -0.575 | -1.747 | +0.501 |
| Cora / order19102 | +0.635 | -0.500 | +0.877 | -0.003 |
| Citeseer / canonical | -1.926 | -2.507 | -1.684 | -1.223 |
| Citeseer / order19102 | +0.511 | +0.209 | +2.614 | -2.901 |
| **四 cell 平均** | **-0.463** | **-0.843** | **+0.015** | **-0.907** |

审计显示四个 cell 的 support `old_response` 均没有超过 C0 reference 加 margin，barrier 实际为 0。因此它没有真正约束到下游旧类下降；这直接说明 construction support 的旧类 InfoNCE 与真实 full-graph 分类保持不一致。

### 9.2 传播算子一致的 outer objective

此版本让 inner Student 和 outer response 都使用 candidate synthetic adjacency。当前 raw delta 只用于构造每个 slot 的文本 prototype，避免用 support graph 的另一套邻接矩阵评价 candidate。

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.694 | -0.985 | -1.518 | +1.561 |
| Cora / order19102 | **+2.367** | +0.833 | **+4.874** | -1.241 |
| Citeseer / canonical | -0.636 | -0.474 | -2.675 | +1.971 |
| Citeseer / order19102 | -1.068 | -1.395 | -2.435 | +0.371 |
| **四 cell 平均** | **-0.008** | **-0.505** | **-0.439** | **+0.666** |

它改善了一个真实的协议错位，但没有带来跨顺序泛化。Cora/order19102 的正收益仍是局部 cell 现象。

### 9.3 算子一致 + construction-time 类 prompt CE

在 candidate graph 新 slot 表示上加入当前到达类别对应的 prompt CE，权重固定为 `0.5`，不使用 report/test 标签。

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -3.306 | -2.463 | -4.885 | +0.187 |
| Cora / order19102 | +0.471 | +0.083 | +0.512 | +0.251 |
| Citeseer / canonical | -0.159 | -0.271 | -1.556 | +2.002 |
| Citeseer / order19102 | -2.167 | -2.371 | -1.824 | -3.226 |
| **四 cell 平均** | **-1.290** | **-1.256** | **-1.938** | **-0.197** |

类别目标更接近最终评价形式，但在当前短 outer horizon 下会把 `X` 推向 prompt 几何，不能稳定改善新类塑性。

### 9.4 冻结 inner Student 的文本投影

此版本在 inner unroll 中固定 `text_proj.weight`、`text_proj.bias` 和 `logit_scale`，避免 outer 梯度通过文本分支降低代理损失；图 encoder 仍按原协议更新。

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.362 | -0.164 | -1.343 | +1.084 |
| Cora / order19102 | +0.288 | -0.583 | +0.328 | -0.582 |
| Citeseer / canonical | -0.026 | +1.084 | -2.651 | +3.490 |
| Citeseer / order19102 | +0.018 | -1.255 | -0.229 | -1.118 |
| **四 cell 平均** | **-0.021** | **-0.230** | **-0.974** | **+0.719** |

冻结文本分支能改变 old/new trade-off，但没有恢复新类 F1。文本 shortcut 是问题之一，不能解释全部退化。

## 10. 当前结论与后续路线

本轮已经把 X-only 外层路线中最重要的四个假设分别做了受控检验：

1. 类内 provenance 路由减少了类级梯度混合，但收益依赖顺序；
2. trust region 半径控制塑性与保持的权衡，不能单独解决问题；
3. 使用同一 candidate propagation operator 能修正协议错位，但不能保证分类收益；
4. 类 prompt CE 和冻结文本投影都不能把短期 proxy 改造成可靠的最终 Student 评价器。

因此下一步不再继续增加 `X` 的损失项，也不直接进入 `P/A` 联合优化。更合理的方向是重新设计**训练响应估计器**：使用多个 Student training checkpoints 的 trajectory response，比较 candidate 与 C0 在相同后续训练前缀上的增量，而不是只依赖 2-step 单点 support loss。该响应估计器应先作为 read-only 选择/诊断，固定 X provenance candidate，不立即修改 `X`；只有它能在独立顺序上预测最终 F1 delta 后，才允许用于有限幅度的 X update。

建议的下一轮最小实验：

- 固定当前 `x-provenance r=0.10` 候选和 C0；
- 使用相同 Student 初始化、batch 顺序和优化器，保存 0/2/4/8 个训练 checkpoint；
- 在 construction-time old/new support 上计算每个 checkpoint 的 delta trajectory；
- 在不更新 memory 的条件下，测 trajectory response 与最终 F1 delta 的相关性；
- 若相关性不足，终止 Student outer 更新，保留 provenance 作为负结果；若相关性成立，再做一个有限半径的 response-guided X revision。

## 11. 产物

- 构造实现：[direct_student_outer.py](../../code/v2_llm_fusion/direct_student_outer.py)
- 运行入口：[run_direct_student_outer.py](../../code/v2_llm_fusion/run_direct_student_outer.py)
- provenance 映射：[prepare_provenance_x_support.py](../../code/v2_llm_fusion/prepare_provenance_x_support.py)
- `r=0.05` 评估：`results/direct_student_outer_20260928/eval_provenance_x_r05b_*.json`
- `r=0.10` 评估：`results/direct_student_outer_20260928/eval_provenance_x_r10b_*.json`
- barrier 评估：`results/direct_student_outer_20260928/eval_provenance_x_r10_barrier_*.json`
- 算子一致评估：`results/direct_student_outer_20260928/eval_provenance_x_r10_operator_*.json`
- 算子一致 + prompt CE：`results/direct_student_outer_20260928/eval_provenance_x_r10_operator_ce_*.json`
- 算子一致 + 冻结文本：`results/direct_student_outer_20260928/eval_provenance_x_r10_operator_freeze_*.json`
