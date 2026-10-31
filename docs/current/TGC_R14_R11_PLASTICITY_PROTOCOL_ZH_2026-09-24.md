# TGC-R14：基于 R11 的分歧加权新类塑性实验协议

## 1. 目的

R11 的 `anchor025` 能够降低 joint 的最差退化，但最终新类 F1 仍低于 C0。R14 不重做 TGC，也不改变文本、X 或拓扑算子；只把固定的全局 anchor 改成按当前新样本的 assignment 分歧加权。

直觉是：与 C0 初始 assignment 分歧很小的样本已经稳定，继续保留 R11 anchor；分歧较大的样本可能包含 teacher 发现的边界或多模态结构，应保留更多 joint assignment，以恢复新类塑性。

## 2. 算法

对当前增量块的每一行 `i`：

\[
d_i=\frac{1}{2}\|P_{joint,i}-P_{initial,i}\|_1
\]

\[
\alpha_i=\operatorname{clip}(0.25+\beta d_i,0,0.75)
\]

\[
P_i=(1-\alpha_i)P_{initial,i}+\alpha_iP_{joint,i}
\]

测试 `\beta=0.25` 和 `\beta=0.50`。`P_i` 行和保持为 1，所有新节点的 X、文本和拓扑仍使用同一 `P` 材料化。

## 3. 对照

| arm | 作用 |
|---|---|
| C0 | 原始可靠主线 |
| R11 anchor025 | 固定 `alpha=0.25` |
| joint | R10/R11 封存的 teacher joint assignment |
| plastic025 | `beta=0.25` 的分歧加权 anchor |
| plastic050 | `beta=0.50` 的分歧加权 anchor |

固定：Cora/Citeseer、`order19601/19602/19603`、base+2、5% growing memory、report seeds `150..155`、两层 384 维 GCN/GraphCLIP、冻结 MiniLM、双向 InfoNCE、batch 16、相同步数、相同文本预算和拓扑规则。

## 4. 选择与约束

- 不使用 report F1 选择候选；全部 arm 完整报告；
- 不读取历史完整 raw graph；只使用旧 C0 synthetic memory、当前 raw delta 和封存 R11 assignment；
- 不使用 test labels；当前 delta 的类别增量边界只用于既有协议和合法预算分组；
- 旧 synthetic 行、旧 edge block、旧文本和旧 mass 不修改；
- 结果只作为开发实验，不能因为某一个 order 的正收益直接晋升。

## 5. 晋升门禁

候选至少需要：最终平均相对 C0 `>=+0.5 pp`、阶段平均不低于 0、新类 F1 平均不下降、最差最终 cell 不低于 `-2 pp`，并且在两个数据集和多个顺序上成立。否则保留为 R11 的诊断扩展。
