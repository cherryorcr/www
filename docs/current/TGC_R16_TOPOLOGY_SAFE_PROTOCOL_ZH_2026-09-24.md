# TGC-R16 拓扑安全实验协议

## 目标

R15 将 R11 的新块内容与 C0 拓扑交叉后，最终 F1 相对 C0 提升 `+0.723 pp`，而 R11 完整版本为 `+0.142 pp`；这说明 R11 的 coarse topology 可能损害新类传播。本轮只修改拓扑，不修改 R11 内容、文本、预算、alpha 或 student。

## 固定条件

- 数据集：Cora、Citeseer。
- 顺序：`order19601/19602/19603`。
- 协议：base + 2 incremental，5% growing memory。
- 内容：封存 R11 `anchor025` 的 X、mass、文本和冻结 MiniLM embedding。
- 文本：不超过 5 个来源、320 字符、拼接后重新编码；本轮不评价文本算子。
- student：两层 384 维 GCN/GraphCLIP，batch 16，steps=`max(50,min(2048,16*rows))`，report seeds `150..155`。
- 历史访问：只读封存 synthetic memory；历史 raw graph reads=0。

## 实验臂

| arm | 内容 | 拓扑 |
|---|---|---|
| `c0` | C0 | C0 |
| `anchor025` | R11 | R11 coarse |
| `anchor_content_c0_topology` | R11 | C0 |
| `degree_safe` | R11 | C0 old-old/cross；新块保持 C0 内部边数和每个新节点内部度数，用 R11 内容做二交换重连 |
| `cross_guard` | R11 | C0 topology fallback；仅当 R11 跨块候选的内容得分比 C0 高固定 margin 且未超过 C0 派生的旧端点容量时替换，否则回退 C0 |

## 不允许的选择

- 不按最终 F1 选择拓扑。
- 不使用历史完整 raw graph。
- 不改旧 synthetic 行和旧旧边。
- 不同时引入动态预算、文本 attention、response selector 或旧记忆 refresh。

## 晋升门禁

相对 C0 的最终阶段平均至少 `+0.5 pp`，阶段平均不低于 0，new F1 平均不下降，最差 final cell 不低于 `-2 pp`，至少 4/6 final cells 非负，且所有 construction/evaluation/reload audit 通过。否则 R16 只作为拓扑机制诊断，不进入主方法。
