# TGC-R17 Teacher-Message Topology Gate 协议

## 目标

R15/R16 表明 R11 内容有正信号，但 coarse topology 会损害新类传播。本轮固定 R11 内容，只检验 teacher message compatibility 是否能在 C0 结构边界内选择少量安全拓扑修改。

## 固定条件

- Cora、Citeseer；`order19601/19602/19603`；base + 2 incremental。
- 5% growing synthetic memory；student、batch、steps、report seeds 与 C0 完全一致。
- 内容固定为 R11 `anchor025`：`alpha_content=0.25`。
- 文本固定为 R11：最多 5 个来源、320 字符、拼接后 MiniLM 重编码。
- teacher 使用 `9000/9001`、128 steps、batch 64、compact sampler。
- 构造只读取旧 synthetic memory、当前 delta 和封存 R11 assignment；历史 raw reads=0。

## 拓扑门

1. C0 old-old edges、new-new edge count、new-node internal degree 和 cross degree 作为硬约束。
2. R11/C0 的已有 new-new/cross edges 组成候选池，不生成无来源的任意边。
3. 用 teacher graph/text representation 聚合到 R11 synthetic rows，得到 teacher message target。
4. 只接受 teacher message loss 下降且 old-node message risk 不增加的 edge swap。
5. 分别限制最多 1 次或 2 次 new-new 交换。
6. 跨块边只有在 message score 提升超过固定 margin 且旧端点容量允许时替换。
7. 最终全局 message loss 不改善时，整个 topology 回退到 C0。

## 实验臂

| arm | 内容 | 拓扑 |
|---|---|---|
| `c0` | C0 | C0 |
| `anchor025` | R11 | R11 |
| `anchor_content_c0_topology` | R11 | C0 |
| `message_gate_1` | R11 | 最多 1 次 message-compatible internal swap + guarded cross |
| `message_gate_2` | R11 | 最多 2 次 message-compatible internal swap + guarded cross |

## 晋升门禁

最终平均至少 `+0.5 pp`，阶段平均不低于 0，new F1 平均不下降，最差 final cell 不低于 `-2 pp`，至少 4/6 final cells 非负，构造和 reload audit 全部通过。否则只保留为机制诊断。
