# TGC-R15：R11 内容/拓扑交叉归因实验协议

R14 表明分歧加权能够改善部分最终 cell，但没有恢复新类 F1。R15 不再调 assignment 系数，而是直接交叉已经封存的 R11 和 C0 材料化结果，判断新类损失来自 R11 拓扑还是内容表示。

| arm | 新块内容 | 新块拓扑 |
|---|---|---|
| C0 | C0 | C0 |
| R11 anchor025 | R11 | R11 |
| `anchor_content_c0_topology` | R11 | C0 |
| `c0_content_anchor_topology` | C0 | R11 |

两个交叉臂分别只替换一个因素：一个保留 R11 内容、回退 C0 拓扑；另一个保留 C0 内容、只使用 R11 拓扑。四个 arm 使用相同的 rows、student、batch、steps、report seeds 和历史访问约束，不使用 report F1 选择。

晋升不是本轮目标；本轮先判断 new F1 的主要损失来源。若 `anchor_content_c0_topology` 相对 R11 明显提高 new F1，后续主方法候选应保留 R11 assignment/content、回退 C0 topology；否则进入新块 X 的受控残差实验。
