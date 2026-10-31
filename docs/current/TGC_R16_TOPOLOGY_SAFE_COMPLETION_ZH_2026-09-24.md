# TGC-R16 拓扑安全实验完成报告

日期：2026-09-24  
状态：**完成；R16 两个拓扑候选均未晋升，R15 的内容正信号得到重复确认**

## 1. 结论先行

本轮回答了四个关键问题：

1. R11 的 `alpha` 是固定实验参数，不是自动学习出来的；R14 的逐行动态 alpha 也没有稳定解决新类塑性。
2. R11 的 X/text/topology 在原始构造阶段使用同一个 assignment，因此不是实现层面的直接错位；真正的问题是这个 assignment 投影出的 coarse topology 会把新节点接入不利的旧传播结构。
3. R15 的 `R11 content + C0 topology` 仍是当前最有价值的研究候选，但它是归因交叉版本，不是完全一致的最终算法。
4. R16 的“只保持度数再按内容重连”没有带来稳定收益，说明拓扑是否安全不由度数和内容相似度单独决定；`cross_guard` 在本轮全部回退到 C0，等价于 R15 的候选。

因此当前仍保留：

> C0 作为可靠共同控制；R11/R15 作为内容与拓扑解耦的机制证据；下一步应做消息兼容约束下的拓扑门控，而不是继续扫描 alpha 或简单重连。

## 2. Alpha 到底是什么

### 2.1 R11 的 alpha

R11 使用：

\[
P_i=(1-\alpha)P_{\mathrm{initial},i}+\alpha P_{\mathrm{joint},i}
\]

本轮的 `anchor025` 在所有新行、所有阶段使用同一个固定值：

\[
\alpha=0.25
\]

`anchor050` 和 `anchor075` 是敏感性对照，不是动态决策。它们不读取最终 report F1，也没有根据每个类别的后验性能自动调节。

### 2.2 R14 的动态 alpha

R14 试过按 assignment 分歧逐行变化：

\[
d_i=\frac{1}{2}\|P_{\mathrm{joint},i}-P_{\mathrm{initial},i}\|_1
\]

\[
\alpha_i=\operatorname{clip}(0.25+\beta d_i,0,0.75)
\]

结果显示 `beta=0.50` 的最终 F1 相对 C0 可达 `+0.571 pp`，但 new F1 仍为负，阶段平均仍为负，最差 cell 仍退化。因此：

> 动态 alpha 这个形式有合理性，但当前使用的“assignment 分歧越大就越信任 joint”代理不够可靠。

下一版不应直接把 alpha 变成更复杂的学习器。更合理的是把它拆成两个门：

- 内容门 `alpha_content`：控制 R11 assignment 对 X/text 的影响；
- 拓扑门 `gamma_topology`：独立决定哪些新旧边可以接受。

这样可以避免一个 alpha 同时承担“内容塑性”和“消息传播安全”两个相反目标。

## 3. X 和拓扑是不是不一致

需要区分两个阶段。

### 3.1 R11 原始构造阶段

R11 的同一个 `P_alpha` 同时生成：

1. 新节点特征：

\[
x'_j=\frac{\sum_i P_{ij}x_i}{\sum_iP_{ij}}
\]

2. 新节点文本来源排序和文本 embedding；
3. coarse support 矩阵；
4. 新新边和新旧跨块边。

所以 R11 本身在材料化逻辑上是“共享 assignment 的一致构造”，不是 X 用一种 assignment、拓扑随意生成的代码错误。

### 3.2 R15 交叉版本

R15 的 `R11 content + C0 topology` 是故意把 R11 的 X/text/mass 与 C0 的 topology 拼在一起，用来做因果归因。它确实不再拥有一个完全共享的 assignment，但结果反而最好：

| 方法 | 最终 F1 相对 C0 | old F1 相对 C0 | new F1 相对 C0 |
|---|---:|---:|---:|
| R11 完整版本 | +0.142 | +1.629 | -2.059 |
| R11 content + C0 topology | **+0.723** | +2.103 | **-1.314** |
| C0 content + R11 topology | -0.473 | +2.245 | -4.522 |

这说明当前主要矛盾不是“X 和拓扑在数学上必须来自同一个 assignment”，而是：

> R11 的内容重构有正向信息，但它的 coarse topology 会改变消息传播路径，并在部分顺序上压制新类。

因此最终算法应追求“内容一致、拓扑受约束”，而不是强迫三种模态完全共用一个自由度很高的 assignment。

## 4. 文本为什么仍然是最多五个来源

当前文本不是随机拼接五个节点，而是：

1. 找到对当前合成行有正 assignment mass 的 delta 来源；
2. 按 assignment 权重从高到低排序；
3. 选前 `min(5, 可用来源数)` 个来源；
4. 将 320 字符预算平均分给被选来源；
5. 截断后拼接；
6. 对拼接结果重新使用冻结 MiniLM 编码。

它作为当前协议是合理的，原因是：

- 预算固定，便于公平比较；
- 选取规则确定，不依赖 report F1；
- 重新编码避免简单平均 embedding 的空间偏差；
- C0、R11、R15、R16 使用同一文本口径。

但它不是已经证明最优的文本算法。主要缺点是：

- `5` 是工程上限，不是由理论推导出的最优值；
- 等字符分配会截断长文本，也可能让短文本信息不足；
- 只按 assignment 权重选代表，缺少多样性和图结构条件；
- 文本顺序和拼接边界可能影响 MiniLM 表示；
- R15 的收益不能单独归因于 X，因为 R11 content 还包含了这套文本重构。

所以文本在 R16 中应保持不变，避免把拓扑实验和 P2 文本实验混在一起。后续 P2 才比较固定 token budget 下的 set attention、graph-conditioned attention 和 concat-reencode。

## 5. 旧记忆为什么全部不变

这不是因为持续 TAG 在理论上必须永远冻结旧记忆，而是当前 R11/R15/R16 的**受控实验协议**明确采用 append-only：

- 旧 synthetic rows 的 X、text、mass、depth、birth stage 不改；
- 旧旧边不改；
- 新阶段只读取旧 synthetic memory 和当前 raw delta；
- 历史完整 raw graph 不回读。

这样做有三个目的：

1. 分离“新块构造问题”和“旧记忆维护问题”；
2. 让每个阶段的历史行为可审计、可复现；
3. 避免一个 refresh/merge/recluster 同时改变内容、文本和拓扑，无法解释收益来源。

这不是最终持续算法的唯一选择。旧记忆遗忘确实可能需要有限 revision，但 P4 的简单 bounded refresh 已经没有形成可靠收益。因此当前先不把旧记忆 revision 和拓扑诊断同时加入。等新类塑性问题的拓扑因素明确后，再做：

- 只修改少量高风险旧行；
- X 改动有 trust region；
- text 和 birth metadata 默认不改；
- 旧类保持和新类收益同时过门；
- 失败就回退 C0。

## 6. R16 实验设置

R16 固定封存的 R11 `anchor025` X/text/mass，只修改新块 topology：

| 方法 | 拓扑规则 |
|---|---|
| C0 | C0 原始 topology |
| R11 | R11 coarse topology |
| R11 content + C0 topology | R11 content，完整回退 C0 topology |
| `degree_safe` | 保持 C0 old-old/cross、new-new 边数和每个新节点内部度数，使用 R11 content score 做二交换重连 |
| `cross_guard` | C0 topology fallback；只有 R11 跨边内容得分超过固定 margin 且旧端点容量允许时才替换 |

所有方法统一 Cora/Citeseer、三个固定顺序、base + 2 incremental、5% growing memory、student、batch、steps 和六个 report seeds。远程 `prepare -> seal -> evaluate -> audit` 全部通过。

## 7. R16 结果

所有数值为六个最终 dataset/order cell 的均值，差值相对 C0。

| 方法 | 最终 F1 | 最终差值 | old F1 差 | new F1 差 | 阶段平均差值 | 最差 final cell | 正收益 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/6 |
| R11 anchor025 | 51.656 | +0.142 | +1.629 | -2.059 | -0.856 | -2.104 | 3/6 |
| R11 content + C0 topology | 52.237 | **+0.723** | +2.103 | -1.314 | -0.314 | -3.155 | 4/6 |
| `degree_safe` | 51.263 | -0.251 | +0.735 | -1.741 | -1.053 | -2.260 | 3/6 |
| `cross_guard` | 52.237 | **+0.723** | +2.103 | -1.314 | -0.314 | -3.155 | 4/6 |

### 7.1 `degree_safe` 说明了什么

`degree_safe` 不是简单失败，而是排除了一个错误解释：

> 只要保持每个新节点的内部度数，拓扑就会安全。

结果最终平均为 `-0.251 pp`，阶段平均为 `-1.053 pp`，最差 cell 为 `-2.260 pp`。它在 Cora/order19603 和 Citeseer/order19601/19602 有局部恢复，但在 Cora/order19601/19602 明显退化。

说明传播效果依赖具体的边端点和路径结构，而不仅是 degree histogram。对小图而言，一次二交换就可能改变跨类消息传播方向。

### 7.2 `cross_guard` 说明了什么

本轮 12 个增量阶段的 `cross_guard` 实际替换边数量均为 0，全部回退到 C0。它不是一个新的正向结果，而是说明当前设定下：

- R11 跨边候选没有足够稳定的内容得分优势；
- 固定 margin 和旧端点容量限制过于保守；
- 简单地从 C0/R11 两组跨边中择优，不能产生可晋升的在线拓扑规则。

因此 `cross_guard` 的数值与 R15 `R11 content + C0 topology` 完全一致，不能重复计为独立方法收益。

## 8. 晋升判定

R16 的两个拓扑候选均未通过：

| 条件 | `degree_safe` | `cross_guard` |
|---|---:|---:|
| 最终平均至少 +0.5 pp | 未通过 | 表面通过，但与 R15 完全相同 |
| 阶段平均不低于 0 | 未通过 | 未通过 |
| new F1 平均不下降 | 未通过 | 未通过 |
| 最差 final cell 不低于 -2 pp | 未通过 | 未通过 |
| 至少 4/6 final cells 非负 | 未通过 | 通过，但沿用 R15 |
| audit | 通过 | 通过 |

## 9. 下一步方法设计

下一版不再做“共享 assignment 直接投影成全部边”，也不再做“只按内容相似度重连”。建议设计为两层结构：

### 9.1 内容层

保留 R11 的有限 anchor：

\[
P^{content}_i=(1-\alpha_i)P_{initial,i}+\alpha_iP_{joint,i}
\]

先固定 `alpha_content=0.25` 做共同控制，不急于动态学习。它只用于新 X、mass 和文本来源。

### 9.2 拓扑层

新旧边和新新边单独生成：

1. 旧旧边严格沿用 C0；
2. C0 提供新块边数、每行总度数和跨块容量的 scaffold；
3. R11 teacher 只提供候选 edge score，不直接决定最终 edge；
4. edge score 同时包含 feature/text 相似度和 teacher message compatibility；
5. 在 C0 的 degree、cross-capacity 和 edge-count 约束下做局部交换；
6. 只有 teacher message loss 有明确下降且新节点风险不升高时才接受；
7. 任一约束失败立即回退 C0。

核心变化是：

> R11 负责提供“新内容应该和谁相似”的候选信息，C0 负责提供“图传播不能偏离到什么程度”的结构边界。

后续实验优先级：

1. 固定 `alpha_content=0.25`，实现 teacher-message-compatible edge score；
2. 只允许每个新块最多少量局部 edge swap，先做 `0/1/2` 次交换敏感性；
3. 用独立的 synthetic-memory message loss 和 cross-risk 做门控，不使用 report F1；
4. 保持 top-5/320-char 文本不变，先确认拓扑模块；
5. 若拓扑模块稳定，再单独回到文本 attention/set encoder；
6. 最后才做 bounded old-memory revision。

## 10. 当前评价

当前最强的科学证据不是“R16 已经超过 C0”，而是：

- R11 content 在多个 cell 中有正向信号；
- R11 coarse topology 会伤害新类塑性；
- C0 topology 是目前更可靠的传播 scaffold；
- 简单 degree-preserving rewire 不足以恢复性能；
- 动态 alpha 和简单跨边选择都还没有可靠的在线判据。

因此论文主方法暂时仍应是 C0，R11/R15/R16 用于形成“内容-拓扑解耦、保持-塑性冲突和拓扑负结果”的机制分析。只有下一版消息兼容拓扑门控在两个数据集、多个顺序上满足晋升门槛，才有理由把它升为完整方法。

## 11. 产物

- 协议：[TGC_R16_TOPOLOGY_SAFE_PROTOCOL_ZH_2026-09-24.md](TGC_R16_TOPOLOGY_SAFE_PROTOCOL_ZH_2026-09-24.md)
- runner：[run_tgc_r16_topology_safe.py](../../code/v2_llm_fusion/run_tgc_r16_topology_safe.py)
- 远程部署：[tgc_r16_remote.py](../../tools/tgc_r16_remote.py)
- 本地结果：[tgc_r16_topology_safe_20260924](../../results/tgc_r16_topology_safe_20260924/)
- 汇总：[methods.csv](../../results/tgc_r16_topology_safe_20260924/methods.csv)、[cells.csv](../../results/tgc_r16_topology_safe_20260924/cells.csv)
