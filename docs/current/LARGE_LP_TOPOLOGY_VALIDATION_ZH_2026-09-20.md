# 大图链路预测拓扑验证报告

更新时间：2026-09-20

状态：**arXiv 与 BookChild 大图筛选已完成；新拓扑候选暂不替换 C0 为统一主方法。**

## 1. 研究问题

本轮验证此前在 Cora/Citeseer 小图上观察到的现象：在不改变合成节点特征、文本表示、节点数量和阶段预算的前提下，通过更规则的合成图拓扑，是否可以改善持续凝聚图在链路预测上的可重训效用。

本轮只改变拓扑构图规则，重点回答三个问题：

1. 小图上看到的度均衡收益能否迁移到大图；
2. 将节点特征 X 加入拓扑相似度是否必要；
3. coverage 惩罚是否比简单的 degree-balanced 构图更稳定。

本轮是大图机制筛选，不是完整的静态 TAG 方法复现，也不构成 SOTA 结论。

## 2. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集 | arXiv、BookChild |
| 持续协议 | 已冻结的 base+4 五阶段类别增量协议 |
| 压缩率 | 5% pool，沿用已有 C0 每阶段 rows |
| 方法臂 | C0、T+degree、XT+degree、T+coverage、XT+coverage、full_raw_equal |
| edge split | 39001、39002，按阶段到达时间分层，独立于方法构造 |
| student | 与既有链路预测实现一致的 GAE student |
| student steps | 256 |
| student seeds | 100、101、102、103 |
| 指标 | held-out real edges 上的 ROC-AUC 与 AP |
| 总规模 | 每个数据集 2×5×6×4=240 次 fit |
| 数据访问 | 构图和训练只使用已见 pool 节点；test/dev 原始节点不进入构图或训练 |

`full_raw_equal` 是完整已见真实图参考：它使用同一阶段可见的真实节点和训练边，student、steps、edge split 与合成方法一致。它不是压缩方法，也不是理论上限。

## 3. 方法定义

### 3.1 C0

C0 是当前冻结共同控制。每阶段读取既有 `append_fixed_s{t}_memory.pt`，保留旧合成 rows 和旧旧边，追加当前阶段的新合成 block 及 C0 原有拓扑。它用于回答新拓扑算子相对于当前方法的差异。

### 3.2 T 与 XT kernel

对候选节点对计算归一化余弦相似度：

\[
K(i,j)=w_x\,\cos(x_i,x_j)+w_t\,\cos(t_i,t_j)+w_g\,\cos(g_i,g_j).
\]

本轮实际使用：

- `T`：\((w_x,w_t,w_g)=(0,1,0)\)，只使用文本表示；
- `XT`：\((4/7,3/7,0)\)，联合使用合成节点特征 X 与文本表示；
- 本轮没有使用 graph signature，因此不能把结果解释为 graph-signature 方法的验证。

每个新 block 的内部边使用同一 kernel 排序。为了避免合成图中出现大量孤立节点，内部边采用无环、每节点度上限为 2 的路径构图，并在最后闭合路径。

### 3.3 degree-balanced 与 coverage

- `degree-balanced`：在 kernel 相似度排序的基础上进行 cap-2/path 约束；跨旧块连接时给旧节点设置均衡容量，避免所有新节点集中连到少数旧节点。
- `coverage`：在 degree-balanced 的候选边分数上加入 0.15 倍的已选边冗余惩罚，试图覆盖更多不同的相似区域。

候选只替换 `edge_index`，不改变 C0 的 `x`、`text_emb`、`mass`、`depth`、`stage` 和文本内容。因此性能差异不能归因于增加 rows、改变 X 合成或增加训练步数。

## 4. 完整结果

### 4.1 arXiv 最终阶段

数值为 8 个配对条件（2 split × 4 student seeds）的均值 ± 标准差，括号为相对于 C0 的百分点差。

| 方法 | ROC-AUC | AP |
|---|---:|---:|
| C0 | 97.685 ± 0.091 | 97.575 ± 0.102 |
| T+degree | **97.742 ± 0.062 (+0.058)** | **97.579 ± 0.090 (+0.003)** |
| XT+degree | 97.501 ± 0.168 (-0.184) | 97.319 ± 0.169 (-0.257) |
| T+coverage | 97.647 ± 0.124 (-0.038) | 97.473 ± 0.148 (-0.102) |
| XT+coverage | 97.616 ± 0.173 (-0.069) | 97.426 ± 0.203 (-0.149) |
| full_raw_equal | 97.733 ± 0.138 (+0.048) | 97.712 ± 0.093 (+0.137) |

T+degree 在 8 个配对条件中有 5 个 AUC、4 个 AP 高于 C0，但绝对提升只有约 0.06/0.003 pp，处于小幅波动范围。arXiv 上不能据此宣称稳定提升。

### 4.2 BookChild 最终阶段

| 方法 | ROC-AUC | AP |
|---|---:|---:|
| C0 | 92.061 ± 0.486 | 91.083 ± 0.540 |
| T+degree | 93.329 ± 0.383 (+1.268) | 92.832 ± 0.412 (+1.749) |
| XT+degree | **93.796 ± 0.523 (+1.735)** | **93.226 ± 0.606 (+2.142)** |
| T+coverage | 93.189 ± 0.356 (+1.127) | 92.731 ± 0.349 (+1.648) |
| XT+coverage | 93.642 ± 0.340 (+1.580) | 93.053 ± 0.316 (+1.970) |
| full_raw_equal | 95.320 ± 0.300 (+3.259) | 95.291 ± 0.331 (+4.207) |

BookChild 上的配对结果更一致：T+degree 在 8 个条件中 7 个 AUC、8 个 AP 高于 C0；XT+degree 在 AUC/AP 上均为 8/8 高于 C0。这里的新拓扑确实改善了压缩图的链路预测效用，但距离完整真实图仍有约 1.52 pp AUC、2.06 pp AP 的差距。

### 4.3 所有阶段均值

阶段均值先对两个 split、四个 student seeds 聚合，再对五个阶段取均值。

| 数据集 | 方法 | stage-average AUC | stage-average AP |
|---|---|---:|---:|
| arXiv | C0 | 97.755 | 97.639 |
| arXiv | T+degree | 97.735 | 97.559 |
| arXiv | XT+degree | 97.620 | 97.434 |
| arXiv | full_raw_equal | 97.830 | 97.803 |
| BookChild | C0 | 91.385 | 90.305 |
| BookChild | T+degree | 93.211 | 92.680 |
| BookChild | XT+degree | **93.584** | **92.964** |
| BookChild | full_raw_equal | 95.299 | 95.272 |

这张表说明：BookChild 的收益不是只出现在最后一个阶段，XT+degree 的五阶段平均也高于 C0；arXiv 的最终阶段小幅正差没有转化为阶段平均正差，因此更应视为持平而非稳定改进。

## 5. 拓扑和成本审计

### 5.1 rows 与 edge 数量

候选方法与 C0 每个阶段的 rows、无向 pairs 完全一致。最终阶段为：

| 数据集 | 方法 | rows | pairs |
|---|---|---:|---:|
| arXiv | C0 / 所有候选 | 5,926 | 6,890 |
| BookChild | C0 / 所有候选 | 2,690 | 3,036 |

因此本轮不是通过增加合成节点或增加边数量获得收益。

### 5.2 孤立节点和最大度

| 数据集 | 阶段 | C0 isolated / max degree | 候选 isolated / max degree |
|---|---:|---:|---:|
| arXiv | 0 | 2,022 / 24 | 0 / 2 |
| arXiv | 4 | 1,776 / 25 | 0 / 5 |
| BookChild | 0 | 1,159 / 89 | 0 / 2 |
| BookChild | 4 | 1,063 / 92 | 0 / 5 |

这是本轮最明显的机制变化：C0 的合成拓扑含有大量孤立节点和少数高连接节点，而 degree-balanced/coverage 将内部 block 变为连通路径，并限制节点度。它提供了一个合理解释：链路预测 student 更容易从每个合成节点获得稳定的消息传播，而不是让大量 synthetic rows 在训练中没有任何邻居。

但这里仍不能把“消除孤立节点”单独称为已证明的因果因素，因为 path constraint、跨块容量、内部相似度排序同时发生了变化。后续需要做只改变一个算子的拆分实验。

### 5.3 构图成本

构图成本只记录 topology construction，不含完整 raw 数据加载、文本编码和 student 重训。

- arXiv：degree 方案首阶段约 5.7--6.6 秒，之后约 3.2--4.1 秒；coverage 首阶段约 35 秒，之后约 3.3--3.9 秒。
- BookChild：degree 首阶段约 1.4--1.5 秒，之后约 0.5--0.8 秒；coverage 首阶段约 11.9--12.0 秒，之后约 0.5--0.8 秒。
- 每个方法的 student 训练预算相同，均为 2 split × 5 stage × 4 seeds × 256 steps。

coverage 的首阶段代价明显高于 degree，但本轮没有发现它相对于 degree 的稳定收益。因此后续默认候选应优先使用 degree-balanced，而不是 coverage。

## 6. 结果解释

### 6.1 目前能确定的结论

1. **大图上度均衡拓扑有真实信号。** BookChild 的 T+degree、XT+degree、T+coverage、XT+coverage 全部超过 C0；其中 XT+degree 最好。
2. **该信号具有数据集依赖性。** arXiv 上所有候选与 C0 基本持平或下降，只有 T+degree 的 AUC 有约 +0.06 pp 的微小差异。
3. **graph signature/更复杂的联合 kernel 不是普适必要条件。** arXiv 上加入 X 反而降低表现；BookChild 上 XT 有帮助。因此应把 X 的作用视为数据集条件变量，而不是默认必须组件。
4. **coverage 不是稳定升级。** 它在 BookChild 有效，但在 arXiv 不如 C0；同时首阶段构图成本明显更高。
5. **压缩图已经保留了大部分链路预测能力，但不是完整图。** arXiv 的 C0 与完整图几乎相同；BookChild 的新拓扑缩小了 C0 与 full_raw_equal 的差距，但没有消除它。

### 6.2 当前不能声称的结论

- 不能声称新拓扑在所有大图上稳定优于 C0。
- 不能声称它超过完整真实图。
- 不能把 BookChild 的提升直接推广到节点分类或其它下游任务。
- 不能把 path/degree-balanced、跨块均衡、kernel 选择分别归因于单独收益。
- 不能将本轮筛选结果称为静态 TAG 凝聚 SOTA 对比；本轮没有补强 TAGSAM 或其它静态凝聚论文的严格同协议复现。

## 7. 当前方法定位

本轮之后，最合理的工程候选是：

> **C0 的合成 rows 与文本/X 保持不变；新 block 使用 T-only 相似度构造 cap-2 连通路径，并以均衡跨块容量追加到旧 memory。**

也就是 `T+degree-balanced append`。

但论文主方法是否从 C0 切换为这个候选，要看研究问题：

- 如果论文主线是**链路预测下的压缩图拓扑有效性**，可以把 `T+degree-balanced` 作为大图候选主方法，理由是它在 BookChild 上有约 +1.27 AUC/+1.75 AP，且构图成本低。
- 如果论文主线仍是**跨任务、跨数据集的持续 TAG 凝聚统一主方法**，目前应保留 C0 为共同控制，并把 T+degree-balanced 标为“有希望的大图 LP 分支”，因为 arXiv 阶段均值没有提升，且分类收益尚未验证。

## 8. 后续优先级

### P0：独立确认 T+degree

只保留 C0、T+degree、XT+degree 三个臂，在一个新的预注册 edge split 和一个未参与开发的类别顺序上复测。要求：

- BookChild 增益保持；
- arXiv 不出现明显退化；
- rows、pairs、student steps 继续严格匹配；
- 同时报告链路预测与节点分类。

### P1：拓扑因果拆分

固定 T kernel、rows 和总边数，分别比较：

1. 只改内部 path、保持原跨块边；
2. 只改跨块 degree capacity、保持内部边；
3. 只消除孤立节点；
4. 完整 T+degree。

目标是判断收益来自内部连通性、跨块负载均衡，还是仅仅来自孤立节点消除。

### P2：X 合成算子与联合评价

在拓扑候选稳定后，再测试 mean-X、medoid-X 和局部多原型 X。不要同时改变拓扑，否则无法判断收益来自 X 还是边。最终至少报告节点分类 Macro-F1、链路 AUC/AP、memory bytes、构图时间和 student 重训时间。

### P3：大图 rate curve

在 BookChild 和 arXiv 上比较 1%、5%、10% memory rate，检查 T+degree 是否只在当前 5% 预算有效。若收益随 rate 消失，说明它是极低预算下的拓扑修复；若保持，才更接近通用机制。

## 9. Artifact 与可复现入口

本地证据：

- `results/large_lp_validation_20260920/arxiv/`
- `results/large_lp_validation_20260920/bookchild/`
- `results/large_lp_validation_20260920/arxiv/summary.json`
- `results/large_lp_validation_20260920/bookchild/summary.json`

远程原始结果：

- `/home/pc/research/TAG_condensation/results/large_lp_validation_20260920/arxiv`
- `/home/pc/research/TAG_condensation/results/large_lp_validation_20260920/bookchild`

实现与测试：

- `code/v2_llm_fusion/large_lp_support.py`
- `code/v2_llm_fusion/run_large_lp_validation.py`
- `code/v2_llm_fusion/test_large_lp_support.py`

远程单元测试结果：4/4 passed。两数据集均完成 240/240 fits。
