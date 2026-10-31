# TGR-R 下一轮归因与优化决策

日期：2026-09-27

## 1. 本轮目的

本轮不继续扩大 Teacher/P/X 联合优化，而是先回答：此前 `X-only + message_mix=0.20` 的正向结果，到底来自合成特征 `X`、拓扑修正，还是二者共同作用。

统一条件为：

- Semantic-Recursive 作为同 cell 的 base memory；
- 合法 support 为 parent synthetic memory 加当前 raw delta；
- Student、文本、memory budget、类别顺序和六个 Student seeds 完全一致；
- Cora/Citeseer，`order19611/order19612`，Stage-1/Stage-2；
- 只使用 report-dev 之外的构造信息，不使用最终 F1 做选择。

## 2. 实现修正

### 2.1 `topology_only` 归因臂

新增 `topology_only` mode：固定初始 sparse assignment `P`，不更新 `X`，不学习 `P`，只将 bounded message kernel 材料化为二值拓扑。

### 2.2 identity/no-op 回退

外层优化先计算未更新 candidate 的 validation proxy。只有更新后的 candidate 严格改善 proxy 才接受，否则返回原始 base memory。

该回退只保证构造 proxy 不退化，不能等价保证最终 Student F1 不下降。当前实验表明 proxy 与下游 F1 仍存在明显错位，因此它只能作为安全审计，不能作为性能 selector。

### 2.3 `message_mix=0` 的精确拓扑控制

修正后，当 `message_mix=0` 时直接复用 base `edge_index`，不再重新材料化边，避免仅因 edge index 排列变化而被误判为拓扑改变。

相关代码：

- [teacher_guided_message_revision.py](../../code/v2_llm_fusion/teacher_guided_message_revision.py)
- [run_teacher_guided_message_revision.py](../../code/v2_llm_fusion/run_teacher_guided_message_revision.py)

## 3. 归因结果

### 3.1 X-only，固定 base topology

该组使用 `message_mix=0`，只允许 `X` 接受 Teacher/Student 梯度更新。

| 数据集 | 顺序 | 阶段 | Final F1 差值 | Accuracy 差值 | New F1 差值 | Old F1 差值 |
|---|---|---:|---:|---:|---:|---:|
| Cora | order19611 | 1 | -0.644 | -1.252 | +0.866 | -1.649 |
| Cora | order19611 | 2 | -3.234 | -2.768 | -8.321 | +1.349 |
| Cora | order19612 | 1 | -4.759 | -5.489 | -8.174 | -1.705 |
| Cora | order19612 | 2 | -1.750 | -0.185 | -1.591 | -4.422 |
| Citeseer | order19611 | 1 | -1.549 | -1.630 | -1.045 | -1.497 |
| Citeseer | order19611 | 2 | -2.607 | -2.769 | -0.647 | -2.832 |
| Citeseer | order19612 | 1 | -1.513 | -1.942 | -2.628 | -1.036 |
| Citeseer | order19612 | 2 | -0.739 | -1.149 | -2.716 | -0.222 |
| **平均** |  |  | **-2.10** | **-2.15** | **-3.03** | **-1.50** |

结论：当前 gradient-consistent X objective 在固定拓扑上没有形成可靠收益。它不是简单的“旧行更新太多”问题，因为这一组仍然整体下降；更可能是 Teacher gradient/trajectory proxy 与最终图文分类任务不一致。

### 3.2 topology-only，`message_mix=0.20`

该组不修改 X，只使用固定初始 P 生成 bounded message topology，单行度数保持不变，最多替换 10% 邻居。

| 数据集 | 顺序 | 阶段 | Final F1 差值 | Accuracy 差值 | New F1 差值 | Old F1 差值 |
|---|---|---:|---:|---:|---:|---:|
| Cora | order19611 | 1 | +0.890 | +0.960 | -0.200 | +2.110 |
| Cora | order19611 | 2 | +2.030 | -0.310 | +5.010 | +0.650 |
| Cora | order19612 | 1 | +0.030 | -4.390 | +4.760 | -2.590 |
| Cora | order19612 | 2 | -0.300 | +0.610 | +0.530 | -1.170 |
| Citeseer | order19611 | 1 | +1.460 | +1.630 | +4.860 | -1.200 |
| Citeseer | order19611 | 2 | -0.880 | -0.840 | -1.970 | +0.030 |
| Citeseer | order19612 | 1 | +1.250 | +0.080 | -0.620 | +4.230 |
| Citeseer | order19612 | 2 | -0.410 | -0.990 | -0.840 | -0.510 |
| **平均** |  |  | **+0.51** | **-0.41** | **+1.44** | **+0.19** |

结论：当前正向信号主要来自 topology transport，而不是 X 梯度更新。收益集中表现为新类塑性和部分旧类保持，但 Accuracy 平均略降，且 Stage-2/Citeseer 仍有负 cell。

### 3.3 topology-only，`message_mix=0.10`

| 数据集 | 顺序 | 阶段 | Final F1 差值 | Accuracy 差值 | New F1 差值 | Old F1 差值 |
|---|---|---:|---:|---:|---:|---:|
| Cora | order19611 | 1 | +1.220 | +1.640 | +1.470 | +1.940 |
| Cora | order19611 | 2 | +1.330 | +2.770 | -1.360 | +4.160 |
| Cora | order19612 | 1 | -4.130 | -5.990 | -3.500 | -4.200 |
| Cora | order19612 | 2 | +1.430 | +1.480 | +1.080 | +1.270 |
| Citeseer | order19611 | 1 | -0.010 | +0.220 | -0.290 | +0.400 |
| Citeseer | order19611 | 2 | -1.350 | -0.570 | -2.830 | -1.490 |
| Citeseer | order19612 | 1 | +0.010 | -0.410 | -1.750 | +3.380 |
| Citeseer | order19612 | 2 | -0.280 | -1.040 | -3.160 | +1.340 |
| **平均** |  |  | **-0.22** | **-0.24** | **-1.29** | **+0.85** |

结论：较小 `message_mix` 并不保证更稳定。Cora/order19612/Stage-1 的大幅下降说明拓扑收益不是由扰动幅度单调决定，而取决于候选边是否适合当前阶段的结构和类别组成。

## 4. 当前最重要的判断

1. 之前的 `X-only + mix=0.20` 不能被解释为“X 梯度优化有效”。它同时改变了最终拓扑。
2. 固定拓扑上的 X-only 结果全部下降，当前不应继续扫描 X learning rate、trust-region 半径或 outer steps。
3. topology-only `mix=0.20` 是目前最有希望的局部组件，但还不能直接升为主方法，因为跨顺序和跨阶段仍不稳定。
4. `message_mix=0.10` 的结果否定了“简单减小拓扑扰动就能解决问题”的假设。
5. 当前 no-op proxy gate 是必要的工程保护，但不能替代独立 Student 评估。它在 X-only 实验中接受了 proxy 更好的候选，而下游 F1 仍下降，说明 proxy 需要重新校准。

## 5. 下一轮唯一主线：结构可靠性 gate

下一轮暂时固定 `X`、文本、预算和 P，不再优化 X。只研究是否应该接受 topology transport。

### 5.1 构造阶段统计量

只使用 support 和 synthetic metadata 计算：

- assignment entropy/coverage；
- new-to-old cross-block message mass；
- 每行 degree change 和 edge overlap；
- projected soft kernel 与 materialized binary topology 的 message discrepancy；
- Teacher seed 间的 topology/message disagreement；
- 新 block 的孤立率和类间连接不平衡。

### 5.2 三档 gate

预注册三个动作：

| 条件 | 动作 |
|---|---|
| 高 entropy、高 disagreement、message discrepancy 超阈值 | 回退 Semantic-Recursive 原拓扑 |
| 中等风险 | `message_mix=0.10`，固定小比例 edge swap |
| 低风险且新旧 message 兼容 | `message_mix=0.20`，固定小比例 edge swap |

gate 不能使用最终 F1、report/test labels 或候选 Student 结果选择。

### 5.3 通过标准

- 2 个数据集、2 个独立顺序、Stage-1/Stage-2；
- 平均 Final F1 不低于 Semantic-Recursive；
- 最差 cell 不低于 `-1 pp`；
- 新类 F1 平均不下降；
- Accuracy 平均不下降超过 `0.5 pp`；
- 拓扑边数和每行度数不变；
- 构造成本不超过 Semantic-Recursive 的 2 倍。

如果 gate 仍不能消除负 cell，拓扑 transport 作为诊断性结果保留，不再继续加入 X/P/text 模块。

## 6. 当前方法定位

当前最合理的主线不是“Gradient-Consistent X Revision”，而是：

> Semantic-Recursive memory with teacher-guided, degree-preserving, bounded topology transport.

其中 topology transport 仍是候选组件，尚未达到稳定主方法标准。X 梯度优化保留为负结果和后续研究方向，不能在论文中作为已经验证的核心增益。
