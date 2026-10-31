# Direct Outer：受限拓扑与 X+拓扑联合实验报告

日期：2026-09-29  
状态：四个小图 cell 的 topology-only、X+topology、selector v1/v2 均已完成；未晋升为主方法组件。

## 1. 本轮目的

上一轮已经完成 `x-provenance r=0.10`、slot representation、inner horizon 和 payload-mass
聚合实验。结果说明，直接对新增 synthetic `X` 做 Student outer 更新可以产生局部信号，但
固定拓扑下的 support loss 与最终新类 F1 不稳定对应。因此本轮只验证一个更小的联合假设：

> 如果只在已有 synthetic memory 的 old-new cross edges 上做守恒的 endpoint transport，
> 并让 Student response 选择交换，是否能改善 X 与传播结构的一致性？

本轮没有重新设计文本、P、预算或 Student，也没有读取历史完整 raw graph。

## 2. 方法与约束

### 2.1 topology-only

给定 direct outer 的 `control_*` memory：

1. 固定所有 `X`、`text_emb`、文本、mass、rows、预算、old-old edges 和 new-new edges。
2. 对每个有 cross edge 的新行，用当前 synthetic X/text 的联合 cosine 相似度构造最多 3 个旧端点候选，
   并保留原始端点。
3. 只枚举 two-switch：两个新行互换一个旧端点。
4. 交换必须同时保持：
   - 每个新行的 cross degree；
   - 每个旧端点的 cross capacity；
   - old-old edges；
   - new-new edges。
5. 用 canonical two-layer GCN + bidirectional graph-text InfoNCE 做短程 Student response，
   不使用 report/test 标签；最多接受一次交换，失败精确回退控制 memory。

### 2.2 selector v1 与 v2

`v1` 使用 3 个 selector seeds、1 个 Student step 的平均 gate，配置为：

```text
new_gain >= 0
old_gain >= 0
total_after <= base_total_after
```

`v2` 只修改 selector 可靠性：使用相同 3 个 seeds、3 个 Student steps，并且要求至少 2/3
selector seeds 各自满足上述约束。它不改变候选池、memory budget 或下游 Student。

### 2.3 X+topology joint

先读取已完成的 `x-provenance r=0.10` memory，再重复同一 cross-edge operator。这样可以分开：

- C0/control；
- X-only；
- topology-only；
- X+topology。

联合版本不修改旧 rows，也不更新 P、文本或预算；它只是把 topology selector 叠加到已更新的新 X 上。

## 3. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 顺序 | canonical、order19102 |
| 阶段 | direct outer 的 stage 1 cell |
| downstream Student | canonical 两层 GCN + 可训练 text projection + 双向 InfoNCE |
| downstream seeds | 150--155 |
| selector seeds | 17、23、29 |
| selector batch | 16 |
| selector lr | 1e-3 |
| candidate top-k | 3 |
| X/text score | 0.5 X cosine + 0.5 text cosine |
| 最大交换数 | 1 |
| 历史 raw 访问 | 0 |
| report/test 标签 | 未使用 |

所有下游差值都是相对同一 cell 的 `control_*` memory，单位为百分点。

## 4. topology-only 结果

### 4.1 逐 cell

| 方法 | Cell | Final F1 | Accuracy | New F1 | Old F1 | 是否接受 swap |
|---|---|---:|---:|---:|---:|---:|
| topology v1 | Cora/canonical | -1.875 | -1.396 | -3.475 | +0.677 | 是 |
| topology v1 | Cora/order19102 | 0.000 | 0.000 | 0.000 | 0.000 | 否，回退 |
| topology v1 | Citeseer/canonical | +1.120 | +2.371 | +0.087 | +2.359 | 是 |
| topology v1 | Citeseer/order19102 | -0.092 | -1.883 | -1.621 | -1.017 | 是 |
| topology v2 | Cora/canonical | -1.875 | -1.396 | -3.475 | +0.677 | 是 |
| topology v2 | Cora/order19102 | 0.000 | 0.000 | 0.000 | 0.000 | 否，回退 |
| topology v2 | Citeseer/canonical | 0.000 | 0.000 | 0.000 | 0.000 | 否，回退 |
| topology v2 | Citeseer/order19102 | -0.092 | -1.883 | -1.621 | -1.017 | 是 |

### 4.2 汇总

| 方法 | 平均 Final F1 | 平均 Accuracy | 平均 New F1 | 平均 Old F1 | Final F1 正向 cell |
|---|---:|---:|---:|---:|---:|
| topology v1 | -0.212 | -0.227 | -1.252 | +0.505 | 1/4 |
| topology v2 | -0.492 | -0.820 | -1.274 | -0.085 | 0/4 |

v2 的确减少了一个可能不可靠的接受，但没有改善最终平均结果。Cora/canonical 的 swap 在 v1/v2
都通过局部 gate，却在完整 Student 上造成 `New F1 -3.475 pp`，说明 selector 目标和最终塑性
之间存在明显错位。

## 5. X+topology 结果

### 5.1 相对 control 的结果

| 方法 | Cell | Final F1 | Accuracy | New F1 | Old F1 | 是否修改 topology |
|---|---|---:|---:|---:|---:|---:|
| X+topology v1 | Cora/canonical | -2.205 | -1.724 | -2.440 | -2.172 | 是 |
| X+topology v1 | Cora/order19102 | +0.635 | -0.500 | +0.877 | -0.003 | 否，回退 |
| X+topology v1 | Citeseer/canonical | -1.498 | -0.474 | -4.025 | +2.119 | 是 |
| X+topology v1 | Citeseer/order19102 | +0.637 | 0.000 | -0.197 | +0.141 | 是 |
| X+topology v2 | Cora/canonical | +0.668 | +0.246 | +1.076 | +0.426 | 否，回退到 X-only |
| X+topology v2 | Cora/order19102 | +0.635 | -0.500 | +0.877 | -0.003 | 否，回退到 X-only |
| X+topology v2 | Citeseer/canonical | -1.926 | -2.507 | -1.684 | -1.223 | 否，回退到 X-only |
| X+topology v2 | Citeseer/order19102 | +0.010 | -1.395 | -1.464 | -0.292 | 是 |

### 5.2 与对应 X-only 的差值

对应的 `x-provenance r=0.10` 结果为：

| Cell | X-only Final F1 | X-only New F1 | Joint v1 相对 X-only Final/New | Joint v2 相对 X-only Final/New |
|---|---:|---:|---:|---:|
| Cora/canonical | +0.668 | +1.076 | -2.873 / -3.516 | 0 / 0 |
| Cora/order19102 | +0.635 | +0.877 | 0 / 0 | 0 / 0 |
| Citeseer/canonical | -1.926 | -1.684 | +0.428 / -2.341 | 0 / 0 |
| Citeseer/order19102 | +0.511 | +2.614 | +0.126 / -2.810 | -0.501 / -4.077 |

平均而言，联合 v1 相对 X-only 仍降低 Final F1 约 `-0.58 pp`，降低 New F1 约 `-1.11 pp`；
联合 v2 相对 X-only 降低 Final F1 约 `-0.15 pp`，降低 New F1 约 `-1.72 pp`。v2 的主要效果是
回退，而不是产生新的收益。

## 6. 审计结果

四个 topology-only 和四个 joint cell 均满足：

- `X`、文本、mass、rows、预算没有改变；
- old-old edges 和 new-new edges 保持；
- 新行集合保持；
- 每个新行 cross degree 保持；
- 旧端点容量保持；
- 历史完整 raw graph 访问为 0；
- report/test 标签未用于 selector。

因此负结果不是由预算变化、内容变化或边数变化造成的。v1/v2 的差异也只来自 selector horizon
和逐 seed guard。

## 7. 结论

本轮支持三个结论：

1. **cross-edge transport 在实现上可控。** 可以在不改 X、文本、预算和内部拓扑的前提下修改旧新连接，
   并严格守恒新行度数和旧端点容量。
2. **短程 Student response 不是可靠的长期分类 selector。** 即使局部 old/new InfoNCE gain
   同时为正，完整 Student 的最终 New F1 仍可能大幅下降。
3. **不能把 X+topology 简单串联成主方法。** v2 虽然更保守，但大多数 cell 只是回退到 X-only，
   唯一真正接受的 cell 也没有 New F1 收益。

因此当前不继续扫描更多 top-k、swap 数量、response 权重或更多 selector seed；那会继续在局部
启发式目标上调参，不能解决已观察到的长期目标错位。

## 8. 后续方法方向

下一步若继续研究梯度和旧记忆更新，应改变优化对象，而不是继续强化当前 edge selector：

1. 让 `X` 更新和最终使用的 synthetic propagation operator 在同一个 inner/outer loop 中计算；
2. 以新类分类塑性或 class-conditional student trajectory 作为外层目标，而不是仅用图文 InfoNCE
   的一步 loss reduction；
3. 对旧 rows 使用 trust-region，只允许小范围 revision，并加入 old-class retention constraint；
4. 对 P、X、A 做联合但有界的更新，任何候选都保留 C0 anchor 和精确回退；
5. 先做一个 cell 的 full-graph student-compatible pilot，再决定是否扩大到四个 cell。

在没有新的目标函数前，当前最合理的主线仍是：保留 `x-provenance r=0.10` 作为 X-only 诊断，
不把 topology selector 晋升进主方法；拓扑实验作为说明“为什么简单 response gate 不足”的负结果。

## 9. 复现产物

- 构造：[run_direct_topology_only.py](../../code/v2_llm_fusion/run_direct_topology_only.py)
- selector：[student_consistent_edge_swap.py](../../code/v2_llm_fusion/student_consistent_edge_swap.py)
- selector 测试：[test_student_consistent_edge_swap.py](../../code/v2_llm_fusion/test_student_consistent_edge_swap.py)
- topology v1：`results/direct_student_outer_20260928/topology_student_v1_*`
- topology v2：`results/direct_student_outer_20260928/topology_student_v2_*`
- joint v1：`results/direct_student_outer_20260928/joint_provenance_topology_v1_*`
- joint v2：`results/direct_student_outer_20260928/joint_provenance_topology_v2_*`
