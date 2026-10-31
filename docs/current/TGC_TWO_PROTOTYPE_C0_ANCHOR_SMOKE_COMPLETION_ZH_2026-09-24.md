# TGC 双原型与 C0 拓扑风险约束小图验证报告

日期：2026-09-24  
状态：完成三格受控 smoke，未通过主方法晋升门禁

## 1. 实验目的

上一轮局部双原型构造在一个 Cora 条件下出现正向信号，但在 Citeseer 和另一条 Cora 顺序上出现 old-class F1 下降。本轮只验证一个问题：

> 如果保留 C0 拓扑，只对传播负荷较高的合成槽位减少内容更新幅度，能否改善历史保持并保留新类塑性？

本轮没有加入 response selector、GraphGPS、额外预算、拓扑重构或历史原图回读。

## 2. 先修正的协议问题

第一次运行发现，同一个 two-prototype runner 与此前结果的文本内容不一致：旧远程快照中部分拼接文本超过 320 字符，而当前协议要求严格 320 字符。原因是远程工作快照中的依赖文件没有与本地当前版本同步，而不是方法本身的性能波动。

因此本报告只采用 `*_sync` 与 `*_gate` 目录中的重跑结果。此前记录的 Cora `+2.328 pp` 不再作为本轮可比数字，不能与严格 320 字符协议混用。

## 3. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集/条件 | Cora `order19601`、Cora `order19602`、Citeseer `order19601`，stage 1 |
| memory | 5% growing memory，总 rows 与 C0 相同 |
| 历史访问 | 不访问历史完整 raw graph；旧 synthetic prefix 不变 |
| 拓扑 | C0 `edge_index` 完全复制，不增删边 |
| Student | 原始两层 GCN、trainable text projection、双向 InfoNCE |
| batch/steps | 与对应 C0 完全一致，batch=16，steps=`protocol_v3.student_steps` |
| 文本 | 同一 source assignment、top-5、320 字符、冻结 MiniLM 重编码 |
| report seeds | 150、151 |
| 选择 | 无 report-F1 selector；所有 arm 直接成对评估 |

## 4. 方法臂

所有非 C0 arm 先构造同一个局部双原型候选：每个新类选择一个高离散 C0 source group 拆成两个原型，同时合并该类最相近的两个 group，以保持该类 rows 不变；三个 proposal 用 joint center 对齐回原 C0 row slots。X 与文本使用同一最终 source piece。

之后只改变候选内容写入 C0 row 的幅度：

1. `two_prototype`：完整写入双原型 X/text。
2. `anchor050`：所有被修改槽位使用 0.50 的候选内容和 0.50 的 C0 内容。
3. `anchor025`：所有被修改槽位使用 0.25 的候选内容和 0.75 的 C0 内容。
4. `degree_anchor`：按 C0 新块 degree 连续降低更新系数，使用
   `alpha = 0.35 + 0.65 * (1 - degree / max_new_degree)`。
5. `degree_gate`：只修改 degree 不高于当前新块 degree 中位数的槽位；高风险槽位完全回退 C0 内容。

X 使用同一 `alpha` 做线性混合；文本使用同一 `alpha` 分配旧/新字符预算，再经过冻结 MiniLM 重编码；拓扑、rows、mass 总预算不变。

## 5. 结果：累计 Macro-F1

数字为相对同一 cell 的 C0 差值，单位为百分点；绝对 F1 放在括号中。

| 条件 | two-prototype | anchor050 | anchor025 | degree-anchor | degree-gate |
|---|---:|---:|---:|---:|---:|
| Cora/order19601 | +0.147 (59.465) | -1.761 (57.557) | -0.678 (58.640) | -1.945 (57.374) | **+0.998 (60.316)** |
| Citeseer/order19601 | -3.823 (55.847) | -2.268 (57.402) | -1.577 (58.094) | -4.094 (55.576) | -4.012 (55.659) |
| Cora/order19602 | +0.384 (58.693) | -0.379 (57.931) | +0.284 (58.593) | -2.299 (56.011) | -0.716 (57.594) |

C0 绝对 F1 分别为：Cora/order19601 `59.318`、Citeseer/order19601 `59.670`、Cora/order19602 `58.309`。

## 6. Old/New 塑性分解

| 条件/arm | old-F1 差值 | new/current-F1 差值 | real new margin 差值 |
|---|---:|---:|---:|
| Cora/order19601 two-prototype | +0.594 | -1.494 | +0.880 |
| Cora/order19601 degree-gate | **+5.352** | -1.323 | +0.571 |
| Citeseer/order19601 two-prototype | -8.246 | +1.332 | +0.363 |
| Citeseer/order19601 degree-gate | -7.256 | -0.248 | +0.545 |
| Cora/order19602 two-prototype | -0.419 | +0.345 | -0.245 |
| Cora/order19602 degree-gate | -1.669 | -0.583 | -0.040 |

## 7. 结果解释

### 7.1 硬风险门控有机制信号，但不是稳定算法

Cora `order19601` 中，degree-gate 只改低传播负荷槽位，使 old-F1 提升 `+5.352 pp`，累计 F1 提升 `+0.998 pp`。这说明“不要把新语义直接写入高传播负荷 C0 槽位”是一个真实的风险控制信号。

但同一规则在 Citeseer 和 Cora `order19602` 上失败，累计差值分别为 `-4.012 pp` 和 `-0.716 pp`。因此它只能作为结构诊断结论，不能称为跨数据集的自适应算法。

### 7.2 anchor 主要改变 old/new trade-off，没有解决根本错配

统一 anchor 在 Citeseer 中把 two-prototype 的 old-F1 损失从 `-8.246 pp` 缓解到 `-4.442 pp`（anchor025），但仍未恢复到 C0；在 Cora 条件下又压低了原型带来的新类收益。连续 degree anchor 的表现更差，说明简单按 degree 做连续缩放并不等价于可靠的风险估计。

### 7.3 当前主要问题不是“更新幅度不够”，而是 row slot 语义与 C0 message role 不一致

同一新类内容即使提高 real-new margin，也可能被写入承担不同邻居传播角色的 C0 row。由于本轮拓扑完全固定，student 看到的是：

1. 节点 X/text 已经换成新的局部 prototype；
2. 该节点在 GCN 中仍保留旧 C0 的邻居和 degree role；
3. 高传播负荷槽位的消息会影响多个旧/新节点；
4. 因而内容改善与历史保持之间出现强 trade-off。

anchor 只能减小这个冲突，不能让内容和 topology role 真正一致。

## 8. 晋升结论

本轮不满足主方法晋升条件：

- 没有至少两个数据集/独立顺序稳定正收益；
- 最差 cell 仍低于 `-2 pp`；
- degree-gate 的正收益只出现在一个 Cora 条件；
- 连续 anchor 和硬 gate 都不能稳定同时改善 old retention 与 new plasticity。

因此：

1. C0 继续作为当前主方法和比较基线。
2. 局部双原型保留为“条件性内容改造 + 拓扑角色风险”的机制实验。
3. anchor/degree-gate 不进入主方法，只进入负结果和机制分析。
4. 暂不上完整三顺序、多阶段或大图扩展。

## 9. 后续建议

下一步不宜继续扫描更多固定 alpha。更有价值的是做一个小规模的 slot-role 诊断：固定同一双原型内容，只比较把它写入低 degree、高 degree、低 cross-load 和高 cross-load 的 C0 槽位，报告 old/new message drift 与 student F1。若确认真正关键的是“槽位角色”而非单纯 degree，后续才考虑允许有限拓扑重分配；否则回到 C0 并把该现象作为持续 TAG 凝聚中的负结果。

## 10. 产物与验证

- Runner：[run_tgc_two_prototype_anchor_smoke.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_tgc_two_prototype_anchor_smoke.py)
- Remote wrapper：[tgc_two_prototype_anchor_smoke_remote.py](D:/Aprojects/continue_tag_condensation/tools/tgc_two_prototype_anchor_smoke_remote.py)
- 结果目录：[tgc_two_prototype_anchor_smoke_20260924](D:/Aprojects/continue_tag_condensation/results/tgc_two_prototype_anchor_smoke_20260924)
- 本地语法检查：`py_compile` 通过
- 构造审计：old prefix、C0 topology、总 rows、报告 seed 和 student steps 均保持一致
