# TGC-R17 Teacher-Message Topology Gate 完成报告

日期：2026-09-24  
状态：实验完成，审计通过，未晋升为主方法

## 1. 实验目的

R16 已经显示：R11 的内容构造有局部正向信号，但直接改变粗粒度拓扑可能损害新类塑性。本轮固定 R11 的合成节点特征和文本，只在 C0 的候选边集合内使用 teacher message compatibility 选择少量安全的内部 edge swap，检验更细粒度的拓扑门是否能够保留内容收益并修复传播问题。

## 2. 固定协议

- 数据集：Cora、Citeseer。
- 类别顺序：`order19601`、`order19602`、`order19603`。
- 任务：base + 2 incremental，评价 stage 0/1/2。
- memory：5% growing synthetic memory。
- 内容：R11 `anchor025`，固定 `alpha_content=0.25`。
- 文本：最多 5 个来源、约 320 字符，拼接后重新使用冻结 MiniLM 编码。
- 学生：沿用 C0 的模型、batch、steps、report seeds。
- teacher：seed `9000/9001`，compact sampler，128 steps，batch 64。
- 可访问数据：旧 synthetic memory、当前 delta 和封存的 R11 assignment；历史完整 raw graph reads = 0。
- 历史记忆：不修改旧 synthetic rows 和旧旧边。

### 实验臂

| 方法 | 内容 | 拓扑 |
|---|---|---|
| C0 | C0 | C0 |
| `anchor025` | R11 | R11 |
| `anchor_content_c0_topology` | R11 | C0 |
| `message_gate_1` | R11 | 最多 1 次 teacher-message-compatible internal swap |
| `message_gate_2` | R11 | 最多 2 次 teacher-message-compatible internal swap |

门控规则同时要求：teacher message loss 下降、old-node message risk 不上升、C0 的边数/度数/跨块容量约束保持；不满足时回退 C0。跨块边还要求超过固定 margin，但本轮没有任何跨块边被接受。

## 3. 主结果

表中 `final delta` 是相对于同一 dataset/order 的 C0 最终阶段 Macro-F1 差值；`stage mean` 是所有评估阶段的平均差值。

| 方法 | 最终 Macro-F1 | final delta | old F1 delta | new F1 delta | stage mean delta | 正向 final cells | 最差 final delta | student 秒/fit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0/6 | 0.000 | 3.602 |
| R11 `anchor025` | 51.656 | +0.142 | +1.629 | -2.059 | -0.856 | 3/6 | -2.104 | 3.601 |
| R11 内容 + C0 拓扑 | 52.237 | **+0.723** | +2.103 | -1.314 | -0.314 | 4/6 | -3.155 | 3.601 |
| `message_gate_1` | 51.557 | +0.042 | +1.845 | -2.675 | -0.555 | 4/6 | -4.919 | 3.607 |
| `message_gate_2` | 51.761 | +0.247 | +1.995 | -2.362 | -0.682 | 4/6 | -4.919 | 3.607 |

### `message_gate_2` 最终六个 cell

| 数据集 | 顺序 | final delta | new F1 delta | old F1 delta |
|---|---|---:|---:|---:|
| Cora | `order19601` | +2.025 | +1.346 | +2.534 |
| Cora | `order19602` | -4.919 | -13.434 | +1.467 |
| Cora | `order19603` | +2.610 | +1.164 | +3.694 |
| Citeseer | `order19601` | +0.945 | -0.942 | +1.889 |
| Citeseer | `order19602` | +1.863 | +0.310 | +2.640 |
| Citeseer | `order19603` | -1.041 | -2.615 | -0.254 |

## 4. 构造审计结果

- Cora/Citeseer 的 prepare、seal、evaluate、audit 全部退出码为 0。
- 6 个 dataset/order cell 的 reload audit 全部通过。
- `historical_raw_reads=0`，没有访问历史完整原始图。
- `report_labels_used=false`，构造和门控没有使用报告标签。
- teacher-enabled 的 12 个增量构造点中，`message_gate_1` 接受 8 个内部 swap，4 个回退；`message_gate_2` 接受的内部 swap 总数为 12，8 个构造点发生了修改，4 个回退。
- 两个 gate 的跨块边改动数均为 0。
- 12 个 teacher-enabled 构造点中有 8 个满足最终 message loss 严格下降；old message risk 增加次数为 0。
- teacher 额外构造时间平均约 0.76 秒/增量构造点；student 训练时间约 3.60 秒，因此 gate 的额外成本主要来自构造阶段，而不是 student 评估阶段。

## 5. 结果解释

### 5.1 已确认的正信号

teacher-message gate 不是空操作。它能够在 C0 的结构边界内找到少量内部边交换，真实降低局部 message loss，并且没有增加 old-node message risk。这说明 message compatibility 可以作为拓扑候选的局部安全约束，优于单纯用 X/text cosine 或度数保持进行重连。

### 5.2 尚未证明的部分

局部 teacher message loss 的下降不能稳定预测最终 student 的 Macro-F1，尤其不能稳定预测 new-class plasticity。`message_gate_2` 的最终平均提升只有 +0.247 pp，低于 +0.5 pp 晋升门槛；stage mean 为 -0.682 pp；new F1 平均下降 -2.362 pp；最差 final cell 为 -4.919 pp。因此它不满足当前主方法晋升条件。

失败的主要原因不是 gate 没有执行，而是 gate 优化的目标仍然是局部、短 horizon 的 teacher message compatibility，而最终评价受以下因素共同影响：

1. R11 内容本身对新类塑性的影响仍随类别顺序变化；
2. 内部 edge swap 只改变局部传播路径，不能修复合成 X、文本和拓扑之间更大的不一致；
3. old risk 不上升不等于 new-class signal 增强；
4. teacher 的 128-step proxy 没有覆盖 student 的长训练轨迹和最终 zero-shot 分类竞争；
5. `order19602` 的 Cora 负 cell 表明，局部拓扑安全性无法消除顺序特定的塑性失败。

因此，本轮结论是：

> teacher message compatibility 是一个有解释力的局部拓扑诊断和安全约束，但当前形式还不是可靠的全局选择准则。

## 6. 与晋升门槛对照

| 门槛 | 要求 | R17 `message_gate_2` | 结论 |
|---|---:|---:|---|
| final 平均收益 | 至少 +0.5 pp | +0.247 pp | 未通过 |
| stage 平均 | 不低于 0 | -0.682 pp | 未通过 |
| new F1 | 平均不下降 | -2.362 pp | 未通过 |
| 最差 final cell | 不低于 -2 pp | -4.919 pp | 未通过 |
| 正向 final cells | 至少 4/6 | 4/6 | 通过 |
| audit | 全部通过 | 全部通过 | 通过 |

## 7. 当前方法决策

### 保留

- C0 作为可靠主线和所有后续比较锚点。
- R11 内容 + C0 拓扑作为当前最强的局部内容诊断版本，但不称为已确认主方法。
- R16 的度保持重连负结果。
- R17 的 teacher-message gate 作为“局部 message loss 可下降、但 downstream plasticity 不稳定”的机制边界结果。

### 暂不纳入主方法

- `message_gate_1`。
- `message_gate_2`。
- 任意跨块边重连。
- 继续增加 edge swap 次数或继续扫描 alpha。

## 8. 后续推进建议

1. 停止继续堆叠拓扑 gate；不再把更多 edge swap 当作下一步主线。
2. 将 R17 的 message loss、old risk、new F1 和最终 Macro-F1 做相关性分析，确认 proxy 失配的具体来源。
3. 优先处理新类塑性：让候选构造或 student-compatible objective 直接约束 new-class response，而不是只约束 teacher message。
4. 若继续做拓扑，只做一个受控诊断：固定 X、文本、行顺序和边数，比较 message loss 与 downstream new F1 的关系；不再把局部 loss 下降直接作为晋升依据。
5. 论文中可将 R17 写入机制分析和负结果部分，不能作为主算法性能提升证据。

## 9. 可复现实验文件

- [R17 runner](../../code/v2_llm_fusion/run_tgc_r17_message_topology.py)
- [R17 remote runner](../../tools/tgc_r17_remote.py)
- [R17 protocol](TGC_R17_MESSAGE_TOPOLOGY_PROTOCOL_ZH_2026-09-24.md)
- [R17 results](../../results/tgc_r17_message_topology_20260924b/)
- [R17 methods summary](../../results/tgc_r17_message_topology_20260924b/methods.csv)
- [R17 cells summary](../../results/tgc_r17_message_topology_20260924b/cells.csv)
- [R17 audit summary](../../results/tgc_r17_message_topology_20260924b/summary.json)
