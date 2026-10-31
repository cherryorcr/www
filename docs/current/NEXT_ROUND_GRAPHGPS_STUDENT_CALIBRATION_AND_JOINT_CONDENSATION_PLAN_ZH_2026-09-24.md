# 下一轮推进计划：GraphGPS Student 校准与联合 Student-Response 凝聚

日期：2026-09-24  
状态：P1 GraphGPS/GCN 静态校准已完成；P2 联合 student-response 尚未启动

## 1. 为什么需要重新规划

当前项目同时存在两个可能的瓶颈：

1. 下游 student 可能过弱。当前使用两层 GCN、冻结 MiniLM、双向 InfoNCE 和 zero-shot prompt 分类，不能充分利用压缩图中的长程结构。
2. 凝聚 memory 仍然存在新类塑性不足。R11 内容 + C0 拓扑的最终阶段平均相对 C0 为 `+0.723 pp`，但 new F1 平均下降 `-1.314 pp`；R17 teacher-message topology gate 的最终平均仅 `+0.247 pp`，new F1 下降 `-2.362 pp`。

此前 R2B 的 student 反向修订、R17 的拓扑 gate 和 R6/R7 的 teacher proxy 校准已经说明：

- 只优化 X 不足以保证完整 student 受益；
- 只优化拓扑不能解决新类塑性；
- 更低 teacher reconstruction/message loss 不能替代实际 student 重训收益；
- teacher 的正向信号主要出现在 X、文本和拓扑共享 assignment 的联合材料化中。

因此下一轮不能直接继续扫描 alpha、edge swap 或单独修改 X，而要先建立一个可信的 student 模型层，再做共享 assignment 驱动的联合凝聚。

## 2. 总体路线

执行顺序固定为：

```text
C0/现有协议冻结
        |
        v
GraphGPS 与两层 GCN 的静态 student 校准
        |
        v
确定最终凝聚目标 student 与输入协议
        |
        v
共享 assignment 的联合 student-response 材料化
        |
        v
小图多顺序确认
        |
        v
链路预测与长链协议
        |
        v
大图验证与论文主方法决策
```

本路线中 GraphGPS 首先作为**模型诊断工具**，只有静态校准通过后，才允许作为凝聚过程的 teacher/student。

## 3. 阶段 P0：冻结公共协议

### 3.1 固定方法

- C0：可靠基线。
- R11 内容 + C0 拓扑：当前最佳内容诊断版本。
- k-center/medoid：强 coreset 参照。
- 完整合法 raw pool：同 student 数据上限参考。
- 官方 GraphCLIP checkpoint：外部预训练模型参考，不纳入同 student 主表。

### 3.2 固定评价

- Cora、Citeseer。
- `order19601/19602/19603` 作为现有开发顺序。
- 5% growing memory。
- 报告 final F1、stage-average F1、old F1、new F1、最差顺序、构造成本和 student 训练成本。
- report/test 标签只用于最终评价，不用于 memory 构造、候选选择或 checkpoint 选择。

### 3.3 固定历史访问边界

- 只能读取旧 synthetic memory 和当前 raw delta。
- 历史完整 raw graph 不回读。
- 第一版联合方法不修改旧 synthetic rows 和旧旧边，先隔离新类塑性问题。

## 4. 阶段 P1：GraphGPS 静态 student 校准

这一阶段不修改凝聚算法，只回答：当前性能不足是否主要来自两层 GCN student。

### 4.1 三种 GraphGPS 评测必须分开

#### A. 官方 GraphCLIP checkpoint

- 使用官方预训练 GPS + 文本塔；
- 不训练、不微调、不选择 checkpoint；
- 使用完整真实图和官方推理输入；
- 只作为外部参考。

当前报告划分上的结果为 Cora `58.98`、Citeseer `57.76`，不能直接作为凝聚方法的同 student 对照。

#### B. 从头训练 GraphGPS student

固定同一 memory，分别训练：

- 完整 raw pool；
- C0 synthetic memory；
- R11 内容 + C0 拓扑；
- k-center coreset。

所有方法使用相同 GraphGPS 初始化、batch、steps、report seeds 和 prompt 规则。

#### C. GraphGPS-targeted condensation

只有 B 组确认 GraphGPS 能稳定消费 synthetic memory 后，才进入这一分支。此时 GraphGPS 才作为凝聚 teacher/student，而不是把 GCN 构造结果直接迁移到 GraphGPS 上。

### 4.2 GraphGPS 输入协议

不能只替换模型名称，必须明确：

- full-graph 还是 ego-subgraph 输入；
- random-walk positional encoding 的计算图；
- root-node 标记；
- edge feature；
- 节点输出与图输出方式；
- synthetic graph 和真实评估图是否使用相同输入处理。

第一版建议建立两条明确路径：

1. **GraphGPS-scratch full-graph**：尽量保持当前 transductive node-level 任务，作为同 student 公平比较。
2. **official GraphCLIP ego/RW-PE**：只作为外部预训练参考。

二者不能混入同一主表。

### 4.3 P1 决策规则

| 观察结果 | 解释 | 后续决策 |
|---|---|---|
| raw 和 synthetic 都明显提升 | 两层 GCN 是主要瓶颈 | GraphGPS 可成为后续目标 student |
| raw 提升但 synthetic 不提升 | 凝聚图无法支持长程/PE | 优先改联合凝聚，不直接宣称模型瓶颈 |
| raw 和 synthetic 都不提升 | GraphGPS 输入或训练协议有问题 | 暂不纳入主方法，先修输入协议 |
| GCN 与 GraphGPS 相对保留率相近 | 凝聚算法问题大于 student 问题 | 继续以 GCN 为主，GraphGPS 作稳健性验证 |

## 5. 阶段 P2：共享 assignment 联合 student-response 凝聚

该阶段的最终目标不是“优化 X”，而是优化一个共享 assignment (P)，并由同一个 (P) 同时生成 X、文本和拓扑。

### 5.1 初始化

对当前增量块使用已有 teacher 联合构造：

\[
P_0=0.75P_{C0}+0.25P_{joint}
\]

它只作为稳定的初始点，不直接把 teacher loss 当作最终选择标准。

### 5.2 共享材料化

每个候选 assignment (P) 都必须同时材料化为：

\[
M(P)=\{X(P),T(P),A(P)\}
\]

- (X(P))：按 (P) 加权聚合当前 delta 特征；
- (T(P))：按 (P) 选择文本来源，在固定 token budget 下拼接并重新编码；
- (A(P))：在 C0/R11 合法候选边集合和 C0 边数、度数、跨块容量约束下生成 binary topology。

不允许分别选出“最好的 X、最好的文本、最好的边”后再拼接。

### 5.3 Student inner loop

对实际硬材料化后的 (M(P)) 训练临时 student：

\[
\theta_{k+1}=\theta_k-\eta\nabla_\theta
\mathcal L_{InfoNCE}(\theta_k,M(P))
\]

第一版只测试 `K=1/2` 个 inner steps。R2B 已显示更长的 `K=4/8` 并不自动带来更好结果，不应一开始扩大 horizon。

### 5.4 联合 response 目标

候选的评价同时考虑：

- 新类图文 margin；
- 旧 memory 保持；
- 不同 student seed/horizon 的方差；
- 额外构造成本。

\[
Q(P)=\Delta R_{new}
-\lambda\max(0,-\Delta R_{old})
-\mu V_{response}
-\nu C(P)
\]

这里不使用最终 dev/test F1 选候选。每个候选最终都必须重新硬材料化，并在真实 synthetic graph 上验证。

### 5.5 更新方式

可以使用连续 assignment 产生候选方向，但最终评价必须采用离散实际结果：

1. 用 (P) 产生一个候选方向；
2. 生成实际合成 X、实际拼接文本和实际 binary edge_index；
3. 在实际 memory 上运行短 student response；
4. 只有完整 tuple `(X, text, topology)` 同时满足约束时才接受。

如果候选失败，整体回退到 R11 内容 + C0 拓扑，而不是只保留其中一个局部更新。

## 6. 阶段 P3：小图验证矩阵

### 6.1 主比较

| 方法 | 目的 |
|---|---|
| C0 | 可靠基线 |
| R11 内容 + C0 拓扑 | 当前最佳内容参照 |
| k-center/medoid | 强 coreset 参照 |
| 完整 raw pool | 同 student 上限参考 |
| 共享 assignment 联合 student-response | 新主方法候选 |

### 6.2 解释性消融

消融只用于解释联合机制，不作为最终方法的开发顺序：

- 去掉 student response；
- 去掉共同 assignment；
- 固定文本，只优化 X；
- 固定 X，只改变文本来源；
- 固定内容，只改变拓扑；
- 去掉 old-retention 项；
- 去掉 C0 assignment anchor。

### 6.3 晋升门槛

新方法至少需要满足：

- final 平均相对 C0 `>= +0.5 pp`；
- stage-average 不低于 0；
- new F1 平均不下降；
- 最差 final cell 不低于 `-2 pp`；
- 至少 4/6 final cells 非负；
- 相对 C0 的额外构造成本不超过 2 倍；
- 在 Cora、Citeseer 和多个顺序上成立。

如果只提高 old F1 而 new F1 继续下降，不进入主方法。

## 7. 阶段 P4：任务与协议扩展

只有 P3 通过后才执行：

1. link prediction：比较 C0、k-center、联合方法在 edge recovery 和 LP AUC/AP 上的差异；
2. 长链协议：base + 4 或 5 stages；
3. 独立类别顺序和构造 seed；
4. 更高 memory rate；
5. 大图 Arxiv、BookChild、SportsFit。

如果 P3 没有通过，不能用大图结果为小图失败的方法背书；大图只作为后续诊断，不作为主线晋升依据。

## 8. 最终决策树

### 情形一：GraphGPS 只提高 raw，不提高 synthetic

说明主要问题仍是凝聚算法与 PE/长程结构不匹配。继续研究联合 assignment，但不把 GraphGPS 当作性能补丁。

### 情形二：GraphGPS 同时提高 raw 和 synthetic，但联合凝聚仍不稳定

说明当前两层 GCN 是瓶颈，但联合凝聚目标仍然不可靠。论文应分开报告模型校准和凝聚负结果。

### 情形三：联合 student-response 在 GraphGPS 和 GCN 上都稳定提升

才可以把“共享 assignment + student-compatible joint materialization”作为完整算法主线。

### 情形四：所有 student 上联合方法都不能稳定超过 C0

保留 C0 为可靠方法，将 teacher 联合 assignment、student response 失配和 topology gate 失败写成机制分析与负结果，不再继续堆叠局部模块。

## 9. 本轮不做的事情

- 不立即把官方 GraphCLIP checkpoint 当作主 student；
- 不把 GraphGPS 和两层 GCN 的分数混成一张主表；
- 不先分别优化 X、文本、拓扑再拼接；
- 不继续扫描 alpha、edge swap 或 weighted adjacency；
- 不在联合方法尚未通过小图验收前上大图；
- 不使用最终 F1 选择候选。

## 10. 计划产出

1. GraphGPS/GCN 静态 student 校准表；
2. raw pool、C0、R11、k-center 在两种 student 下的保留率；
3. 共享 assignment 联合材料化的算法实现；
4. student response 与最终 F1 的相关性及 regret 报告；
5. 小图多顺序晋升或失败报告；
6. 通过后再生成长链、大图和论文主表。

## 11. P1 实际验收结果（2026-09-24）

P1 已完成 2 个数据集、3 个阶段、4 个 arms、6 个 student seeds 的正式矩阵，共 144 个 seed runs。完整数字见：

`GRAPHGPS_P1_CALIBRATION_REPORT_ZH_2026-09-24.md`

结论不是 GraphGPS 已经晋升为新主 student，而是：

- runner、GraphGPS GPS tower、2-hop ego graph、RW-PE、冻结 MiniLM 和 report-dev 推理均通过；
- GraphGPS scratch student 的长链 seed 方差较大，不能直接替换 GCN；
- raw pool 不是当前 zero-shot InfoNCE 协议下的理论上限，Cora 多阶段 raw 反而低于 synthetic C0，Citeseer 只在最终阶段略高；
- R11 和 k-center 均未形成跨阶段、跨数据集的稳定晋升；
- 不启动 GraphGPS-targeted joint student-response 凝聚。

因此下一步改为：先做一次小范围 GraphGPS stage-wise training calibration；若仍不稳定，则保留 GraphGPS 为诊断分支，回到 GCN/C0 进行共同 assignment 联合材料化实验。P1 结果不能支持将 GraphGPS、raw pool 或单次 R11/k-center 高分写成新的主方法优势。
