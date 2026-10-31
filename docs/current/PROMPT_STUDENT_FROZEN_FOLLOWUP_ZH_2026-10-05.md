# Prompt Student 冻结与后续实验结果

日期：2026-10-05  
状态：完成独立 construction seed、bounded X refinement 和最小合成拓扑验证。

## 1. 冻结协议

当前统一 Student 为 `GraphTextPromptStudent`：

- 两层对称归一化 GCN；
- MiniLM 文本编码器冻结；
- 训练一个文本线性投影和 logit scale；
- 没有可训练分类器头；
- 测试时只计算图表示与类别 prompt 的余弦相似度；
- 同类别 memory 行在图文对比损失中视为多个正样本。

训练目标为：

\[
L=0.5L_{multi-positive}+1.0L_{prompt-CE}+0.1L_{margin}.
\]

所有本轮小图实验使用 Cora/Citeseer、三个类别顺序、三阶段、5% 累积 memory、六个 Student seeds。构造只读取当前 delta 和已材料化 parent，不读取历史 raw graph；report/test 节点不参与构造。edge-free 版本由 Student 自动补 self-loop。

## 2. Formal prompt-margin 独立 seed

canonical construction seed 为 `270927`，独立确认使用 `271927`。六个数据集-顺序 cell 的 pooled 结果如下：

| Construction seed | Final F1 | Accuracy |
|---|---:|---:|
| 270927 | 77.09 | 79.77 |
| 271927 | 77.01 | 79.83 |
| Difference | -0.08 | +0.06 |

独立 seed 没有推翻 formal prompt-margin，但顺序方差仍存在：Cora `order19612` 下降约 `1.45 pp`，其它 cell 变化较小。因此该方法可以冻结为统一参照，但不能宣称 construction seed 完全不敏感。

结果目录：

- `results/prompt_zero_shot_small_20261006_merged/`；
- `results/prompt_zero_shot_seed271927_cora/`；
- `results/prompt_zero_shot_seed271927_citeseer/`。

## 3. Formal prompt-margin + bounded X refinement

该版本保持 formal prompt-margin 材料化文本完全不变，只对当前阶段 CaT X 做有界二次 refinement：

\[
L_X=L_{CaT}+\lambda_tL_{transport/text}
+\lambda_a\lVert X-X_0\rVert_2^2,
\]

并限制每个 synthetic row 相对 formal X 的偏移不超过 `5% trust radius`。旧 memory、预算、文本、拓扑和访问边界均不变。

六个 cell 的相对变化：

| 指标 | 平均变化 | 正向 cell |
|---|---:|---:|
| Final F1 | -0.15 pp | 3/6 |
| Accuracy | -0.10 pp | 2/6 |
| Old F1 | -0.29 pp | 3/6 |
| New F1 | +0.06 pp | 4/6 |

按数据集 pooled：

| 数据集 | Formal F1 | Bounded-X F1 | Formal Accuracy | Bounded-X Accuracy |
|---|---:|---:|---:|---:|
| Cora | 83.81 | 83.57 | 85.49 | 85.40 |
| Citeseer | 70.19 | 70.12 | 73.96 | 73.84 |

结论：当前 X refinement 只带来极弱的新类信号，同时牺牲旧类保持，不能替代 formal prompt-margin。它说明 trust region 能限制破坏幅度，但不能自动保证 X 更新方向与最终 Student 任务一致。该版本保留为负结果和组件诊断，不进入主方法。

结果目录：

- `results/prompt_bounded_x_canonical_cora/`；
- `results/prompt_bounded_x_canonical_citeseer/`。

## 4. 最小合成拓扑实验

只改变 memory 拓扑，X、文本和 Student 完全保持 formal prompt-margin：

1. 每个 incoming block 内按文本 embedding 建立 text-kNN 边；
2. append 时保留旧旧边；
3. 每个新节点连接一个文本相似度最高的旧节点；
4. 不修改旧 X、旧文本和旧旧边；
5. 使用同一个 prompt-only Student 重新训练。

与 edge-free formal prompt-margin 的差值：

| 指标 | 平均变化 | 正向 cell |
|---|---:|---:|
| Final F1 | -7.77 pp | 0/6 |
| Accuracy | -7.18 pp | 0/6 |
| Old F1 | -7.14 pp | 0/6 |
| New F1 | -8.98 pp | 0/6 |

按数据集 pooled：

| 数据集 | Edge-free F1 | Text-kNN topology F1 | 差值 |
|---|---:|---:|---:|
| Cora | 83.77 | 75.40 | -8.38 |
| Citeseer | 70.19 | 63.03 | -7.16 |

结果说明：简单文本相似度边不能直接代表对 GCN 有益的消息传播。问题可能来自边选择与 X/text 语义不一致、跨阶段连接引入错误消息，以及合成图的度归一化和边预算没有经过 message compatibility 校验。拓扑模块不能直接加入主方法。

结果目录：

- `results/prompt_topology_canonical_cora/`；
- `results/prompt_topology_canonical_citeseer/`。

## 5. 当前主方法决定

当前暂定主方法冻结为：

> CaT feature condensation + formal prompt-margin payload assignment + prompt-only GraphText Student。

不纳入主方法：

- bounded X refinement；
- 直接 text-kNN synthetic topology；
- straight-through Sinkhorn 作为独立主方法结论；
- 动态 selector、GraphGPS Student 和无界 old-memory revision。

当前方法的贡献应表述为受限持续 TAG 凝聚下的语义 payload assignment 和统一 prompt-only 评价协议，不能声称已解决完整合成拓扑凝聚。

## 6. 后续唯一合理方向

如果继续补拓扑，只实现一个受限 message-aware 版本：

1. 先由 X/text 相似度产生很小的候选旧节点池；
2. 使用固定的 teacher message compatibility 评价候选边；
3. 保持 cross-edge 数、旧旧边和 degree capacity 不变；
4. 只接受能降低局部 prompt/message loss 的 edge swap；
5. 任意风险升高时回退到 edge-free 或 CaT topology；
6. 不再改变 X、文本、预算和 Student。

若该受限版本仍不能在至少两个数据集上恢复 edge-free 基线，则论文应把拓扑定义为明确的失败边界，而不是继续堆叠拓扑启发式。

代码入口：

- `code/v2_llm_fusion/graphtext_prompt_student.py`；
- `code/v2_llm_fusion/run_prompt_zero_shot_small_20261006.py`；
- `code/v2_llm_fusion/x_conditioned_payload_partition.py`。
