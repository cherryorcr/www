# 链路预测：联合核结构签名必要性对照协议

日期：2026-09-20。在本轮新增臂分数产生前冻结。沿用前轮已知 edge splits，属于机制开发诊断，不是新的独立泛化确认。

## 1. 问题与矩阵

前轮已验证 coverage 相对 C0 的链路预测收益，但原始联合核未稳定改善。本轮仅回答：度/覆盖约束下，真实图结构签名是否必要；C0 的文本核加上同样约束能否达到类似收益？

| 连边核 | 度均衡 | 度均衡 + coverage |
|---|---|---|
| T：文本 cosine，权重 (0,1,0) | 新增 `t_degree` | 新增 `t_coverage` |
| XT：X/text cosine，权重 (4/7,3/7,0) | 新增 `xt_degree` | 新增 `xt_coverage` |
| XTG：权重 (0.4,0.3,0.3) | 复用前轮 `degree_balanced` | 复用前轮 `coverage` |

另外保留原 C0、无度约束 joint_kernel、full_raw_equal、无训练余弦对照；新增 `c0_replay` 重复训练作为数值波动诊断，不是新算法。每个数据集新增 4×3 splits×3 stages×6 seeds=216 次候选训练，另有 54 次 C0 重复训练；两图合计 **432 次候选训练 + 108 次 C0 重复训练 = 540 fits**。

## 2. 固定项

- Cora/Citeseer、canonical 三阶段、5% 合成节点、edge split seeds 29001/29002/29003、student seeds 100..105。
- 原 LP 脚本的 PyG GAE、输入 X+text、hidden384/out128、BCE、Adam0.001、256 updates、相同边 batch 规则；不得为了新臂改变 student。
- 使用已经审计的前轮 train-only split 与真实图推理输入；eval tensors 和父实验 manifest/result/audit 的 SHA256 冻结。
- 四个新臂直接读取 C0 合成图文行的新后缀，不读取原始图或 graph signature；真实图仅供共同的下游评估器。
- old rows、old-old edges 不可修改；各臂的 X/text/texts/mass/depth/stage/rows/row order 与 C0 完全一致，各阶段无向边数也一致。
- 度约束/coverage 的实现直接复用 `safe_joint_kernel_append.select_internal` 和 `cross_pairs`，只传入不同核权重。新 block 内度上限 2、在当前大小条件下构成闭环，跨块每个新行连接一个旧行、旧行容量 ceil(new/old)。coverage 惩罚仍为 0.15，不新增调参。
- T 的评分与 C0 同为归一化文本余弦，但增加了约束的选边过程；不把它描述成完全相同的 top-k 实现。XT 去掉 G 后按原 X:T 比例归一化，不保留无用签名占字节。
- 约束相同不表示每个节点的最终度序列严格相同，故本轮不是同度随机化消融。

## 3. 比较与审计

主对照：`t_degree - C0`；`degree_balanced - xt_degree`（固定约束看结构签名的条件增益）；`coverage - xt_coverage`；`t_coverage - t_degree`、`xt_coverage - xt_degree`（覆盖惩罚条件增益）。报告两个数据集全部对照，不根据 test 选核或超参数。

报告最终/阶段平均 AUC、AP，旧/新关系切片，各 split 的成对差值、逻辑字节与训练时间；复用结果必须标记非本轮重新训练。保留文本余弦强对照和完整图非理想上限的限定。

审计包括：父实验哈希、共同 eval tensors、split 无泄漏断言、无原始图/签名依赖、与 C0 图文行一致、旧块冻结、等边数、度与跨块容量约束、memory 重建、checkpoint SHA、保存分数指标精确复算、重载分数 allclose，以及 CPU 双精度指标控制。

遵循前轮公开的浮点审计说明：指标排序差超过 1e-4 时保留告警，不以放宽阈值隐藏；分数数值等价与严格排序检查分列。C0 重复训练差值仅用于展示数值波动，不据此选择较好的 C0。

## 4. 结论边界

如果 T/XT 在相同约束下仍明显较差，支持结构签名在当前设置中的条件效用，但不证明其不可替代或具备文献新颖性；如果接近或更好，则优先保留更简单的候选，而非继续把整体增益归因于复杂核。三次已知 edge splits 不足以决定大图或分类主方法晋升。

本轮不增加 harder negatives、其它 rate/order/student、大图或随机同度拓扑；这些仍是后续任务，避免一次同时改变多个实验问题。
