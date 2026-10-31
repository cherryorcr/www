# TGC Message-Conditioned Synthetic Feature Projection 完成报告

日期：2026-09-26  
状态：**构造审计和独立 Student 评估已完成；该版本不晋升为主方法。**

## 1. 实验问题

本轮只检验一个明确假设：如果把新合成节点的特征 `X` 投影到当前增量真实子图的一阶/二阶 message target，是否能在不改变 C0 拓扑和文本的情况下改善 Student 的新类塑性。

这不是 selector，也没有使用 report F1 选择候选。它保持 C0 的历史 memory、边集合和文本，只修改当前 changed synthetic rows 的 `X`。

## 2. 方法流程

对每个数据集、类别顺序和增量阶段：

1. 使用冻结的 two-prototype structural proposal 生成新块；
2. 读取当前合法增量的 induced subgraph，计算 raw one-hop/two-hop message target；
3. 在 C0 固定拓扑的 synthetic slots 内，对每个 changed row 做闭式特征投影；
4. 以 source cluster dispersion 作为 trust-region，限制 `X` 位移；
5. 同步更新所有 changed rows 后重新审计 message error；
6. 保持文本、mass、depth、stage、旧 prefix 和 `edge_index` 与 Identity/C0 一致；
7. 以相同 v3 Student 协议重训 `C0`、`Identity` 和 `message_x`。

局部投影目标可写为：

\[
\min_{x_i}\;\|M_1^{C0}(x_i)-\bar m_{1,i}\|^2+
\|M_2^{C0}(x_i)-\bar m_{2,i}\|^2
\quad\text{s.t.}\quad \|x_i-x_i^{proposal}\|\le r_i.
\]

其中 `M_1/M_2` 是固定 C0 拓扑下的一阶/二阶传播贡献，`\bar m` 来自当前增量 induced subgraph，`r_i` 由源簇离散度给出。

## 3. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19605`、`order19606`、`order19608` |
| 阶段 | stage 1/2；base prefix 已冻结 |
| memory | 5% growing synthetic memory |
| arms | C0、Identity、Message-X |
| Student | 原两层 GCN + trainable text projection + frozen MiniLM |
| 目标 | 双向图文 InfoNCE，不使用分类标签训练 |
| batch | 16 |
| Student seeds | `150..155` |
| steps | v3 `student_steps(len(memory))` |
| 选择 | 无 report-F1 选择；所有 memory 封存后直接评估 |
| 访问约束 | 只读旧 synthetic memory 与当前 raw delta；历史 raw reads = 0 |

## 4. 构造审计

12 个 cell 共修改 81 个 changed rows。

| 审计项 | 结果 |
|---|---:|
| changed rows | 81 |
| 全局 message error 均下降的 rows | 81/81 |
| mean error before | 1.374 |
| mean error after global | 0.968 |
| 相对下降 | 约 29.5% |
| trust-region 触顶/收缩 rows | 33/81 |
| mean applied displacement | 1.393 |
| historical raw reads | 0 |
| 旧 prefix / C0 topology / text 是否改变 | 否 |

因此实现和局部代理目标本身是有效的：不是代码没有改变，也不是 message error 统计错误。

## 5. Student 结果

以下均为 6 个 Student seeds 的均值；差值为 `message_x - C0`，单位为百分点。

| 数据集 | 顺序 | stage | C0 F1 | Message-X F1 | 累计 F1 差 | old F1 差 | new F1 差 |
|---|---|---:|---:|---:|---:|---:|---:|
| Citeseer | order19605 | 1 | 53.379 | 52.547 | -0.832 | -3.483 | +1.819 |
| Citeseer | order19605 | 2 | 55.330 | 54.719 | -0.611 | +0.182 | -2.195 |
| Citeseer | order19606 | 1 | 65.691 | 57.944 | -7.747 | -1.781 | -13.713 |
| Citeseer | order19606 | 2 | 51.452 | 49.094 | -2.358 | -0.352 | -6.371 |
| Citeseer | order19608 | 1 | 54.961 | 58.199 | +3.238 | +5.573 | +0.902 |
| Citeseer | order19608 | 2 | 46.950 | 47.686 | +0.735 | +0.891 | +0.424 |
| Cora | order19605 | 1 | 50.374 | 50.800 | +0.426 | +11.528 | -10.677 |
| Cora | order19605 | 2 | 50.563 | 51.985 | +1.423 | +3.372 | -1.177 |
| Cora | order19606 | 1 | 57.252 | 54.781 | -2.471 | +0.254 | -5.197 |
| Cora | order19606 | 2 | 44.111 | 41.553 | -2.557 | -6.369 | +2.524 |
| Cora | order19608 | 1 | 52.973 | 52.402 | -0.571 | +2.941 | -4.083 |
| Cora | order19608 | 2 | 52.655 | 54.095 | +1.440 | +4.952 | -3.242 |

### 汇总

| 范围 | 累计 F1 差 | 正 cell | old F1 差 | new F1 差 |
|---|---:|---:|---:|---:|
| 全部 12 cell | -0.824 | 5/12 | +1.476 | -3.415 |
| Cora 6 cell | -0.385 | 3/6 | +2.780 | -3.642 |
| Citeseer 6 cell | -1.262 | 2/6 | +0.172 | -3.189 |
| stage 1 | -1.326 | 2/6 | +2.506 | -5.158 |
| stage 2 | -0.321 | 3/6 | +0.446 | -1.673 |

该版本未通过预设晋升条件：平均至少 `+1 pp`、至少 `8/12` 正 cell、最差不低于约 `-2 pp`、new-class F1 不下降。

## 6. 结果说明

### 6.1 为什么 message error 下降但 F1 下降

这说明 raw induced message 不是当前 Student 的充分目标。具体原因有四个：

1. **目标拓扑不一致。** 投影 target 来自当前增量真实 induced subgraph，而 Student 实际使用的是 C0 的凝聚拓扑。即使 synthetic node 的局部 message 更接近 raw target，也不代表它在 C0 的二层 GCN 中更适合传播。
2. **只改 X 破坏图文共同表示。** 文本和拓扑保持不变，X 被推离 two-prototype 的语义位置；双向 InfoNCE 会同时看到 graph/text 不一致，尤其伤害新类。
3. **局部闭式目标没有 Student 参数和训练轨迹。** Student 的可学习 GCN 权重、text projection、batch 负样本和 prompt competition 都没有出现在 message error 中。
4. **误差改善与任务效用几乎不相关。** 12 个 cell 中，message error 降幅与累计 F1 差的相关系数约 `0.121`，与 new-class F1 差约 `0.151`，不足以作为选择或优化依据。

Message-X 的主要表现是 old F1 平均提高 `+1.476 pp`，而 new F1 平均下降 `-3.415 pp`。它把传播目标向已有结构的稳定方向拉近，却没有提供新类可分辨、可重训的内容信号，表现为典型的“保持改善、塑性恶化”。

### 6.2 对之前 message proxy 假设的修正

本轮不能写成“结构 message 对 Student 无用”。更准确的结论是：

> 在固定 C0 拓扑、固定文本、只修改 `X` 的条件下，当前 raw induced one/two-hop message proxy 不能可靠预测 Student 的新类塑性。

它可以作为诊断量，但不能单独作为 `X` 优化目标，也不能据此动态分配预算。

## 7. 当前方法层级更新

- **C0：** 继续作为可靠默认主线和共同基线；
- **role/semantic permutation：** 保留为条件性槽位匹配机制，已有正信号但顺序依赖；
- **R10/R11 shared-P：** 证明 content/topology 联合构造存在局部正交互，但新类塑性不稳定；
- **Message-X：** 已完成独立确认，作为“局部 message proxy 与 Student utility 失配”的负结果；
- **主方法晋升：** 本轮没有新版本通过门禁。

## 8. 下一步：从 Message-X 转向真正的 P+X 联合

下一步不再在 C0 固定拓扑上单独修 `X`，也不继续堆叠 selector。应重新做一个最小的 **shared-P + bounded-X** 版本：

1. `P` 同时决定新节点的 `X`、文本来源和拓扑支持；
2. `X` 的候选目标不使用 raw message error 单独决定，而以 `P` 的图文一致性和 Student-facing short-horizon response 作为联合约束；
3. 对 `P` 保留 C0/初始 assignment anchor，限制偏移；
4. 对 `X` 使用 source-cluster trust region，但 trust radius 不再由 message error 单独放大；
5. 用 matched C0、shared-P、shared-P+bounded-X 三臂比较，并报告 old/new F1、P 漂移、X 位移、构造成本；
6. 若该版本仍只提高 old F1 而伤害 new F1，则关闭“局部传播代理优化”路线，保留 C0 + 条件性 role assignment 的论文定位。

## 9. 证据位置

- 构造结果：`results/tgc_message_conditioned_x_independent_20260926b/`
- Student 结果：`results/tgc_message_conditioned_x_eval_20260926/`
- 构造代码：`code/v2_llm_fusion/run_tgc_message_conditioned_x_20260926.py`
- 评估代码：`code/v2_llm_fusion/run_tgc_role_aware_detailed_eval_20260926.py`
