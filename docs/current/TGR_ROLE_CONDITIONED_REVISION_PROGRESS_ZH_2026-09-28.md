# TGR 受限旧记忆修订：实现与实验进展

日期：2026-09-28  
状态：完成最小实现、单元测试和多组真实小图 smoke；尚未晋升为主方法。

## 1. 本轮目标

本轮没有因为旧记忆更新效果不稳定而取消该方向，而是把它改造成一个更窄、更可审计的组件：

> 允许旧 synthetic memory 更新，但只更新构造期判定为可适应的少量行，并对每个角色使用不同 trust region。

同时固定：

- 初始稀疏分配 `P`；
- 最终拓扑；
- 文本 embedding 和文本内容；
- memory budget；
- canonical 两层 GCN Student；
- 历史完整 raw graph 不可回读约束。

## 2. 当前实现流程

### 2.1 合法输入

阶段 `t` 只读取：

```text
parent synthetic memory + 当前阶段 raw delta
```

历史 raw 节点、历史 raw 文本和历史 raw 边不进入构造过程。

### 2.2 初始 `P`

使用现有 sparse top-`r` 图文联合距离构造 `P`。本轮不更新 `P`，因为先要区分旧记忆修订和 assignment 漂移的影响。

### 2.3 旧行角色

每个旧 synthetic row 根据三个构造期统计量打分：

1. **Semantic boundary score**：旧行和当前 raw delta 的图文联合相似度；
2. **Teacher gradient sensitivity**：canonical Teacher loss 对该旧行的梯度范数；
3. **Degree score**：旧行在 base synthetic topology 中的出度。

旧行按综合分数分为：

| 角色 | 默认比例 | 默认相对位移半径 | 作用 |
|---|---:|---:|---|
| core | 50% | 0 或 0.02 | 保护稳定旧记忆 |
| regular | 35% | 0 或 0.05 | 允许普通适应 |
| boundary | 15% | 0.005、0.02 或 0.08 | 处理新旧类边界 |

新行使用单独的 `new_radius`，常用值为 `0.15`。

### 2.4 梯度目标

Candidate Student 从 Teacher 的同一初始化开始做短 unroll。外层 Adam 更新 `X`：

- 新行可以更新；
- 旧行只在角色半径内更新；
- 每轮更新后投影回 trust region；
- 旧行和新行不是同一个学习率约束。

### 2.5 旧类关系保持

目标中加入：

- 旧类 graph center 保持；
- 旧类之间的 relation matrix 保持；
- 旧类 graph/text center consistency；
- 角色加权的位移正则。

旧类目标最初使用 support Teacher，后改为在 base topology 上单独训练 synthetic-only Teacher，避免 support topology 与最终候选 topology 不一致。

### 2.6 Student response probes

为了分析 teacher proxy 与最终 Student 的错位，先后加入了以下 construction-time probe：

- synthetic old/new rows 的 InfoNCE；
- 当前 raw delta 子图上的 InfoNCE；
- 当前 raw delta 的 construction-label text prototype loss；
- 当前 raw delta 对官方 class prompt 的分类 loss；
- 旧 synthetic rows 对官方 class prompt 的保持 loss。

这些 probe 只使用当前合法 delta、synthetic memory 和公开 prompt，不使用 report/test 标签。

### 2.7 Gradient surgery

对旧行增加了可选的一阶梯度投影：如果总梯度方向会增加 old prompt loss，则投影掉冲突分量；其余方向继续保留。因此它不是冻结旧记忆，而是：

```text
允许 old X 更新，但不允许其一阶方向直接破坏 old prompt response
```

## 3. 实现文件

- `code/v2_llm_fusion/tgr_role_conditioned_revision.py`
- `code/v2_llm_fusion/run_tgr_role_conditioned_revision.py`
- `code/v2_llm_fusion/test_tgr_role_conditioned_revision.py`

当前测试：

```text
13 passed
```

## 4. 真实小图结果

所有结果均为候选相对同一 Semantic-Recursive base memory 的 paired difference，单位为百分点；Student seeds 为 `150--155`。

### 4.1 归因实验，order19611 / Stage-1

| 方法 | Cora Final | Cora New | Cora Old | Citeseer Final | Citeseer New | Citeseer Old |
|---|---:|---:|---:|---:|---:|---:|
| role revision v1，old+new，半径 0.08/0.15 | -1.51 | -0.27 | -1.88 | -0.80 | +0.41 | -1.56 |
| role revision v2，base Teacher old anchor | -2.92 | -1.65 | -4.06 | -2.96 | -2.89 | -2.47 |
| new-only，new radius 0.15 | +0.20 | +0.20 | +1.01 | +1.42 | +1.43 | +2.07 |
| old-only，old radius 0.02/0.05/0.08 | -1.22 | -0.19 | -2.07 | -1.11 | +0.88 | -2.07 |
| boundary-only，boundary radius 0.02 | -0.94 | +0.23 | -1.27 | -0.92 | +0.23 | -1.49 |
| boundary-only + synthetic probe gate | 0.00 | 0.00 | 0.00 | -1.41 | -1.06 | -0.98 |
| joint strict，boundary 0.005 + response gate，new 0.15 | +0.29 | +0.42 | +0.09 | +0.37 | +1.94 | -0.15 |

其中最后一行是目前最有希望的局部配置：旧行只允许少量梯度兼容的 boundary row 更新，新行继续更新。

### 4.2 独立顺序确认

同一 joint strict 配置换到 `order19612 / Stage-1` 后：

| 数据集 | Final | New | Old | 结论 |
|---|---:|---:|---:|---|
| Cora/order19612 | -2.54 | -4.70 | -0.53 | 失败 |
| Citeseer/order19612 | 0.00 | 0.00 | 0.00 | construction gate 回退 |

这说明局部正向结果不能直接称为稳定晋升。

### 4.3 Prompt probe 和 gradient surgery

在 `order19611 / Stage-1` 上加入官方 prompt probe 后：

| 配置 | Cora Final | Cora New | Cora Old | Citeseer Final | Citeseer New | Citeseer Old |
|---|---:|---:|---:|---:|---:|---:|
| prompt joint strict | -0.03 | +0.00 | -0.07 | -0.11 | +2.19 | -1.58 |
| prompt retention | -0.03 | +0.00 | -0.07 | -0.11 | +2.19 | -1.58 |
| prompt + gradient surgery | -0.03 | +0.00 | -0.07 | -0.11 | +2.19 | -1.58 |

结论是：prompt-space 目标明显提高了部分新类 F1，但目前仍不能保护真实图上的 old F1；gradient surgery 没有在这两个 cell 上改变最终结果。

### 4.4 Trust region 幅度控制

只保留干净 Student 目标、new-only、把 new radius 降到 `0.05` 后：

| 数据集/顺序 | Final | New | Old |
|---|---:|---:|---:|
| Cora/order19611 | -1.24 | -0.83 | -1.42 |
| Citeseer/order19611 | +0.67 | +2.02 | -0.44 |
| Cora/order19612 | -4.28 | -6.08 | -2.74 |

性能不是随半径单调变化，不能简单认为“半径越小越安全”。

## 5. 结果说明

### 5.1 已确认的正确信号

1. 旧记忆更新不能被简单取消；它确实需要 role/trust-region 约束。
2. 全量 old revision 是主要风险源，old-only 两个数据集都明显下降。
3. 少量 boundary old revision 与 new X 联合时，在 order19611 上得到双数据集正向结果。
4. 新类塑性和旧类保持存在真实 trade-off，单一 Teacher proxy 不能同时代表两者。
5. prompt probe 比普通 synthetic InfoNCE 更接近最终下游，但仍不能完全消除真实图传播差异。
6. 顺序影响非常大。order19612 的 Cora 失败说明当前模块还不是稳定主方法。

### 5.2 当前仍未解决的问题

- Teacher/support topology 与最终真实图 topology 仍有 gap；
- synthetic old prompt response 与真实 old-class F1 仍不完全一致；
- new X 的最优方向依赖类别顺序；
- 当前 P 固定，尚未验证 P 与 role-conditioned X 联合更新是否有益；
- 当前拓扑仍固定，没有把安全的 old/new cross-edge transport 纳入这一版；
- 目前只完成 Stage-1 小图验证，未完成多阶段和大图确认。

## 6. 当前判断

当前不能把 role-conditioned revision 晋升为稳定主方法，也不能宣称已解决持续 TAG 凝聚中的旧新协同问题。

但本轮已经把问题从“梯度更新效果不好”细化为三个可验证子问题：

1. **更新对象问题**：old rows 不能全部更新，只有少量 boundary rows 有希望；
2. **目标错位问题**：InfoNCE、Teacher embedding、prompt classification 和真实图 F1 并不等价；
3. **协议泛化问题**：order19611 的正向结果不能代表 order19612。

## 7. 下一步建议

### P1：先做多验证 seed 共识 gate

固定当前 joint strict 结构，只把 validation Teacher 从 1 个改成 3--5 个，要求候选对所有 validation seed 都不恶化，避免单个 proxy seed 选择错误。

### P2：加入合法的 current-delta prompt response 作为必要条件

保留 prompt probe，但不把它和大量其它损失继续相加；先做单项 ablation：

- old retention only；
- current-delta prompt only；
- 二者联合；
- no revision。

### P3：再考虑 `P` 更新

只有当 fixed-`P` 版本在独立顺序上不再严重下降，才允许 `P` 在同一 response-alignment gate 下更新；否则会同时放大 assignment 和 X 的不确定性。

### P4：拓扑作为单独组件

保持 `X` 和 prompt 目标不变，只允许少量 message-compatible new-old edge swap；用同一 hard materializer 做 forward/evaluation 对齐。

### 晋升条件

至少满足：

- 两个数据集、至少两个独立顺序平均正收益；
- 最差 cell 不低于 `-1 pp`；
- old F1 平均不下降；
- new F1 平均有正收益；
- 构造成本不超过 Semantic-Recursive 的 `2x`；
- 所有回退和 role assignment 都有审计记录。

在满足前，不把它写成主方法；应作为 Semantic-Recursive 的受限 revision 候选和负结果分析。

