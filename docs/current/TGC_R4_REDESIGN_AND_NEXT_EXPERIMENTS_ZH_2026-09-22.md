# TGC-R4 重新设计与下一轮实验路线

日期：2026-09-22。状态：历史设计草案；R4 尚未运行。2026-09-23 审查后，本文件的 MB-LRR 配方及执行决策树已被取代。

> **不要按下文旧配方直接启动实验。** 现行设计审查见 [TGC-R4 证据审查、设计修正与实现路线](TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md)。本文件保留作为历史方案记录，不是有效预注册协议。
>
> 主要更正：审计通过不排除所有实现问题；mass 是来源数而非质量；采样不平衡尚未被证明是失败原因；局部 basis 存在坐标空间/零方向/历史访问问题；半径 0.05 并非未试新机制；旧行 quota 必须用上一阶段 rows 计算；mass-support 失败不能否定全部 TGC；原构造成本 3x 门槛撤回，需在新确认协议前冻结完整成本标准。下文对应因果断言、默认参数和决策树均不再作为执行依据。

## 1. 结论先行

TGC-R3 没有证明 TGC 思路无效，但已经证明当前配方不能继续作为主线：

- `warm-up=64 + suffix=1` 的候选在独立条件下最终 s2 平均为 `-0.64 pp`；
- 从头训练的 `new64`、`mixed64` 和 `response curve` 选择器分别为 `-1.01`、`-0.51` 和 `-0.50 pp`；
- teacher 对照 s2 为 `+0.60 pp`，但 s1 为 `-0.64 pp`，全程基本持平，不能直接升格为完整算法；
- R3 的审计、重载和指标重算均通过，失败属于机制和目标设计问题，不是实现错误；
- 主策略构造成本约为 C0 已计时构造的 `21.91x`，加一个正式 student 后约为 `2.33x`，当前收益不足以支撑该成本。

因此，R4 不应继续调整旧 gate 的阈值、位移半径或 response curve。应重做候选生成器：

> 先把当前增量压缩成质量平衡的 teacher support，再用 teacher 的旧新统计量指导局部、低维、可回退的 revision；最后只用与正式 student 更接近的无标签响应作为选择信号。

R4 的目标不是一次性加入文本、预算、拓扑和 student 反向传播全部模块，而是先验证一个新的候选机制是否能够稳定优于 C0 和 matched-random。

## 2. R3 具体哪里出了问题

### 2.1 teacher 的训练支持集与持续 memory 的质量不匹配

当前 teacher 直接在 `旧 synthetic memory + 当前 raw delta` 上训练：

```text
L_teacher = InfoNCE(old synthetic rows ∪ current raw rows)
```

但 old synthetic row 带有 `mass`，而当前实现的 teacher batch 将行近似等权处理。于是：

- 当前 delta 可能有数百到上千行，旧 memory 只有几十到一百行；
- teacher 的梯度更容易被当前 raw delta 的数量和局部结构支配；
- 旧 memory 的压缩误差和历史覆盖质量没有按 mass 进入 teacher 目标；
- 旧类保持只在 synthetic probe 上检查，不能代表历史真实分布。

这解释了为什么 teacher 在某些 s2 条件能提高旧类和新类，但在 s1 可能明显波动：它不是稳定的 continual teacher，而是一个对当前 delta 比较敏感的短期拟合器。

### 2.2 候选生成与最终 student 的训练轨迹不一致

R2B/R3 的候选主要来自：

1. 在 C0 上训练 64 步；
2. 固定并 detach 该 prefix；
3. 对旧 X 做 1 步可微 suffix；
4. 用短期 old/new/mixed InfoNCE 选择。

正式 student 却是从随机初始化开始训练约 `608--1776` 步，并用共同类别 prompt 的 zero-shot Macro-F1 评估。两者之间存在三层错位：

- prefix student 与正式 scratch student 不同；
- 32/64/128 步 InfoNCE 与长程收敛状态不同；
- 实例级对齐 loss 与共同 prompt 竞争下的类别 Macro-F1 不同。

因此，selector 不是简单“预测不准”，而是在预测另一个任务。

### 2.3 X 的修改破坏了图文和拓扑的一致性

当前只改旧 synthetic `X`，同时冻结：

- 旧 text 和 text embedding；
- 旧旧边和新旧拓扑；
- row mass、cluster identity、birth stage。

但是 GCN 的图表示由 `X + adjacency` 共同决定。X 发生较大相对位移后，可能出现：

- graph embedding 向新 delta 靠近，但 text prototype 仍代表旧语义；
- 旧节点的消息邻居没有同步变化；
- teacher 的局部 loss 下降，但最终 prompt 分类 margin 变差；
- 位移方向更像“提高当前 batch 对齐”，而不是修复可重训的合成节点。

R3 的 stop-gradient 与 random 对照只能说明当前实现有差异，不能说明这个差异是有用的结构修复。

### 2.4 修改行的选择信号偏向高敏感行，而不是高价值行

当前 row utility 主要由 gradient norm、residual 和 delta affinity 构成。高 gradient norm 表示“对当前 teacher loss 敏感”，不表示“修改后能改善最终 student”。在只有 16、38、94 行的早期 memory 中，10% revision quota 还会退化为修改 1、3、9 个行，单个行的误选影响很大。

### 2.5 preservation 项没有真正工作

R2B 中旧 loss hinge 的最大增幅低于 tolerance，激活数为 0；R3 中旧 synthetic probe 保护也不能预测真实 old F1。当前 preservation 更接近一个形式上的约束，而不是历史任务风险证书。

### 2.6 当前实验还不是递归 TGC

R3 的 s1/s2 都从对应阶段的 C0 parent memory 出发，不把上一阶段的 revision 结果传到下一阶段。这样做对机制确认是正确的，但不能说明 revision 在长链上能否累积，也不能用 R3 结果声称完整 continual rollout 已被验证。

## 3. R4 的核心重设计：Mass-Balanced Local Response Revision

暂定名称：**MB-LRR：Mass-Balanced Local Response Revision**。它仍属于 TGC，但与当前的“teacher gradient 直接改 X”不同。

### 3.1 阶段一：构造质量平衡的 teacher support

给定上一阶段 memory `M_{t-1}` 和当前合法 raw delta `Delta_t`，先构造临时 support：

```text
S_t = Replay(M_{t-1}, mass-aware) ∪ Compress(Delta_t, balanced)
```

建议固定如下规则：

- old synthetic rows：按 `mass` 做 capped replay，避免大 mass 行被忽略，但也不允许单行重复支配 batch；
- current delta：每个到达类别或局部 cluster 至少保留相同数量的 probe rows；
- old/new 总体比例固定为 `1:1` 和 `1:2` 两个候选，不同时搜索更多比例；
- teacher 只读取 old synthetic memory 和当前 raw delta；不读取历史 raw graph、report/test 标签；
- teacher 只作为构造期工具，不能把 teacher 参数传给最终 student。

第一轮先只验证 `1:1 mass-balanced`，`1:2` 作为控制。这样可以先判断 R3 的主要问题是否来自支持集偏置。

### 3.2 阶段二：用局部子空间而不是全维 X 位移

对每个可编辑旧行 `i`，从合法支持中提取一个局部 basis：

```text
U_i = orth([x_i - x_neighbour, x_delta - x_i, x_text - x_i])
```

第一版只保留两个方向：

1. `u_affinity`：旧行到当前 delta 的联合图文亲和方向；
2. `u_preserve`：旧行到其旧 joint-cluster prototype 的保持方向。

令 `a_i, b_i` 为有界标量：

```text
x'_i = x_i + a_i u_affinity + b_i u_preserve
```

约束：

- `||x'_i - x_i|| / max(||x_i||, eps) <= 0.05`，先比 R3 的 `0.10` 更保守；
- `|a_i| <= 0.05`、`|b_i| <= 0.05`；
- protected coverage core 永不修改；
- revision quota 仍为旧 rows 的 10%，但不足 4 行时改用固定最小候选数 1，不增加总 memory；
- 所有修改行、方向、幅度、理由和 hash 写入 revision certificate。

这个设计的关键不是“位移更小”本身，而是禁止 teacher 将 X 推向没有语义解释的任意高维方向。

### 3.3 阶段三：teacher 目标拆成三类统计量

不再只用一个 global InfoNCE loss。对每个局部候选定义：

```text
L_MB-LRR = L_mix
          + λ_old L_old_response
          + λ_geom L_cluster_consistency
          - λ_new L_delta_affinity
```

其中：

- `L_mix`：质量平衡 old/new joint InfoNCE；
- `L_old_response`：修改前后旧 synthetic 行的 graph-text alignment、局部相似度和邻居消息统计保持；
- `L_cluster_consistency`：修改后仍接近原 joint cluster 的局部 prototype；
- `L_delta_affinity`：只鼓励可解释地接近当前 delta，不鼓励无界追逐 raw rows。

第一轮只使用确定性加权和，不引入可训练 gate：

```text
λ_old = 1.0, λ_geom = 0.25, λ_new = 0.50
```

这组权重只是冻结的初始协议，不代表最优值。R4-A 先只跑 `λ_old=0` 与上述完整目标两个臂，分离 preservation/geometry 是否带来实际收益。

### 3.4 阶段四：候选选择改为 conservative accept-or-reject

候选生成和候选选择必须分开：

- `candidate`：MB-LRR 生成的固定 candidate；
- `random`：相同修改行和相同位移范数的随机局部方向；
- `c0`：不修改；
- `teacher-only`：质量平衡 support 但使用旧 teacher 的全维 gradient，作为机制对照；
- `geometry-only`：只使用局部子空间，不使用 student response。

第一轮不做昂贵的多策略 selector。只做一个无标签 acceptance：

```text
accept iff
  balanced mixed response(candidate) <= response(C0) - ε
  and old response drift <= τ
  and cluster consistency does not worsen
```

如果 candidate 本身不优于 matched-random，就不进入下一轮；不能通过放宽 gate 把它救回来。

## 4. R4 实验分层

### R4-A：支持集偏置诊断

固定 C0 的 rows、X、text、topology 和 quota，只改变 teacher support：

| 臂 | support | update |
|---|---|---|
| C0 | 无 teacher | 无修改 |
| teacher-old | old synthetic only | R3 全维 gradient |
| teacher-mix | old/new 等权 | R3 全维 gradient |
| MB-support | mass-balanced old/new | R3 全维 gradient |

目的：判断 R3 的 teacher 正负波动是否主要来自 old/new support imbalance。若 `MB-support` 仍不优于 `teacher-mix`，不进入 R4-B。

### R4-B：局部子空间诊断

固定 `MB-support`，比较：

| 臂 | 更新方向 |
|---|---|
| full-gradient | R3 当前全维 gradient |
| local-affinity | `u_affinity` |
| local-preserve | `u_preserve` |
| local-joint | `u_affinity + u_preserve` |
| matched-random | 相同范数随机方向 |

目的：判断性能是否来自方向结构，而不是任意 X 扰动。

### R4-C：response 对齐诊断

只在 R4-A/B 有一个候选通过时进行：

- 训练 student 的真实目标仍为双向 InfoNCE；
- 从头训练 0/64/256 步的短轨迹只用于构造期诊断；
- probe 同时包含 old/new mixed negatives；
- 选择信号还要记录 prompt-margin proxy、old/new response、cluster drift 和最终 report F1；
- 不把任何单一短期信号称为最终指标的无标签保证。

### R4-D：递归 rollout

只有 R4-C 在新的类别顺序上达到门槛才启动：

- Cora/Citeseer；canonical + 两个新顺序；
- 3 stages；6 report seeds；
- C0、MB-LRR、matched-random；
- s1/s2 使用上一阶段真正输出的 memory；
- 额外报告 revision count、old/new drift、cost 和 memory bytes。

## 5. 晋升门槛

R4 的 candidate 必须同时满足：

- s2 四个小图 cell 等权平均不低于 C0，目标至少 `+0.5 pp`；
- 至少 3/4 cell 不低于 `C0 - 1 pp`；
- 最差 cell 不低于 `C0 - 2 pp`；
- old F1 最差下降不超过 `2 pp`；
- new F1 平均不下降；
- 至少不劣于 matched-random；
- 构造成本不超过 C0 的 `3x`；
- 逻辑 memory 行数、边数、文本预算不增加；
- 结果在未参与 R4-A/B 设计的新顺序或新 seed group 上复现。

若只在一个顺序或一个阶段变好，保留为机制诊断，不升为 TGC 主方法。

## 6. 暂停或删除的方向

以下方向在 R4 前暂不继续：

- 继续调 R3 的 `old tolerance`、`gain threshold`、`warm-up` 和 `suffix`；
- 递归 rollout 当前 `warm-up=64 + suffix=1` candidate；
- 同时修改 X、text、topology、budget；
- 直接上大图；
- 将 teacher s2 的局部正收益写成已验证主方法；
- 用最终 report F1 反向挑选最好的 cell 或 gate。

## 7. 最终决策树

```text
R4-A support balance
    ├─ fail: TGC-R 的主要问题是 teacher support，回到 C0 + 负结果分析
    └─ pass
        ↓
R4-B local subspace
    ├─ fail: 任意 X revision 不可靠，考虑只做统计量/新增 block 机制
    └─ pass
        ↓
R4-C response alignment
    ├─ fail: candidate 有效但 selector 不可靠，固定 candidate 不做动态 gate
    └─ pass
        ↓
R4-D recursive continual rollout
```

## 8. 对论文主线的影响

当前论文不能写成“R3 已经提出并验证了 TGC-R”。准确表述是：

> 本项目提出了一个受历史 raw graph 不可回读约束的 teacher-guided continual revision 框架，并系统发现：直接用短程 student response 或 teacher gradient 修订旧 synthetic X，在小图上不能稳定预测长程 student utility。R4 将验证质量平衡和局部可解释 revision 是否能修复这一失配。

只有 R4-D 通过新的顺序和递归 rollout，才可以把 TGC 写成主算法；否则 TGC 应作为机制探索和负结果章节，C0 作为可复现的主基线。
