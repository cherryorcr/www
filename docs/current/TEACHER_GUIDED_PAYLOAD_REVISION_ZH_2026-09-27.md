# Teacher-Guided Payload Revision 实现与诊断

日期：2026-09-27

## 1. 本轮目标

本轮完整实现 Teacher--P--X 联合优化，不根据第一轮结果提前删掉梯度反馈。方法以
Semantic-Recursive memory 为初始化，使用上一阶段 synthetic memory 与当前 raw delta
构成合法 support，并新增：

- 在合法 support 上短暂训练的 canonical Teacher；
- 当前 raw delta 到新 synthetic slots 的稀疏软分配 `P`；
- 新旧 `X` 的 trust-region revision；
- 2 步 Student-compatible unroll；
- payload、Teacher feature、trajectory 和 old retention 约束。

第一版固定文本、预算和拓扑，目的是先验证 Teacher/P/X 主机制，而不是同时改变所有模块。

## 2. 完整流程

### 2.1 合法 support

阶段 `t` 的 support 为：

\[
S_t=M_{t-1}^{synthetic}\cup \Delta_t^{raw}.
\]

历史完整 raw graph 不会被读取。旧 synthetic rows 的 class id 在第一版记为 old group，
当前 delta 使用其合法到达类别。

### 2.2 稀疏 P

对当前 raw delta 和 semantic synthetic block 构造联合 graph-text 表示，按距离选择
top-8 候选，并对候选使用 softmax 权重：

\[
p_{ij}=\operatorname{softmax}(-d(h_i,h_j)/\tau).
\]

P 同时接受：

- 类别一致性约束；
- top-r 覆盖约束；
- assignment entropy 监控。

### 2.3 Teacher

Teacher 使用原始两层 GCN、可训练 text projection 和双向 InfoNCE，在合法 support 上
训练 4 步。它不访问 report/test 标签，也不使用历史完整 raw 数据。Teacher 输出：

- support graph representations；
- 训练后的参数状态；
- 从初始参数到训练参数的短轨迹。

### 2.4 X/P 联合更新

Candidate Student 从相同初始化出发，在 synthetic memory 上进行 2 步 differentiable
unroll。外层同时更新：

- 所有 old/new X 的相对位移；
- 新 block 的 P logits。

目标包括：

\[
L=\lambda_fL_{feature}+\lambda_tL_{trajectory}
 +\lambda_xL_{payload}+\lambda_oL_{old}
 +\lambda_pL_{assignment}+\lambda_aL_{anchor}.
\]

旧行半径为 `0.05`，新行半径为 `0.15`。文本、预算和 edge_index 保持不变。

## 3. 小图完整运行

配置：Teacher steps=4，Student unroll=2，outer steps=4，P top-r=8，Student
评估 seeds=`150--155`。

| 数据集 | 顺序 | Semantic-Recursive F1 | Teacher/P/X F1 | F1 差值 | Accuracy 差值 | 新类 F1 差值 | 旧类 F1 差值 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | order19611 | 52.077 | 50.957 | -1.120 | -2.119 | -0.046 | -1.763 |
| Cora | order19612 | 35.861 | 33.808 | -2.053 | -3.892 | -6.632 | +2.432 |
| Citeseer | order19611 | 62.815 | 60.140 | -2.674 | -2.963 | -2.120 | -2.984 |
| Citeseer | order19612 | 59.067 | 57.148 | -1.920 | -1.861 | -1.151 | -1.951 |

这些结果只说明第一版实现可以运行，不能说明 Teacher/P/X 思路没有价值。

## 4. 数据与实现审计

四个 cell 均满足：

- P top-r=8，类别字段与当前 raw delta 对齐；
- `text_changed=false`；
- `budget_changed=false`；
- `topology_changed=false`；
- `historical_raw_reads=0`；
- `report_labels_used=false`；
- X 的 old/new trust-region 均生效。

P 平均熵约为 `1.64--1.79`，不是单一 raw 节点塌缩。当前 raw 类别和 slot 类别均有
覆盖，因此没有发现数据加载、类别映射或 P 全部塌缩错误。

## 5. 已定位的问题

### 5.1 trajectory 项尺度过小

当前参数轨迹 MSE 约为 `4e-6`，相比 feature loss 的 `0.1` 量级几乎不起作用。
因此代码虽然包含 trajectory 项，但实际优化主要由 feature/payload/assignment 项驱动，
还没有真正实现有效的 Student trajectory guidance。

### 5.2 P 的类别候选过早硬限制

初始化时 top-r 候选优先全部来自同一类别，导致 class penalty 接近零。P 只能在同类
内部重新加权，无法纠正 slot 的初始类别归属，也无法表达边界样本的跨类语义关系。

### 5.3 Teacher 与 Candidate 的图拓扑不一致

Teacher 在 `parent + raw delta` 的 support topology 上训练，Candidate 在固定的
Semantic-Recursive/C0 topology 上训练。逐 slot 比较两边的 graph representation
并不完全可行，X 很难仅靠移动特征补偿邻接矩阵差异。

### 5.4 old group 过粗

第一版没有历史原始节点可读权限，因此旧 synthetic rows 暂时全部归为一个 old group，
无法精确约束每个旧类别的保持。后续应在 memory 构造时保存合法的 slot-level class
provenance，而不是重新读取历史 raw graph。

## 6. 下一版调整

下一版不删除梯度和联合优化，而是修正上述四点：

1. 将 trajectory MSE 改为 layer-wise normalized update cosine，并提高到与 feature loss
   同量级；
2. P 使用“同类候选 + 全局近邻候选”的并集，保留类别惩罚但允许 P 修正初始归属；
3. Teacher 目标改为 class-level pooled feature/trajectory target，避免把不同拓扑下的
   graph node embedding 强行逐行等同；
4. 在 stage-0 生成 memory 时保存 synthetic slot 的 class provenance，后续 old
   retention 按类别计算；
5. 先只改 X/P 和归一化 trajectory，再单独验证 topology revision，避免再次混合多个
   未校准模块。

## 7. 产物

代码：

- `code/v2_llm_fusion/teacher_guided_payload_revision.py`
- `code/v2_llm_fusion/run_teacher_guided_payload_revision.py`

构造结果：

- `results/tgr_full_small_20260927/`

Student 评估：

- `results/tgr_full_small_eval_20260927/`

## 8. 修正后的完整复现

上一节的 `tgr_full_small_20260927` 是第一版实现的完整运行。随后只修正了已经审计出的实现问题，
没有改变文本、预算、拓扑、Student 架构或评估协议。修正内容为：

1. 将参数轨迹 MSE 改为归一化参数更新方向 cosine；
2. 将 P 的候选集合改为“同类近邻 + 全局近邻”并集；
3. 将 Teacher/Candidate 的新节点特征比较改为类级 pooled feature；
4. 保留 X/P 联合更新、old/new trust region 和原有 Student unroll。

修正版构造和评估均已完成：

- 构造：`results/tgr_revised_small_20260927/`
- Student 评估：`results/tgr_revised_small_eval_20260927/`
- 评估 Student seeds：`150--155`
- 四个 cell 均为 stage 1，使用 Cora/Citeseer 的两个类别顺序。

### 8.1 修正版 Student 结果

| 数据集 | 顺序 | Semantic-Recursive F1 | Teacher/P/X F1 | F1 差值 | Semantic-Recursive Accuracy | Teacher/P/X Accuracy | Accuracy 差值 | 新类 F1 差值 | 旧类 F1 差值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | order19611 | 52.077 | 51.546 | -0.531 | 51.156 | 50.289 | -0.867 | -0.092 | -0.523 |
| Cora | order19612 | 35.861 | 33.442 | -2.419 | 44.311 | 41.417 | -2.894 | -5.015 | -0.379 |
| Citeseer | order19611 | 62.815 | 62.314 | -0.500 | 57.926 | 57.259 | -0.667 | -0.134 | -0.657 |
| Citeseer | order19612 | 59.067 | 58.733 | -0.334 | 66.181 | 65.210 | -0.971 | -3.187 | +1.715 |
| **四格平均差值** |  |  |  | **-0.946** |  |  | **-1.350** | **-2.107** | **+0.039** |

相对于第一版四格平均差值：

| 版本 | F1 差值 | Accuracy 差值 | 新类 F1 差值 | 旧类 F1 差值 |
|---|---:|---:|---:|---:|
| 第一版 Teacher/P/X | -1.942 | -2.709 | -2.487 | -1.067 |
| 修正版 Teacher/P/X | -0.946 | -1.350 | -2.107 | +0.039 |
| 修正带来的改善 | +0.996 | +1.359 | +0.380 | +1.106 |

这说明前两项实现修正确实消除了部分损失，但当前联合优化仍没有超过 Semantic-Recursive。
它暂时不能作为主方法晋升版本。

### 8.2 修正版审计

四个 cell 均满足：

- `historical_raw_reads=0`；
- `report_labels_used=false`；
- 文本、预算和 `edge_index` 未改变；
- old/new 最大相对位移分别受 `0.05` 和 `0.15` trust region 约束；
- Teacher、Candidate、support 的行数和维度一致；
- P 的熵约为 `1.67--1.80`，不是单点塌缩。

归一化 trajectory loss 在四个 cell 约为 `0.59--0.73`，不再是旧版的 `4e-6`，说明
Student trajectory 项已经真正进入优化目标。P 的最终变化仍然很小：平均绝对权重变化约
`0.0019--0.0023`，KL 约为 `0.00015--0.00018`，top-1 assignment 改变比例为 `0--5.56%`。

因此，这一轮没有证据表明结果差是数据加载、标签映射、预算或统计错误造成的。Semantic-Recursive
的 base 结果在两轮评估中保持一致，说明对照端稳定。

## 9. 当前真正的问题

### 9.1 P 还没有成为真正的结构变量

当前 P 只参与新 X 的加权 target、payload loss 和 assignment penalty；文本和拓扑仍被冻结。
因此 P 即使变化，也不能改变 Candidate Student 实际看到的邻接传播或文本输入。它更像一个
软目标重加权器，而不是完整的 payload transport operator，这解释了 P 梯度存在但实际变化很小。

### 9.2 Teacher 和 Candidate 的传播算子不一致

Teacher 在 `parent synthetic + raw delta` 的 support topology 上训练，Candidate 在
Semantic-Recursive 已材料化的固定 topology 上训练。两者的 graph message 不是同一个算子，
仅靠 X 的小范围移动无法完全对齐 Teacher target。

### 9.3 外层短视目标和最终 Student 目标仍有差距

Teacher 只训练 4 步，Candidate 只 unroll 2 步，而最终报告 Student 使用与 memory 规模相关的
完整训练步数。当前目标能优化短期响应，却没有可靠地预测长训练轨迹中的新类塑性。

### 9.4 X 更新已经触及 trust-region 边界

四个 cell 的 new-row 最大位移都约为 `0.15`，old-row 最大位移都约为 `0.05`。这不是非法更新，
但说明外层梯度持续推动 X，当前半径更像硬截断，而不是经过校准的自然最优幅度。

## 10. 下一步

本轮先不继续叠加文本 attention、GraphGPS、动态预算或复杂 selector。下一轮应只修正一个核心
失配：让 Teacher 和 Candidate 使用同一个可微的传播材料化算子，并让 P 同时决定 X 的 payload
和受控的边/消息材料化。具体顺序为：

1. 保持当前 X/P loss，先把 Teacher support 的旧块、当前 delta 和 Candidate synthetic slots
   映射到同一 slot-level message kernel；
2. 做 `P-only`、`X-only`、`joint P+X` 三个严格对照，确认收益到底来自 assignment 还是特征更新；
3. 将 old retention 从单一 old group 改为合法保存的 slot-level class provenance；
4. 只在上述三项通过后，再校准 trust-region 半径和 unroll horizon；
5. 每一步仍先在这四个小图 cell 上审计，再决定是否扩展到 stage 2 或大图。

当前阶段的结论是：Teacher--P--X 已经完整实现并真实运行过；数据和协议没有发现致命问题；
主要失败来自 P 没有进入结构传播、Teacher/Candidate 拓扑不一致以及短期目标和最终 Student
训练目标不一致。后续应针对这些机制问题做最小改动，而不是回到大量启发式模块叠加。

## 11. bounded message correction 后续实验

### 11.1 provenance 修正

第一次加入旧类 provenance 时，错误地把当前新增 block 的 `c0_row_classes` 当成旧 parent
rows 的类别。该字段长度等于新增 rows，不等于 old rows，因此代码安全地回退成统一 old group；
该轮结果与旧 bounded 版本逐字节一致，不能作为 provenance 实验。

随后修正为只读取封存的合成元数据：stage-0 memory 的 `allocation={class_id: row_count}`，
以及 stage-1 以前 block 的 `build_audit.c0_row_classes`。没有读取历史完整 raw graph。修正后
provenance 确实生效，但结果下降：

| 版本 | F1 差值均值 | Accuracy 差值均值 | 新类 F1 差值均值 | 旧类 F1 差值均值 |
|---|---:|---:|---:|---:|
| bounded joint，mix=0.10，new radius=0.15 | -0.642 | -1.363 | -0.874 | -0.170 |
| full class-balanced provenance | -1.084 | -1.315 | -1.448 | -0.247 |
| provenance mix=0.25 | -1.947 | -2.781 | -1.445 | -1.423 |

结论是：旧类类别均衡采样改变了 Teacher 的 batch 分布，当前不能作为默认策略。provenance
元数据本身是有用的审计信息，但不应直接主导训练采样。

### 11.2 message mix 与 X trust-region

在固定 Student、文本、预算、P 构造和 6 个 Student seeds 下，比较 bounded message correction：

| 设置 | F1 差值均值 | Accuracy 差值均值 | 新类 F1 差值均值 | 旧类 F1 差值均值 |
|---|---:|---:|---:|---:|
| mix=0.05，new radius=0.15 | -1.957 | -2.449 | -1.667 | -2.013 |
| mix=0.10，new radius=0.15 | -0.642 | -1.363 | -0.874 | -0.170 |
| mix=0.20，new radius=0.15 | -0.729 | -2.031 | +1.220 | -1.259 |
| mix=0.15，new radius=0.25 | -1.981 | -3.173 | -3.028 | -0.705 |
| mix=0.20，new radius=0.25 | **+0.153** | -1.409 | +0.452 | +0.462 |

`mix=0.20, new radius=0.25` 有正向信号，但 Accuracy 仍下降，且最差 cell 的新类 F1 为
`-6.548 pp`，因此不能仅凭均值晋升主方法。

### 11.3 X-only 与 P-only 消融

在上述最有希望的设置 `mix=0.20, new radius=0.25` 下，分别冻结 P 或冻结 X：

| 方法 | F1 差值均值 | Accuracy 差值均值 | 新类 F1 差值均值 | 旧类 F1 差值均值 | 正向 F1 cell |
|---|---:|---:|---:|---:|---:|
| joint P+X | +0.153 | -1.409 | +0.452 | +0.462 | 3/4 |
| P-only | +0.288 | -0.749 | **+3.096** | -1.275 | 2/4 |
| **X-only** | **+0.774** | -0.549 | +1.013 | **+1.650** | **3/4** |

X-only 的 stage-1 四格结果为：

| 数据集/顺序 | F1 差值 | Accuracy 差值 | 新类 F1 差值 | 旧类 F1 差值 |
|---|---:|---:|---:|---:|
| Cora/order19611 | +1.213 | +1.349 | +0.039 | +2.972 |
| Cora/order19612 | +1.684 | -3.094 | +4.305 | +2.252 |
| Citeseer/order19611 | -0.043 | +0.519 | -0.596 | +0.861 |
| Citeseer/order19612 | +0.243 | -0.971 | +0.305 | +0.514 |

这说明当前正向信号主要来自受限的 X 更新，而不是 P assignment 更新。P 的平均变化很小；
在 X-only 中 P 完全冻结，拓扑只由初始 P 产生的 bounded message correction 决定。

### 11.4 stage-2 检查

将 stage-1 最优 X-only 设置直接推进 stage-2：

| stage-2 设置 | F1 差值均值 | Accuracy 差值均值 | 新类 F1 差值均值 | 旧类 F1 差值均值 |
|---|---:|---:|---:|---:|
| mix=0.20，new radius=0.25 | -0.161 | -0.391 | +0.577 | -0.999 |
| mix=0.10，new radius=0.25 | -1.099 | -0.266 | -1.172 | -1.996 |
| mix=0.20，new radius=0.15 | -0.391 | -0.012 | +0.335 | -1.872 |

因此，X-only 的 stage-1 正向结果尚未证明能沿持续链稳定传播。stage-2 需要阶段感知的
retention/plasticity 控制，不能直接复用 stage-1 的固定 mix；单纯减小 new radius 也没有
解决旧类保持问题。

### 11.5 结构审计与当前判断

X-only `mix=0.20, new radius=0.25` 的拓扑变化是受限的：stage-1 相对 Semantic-Recursive
的边重叠约为 `95.4%--97.5%`，stage-2 约为 `96.2%--99.2%`；old rows 最大相对位移保持
在 `0.05`，new rows 最大位移不超过 `0.25`，`historical_raw_reads=0`，报告标签没有用于
构造。它是一个可审计的 Student-compatible X revision 候选，但还不是稳定主方法。

当前最合理的研究结论是：

1. 受限 message kernel 能把 Student 训练反馈传给 X，并在部分阶段改善 F1 与新类塑性；
2. P 的自由优化不是当前瓶颈，反而可能引入 assignment 噪声；
3. full provenance balancing、低 message mix 和 stage-2 固定配置都没有稳定收益；
4. 当前应保留 X-only 作为候选分支和负结果，不应把某一个正向 cell 写成已验证主方法。

后续只值得做一个方向：把 `message_mix` 和 X 半径改为由 old-retention risk 与 new-class
response 的局部差异共同决定的有界 stage-aware gate，并与固定 `mix=0.20/radius=0.25`、
固定 `mix=0.10/radius=0.15`、Semantic-Recursive 三者做独立顺序确认。若 gate 仍不能降低
stage-2 最差 cell，再将 bounded X revision 定位为诊断性机制，主方法回到 Semantic-Recursive。
