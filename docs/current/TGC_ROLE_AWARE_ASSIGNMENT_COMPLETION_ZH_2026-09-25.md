# TGC Role-Aware Conserved Assignment 小图验证完成报告

日期：2026-09-25  
状态：12 个 stage-cell 全部完成；结果、守恒审计和协议文件齐全；本报告用于判断该机制是否值得进入下一轮主线确认。

**独立确认更新（2026-09-26）：** 后续未参与开发的 `order19605/19606/19608`、Cora/Citeseer、stage 1/2 共 12 个 cell 已完成 direct old/new F1 和构造成本评估。`role_perm` 相对 C0 的累计 F1 平均为 `+1.468 pp`、`6/12` 正、最差 `-7.444 pp`，new-class F1 平均 `+0.558 pp`；未通过独立晋升门禁，不能升为默认主方法。完整结果见 [独立顺序确认报告](TGC_ROLE_AWARE_INDEPENDENT_CONFIRMATION_ZH_2026-09-26.md)。

## 1. 结论先行

本轮最强的是 `role_perm`：在固定 C0 拓扑、固定 Student、固定文本预算和固定 memory budget 下，将双原型 proposal 的完整 payload 在同类 C0 槽位内按照“图文语义 + 拓扑角色 + mass”进行守恒双射匹配。

在 Cora/Citeseer、3 个类别顺序、stage 1/2、6 个 report seeds 的 12 个 cell 上：

| 方法 | 相对 C0 的累计 Macro-F1 均值 | 正 cell | 最差 cell | 当前阶段 F1 变化 | 旧阶段 F1 变化 |
|---|---:|---:|---:|---:|---:|
| Identity proposal | +0.513 pp | 6/12 | -3.509 pp | +0.595 pp | +1.210 pp |
| Semantic permutation | +2.541 pp | 7/12 | -1.539 pp | +3.484 pp | +2.858 pp |
| **Role permutation** | **+3.579 pp** | **10/12** | **-1.155 pp** | **+5.958 pp** | **+2.945 pp** |
| Random permutation | +1.297 pp | 8/12 | -3.905 pp | +0.739 pp | +2.950 pp |
| Message-role proxy | +0.006 pp | 3/12 | -5.431 pp | +0.825 pp | +0.953 pp |
| Plasticity-role proxy | +0.419 pp | 5/12 | -5.325 pp | +0.718 pp | +1.700 pp |
| Adaptive-role proxy | +0.410 pp | 5/12 | -5.431 pp | +0.910 pp | +1.470 pp |

这里的“当前阶段 F1”是当前已见类别上的 all-prompt competition Macro-F1，不等于严格意义上的 new-class F1；“旧阶段 F1”是旧类别阶段的 retention proxy。因此本轮可以确认的是整体任务效用和旧类保持改善，不能直接宣称新类塑性已经被证明改善。

**判定：** `role_perm` 是目前第一个同时满足本轮预设小图门槛的机制候选：平均提升超过 `+0.5 pp`、10/12 cell 为正、最差 cell 高于约 `-2 pp`、旧阶段 F1 平均不降、全部守恒和访问审计通过。但由于还没有直接 new-class F1、独立构造成本计时和更长 canonical 协议确认，它应先进入“下一轮候选主线”，不能立即替代冻结的 C0 论文主表。

## 2. 实验对象与严格协议

### 2.1 数据和任务

- 数据集：Cora、Citeseer。
- 类别顺序：`order19600`、`order19609`、`order19610`。
- 增量阶段：stage 1、stage 2。
- memory rate：5%。
- 每个 cell 的 report seeds：`150..155`。
- 共 `2 x 3 x 2 = 12` 个 stage-cell。

### 2.2 固定比较条件

所有 arm 使用完全相同的：

- C0 的历史 old prefix；
- C0 的 `edge_index`；
- 当前增量和合法 source pool；
- 双原型结构 proposal 的生成方式；
- top-5/320 字符文本来源和冻结 MiniLM 重编码；
- 原始两层 GCN、可训练文本投影和双向 InfoNCE Student；
- batch size 16、同一 student steps、同一 report seeds；
- 评价 mask 和 prompt competition 规则。

候选构造没有读取历史完整 raw graph，也没有用 report/test F1 选择候选。报告标签只在构造完成后用于统一评估。

### 2.3 审计结果

12 个 cell 均生成 `results.json`，没有发现失败 cell 或构造异常。所有置换 arm 均通过：

- `payload_multiset_preserved`；
- `source_coverage_preserved`；
- `same_c0_topology`；
- `same_old_prefix`。

所有 arm 的 `historical_raw_reads=0`，构造协议中的 `report_labels_used=false`。本轮总计有 81 个发生 split/merge 的 proposal slot；`role_perm` 共移动 71 个完整 payload，未发生 class-level fallback。其余 proxy arm 的 fallback 次数如下：

| arm | class-level fallback 次数 |
|---|---:|
| `anchor_role_perm` | 9 |
| `message_role_perm` | 11 |
| `plasticity_role_perm` | 10 |
| `adaptive_role_perm` | 10 |
| `role_perm` | 0 |

这里的 fallback 不是按最终 F1 回退，而是 assignment proxy 相对 identity 不满足固定 margin 时的确定性 identity 回退。

## 3. 方法具体流程

本轮的核心不是重新生成整张图，而是对已经生成的合成 payload 做“槽位分配”。

### 步骤 1：读取 parent memory 和当前增量

在阶段 `t`，读取上一阶段 synthetic memory `M_{t-1}` 和当前合法原始增量 `Delta_t`。旧 memory 的 payload 和旧旧边不被重写，也不读取历史完整原始图。

### 步骤 2：生成固定双原型 proposal

先沿用当前已冻结的 two-prototype structural proposal：在当前类别内部对 source pool 做确定性 split/merge，得到 proposal payload：

\[
P_j=(x_j^{syn}, e_j^{text}, texts_j, mass_j, depth_j, stage_j).
\]

本轮不使用 P-guided response 重新生成内容，不改变文本 cap，也不改变拓扑边集合。因此可把“proposal 内容变化”和“payload 放入哪个传播槽位”分开分析。

### 步骤 3：提取目标槽位的拓扑角色

对 C0 的新块槽位提取以下角色特征：

- `degree`：节点总度数；
- `cross_degree`：与旧块相连的边数；
- `new_new_degree`：新块内部连接数；
- `old_neighbor_ratio`：邻居中旧块节点的比例；
- `old_endpoint_load`：旧节点作为跨块端点承担的连接负载。

这些量描述的是“一个 payload 被放到该槽位后，它会以什么方式参与 GCN message passing”，而不是只比较 payload 自身的 X/text cosine。

### 步骤 4：计算同类内 assignment cost

对每个类别，只在该类别已有的 C0 slots 内做匹配。`role_perm` 的 cost 为：

\[
C(P_j,s_i)=z(C_{joint})+0.50z(C_{role})+0.10z(C_{mass}),
\]

其中：

- `C_joint`：proposal 与目标槽位 payload 的图文联合距离；
- `C_role`：proposal 来源角色与目标槽位角色的距离；
- `C_mass`：proposal mass 与目标槽位 mass 的对数差异；
- `z`：在当前类别 assignment 矩阵内做标准化。

这里没有用最终 Macro-F1，也没有用测试标签调节权重。`message_role_perm`、`plasticity_role_perm` 和 `adaptive_role_perm` 只是额外加入 C0 one-hop message proxy、prompt margin proxy 或 bounded graph/text alpha 的对照。

### 步骤 5：Hungarian 双射匹配

对每个类别求最小 cost assignment。被选中的 proposal payload 作为一个整体搬到目标 slot；其 `x`、`text_emb`、文本来源、mass、depth 和 stage 必须一起搬移。剩余未参与 split/merge 的 slots 保持 identity，最后得到完整行级双射。

### 步骤 6：复用 C0 拓扑并训练 Student

完成 payload placement 后，直接复用 C0 的 `edge_index`。因此本轮测量的是：

> 同一张合成拓扑中，合成内容放置到不同传播角色槽位后，Student 的可重训效用是否改变。

每个 arm 单独从头训练相同 Student，并用相同的 report seeds 评估。

## 4. 详细结果

### 4.1 按数据集

| 数据集 | arm | 累计 F1 相对 C0 | 当前阶段 F1 相对 C0 | 旧阶段 F1 相对 C0 | 正 cell | 最差 cell |
|---|---|---:|---:|---:|---:|---:|
| Cora | Identity | -0.521 | +0.420 | -0.120 | 2/6 | -3.509 |
| Cora | Semantic | +2.536 | +4.562 | +2.879 | 3/6 | -1.539 |
| **Cora** | **Role** | **+4.489** | **+7.946** | **+3.358** | **6/6** | **+1.968** |
| Citeseer | Identity | +1.546 | +0.770 | +2.541 | 4/6 | -2.137 |
| Citeseer | Semantic | +2.547 | +2.406 | +2.838 | 4/6 | -1.208 |
| **Citeseer** | **Role** | **+2.669** | **+3.970** | **+2.532** | **4/6** | **-1.155** |

Role assignment 在 Cora 上更稳定，6/6 cell 为正；Citeseer 上仍有两个轻微负 cell，说明该机制不是无条件稳定增益。

### 4.2 按阶段

| 阶段 | 累计 F1 相对 C0 | 当前阶段 F1 相对 C0 | 旧阶段 F1 相对 C0 | 正 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|
| stage 1 | +5.373 | +6.503 | +5.604 | 5/6 | -0.270 |
| stage 2 | +1.785 | +5.413 | +0.286 | 5/6 | -1.155 |

收益在 stage 1 更强，stage 2 仍保持正均值但旧阶段保持收益明显变小。这提示后续需要在更长 stage chain 上验证 role assignment 是否会累积误差。

### 4.3 每个 cell 的 role 结果

| cell | C0 累计 F1 | Role 累计 F1 | 差值 | 当前阶段差值 | 旧阶段差值 | Identity 相对 C0 |
|---|---:|---:|---:|---:|---:|---:|
| Citeseer/order19600/s1 | 66.608 | 71.253 | +4.645 | +8.149 | +1.703 | -1.417 |
| Citeseer/order19600/s2 | 57.377 | 56.222 | -1.155 | +3.284 | -2.301 | -2.137 |
| Citeseer/order19609/s1 | 36.161 | 43.583 | +7.422 | +1.917 | +13.515 | +5.189 |
| Citeseer/order19609/s2 | 51.896 | 55.759 | +3.863 | +1.443 | +4.623 | +4.427 |
| Citeseer/order19610/s1 | 66.453 | 66.183 | -0.270 | +2.039 | -1.535 | +1.971 |
| Citeseer/order19610/s2 | 53.212 | 54.720 | +1.508 | +6.985 | -0.812 | +1.243 |
| Cora/order19600/s1 | 41.667 | 45.928 | +4.262 | +12.114 | -2.151 | -3.509 |
| Cora/order19600/s2 | 45.228 | 47.196 | +1.968 | +7.930 | +0.160 | +2.683 |
| Cora/order19609/s1 | 42.314 | 44.346 | +2.032 | +6.130 | -1.904 | +2.889 |
| Cora/order19609/s2 | 45.101 | 47.630 | +2.529 | +5.581 | +0.007 | -1.981 |
| Cora/order19610/s1 | 39.444 | 53.591 | +14.148 | +8.668 | +23.997 | -2.080 |
| Cora/order19610/s2 | 49.596 | 51.593 | +1.997 | +7.255 | +0.038 | -1.125 |

### 4.4 是否真的来自 role assignment

不能只比较 `role_perm` 和 C0，因为两者之间同时包含了 proposal 内容变化和 slot placement。更严格的分解是：

- Identity proposal 相对 C0：平均 `+0.513 pp`；
- Role permutation 相对 Identity：平均 `+3.066 pp`，8/12 cell 为正，最差 `-2.241 pp`；
- Role permutation 相对 C0：平均 `+3.579 pp`。

因此本轮主要信号不是“只要换一个 X 就会变好”，而是：在同一 proposal 内容下，把 payload 放到更匹配其传播角色的 C0 slot，确实解释了额外的大部分收益。不过 role 与 semantic assignment 的 mapping 在 12 个 cell 中有 5 个完全相同，说明 role term 不是每个 cell 都真正改变决策；收益主要集中在它改变 mapping 的 cell。

## 5. 为什么 message/plasticity/adaptive proxy 没有继续提升

本轮结果不支持继续把更多 proxy 叠加到 cost 中：

1. `message_role_perm` 平均仅 `+0.006 pp`，而 role-only 是 `+3.579 pp`；
2. `plasticity_role_perm` 和 `adaptive_role_perm` 的平均收益只有约 `+0.42 pp`；
3. 这些 arm 分别在 27 个 class-level decision 中触发 10、10、11 次回退，说明 proxy 经常认为 assignment 不够安全；
4. C0 one-hop message 与最终两层 GCN + 图文 InfoNCE 的实际训练响应仍有目标错位；prompt margin 也不是最终 all-class Macro-F1 的可靠替代目标；
5. bounded alpha 只能改变 cost 的图文比例，不能消除 slot role、proposal 内容和 Student 优化轨迹之间的非线性耦合。

因此下一步不应继续堆叠 message/plasticity/adaptive proxy，而应先确认 role-only gain 是否能在更长协议和独立顺序上复现。

## 6. 审计与成本解释

### 6.1 守恒和访问审计

本轮所有 arm 都满足：

- 旧 memory prefix 不变；
- C0 `edge_index` 完全不变；
- payload 字段整体搬移，没有拆开 X、text、mass 或来源记录；
- 每个 source payload 恰好覆盖一次；
- 不访问历史完整 raw graph；
- 不使用 report/test F1 选择 assignment。

### 6.2 训练时间不能当作构造成本

当前 `results.json` 的 `wall_seconds_total` 主要记录每个候选的 Student report 训练时间，所有 arm 的 Student 训练预算相同，因此 Role/C0 之间约 17 秒量级的差异只能视为噪声，不能当成 assignment 的真实额外成本。

本 runner 尚未单独记录：

- role feature 提取时间；
- cost matrix 构造时间；
- Hungarian 求解时间；
- proposal 生成时间。

因此下一轮必须增加独立 construction timer，并报告 `construction time / student training time / total wall time` 三项，而不能用本轮 `wall_seconds_total` 宣称成本不变。

## 7. 当前决策

### 7.1 可以确认的内容

1. conserved payload assignment 是有意义的机制，不是单纯的随机置换收益：Role 相对 Identity 平均多出 `+3.066 pp`，且最差 cell 明显优于 random。
2. 低成本的拓扑角色特征比当前 message/plasticity proxy 更有用。
3. 收益同时体现在当前阶段效用和旧阶段 retention，不是只改善一个指标。
4. 该机制没有破坏 C0 拓扑、历史 prefix 或 source coverage。

### 7.2 不能确认的内容

1. 尚不能说已经稳定解决 new-class plasticity，因为本轮没有直接保存并汇总 new-class F1。
2. 尚不能说跨数据集、跨长链协议普遍有效；当前只有两个小图、三个顺序和两个增量阶段。
3. 尚不能说比 k-center、TAGSAM 或其它强静态方法更强；本轮是 C0 内部机制控制实验。
4. 尚不能说总计算更低；构造成本还没有独立计时。

### 7.3 晋升结论

`role_perm` 通过本轮小图内部晋升门槛，进入下一轮候选主线，建议使用一个明确名称：

> **Topology-Role-Aware Conserved Payload Assignment**：在同类合成槽位内，根据 payload 的图文语义和目标槽位的传播角色做确定性双射匹配，同时保持 C0 拓扑、旧 prefix 和 payload 来源守恒。

但 C0 仍保留为论文共同控制和冻结默认方法。只有在下一轮满足以下条件后，才考虑把 Role assignment 写入主方法：

- 独立类别顺序至少 2 个数据集均值为正；
- canonical base+4 或更长链上平均提升仍为正；
- 直接 new-class F1 平均不下降；
- 最差 cell 不低于预设容忍线；
- 独立 construction timer 显示额外成本可接受；
- 与 identity、semantic、random 和强 coreset 的 matched 对照仍成立。

## 8. 下一轮最小实验

按优先级执行：

1. **独立顺序复现**：冻结本轮 role cost 和所有权重，不再调参；Cora/Citeseer 追加未参与本轮的类别顺序。
2. **直接 new/old F1**：在 report 阶段保存按 seen/old/new 类别切分的 F1，不能再只用 current F1 代替 new F1。
3. **构造成本计时**：记录 proposal、role feature、cost matrix、Hungarian 和最终 materialization 的 CPU 时间。
4. **长链确认**：至少一个 base+4 或 4-stage 小图协议，检查 role assignment 是否随 stage 累积漂移。
5. **只保留 role-only**：message/plasticity/adaptive 先作为负结果，不再叠加进主候选。
6. **再决定大图**：只有上述确认通过，才在 arXiv/BookChild/SportsFit 上做固定协议的大图实验。

## 9. 可复现实验文件

- Runner：[run_tgc_role_aware_assignment_20260925.py](../../code/v2_llm_fusion/run_tgc_role_aware_assignment_20260925.py)
- 结果目录：[tgc_role_aware_assignment_20260925](../../results/tgc_role_aware_assignment_20260925/)
- 每个 cell 的协议：`protocol.json`
- 每个 cell 的完整结果：`results.json`
- 每个 cell 的构造审计：`build_audit.json`
