# TGR 受限新旧跨边传输：严格小图实验报告

日期：2026-09-28  
状态：严格控制版已完成；不晋升为默认主方法。

**2026-09-28 口径核对：** 以下沿用脚本历史 `C0/c0` 名称的比较基底实际为
`semantic_perm_memory.pt`，即 Semantic-Recursive（SR），不是原始 C0。
第 5/8 节总体表的 `Final F1` 汇总了 stage 1/2 的累计 F1 差，应读作“增量
stage-cell 累计 F1 平均”；`Stage-average F1` 是同一模型在已见类别组上的平均，
不是训练过程三阶段平均；表中标准差是 cell 差值的标准差。数值本身未改动。
本轮各阶段共享 SR parent，尚未完成跨边候选自身拓扑的递归长链验证。
第三顺序将按[新协议](TGR_CROSS_EDGE_THIRD_ORDER_PROTOCOL_ZH_2026-09-28.md)
单列真正的最终阶段和过程平均。

## 1. 本轮要回答的问题

TGR 的一个候选机制是：在只访问“上一阶段 synthetic memory + 当前 raw delta”的条件下，
为新增 synthetic rows 选择少量旧 synthetic 邻居，使新旧块之间产生受控消息传播。

本轮只改变新旧跨边端点，固定：

- `X`；
- 文本内容和 `text_emb`；
- rows、mass、stage 和 memory budget；
- 初始 `P` 和 hard materializer；
- canonical 两层 GCN Student 与双向图文 InfoNCE；
- Student seeds `150--155`；
- 两个数据集、两个类别顺序、stage 1/2。

历史完整 raw graph 不被用来恢复旧 synthetic 节点。跨边候选只使用旧 synthetic/current delta
的合法状态；真实 incident provenance 缺失时不伪造映射。

## 2. 实验 arms

| arm | 定义 |
|---|---|
| `none` | 严格 no-op，完整复制 C0 拓扑，用于确认统计基准 |
| `random` | 对每个新节点随机选择保持容量的旧节点 |
| `semantic` | 按旧/新 graph-text 联合 cosine 选择旧节点 |
| `message` | 按一跳 graph message compatibility 选择旧节点 |

所有非 no-op arm 使用同一个 cross-only materializer：旧旧边和新新边逐边复制，
每个新节点的跨边数量不变，边总数不变，只替换跨边的旧端点。

## 3. 实现流程

对阶段 `t`：

1. 读取 parent synthetic memory 和当前 delta，构造 support graph；
2. 将支持图分为旧旧、新新和新旧跨块三部分；
3. 对每个新节点计算候选旧节点分数；
4. 用固定 cross-degree 选择候选端点；
5. 用容量保持的 hard materializer 只替换跨边端点；
6. 不更新 `P`、`X`、文本或预算；
7. 用同一批 Student seeds 训练并评估 Macro-F1、Accuracy、new-class F1、old-class F1。

`message` 分数为旧节点和新节点一跳 feature message 的 cosine，并混合少量直接 feature
相似度；它不是使用最终 report/test 标签的 response selector。

## 4. 严格审计结果

8 个构造 cell、40 个拓扑 memory 全部完成。审计均通过：

| 审计项 | 结果 |
|---|---:|
| 构造 cell | 8/8 |
| 每个 cell arms | 5（C0、none、random、semantic、message） |
| 非 C0 边数变化 | 0 |
| old-old 边保持 | 40/40 |
| new-new 边保持 | 40/40 |
| 每个新节点跨边度保持 | 40/40 |
| 内容 `X/text/text_emb` 改变 | 0 |
| historical raw feature/text 读取 | 0 |
| report/test 标签用于选择 | 否 |
| 本地单元测试 | 7 passed |

第一版实验的 `none` 曾经在零分数下重新排列跨边，因此不是严格 no-op。该问题已修正；
旧版结果仅作为诊断，不作为正式对照。严格版的 `none` 与 C0 的所有差值为 0。

## 5. 严格下游结果

以下均为相对严格 C0 的百分点变化，8 个 cell 的 Student seed 均为 `150--155`。

### 5.1 总体汇总

| arm | Final Macro-F1 均值 | 标准差 | Accuracy | New F1 | Old F1 | Stage-average F1 | 正收益 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | +1.883 | 3.779 | +0.224 | +1.707 | +3.746 | +2.271 | 4/8 | -2.461 |
| `message` | +1.165 | 3.635 | +0.058 | -0.169 | +2.181 | +1.494 | 5/8 | -3.867 |
| `semantic` | -0.812 | 2.920 | -1.214 | -1.871 | +0.107 | -0.534 | 3/8 | -5.361 |
| `none` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/8 | 0.000 |

这里的总体均值不能被解释为稳定优于 C0：`random` 的均值最高，但只有 4/8 cell 正收益，
说明保持度数的跨边重连本身就可能改变结果；`message` 的均值为正，但 new F1 平均略降，
且最差 cell 为 -3.867 pp。

### 5.2 按数据集

| 数据集 | arm | Final F1 | Accuracy | New F1 | Old F1 | Stage-average F1 |
|---|---|---:|---:|---:|---:|---:|
| Cora | `message` | +2.514 | +1.905 | -1.079 | +5.490 | +2.997 |
| Cora | `random` | +2.233 | +1.570 | -0.702 | +6.086 | +2.534 |
| Cora | `semantic` | +1.452 | +1.986 | -1.279 | +4.744 | +2.296 |
| Citeseer | `message` | -0.185 | -1.788 | +0.741 | -1.128 | -0.010 |
| Citeseer | `random` | +1.534 | -1.122 | +4.116 | +1.406 | +2.008 |
| Citeseer | `semantic` | -3.077 | -4.413 | -2.463 | -4.530 | -3.365 |

### 5.3 按阶段

| stage | arm | Final F1 | Accuracy | New F1 | Old F1 | Stage-average F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `message` | +3.933 | +2.357 | +5.006 | +3.853 | +4.430 |
| 1 | `random` | +2.280 | -0.305 | -1.999 | +7.549 | +2.775 |
| 1 | `semantic` | -1.844 | -2.627 | -1.813 | -2.028 | -1.921 |
| 2 | `message` | -1.604 | -2.241 | -5.344 | +0.508 | -1.442 |
| 2 | `random` | +1.487 | +0.753 | +5.413 | -0.057 | +1.766 |
| 2 | `semantic` | +0.219 | +0.199 | -1.928 | +2.242 | +0.852 |

## 6. 机制诊断

### 6.1 Message 选择存在旧节点负载集中

在 `message` arm 中，很多新节点被分配到少数旧节点。以构造期 support 选择为例，
不同 cell 中前 5 个旧节点承担约 `28.7%--80.2%` 的跨边；随机选择约为 `11.7%--22.9%`。
这说明 message compatibility 只回答了“哪个旧节点最相似”，没有回答“如何避免旧节点过载”。

因此 support 候选可能集中在少数旧行，影响 projected message kernel 的排序和端点竞争。
在本轮 cross-degree=1 且两侧容量保持的最终图上，不能将下降解释成旧节点度数改变；
端点对应关系改变仍会改变传播内容。负载集中是一个待对照验证的机制解释。

### 6.2 Semantic 分数不等于 Student message utility

`semantic` 在总体、Citeseer 和部分 Cora cell 上为负，说明静态 graph-text cosine
并不能可靠预测最终 GCN message-passing 的有用跨边。

### 6.3 进入 stage 2 后 message 收益消失

stage 1 的 `message` 平均 Final F1 为 `+3.933 pp`，stage 2 变为 `-1.604 pp`。
这说明效果依赖阶段。由于 stage 2 使用 SR parent 而非 message arm 的 stage 1 输出，
本实验不能将变化归因于候选跨边的递归累积；需要另做递归拓扑实验。

## 7. 本轮结论

1. “新旧跨边不能直接 append”这一问题被严格实现并完成了因果实验；
2. message-aware 跨边有真实局部信号，尤其在 Cora 和 stage 1；
3. 该信号不能称为稳定增益，不能直接进入主方法；
4. semantic cross-edge 目前不值得作为默认策略；
5. 当前最合理的状态是：保留 `message` 作为受限拓扑候选/诊断组件，不与 TGR 的旧 X/P 更新同时叠加。

## 8. 容量感知 message follow-up

针对第 6 节发现的候选负载集中，新增一个只改变 support 候选选择的
`message_balanced` arm。对每个新 support 节点仍按 message compatibility 排序，
但设置：

\[
\mathrm{cap}=\left\lceil\frac{N_{new}\times d_{cross}}{N_{old}}\right\rceil
\]

每个旧 support row 在候选阶段最多被选 `cap` 次。该约束不改变最终 C0 的 cross-edge
容量；最终 hard materializer 仍保持 C0 的跨边数量和旧端点容量。

### 8.1 容量感知构造审计

8 个 cell 全部完成。前 5 个旧 support row 的候选负载占比从原始 message 的
`28.7%--80.2%` 降为 `6.9%--18.6%`，最大负载均不超过预设 cap。

### 8.2 容量感知 Student 结果

以下为相对严格 C0 的百分点变化，Student seeds 仍为 `150--155`。

| arm | Final Macro-F1 均值 | 标准差 | Accuracy | New F1 | Old F1 | Stage-average F1 | 正收益 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 `message` | +1.165 | 3.635 | +0.058 | -0.169 | +2.181 | +1.494 | 5/8 | -3.867 |
| `message_balanced` | **+2.211** | **2.272** | **+0.665** | **+3.001** | **+2.750** | **+2.623** | **7/8** | **-1.813** |

按数据集平均：

| 数据集 | Final F1 | Accuracy | New F1 | Old F1 | 正收益 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|
| Cora | +2.886 | +1.130 | +3.143 | +4.495 | 4/4 | +0.261 |
| Citeseer | +1.535 | +0.201 | +2.860 | +1.005 | 3/4 | -1.813 |

按类别顺序平均：

| 顺序 | Final F1 | New F1 | Old F1 |
|---|---:|---:|---:|
| `order19611` | +0.820 | +4.818 | -1.350 |
| `order19612` | +3.602 | +1.185 | +6.850 |

### 8.3 对结果的判断

这轮结果支持一个比原始 message 更具体的机制解释：

1. message compatibility 本身有用，但逐新节点独立取最大分数会造成 support 候选质量
   集中；
2. 加入有界负载约束后，New F1 从 `-0.169` 提升到 `+3.001`，且方差明显下降；
3. 该机制在两个数据集和两个顺序上均保持平均正收益，说明不是只在一个 Cora cell 上
   有效；
4. Citeseer `order19611/s2` 仍为 `-1.813 pp`，所以尚不能宣称稳定晋升或 SOTA；
5. 目前得到的是一个有正向证据的拓扑候选，不是已经和 TGR 旧 X/P revision 联合后的最终方法。

## 9. 后续推进建议

## 9A. 第三顺序确认结果（2026-09-28）

第三顺序 `order19613` 使用相同三阶段协议和 Student seeds `150--155`。这里的
`message_balanced` 仍只替换 SR memory 的新旧跨边；stage 2 的 parent 是 SR stage 1，
不是上一阶段的 message-balanced memory。所有 60 个唯一 Student checkpoint 均已保存，
两个数据集的 reload audit 最大指标误差为 `0.0`。

| 数据集/阶段 | SR Final F1 | message_balanced Final F1 | Delta | SR Accuracy | message Accuracy | SR New F1 | message New F1 | SR Old F1 | message Old F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora s0 | 38.734 | 38.734 | 0.000 | 51.546 | 51.546 | 38.734 | 38.734 | - | - |
| Cora s1 | 41.965 | 43.487 | +1.522 | 48.366 | 49.673 | 50.260 | 49.934 | 39.588 | 44.117 |
| Cora s2 | 52.653 | 48.684 | **-3.969** | 60.025 | 56.950 | 73.579 | 61.601 | 53.530 | 59.450 |
| Citeseer s0 | 48.470 | 48.470 | 0.000 | 37.900 | 37.900 | 48.470 | 48.470 | - | - |
| Citeseer s1 | 41.742 | 36.272 | **-5.470** | 38.169 | 31.865 | 45.959 | 44.134 | 47.047 | 33.059 |
| Citeseer s2 | 51.725 | 52.145 | **+0.420** | 58.098 | 57.889 | 85.263 | 85.466 | 49.127 | 48.855 |

按真正的最终阶段（s2）汇总，Cora 为 `-3.969 pp`，Citeseer 为 `+0.420 pp`；
两者平均为 `-1.775 pp`。按 s1/s2 四个增量 cell，平均为 `-1.874 pp`，最差
`-5.470 pp`，只有 `2/4` 个 cell 正收益。因此第三顺序不满足预注册晋升条件，
不进入旧记忆 revision 组合，也不作为默认拓扑。

完整汇总、配对 seed CI、三阶段过程均值和成本在：
`results/tgr_cross_edge_confirmation_20260928/summary.json`；构造与重载产物在远端
`results/tgr_cross_edge_confirmation_20260928/`，本地 metadata 证据在
`tmp/tgr_cross_edge_confirmation_remote_20260928/`。

## 9B. 失败机制的只读归因

第三顺序的 message-balanced support 代理分数提高，但并未提高最终 Student F1：

- Cora s2：最终 F1 `-3.969 pp`，New F1 `-11.978 pp`，Old F1 `+5.920 pp`；
- Citeseer s1：最终 F1 `-5.470 pp`；
- 三阶段过程均值：Cora `-0.816 pp`，Citeseer `-1.683 pp`。

归因脚本重放了封存的 routing 和 hard materializer，没有重新选择边。结果显示：

1. `raw_fallback_fraction` 约为 `0.61--0.67`。大量 current raw rows 没有足够的
   top-r assignment mass，随后被 nearest-slot fallback 接收；message support 对这些
   行并没有形成可靠的 synthetic slot 路由。
2. `proxy_objective_regret`（容量约束下逐新节点 greedy 与离线全局 assignment proxy
   的差异）在第三顺序为 Cora s2 `19.95`、Citeseer s1 `32.43`、Citeseer s2 `20.12`。
   这不是 Student F1 的 oracle，也没有用于改结果；它说明局部 greedy 不是全局容量分配。
3. 第三顺序中最终跨边发生变化的比例为 Cora s2 `89.7%`、Citeseer s1 `65.1%`、
   Citeseer s2 `63.6%`。变化越多不等于越好；旧类保持提升伴随新类塑性下降，尤其 Cora s2。
4. Cora s2 的一跳 message 相对变化约为旧行 `19.9%`、新行 `30.1%`。因此仅凭
   compatibility 排序没有保证新类的传播方向与原 Student 目标一致。

结论不是“跨边机制没有任何信号”，而是：**support message proxy -> 容量受限
greedy routing -> 二值端点替换 -> Student InfoNCE/F1** 这条链路中，后两步没有被
当前 proxy 充分约束。开发顺序的 `+2.211 pp` 不能外推为稳定方法收益。

## 9C. 后续决定

暂缓 `message_balanced + old revision`、`new-only X revision` 和任何多模块联合，
因为当前拓扑组件在独立顺序已失败。保留以下最小候选方向，另立新版本和新结果目录：

1. **Class-constrained global transport**：在已有 support routing 上做同类约束的全局
   min-cost flow/assignment；禁止大比例 fallback；先固定所有 X、文本、预算和 Student。
2. **Bounded edge swap**：限制每个新 slot 的跨边最多替换一部分，并保留 C0 的 message
   baseline；不接受只提升 proxy、却没有旧/新双侧收益的交换。
3. **Student-consistent edge objective**：只在固定小步数的 Student message/InfoNCE
   代理上验证，不能用最终 report-dev F1 选边；失败立即回退 SR。

只有其中一个单模块在第三顺序和至少一个新顺序上通过固定验收，才重新考虑与旧记忆
`X` trust-region 更新组合。当前主线仍是 SR；`message_balanced` 是已证伪的拓扑候选，
不是主方法新增组件。

下一轮先做独立顺序和更长链确认，再考虑联合旧记忆 revision。

## 9D. 全局容量 assignment 追加诊断（2026-09-28）

### 9D.1 动机与严格控制

第 9B 节发现，`message_balanced` 的局部 routing 与容量约束之间仍可能存在
proxy regret。因此新增 `message_global`，只替换最终跨边的容量分配方式：

1. 完全复用 `message_balanced` 的 support routing；
2. 保持旧旧边、新新边、每个新节点的跨边度和每个旧端点的 C0 容量不变；
3. 将每个旧端点的容量展开成 slot；
4. 用全局 `linear_sum_assignment` 最大化 projected message score；
5. 不改变 `X`、文本、`text_emb`、预算、初始 `P`、Student 和训练步骤；
6. 当前协议固定 `cross_degree=1`，更高跨边度显式拒绝，避免重复端点被错误支持。

因此该实验只回答：**把局部 greedy 替换为全局容量最优 assignment，是否能修复
message routing 的代理失配？** 它不是新的完整凝聚方法。

### 9D.2 构造审计

开发集 8 个 cell 全部完成，4 个 arm 共 32 份审计；全局臂的 8/8 个 cell 均满足：

| 审计项 | 结果 |
|---|---:|
| `old_old` 边保持 | 8/8 |
| `new_new` 边保持 | 8/8 |
| 每个新节点跨边度保持 | 8/8 |
| 旧端点容量精确守恒 | 8/8 |
| assignment | 全部 `global_capacity_linear_sum_assignment` |
| `X`、文本、预算、初始 `P` | 未改变 |
| 独立 `order19613` 构造 | 4/4 完成 |

本地针对性测试为 `9 passed`，远端构造环境测试为 `7 passed`。因此后续性能差异不能
归因于全局 assignment 的容量或边块守恒错误。

### 9D.3 开发顺序结果

下面的数值是相对 SR（评估 JSON 中历史 `c0` 键实际指 sealed
`semantic_perm_memory`）的 8 个增量 stage-cell 平均百分点变化；每个 cell 使用
Student seeds `150--155`。

| 方法 | Final Macro-F1 | Accuracy | New F1 | Old F1 | Stage-average F1 | 正收益 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| `message_balanced` | +2.211 | +0.665 | +3.001 | +2.750 | +2.623 | 7/8 | -1.813 |
| `message_global` | +0.402 | -1.049 | -0.003 | +1.628 | +1.081 | 5/8 | -4.953 |

直接比较两个 routing 臂，`message_global - message_balanced` 为：

| Final Macro-F1 | Accuracy | New F1 | Old F1 | Stage-average F1 |
|---:|---:|---:|---:|---:|
| -1.809 | -1.714 | -3.005 | -1.122 | -1.543 |

### 9D.4 独立 `order19613` 结果

该顺序没有参与开发集选择，仍使用相同的六个 Student seeds。它用于判断全局
assignment 是否能跨顺序泛化，而不是用于调参。

| 方法 | Final Macro-F1 | Accuracy | New F1 | Old F1 | Stage-average F1 | 正收益 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| `message_balanced` | -1.874 | -2.070 | -3.481 | -0.953 | -1.491 | 2/4 | -5.470 |
| `message_global` | -0.951 | -1.500 | +1.898 | -4.297 | -0.836 | 1/4 | -2.303 |

全局 assignment 在独立顺序上稍微改善了新类平均 F1，但以明显的旧类保持损失为
代价，最终总体仍为负。因此它没有通过“两个数据集/独立顺序的稳定晋升”要求。

### 9D.5 机制结论与决策

这轮结果把问题进一步分开了：

- 局部 greedy 的确不是容量约束下的全局最优 assignment；
- 但 projected message score 的全局最优也不是 Student InfoNCE/F1 的全局最优；
- 全局 assignment 倾向于保留部分旧类传播，不能稳定改善新类塑性；
- Cora 开发顺序 stage2 出现 `-4.953 pp`，独立顺序仍只有 `1/4` 正收益；
- 因而“容量分配不够全局”不是当前性能失败的唯一原因，继续扫描 assignment 规则的收益有限。

`message_global` 不进入主方法，也不与旧 memory revision、X/P 更新或文本模块联合。
`message_balanced` 仍保留为已证伪的拓扑对照。当前主方法继续固定为
Semantic-Recursive；下一步若继续拓扑研究，应转向**受限 Student-consistent edge swap**，
直接约束短程 InfoNCE/message 代理并设置 C0 回退，而不是继续优化 projected-score
assignment。任何新版本都必须先在独立顺序通过，再考虑与旧 memory 更新联合。

## 9E. 产物与复现入口

- 全局 assignment 实现：[delta_cross_edge_transport.py](../../code/v2_llm_fusion/delta_cross_edge_transport.py)
- 开发构造启动器：[launch_tgr_cross_edge_global_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_global_20260928.sh)
- 开发评估启动器：[launch_tgr_cross_edge_global_eval_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_global_eval_20260928.sh)
- 独立顺序构造启动器：[launch_tgr_cross_edge_global_independent_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_global_independent_20260928.sh)
- 独立顺序评估启动器：[launch_tgr_cross_edge_global_independent_eval_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_global_independent_eval_20260928.sh)
- 开发评估结果：[tgr_cross_edge_global_eval_20260928](../../results/tgr_cross_edge_global_eval_20260928)
- 独立评估结果：[tgr_cross_edge_global_independent_eval_20260928](../../results/tgr_cross_edge_global_independent_eval_20260928)

固定 `message_balanced`，新增一个未参与开发的类别顺序，至少检查：

- Final Macro-F1；
- Accuracy；
- New/Old F1；
- 最差阶段；
- 候选负载分布；
- 构造和 Student 成本。

第三顺序已不满足该条件，因此以下组合实验暂不启动：

1. 固定 `message_balanced` topology；
2. 只更新少量 boundary old rows；
3. 保持 `P` 固定；
4. 用 old-retention 和 current-delta plasticity 双 gate；
5. 任一 gate 失败立即回退到 `message_balanced`，再失败回退到 C0。

组合实验必须同时包含：

- `C0`；
- `message_balanced`；
- `message_balanced + old revision`；
- `message_balanced + new-only X revision`。

在第三顺序完成前，不同时改变文本、预算、P、X 和拓扑；也不把当前结果直接写成最终主方法。

## 9. 产物

- 构造代码：[delta_cross_edge_transport.py](../../code/v2_llm_fusion/delta_cross_edge_transport.py)
- 构造入口：[run_tgr_cross_edge_transport.py](../../code/v2_llm_fusion/run_tgr_cross_edge_transport.py)
- 评估入口：[run_tgr_cross_edge_eval.py](../../code/v2_llm_fusion/run_tgr_cross_edge_eval.py)
- 严格构造启动器：[launch_tgr_cross_edge_transport_strict_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_transport_strict_20260928.sh)
- 严格评估启动器：[launch_tgr_cross_edge_eval_strict_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_eval_strict_20260928.sh)
- 容量感知构造启动器：[launch_tgr_cross_edge_balanced_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_balanced_20260928.sh)
- 容量感知评估启动器：[launch_tgr_cross_edge_balanced_eval_20260928.sh](../../code/v2_llm_fusion/launch_tgr_cross_edge_balanced_eval_20260928.sh)
- 汇总脚本：[summarize_tgr_cross_edge_eval_20260928.ps1](../../tools/summarize_tgr_cross_edge_eval_20260928.ps1)
- 严格结果本地证据：`tmp/tgr_cross_edge_strict_eval_remote_20260928/`
- 容量感知结果本地证据：`tmp/tgr_cross_edge_balanced_eval_remote_20260928/`
