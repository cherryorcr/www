# P2-A 图条件文本处理完成报告

日期：2026-09-18  
状态：小图正式矩阵完成；工程审计通过；图条件文本未通过晋升门禁，暂不替换冻结 v4 的 `joint_g5_c320 + center concat`。

## 1. 目标与边界

P2-A 只研究一个问题：在合成 X、rows、节点顺序和拓扑全部固定时，当前 delta 的图结构能否帮助选择更有用的文本片段。

本轮没有加入有界跨边、单向跨块传播、response selector、extra budget、可训练 query、历史 raw 回读或大图。P3-B 的这些拓扑设计继续后置；本轮不修改冻结 v4、P1-X 或 P3-B 结果。

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `canonical`、`order1701`、`order1702` |
| 阶段 | 3-stage class-incremental |
| 文本组 | 8 组 |
| Student seeds | 100..105，共 6 个 |
| 训练总数 | 864 fits |
| memory | 合法 pool 的 5%，Mean-X，dispersion allocation |
| Student | 两层 GCN、冻结 MiniLM、双向 InfoNCE、batch 16、Adam lr 0.001 |
| steps | `max(50, min(2048, 16 * rows))` |
| 主指标 | report-dev、所有类别 prompts 竞争的累计 Macro-F1 |
| 数据隔离 | selector-dev/test/report-dev 不进入构造；student 不使用分类标签 |
| 共同变量 | source clusters、X、rows、row order、mass、old/new edges、topology |

各文本组共用同一份源簇、Mean-X 和 legacy frozen-old 拓扑。拓扑不是由候选文本重新计算，因此差异归因于文本目标及其对应的 student 训练效果。

这里的 `center_concat` 是同 64-token 的新对照，**不是冻结 v4 的历史分数**；固定 source clusters、response off，不能用本表替换或重命名 v4。

## 3. P2-A 算法

每个当前 delta 的合法源文本最多取前 240 content tokens，再按 32-token 片段拆分，用冻结 MiniLM 编码。每个合成行最多保留 64 content tokens，最多选 5 个片段；选片分数为：

`score_i = 0.8 * cosine(query, key_i) - 0.2 * max_j cosine(key_i, key_j)`

最后按源顺序拼接并重新编码。没有可训练 query 参数，也没有通过 report-dev F1 反传选择器。

属性 query 不直接把原始 X 与 MiniLM 向量点乘。先利用 X 与 Mean-X 的代表性得到权重，再用同一源节点的 text embedding 映射到 MiniLM 空间：

`w_i = softmax(cos(X_i, mean(X_C)) / 0.2)`

`q_attr = normalize(sum_i w_i * normalize(text_emb_i))`

当前 delta induced graph 中、簇外的一跳邻居提供：

`q_neighbor = normalize(mean_j_in_N(C)(normalize(text_emb_j)))`

`q_graph = normalize(q_attr + q_neighbor)`

运行 manifest 中的 graph-query 字段采用了简写；上式及冻结实现 `p2a_graph_text.py` 明确包含邻域均值的再次归一化。归档 manifest 和训练代码不因报告补充而重签。

没有邻居时退回属性 query。`shuffled_graph` 只对邻域摘要做固定置换，保留自身属性 query 和候选文本，用来判断真实邻域是否提供额外信息。

## 4. 文本组

| 组 | 做法 | 目的 |
|---|---|---|
| `center_concat` | 中心代表文本，同 64-token 预算拼接后重编码 | 当前文本主控制 |
| `mean_pool` | 同一批代表文本分别编码后均值 | 平均表示控制 |
| `text_attention` | 同一批代表文本的已有内容 attention pooling | text-only attention 控制 |
| `text_selection` | text-only query 选片，拼接后重编码 | 选片本身控制 |
| `attribute_selection` | 属性 query 选片后重编码 | 排除真实图邻域 |
| `graph_selection` | 真实当前 delta 邻域 query 选片后重编码 | P2-A 主候选 |
| `graph_pool` | 与 graph_selection 选到同一批片段，但 attention pooling | 区分选片与表示接口 |
| `shuffled_graph` | 打乱邻域 query 后选片 | 图条件反事实 |

## 5. 结果总表

数值为 3 个顺序 × 6 个 student seed 的描述性均值，单位为 Macro-F1 百分点。`old/new` 是最终阶段历史类别/当前类别 F1。

具体为真实 report-dev 节点的零样本类别匹配：在全部类别 prompt 竞争下，对已到达类别的节点计算逐类 F1 后取平均；不是文本摘要质量分数，也不是监督分类训练。

| 数据集 | 方法 | Final | Stage mean | Old | New | Forgetting |
|---|---|---:|---:|---:|---:|---:|
| Cora | center concat | 48.46 | 49.88 | 55.89 | 38.55 | 2.29 |
| Cora | mean pooling | 49.51 | 49.12 | 55.32 | 41.75 | -2.53 |
| Cora | text attention | 51.25 | 50.58 | 56.19 | 44.67 | -0.62 |
| Cora | text selection | 50.28 | 49.70 | 56.19 | 42.41 | 5.18 |
| Cora | attribute selection | 49.33 | 48.92 | 55.87 | 40.61 | 3.25 |
| Cora | graph selection | 51.51 | 50.16 | 55.86 | 45.71 | 2.52 |
| Cora | graph pooling | 49.46 | 49.42 | 55.05 | 42.01 | 2.32 |
| Cora | shuffled graph | 52.32 | 51.58 | 56.46 | 46.79 | 1.93 |
| Citeseer | center concat | 54.97 | 61.07 | 58.86 | 47.19 | 7.29 |
| Citeseer | mean pooling | 53.40 | 61.03 | 57.07 | 46.05 | 9.22 |
| Citeseer | text attention | 54.08 | 61.48 | 57.86 | 46.52 | 9.60 |
| Citeseer | text selection | 54.54 | 60.15 | 58.88 | 45.86 | 5.08 |
| Citeseer | attribute selection | 54.46 | 62.14 | 59.71 | 43.96 | 7.03 |
| Citeseer | graph selection | 53.12 | 57.48 | 57.85 | 43.65 | 3.46 |
| Citeseer | graph pooling | 54.10 | 57.40 | 57.86 | 46.58 | 2.44 |
| Citeseer | shuffled graph | 53.94 | 57.48 | 57.92 | 45.98 | 3.35 |

## 6. 关键配对结论

### 6.1 graph selection 相对 center concat

| 数据集 | Final gain | Stage-mean gain | Old gain | New gain | 正增益 cells |
|---|---:|---:|---:|---:|---:|
| Cora | +3.05 | +0.28 | -0.03 | +7.16 | 14/18 |
| Citeseer | -1.85 | -3.59 | -1.00 | -3.54 | 7/18 |

Cora 上图条件选片提高了新类塑性，但 Citeseer 同时损害阶段平均、旧类和新类表现，不能称为跨数据集稳定改进。

逐顺序最终增益（graph selection 减 center concat；各取六 seed 均值）：

| 数据集 | canonical | order1701 | order1702 |
|---|---:|---:|---:|
| Cora | +0.31 | +5.80 | +3.05 |
| Citeseer | -3.18 | -5.55 | +3.18 |

因此不能仅看总体均值。Cora canonical 相对 text-only selection 也低 5.54 点，触发逐顺序退化门禁。

### 6.2 graph selection 相对 shuffled graph

这是 P2-A 最关键的反事实比较：

| 数据集 | Final gain | Stage mean | Old | New |
|---|---:|---:|---:|---:|
| Cora | -0.81 | -1.42 | -0.60 | -1.08 |
| Citeseer | -0.82 | -0.01 | -0.07 | -2.33 |

真实邻域 query 没有优于打乱邻域 query。当前证据说明：这版邻域摘要没有被可靠地转化为更有用的文本选择信号；它不证明图条件文本总体不可行，但否定了当前第一版的晋升资格。

### 6.3 pooling 与选片

Cora 上 `text_attention` 达到 51.25，说明固定 token 下内容 attention 可能改善该数据集的文本目标；但 Citeseer 为 54.08，低于 center concat 的 54.97，仍不是稳定的跨数据集方案。`graph_pool` 的结果具有数据集依赖：Citeseer 的最终 F1（54.10）高于 `graph_selection`（53.12）和 `shuffled_graph`（53.94），但仍低于 center concat，且阶段平均明显偏低；Cora 也没有形成稳定优势。因此不能把 attention pooling 直接写成新方法贡献。

## 7. 审计和成本验收

- 864/864 fits 完成。
- 18 个共享 stage-control cells 检查通过。
- 文本预算违规：0。
- Cora/Citeseer 独立 artifact audit 均通过；分别检查 72 个保存 memory、2256/2664 个新增文本行。
- 所有 checkpoint 在 runner 内重新加载验证；文本源、X、rows、edge、old-memory hash 均通过。
- test labels evaluated：否；report-dev 不参与 query 参数训练。
- 仅当前 delta induced graph 用于 graph query；没有历史 raw 节点/文本回读。
- 独立 artifact audit 为训练结束后的离线核验，可以读 raw 检查来源和 Mean-X；该核验不向凝聚流程反馈信息。
- 本地最终测试 27/27 通过，含六项汇总器测试；正式训练所依赖的 11 个源文件与 manifest hash 一致。
- 1052 份原始 JSON 在本地与远端逐文件 SHA-256 一致；本地从这些证据重新生成汇总，并另用独立计算复核 center/graph/shuffled 的 final 与 stage mean。汇总文件含本地路径，因此不要求与远端 summary 文件逐字节一致。

成本按“共享预处理”和“候选臂专属成本”拆开核算。下表是三顺序平均、三阶段轨迹合计；student 秒数是六个 report seed 合计，不能直接当作一次部署成本：

| 数据集 | 共享 cluster+legacy topology | 共享 bank/query | 其中 bank 编码 | 候选 tokens | graph 文本构造/最终编码 | graph student（6 seeds） |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.17 s | 9.35 s | 1.00 s | 318,940 | 0.70 / 0.26 s | 37.66 s |
| Citeseer | 0.19 s | 12.03 s | 1.25 s | 413,524 | 0.84 / 0.30 s | 47.98 s |

若共享 bank 只构造一次，并将 student 训练按单个 seed 折算，graph-selection 的暖缓存记账约为 Cora 16.50 s、Citeseer 21.07 s；center control 约为 7.59 s、9.52 s。这里是由已记录组件相加得到的核算值，不是隔离环境下的端到端 wall-clock；原始 source MiniLM 缓存编码、IO 和审计开销未计入。所有文本臂的 student steps 相同，但片段候选编码使 graph 条件方案并非零额外成本。

最终编码已包含在文本构造时间内，bank 编码已包含在 bank/query 内，不重复相加。共享预处理不能按八个实验臂或六个 seed 摊薄后声称一次使用更便宜；该计时包含各类控制 query 的公共准备，是当前实现的保守记账，不是已优化的单臂最小开销。graph-selection 最终逻辑 payload 为约 321.31/380.09 KiB（Cora/Citeseer），不含审计来源库和 student checkpoints。

## 8. 晋升判定

预先冻结的门禁要求分别对 center_concat、text_selection、attribute_selection、shuffled_graph 四个控制满足：两个数据集各自平均 final 至少 +0.5、stage mean 不下降、各顺序 final 不低于 -2、数据集平均 old/new 增益均不低于 -2。结果未通过：

1. Cora 的真实 graph query 低于 shuffled control。
2. Citeseer 的 graph selection 比 center concat 低 1.85，stage mean 低 3.59。
3. 真实 graph query 没有跨数据集提升新类塑性。

因此当前默认继续 `Mean-X + dispersion + joint_g5_c320 center concat + current cross-edge topology`；不启动 P2-A 独立 order1703/1704 confirmation，不将 graph-conditioned text 命名为 v5 或主方法，不删除负结果。

P3-B 的有界跨边、单向跨块传播仍按原计划后置，不与本轮文本结果混合。

## 9. 后续建议

如果继续研究 P2，应先修正信号接口，而不是扩大候选数量：

1. 将邻域摘要从简单 text mean 改为边权/结构角色加权，并保持 source-only、固定参数。
2. 分离簇外邻域文本与簇内文本的独立贡献，继续保留无邻域和随机邻域反事实。
3. 使用图条件选片但保留 center anchor，避免 query 把文本带离 MiniLM 的簇中心。
4. 分开报告候选内容预算和候选来源编码成本。
5. 先在单阶段固定旧 memory 上确认机制，再决定是否重跑持续链。
6. 只有图条件信号通过诊断与独立确认，才考虑可训练 attention 或升级集合表示；本轮无参数 graph_pool 已测，不能再记为未测候选。

### 9.1 后置的拓扑实验：有界跨边与单向跨块传播

本轮不把拓扑算子变化混入 P2-A。后续若启动 P3-B-2，应先固定 X、文本、rows、row order 和当前跨边端点，只改变跨块传播；以下是待冻结的小规模候选，不表示已跑或晋升：

1. **有界跨边。** 先固定端点和边数，仅比较全局跨边权重 `0.25/0.5/1.0`。若再测 degree cap，应同时约束旧节点接收的跨边总权重和新节点跨度，另配同边数/同度分布的随机端点控制；不把少边和更好端点混为一项增益。保留无跨边与当前拓扑。权重必须真正进入传播及度计算，不能只写 metadata。
2. **单向跨块传播。** 消息方向明确按发送者到接收者定义。以 old-to-new 为主要保护候选，new-to-old 为反向对照，双向为共同控制。矩阵按“行接收、列发送”记：`P=[[P_oo,0],[lambda*P_no,P_nn]]` 表示旧块向新块发送；固定旧块归一化 `P_oo`，否则仅删反向边仍可能改变旧块。degree correction/block normalization 单列算子因素，先单阶段同旧 memory 起点诊断，再有条件运行持续链，避免一次跑全部笛卡尔积。
3. **必须记录的机制指标。** 旧块归一化邻接漂移、跨边消息范数、旧 synthetic 表征漂移、old/new F1、阶段平均、forgetting、边数和训练成本。旧边 hash 不变并不等于旧传播不变。

改变加权、定向或分块归一化传播时，须单列 operator-adaptation 表：Mean-X 控制、强 coreset 和完整数据参考都使用相同的算子定义及可用阶段信息，不能只给本方法改 student 后与旧表比较。即使 `P_oo` 完全保持，student 参数仍重新训练，所以不保证旧类预测不遗忘。

只有拓扑控制在至少两个小图、预注册顺序和共同 student 下稳定改善，才考虑进入长链或大图；不能用单向传播的 operator 改动掩盖 P2-A 的文本负结果。

### 9.2 解释边界

- 评估沿用完整真实图上的传导式推理，未标注的 future/holdout 特征和边仍可参与消息传播，因此不是严格归纳式评估。
- P2-A 的 64 content-token 是严格控制；冻结 v4 的 `g5_c320` 是字符预算，两者不能直接当作同一文本预算。
- `forgetting = mean_{j<T}[max_{s=j..T-1}F1(s,j)-F1(T,j)]`，排除最终阶段取历史最大值，允许负值表示后向改善。初期学得较差也会让该数值变小；Citeseer 图条件组 forgetting 较小但总体 F1 更低，不能解释为保持能力更强。
- 18 个 order-seed 单元是配对诊断格，不是 18 个独立数据集试验；均值和标准差不提供独立样本置信区间。
- 当前负结果否定的是这一版邻域均值 query + MMR 接口的晋升资格，不是定理意义上否定所有图条件文本注意力。

下一阶段优先保留 P2-A 负结果，完成计划中尚未验收的低成本预算/分配对照，随后再决定是否进入 P4 历史记忆管理。大图和 P3-B 有界传播暂不因本轮结果提前启动。

## 10. 复现入口

- [P2-A 冻结协议](P2A_GRAPH_CONDITIONED_TEXT_PROTOCOL_ZH_2026-09-18.md)
- [P2-A 代码](../../code/v2_llm_fusion/p2a_graph_text.py)、[runner](../../code/v2_llm_fusion/run_p2a_graph_conditioned_text.py)
- [P2-A 测试](../../code/v2_llm_fusion/test_p2a_graph_conditioned_text.py)
- [汇总器测试](../../code/v2_llm_fusion/test_p2a_summary.py)、[独立产物审计器](../../code/v2_llm_fusion/audit_p2a_artifacts.py)
- [汇总 JSON](../../results/p2a_graph_conditioned_text_20260918/summary/summary.json)、[指标表](../../results/p2a_graph_conditioned_text_20260918/summary/metrics.md)
- [Cora 验收](../../results/p2a_graph_conditioned_text_20260918/cora/acceptance.json)、[Citeseer 验收](../../results/p2a_graph_conditioned_text_20260918/citeseer/acceptance.json)
- 结果标识：`p2a-graph-text-v1`。

本地已同步完整逐 seed JSON、文本/来源审计和计划；memory 与 student 的 `.pt` 留存在服务器结果目录 `/home/pc/research/TAG_condensation/results/p2a_graph_conditioned_text_20260918`，不声称本地已有全部 checkpoint。实际训练使用 `tagsam`、RTX 4090，未使用本地测试环境训练正式模型。

从项目根目录重新汇总（无需重新训练）：

```text
python code/v2_llm_fusion/summarize_p2a_graph_conditioned_text.py --root results/p2a_graph_conditioned_text_20260918 --output results/p2a_graph_conditioned_text_20260918/summary
```

从 `code/v2_llm_fusion` 运行测试：

```text
python -m unittest test_p2a_graph_conditioned_text test_p2a_summary test_p2_text_operators test_p2_compatibility test_p3b_topology_diagnostic
```
