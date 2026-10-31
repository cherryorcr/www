# P2-A 图条件文本：冻结实验协议

日期：2026-09-18。状态：正式训练前冻结；短步 smoke 只验工程，不用于选方案。

## 1. 范围与不做的事

本轮仅检验当前合法图邻域是否能改善合成文本目标。继续使用 Mean-X、dispersion、累计合法构造节点的 5% 预算、response off。合成节点不是 coreset 真实节点。冻结 v4 不改名、不覆盖。

P3-B 的有界跨边、分块归一化、单向跨块传播列入后续拓扑实验，本轮不实现。大图、extra budget、可训练 attention、旧文本刷新均不混入本轮。

## 2. 共同协议

- Cora/Citeseer，canonical、order1701、order1702；三阶段，report seeds 100..105。
- 8 个文本组，共 2×3×3×8×6 = 864 次独立从头 student 训练。
- 相同两层 GCN/MiniLM、双向 InfoNCE、Adam lr=0.001、batch=16；steps=max(50,min(2048,16×rows))。
- 原始 ID 模 10：1 为 selector-dev，2 为 test，3 为 report-dev，其余为构造源。本轮不调用 response selector，不用 test 标签。
- 仅当前 delta 源标签用于类别划分和配额；student 不使用分类标签。
- 主指标沿用既有审计比较的 report-dev、所有类别 prompts 竞争下的 seen-node Macro-F1；另报 seen-only prompts，不能混表。
- 评估沿用完整真实图的传导式推理，包括未标注未来/holdout 的 X 和边；这不是全流程严格归纳式协议。
- 各组固定 source clusters、Mean-X、mass、depth、stage、row order、累计 rows、全部合成边。共同拓扑来自 legacy joint_g5_c320，避免文本变化同时改变连边。
- 历史合成 X、文本、text_emb 和旧旧边逐字段保持。旧传播仍可受新增跨边影响，不能称传播完全冻结。

## 3. 图条件机制

不假定原始 X 与 MiniLM embedding 处于同一坐标系。

对源簇 C 的属性 query，先根据 X 与 Mean-X 的余弦代表性得到权重，再通过同一源节点的文本 embedding 映射到 MiniLM 空间：

`w_i = softmax(cos(X_i, mean(X_C)) / 0.2)`

`q_attr = normalize(sum_i w_i * normalize(text_emb_i))`

text-only query 是簇内归一化文本向量的均值，不使用上述属性权重。

邻域只来自当前 delta 的 induced raw graph。N(C) 是与 C 直接相邻、但不在 C 内的唯一合法节点集合；对有向输入按任一方向相邻处理。可跨当前阶段类别，但不能接入旧 raw、未来阶段或 holdout 节点：

`q_neighbor = normalize(mean(normalize(text_emb_j), j in N(C)))`

`q_graph = normalize(q_attr + q_neighbor)`

孤立簇令邻域向量为零，退回属性 query。邻居数量、源节点数、属性权重有效支持数单独记录，第一版不再给这些统计量引入调优权重。

shuffled control 对各簇邻域摘要做确定性无固定点置换，仅置换邻域，保留本簇属性 query 和候选文本。shuffle seed=9001+1009×stage。它用于机制反事实，不是更多独立数据样本。

## 4. 文本候选与预算

每个合法 delta 源文本取至多前 240 个内容 tokens，以 32-token 窗口拆片，并用冻结 MiniLM 编码。各组有相同候选来源；片段编码是额外预处理，必须计入成本，不能声称整个构造只读 64 tokens。

最终每个合成节点保留至多 64 内容 tokens。注意力选片采用固定 `0.8×query relevance - 0.2×max redundancy` 贪心分数，最多 5 片；以源 ID/片段偏移规范排序，统一 water-fill/truncation 并再次检查拼接和分片编码预算。无可训练参数、无 response 选片、无生成式摘要。

| 组 | 操作 | 主要控制 |
|---|---|---|
| center_concat | 簇中心最近的最多 5 篇前缀，同 64 tokens 后重编码 | 公平主控制 |
| mean_pool | 相同中心前缀，分别编码后均值归一化 | 复用 P2 已有操作，不称新方法 |
| text_attention | 相同中心前缀，已有内容 attention pooling，温度 0.2 | 加权集合对照 |
| text_selection | 簇文本 query，片段 MMR 选取后重编码 | 隔离“选片”与“pooling”差别 |
| attribute_selection | 属性 query 选片后重编码 | 是否只是原始属性语义再加权 |
| graph_selection | 真邻域 query 选片后重编码 | 预注册主候选 |
| graph_pool | 与 graph_selection 完全相同的片段，attention pooling | 单独检验目标表示接口 |
| shuffled_graph | 打乱邻域 query，选片后重编码 | 图条件反事实 |

最终模型输入的内容 tokens 和特殊 tokens 分别报告。共享候选预处理、每组最终编码分别计费；不同组序列数量不同，不能称等 FLOPs。64 是相同上限，不保证短源簇恰好用满 64。

## 5. 审计、筛选与停止

工程门禁：预算零违规；共同 X/边/行 hash 相同；旧记忆不变；相同 seed 初始 student 参数 hash 相同；全部检查点重新加载验证指标；配置、数据、编码器和代码 hash 冻结；续跑拒绝不一致缓存。

历史 3 个顺序仅用于机制诊断。graph_selection 的独立确认启动门禁预先固定如下，均与 center_concat、text_selection、attribute_selection、shuffled_graph 配对比较：

- 两个数据集各自的平均最终 F1 至少 +0.5 个百分点；
- 两个数据集各自的阶段平均增益不为负；
- 各顺序最终 F1 不能低于对照超过 2 个百分点；
- 数据集平均最终 old/new F1 不能低于对照超过 2 个百分点。

门禁通过后才以固定方案运行 order1703/1704、student seeds110..115 的独立确认；不得在已看到分数后更改参数或换主候选。未通过则停止晋升，保留全部正负结果。graph_pool 是表示诊断，不能事后挑其最好分数替换主候选。

报告 final、stage mean、old/new、按历史任务最大值计算的 forgetting、配对 seed 差值和逐顺序差值。seed 不等同独立数据集，不能把 18 个 seed/order 单元当 18 个独立实验作强统计宣称。

## 6. 复现与产物

代码：`p2a_graph_text.py`、`run_p2a_graph_conditioned_text.py`、`test_p2a_graph_conditioned_text.py`。

服务器使用 conda tagsam 的 Python；Cora cuda:0、Citeseer cuda:1。正式输出 `results/p2a_graph_conditioned_text_20260918/{dataset}`，smoke 输出单列。保存 plan、分阶段 source/cluster/query 审计、文本证据、memory、student checkpoints、结果和验收文件。诊断原始数据和共享候选缓存不计入部署 memory，存储成本必须明确区分。
