# 冻结 v4 后的外部 TAG 基线比较协议

## 目的与优先级

依照最新要求，暂停 v4 消融、全局 v4 开发与新组件调参，先比较外部方法。不是要求凝聚超过完整数据，而是测量用约 5% 节点训练 student 能保留多少完整数据性能。只做 Cora/Citeseer，不启动大图。

主指标为最终阶段 report-dev Macro-F1、六个 student seed 的均值/标准差，以及均值比 `100 * method_F1 / raw_long_selected_F1`。阶段平均另报，不能与最终 F1 混算。完整数据分母是相同 GraphCLIP 任务的 8192 步、selector-dev 检查点选择参考，不是理论上界。监督参考另表，不作同任务保留率分母。

## 统一不变项

- 冻结 v4 已审计 memory，禁止根据新报告分数修改主方法。
- 数据、类别顺序、pool/selector-dev/report-dev/test 划分沿用五层协议。
- 三阶段累计合成预算：Cora 25/67/94，Citeseer 38/86/111。
- 相同 GCN384、冻结 MiniLM、InfoNCE、Adam0.001、batch16、student seeds100..105、按合成预算计算的相同步数。
- 保留 baseline 自己的合成拓扑，STAD 边权不得二值化或重复归一化。
- 原始历史可读的静态方法明确标注，不当作满足持续约束的方法。
- 本轮不评价 test，不按 report-dev 选构造设置。外部方法的合成节点可以使用合法 pool 标签，标签监督和生成模型来源必须披露。

## STAD 首轮固定适配

来源：<https://arxiv.org/abs/2607.20477>，作者仓库 <https://github.com/laiyurui/STAD-Semi-Supervised-Text-Attributed-Graph-Distillation>，提交 `5aaf1a06e44638499373b1358c85d1546187edae`，MIT 许可。名称固定为 **STAD-local-canonical**，不是官方完整设置复现。

1. 每阶段只将累计已见合法 pool 的属性、文本、诱导边交给构造器。预算取该 pool 的 5%，不是全图节点的 5%。
2. 使用官方 GCN+MLP+attention 自训练代码。为保留半监督机制，在 pool 内按类别分层取 80% 初始标注、20% 内部验证兼伪标签集合，随机种子 `15+stage`；不会访问 report-dev/test。保留官方每轮600步、最多10轮、置信度比例0.1逐轮增加0.1及提前停止。
3. 官方 fused logits 聚类、类别-簇再划分、均值属性和加权邻接压缩。使用固定 KMeans `n_init=10`，保留种子。
4. 官方 TF-IDF + MiniLM 关键词排序，最多256个关键词；每个合成节点生成3个文本候选，每候选最多256生成 token，temperature1.2、top_p0.85。
5. 使用服务器本地 `Qwen3-32B-AWQ-vLLM`，不是官方默认 `qwen3-max`。所有文本留在服务器；缓存请求/响应、种子与 token 用量。失败必须报错，不允许静默退化为关键词串。
6. 官方100个组合候选、两步图平滑及 Sinkhorn 距离，筛选前5个；独立构造选择 seed `7000+stage` 的监督 GCN600步，按 selector-dev 选择。用确定性的原始候选序号处理平分，不看 report-dev。
7. 合成文本重新编码：未归一化 MiniLM embedding 作为 STAD 合成图属性，单位归一化 embedding 作为 canonical student 文本输入；不使用真实节点作为合成节点。真实图推理属性仍为已有 canonical 属性。
8. 官方的 test 评分路径不执行。下游六 seed 使用统一 GraphCLIP，而非 STAD 官方监督交叉熵；因此评价的是迁移到本任务后的凝聚效用。

这是等节点预算，不是等文本 token、等LLM算力或等历史访问权限。逐阶段记录文本量、边数、构造耗时、下游耗时及峰值显存。一次构造、多 student seeds 尚不能证明跨构造随机性的稳定性。

## 其他方法排查

- TAGSAM：已有 self-loop/semantic 响应匹配适配。完整 NPMI 文本与官方编码器/训练时长仍未复现，保留此限制。
- DANCE：<https://arxiv.org/abs/2601.16519>，联邦 TAG、轮次刷新及 evidence pack 机制。论文 HTML 未提供作者实现链接，当前 GitHub 检索未找到可确认官方仓库。先列相关工作和接入阻塞，不能虚构分数，也不把单机自行简化实现称为官方 DANCE。
- 通用 GCond/GCDM 等只能列为 TAG 特征上的适配方法，不能算新增直接 TAG 方法。

## 验收

保存源文件哈希、官方提交、协议 manifest、pool 来源、合成文本候选、normalized edge weights、逐 seed 模型和结果。复载所有报告模型重算指标，核对预算、独立初始化、实际训练步数、无 holdout 来源及官方函数使用。旧无权图分支必须与历史 `norm_adj` 逐值一致。

## 补充诊断：原生风格监督检查

该诊断是在观察到 STAD 的 GraphCLIP 迁移分数较低后提出的，不是事前提出的独立确认实验。以下配置在读取本诊断报告分数前固定，不据此修改 STAD 合成图或 v4。

- 仅 Cora/Citeseer 最终阶段，复用冻结 STAD 的 `memory.pt` 和 `core.pt` 中的合成标签，不重新构造或调用 LLM。
- 两臂：STAD 合成图；完整合法已见 pool 诱导图。两者均使用官方两层 GCN，hidden64、dropout0.5、Adam lr0.01、weight decay0.0005、600 full-batch epochs、seeds100..105。
- STAD 保留其已归一化加权邻接；raw 使用 canonical 对称归一化邻接。共同推理图与 canonical 完全一致，为真实全图。
- 每轮仅用 selector-dev accuracy 选检查点，平分保留最早；训练结束后再评价终点和所选检查点的 report-dev Macro-F1/accuracy。禁止访问 test 标签。共24次fit，全部复载验算。
- STAD 以合成伪标签训练，raw 以合法 pool 真标签训练。两臂用于检查监督训练效用，不作为所有方法的公平监督排行榜。
- 名称为“STAD 原生风格监督检查”，不是官方分数复现。数据划分、LLM、文本编码器等适配限制仍然存在；本结果不能用作主表的性能保留率分母。
