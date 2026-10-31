# 持续图凝聚相关工作复核与外部基线推进

日期：2026-09-27。状态：文献复核及 CaT/GECC 小图 TAG 机制适配已完成，360 次正式 Student fit 与重载审计通过；PUMA 尚未实现。见[外部基线结项报告](EXTERNAL_CONTINUAL_ADAPTER_COMPLETION_ZH_2026-09-27.md)。

## 1. 本轮决策

Semantic-Recursive 保留为冻结主线。正式性能比较面向外部 coreset、公开 TAG 凝聚适配和持续图凝聚适配；C0 仅作内部消融、资源控制和回退。独立顺序上相对 C0 接近零，不等于相对所有外部基线无效；反过来，外部方法未测也不能视为已胜出。

方法形式化优先保持输出一致，准确描述现有实现，不为加强理论叙事改写算法或回选最好 cell。

## 2. 是否已有持续图凝聚

有。CaT 和 PUMA 明确研究持续图学习中的图凝聚；GECC 研究演化图的增量凝聚；OpenGC 研究凝聚图在 open-world 分布变化下的泛化。因此不能写“此前没有持续图凝聚”。

本轮核对的文献尚未提供同时包含文本 payload、历史 raw 不可回读和统一图文对比 Student 的现成严格对照。该结论是检索范围内的证据状态，不是不存在相关工作的证明，也不能通过要求所有方法遵守本文内部设计来排除竞争方法。旧 memory 不可改写是当前方法选择，不是外部基线的强制资格条件；用户认可的核心约束是历史完整 raw 不可回读。

## 3. 相关工作与边界

| 工作 | 原方法核心 | 与本文的可比部分 | 不能直接等同的部分 |
|---|---|---|---|
| CaT | 对当前图作按类分布匹配，生成 synthetic memory，在 memory 中训练 | 当前增量凝聚与历史压缩记忆 | 数值特征、监督学习；没有文本合成与图文目标 |
| PUMA | CaT 扩展：伪标签、单次传播、宽随机编码器、edge-free memory、从头重训 | memory 可重训性、新旧学习失衡 | 监督/伪标签、MLP 训练与 GNN 推理；并非本文 InfoNCE |
| GECC | propagated-feature clustering、中心代表、演化时中心继承 | 聚类凝聚、可追溯性和增量成本 | 不能仅凭继承中心就认定 no-old-raw；累计图访问须按原论文与代码核验 |
| OpenGC | 时间不变凝聚、结构感知分布偏移、KRR 式训练 | 图变化后的泛化与高效凝聚 | 原目标不等于逐阶段 no-old-raw TAG memory append |
| TAGSAM / STAD | 静态 TAG 凝聚 | 文本和合成图构造 | 当前本地适配与原生监督/模型设定须分别标注 |
| CCC | 每个动态图快照生成 condensed graph，用动态 GNN 顺序处理历史 condensed snapshots，并对结构变化区域选择性拼接历史 embedding | 动态拓扑变化下的历史凝聚信息利用 | 保存/更新的是动态模型表示，使用数值特征；不是本文独立重训的 TAG memory-only Student |
| CA-DGCL | 跨时间 condensed snapshot 做节点链匹配，构造 feature-time tensor，经 Tucker 分解生成节点并连接到当前图 | 跨阶段历史信息和新节点生成 | 需要动态图快照、节点链、训练中的重生成/重连；不是本文 append-only graph-text payload placement |

PUMA 已有从头重训策略，不能把“每阶段重训 Student”单独写成本项目首创。GECC 已有演化聚类与可追溯来源，不能把“继承中心”或“可追溯”单独作为全新贡献。本文需要强调并验证受限访问下的图文 payload 一致性、固定容量下的 placement 机制及跨阶段效用。

## 4. 基线分层与公平性

| 层级 | 方法 | 实验角色 | 当前状态 |
|---|---|---|---|
| 主比较 | Semantic-Recursive；random、k-center/FPS、herding、medoid | 相同合法输入、预算、Student 和评价 | 小图两个顺序已有结果 |
| 公共持续适配 | CaT-MMD-TAG、PUMA-inspired TAG | 检验持续分布匹配机制 | CaT 适配四条件完成，平均 Final F1 54.62；PUMA 尚未实现 |
| 演化聚类适配 | GECC-hard-TAG | 聚类凝聚比较 | 受限 hard 分支四条件完成，平均 48.70；未复现累计 raw evolving/fuzzy 分支 |
| 静态/离线 | TAGSAM、STAD、全历史重凝聚 | 额外历史访问的参考 | 既有适配仅按原实验口径报告 |
| 完整数据 | equal-step raw；long-train raw | 同 Student 性能保留率分母 | 按数据集、顺序和指标精确配对 |
| 模型持续学习 | delta-only fine-tune；replay + delta | 持久模型状态的对照 | 单列状态资源及成本 |
| 内部消融 | C0；identity placement 等 | 解释算子来源 | 不当作外部 SOTA |

同 topology 的 coreset 表是控制内容的实验，不完全代表每个 coreset 的最佳端到端构造。应保留它，同时追加允许各方法使用其合法原生拓扑的公平主比较。外部方法不应被迫复用 SR 的 slot/topology，除非明确标为 controlled adaptation。相同节点数还不等于相同存储：文本长度、边、优化器/模型状态和辅助 synthetic memory 均需记账。

## 5. 最小推进顺序

1. 冻结现有 SR 结果与代码来源，先完成准确形式化和输出等价审计。
2. 核验公开 CaT/PUMA/GECC 代码、标签权限、原生 memory、raw 访问范围和文本补全接口。
3. 在 Cora/Citeseer 先做 CaT-inspired 与 GECC-style 最小适配；保留合成 X，不用真实节点 coreset 冒充凝聚；文本附着规则统一、来源合法，并单列适配改变。PUMA 不能只以 CaT 改名，须实际实现其独有机制。
4. 先跑小规模 correctness/smoke，再冻结构造超参与独立 seeds，补同协议多顺序比较。若 adaptation 引入标签监督，匹配构造标签权限并披露，不能将本文按类分组误称全流程无标签。
5. 同步报告 fixed-rate 和 matched-budget 两层；matched-budget 仅固定每阶段 memory 轨迹，不应悄悄减少/重采样 raw pool。
6. 评价以外部基线 paired 差异为主，C0 用于归因；在同一数据集/顺序/Student 下另报告相对 full-raw 的 F1、Accuracy 保留率，保留大于 100% 的比值但不解释为理论超越上界。

多顺序推断以 dataset/order 为主要分组单位；Student seeds 是组内重复。既有把四个 cell×六个 seeds 合并的 t 区间仅作描述，不能当成 24 个独立到达顺序的泛化置信区间。

## 6. 检索结论与仍需核验的内容

本轮进一步读取了 CCC（arXiv:2512.11317）和 CA-DGCL（arXiv:2607.11112）的 arXiv HTML/API。两者都属于动态图持续学习中的凝聚/历史信息 replay 方向，但都没有文本属性图、冻结 MiniLM、图文 InfoNCE 或本文的独立重训 memory-only 接口。它们应进入 related work 和 closest external 层，不应被写成已经完成的同协议跑分。

检索范围内目前可确认的分层结论是：

1. CaT、PUMA、GECC、CCC、CA-DGCL 和 OpenGC 证明持续/演化图凝聚已经有公开先例。
2. 本轮没有确认同时满足 `text-attributed graph + continual condensation + historical raw inaccessible + graph-text Student retraining` 的公开方法。
3. 这只能支持“本文研究接口仍有空白”，不能支持“本文已经击败 continual TAG SOTA”。
4. CaT/GECC 官方代码与主要接口已核验，适配限制见冻结协议；PUMA、CCC/CA-DGCL 的实现、标签和历史访问细节仍需进一步核验。没有把 CaT 的适配当成 PUMA 的完整实现。

后续检索保留 `continual graph condensation`、`evolving graph condensation`、`text-attributed graph condensation` 与 `continual text-attributed graph` 关键词，并保存检索日期、原文链接和排除理由。

## 7. 已核对的原始资料

- CaT: Balanced Continual Graph Learning with Graph Condensation，arXiv:2309.09455，重点核对 Condensed Graph Memory / Train in Memory 与实验设置。https://arxiv.org/html/2309.09455v2
- PUMA: Efficient Continual Graph Learning for Node Classification with Graph Condensation，arXiv:2312.14439，重点核对 retraining、pseudo-label、edge-free memory 和 one-time propagation。https://arxiv.org/html/2312.14439v2
- Scalable Graph Condensation with Evolving Capabilities，arXiv:2502.17614，重点核对 propagated-feature clustering、evolving inputs 与 centroid initialization。https://arxiv.org/html/2502.17614v3
- Graph Condensation for Open-World Graph Learning，arXiv:2405.17003，重点核对 temporal invariance 与 evolving evaluation。https://arxiv.org/abs/2405.17003
- Condensation-Concatenation Framework for Dynamic Graph Continual Learning，arXiv:2512.11317，重点核对动态快照凝聚、历史 embedding 生成和 change-region replay。https://arxiv.org/html/2512.11317
- CA-DGCL: Dynamic Graph Continual Learning via Condensation and Attachment，arXiv:2607.11112，重点核对 cross-timestamp node chains、Tucker decomposition 和 generated-node attachment。https://arxiv.org/html/2607.11112

上述条目用于机制与协议核对，不冒充已完成官方复现。出版年份/venue 以 publisher 元数据核准后进入正式 BibTeX。

## 8. 已完成的小图外部比较

本轮保留 SR 原产物，不再要求外部方法使用 C0 stage0 或 C0 topology。所有方法同条件同 rows、原两层 GCN/InfoNCE、Student seeds 150-155、同 report-dev。最终四条件均值为 SR 53.02、CaT 适配 54.62、GECC 适配 48.70、k-center 独立构图 54.18、medoid 独立构图 47.32；Accuracy 分别为 59.59、59.96、52.53、59.43、53.11。

SR 相对 GECC/medoid 在 4/4 个条件均较高，相对 CaT/k-center 各 1/4 较高。SR 逐 seed 复现旧结果，但旧共享初始图/拓扑 coreset 表的优势不能推广为独立构造下的稳定优势。两种协议都应保留，分别回答内容控制和端到端构造问题。

完成范围是两图、两个既有顺序、单组构造 seed、六 Student seeds，不是 3-5 个新独立顺序或原生方法完整复现。下一步先用外部强对照作独立确认和预算/文本资源控制，再讨论不大改 SR 的有限优化。
