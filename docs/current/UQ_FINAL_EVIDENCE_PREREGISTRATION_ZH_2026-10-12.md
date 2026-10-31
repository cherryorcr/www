# Unified-Q-EF 最终证据补充：预注册

本轮回答三个问题：高保真相关方法是否仍有竞争力；共享分配的实际作用是什么；ArXiv 在合法池真实 5% 下是否稳定。既有 report-dev、sealed test 和负结果不覆盖、不改名为新实验。

## 冻结设置

- 主方法：Unified-Q-EF，当前增量图感知、无显式合成边、冻结历史 synthetic prefix。
- 复用 2026-10-10 构造快照；不引入 EM、attention、拓扑或旧记忆更新。
- 固定累计比例 5%，分母为排除 report-dev/test 后的合法构造池；每阶段仍保证新类别至少一行，记录任何预算修正。
- ArXiv 确认：orders 9101/9102/9103 × construction seeds 271006/271108/271109 × Student seeds 150/151/152；方法为主方法、CaT-CCTP、GECC-hard。
- Q 优化 30 步，温度/学习率/损失权重不变；Sinkhorn 最大 4096 次，容差 1e-4，不以结果调整数值设置。
- 共同 Prompt GCN Student：2048 步、batch 24、相同 MiniLM 和 prompts。原始图用于共同 transductive inference，不向构造器提供历史原始特征或文本。
- report-dev 与 sealed test 使用同一次训练的预测，分别保存；实验前固定版本，不根据任一新测试结果选择方法或超参数。
- 新结果写入 `uq_final_evidence_20261012`。仅最终确认可保存压缩 memory；不持久保存 dense Q。

## 基线保真度

1. TAGSAM：核对官方邻域提取、语言模型互信息评分、去冗余摘要及响应蒸馏。共同文本编码器和持续访问限制属于明确适配；不称官方原协议 reproduction。
2. STAD：保留双路径协同、自训练、聚类纯化、关键词生成和 Wasserstein 文本候选选择；记录局部生成模型和原生 API 的差别。若无法保留核心，则列为 incomplete/inspired，不冒充完成。
3. GECC-soft：类内 fuzzy c-means、propagated features、balanced restart selection、软分配池化；文本用与 GECC-hard 相同的 CCTP。soft fuzziness 固定 1.3，不做测试集调参。

### 实现前的明确适配

- TAGSAM 直接执行 pinned 官方 GPT2-large 词概率与 PMI，保留随机中心、最多四跳/四个来源、五句成组、60% 去冗余选择及加权特征。修复官方尾批未计算、空组除零；统一候选及最终 320 字符预算。响应蒸馏沿用已审计的 100×10 unroll，共同 Student 不使用其学出的学习率。不能称为原协议完整复现。
- STAD 使用本地 Qwen2.5-1.5B-Instruct 替代官方 qwen3-max API（记录 resolved revision），不重新启动 vLLM；保留三候选、100 个组合、Wasserstein 前五和内部验证选择。生成器容量差异必须进入限制表。
- STAD 在合法当前池内按类分出 10% supervised、10% internal-validation、80% pseudo-label pool；不读取外部 report-dev/test。使用共同 MiniLM 表示执行官方双路径自训练、聚类纯化和文本匹配，另外池化原始 X 以满足共同 Student 输入维度。该半监督标签用量与全监督主方法不同，必须单列说明。
- STAD 保留加权合成拓扑用于原生文本候选选择；共同 Student 现有邻接接口使用对应二值边，不声称加权邻接完全相同。主方法仍为 EF，STAD 的拓扑不可由 EF runner 静默丢弃。
- STAD 生成模式固定：每个 cluster 三个样本，最多三次格式重试；要求每个样本保留官方 `[[LABEL]]`/`[[SUMMARY]]` 格式，失败则停止并记录，不改用关键词拼接代替生成。空关键词词表保留类别与数据集背景进行生成，并计数。格式重试、容量差异及 320 字符处理均列入适配偏差表，不因结果改变规则。
- STAD 生成模型固定 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，权重大小 3,087,467,144 bytes，SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`；使用可信 HTTPS 下载，必要时通过 SSH/Tailnet 传输，目标端再次校验，不关闭 TLS 验证。服务器 Hugging Face 下载遭遇不可信证书、SSH/Tailnet 传输过慢后，改用可信 ModelScope HTTPS 镜像；仍以固定的官方 Hugging Face LFS 摘要校验全部权重，记录真实下载来源。

## Q 归因

- 共享/独立分配使用完全相同初始化、目标、迭代数和 residual 设置。
- 分开比较有/无 residual；报告自然参数量差异，并增加相同 active parameter budget 的共享/独立比较。
- 独立分配不以翻倍参数量冒充等参数预算；等预算控制的参数化和初始化误差必须显式报告。
- soft-only 对照在构造前向及最终 memory 中都保存软文本 embedding；字符串仅为参考来源，不声称其能重编码恢复 embedding。
- 输出 slot 覆盖、assignment entropy、文本来源多样性、soft/hard cosine gap、图文支持一致性、梯度路径和构造时间/显存。

## 汇总准则

报告 F1/Acc、Old/New、阶段均值、构造时间、峰值显存、配对差值的 CI、positive-cell ratio 与 worst cell。27 个 Student 读数有共享 memory 的依赖，主 CI 以 9 个 order/construction cells 为配对单位，Student seeds 先在 cell 内平均，不把 27 个读数当独立样本。

旧 ArXiv 6773 行 / 118539 合法节点 = 5.7137%，保留为 legacy-budget 参考，不能标为真实 5%。已完成的 canonical 真 5% 5927 行结果可复用作单顺序补充，但不替代新增多顺序确认。
