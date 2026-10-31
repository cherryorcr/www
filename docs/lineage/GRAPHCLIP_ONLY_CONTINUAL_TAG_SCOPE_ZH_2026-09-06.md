# 持续 TAG 凝聚：GraphCLIP 唯一下游模型的研究范围

**历史范围，已由本日后续用户澄清更新：** 用户重新开放 GraphCLIP 与 TAGSAM 原生下游的选择，时间/类别到达亦未固定。“GraphCLIP 唯一”不再是当前约束。当前默认建议见 [首版设置、模型与修复顺序](D:/Adocs/简历/research/graph-condensation/TAGSAM/STARTING_ROUTE_SETTING_AND_MODEL_DECISION_ZH_2026-09-06.md)，尚未启动新实验。下文作为当时选择及 GraphCLIP 适配边界的记录保留。

日期：2026-09-06。依据：用户最新明确选择。本文覆盖此前“监督 CaT/PUMA 作为主任务”“GraphCLIP 仅作可选迁移诊断”“扩展其他下游模型”的建议，不改变历史实验结果。本轮未执行实验。

## 1. 已确定的目标

研究对象是随数据到达而持续更新的凝聚 TAG，GraphCLIP 是唯一目标训练与评测模型。

\[
(\widetilde G_{t-1},\Delta G_t)\longrightarrow\widetilde G_t,
\qquad
\operatorname{Train}_{\rm GraphCLIP}(\widetilde G_t)
\longrightarrow\text{下游任务效用}.
\]

核心判据是每阶段导出的 synthetic TAG 能否支持有效的 GraphCLIP 训练，并保留累计图文信息。模型是否沿用上一阶段权重是需要明确的评测选择，不替代独立重新加载基座后的数据效用评估。

后续不开展其他下游模型的训练或跨架构迁移实验。TAGSAM 原始 GCN+BERT 代码仍可用于只读理解方法，既有 GCN/GraphSAGE/D2/A2 结果保留为历史证据，不再扩展为新的下游研究分支。

## 2. 方法继承关系

- TAGSAM 是主要静态 TAG 凝聚参照：真实文本片段选择、合成结构/特征构造、student 在小 TAG 上训练、ASM 匹配图文相似度。
- GraphCLIP 是唯一目标模型。其现有图编码器、文本处理、预训练和训练优化应先按实际分支与 checkpoint 核实；不预设所有已发表优化都已存在于本地版本，也不未经核实替换掉其原生设计。
- CaT/PUMA 提供持续 memory 更新、Append 和训练策略的参考；不据此将主任务切换为它们的监督分类模型。
- DANCE 提供文本证据更新等近邻参照；本地可读 raw 的刷新条件与本项目受限历史访问分开讨论。

## 3. 区分两种 TAGSAM 到 GraphCLIP 的应用方式

| 方式 | 实际含义 | 地位 |
|---|---|---|
| 凝聚数据迁移 | 将已有 TAGSAM 格式的小图和真实文本转换为 GraphCLIP 可消费的训练数据 | 可作接口和数据迁移对照，不自动保证效果 |
| 凝聚目标适配 | 使用 GraphCLIP teacher/student，继承 TAGSAM 文本选择与 ASM 思路，优化能训练 GraphCLIP 的 synthetic TAG | 更直接面向目标模型的主要方法候选，尚未实现验证 |

原始 TAGSAM 论文默认 GCN+BERT，不能称其已完成 GraphCLIP 适配。现有 GraphCLIP C0 使用 GCond-style 梯度匹配，不是 TAGSAM ASM；其失败不等于 TAGSAM 到 GraphCLIP 的路线已失败。

GraphCLIP 的选择也不使旧 C0 自动变成必须成功的算法。后续比较的是适配后的 TAGSAM/ASM、已有强小集和持续更新方式，而非无限期只修当前梯度匹配目标。

## 4. 适配前需核对的接口

1. GraphCLIP 实际版本、checkpoint 来源及已有训练优化：明确图编码器、可训练参数、冻结文本参数和预训练暴露范围。
2. 图输入：synthetic X 的维度/来源、合成邻接、ego 采样、根节点标记和位置编码；不能把原始 GCN 节点输入直接视为已兼容 GraphCLIP。
3. 文本输入：保存来源明确的真实字符串，由目标 GraphCLIP 的 tokenizer/文本编码器重新编码；BERT embedding 不可冒充另一文本编码器的输出。
4. ASM 路径：GraphCLIP student 在合成数据上训练，再与 teacher 在相同、合法可访问的参考样本上比较图文关系。静态完整原图目标不能在持续阶段偷偷继续读取旧 raw。
5. 导出与评估：独立加载同一登记基座，训练冻结初始化/凝聚产物；列明文本、embedding、结构、adapter 及必要状态的真实成本。

## 5. 下游任务与标签权限

当前已有 GraphCLIP 开发协议使用 heldout-node zero-shot Macro-F1 为主指标，Accuracy、BA 和图文检索指标为辅助。用户本次确定的是唯一模型，并未要求改变任务或引入类别监督，因此先保留这一口径；后续核对 GraphCLIP 原生任务后，任何调整都须先明确协议再看新结果。

图文对齐训练与零样本节点分类评测可以共存，不等于采用标签监督训练。CaT/PUMA 的 class-wise matching 或伪标签来源不能直接混入当前 label-free 凝聚过程。原文复现与 GraphCLIP 同接口适配对照须分别命名。

静态 TAGSAM 的 five-way Accuracy 与本项目当前 Macro-F1 不是相同评测口径；旧 ArXiv t3 已暴露，所有现有 t1 结果仍是开发证据。

## 6. 后续研究顺序

先核实 GraphCLIP 的实际训练设计及 TAGSAM 文本/ASM 接口，形成同一 GraphCLIP 模型下的适配方案；再验证导出小 TAG 的训练效用。随后研究只使用旧 synthetic TAG 与当前 delta 时，文本选择、结构构造和匹配参考如何持续更新。

持续阶段比较同模型、同预算的 Append、普通重凝聚、full-history 参考及有证据支持的单一保持机制。前文提出的“图文对应在替换中丢失”仍是候选假设，不因本次选择 GraphCLIP 就视为已验证瓶颈或最终创新。

压缩预算仍主报固定 root 比例及真实字节，固定总容量是另一协议。主线目标仍为具有实质算法贡献的持续 TAG 凝聚研究，不新增其他下游模型来替代当前问题。

## 依据

- [TAGSAM 方法与评测原文摘录](D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/literature/sources/tagsam_blocks.txt:49)。
- [当前实验进度与归因](D:/Adocs/简历/research/graph-condensation/TAGSAM/CURRENT_PROGRESS_AND_ATTRIBUTION_ZH_2026-09-06.md)。
- [此前 CaT/DANCE 研究草案](D:/Adocs/简历/research/graph-condensation/TAGSAM/CAT_DANCE_BASED_CONTINUAL_TAG_RESEARCH_DESIGN_ZH_2026-09-06.md)：其文献差异和候选机制可参考，模型及主路线选择以本文为准。
