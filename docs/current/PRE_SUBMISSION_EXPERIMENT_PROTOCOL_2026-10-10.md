# WWW 提交前实验协议与证据边界

## 目的

本文件冻结 Unified-Q edge-free 的提交前验证范围，避免用不同 Student、不同推理算子或不同历史数据访问边界拼接出不公平的主表。

## 主协议

- 数据集：WikiCS、SportsFit、BookHis。
- 持续划分：每个数据集 3 个预注册类别顺序；当前主 matched-operator 表使用 `9101/9102/9103`。
- 构造种子：`271006/271108/271109`。
- Student seeds：`150/151/152`。
- Student：Prompt GCN 为主；Prompt SAGE 为跨架构验证。
- 训练：2048 steps，batch 24；冻结 MiniLM；推理使用同一 `real_adj`。
- 构造访问：当前增量 raw delta 与自身 frozen synthetic prefix；不回读历史 raw。
- Memory：固定累计 budget；当前主表为 edge-free memory，Student 使用统一图算子。
- 指标：Final Macro-F1、Accuracy、stage-average F1/Accuracy、Old F1、New F1、worst order/stage、construction time、Student time、peak memory。
- sealed test：方法冻结后单独运行，不能用于选择组件或参数。

## 证据分层

1. **主方法**：Unified-Q edge-free：共享 class-masked Sinkhorn assignment `Q`、in-loop distribution matching、Q 诱导的文本硬/软一致材料化。
2. **外部适配基线**：CaT-CCTP、GECC-hard TAG adapter、PUMA-inspired TAG adapter、TAGSAM-response adapter。它们不是官方完整复现，论文中单列适配规则。
3. **内部控制**：no-GM、no-DM、fixed-Q、frozen-text、no-propagation、no-residual、no-text-init、no-Q-anchor、DM-only，以及 shared-Q 对 independent-Q 的归因。
4. **补充数据集**：Cora、Citeseer 的旧 Unified-Q 对照结果作为补充证据；其 topology-enabled 和旧 Student 协议不并入当前 edge-free 主排名。

## 不允许的结论

- 不把 Prompt Student 相对 Legacy Student 的共同提升写成 Unified-Q 的方法收益。
- 不把 GECC-hard 写成官方 GECC 原样结果。
- 不把三次类别顺序当成充分的统计显著性证明。
- 不在 sealed test 上调参。
- 在没有合成拓扑验证前，不宣称 Unified-Q 已经解决 TAG 的结构保持或链路预测。

## 当前完成状态

- Matched Prompt/Legacy GCN：72 cells，已完成。
- Construction-seed shared-Q 归因：26/27 cells，正在补齐唯一缺失 cell。
- Prompt SAGE 冻结 memory transfer：27/27 cells，已完成。
- 核心单 cell 消融矩阵：已完成，结果需与 seed 归因一起解释。
- Cora/Citeseer：已有旧协议补充结果，需单列并注明协议差异。
- Sealed test：已有旧 sealed 结果；最终候选冻结后需重新生成一次 sealed manifest。
