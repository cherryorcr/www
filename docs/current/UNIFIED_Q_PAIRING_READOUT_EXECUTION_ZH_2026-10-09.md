# 固定 memory 的配对与 readout 诊断：执行登记

本轮在查看新对照分数前登记。目标是回答：当前节点级 X/文本配对是否被现有下游 Student 使用，以及完整候选 GM 与小批量训练是否一致。不修改凝聚算法或 Student，不直接增加 slot alignment loss。

## 设计

- 复用 `uq_text_path_20261008` 中已审计的原 In-loop CaT-Q、CaT-nearest5、GECC-hard 最终 memory。
- WikiCS/SportsFit 顺序 9201，BookHis 顺序 9202；construction seeds 271108/271109。这些都是开发数据，不算新的独立顺序。
- 每张 memory 以固定 permutation seeds 271011/271012，在同一到达阶段、同一类别内均匀随机置换文本字符串、embedding 和来源 ID。允许随机固定点，不挑选有利置换。
- X、标签、质量、深度、阶段、拓扑、文本多重集不变。置换按阶段可独立生成，不读取或生成新文本，不修改历史原档案。它是构造后的归因对照，不是新凝聚算法。
- 同一 Prompt-GCN、Prompt-SGC，Student seeds 150/151/152，2048 steps、batch 24。推理保持完整 transductive 图、prompt 分类；仅 report-dev。
- 6 个数据集/构造 cell，每 cell 三方法、两置换、两架构、三个 Student seeds，共 216 次新训练；复用 108 次已审计原始训练。
- 用固定初始化模型比较完整 group-class、完整 node-mean、32 个均匀配对 minibatch 的梯度。报告同类置换敏感性、与批采样噪声的相对量级。不将初始化梯度当作完整训练轨迹。
- 统计最终 memory 内文本唯一率、重复率和类内方差。它们不是原始图语义覆盖率，不能仅凭字符串不同断言覆盖充分。此前软硬失真诊断直接引用已有结果，不重新凝聚。

## 归因与下一步条件

先平均同一 memory 的三个 Student seeds，再平均两次置换；construction cell 是配对统计单位。两个置换不能冒充独立数据集或构造重复。

只有原版相对置换在两种架构的三图平均 F1 都至少高 0.5 pp、且至少两图方向一致，才把“必须强化类内对应关系”作为下一轮优先假设。即便通过也不能推出新 alignment loss 有效。

未通过时不自动新增逐行损失或改 Student。重点转为代理训练分布、特征表达能力或持续 setting 证据；保留真实负结果。Full-candidate GM 的置换不变性不等于小批量 Student 不变，也不等于文本无用。

不删除历史文件，不提交 Git。不保存重复 X/memory，仅写置换索引、哈希、结果与诊断 JSON。现有独立确认结果继续单列。
