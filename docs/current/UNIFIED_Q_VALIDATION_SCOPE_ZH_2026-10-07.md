# In-loop CaT-Q 确认实验范围

## 冻结内容

主候选仍为 edge-free In-loop CaT-Q；no-GM 是预注册的简化消融，不按每个数据集的最好结果替换主方法。构造阶段只访问当前 raw delta 和本方法旧 synthetic memory，旧前缀冻结。Student、MiniLM、2048 steps、batch 24、Student seeds 150/151/152、30 outer steps、320 字符和 top-1 硬文本前向不变。

本轮不做参数搜索、不删除旧结果、不提交远程仓库。所有分类结果仍为 report-dev，不是封存测试，也不是未见类别或跨数据集 zero-shot 证据。推理使用完整原图的传导式评估算子；历史 raw 访问限制针对构造器，而不是该评估器。

Student 每阶段从该阶段累积 memory 重新训练，种子配对，不跨阶段继承优化器或模型权重。持续保存的是凝聚 memory；不能将这张表描述为直接模型持续学习实验。

## 实验矩阵

| 部分 | 数据与重复 | 比较对象 | 作用 |
|---|---|---|---|
| 补齐控制 | SportsFit、BookHis；参考顺序；构造 seed 271006；3 Student seeds | no-GM、fixed-Q | 与已有带 GM、CaT-CCTP 配对 |
| 独立确认 | WikiCS、SportsFit、BookHis；3 顺序 × 3 构造 seed；3 Student seeds | 主候选、no-GM、CaT-CCTP | 每方法每图 9 个顺序/构造 cell，共 81 个方法 cell |
| 同协议基线 | 三图；参考顺序与构造 seed；3 Student seeds | GECC-hard、PUMA-inspired、随机 coreset、k-center | 使用同一冻结评估器，不能与旧 Student 数值直接混表 |
| 固定内容归因 | 三图；5 阶段；3 Student seeds | 不变、自环+空文本、自环+CCTP 文本、自环+类内文本置换、二值图、加权图、同边数置换图 | 同一 Q/X/text 检查拓扑；同一 X 检查文本 |

参考顺序保留现有 groups，称为 9101，不重新调用随机排列以免实际类别顺序改变。新顺序 9102/9103 保留 base 类别数和各阶段类别数，仅对类别排列。构造种子为 271006/271007/271008。

独立顺序使用参考 pilot 的相同逐阶段预算表，即 fixed-schedule。复用 pilot 的参考顺序/构造 seed 时，必须逐项验证实际 groups、budgets、pool/report masks、stage IDs、Student、训练步数以及评估代码哈希一致；不能仅因为都叫“5%”就混合。其它顺序不重新按 5% 改预算。

## 基线名字与做法

| 名字 | 实际实现 | 改动与限制 |
|---|---|---|
| CaT-CCTP adapter | 随机 Encoder class-wise MMD，3000 次优化自由 X；graph-nearest top-5 文本重编码 | current-delta-only，旧前缀冻结，self-loop，同一 Prompt Student；不是官方原生 TAG 结果 |
| GECC-hard TAG adapter | 固定 hop 加权传播，hard Lloyd 聚类，20 restarts，SSE 与 cluster-size penalty 选 restart | 仅当前 delta；不是官方 fuzzy/累积 raw GECC；top-5 TAG 文本附着 |
| PUMA-inspired TAG adapter | 当前 delta 一跳传播，线性 pseudo teacher，0.8 置信阈值，CaT matching | 在标注 pool 中确定性留出一部分作 pseudo 候选；非官方完整 PUMA；不得隐藏这种监督使用差别 |
| Stratified random coreset | 按相同 class-ratio 配额随机选真实节点 | 真实 X 和对应文本；320 字符后重新编码；self-loop |
| Stratified graph-text k-center | 同类联合归一化 X/text 空间，exact farthest-first | 相同配额与真实文本；不使用未披露的候选池下采样 |

这些是可审计的机制适配对照，不等于官方复现。旧报告的 Static-TAG full-history 实际是项目内部 joint graph-text cluster 构造器，不是官方 TAGSAM/STAD；不能把它重新命名成公开静态 TAG 强基线。官方静态 TAG 凝聚仍需另外的适配/复现证据，本轮不虚构这部分完成状态。

## 固定内容诊断

每阶段只构造一次主候选，读取最终 Q 的只读诊断视图，不将稠密 Q 保存为长期 memory。文本和拓扑变体不回传影响下一阶段主候选的构造。

- 文本：空字符串是“无有用节点文本”控制，仍保留相同 prompt 和 Student 损失；CCTP 使用同一 X 为 query；类内置换守恒文本多重集，打乱 slot 与文本对应。来源 ID 随文本一起改变。
- 拓扑：固定 Q/X/text，比自环、同一 Q 的二值图、加权图、同边数置换图。每个旧诊断前缀本身冻结，不生成跨阶段边。
- 加权图：使用 Q 诱导权重与对称归一化、单位自环；同一 Student 网络、损失、种子与步数，只注入不同训练传播算子；原图推理不变。
- hard 与 soft 差异是材料化误差诊断。现有主算法已经是 hard forward，不能再声称它“训练时用 soft、保存时变 hard”作为既定失败原因。
- 记录一/二跳消息范数、类别均值漂移、hard-soft 消息差、边数与同质性。它们用于分析，不用于按 F1 调参。

固定内容实验只能说明对同一份特征/文本增加某种传播的影响，不能直接证明一个重新联合优化的拓扑方法有效或无效。boundary sketch、旧记忆更新、长链和混合到达不属于这一最小确认批次。

### 完成后的可复现性审计说明

固定内容诊断与旧 pilot checkpoint 并非按位一致；它们是同配置的独立重建，不是同一份材料化 memory。最终汇总将诊断单列，并只与该次构造的 unchanged/self-loop 配对，不纳入九-cell 主结果，也不拿其不同拓扑与旧 pilot 相减。

逐阶段哈希和最终存档确认诊断内部所有 arm 的 X/labels/mass 相同，拓扑 arm 的文本也相同。额外 WikiCS 实验确认 callback 不改变 Q、memory 或随机状态；无 callback 的重复构造本身仍存在差异。故此前 RNG 污染只是未证实猜测，不能将预防性的 RNG 保存/恢复说成根因已解决。完整结果报告保留数值可复现性警告。

## 统计与成本

先平均三个匹配 Student seeds，再报告 9 个顺序/构造 cell 的均值与标准差；配对差的 bootstrap 同时重采样顺序轴与构造 seed 轴，不把 27 次 Student 运行伪装成 27 次独立构造。仅三个顺序，CI 仍属探索性稳定性证据。

记录 F1、Accuracy、stage-average、final Old/New F1、worst-stage、positive-cell ratio、worst paired cell、memory rows、构造时间、Student 时间及峰值显存。不能用 incomplete cell 的有利前几个阶段替代 final 结果。

服务器空间保护阈值为 1.5 GiB，旧文件不删除。只保存每个实验臂最后一个可恢复 memory，并保留逐阶段指标与哈希。存储账需包括来源/support IDs 等辅助元数据，不能只统计 tensor 行数。

最初几分钟一条基线辅助任务曾与确认任务共享 GPU，已停止并转为后置队列；可能受影响的运行耗时不能作为独占 GPU 效率结论。分类指标与协议不因此更改。

## 实现与输出

- `tools/run_unified_q_inloop_validation_20261007.py`：冻结确认队列。
- `tools/run_unified_q_aux_validation_20261007.py`：基线与固定内容诊断队列。
- `code/v2_llm_fusion/run_unified_q_matched_baselines_20261007.py`：同评估器的公开机制适配。
- `code/v2_llm_fusion/run_unified_q_fixed_content_20261007.py`：固定内容诊断。
- `tools/summarize_unified_q_validation_20261007.py`：逐指标复算、协议核对和配对统计。
- 本地汇总：`results/unified_q_inloop_validation_20261007/`。
- 远程同名结果目录，核心构造保持原 `experiments/unified_q_inloop_cat_20261007/code` 快照。
