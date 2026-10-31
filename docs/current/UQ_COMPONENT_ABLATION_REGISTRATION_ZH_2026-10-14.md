# Unified-Q-EF 当前协议组件消融登记

本轮承接五数据集基线汇总，组件表与外部论文方法表分开。保留所有已有结果，包括共享 Q 对独立 Q 未稳定领先的结果，不按 sealed 分数挑选模块或参数。

## 已有证据与复用边界

- 五数据集主基线面板 50/50 完成，按数据集列出的 F1/Accuracy、适配说明与图表见 `results/uq_baseline_expansion_20261013/README.md`。额外原生参考继续运行，缺失值不写成零。
- 三个大图已有共享/独立分配、有无 residual 的 3 construction seeds 控制；equal-count 子空间只是敏感性实验。hard 前向/保存与真正 soft 前向/保存的对照已有主构造种子结果。
- 旧 random-Q 同时关闭优化，旧 random-5 使用 CaT 特征；旧大图消融多数只评估 report-dev 且使用旧固定预算，不能拼入本轮 exact-5% sealed 表。
- 复用须核对类别顺序、各阶段行数/预算、pool/report/test 掩码、阶段 ID、Student 配置/种子/预算和共同构造源码哈希。

## 新增实验矩阵

数据集：ArXiv、SportsFit、BookHis。固定 order=9101，construction seeds=271006/271108/271109，Student seeds=150/151/152。先完成首个构造种子，再执行其余两个种子，执行顺序预先固定，不随结果变化。

| 组件对照 | 唯一变化或解释 |
| --- | --- |
| no-GM | 仅令 gradient matching 权重为零，其余保持主配置 |
| no-DM | 仅令 distribution matching 权重为零 |
| structured fixed-Q | 固定结构化 Q0，仍执行 30 步 residual 优化 |
| randomized-init learned-Q | 对同类初始代价的 raw 行随机置换，保留分值尺度、类别掩码和容量；仍学习 Q/residual |
| randomized-init fixed-Q | 同一随机置换，冻结 Q，仍优化 residual；与上一项构成初始化×学习的控制 |
| class-shared payload | 固定主方法所有 X/标签/质量/阶段属性，每类使用一个真实文本 medoid |
| random-5 payload | 固定上述属性，每节点随机选五条当前同类文本，共享 320 字符总预算并重新编码 |

后两项为 **存储文本干预**：主构造反馈仍使用主方法文本，Student 使用替换后的存储文本；它们不等同于端到端重新训练文本构造。各阶段只附着当前 delta 的文本，替换分支自己的历史前缀也冻结。若历史主方法 memory 未保存，必须重建并逐阶段核对原 memory hash，复用已有 Student 分数而不重复评估主方法。

## 协议与诊断

统一累计合法池 5%（含整数舍入/类别覆盖下限）、current-delta-only、frozen synthetic prefix、edge-free、共同 MiniLM、Prompt GCN 2048 steps/batch 24。report-dev 与 sealed test 分开保存。新增对照是事后机制诊断，不称为独立盲测；不会用它们改写已冻结主表。

报告最终 F1/Accuracy、Old/New F1、阶段均值、构造 wall time 和 CUDA allocated peak。三 Student 种子的标准差与三构造单元间标准差分开；paired CI、正单元比例和最差单元以构造单元为统计单位，不能把 9 个 Student 评估当作 9 个独立构造。

报告 hard slot coverage、positive-mass coverage、行分配熵、列支持熵/有效支持数、unique source/text 比例、类内文本 cosine、多模态支持一致率、soft/hard cosine gap，以及 Q/residual 梯度和相对更新。平衡 Q 的 positive-mass coverage 接近 100% 是约束结果，不当作有效性证明。

## 实施约束

原 frozen snapshot 不改动；随机初始化使用从已核验控制实现派生的独立源码。运行前检查主配置数值等价、fixed-Q 数值等价、随机初始化/学习可区分、文本干预不改变 X/预算、诊断不改变 RNG。仅使用空闲 GPU1；不重复基线 GPU0 队列。阶段 checkpoint 可恢复，不持久化 dense Q；低于 8 GiB 磁盘空间停止新增构造，不自动删除文件。

结果根目录：`results/uq_component_ablations_20261014/`。主表与旧 sealed 结果均保持原样。
