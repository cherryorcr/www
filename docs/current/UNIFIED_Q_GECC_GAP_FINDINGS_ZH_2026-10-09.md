# GECC 差距诊断与两条改进路线：完整开发实验报告

日期：2026-10-09。

**结论：本轮两条候选改动均不升级。继续保留原 Edge-free In-loop CaT-Q 为研究对照与主方法候选，不能据本轮结果宣称稳定超过 GECC。** 本轮得到的关键进展是明确了当前 GM 代理对类内图文配对的不可辨识性，以及“拟合统计量更好”不保证下游更好的边界。

## 1. 实际完成范围

| 阶段 | 完成内容 | 状态 |
|---|---|---|
| P0 | 三图五阶段表示诊断；固定原版/GECC 的 X，分别附着共同 nearest1/nearest5 | 6/6 cell 完成 |
| P1 | 仅把 Q 的 feature pooling 从 raw-X 改为已有一跳消息 H | 6/6 cell 完成，不升级 |
| P2 | 正确配对、边际统计、类内打乱 reference 配对三个目标 | 18/18 cell 完成，不升级 |
| 审计 | 48 个新 memory/文本 sidecar 与 Student 输入的哈希、预算、前缀、文本来源核对 | 通过 |
| 测试 | 125 项相关回归测试，加 1 项 GM 配对置换性质测试 | 126 项通过 |
| P3 | 新顺序、新 construction seeds 的独立大矩阵 | 未执行：候选没有通过预先登记的开发门槛 |
| 后续协议 | 预算曲线、混合到达、长链、full-raw 参考 | 本轮未执行，不算已完成 |

总计 30 个开发 cell、150 次逐阶段构造/辅助重放调用、288 次新增 Student 训练；另复用 108 次相同协议的既有 Student 训练结果。初次严格 bitwise 重放检查失败的日志保留，但不计入完成 cell。

本轮仅评价最终 memory。没有逐阶段分类结果，因此不提供虚构的 stage-average、worst-stage 或遗忘曲线。

## 2. 冻结协议与可比性

- 数据集：WikiCS、SportsFit、BookHis。开发顺序分别为 9201、9201、9202，均为之前已用顺序。
- construction seeds：271108、271109。每个 memory 共用 GCN/SGC 两个架构，每架构 Student seeds 150/151/152，2048 steps、batch 24。
- 五阶段 fixed-schedule；最终 memory rows 分别为 410、6057、1454。同一 cell 中所有方法的阶段预算、类别顺序、划分哈希一致。
- 构造接口只接受当前 delta、自己的历史 synthetic prefix；旧前缀冻结，无合成非自环边，文本上限 320 字符，MiniLM 冻结。
- 推理使用同一个完整 transductive 图和类别 prompt。数据是 report-dev，不是 sealed test，也不是严格未见类别 zero-shot。
- 数据加载器还为评价加载完整图；“不回读历史 raw”是构造接口级约束和代码审计，不是操作系统级数据隔离。
- 均值先在同一构造 cell 内平均三个 Student seeds，再跨两个 construction seeds 汇总。完整表的 std 是构造 cell 间标准差，不把六次 Student 训练当成六个独立构造重复。
- 只有一个开发顺序，不能据此作强显著性或不劣证明。此前两顺序确认表仍有效，本表不覆盖它；同名原版数值不同是汇总范围不同，不是换了 Student。
- 不扫描新参数，不删除历史文件，不提交仓库。代码默认仍为 `feature_pooling="raw"`、`lambda_joint_distribution=0`。

## 3. 外部基线与内部消融分开

| 名称 | 来源/类别 | 本轮实际做法与适配 |
|---|---|---|
| CaT + nearest5 / `cat_cctp` | 外部论文机制适配 | 当前 delta 的 CaT 类分布匹配，3000 步优化合成 X；按同类 graph-nearest 选择五条原文，固定字符预算拼接、MiniLM 重编码；旧块冻结、edge-free |
| GECC-hard + TAG / `gecc_hard` | 外部论文机制适配 | 当前 delta 上计算 `0.3 A_hat X + 0.9 A_hat^2 X`，按类 hard Lloyd 聚类，20 次重启按 SSE 与大小不均衡罚项选择；nearest5 附着原文，旧中心冻结 |
| 原 In-loop CaT-Q | 本文研究候选，不是外部基线 | 共享 Q 生成 raw-X pooling 与硬一致文本，5% 有界 residual，原 DM/GM/传播目标；不是提前完成 CaT 再微调 |
| 原版/GECC X + common nearest1/5 | 固定内容归因对照 | 精确复用原档案 X、质量、标签、阶段与拓扑；逐阶段只替换文本，随后冻结 |
| Message-space Q | 单因素改动 | 仅切换 Q 聚合的特征空间；Q 初始化、文本、损失权重、Student 等不变 |
| Q + paired/marginal/shuffled MMD | 单个新增目标的机制控制 | 从原版 raw-X pooling 出发，三种目标使用相同特征映射、候选文本、Q 初始化、优化步数和预算 |

前两项不是官方完整原协议复现。尤其 GECC 没有累计历史 raw 重聚类，也不是其完整 fuzzy 版本。构造步数和算法成本不强行相同，Student 预算相同；不能混淆这两种“同预算”。

共同 nearest 控制采用按类批量 float64 cosine 查询，原生适配器为 float32 逐行查询。近似并列会改变文本来源排序，故“共同 nearest5”与“原生 GECC”分开报告，绝不替换更强的原生结果。同策略也不等于两种构造器得到了相同文本。

## 4. 主要结果

下表每格为 **Final Macro-F1 / Accuracy，单位 %**。完整标准差、Old/New F1、配对差、worst cell、positive-cell ratio 见[统一结果表](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/summary.md)。

| 数据集 | Student | 原 In-loop CaT-Q | CaT 适配 | GECC 适配 | Message-space Q | Q + 配对 MMD |
|---|---|---:|---:|---:|---:|---:|
| WikiCS | GCN | 79.77 / 81.45 | 80.97 / 83.43 | 79.32 / 82.07 | 79.57 / 81.61 | 77.32 / 80.53 |
| WikiCS | SGC | 78.22 / 80.93 | 73.64 / 76.48 | 78.65 / 81.78 | 75.64 / 77.89 | 77.79 / 81.14 |
| SportsFit | GCN | 77.87 / 82.62 | 74.12 / 81.53 | 76.35 / 81.63 | 75.05 / 76.92 | 77.35 / 83.71 |
| SportsFit | SGC | 73.33 / 79.44 | 53.16 / 54.35 | 73.93 / 77.77 | 74.56 / 78.49 | 72.68 / 79.66 |
| BookHis | GCN | 41.58 / 78.42 | 19.83 / 66.60 | 39.57 / 77.25 | 38.59 / 76.63 | 40.81 / 77.16 |
| BookHis | SGC | 36.88 / 74.63 | 17.81 / 65.99 | 37.00 / 76.02 | 37.95 / 74.51 | 37.64 / 74.53 |

Message-space Q 相对原版：跨三图等权平均，GCN F1 **-2.00 pp**、SGC **-0.09 pp**；12 个“数据集 × 构造种子 × Student 架构”比较中仅 3 个正向。

配对 MMD 相对原版：GCN F1 **-1.25 pp**、SGC **-0.11 pp**；12 个比较中 4 个正向。SportsFit/GCN 的 Accuracy 上升约 1.09 pp，但 F1 下降约 0.52 pp，不能只选 Accuracy 宣称整体升级。

![Final F1](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/final_f1.png)

![Accuracy](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/accuracy.png)

## 5. P0/P1：均值可达性并不是充分解释

平衡 Q 的 raw-X pooling 在理想精确约束下保持类均值。辅助重放中，raw/message 均值距离与宽松 residual 均值移动上界的平均比值为 WikiCS 2.98、SportsFit 2.40、BookHis 1.35；后两图 residual 触界率约为 100%。

但 raw/message 类均值余弦分别约 0.999856、0.999882、0.995565，很多差异来自尺度。线性均值不可达不能自动推出 F1 会下降。

P1 确实缩小了均值误差，却没有解决分类与非线性分布拟合问题：

| 图 | 原版 -> message pooling：平均类均值误差 | 合成类内方差 / message 类内方差 | Holdout DM 每类误差 |
|---|---:|---:|---:|
| WikiCS | 0.41464 -> 0.06528 | 0.350 -> 0.244 | 0.000779 -> 0.001180 |
| SportsFit | 0.11117 -> 0.01435 | 0.674 -> 0.470 | 0.002829 -> 0.004678 |
| BookHis | 0.05616 -> 0.01868 | 0.376 -> 0.278 | 0.036229 -> 0.037067 |

这些是辅助重放的汇总，原档案本身的只读几何另保存在各 cell 的 JSON。结果与“传播后再软平均会进一步收缩类内变化”的解释一致，但尚未单独证明它是分类退化的唯一原因。不能把原来的可达性推导升级为因果结论。

固定 X、统一 nearest5 后，原版与 GECC 的 SGC F1 差值仍为 -0.90/-2.04/-0.91 pp；仅换文本附着政策不足以消除差距。与此同时，不同文本政策确实改变分数，所以不能反过来说文本完全没用。

## 6. P2：真正配对目标有梯度，但还没有稳定收益

本轮不是把两种边际损失相加后改称“联合”。使用固定随机非线性映射：

\[
\phi_g(h)=\operatorname{norm}(\operatorname{ReLU}(hW_g)),\qquad
\phi_t(e)=\operatorname{norm}(\operatorname{ReLU}(eW_t)).
\]

每类的联合统计和新增目标是：

\[
M_c=\frac1{n_c}\sum_{i:y_i=c}\phi_g(H_i)\phi_t(e_i)^\top,
\qquad L_{joint}=\sum_c\|M_c^{real}-M_c^{syn}\|_F^2.
\]

这是两种随机特征核的乘积核 MMD。维度固定 64、两个随机映射 bank、权重 1；目标只来自当前 delta。通过 `phi_g.T @ phi_t` 计算统计，不构造 raw 节点两两核矩阵。有限特征不保证唯一恢复整个联合分布。

合成侧使用当前 X 和实际硬文本前向，文本反向经 ST 进入 Q。三种控制只改变新增统计目标：正确配对；两个边际均值；reference 类内打乱配对。实际候选池、来源 ID 和 Q 初始化没有被打乱。

所有阶段的抽查点都测到非零新增 text/feature 联合目标到 Q 的梯度；正确配对版本 30 个阶段中 29 个的最终材料化目标低于初值。因此不能将负结果归因于“忘记接梯度”或“完全没优化”。

| 图 | Student | 正确配对 - 边际控制 F1 | 正确配对 - 打乱配对 F1 |
|---|---|---:|---:|
| WikiCS | GCN | -0.55 | -1.41 |
| WikiCS | SGC | -0.32 | -0.48 |
| SportsFit | GCN | -0.78 | +1.28 |
| SportsFit | SGC | +0.15 | -0.78 |
| BookHis | GCN | -0.60 | +0.90 |
| BookHis | SGC | +1.25 | +1.29 |

BookHis/SGC 的两个 seed 都支持正确配对优于这两个控制，是应保留的局部正信号。但它没有在其他图/架构上稳定重现，也没有让候选整体优于原版，不能据此晋升。

![Pairing controls](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/pairing_controls.png)

## 7. 新定位的目标性质：GM 并不识别类内一一对应

当前代理的 pair loss 将所有同类行视为正样本。记 Pi 为保持类别的文本置换，且被置换行的权重相同。全候选计算满足：

\[
L_{mm}(X,\Pi T,Y)=L_{mm}(X,T,Y),\qquad
\nabla_\theta L_{mm}(X,\Pi T,Y)=\nabla_\theta L_{mm}(X,T,Y).
\]

原因：图到文本的相似度矩阵只做列置换，同类正集合和全部候选集合不变；反向文本到图的 anchor 求和也只改变顺序；prompt/margin 项不依赖逐行 payload 配对。当前互斥类别到达下，group-class 权重在同类内恒定，条件成立。

代码级 CPU 诊断覆盖 node-mean/group-class、整块/分块计算：同类置换的 loss 差为 0，参数梯度最大差不超过 4.77e-7；跨类置换 loss 差约 0.21-0.22。见[原始诊断](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/pair_invariance_probe.json)。

**这不是“文本没有梯度”或“图文完全没耦合”。** 文本内容、类级图文分布以及共享 Q 参数化仍然影响训练；但当前 GM 目标不直接要求辨认同类内部哪一个 graph slot 应对应哪一个 text payload。因此，单凭共享 Q 和非零梯度，不能证明保留了对下游有用的节点级对应关系。

实际 Student 每步只抽 24 行，全局同类置换会改变某一小批中的文本样本，因此不能声称其训练轨迹或 F1 严格不变。诊断中小批 loss 确实变化。严格不变性针对当前 full-candidate GM 代理，不是对所有训练过程的笼统断言。

这个性质为此前换成 multi-positive Prompt Student 后，文本精细处理的相对收益缩小提供了一个可检验解释；本轮没有把跨历史协议的变化直接当成已证明的因果结论。

## 8. 成本、数值稳定性与限制

- 构造计时和 peak allocated GPU memory 都已保存到 summary.json。计时包含只读诊断，且两卡并发，不是独占硬件端到端效率 benchmark，不能据此主张新增目标更快。
- P0 辅助重放不总能逐 bit 复现旧 Q/X/文本。相同初始原型下也出现微小前向差异和硬选择变化；诊断回调在 CPU 测试中不改变结果。具体 GPU 核级原因尚未完全定位，不把它伪装成已经修复的 bug。
- 固定内容对照使用的是精确旧档案 X，而不是这些重放 X。来源、重编码、旧前缀和输入哈希均有单独检查。
- 共同 nearest 的近似并列现象进一步说明硬选择对数值精度敏感。原生基线成绩没有被较弱的共同 nearest 版本替换。
- 没有封闭 test 结论、没有本轮跨独立新顺序结论、没有“再也不能优化”的证明。

## 9. 对主方法与论文的决定

1. 原 In-loop CaT-Q 仍是本轮整体更稳妥的研究版本。它具有竞争力，但此前多顺序的 GECC/SGC 差距仍需保留，不能改写成稳定 SOTA。
2. 不升级 message-space pooling 或新增配对 MMD，不按数据集切换版本；两者保留为可复现消融。
3. P3 条件未满足，不在退化候选上继续扩大到 108 条链。原版是否开展更完整独立确认与 setting 证据，是下一轮单独决策，本轮不冒充已完成。
4. 当前可以主张可审计的共享参数化和真实文本材料化；“节点级配对保真带来稳定收益”仍未被充分支持。不能靠组件数量或新增梯度项本身论证 WWW 级创新。
5. 下一步优先检查代理目标与实际 readout 的一致性，包括 full-candidate/group-class 代理与小批均匀采样的区别，以及类内覆盖收缩和硬选择稳定性。若要主张节点级图文保真，需要另外预注册真正使用这种对应关系的评价，所有方法一同测试，并保留本轮分类主表；不能为了获得好看的结论换掉当前 Student。
6. 预算曲线、旧类重新到达等仍有价值，但这些是尚未执行的 setting 证据，不能先用叙述替代实证。当前证据不足以给出可靠 WWW 录用概率。

## 10. 文件与复现入口

- [完整表及配对统计](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/summary.md)
- [机器可读结果、梯度/成本诊断与开发门槛](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/summary.json)
- [48 个归档审计](D:/Aprojects/continue_tag_condensation/results/uq_gecc_gap_20261009/archive_audit.json)
- [执行登记与条件门](D:/Aprojects/continue_tag_condensation/docs/current/UNIFIED_Q_GECC_GAP_EXECUTION_ZH_2026-10-09.md)
- [可选 pooling 与联合目标接入](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py)
- [乘积核配对统计](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_pair_distribution.py)
- [实验 runner](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_unified_q_gecc_gap_20261009.py)
- [置换不变性诊断代码](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/probe_uq_pair_invariance_20261009.py)

所有新 memory/sidecar 在服务器独立实验目录保留；本地保留汇总、完整日志、来源哈希和新归档副本。下载后的 48 份归档均与服务器审计记录的 SHA-256 一致；三张结果图已经过目视检查。此前结果与失败试验均未删除。
