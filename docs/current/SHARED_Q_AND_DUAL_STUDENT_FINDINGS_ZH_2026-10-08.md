# 共享 Q 归因与双 Student 基线比较

日期：2026-10-08。保持 `uq_dm_edgefree` 构造器不变，不按新结果选参数，不删除历史文件。本轮全部评估使用 report-dev，未使用封存 test。

## 1. 两个实验回答不同问题

| 实验 | 固定内容 | 变化内容 | 可以回答 | 不能回答 |
|---|---|---|---|---|
| A：共享 Q 归因 | 当前三图、三个类别顺序、预算轨迹、损失、初始化规则、步数、Student | 特征与文本共用 Q，或各自学习 Q_X/Q_T | 在当前有限优化预算下，参数共享是否有帮助 | 共享 Q 必然优于所有解耦算法 |
| B：交接顺序的双 Student | 每个方法同一份 X、文本、拓扑、来源；各 Student 内相同训练预算 | legacy InfoNCE 或 Prompt Student | 方法的相对效果是否依赖训练协议 | 跨架构泛化、严格未见类 zero-shot |

实验 B 中，主方法只按交接包顺序重新构造一次。不为了 legacy Student 改构造 proxy，不重新优化 memory。外部 memory 原样复用；PUMA 没有 checkpoint，明确标成重建。不同 Student 的绝对分数不能作为凝聚算法的净收益。

## 2. 交接包能否复用

输入目录：`D:/Aprojects/LARGE_GRAPH_BASELINE_HANDOFF_20261008/large_graph_baseline_20261008`。178 个文件的 SHA256 校验通过。

可复用的是通过来源、内容 hash、划分和前缀检查的 memory，不是直接把旧表里的数字拼进当前主表。两套类别顺序和预算轨迹不同：

| 数据集 | 当前主表 rows | 交接表 rows |
|---|---|---|
| WikiCS | 247, 324, 356, 379, 410 | 285, 330, 368, 391, 409 |
| SportsFit | 3817, 4486, 4812, 5305, 6057 | 4491, 4965, 5430, 5802, 6056 |
| BookHis | 579, 592, 601, 1429, 1454 | 1395, 1423, 1436, 1445, 1454 |

特别注意：最终 rows 接近，并不代表整条持续链相同。BookHis 的大类到达时间明显不同，不能跨表相减。

- 五个核心基线的 memory 存在，可核验后复用。
- WikiCS、BookHis 的 TAGSAM 顺序与交接核心表一致；SportsFit、arXiv 的 TAGSAM 不一致，未硬塞进同协议表。
- PUMA 的原始运行只有指标和 hash，不能由 hash 恢复 memory。因此重建并单列构造来源。
- 旧 checkpoint 通常不含 `labels`。WikiCS/BookHis 真实 coreset 通过已存原始 ID 与 X 一致性恢复；SportsFit 缺少这些 ID，14 个块可通过当前 delta 中完全一致的 X、文本 embedding 和字符串恢复。base random 含跨类内容重复，因此重放原无标签 selector，并确认 seed、全部特征、文本、mass/depth/stage 完全一致，再采用重建来源的标签，未任意指定重复行的类别。CaT/GECC 通过已存 `row_classes` 和文本来源恢复；TAGSAM 使用其原构造中排序的 dispersion 类别配额，并核对 base allocation。均不使用 report/test 标签推断。
- SportsFit、arXiv 的交接 `cat_mmd_tag_cctp_top5` 实际构造模式为 `text_center`，不能按名称视为 nearest-5 CCTP。WikiCS/BookHis 的该项是正确的 nearest-5。本轮单独修复 SportsFit：固定原 CaT X、标签、mass、depth、stage 和拓扑，重新附着 nearest-5，再评估两种 Student；错误实现作为历史内部对照单列。arXiv 仅完成审计，未纳入这轮三图比较。
- 外部 coreset 保留原文本和合成拓扑；文本长度可能远超 320 字符。此表是相同 **rows**，不是相同文本字节数、总存储或边数，成本表另行公开。
- 构造函数只接收合法当前 delta；评价进程仍持有完整原图做 transductive 推理。这是函数接口与来源审计，不是 OS 隔离证明。

原始记录保持不变；初始复用清单位于 [handoff_audit](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/handoff_audit/audit.json)，补充实际文本模式及 selector 来源的审计位于 [handoff_audit_v2](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/handoff_audit_v2/audit.json)。SportsFit 在旧来源检查处停止的结果保留，正式补跑使用独立的 `handoff_sportsfit_verified`，不是覆盖旧协议。

## 3. 两种 Student 的准确含义

| 项目 | Legacy Student | Prompt Student |
|---|---|---|
| 图编码器 | 两层 GCN，hidden/out=384 | 两层 GCN，hidden/out=384 |
| 文本编码 | 冻结 MiniLM + 可训练线性投影 | 相同类型 |
| 配对训练 | 行对行双向 InfoNCE | 同类多正样本图文 InfoNCE |
| 合成标签参与 Student 训练 | 否 | 是，用于多正样本、prompt CE 和 margin |
| batch | 16 | 24 |
| 每阶段 steps | 2048 | 2048 |
| seeds | 150, 151, 152 | 150, 151, 152 |
| 邻接处理 | 保留输入方向/重复边后加自环、归一化 | 对称化、去重、自环、归一化 |
| 预测 | 原图表示与类别 prompt 的余弦相似度 | 同类预测接口 |

两者每阶段都重新训练，不是保留模型状态的 direct continual learning。它们都使用凝聚文本训练；推理时读取原图 X/边和类别 prompt，不是把每个测试节点原文直接交给文本分类器。

训练监督、batch 与邻接算子一起变化，因此不能把差异归因于某一个 loss。它们也不是 GCN 对 GraphSAGE/GAT 的跨架构实验。所谓 prompt inference 指不在评估阶段拟合新的分类头，不意味着这些类别从未在构造或训练中出现。

## 4. 共享 Q 的直接实验

实验 A 完成三图 × 三顺序，每条链五阶段；每份 memory 三个 Student seeds。共 90 个原生评估与 18 个最终统一文本评估，即 324 次 Student 训练。只有一个 construction seed，顺序为 9101/9102/9103，不能写成三个独立构造种子。

独立 Q 对照为：

$$
X = D_{Q_X}^{-1}Q_X^\top X_{raw}+R,\qquad
T^{soft}=\operatorname{Norm}(D_{Q_T}^{-1}Q_T^\top E),\qquad
T_j^{hard}=E_{\arg\max_i Q_{T,ij}}.
$$

两个 Q 从同一初始化出发，分别接受 Adam 更新，均遵守类内 mask 和容量约束。Q-anchor 取两分支 anchor 的平均值，参数绑定时目标退化为共享版本。共同的多模态 GM 仍连接 X 与文本，因此这不是完全取消图文关联。独立版本分配参数数目为两倍，优化几何也随之改变，不能宣称是关于表示能力的普适证明。

| 数据集 | 共享 Q F1 | 独立 Q F1 | 共享减独立 F1 | 95% 配对顺序区间 | 共享 Q Acc | 独立 Q Acc |
|---|---:|---:|---:|---|---:|---:|
| WikiCS | 80.06 | 80.75 | -0.70 | [-1.74, +0.34] | 82.94 | 83.24 |
| SportsFit | 78.64 | 78.24 | +0.40 | [-0.97, +1.77] | 84.49 | 84.04 |
| BookHis | 42.00 | 41.92 | +0.08 | [-9.07, +9.23] | 80.17 | 80.28 |

正收益顺序分别为 0/3、2/3、1/3。将最终文本统一替换为归档 CaT payload 后，F1 差值分别为 -0.75、+0.28、+0.68 pp，也未建立稳定优势。该操作发生在构造结束后，仅用于诊断，不能代替完整闭环优化。

**结论：目前未证明共享 Q 带来稳健的准确率收益。** 区间跨零也不证明两者等效。不能从这组结果写“共享 Q 是全部提升的核心原因”，也不能反过来断言所有耦合都无用。

共享版本可确认的实现优势是少一套分配参数。当前观察到的平均构造耗时为 WikiCS 19.22 vs 24.39 秒、SportsFit 157.78 vs 198.33 秒、BookHis 73.22 vs 89.51 秒；属于本批机器负载下的描述性计时，不是独占硬件的严格性能定理。参数共享可以作为紧凑建模选择，但不能替代效果证据。

### 4.1 同一 memory 换旧 Student 后怎样

在 Prompt 结果已知后，追加了全三图、全三顺序的 legacy 最终阶段评估，共 18 份冻结 memory、54 次 Student 训练。没有重新凝聚，也没有只选择有利的顺序。该实验是探索性补充，不冒充事前注册确认。

| 数据集 | 共享减独立 F1 | 95% 配对顺序区间 | 正收益顺序 |
|---|---:|---|---:|
| WikiCS | +8.83 | [-10.29, +27.96] | 3/3 |
| SportsFit | -4.45 | [-8.00, -0.90] | 0/3 |
| BookHis | +1.69 | [+0.20, +3.17] | 3/3 |

共享 Q 效应与 Student 协议、数据集有关：旧 Student 下 BookHis 正向、SportsFit 负向，不能只引用前者。这里进行了多个探索性比较，未做多重检验校正。结果仍不支持“共享 Q 在不同 Student 下普遍提高精度”。

## 5. 基线名称和适配对应

| 原方法/类型 | 本轮名称 | 保留机制 | 适配或限制 |
|---|---|---|---|
| CaT | CaT-MMD + CCTP nearest5 | 类条件随机表示分布匹配，直接梯度优化自由 X | 只构造当前 delta；旧前缀冻结；每行近邻 top-5 原文附着；统一 Student；SportsFit 的错误 text_center 单列并固定 X 修复 |
| GECC | GECC-hard + TAG attachment | 0/1/2-hop 权重 0/0.3/0.9；类内 hard Lloyd，20 次重启，SSE+平衡罚项选解 | sklearn 实现而非官方 Torch kernel；统一配额、访问边界、文本与 Student；不宣称复现累计 raw 演化或 fuzzy 模式 |
| PUMA | PUMA-inspired TAG，重建 | 一次传播、伪标签 teacher、置信过滤、CaT 风格匹配 | 在合法 pool 内划出模拟未标注子集；teacher 和监督设置有改动；不是官方完整流水线 |
| 静态 TAGSAM | TAGSAM-response adapted | 官方固定版本 wBCELoss/response unroll 核心 | dispersion 初始化替代官方 text selector；current-delta 持续适配；统一最终 Student，旧前缀冻结 |
| 通用真实子集 | Random real coreset | 当前 delta 内无逐节点标签的随机选点 | 原文与其文本拓扑保留，不是假造 synthetic noise |
| 通用覆盖子集 | K-center coreset | 联合图文空间覆盖 | 保留所选真实特征、文本和构造拓扑 |
| 通用代表点子集 | Medoid coreset | 当前 delta 联合图文空间聚类代表节点，非类内标签分组 | 保留所选真实特征、文本和构造拓扑 |

这四个论文来源与三个 coreset 不能合称“七篇独立凝聚论文”。共享/独立 Q、no-GM、CCTP 文本变体是内部消融，不算外部基线。

补核 [GECC 官方固定提交](https://github.com/Emory-Melody/GECC/tree/d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f) 的 `graphslim/config.py` 与 `condensation/gecc.py` 后，确认官方同时支持 hard/fuzzy，且默认 `fuzziness=1.0`、20 次重启。因此“不是 fuzzy”本身不足以判定 GECC-hard 是弱基线。真正需要列明的是 kernel、协议、文本和下游接口的改动，不能只靠名称判断保真程度。

## 6. 结果入口与尚未冻结的结论

双 Student 三图五阶段全部完成。全量表、Old/New、阶段平均、最差阶段、存储和计时由脚本汇总：[统一表](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/summary.md)、[结构化结果](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/summary.json)。不与此前不同顺序/预算轨迹的主表混算。

### 6.1 外部论文机制的适配比较

下表每格为最终 Macro-F1 / Accuracy（%，三个 Student seeds 的均值）。这不是三个独立凝聚种子；标准差和条件配对区间见全量表及 JSON。

**Prompt Student**

| 数据集 | 主方法 In-loop CaT-Q | CaT + nearest5 | GECC-hard + TAG | PUMA-inspired + TAG | TAGSAM-response 适配 |
|---|---:|---:|---:|---:|---:|
| WikiCS | 81.45 / 83.90 | 81.53 / 84.19 | 80.58 / 83.56 | 81.22 / 83.70 | 45.38 / 55.95 |
| SportsFit | 79.50 / 85.47 | 74.49 / 87.47 | 80.24 / 89.29 | 78.51 / 89.20 | 原始顺序不兼容 |
| BookHis | 46.33 / 81.85 | 28.98 / 81.12 | 45.15 / 82.54 | 36.38 / 81.48 | 32.73 / 80.67 |

**Legacy InfoNCE Student**

| 数据集 | 主方法 In-loop CaT-Q | CaT + nearest5 | GECC-hard + TAG | PUMA-inspired + TAG | TAGSAM-response 适配 |
|---|---:|---:|---:|---:|---:|
| WikiCS | 51.36 / 55.53 | 52.41 / 57.89 | 51.42 / 53.68 | 47.94 / 49.97 | 36.64 / 45.73 |
| SportsFit | 28.03 / 32.14 | 30.27 / 30.55 | 25.93 / 29.51 | 26.75 / 30.31 | 原始顺序不兼容 |
| BookHis | 15.57 / 20.51 | 16.19 / 27.16 | 16.61 / 27.62 | 15.75 / 26.83 | 18.96 / 40.93 |

Random/K-center/Medoid 也已在两种 Student 下跑完，放在全量表的独立行及下图中；不把这些通用子集方法称作凝聚论文。SportsFit 标准 CaT 数值来自独立 fixed-X nearest-5 修复。原 `text_center` 的 F1/Acc 为 legacy 26.18/48.59、Prompt 75.94/87.93，作为历史文本组件对照单列，不隐藏也不冒充 top-5 nearest。

![三图双 Student Macro-F1](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/handoff_final_f1.png)

![三图双 Student Accuracy](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/handoff_accuracy.png)

### 6.2 客观解读

- WikiCS 的 Prompt F1 与 CaT 基本持平，没有明显领先。
- SportsFit 的 Prompt F1 相对 CaT 提高约 5.02 pp，但低于 GECC 约 0.74 pp；Accuracy 比 CaT 低约 2.00 pp，不能写成所有指标同时提升。
- BookHis 的 Prompt F1 相对 CaT 提高约 17.35 pp，相对 GECC 约 1.18 pp；但 Accuracy 仍低于 GECC。结论是宏平均分类表现和长尾权衡较好，而非全面 SOTA。
- Legacy Student 下，主方法三图 F1 的本次均值均未超过 CaT。不能只展示 Prompt 表并宣称对任意 Student 都有优势。
- 共享 Q 的直接消融也有 Student 依赖；应将共享描述成统一且紧凑的参数化，而不是已经证明不可替代的精度来源。既有 in-loop DM、Q 参数化归因与本轮共享/独立 Q 问题不同。

![共享 Q 效应的 Student 依赖](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/shared_q_student_effect.png)

### 6.3 数值可复现性限制

同一份 WikiCS base K-center memory、同一 seed=150、2048 steps，旧 Student 的重复训练诊断如下。诊断没有替换正式表中的任何分数。

| 执行方式 | 第一次 F1 | 第二次 F1 | 两次模型最大参数差 |
|---|---:|---:|---:|
| CUDA sparse，默认 | 44.58 | 64.74 | 0.1613 |
| CUDA sparse，开启确定性选项 | 49.98 | 41.21 | 0.1493 |
| CUDA dense，数学等价邻接 | 59.58 | 59.58 | 0 |
| CPU sparse | 46.75 | 46.75 | 0 |

同入口无边 GECC 重复结果为 62.16/62.16，参数完全一致。旧评价入口与新 wrapper 对相同预测的指标也一致。这支持将此例的波动定位到 GPU 稀疏训练数值执行路径及优化敏感性，而不是汇总公式错误；尚未逐一证明每个底层算子的责任，也没有证明所有数据集或 Prompt Student 都具有同样的波动。

因此带边 coreset 尤其是旧 Student 的细小排名差异，不能作为强结论；三个 Student seeds 的标准差也没有单独分离同 seed 数值波动。发表前应冻结更稳定的稀疏执行实现，或补预先定义的重复运行统计，再进行封存 test。不能根据哪次重复分数更好来替换结果。

诊断原件：[同入口重复](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/legacy_sparse_determinism_diagnostic.json)、[dense/CPU 对照](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/legacy_operator_repeat_diagnostic.json)。

### 6.4 完成范围与审计

- 共享 Q：9/9 数据集顺序 cell，108 个阶段/文本控制评估记录；另有 18 个 legacy 最终阶段读出。
- 双 Student 基线：240 个阶段评估记录，其中 10 个是错误 `text_center` 的历史内部对照，其余 230 个是本轮主方法和基线比较记录；每个记录三个 Student seeds。
- 上述主结果合计 1098 次 Student 训练，不含 smoke、失败运行和数值诊断。这些不是 1098 个独立构造样本。
- 原始交接文件 178 个 checksum 通过；原五个核心基线的 75 条完整候选 delta ID 数组与新主方法实际阶段 hash 一致，见 [候选节点协议审计](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/source_delta_audit.json)。
- 归档重建核验：90 个共享/独立 Q 原生读出、18 个相同最终文本读出、60 个新构造的 handoff 读出、10 个 fixed-X 修复读出全部一致，见 [归档审计](D:/Aprojects/continue_tag_condensation/results/shared_q_baseline_20261008/archive_audit.json)。外部 memory 原文件 hash 在服务器加载时检查。
- 本轮没有执行 arXiv、跨架构 GAT/GraphSAGE、独立新 construction seeds 或封存 test；没有把这些未完成内容计为已完成，也没有改动主算法来追逐新表结果。

已有当前主协议三顺序 CaT/GECC/PUMA/coreset 对照仍单独保留：[旧主协议基线表](D:/Aprojects/continue_tag_condensation/results/unified_q_matrix_20261008/summary.md)。本轮不是说这些比较以前完全没做过，而是增加经过审计的外部 memory、另一条类别顺序及双 Student 证据。

主方法仍可作为研究主线；目前最可靠的机制证据主要来自 Q 参数化与 in-loop DM 的归因，而不是“文本直通梯度和共享 Q 均已证明必要”。不要为保留复杂度而夸大所有组件作用。发表前仍需补独立构造确认、更接近原方法的完整基线或独立原生参考、真正的跨架构评估、封存 test，以及存储/时间预算的充分披露。

## 7. 可复核代码

- [交接包审计](D:/Aprojects/continue_tag_condensation/tools/audit_large_baseline_handoff_20261008.py)
- [共享/独立分配控制](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/shared_q_controls.py)
- [共享 Q 实验入口](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_shared_q_attribution_20261008.py)
- [双 Student 入口与标签恢复](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_handoff_student_transfer_20261008.py)
- [结果与配对区间汇总](D:/Aprojects/continue_tag_condensation/tools/summarize_shared_q_baseline_20261008.py)
- [阶段块重建审计](D:/Aprojects/continue_tag_condensation/tools/audit_shared_q_archives_20261008.py)

本轮相关单元测试共 39 项通过，涵盖共享/独立 Q、来源恢复、固定 X 修复、跨 Student 同一 memory、完整 delta 一致性、协议检查和汇总。主构造器与 Student 源码未改动。任何同 seed 重建漂移都应独立披露，不并入既有结果充当新 seed。
