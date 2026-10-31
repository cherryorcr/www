# CoTAG-Q 实验结果与论文叙事整理

2026-10-22，依据最新讨论整理。本文件更新此前四项证据修改方案的展示建议。本轮只核对和整理已有结果，没有新增模型训练，没有修改论文正文或推送 Overleaf。

## 1. 建议采用的四项处理

| 问题 | 处理方式 |
|---|---|
| 共享 Q 与独立优化 | 只用同一受控实验的性能与资源做表。两个独立 Q 和直接优化 X 分开，不与主表成绩拼接 |
| TAGSAM | 全部称为共同协议下的适配。新增原生核心适配仅覆盖 Cora；扩展后再统一更新主表 |
| 时间增量 | 证明出版年份流的适用性。正文可以重点讨论 CaT，但已有 GECC 结果保留并提供明确入口 |
| 资源 | 正文重点报告凝聚记忆与原始图的下游训练成本。内部构造优化另列，跨方法构造成本放补充材料 |

论文主线建议表述为：**在历史训练数据访问受限的 TAG 数据流中，构造同时包含数值特征与可读文本的紧凑训练记忆，验证其训练效用、跨阶段适用性和后续模型训练成本。**

共享 Q 是协调来源和减少分配参数的机制。方法效用由学习 Q、分布保持、直接特征优化对照及匹配基线共同支撑，不把全部贡献押在共享本身的精度提升上。

## 2. 共享 Q 的同批次性能与资源表

### 2.1 已有受控实验，可以放 Results

此前确实有共享 Q 与两个独立 Q 的受控实验，匹配初始化、损失目标、特征修正及更新预算。本次核对 18 条运行的原始 JSON 与登记哈希，重算最终性能、全流构造峰值及构造总时间。因此，不需要从主表取性能、再从另一资源批次取显存。

| 数据集 | 分配方式 | Macro-F1，均值 ± SD | Accuracy，均值 ± SD | 构造峰值 GiB | 构造时间 s |
|---|---|---:|---:|---:|---:|
| ArXiv | 两个独立 Q | 53.20 ± 0.12 | 70.96 ± 0.63 | 15.986 | 292.48 |
| ArXiv | 共享 Q | 53.24 ± 0.09 | 70.71 ± 0.40 | 12.083 | 241.30 |
| SportsFit | 两个独立 Q | 80.70 ± 0.66 | 85.87 ± 0.53 | 32.595 | 226.74 |
| SportsFit | 共享 Q | 80.33 ± 0.34 | 85.32 ± 0.72 | 24.724 | 177.91 |
| BookHis | 两个独立 Q | 48.47 ± 1.18 | 81.21 ± 0.48 | 3.262 | 81.60 |
| BookHis | 共享 Q | 46.55 ± 0.93 | 80.87 ± 0.12 | 2.766 | 66.86 |

建议表题为 **Prediction quality and construction cost of shared and separate assignments.** 正文可以保留 F1、峰值与时间，Accuracy、资源 SD 和逐种子结果放补充材料。

表注说明：一种类别顺序、三个构造种子，每份记忆三个 student 训练种子；同一记忆的训练分数先平均，再计算三个构造均值的样本标准差。每次构造 30 次更新，student 训练 2,048 次更新。全流峰值取各阶段构造峰值的最大值，时间为各阶段之和。

**ArXiv 是存档三阶段类别协议，SportsFit 和 BookHis 为五阶段。** 这是此前稠密分配实现的受控结果，不是最新分块实现，也不是 ArXiv base+4 的新实验。历史字段名为 `sealed_test`，现在这些分数已经查看过，不称为新盲测。

| 数据集 | 共享减独立 Q 的 F1 百分点 | 配对 95% 区间 | 构造峰值下降 | 构造时间下降 |
|---|---:|---|---:|---:|
| ArXiv | +0.04 | [-0.43, 0.50] | 24.42% | 17.50% |
| SportsFit | -0.37 | [-1.21, 0.46] | 24.15% | 21.54% |
| BookHis | -1.92 | [-5.02, 1.17] | 15.19% | 18.06% |

可以说共享分配减少了重复的支持参数与构造成本，性能体现数据集相关的取舍。ArXiv 和 SportsFit 的平均 F1 差较小，BookHis 的下降保留在表中。区间跨零不是精度等效证明。参数减半仅指两套分配 logits 合为一套，不指全部模型参数。时间为这组受控运行的观测值，不外推为部署加速保证。

正文可用：

> Sharing the support matrix reduces peak construction memory by 15.2% to 24.4% relative to separate feature and text assignments. Table X reports the corresponding prediction quality and construction time under the same controlled setting. The compact assignment incurs dataset-dependent changes in Macro-F1.

### 2.2 与直接优化 X 的实验分开

两个独立 Q 仍通过支持关系构造特征和文本。直接优化 X 保留文本 Q，但取消 Q 池化特征参数化、特征信赖域及 X anchor。它们是不同干预，不能合成一个“独立优化”组。

直接 X 对照中，本方法在 SportsFit 的 GCN 和 GraphSAGE 上分别高 5.03 和 5.26 个 F1 百分点，配对区间分别为 [4.48, 5.57]、[3.93, 6.58]。ArXiv GCN 与 BookHis 的区间跨零。因此更适合论证有限预算下支持池化与约束组合的使用价值，不能单独归因于共享 Q，也不能解释为优于所有充分调参的自由特征优化器。

消融段建议依次展示学习 Q、DM、直接 X，然后用本节表格解释共享表示的成本。GM 写成通过代理目标提供反馈的辅助项，去掉“数值相近但仍然必需”的结论，保留 no-GM 结果。

## 3. TAGSAM 的覆盖范围与主表更新

| 实验层次 | 数据集覆盖 | 已有证据 | 尚未补齐 |
|---|---|---|---|
| 原 bounded 适配 | ArXiv、SportsFit、BookHis、Cora、Citeseer | 原共同协议参考性能 | 恢复更多原生模块后的性能 |
| 旧适配器完整阶段链预算检查 | ArXiv、SportsFit、BookHis | 增加该适配器更新预算的效果 | 原生完整 inner epochs 的竞争力 |
| 保留原生核心的完整阶段链适配 | **仅 Cora** | 原生构造及 teacher 预算影响 | 更多数据集及构造重复 |

新增适配恢复 BERT CLS、原生图文模型与残差投影、weighted BCE、重采样、SGD/StepLR 和完整合成数据 inner epochs。历史访问、固定凝聚率、文本长度、冻结记忆、部分图计算和共同 student 等仍是适配条件。统一称为 **TAGSAM adaptation**，补充材料说明恢复哪些组件，不称为官方端到端复现。

| Cora 构造设置 | Macro-F1 | Accuracy |
|---|---:|---:|
| 原 bounded 适配 | 60.55 ± 3.22 | 67.28 ± 1.49 |
| 原生核心，teacher 3 epochs | 41.50 ± 2.05 | 53.51 ± 1.69 |
| 原生核心，teacher 2,048 updates | 63.80 ± 0.43 | 73.19 ± 0.56 |

每个设置只有一份构造记忆，SD 来自三个 student 种子。teacher 预算增加后的改善说明适配训练条件重要，但不说明已恢复官方论文性能。官方 5,000 × 15 的 15 是完整合成数据 epochs；大图旧适配器扫描的 15 是 minibatch updates，不能称为相同预算。

**建议扩展后再更新主表，优先级高于继续增加 inspired 行。** 如果只是报告适配敏感性，Cora 结果放附录即可。如果要强化与最近 TAG 方法的比较，至少再覆盖 Citeseer 和一个代表性大图，可优先 BookHis；最终同一冻结版本覆盖全部主表数据集最清楚。

后续实验顺序建议如下，本轮未启动：

1. 固定改进适配器和预算规则，依据代码审计与开发集确定设置，不按各测试集挑选更好版本。保留已经观察过的两种 teacher 预算。
2. 对 Citeseer 和一个大图先检查单阶段资源，再完成从阶段 0 开始的完整记忆链。单末阶段诊断不足以替换主表。
3. 冻结版本后补三个构造种子，每份记忆三个 student 种子，并与本方法使用相同类别顺序、数据访问、记忆率和推断范围。资源允许时扩展剩余大图。
4. 主表只替换实际完成的匹配条目，说明重复次数。不能用旧适配器填未跑数据集，却把整行重命名为新增原生核心适配；覆盖不完整时单列小表。

ArXiv 三阶段存档与 base+4 保持分开。若决定统一主表为 base+4，各主比较方法均须使用该协议确认，不能直接改阶段标签。

## 4. 时间增量的比较范围

正文可以将研究问题限定为“能否在出版年份增量下构造有效记忆，并与持续图凝聚参照比较”，重点讨论 CaT。**如果正文只放 CaT，补充材料保留完整 GECC 结果，正文明确引用。** 不能将省略后的表解释成全部时间流方法的排名。

优先推荐正文保留以下紧凑表，三方法乘三数据集比两行表只多一行。

| 方法 | ArXiv | ACM | DBLP |
|---|---:|---:|---:|
| CaT 适配 | 33.66 ± 1.18 | 86.62 ± 0.75 | 80.74 ± 1.06 |
| GECC 适配 | 53.07 ± 0.32 | 88.38 ± 0.15 | 82.24 ± 0.58 |
| CoTAG-Q | 53.83 ± 0.29 | 87.87 ± 0.37 | 81.50 ± 0.56 |

指标为第五阶段、已到达图推断、累计测试节点的 Macro-F1，均值 ± 构造 SD。三个构造种子，每份记忆三个 student 种子。类别可在后续年份再次出现，与类别增量主表分开。

| 数据集 | 本方法减 CaT 的 F1 百分点 | 配对 95% 区间 | 本方法减 GECC 的 F1 百分点 |
|---|---:|---|---:|
| ArXiv | +20.16 | [16.96, 23.37] | +0.75 |
| ACM | +1.24 | [-0.65, 3.14] | -0.51 |
| DBLP | +0.76 | [-3.27, 4.79] | -0.74 |

差值由未四舍五入的分数计算。可说三图最终平均 F1 高于 CaT，不能说三图都显著优于 CaT。GECC 在 ACM、DBLP 均值更高；当前快照和阶段平均的完整结果保留在补充材料。

正文可用：

> Chronological evaluation follows publication years and allows categories to recur across five stages. CoTAG-Q achieves final Macro-F1 scores of 53.83, 87.87, and 81.50 on ArXiv, ACM, and DBLP. The results extend the evaluation from new-category arrival to chronological document arrival, with higher final mean scores than the CaT adaptation on all three streams. Additional comparisons and stage-wise results are provided in Appendix X.

若完整三行表放正文，末句改为直接说明 GECC 的数据集差异。时间流支撑适用性，不作为普遍领先的证据。ACM/DBLP 是整理的文献子集，三图都是学术网络；边按已到达端点过滤，不是带边事件时间的真实平台日志。数值求解修复在设置中交代一次即可。

## 5. 资源贡献怎样呈现

### 5.1 正文重点放下游训练

可以不在正文做跨凝聚算法的构造显存排名。重点比较同一 student 和训练预算下，凝聚记忆与全部合法原始训练节点的训练资源。

| 数据集 | 记忆 / 原始训练节点 | 记忆 / 原图峰值 MiB | 显存下降 | 记忆 / 原图训练时间 s | 时间下降 |
|---|---:|---:|---:|---:|---:|
| ArXiv | 5,927 / 118,539 | 100.69 / 1,614.62 | 93.76% | 7.75 / 63.61 | 87.81% |
| SportsFit | 6,057 / 121,137 | 102.18 / 1,656.25 | 93.83% | 7.73 / 66.64 | 88.41% |
| BookHis | 1,454 / 29,086 | 43.14 / 411.22 | 89.51% | 7.42 / 15.11 | 50.91% |

边界为 student 训练，2,048 次更新，batch size 24，两个预热更新后测量，三个训练种子。文本嵌入提前计算，编码器不驻留 GPU，构造和原图推断不计入。GPU 指标为 peak allocated memory。原图参考是最终合法训练池与其诱导边，不是全部无标签节点或无限预算模型。

正文可用：

> Training on the condensed memory reduces peak GPU memory by 89.5% to 93.8% and training time by 50.9% to 88.4% relative to training on all observed training nodes under the same update budget. The resulting memory provides a compact training set for subsequent model updates.

贡献落在“受限访问下构造有训练效用的图文记忆，并降低后续训练成本”。相同大小的其它凝聚记忆也享有这类节省，不将节点数缩小带来的全部收益归因于 Q。原始 student 在本实验中本来即可运行，也不写成解决原始模型 OOM。

### 5.2 凝聚内部优化与其它对照

| 数据集 | 原始实现峰值 GiB | 优化实现峰值 GiB | 下降 | 原始 / 优化 F1 |
|---|---:|---:|---:|---:|
| ArXiv | 11.537 | 5.252 | 54.48% | 51.640 / 51.660 |
| SportsFit | 23.802 | 17.514 | 26.42% | 79.681 / 79.413 |

主要优化来自仅存储合法类别支持块，减少 logits、梯度及优化器状态；构造时移除原图推断缓存仅节省约 0.3 GiB。完整 GM、DM、传播目标和 30 次更新保留。SportsFit 一个构造下降约 1.09 个 F1 百分点，因此不能统一称为精度完全不变。

这属于同方法内部的实现改进，正文短段或补充表足够。不把旧主表分数与优化实现的成本拼接；若优化实现作为提交默认版本，需要在主表协议确认。

同一优化构造器的分阶段对一次全历史构造，峰值分别下降 ArXiv 41.25%、SportsFit 7.09%。SportsFit 初始阶段含全流 90.62% 的合法分配项，最大类块几乎没有变小。全历史一次 30 次更新，增量分别三次和五次、每次 30 次更新，总预算不同；这是峰值与访问方式对照，不是上节的原始图 student 训练。该表放补充分析即可。CPU 加载器仍持有完整源数据，未建立物理流式加载的主机内存收益。

已有共同边界测量显示 CaT、GECC 构造峰值更低。完整质量与成本表保留补充材料，正文不展开排名，无需继续补跑。方法差异大不等于资源不可比较，只是本论文的主资源结论应落在下游训练收益。目前也没有端到端成本更低或复用多少次必然回本的实测结论。

## 6. Results 的组织和统计说法

| 顺序 | 回答的问题 | 正文重点 | 补充内容 |
|---|---|---|---|
| 基线对比 | 记忆有没有训练效用 | 五图共同协议比较；BookHis 配对及旧类效用 | 完整适配条件与逐种子结果 |
| 方法分析 | 支持学习和特征构造有何作用 | 学习 Q、DM、直接 X；共享与独立 Q 的质量成本表 | 文本细分、完整区间、no-GM |
| 跨模型与历史效用 | 记忆是否可迁移，保留多少原始效用 | 固定记忆的 GraphSAGE 对照；full-history training 参照 | 更多阶段及分组指标 |
| 时间增量 | 能否处理按年份到达的内容 | 三个出版年份流，优先完整三方法表 | 当前快照、阶段均值、单来源文本 |
| 资源 | 后续训练省多少，构造优化省多少 | 记忆与原图训练表，内部优化短段 | 全历史构造、跨算法构造成本、CPU |

原 RQ 编号可以保留，时间流作为扩展实验，不必增加额外命名。统计用常见表述 **three construction seeds, with three student training seeds per memory**；有多个类别顺序时写 **three class orders**。说明 mean ± sample standard deviation 的计算层级。配对区间以构造均值为独立单位，同一记忆的三个训练种子不算三个独立构造，不再用 cell 作为正文统计单位。

摘要和 Introduction 保持历史访问限制、数值特征与可读文本共同构造、训练效用这条主线。资源用一句点出后续训练节省；合法块作为可扩展实现，不新增一个由稠密 Q 自身引入的普遍性 challenge。

当前新增实验的首要项是 TAGSAM 改进适配的跨数据集完整链及构造重复。其余已有证据可以先整理，无需重复训练。另两项条件性确认是统一 ArXiv base+4 和采用最新分块实现，取决于最终提交协议与版本，不能用开发诊断直接替换历史主结果。

## 7. 数据入口

- [共享与独立 Q 的同批次汇总](/D:/Aprojects/continue_tag_condensation/results/cotag_paper_presentation_20261022/shared_q_matched_summary.csv)，[配对变化](/D:/Aprojects/continue_tag_condensation/results/cotag_paper_presentation_20261022/shared_q_matched_pairs.csv)。
- [时间流最终阶段表](/D:/Aprojects/continue_tag_condensation/results/cotag_paper_presentation_20261022/temporal_final_summary.csv)，[CaT 与 GECC 的配对区间](/D:/Aprojects/continue_tag_condensation/results/cotag_paper_presentation_20261022/temporal_final_pairs.csv)。
- [表格整理脚本](/D:/Aprojects/continue_tag_condensation/tools/summarize_cotag_paper_evidence_20261022.py)，[来源哈希与核验范围](/D:/Aprojects/continue_tag_condensation/results/cotag_paper_presentation_20261022/verification.json)。
- [TAGSAM 报告](/D:/Aprojects/continue_tag_condensation/docs/current/COTAG_TAGSAM_FIDELITY_EVIDENCE_ZH_2026-10-21.md)，[机制证据](/D:/Aprojects/continue_tag_condensation/docs/current/COTAG_THREE_REVIEW_CONCERNS_EVIDENCE_ZH_2026-10-22.md)。
- [下游训练资源](/D:/Aprojects/continue_tag_condensation/results/cotag_resource_comparison_20261022_support/student_resources.csv)，[配对节省比例](/D:/Aprojects/continue_tag_condensation/results/cotag_resource_comparison_20261022_support/student_reductions.csv)。
- [完整资源与四项证据](/D:/Aprojects/continue_tag_condensation/docs/current/COTAG_FOUR_CONCERNS_AND_RESOURCE_EVIDENCE_ZH_2026-10-22.md)，[内部显存优化](/D:/Aprojects/continue_tag_condensation/docs/current/COTAG_MEMORY_OPTIMIZATION_RESULTS_ZH_2026-10-22.md)。
