# CoTAG-Q 四项证据对应的论文修改方案

2026-10-22。本文是早期修改建议稿，不是已经改动的论文正文。按用户最新讨论调整的展示方式，以[实验结果与论文叙事整理](/D:/Aprojects/continue_tag_condensation/docs/paper/COTAG_RESULTS_NARRATIVE_ZH_2026-10-22.md)为准：共享 Q 使用同批次性能成本表，TAGSAM 保持适配定位，时间流保留完整比较入口，正文资源重点转为下游训练。以下原方案保留作参考，各项事实以配套实验报告为准。

## 1. 主线与两个挑战

论文要回答的具体问题是：**链接内容持续到达、模型更新只能访问近期训练数据时，如何构造保留类别训练效用和可读文本的紧凑记忆？** 类别增量对应新内容类别加入，出版年份流对应已有类别中的文献持续增加。

建议保留两个挑战，不把实现显存优化另包装成一个与问题无关的理论挑战。

**挑战一：历史训练依据不再可用，紧凑记忆需要承担跨阶段训练作用。** 旧节点、文本及训练关系不能重新进入构造目标，后续构造只能使用当前增量和已有记忆。困难是怎样保留历史类别效用并吸收新内容。冻结旧记忆是本文采用的更新策略，不能说它由历史访问限制逻辑推出，也不能声称其它持续图工作从未讨论历史访问或遗忘。

**挑战二：连续特征和离散文本需要共同构造，同时控制支持优化的开销。** 数值特征可以连续调整，可读文本来自离散来源。单独优化两者不直接保证采用一致的来源依据；简单枚举全部来源与槽位又会带来很大的分配及优化状态。本文用学习到的支持关系协调二者，用受限特征修正提供自由度，并利用已知类别约束只存储合法支持块。共享支持是实现协调的一种设计，不是所有解决方案必须使用的形式。

资源在这里是协调构造需要满足的实际约束。稠密 Q 存储是本文实现的选择，不将它引入的冗余描述成所有现有方法的共同缺陷；类别分块的细节留到方法与资源分析。

可采用的英文衔接段：

> Two challenges arise when condensed data serves as training memory for an evolving text-attributed graph. First, later construction stages must preserve useful information about earlier categories using the memory retained from previous increments. Second, numerical feature construction and readable text selection must be coordinated despite their different optimization properties. A support representation over source nodes and memory items provides this coordination, but its size can make construction expensive even when the resulting memory is small.

接方法设计的段落：

> CoTAG-Q learns a support matrix that guides feature pooling and source text selection. A bounded feature correction gives the pooled representation additional flexibility. Each stage constructs new memory items from the current increment and carries the historical memory forward for subsequent training. We implement the class constraints through separate legal support blocks, reducing the storage of assignment parameters and their optimization states while preserving the construction objective.

这些段落解释“困难、设计、作用”的关系。精度收益放到实验中验证，不把训练目标写成理论性能保证。

## 2. 贡献应该分哪几层

1. **问题与评价范围。** 在历史训练数据访问受限的条件下研究持续 TAG 凝聚，提供类别增量和出版年份增量两种评价。限定到经过验证的链接文档和商品网络，不声称完整在线部署。
2. **构造机制。** 用可学习支持关系共同组织数值特征与可读文本，结合受限修正和分布保持。学习 Q 的消融支持支持关系需要更新，直接 X 对照进一步支持有限预算下特征参数化的使用价值。
3. **实证效用与可扩展实现。** 以强基线配对结果、旧类效用、跨 student 迁移及分别测量的构造和训练成本说明使用价值。合法支持分块可作为实现贡献的一部分，不单独宣称一种新的学习理论。

贡献条目示例：

> We study continual condensation of text-attributed graphs under restricted access to historical training data, covering both category arrival and chronological document arrival.
>
> We develop a learned support representation that coordinates feature construction with readable text selection. Controlled comparisons identify the value of learning the support and organizing feature optimization around the available data.
>
> We evaluate training utility, architectural transfer, and resource use separately. A class-block implementation reduces the construction memory of CoTAG-Q, while the resulting compact memories substantially reduce the resources needed for downstream training.

第三条需要与具体数据表相连。不要写成“解决大图模型原本无法训练的问题”：目前原始训练池上的 student 本身可以运行，实验测得的是资源下降，不是从 OOM 到可运行的阈值证明。

## 3. 机制证据怎样写

先讲学习支持与 DM，再讲直接优化特征，最后说明共享支持的成本价值。三种对照回答不同问题。

> Fixing the support matrix or removing distribution matching reduces the quality of the condensed memory. We further compare the proposed feature construction with independently optimized features under the same finite update budget. On SportsFit, the proposed construction improves Macro-F1 by 5.03 points with GCN and by 5.26 points with GraphSAGE. The paired intervals are [4.48, 5.57] and [3.93, 6.58], respectively. These results support the practical value of organizing feature optimization through the learned support and bounded correction.

表注写清楚 Independent X 的干预：保留学习到的文本 Q 和 GM、DM、传播目标，取消 Q 池化特征参数化、特征信赖域及 X anchor。这不是仅把一个共享 Q 改成两个 Q，也不是自由 X 所有优化设置中的最优结果。

共享与独立分配的成本另用短段：

> A shared support matrix replaces separate feature and text assignments. In the controlled implementation, this reduces the number of assignment logits by half and peak construction memory by 15.2% to 24.4%. The full comparison reports the corresponding accuracy and time, so that the compact parameterization can be assessed as a design tradeoff.

完整表必须保留 BookHis 的精度变化。ArXiv 差值 +0.04、SportsFit -0.37、BookHis -1.92 个 F1 百分点，不能统一说为“精度等效”。总模型参数量、相对自由 X 的内存节省也不能从分配 logits 减半推得。

GM 的描述调整为功能说明：

> Gradient matching supplies feedback through the graph and text proxy objective. In the component comparison, removing this term gives similar scores, while the largest observed gains come from learning the support and preserving distributions.

保留 no-GM 表项即可，不再增加一段未经验证的“二阶噪声”解释。此次显存优化仍保留完整 GM。

## 4. TAGSAM 的适配怎样呈现

主表保留共同协议的适配结果，补充材料用一张表对照官方模块、预算单位及已经恢复的模块。不要把仅增加有界适配器预算的结果标作官方完整预算。

> We examine TAGSAM over complete continual construction chains and restore its native construction core on Cora. With the native components, changing the teacher budget from three epochs to 2,048 updates raises common-student Macro-F1 from 41.50 to 63.80, compared with 60.55 for the earlier bounded adapter. This identifies teacher training as an important condition when applying the method to small increments.

必须在表注说明：一份构造记忆、三个 student；官方 15 次内循环单位是完整合成数据 epoch；大图预算扫描中的单位是 minibatch update。大图原生完整 TAGSAM 尚未完成，因此不能写“已经恢复其论文性能且仍显著落后”。Cora 的高保真结果更有说明力，但不是全部 TAG 基线保真度问题的终结。

## 5. 类别增量与时间增量如何组织

两种协议都保留。Introduction 和设置先用自然内容到达建立场景，再说明两种互补变化。主实验保持证据更完整的类别增量比较，随后用独立小节展示出版年份流，不用某一种协议的有利排名替换另一种。

> Class-incremental evaluation isolates the arrival of new categories. Chronological evaluation instead follows publication years and allows categories to recur as documents accumulate. Together, the two settings test different forms of change in linked content collections.

时间流结果段：

> On the five-stage publication-year streams, CoTAG-Q obtains final Macro-F1 scores of 53.83 on ArXiv, 87.87 on ACM, and 81.50 on DBLP. It improves over the GECC adaptation on ArXiv, while GECC obtains higher mean scores on ACM and DBLP. The evaluation establishes that the construction supports chronological content arrival, with relative utility depending on the stream and the available inference context.

主文放三方法乘三数据集的表。附录保留当前快照推断、阶段平均、误差与单来源文本对照。构造访问与推断访问在设置中用一段清楚定义即可。ACM/DBLP 是整理的文献子集，边按已到达端点截取；不称为带真实边事件时间的完整平台日志。

## 6. 资源结果的组织

资源部分依次回答三个问题：

1. 本构造器的优化减少多少显存，并伴随怎样的精度变化？
2. 同一优化构造器分阶段处理与一次全历史构造相比，峰值差多少？
3. 与其他凝聚算法以及原始数据 student 训练相比，成本分别发生在哪个阶段？

主文放一张构造显存柱状图和一张紧凑的训练资源表。CPU 内存、reserved 显存、逐种子耗时、配对区间放补充材料。所有方法的构造 GPU 峰值保留同一测量边界，不省略成本更低的 CaT/GECC。

英文资源段的基础文字：

> Storing only legal class blocks reduces the mean construction peak from 11.54 to 5.25 GiB on ArXiv and from 23.80 to 17.51 GiB on SportsFit. The complete objective, including gradient matching, remains enabled. The corresponding Macro-F1 means change from 51.64 to 51.66 and from 79.68 to 79.41. We report these costs with the scores produced by each implementation.

新增分阶段与全历史构造段：

> With the same class-block constructor and final class quotas, incremental construction reduces peak GPU memory by 41.25% on ArXiv and 7.09% on SportsFit relative to constructing once from all final legal training nodes. The difference reflects the size of the largest active increment. On SportsFit, the base stage contains 90.62% of the legal assignment entries in the entire stream, so staging provides limited additional peak reduction after class blocking.

表注必须说明：全历史参考一次 30 次更新，增量分别三次和五次构造、每次 30 次更新，总优化预算不同。ArXiv 是存档三阶段资源诊断，不把这个数字接在统一 base+4 主表后而省略协议区别。

共同基线对照接一句即可：

> The optimized constructor still uses more peak GPU memory than the CaT and GECC adaptations. We therefore distinguish the reduction within CoTAG-Q from comparisons across algorithms and report prediction quality, GPU memory, CPU memory, and construction time together.

这句话与完整数值表足够交代边界，不需要反复防御。优先给读者看到优化前后 54.5%/26.4% 的实测改善，以及这种改善解决了哪部分冗余。

下游训练段：

> Training on the condensed memory reduces peak allocated GPU memory by 89.5% to 93.8% and training time by 50.9% to 88.4% relative to training on all final legal source nodes under the same update budget. These measurements exclude construction and inference. Similar-sized memories from other condensation methods have similar student training costs, so the role of CoTAG-Q is to produce a useful memory under the specified historical access constraint.

避免混淆两个全历史参考：资源新实验中的 full-history construction 是用全部合法历史训练节点构造一份凝聚记忆；原文 retention 的 full-history training 是直接用全部合法历史训练节点训练 student。二者应采用不同、明确的说法。

## 7. 正文修改顺序

| 位置 | 修改重点 | 主文保留的证据 |
|---|---|---|
| 摘要、Introduction | 具体内容更新问题；两个挑战；支持构造和可扩展实现 | 代表性类别效用，构造峰值改善与下游节省分开 |
| Problem setting | 类别增量与出版年份流；训练访问与推断访问 | 清晰定义，不展开全部超参数 |
| Method | 支持池化、受限修正、文本选择的作用；合法块的表示 | 核心公式与流程，GM 为辅助反馈 |
| Main results | 保留匹配强基线；强调 BookHis 的类别与旧类效用 | 全部数据集 F1/Acc，重复范围明确 |
| Mechanism analysis | 学习 Q、DM、直接 X；共享设计的成本 | 不混用自由 X 与独立 Q 对照 |
| Chronological evaluation | 内容自然到达的适用性 | 三数据集完整对比，不删更强基线 |
| Resources | 内部实现优化、分阶段对整图、下游训练 | 质量与成本同时报告 |
| Appendix | TAGSAM 模块预算、逐种子区间、原始指标入口 | 保留影响结论的对照与单位 |

建议保留原主表分数，资源开发实验作为补充，不用原实现的精度拼接优化实现的成本。若决定以优化实现作为提交默认版本，需要冻结该版本并在实际主表协议上确认；本轮 ArXiv 资源实验仍是三阶段，不替代五阶段协议结果。
