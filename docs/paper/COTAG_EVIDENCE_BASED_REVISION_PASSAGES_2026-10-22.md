# CoTAG-Q: Evidence-Based Revision Passages

Prepared on 2026-10-22. These are proposed passages, not edits to the manuscript. Numerical sources and remaining gaps are documented in [the evidence report](D:/Aprojects/continue_tag_condensation/docs/current/COTAG_THREE_REVIEW_CONCERNS_EVIDENCE_ZH_2026-10-22.md).

## 1. Introduction: State the Practical Question

放在问题背景与研究空缺之间。用具体问题建立 Web 相关性，不添加新的部署承诺。

> Linked content collections must update their classification models as new documents and categories arrive. When model updates have access only to recent training records, previously condensed data becomes the training interface to earlier content. For text-attributed graphs, this memory must preserve both information derived from graph relations and readable text that can be consumed by the text encoder. We study how to construct such a memory so that later updates retain useful category information without repeatedly processing the historical training graph.

关于第二个挑战，将现有 Introduction 中 “a continual TAG condenser needs a common support matrix” 改为下面这种论述。它解释设计选择，不把唯一实现强加为问题的必要条件。

> Continuous feature synthesis and discrete text selection must remain coordinated as the memory is optimized. We address this problem with a learned support matrix that guides both feature pooling and the selection of readable source text. A bounded feature correction supplies flexibility while keeping the synthetic representation close to its selected support.

贡献条目可围绕以下三个层次，不另设“共享 Q 精度优势”条目。

> We formulate continual condensation of text-attributed graphs under restricted access to historical training data and evaluate both category arrival and chronological document arrival.
>
> We develop a support-based construction that coordinates graph-informed features with readable text. Learning the support and preserving feature distributions produce useful memories under a finite construction budget.
>
> We characterize the resulting training utility through matched baselines, feature construction controls, architecture transfer, and separate measurements of construction and student training costs.

不要将这里的 “graph-informed” 继续换写成多个新命名术语。方法名称仍沿用正文的 CoTAG-Q。

## 2. Problem Setting: State Access Positively

放到第二部分，取代目前只说实验全部是模拟类别流的句子。具体年份和数据量留到实验设置。

> We evaluate two forms of arrival. Class-incremental streams introduce disjoint groups of categories, while chronological streams order documents by publication year and allow categories to recur. At construction time, the method receives the current training increment and the condensed memory from earlier stages. At inference time, we consider an arrived graph containing all nodes observed so far and a current graph containing only the latest increment. Comparing these inference modes on the same current queries isolates the effect of available graph context.

访问限制属于训练更新问题。不要用“平台永久删除所有历史信息”去解释累计快照推断。图中保留已到达节点之间的边，不引入未来节点；边的事件时间不单独可用。

## 3. Method: Explain the Design, Not Empirical Necessity

放在介绍支持矩阵和特征池化的段落中，配合现有数学定义。

> The support matrix defines how the current nodes contribute to each memory item. Pooling through this matrix restricts the initial synthetic feature to a combination of available source features. The bounded correction then adjusts that representation without discarding its dependence on the support. Updating the support therefore changes the feature itself, as well as the source distribution used for text selection. This parameterization organizes the search around the current data rather than treating every synthetic feature as an unconstrained variable.

GM 的功能说明可以写为：

> Gradient matching supplies feedback from the graph and text proxy objective to the construction variables. Together with distribution and propagation matching, it encourages the new memory items to support the proxy's training objective while retaining information extracted from the current increment.

这里描述实际优化作用，不声称消融已经证明 GM 必需，也不写“二阶噪声造成无 GM 更好”这种未经隔离的原因。

## 4. Main Results: Lead With the Most Supported Benefit

替换主结果开头泛泛的“保存图知识”总结。主表仍完整保留所有可比方法与两个指标。

> The strongest paired gain over GECC occurs on BookHis. CoTAG-Q improves final Macro-F1 from 45.34 to 48.81, a relative improvement of 7.7%, with positive differences in all nine combinations of class order and construction seed. The improvement also appears in old-class F1, which increases from 45.48 to 48.39, and in the average across stages, which increases from 50.43 to 51.82. These results show useful category-level retention throughout the incremental evaluation. Accuracy remains close to GECC, while the relative performance on ArXiv and SportsFit varies by dataset.

旧类识别的证据可以正文加一句：

> Resampling class orders and construction seeds as crossed factors gives exploratory 95% intervals of [1.81, 5.01] points for final Macro-F1 and [1.77, 4.40] points for old-class F1.

这里不写 statistically guaranteed 或所有阶段全面胜出。阶段均值有正增益，Base 阶段的平均差值是负的。

逐类结果更适合接在这段后面作短分析，或作为附录中的一个图段。

> A separate per-class replay helps explain the difference between Macro-F1 and Accuracy. Classes were grouped by training frequency before inspecting their individual scores. The eight classes in the lower two frequency groups account for 92.8% of the net Macro-F1 gain but only 7.3% of the test queries. Recall losses in the highest-frequency group offset these gains in Accuracy. This decomposition locates the observed benefit in broader category coverage rather than a uniform improvement across queries.

该段必须配图注或说明：一种顺序、三个构造的独立诊断；不是历史九组主表逐类拆解；一个极低频类没有测试查询。若正文空间紧，把逐类段放附录，正文保留已经更稳健的旧类结果。

## 5. Ablations: Separate Learning Support From Constructing Features

RQ2 建议顺序：learned/fixed Q、DM、direct X、文本初始化、GM。表中保留 no-GM，不再用它作为主要创新的证明。

> Learning the support and preserving feature distributions provide the clearest component gains. Fixing Q reduces Macro-F1 by 12.76, 21.67, and 8.91 points on ArXiv, SportsFit, and BookHis, respectively. Removing distribution matching also lowers the score on all three datasets. These comparisons establish the utility of adapting the support within the proposed construction.

上述固定 Q 差值来自原核心消融批次，完整方法 F1 为 52.96、80.02、48.53；不要混入直接 X 批次重建后的 80.06、49.00。正式排版前从同一批 CSV 自动填表，避免两位小数相减误差。

> We further compare this construction with independent feature optimization, retaining learned text selection and the same finite update budget. On SportsFit, the support-based feature construction improves Macro-F1 by 5.03 points with GCN and by 5.26 points with GraphSAGE. The corresponding paired intervals are [4.48, 5.57] and [3.93, 6.58]. ArXiv also shows a positive transfer result with GraphSAGE. This evidence supports the practical value of the pooled parameterization and bounded correction, particularly under the tested finite budget.

表注一次说清楚干预：Independent X 保留文本 Q 与损失目标，但取消池化特征参数化、信赖域和 X anchor，因此检验的是这一特征构造设计整体。三数据集、两种 student 全表放正文或附录，不只展示两个最佳数字。

用下面这一段替换 `07_ablation_discussion.tex` 中 “making it essential” 的尾部内容：

> Text-informed initialization contributes 2.51 Macro-F1 points on SportsFit, indicating that semantic information can help organize the initial support. GM provides feedback through the graph and text proxy objective, while the no-GM variant obtains similar scores in this ablation. The largest measured component gains are associated with learning the support and distribution matching.

此段已经承认 no-GM 结果，但重点仍是已验证的有用设计，不必再跟一整段防御性声明。

## 6. Shared Support: Put Its Cost Benefit in the Right Comparison

放资源或机制补充段，附上共享/独立的完整精度和成本表。

> Using one support matrix for feature pooling and text selection halves the number of assignment logits relative to separate feature and text assignments. In the matched controlled implementation, it reduces peak construction memory by 15.2% to 24.4% and construction time by 17.5% to 21.5%. This provides a compact common-support design, with the accuracy and cost tradeoff reported in the supplementary comparison.

“halves” 只修饰 assignment logits。不能改成总参数量减半、相对 GECC 更省显存或无精度损失。BookHis 独立分配的 F1 均值较高，表中应保留。

## 7. Baseline Fidelity: Use One Compact Diagnostic

主文实验设置一次定义共同 student 和构造适配，具体模块对应放附录。保真度段落要同时包含得到改善的原生核心结果。

> To examine sensitivity to the adaptation, we evaluate TAGSAM over complete continual construction chains and separately restore its native construction components on Cora. Extending the bounded adapter alone does not improve its final scores. With the native core, increasing the teacher budget from three epochs to 2,048 updates raises the common-student Macro-F1 from 41.50 to 63.80, compared with 60.55 for the earlier bounded adapter. This result identifies teacher training as an important condition when transferring the method to small increments.

紧接表注或附录设置说明：

> The original inner budget denotes full synthetic-data epochs. The large-graph bounded-budget sweep instead counts minibatch updates. We report these settings separately, together with the retained construction modules and the common downstream evaluation.

这一单位差别不能删。不要将最高有界预算那三个很低的分数拿来替换主表原行。大图原生完整 TAGSAM 和 STAD 仍未完成，相关主张应明确限制为所报告的适配比较。

## 8. Chronological Evaluation: Treat Applicability as a Research Question

标题可用 `Chronological Arrival`，不再创造一个新的方法名。正文给出三数据集全部结果，不需要展开每个失败或数值修复过程。

> We next evaluate publication-year streams on ArXiv and two AMiner citation subsets, ACM and DBLP. Each stream contains five stages with recurring categories. Under arrived-graph inference, CoTAG-Q obtains final Macro-F1 scores of 53.83, 87.87, and 81.50. It improves over the GECC adaptation on ArXiv, whereas GECC obtains higher mean scores on ACM and DBLP. These results extend the evaluation from category arrival to chronological content arrival and show that the relative benefit depends on the stream.

推断访问用一段说明：

> We also evaluate the current queries using only the current graph snapshot. On ArXiv, CoTAG-Q obtains 49.83 Macro-F1 compared with 48.10 for GECC. The difference is small on ACM and favors GECC on DBLP. Evaluating the same current queries with both graph snapshots distinguishes the effect of available inference context from the quality of the condensed training memory.

GECC 单来源敏感性最多一句，放附录即可：

> Replacing the five-source GECC text attachment with a single nearest source changes final cumulative Macro-F1 by less than 0.02 points on all three chronological datasets.

不要从这句话推出两者已通过等效性检验。

## 9. Resources and Conclusion

资源段落重点说明效益发生在哪个阶段：

> The condensed memory reduces the data processed during subsequent student training. Under the matched training budget, it reduces peak training memory by 89.5% to 93.8% and training time by 50.9% to 88.4% relative to the full-history reference. Construction is a separate investment in producing that memory, and its cost is reported independently from downstream training and graph inference.

这体现凝聚的使用价值，不是相对其它同等压缩方法的独有优势。是否值得多次使用，应由实际重复训练次数和完整构造成本计算；当前不加入未经测量的盈亏平衡次数。

结论建议：

> CoTAG-Q provides a concrete way to construct training memory for text-attributed graphs when historical training data cannot be revisited. Its learned support coordinates graph-informed feature construction with readable text selection. The experiments identify benefits in finite-budget feature construction, category-level performance on BookHis, and transfer across student architectures. Chronological evaluations further establish its applicability to evolving document collections, while exposing the dependence of relative performance on the arrival stream and available graph context.

## 10. Table and Figure Plan

| 内容 | 主文安排 | 补充材料 |
|---|---|---|
| 匹配基线主表 | 全部数据集、F1/Accuracy、按适配类别分组；不删除 GECC | 模块对应与每行重复范围 |
| 核心机制表 | Full、Fixed Q、No DM、No GM；配直接 X 的紧凑对照或两个 student 列 | 三构造区间、文本路径拆分、长预算诊断 |
| 时间流表 | 三数据集的 CoTAG-Q、CaT、GECC 原五来源 | 单来源 GECC、逐阶段和两种快照的完整结果 |
| BookHis 图 | 正文优先放阶段效用或旧类结果 | 本轮逐类频次图、全部类别和不确定性 |
| 共享分配成本表 | 空间允许时用一句话引用 | 精度、显存、时间同表，避免只展示成本 |
| TAGSAM 保真度 | 正文一段说明 teacher 条件的重要性 | 官方 epoch 单位、原生组件与 Cora 诊断表 |

统计用语使用 “three construction seeds, each averaged over three student seeds” 或 “three class orders and three construction seeds”。不把九次 student 训练写成九次独立构造。SD 在表注中说明来自哪个层级。

主体 8 页的空间优先保证问题、方法和决定性结果自洽。来源审计、失败日志、历史版本漂移等保留可复核入口，不逐段重复讲述。影响结论的范围和对照定义仍需在对应表注或正文出现一次。
