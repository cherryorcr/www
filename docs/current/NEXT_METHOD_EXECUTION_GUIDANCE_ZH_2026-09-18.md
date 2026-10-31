# 持续 TAG 凝聚后续方法推进指导

日期：2026-09-18  
适用版本：冻结 v4 及其后续受控实验  
状态：P0、P3-A、P1-X、P3-B、P2-A 第一版、P1-B 固定预算 guarded response transfer、P4 最低闭环、C3 单阶段传播诊断和 C1 类内多原型第一轮均已完成本轮小图诊断；P5 小图 rate curve、5-stage/base+4 长链、一组独立 frequency 顺序以及 arXiv/BookChild/SportsFit 大图 base+4 首轮也已完成；新增候选未晋升。P4 bounded-refresh 10% 未通过门禁，C3 方向/衰减算子未形成跨数据集稳定替代，C1 在 Cora 有信号但在 Citeseer 失效，C0 在长链最后阶段出现塑性回落且对顺序敏感。

## 0.1 C1 收尾后的状态更新

C1 已完成 Cora/Citeseer、3 个类别顺序、4 个 arm、6 个 student seeds 的完整小图实验。子簇 Mean-X、Weighted-X、Moment-X 同时改变了类内复杂度分配和 X 构造，因此只能作为联合候选评价，不能把 Moment-X 的局部高分单独归因于某一个算子。

聚合结果为：Cora 的 final/stage-average F1 分别为 C0 `48.40/49.04`、子簇 Mean-X `49.55/52.93`、Weighted-X `49.21/52.26`、Moment-X `50.36/55.30`；Citeseer 分别为 C0 `54.63/60.97`、`53.55/54.31`、`53.98/53.92`、`53.79/55.77`。因此 C1 标记为“完成但未晋升”，不建立 v5，也不进入默认 C0。

完整审计和逐顺序结果见 [C1 完成报告](C1_MULTI_PROTOTYPE_COMPLETION_ZH_2026-09-19.md)。后续 P5 小图与大图首轮已完成，见 [P5 总进展](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)。[指标审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)进一步确认 P5 的 task-slice old/new 不能与历史 pooled 指标混比；下一轮先保存预测与完整成本，再预注册 source-size 匹配顺序。不继续无界扫描手工 X 算子。

## 1. 总体判断

当前冻结 v4 已经证明：在历史原始数据不可回读、约 5% 合成节点预算下，持续合成图可以保留一定的下游任务效用。但目前尚未证明它稳定优于强 coreset，也没有一次使用场景下的端到端加速优势。

因此，后续主线不应继续无界扩大 response 搜索，而应按以下逻辑推进：

> 先证明合成节点本身有价值，再验证 response 选择是否可靠，最后扩展拓扑、文本和历史记忆管理。

当前最重要的负结果是：k-center 和 kmeans-medoid 在部分数据集和类别顺序上更强；v4 的 response 选择占构造时间约 99%，但只有少数阶段改变了配额。后续工作应围绕“如何让合成节点比简单真实选点更有价值”展开。

## 2. 当前版本与比较基线

### 2.1 低成本合成控制

后续实验统一使用以下低成本合成控制作为第一参照：

```text
dispersion allocation
+ 类内图文联合聚类
+ mean-X
+ joint_g5_c320 文本构造
+ frozen-old topology
+ response off
```

它用于判断新增组件是否真正带来收益，不替代已经归档的冻结 v4。

### 2.2 冻结 v4

冻结 v4 保留为当前正式比较版本：

```text
dispersion allocation
+ 类内图文联合聚类
+ mean-X
+ joint_g5_c320
+ frozen-old topology
+ 固定总预算内的 response 配额转移
```

冻结 v4 不因后续候选结果而被事后替换。后续实验使用 `v4_<component>_<date>` 命名，只有完成配置冻结、回归和独立确认后，才考虑新的正式版本号。

### 2.3 k-center 的解释边界

k-center 不是与当前方法完全相同的“只换一个选择器”。它保留真实节点的原始 X、文本和文本表示；当前方法则将多个节点压缩成合成 X，并对文本进行预算化融合。因此 k-center 的优势可能来自：

- 几何覆盖更充分；
- 不发生均值导致的多模态损失；
- X 与原始文本天然对应；
- 真实文本承载的信息量更大。

现有 k-center 结果主要是同 rows 的强 coreset 参照，不是严格同字节预算的最终结论。后续 X-only 实验必须固定簇、文本和拓扑，避免把多个差异误归因于均值操作。

## 3. 总体推进顺序

| 阶段 | 任务 | 主要问题 | 当前状态 |
|---|---|---|---|
| P0/control | 统一行枚举、seed、hash 和训练口径 | 结果差异是否只是实现混淆 | 已完成比较控制 |
| P3-A | response 可靠性和成本诊断 | 短训排序能否预测完整训练 | 已完成配对诊断，response-on 未晋升 |
| P1-X | 合成 X 改进 | 均值是否丢失任务相关信息 | Weighted/Moment 第一版已测，默认保留 Mean-X |
| P3-B/C3 | 受控拓扑消融 | 旧边冻结是否等于旧传播保持 | **已完成；跨边方向重要但无稳定替代规则** |
| P2-A | 图条件文本选择/注意力 | 图结构是否能改善文本利用 | **小图已完成；未通过晋升，保留负结果** |
| P1-B | adaptive-rate 与 extra budget | 更多 rows 是否真正值得 | 依赖 P3-A |
| P4 | bounded refresh、merge、split | 历史合成记忆如何管理 | **10% bounded refresh 已完成；未晋升，按负结果收尾** |
| P5 | rate、长链、更多顺序、大图 | 结果能否泛化 | rate/长链/一组独立顺序/大图首轮已完成；更多顺序与统一五层对照未完成 |

下文保留各阶段的目标设计，不表示所有进阶候选都已完成。实际状态以本表和对应验收报告为准；selector 未通过可靠性门禁，response-gain extra budget 不作为默认主方法。

## 4. 共同实验协议

除被明确研究的组件外，所有实验必须固定：

- 数据集、类别映射和类别到达顺序；
- 合法 raw pool 及其 hash；
- 累计 5% budget 规则和每阶段新增 rows；
- 类块排列、节点枚举和 cluster assignment；
- 文本来源、token/character budget 和 MiniLM checkpoint；
- topology、cross-degree 和 old-memory 冻结规则；
- student 结构、InfoNCE、batch、optimizer、steps 和 report seeds；
- selection seeds、selector-dev 和 report-dev 的权限；
- 代码、配置和产物 hash。

每轮新增实验必须保存 manifest，并记录：最终/阶段平均 Macro-F1、old/new F1、forgetting、rows、实际压缩率、边数、文本/特征字节、selector/student steps、构造与训练耗时、显存和历史 raw 访问日志。

## 5. 第一阶段：P0 比较控制

### 5.1 目标

建立新的 `v4_control`，排除当前 v4 与 dispersion 在随机类别顺序下可能存在的类块排列和节点编号差异。

### 5.2 做法

1. v4 和 dispersion 使用完全一致的类块排列。
2. 在相同配额、相同 cluster assignment 下检查生成的 X、文本、边和 row order。
3. 对生成记忆计算逐字段 hash。
4. 只切换 response on/off，构造严格配对的消融。
5. 使用独立的 selection seed group 和报告 student seeds。

### 5.3 验收

相同配额且 response 未改变时，两条路径应产生相同的合成产物。若不能满足，先修复实现，不解释性能。

Cora 1702 和 Citeseer 的已有失败顺序仅用于机制诊断。正式提升结论必须使用运行前预注册的新顺序，不能把开发阶段重复测试同一失败条件当成独立泛化证据。

## 6. 第二阶段：P1-X 合成 X 改进

这是当前最值得新增的组件分支。目标不是复制真实节点，而是在固定 rows 下生成更有任务效用的合成节点。

### 6.1 X-only 控制原则

所有候选必须固定：

- 类别配额；
- cluster assignment；
- 源节点集合；
- 合成文本；
- topology 和 row order；
- student、seed 和训练步数。

只改变合成 X 的生成方式。

### 6.2 候选方法

#### A. Mean-X

当前控制：

\[
\tilde{x}_j=\frac{1}{|C_j|}\sum_{i\in C_j}x_i
\]

它最小化簇内平方重构误差，但不保证保留多模态、尾部方向或 student 训练所需的特征结构。

#### B. Weighted-X

使用图文联合相似度、局部结构和覆盖性产生凸组合：

\[
\tilde{x}_j=\sum_{i\in C_j}\alpha_{ji}x_i,
\qquad \sum_i\alpha_{ji}=1
\]

权重需要设置最大值或有效支持数约束，避免退化成复制一个真实节点。第一版应使用确定性权重，避免同时引入可训练 attention 和 X 优化。

#### C. Moment-X

先以簇均值初始化，再在固定 rows 下匹配均值、方差或低秩协方差，并加入合成节点间排斥：

\[
L_{mom}=L_{mean}+\lambda_{cov}L_{cov}+\lambda_{rep}L_{rep}
\]

它主要针对均值造成的方差收缩和多模态丢失。

#### D. Response-X

以 Mean-X 初始化，使用当前合法 delta 和 probe student 优化合成 X，使图文损失梯度或多层响应接近真实 delta。该方案最直接，但额外成本和过拟合风险最高，因此最后实现。

### 6.3 推荐顺序

```text
Mean-X -> Weighted-X -> Moment-X -> Response-X
```

先做单阶段、同一旧记忆起点，再做完整三阶段递归链。每个候选都要与 k-center 和 kmeans-medoid 比较，但不能把真实节点基线误称为合成方法。

### 6.4 建议晋升门禁

门禁阈值必须在正式运行前写入 manifest。建议至少包含：

- 相对 Mean-X 在两个数据集上有稳定改善，或一个数据集明显改善且另一个不退化；
- 预注册顺序上不存在明显灾难性下降；
- old F1 和 new F1 不出现无法接受的单侧退化；
- 与 k-center 的性能差距缩小，而不是仅增加 rows 或训练步数；
- 构造成本和附加优化步数单独报告。

如果 X 改进后仍不能超过 Mean-X，不应继续堆叠更复杂的 selector，而应重新评估文本和 student 任务接口。

## 7. 第三阶段：P3-A response 可靠性与成本

### 7.1 核心问题

当前 selector 的问题不是候选数量不够，而是：

> 短训 ranking 是否能预测 canonical training ranking？

### 7.2 实验设计

1. 只保留少量预注册候选：dispersion、有限 class-row transfer 和少量拓扑候选。
2. 对同一候选记录短、中、canonical 三种训练长度的 response。
3. 使用按 seed 配对的增益、old/new 分项和 worst-horizon 结果。
4. 统计短训与 canonical 排名的一致性。
5. 统计选择后的 regret、负增益频率、fallback 率和 selector wall time。
6. 只对少量入围候选做完整 confirmation。

batch 和 horizon 是压力条件，不能直接当成更多独立样本量。LCB 或风险代理也不能解释为严格覆盖保证。

### 7.3 成本优化

训练前缀复用只能在以下条件通过数值等价检查后采用：

- seed、batch 顺序和 RNG 完全一致；
- optimizer state 和 scheduler 状态一致；
- 中间评估不改变训练状态；
- 共享前缀与独立重训的 loss/参数轨迹在允许误差内一致。

若短训排序没有预测力，应缩小搜索或关闭 selector，而不是继续增加训练成本。extra budget 必须暂时关闭。

## 8. 第四阶段：P3-B 拓扑诊断（已完成）

### 8.1 固定变量

固定 X、文本、rows、row order 和 student，仅改变新旧连接：

1. 当前跨边规则；
2. 不添加跨边；
3. 相同数量的随机端点跨边；
4. 相同数量的质量筛选跨边。

全 self-loop 可用于大幅删边诊断，但由于同时改变旧旧边和边预算，不属于严格的 frozen-old 主比较。

### 8.2 观测量

必须测量：

- 旧节点归一化邻接块漂移；
- 跨边消息量；
- 旧节点表征漂移；
- old/new Macro-F1；
- forgetting 和阶段平均性能。

即使旧旧边没有被改写，新跨边仍然会改变节点度和归一化传播：

\[
\hat A=D^{-1/2}(A+I)D^{-1/2}
\]

因此“旧边冻结”不等于“旧传播完全保持”。本轮结果见 [P3-B 完成报告](P3B_TOPOLOGY_DIAGNOSTIC_COMPLETION_ZH_2026-09-18.md)：跨边相对无跨边在 Cora/Citeseer 最终 F1 平均提升 2.97/2.02 个百分点；random 在 Cora 的部分顺序较高但波动显著，quality 仅在 Citeseer 平均略高。因此 current 暂时保留。下一步才考虑分块归一化、单向跨块传播或有界跨边权重。

## 9. 第五阶段：P2-A 图条件文本处理（第一版已完成，未晋升）

当前 `joint_g5_c320` 继续作为固定文本控制。P2 不能因为 concat 简单就直接替换。

正式结果见 [P2-A 完成报告](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md)：864 fits、144 memory 审计、零预算违规；graph selection 相对同 64-token center control 的最终增益为 Cora +3.05、Citeseer -1.85，均未超过 shuffled control。独立确认未触发；字符版 v4 未改名或替换。后置拓扑实验的方向与公平算子控制见该报告第 9.1 节。

### 9.1 第一版方法

使用当前合成 X 和合法当前邻域摘要作为 query，使用源文本片段作为 key/value，在固定 character/token budget 下选择互补片段，最后仍由冻结 MiniLM 重编码。

### 9.2 必须对比

- 同预算 concat；
- text-only attention；
- attribute-only query；
- 合法图结构条件；
- 打乱图结构条件；
- 集合 pooling。

必须固定源簇、合成 X、rows、edges、文本预算和 student。新增可训练参数、片段编码、选择和重编码成本都要计入。

如果 graph-conditioned 结果不优于 text-only 或 shuffled graph，就不晋升 attention。普通 text attention/DeepSets 的历史结果不能换名称作为新的正面证据。

## 10. 第六阶段：P1-B 动态预算与 extra budget

P1-B 的固定预算 guarded response transfer 已完成小图验收，结果见 [P1-B 完成报告](P1B_GUARDED_RESPONSE_COMPLETION_ZH_2026-09-18.md)。平均最终 F1 相对 dispersion 在 Cora/Citeseer 分别为 `+1.82/+1.62` 个百分点，但仅有 `4/9` 和 `2/9` 个阶段真正接受转移，构造成本约为低成本 dispersion 的 `75--130x`。此外，Citeseer order1702 的 matched random transfer 高于 response，说明收益不能完全归因于 response。该结果只能支持“固定预算候选有潜力”，不能晋升为默认主方法，也不能直接上大图。

因此，以下动态预算和 extra budget 仍依赖后续独立确认，不能作为当前主方法。 

### 10.1 先做固定总预算分配

先研究给定新增 rows 时，如何在类别和子簇之间分配：

- dispersion allocation；
- 类内复杂度 allocation；
- 固定预算的 response-based marginal allocation。

### 10.2 再做 extra budget

response-gain extra budget 必须至少有两个对照：

- 同总 rows 的固定规则；
- 同样增加 rows、同样增加 student steps 的固定规则。

否则无法判断收益来自更聪明的配额，还是仅来自更多合成节点和更多训练更新。

动态预算应记录每阶段：信息增益、rows、实际压缩率、old/new F1、selector/student steps 和耗时。信息量驱动压缩率是合理研究问题，但目前不是已验证机制。

## 11. 第七阶段：P4 历史合成记忆管理

按从弱到强的顺序推进：

```text
re-score only
-> bounded refresh
-> merge + reallocate
-> synthetic split
-> importance eviction
```

所有操作只能读取旧 synthetic memory、合法保存的统计和当前 delta，不能回读历史真实特征、文本或邻接。每个动作都要保留 fallback，并记录 birth stage、改写比例、来源和边变化。

先比较 keep 与单一动作，再考虑组合。不能同时刷新 X、重写文本和重建拓扑后，将全部收益归因于一个 memory manager。

## 12. 第八阶段：P5 稳健性与扩展

小图阶段应继续补充：

- 1%、5%、10% 的 matched-quota rate curve；
- 更多预注册类别顺序；
- 5-stage 与 base+4；
- 更长持续链；
- 固定和增长 budget 的成本曲线。

低 rate 下如果 rows 小于 batch16，必须单独报告实际 batch、负例数和有效更新次数，不能把 batch 混淆成压缩率收益。

Cora/Citeseer 类别数量有限，不能强行拆成 8/10 个非空 class-incremental 阶段。若同一类别分批到达，应另立 data-incremental 协议。

只有小图候选在独立顺序、独立 seed 和成本账上通过后，才进入大图 base+4 主协议；3-stage 作为快速消融，8/10-stage 只适用于类别数量足够的数据集或另行定义的协议。

## 13. 统一验收规则

### 13.1 正确性门禁

- 训练源、selector-dev 和 report-dev 权限分离；
- 历史 raw 访问可审计；
- synthetic-only 主方法不复制真实节点；
- budget、rows、文本预算和 topology 规则一致；
- 冻结版本回归通过；
- 所有预注册运行和 artifact audit 完整。

### 13.2 性能门禁

正式运行前固定：

- 主指标和汇总方式；
- old/new 最大允许退化；
- 独立初始化和类别顺序；
- 选择后 regret 和 fallback 的报告方式；
- 成本是否属于主方法约束。

不能只报告数据集平均值来掩盖一个顺序的明显失败，也不能用 selector-dev 反复调整阈值后再将同一 split 当作独立证据。

### 13.3 一次只晋升一个改变

如果 X、topology、text 和 selector 同时变化，无法判断性能来源。每一轮只晋升一个通过门禁的组件，其余候选保留负结果。

## 14. 推荐的下一批具体任务

下列原始执行顺序保留为任务索引，补充当前完成状态：

1. P0 已统一类块和节点枚举，后续沿用。
2. P1-X 已完成第一版 X-only 实验，Weighted-X/Moment-X 未晋升。
3. X 独立确认没有通过的入围候选，不因流程列表而继续扩大搜索。
4. P3-A 已完成 response 配对与成本诊断，默认 response off；更强 ranking 估计仍需另立候选协议。
5. P3-B 四臂拓扑诊断已完成，current 保留；有界跨边、单向跨块传播只作后置设计。
6. P2-A 第一版已完成但未通过：真实 graph query 没有优于 shuffled query，Citeseer 还低于 center concat；因此暂不晋升，保留负结果。
7. 只有 selector 可靠且成本可接受后，再做 adaptive-rate 和 extra budget。
8. 最后做 P5 独立顺序与统一成本账，再据此决定是否进入大图；rate curve 与 5-stage 结果见 [P5 小图扩展完成报告](P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md)。

## 15. 论文定位

在后续候选尚未通过前，论文不应宣称 SOTA、普遍超过完整数据、稳定优于强 coreset 或一次使用总计算更快。

当前可以稳妥表述为：

> 本项目研究了历史原始数据不可回读条件下的合成图文追加，并给出了可审计的旧记忆保持、预算分配、文本压缩和 student response 选择协议；实验同时揭示了合成均值、短训选择、跨阶段拓扑和动态预算的适用边界。

如果后续 X 构造通过，主线可进一步收敛为“任务感知的合成节点构造 + 可审计持续追加”。如果 selector 或 attention 通过，再将其作为独立机制加入，而不是预先把所有候选包装成完整新版本。

## 16. 参考文档

- [v4 状态与 P0-P5 路线](V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md)
- [小图扩展与成本比较](SMALL_ORDER_COST_COMPARISON_REPORT_ZH_2026-09-17.md)
- [图条件注意力文本后续计划](P2_GRAPH_CONDITIONED_ATTENTION_FOLLOWUP_ZH_2026-09-17.md)
- [冻结 v4 主线目标](V4_TARGET_METHOD_AND_OPTIMIZATION_PLAN_ZH_2026-09-15.md)
- [P2-A 图条件文本协议](P2A_GRAPH_CONDITIONED_TEXT_PROTOCOL_ZH_2026-09-18.md)
- [P2-A 图条件文本完成报告](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md)

## 17. 综合评估入口

当前各组件的证据等级、理论解释、当前最优默认配置，以及 P4/P5 的分阶段执行方案，统一见：

[当前持续 TAG 凝聚方法综合评估与后续推进方向](CURRENT_METHOD_COMPREHENSIVE_ASSESSMENT_ZH_2026-09-18.md)

该报告特别区分：

- `dispersion + Mean-X + center concat + current cross-edge`：当前低成本主控制；
- `response_guarded`：小图平均有积极信号，但尚未通过成本、稳定性和因果归因门禁；
- P4：历史 synthetic memory 管理，当前尚未完成系统验收；
- P5：小图 rate curve、5-stage/base+4、一组独立 frequency 顺序和三个大图的 base+4 首轮已完成；结果显示 C0 存在明显顺序依赖，且大图长链累计 F1 有数据集差异。更多独立顺序和统一五层对照仍未完成，见 [P5 大图报告](P5_LARGE_BASE4_COMPLETION_ZH_2026-09-19.md)。
