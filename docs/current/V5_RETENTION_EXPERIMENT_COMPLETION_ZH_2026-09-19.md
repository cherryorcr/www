# v5 Retention-Constrained Append 实验结项与验收报告

日期：2026-09-19。冻结协议见 [V5_RETENTION_FROZEN_PROTOCOL_ZH_2026-09-19.md](V5_RETENTION_FROZEN_PROTOCOL_ZH_2026-09-19.md)。

## 1. 结论与完成边界

**v5 候选已经实现，预注册的 288 次小图训练全部完成并通过工件审计；三个候选均未通过性能晋升门槛，因此不将它们命名为已成立的 v5 主方法，继续保留 C0 作为低成本合成控制。**

|任务|验收状态|实际产物|
|---|---|---|
|冻结定义、四臂、参数、指标、晋升条件|完成|协议、代码快照、运行 freeze、顺序 registration|
|Cora/Citeseer 四臂 × 两顺序 × 三阶段 × 六 seeds|完成|288 次完整 student 训练，48 张阶段 memory|
|独立确认条件|完成|仅由合法 pool 类规模选取新顺序，使用新的 9100..9105 seeds|
|预测、checkpoint、漂移、保护区、预算、成本审计|通过|288 份预测复算、48 张 memory 审计、15 项测试|
|单组件与组合效应分析|完成|配对差、置信区间、interaction、旧类/新类和传播漂移|
|arXiv base+4 条件性扩展|未触发，不启动|小图门槛未通过，严格执行预注册的停止规则|
|主方法版本晋升|不通过|保留 C0；core 作为有局部收益的研究候选|
|论文长文写作|本轮未开始|按用户先完成方法实验的顺序执行|

没有把未触发的大图实验记作“已跑完”，也没有查看本轮结果后修改参数重跑、挑选最佳 seed 或拼接各条件的最好结果。

## 2. 本轮方法实际做了什么

共同底座仍为：累计合法 pool 的 5% growing budget、dispersion 类预算、合成 Mean-X、joint_g5_c320 中心文本拼接后冻结 MiniLM 重编码、旧原始合成边列表保留、最近文本双向跨边。没有加入昂贵 response selector。

|组别|旧合成 X|文本|训练传播支持|
|---|---|---|---|
|C0|全部冻结|全部旧文本和 embedding 冻结|现有双向跨边、整体度归一化|
|V5-core|永久保护区冻结，固定缓冲区有界更新|同 C0|同 C0|
|V5-topology|同 C0|同 C0|受旧传播漂移约束的跨块算子|
|V5-full|同 V5-core|同 C0|受约束算子，计入实际 X 更新造成的漂移|

**core/buffer：**每批新合成节点入库时，以 mass 升序确定 floor(10% × 新块 rows) 的缓冲行，其余为永久保护区。缓冲成员不在后续重新洗牌。保存缓冲 X 的入库 anchor；只用当前新合成块中的图文最近行提出 0.25 插值更新。更新限制在 anchor 的逐行相对 5% 球内，并缩小到上阶段传播算子的一跳/两跳 X 响应相对漂移都不超过 1%。mass、文本、stage 和保护区 X 不改写。

**topology：**先构造 `S_base = blockdiag(S_old, S_new)`，再构造现有双向跨边的归一化提案 `S_prop`。实际支持为：

\[
S(\lambda)=(1-\lambda)S_{\mathrm{base}}+\lambda S_{\mathrm{prop}}.
\]

用可审计的线性/二次范数上界求最大可行 lambda，使历史行算子相对漂移不超过 5%，历史一跳/两跳 X 响应漂移不超过 2%。没有用 report 分数调整 lambda，没有额外训练 student。原始边与已归一化支持分开保存，避免重复归一化和重复添加自环。

本轮“新信息效用”采用接近新合成特征和增大可行交互强度的代理，没有直接最大化真实 student 的新类收益。因此它完成的是冻结协议中的最小可行 v5 候选检验，不能表述为已经求解了真实保持-塑性最优问题。

这仍是**合成节点方法**。新 X 来自簇内均值，缓冲更新的目标也是新合成节点，不是把 k-center 真实节点拿来当凝聚节点。新阶段构造只读取当前阶段合法节点；append 接口只接收旧合成 memory 和当前合成块。离线实验框架为编码、划分和真实图 transductive 评估加载源数据，不代表 append 可以回放旧原始数据。

## 3. 数据、顺序和 student 协议

|数据集/条件|阶段类别|各阶段新 pool 节点|累计合成 rows|student seeds|
|---|---|---|---|---|
|Cora canonical|[0,1] / [2,3] / [4,5,6]|501 / 858 / 536|25 / 67 / 94|100..105|
|Cora matched_unseen|[0,3] / [2,6] / [1,4,5]|497 / 824 / 574|24 / 66 / 94|9100..9105|
|Citeseer canonical|[0,1] / [2,3] / [4,5]|778 / 953 / 498|38 / 86 / 111|100..105|
|Citeseer matched_unseen|[2,4] / [0,3] / [1,5]|801 / 874 / 554|40 / 83 / 111|9100..9105|

新顺序从保持阶段类别数、排除已记录完整划分、且每个阶段类集合都改变的方案中，仅按 pool 数量误差选择。Cora 有 185 个合格划分，Citeseer 有 71 个。它们是近似规模匹配，不是每阶段精确等规模。四个 arm 在同一个条件内的 rows、源块、文本、steps、seeds 完全相同。

两层 GCN，hidden/out=384，冻结 MiniLM=384，可训练 text projection 和 temperature；每阶段 student 从头训练，Adam lr=0.001、batch=16、steps=`max(50,min(2048,16*rows))`；目标为双向图文 InfoNCE，不使用节点分类 CE。类标签用于合法 pool 的分组构造，这与 student 不使用分类训练标签是两回事。

selector/report/test 分别为 original_id % 10 的 1/3/2，均排除出构造 pool。本轮不使用 selector 评分，也不评价 test。真实图评估仍是 transductive，包含未标注的 holdout/future 输入，与 C0 相同。

主指标是**节点分类 Macro-F1**：全部类别 prompt 竞争，只在已见类别的 report 节点合并支持上计算；old/new 均从该合并支持的逐类 F1 得出。历史 task-slice 指标和 seen-prompt 指标另存，不混表。新顺序与新 seed group 提供新的确认条件，但重复使用的 report-dev 不等于从未访问过的论文测试集。

## 4. 完整性能结果

最终阶段 Macro-F1，六 seeds 的 mean ± sample std：

|条件|C0|V5-core|V5-topology|V5-full|
|---|---:|---:|---:|---:|
|Cora canonical|49.85 ± 3.28|51.39 ± 2.93|47.67 ± 2.07|44.83 ± 2.19|
|Cora 新顺序|44.78 ± 2.97|43.75 ± 3.35|42.76 ± 1.52|44.06 ± 2.84|
|Citeseer canonical|59.17 ± 1.72|58.03 ± 2.93|51.72 ± 3.41|50.81 ± 1.28|
|Citeseer 新顺序|52.38 ± 2.70|54.22 ± 1.76|54.61 ± 1.38|54.82 ± 3.45|

与同条件 C0 的最终 F1 配对差，单位为百分点，直接由未舍入结果计算：

|条件|core - C0|topology - C0|full - C0|
|---|---:|---:|---:|
|Cora canonical|+1.54|-2.18|-5.02|
|Cora 新顺序|-1.03|-2.02|-0.71|
|Citeseer canonical|-1.14|-7.46|-8.36|
|Citeseer 新顺序|+1.85|+2.23|+2.45|
|四条件等权均值|+0.30|-2.36|-2.91|

四条件平均只是本轮描述统计，不是新增数据集或跨域泛化置信度。

阶段平均 Macro-F1：

|条件|C0|core|topology|full|
|---|---:|---:|---:|---:|
|Cora canonical|53.08|53.06|53.16|53.43|
|Cora 新顺序|53.49|52.57|48.75|49.70|
|Citeseer canonical|64.29|63.79|60.08|59.48|
|Citeseer 新顺序|49.32|50.25|49.48|49.95|

最终 pooled old / new F1：

|条件|C0|core|topology|full|
|---|---:|---:|---:|---:|
|Cora canonical|65.44 / 29.06|67.45 / 29.97|60.60 / 30.42|56.42 / 29.39|
|Cora 新顺序|45.77 / 43.46|43.07 / 44.65|44.38 / 40.58|45.13 / 42.64|
|Citeseer canonical|66.46 / 44.61|65.71 / 42.69|60.60 / 33.95|60.05 / 32.33|
|Citeseer 新顺序|64.46 / 28.21|66.14 / 30.40|65.68 / 32.48|65.24 / 33.99|

## 5. 晋升为什么没有通过

core 是三个候选中最值得保留的：两个条件超过 +1.5，新类平均 +0.59，构造成本接近 C0。但另外两个条件最终下降 1.03/1.14，Cora 新顺序旧类下降约 2.70，未满足全部条件不明显退化和旧类保持要求。

两个正收益条件的普通配对 95% t 区间分别为 [-1.31,+4.38]、[-2.47,+6.16]，均跨过零。六 seeds 的证据不足以把正均值认定为稳定提升。所有候选也未通过预注册的 12 项多重校正非劣下界；这属于证据不足，不能反过来证明 core 没有价值。

topology/full 在 Citeseer 新顺序改善，但其他条件退化，尤其 canonical Citeseer 分别 -7.46/-8.36。full 在 Cora canonical 的普通配对 95% 区间为 [-7.58,-2.45]，在 Citeseer canonical 为 [-10.20,-6.53]，本轮有明确的退化证据。

|门槛|core|topology|full|
|---|---|---|---|
|所有条件最终平均差 >= -1|未过|未过|未过|
|至少两个条件 >= +1.5，含新顺序|通过|未过|未过|
|所有条件阶段平均差 >= -1|通过|未过|未过|
|所有条件 pooled old 差 >= -2|未过|未过|未过|
|pooled new 总体正收益且至少两个条件为正|通过|未过|未过|
|构造成本 <= 2 倍 C0|通过|通过|通过|
|多重校正的非劣证据|未过|未过|未过|
|工件与数值审计|通过|通过|通过|

## 6. 机制是否真正起作用，组合是否有用

**约束确实生效。** topology 的 lambda 约 0.0263–0.0321，full 约 0.0131–0.0162，表示只沿 C0 跨块归一化提案前进约 1.3%–3.2%。full 不是“跨边权重固定为 0.25”的 C3 旧实验；它还限制整个历史传播响应，并使用永久 core/buffer。

以 canonical 最终阶段为例：

|数据集/组别|旧两跳 X 相对漂移|固定当前 student 参数下的旧合成 embedding 相对漂移|最终 pooled old F1|
|---|---:|---:|---:|
|Cora C0|0.1831|0.3048|65.44|
|Cora topology|0.0195|0.0182|60.60|
|Cora full|0.0160|0.0133|56.42|
|Citeseer C0|0.1616|0.2667|66.46|
|Citeseer topology|0.0195|0.0489|60.60|
|Citeseer full|0.0158|0.0149|60.05|

所以可以支持的结论是：**本实现能控制指定漂移，却不能据此保证最终分类保持。** 不能解释成约束没有实现、梯度没有生效或额外训练成本过高。

原因要区分事实和待验证解释：

1. 事实：每个阶段 student 都重新训练。约束主要控制固定参数下输入/传播的变化，未约束新训练参数与旧训练参数的差。即便 `f(theta_t, new_memory)` 与 `f(theta_t, old_memory)` 很接近，也不能推出它接近 `f(theta_{t-1}, old_memory)`，更不能推出真实图上的分类边界不变。
2. 事实：漂移界衡量旧合成图，而报告指标来自真实完整图，两者不是同一支持。分类 F1 也不是输入范数的连续保证。
3. 待验证解释：2% 响应界使新旧交互几乎退回块对角传播，可能牺牲真实图推理需要的跨类/跨阶段传播。本轮没有宽松 epsilon 消融，不能断言单独放宽阈值就能恢复性能。
4. 待验证解释：class-incremental 下，新块都是新类；缓冲 X 向最近新合成行移动，却保持旧文本不变，仍可能削弱图文对应。低 mass 也可能代表少数模式，而不等于应被更新的“低价值记忆”。anchor 和小范数界没有直接保护分类 margin。

组合效应 `full - core - topology + C0` 的最终 F1 配对均值依次为 -4.37、+2.34、+0.23、-1.64，四条件等权均值约 -0.86；四个普通 95% 区间均跨零。**没有证据表明这两个组件组合后存在稳定正协同。** 因为 full 的更新约束依赖自己的历史算子，interaction 包含真实的跨阶段路径依赖，不能解释成静态参数相加。

另外从保存的阶段预测计算了旧节点预测翻转和正确/错误迁移，见 `posthoc_prediction_transitions.csv`。这是结果后的只读诊断，不参与选择、调参或晋升门槛。

Cora canonical 最后一次阶段转移中，C0 的旧节点预测翻转率为 35.80%，full 为 29.97%；对应旧节点 accuracy 分别由 43.92% 升至 62.48%、由 45.57% 升至 52.30%。更少预测变化并不自动对应更好结果。这里是固定旧节点支持上的 accuracy，不能与 pooled old F1 混用；阶段前后训练预算也会随 rows 改变，因此这项诊断不能单独证明具体因果路径。

## 7. 计算成本和内存

以下构造时间包括三阶段共同新块构造及对应 append 算子；训练为该方法三个阶段 × 六 seeds 的总训练段墙钟秒数，均有同步计时：

|条件|C0 构造秒|full 构造秒|full/C0 构造|C0 18 fits 训练秒|full 18 fits 训练秒|
|---|---:|---:|---:|---:|---:|
|Cora canonical|0.518|0.525|1.012|39.71|39.49|
|Cora 新顺序|0.418|0.424|1.013|39.02|39.13|
|Citeseer canonical|0.511|0.518|1.012|50.60|50.26|
|Citeseer 新顺序|0.561|0.569|1.014|50.45|49.99|

core 构造为 C0 的约 1.006 倍，topology 为约 1.007 倍，full 为约 1.012–1.014 倍。本轮核心问题是效果稳定性，成本不是拒绝原因。约 1% 的训练时长差不能作为加速贡献。

新鲜合法 pool 文本编码：Cora 0.903 秒、Citeseer 1.050 秒，共享每个数据集一次；独立部署一条方法链仍需承担这个成本，未把缓存写成零。初始化、评估、checkpoint 重载、漂移诊断、存盘及构造审计时间另列。288 次训练段合计 716.49 秒，是并发任务各段墙钟时间之和，不是整轮自然耗时或 GPU 满负荷时长。

canonical 最终逻辑 payload：Cora C0 326,216 bytes，full 346,968 bytes；Citeseer C0 384,817 bytes，full 408,293 bytes。额外空间来自 buffer anchors、显式权重和原始边记录。两类方法有效归一化支持的非零位置相同；topology/full 存储 edge entries 含显式自环，不能直接拿它与 C0 不含自环的原始边数比较。成本表分别提供 raw edge entries 与 normalized support entries。

每次训练的 PyTorch peak allocated 约 115.92/116.64 MiB，不包含驱动上下文等进程级显存口径。实验使用两张 RTX 4090、conda tagsam。没有启动额外 response 候选 student 训练。

## 8. 可复现性与交付

- 核心/门槛测试：本地和远程均 15 passed。
- C0 canonical 的 6 张 memory 与历史归档的 X/text/mass/stage/depth/edges 逐项一致；最终 F1 复现 49.85/59.17。
- 48 张 memory：预算、新块、保护区、永久 buffer、累计 anchor、原始旧边、实际传播界及内容哈希均通过。
- 288 次训练：checkpoint 哈希、参数哈希、支持哈希、steps、report ID、保存 logits 重算的 pooled/task-slice 指标均通过。
- 24 组阶段零“同条件同 seed 四臂”重复：输入、初始化、最终 state 均逐位一致，F1 极差为零。
- 测试标签未作评估；没有 response/dev 选择成本；结果未用来改变本轮候选配置。

全部工件取回本地后再次独立审计通过，见 `summary_local_audit/`。本地 PyTorch 2.9.1 CPU 与远程 2.4.1 CUDA 的漂移复算最大绝对差为 1.11e-16。派生的双精度漂移数值采用 atol=1e-12、rtol=1e-10；所有文件、原始 tensor、训练支持和 state 哈希仍要求完全一致，未放宽实验漂移界、指标或晋升条件。两端最终决策相同。

主要文件：

- 算子：`code/v2_llm_fusion/retention_constrained_append.py`
- runner：`code/v2_llm_fusion/run_v5_retention.py`
- 独立审计/汇总：`code/v2_llm_fusion/audit_v5_retention.py`
- 测试：`code/v2_llm_fusion/test_v5_retention.py`
- 本地结果：`results/v5_retention_20260919/`
- 明细：`summary/cell_summary.csv`、`stage_summary.csv`、`paired_differences.csv`、`costs.csv`、`mechanism.csv`、`interaction.csv`、`audit.json`、`decision.json`
- 服务器训练源快照：`/home/pc/research/TAG_condensation/experiments/v5_retention_20260919/`
- 服务器原始工件：`/home/pc/research/TAG_condensation/results/v5_retention_20260919/`

训练使用的源码/协议快照保存在 `source_snapshot.tar.gz`；运行 freeze 记录实际参与训练的模块哈希。汇总器单独记录 auditor 哈希，新增只读诊断未改变训练快照。每个条件保留全部 checkpoint、报告预测、阶段 memory 和构造记录。

## 9. 当前方案与后续取舍

当前继续使用 **C0：5% fixed-rate + dispersion + Mean-X + concat/re-encode + frozen-old append + response off**。这表示它是目前稳定的低成本合成参照，不表示它在所有条件优于 k-center、完整数据或强静态 TAG 方法；历史 v4（response on）仍单独保留，不改名。

core 的局部收益值得记录，不能把它写成已经证实优于 C0。若继续研发 v5，应先解释低 mass 缓冲选择和跨新类 X 更新的失效，再提出单一改动、冻结参数并另留确认条件。不要把本轮两个有利条件挑出来晋升，也不要为得到正结果事后放宽本轮验收线。

拓扑方向目前不宜直接加入主方法。可以在独立的开发实验中研究响应界强度与有效跨块交互之间的曲线，但本轮已经不能支持“越保守的传播越能保留旧知识”这一简单假设。大图是否会改变该结论仍未知，本轮没有数据支持肯定或否定；按既定顺序，先有小图验收通过的候选再投入 arXiv 长链。

本轮形成的是一个已执行、有审计证据的 v5 候选结项，尚未形成可取代 C0 的新版主方法。论文应保留这种边界，不宣称 v5 已晋升或 SOTA。
