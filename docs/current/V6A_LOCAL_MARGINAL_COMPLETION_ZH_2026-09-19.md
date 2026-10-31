# v6-A 局部合成边际编辑实验结项报告

日期：2026-09-19。冻结协议见 [V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md](V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md)。审计数值说明见 [V6A_AUDIT_NUMERICAL_AMENDMENT_ZH_2026-09-19.md](V6A_AUDIT_NUMERICAL_AMENDMENT_ZH_2026-09-19.md)。

## 1. 结论

v6-A 完成了预注册的 560 次 student fits 和独立工件审计，但没有通过进入递归 v6-B 的门禁。因此当前主方法仍保留 C0，v6-A 不晋升为 v6 主方法，也不启动动态总预算或大图扩展。

本轮最重要的结论不是“局部编辑无论何时都无效”，而是：

1. 在固定总 rows、冻结历史 memory 的条件下，局部合并/拆分没有提供稳定的可选增益；
2. 受影响类别的整类重聚类控制在若干 stage 有明显正收益，但它不能被重新包装为局部边际方法，且短训排序与独立 report 排序不够稳定；
3. v6 的新假设没有解决当前核心问题：student response 排序不能可靠地把有限额外构造成本转换成稳定的持续性能收益。

## 2. 本轮方法

共同底座是 C0：累计合法 pool 的 5% growing rows、dispersion 类预算、类内 joint 聚类、Mean-X、`joint_g5_c320` 文本拼接后冻结 MiniLM 重编码、旧合成行与旧旧边冻结、cross-degree=1。

v6-A 只研究 stage 1/2 的单阶段反事实，不把不同 stage 的赢家拼接成一条持续轨迹。

|候选|具体操作|保持不变的条件|
|---|---|---|
|C0|不转移当前新阶段的类配额|总 rows、旧 history、训练协议|
|local|donor 类合并 Ward 增量最小的一对簇；recipient 类将 SSE 最大的非 singleton 簇二分；只重算三个新槽位|源池、总 rows、类总配额、其它新行、旧 memory|
|recluster|同一 donor/recipient 配额变化，但完整重聚类两个受影响类；用 Hungarian center matching 对齐槽位|源池、总 rows、类总配额、未涉及类别、旧 memory|

所有新节点仍是源簇的合成均值，不使用真实节点 coreset 作为凝聚节点。local 与 recluster 的区别是编辑范围，不是下游 student 或文本模型。

## 3. 实验规模与审计

|项目|结果|
|---|---:|
|数据集|Cora、Citeseer|
|顺序|每个数据集 canonical + 1 个仅按 pool 规模匹配的新顺序|
|阶段|每个顺序 stage 1、stage 2|
|候选|每个 cell 5 个：C0 + 2 个 local + 2 个 recluster|
|训练 seed|selection 6200/6201、6300/6301；独立 report 9200..9205|
|student fits|560|
|checkpoint observations|4480 个，每个同时保存 selector/report|
|优化步数|810240|
|student 训练时间|约 1787 秒；评估/快照约 27 秒|
|测试标签|未读取、未评分|
|工件审计|通过|
|checkpoint 重载|560 个模型、1120 个 split logits 复核通过|

重载 logits 的最大绝对误差为 `5.72e-6`，在 `rtol=1e-4, atol=1e-5` 内。这个差异来自 GPU forward 的浮点非确定性，不是 checkpoint 或指标不一致。

## 4. 性能与选择结果

下表的 gain 都是相对同一 cell 的 C0，单位为 Macro-F1 百分点；oracle 使用独立 report seeds 的完整训练结果，selector gain 是按冻结规则在 selection seeds 的 1/4、1/2 前缀中选择后，再在独立 report seeds 上测得的结果。

|数据集|顺序|stage|C0 F1|local oracle headroom|local selector gain / 选择|recluster oracle headroom|recluster selector gain / 选择|
|---|---|---:|---:|---:|---:|---:|---:|
|Cora|canonical|1|54.87|0.00|0.00 / C0|0.00|0.00 / C0|
|Cora|canonical|2|47.40|+6.55|0.00 / C0|+3.66|+3.17 / recluster|
|Cora|confirmation|1|53.56|0.00|0.00 / C0|0.00|0.00 / C0|
|Cora|confirmation|2|48.26|+3.18|0.00 / C0|+3.33|+3.33 / recluster|
|Citeseer|canonical|1|68.64|+1.42|0.00 / C0|+2.57|0.00 / C0|
|Citeseer|canonical|2|58.61|+0.21|0.00 / C0|+0.99|+0.99 / recluster|
|Citeseer|confirmation|1|57.38|+0.35|0.00 / C0|+3.91|+2.36 / recluster|
|Citeseer|confirmation|2|50.23|+0.50|0.00 / C0|+1.13|+1.13 / recluster|

local 在 8 个 cell 全部回退 C0。recluster 在 5/8 个 cell 被 selection 规则接受，说明“受影响类的完整重聚类”比“三槽位局部编辑”更有潜力，但这只是控制条件的诊断证据，不能当作已注册的 v6 主方法结果。

## 5. 晋升门禁

|门禁|要求|实际|判定|
|---|---|---:|---|
|新顺序 oracle headroom|至少 2 个 cell >= 1 pp|1 个（local family）|失败|
|短训排序|平均 Spearman >= 0.3|`-0.018`|失败|
|local 选择收益|平均 >= +0.5 pp，且不低于 geometry/uniform|0.00 pp；geometry +0.683 pp|失败|
|新顺序稳定性|每个数据集阶段平均不低于 0，且至少接受一次非 C0|两者均未接受 local|失败|
|保持约束|每个 cell cumulative >= -2 pp、old >= -3 pp|通过|通过|
|工件审计|源、预算、memory、checkpoint、指标一致|通过|通过|

综合判定：`continue_v6b = false`。这不是“局部编辑被证明没有任何价值”，而是它没有达到继续递归化、动态加预算或上大图所需的稳定证据。

## 6. 成本与实际收益

本轮完整诊断消耗 560 次训练。若把 5 候选和 4 个 selection seeds 的 1/2-prefix 训练作为一次部署选择成本，估算成本约为 C0 单次构造加一个正式 student 的 6.6--14.7 倍，stage 2 更高。当前 local 选择最终全部回退 C0，因此这些额外成本没有转化为部署收益。

recluster 的正收益不能简单解释为“三个槽位局部修改更好”：它重聚类了两个受影响类别，改变了更多合成源簇。它也不能直接成为主方法，因为本轮的选择排序仍受 stage、数据集和 seed 影响，且 v6-A 的注册假设是 local marginal edit。

## 7. 对当前研究主线的影响

当前可以保留的主线仍然是：

- C0 的低成本、固定协议、历史冻结的持续 TAG 合成控制；
- 合法源池、预算、旧 memory 不改写和 student 可重训性的完整审计；
- 用 response/候选选择研究“student 可重训性”这一机制问题，但不宣称它已经带来稳定收益；
- 将局部编辑、受影响类重聚类、短训排序失配和成本放入负结果/机制分析。

当前不能写成：v6 局部边际方法普遍优于 C0、已经实现动态最优预算、已经解决历史保持-新类塑性，或已经达到 SOTA。

## 8. 后续建议

1. 不启动递归 v6-B，不把 local 或 recluster 的 cell 最优结果拼成新主表。
2. 如果继续追求方法创新，应新建独立协议专门验证“受影响类别完整重聚类 + 低成本预筛选”，并在新的顺序/seed 上预注册；不能沿用本轮 report 结果调参。
3. 优先研究更便宜的候选排序代理，验证能否达到与 geometry 相当的成本和稳定性；当前昂贵 response 选择的收益不足以抵消成本。
4. C0、v5 负结果、P2 文本负结果、P3 response 负结果和 v6-A 负结果可以组成论文的机制分析主线；论文应把“哪些算子有效、哪些算子在何种条件下失效”作为贡献，而不是强行包装单一新模块。

## 9. 复现工件

- 远程实验目录：`/home/pc/research/TAG_condensation/results/v6a_local_marginal_20260919/`
- 代码快照：`/home/pc/research/TAG_condensation/experiments/v6a_local_marginal_20260919/`
- 紧凑汇总：[v6a_audit_summary.json](../../results/v6a_local_marginal_20260919/audit_summary_reload.json)
- 冻结协议：[V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md](V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md)

