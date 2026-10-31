# X-conditioned Slot Partition 变体实验报告

> 本轮比较四个版本：Current、ST-Sinkhorn、coherence anchor、ST+anchor+hard consistency。所有变体在同一远程进程中使用相同的 base+4、5% cumulative memory、Student seeds 100--105、construction seed 270927、current-delta-only 和 edge-free memory 协议。

## 1. 三图结果

|数据集|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|Worst stage|构造秒数|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|arxiv|Current XSP|22.60|34.10|34.77|25.03|8.87|22.60|320.94|
|arxiv|XSP + ST-Sinkhorn|23.44|37.06|34.73|25.91|9.43|23.44|329.85|
|arxiv|XSP + coherence anchor|22.32|34.15|34.39|24.84|8.04|22.32|325.75|
|arxiv|XSP + ST + anchor + hard consistency|23.62|38.69|34.52|26.50|7.33|23.62|1008.12|
|bookchild|Current XSP|12.72|12.79|36.70|13.44|7.67|12.72|98.06|
|bookchild|XSP + ST-Sinkhorn|14.10|14.43|39.59|14.82|9.01|14.10|105.04|
|bookchild|XSP + coherence anchor|13.58|12.60|37.03|14.41|7.76|13.58|105.03|
|bookchild|XSP + ST + anchor + hard consistency|14.93|14.84|35.74|15.81|8.72|14.78|246.05|
|sportsfit|Current XSP|26.74|30.08|50.97|27.24|24.02|26.74|403.48|
|sportsfit|XSP + ST-Sinkhorn|30.55|31.61|55.44|32.03|22.43|30.55|397.40|
|sportsfit|XSP + coherence anchor|29.09|33.01|54.29|29.96|24.30|25.20|400.21|
|sportsfit|XSP + ST + anchor + hard consistency|31.52|34.51|53.34|33.09|22.84|27.03|962.62|

## 2. 等权三图汇总

|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|Worst stage|构造秒数|
|---|---:|---:|---:|---:|---:|---:|---:|
|Current XSP|20.69|25.66|40.81|21.90|13.52|20.69|274.16|
|XSP + ST-Sinkhorn|22.69|27.70|43.26|24.25|13.62|22.69|277.43|
|XSP + coherence anchor|21.66|26.58|41.90|23.07|13.37|20.37|277.00|
|XSP + ST + anchor + hard consistency|23.36|29.35|41.20|25.13|12.96|21.81|738.93|

## 3. 相对 Current 的配对差值

|数据集|方法|F1差值|Accuracy差值|Stage-average差值|Old F1差值|New F1差值|构造成本倍数|
|---|---|---:|---:|---:|---:|---:|---:|
|arxiv|XSP + ST-Sinkhorn|+0.83|+2.95|-0.04|+0.88|+0.56|1.03x|
|bookchild|XSP + ST-Sinkhorn|+1.37|+1.63|+2.89|+1.38|+1.33|1.07x|
|sportsfit|XSP + ST-Sinkhorn|+3.81|+1.53|+4.47|+4.79|-1.59|0.98x|
|arxiv|XSP + coherence anchor|-0.28|+0.04|-0.38|-0.19|-0.83|1.01x|
|bookchild|XSP + coherence anchor|+0.86|-0.19|+0.33|+0.97|+0.09|1.07x|
|sportsfit|XSP + coherence anchor|+2.35|+2.93|+3.32|+2.72|+0.28|0.99x|
|arxiv|XSP + ST + anchor + hard consistency|+1.02|+4.59|-0.25|+1.47|-1.54|3.14x|
|bookchild|XSP + ST + anchor + hard consistency|+2.20|+2.05|-0.95|+2.37|+1.05|2.51x|
|sportsfit|XSP + ST + anchor + hard consistency|+4.77|+4.44|+2.37|+5.85|-1.18|2.39x|

## 4. 与公开/适配基线的 F1 对照

|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|
|---|---:|---:|---:|---:|---:|
|Append-edge control|19.44|23.51|38.06|20.72|11.83|
|X+P control|21.59|27.99|39.75|23.09|12.82|
|Random real-node coreset|19.21|22.62|40.44|20.52|11.55|
|K-center coreset|20.45|24.87|38.17|22.04|11.21|
|Medoid coreset|19.35|22.06|39.70|20.57|12.27|
|CaT-TAG CCTP top-5|17.51|41.13|19.19|18.89|9.46|
|GECC-TAG hard|20.31|24.14|38.13|21.77|11.76|
|CaT-TAG class-shared payload|25.69|42.36|31.70|27.69|14.41|
|Current XSP|20.69|25.66|40.81|21.90|13.52|
|XSP + ST-Sinkhorn|22.69|27.70|43.26|24.25|13.62|
|XSP + coherence anchor|21.66|26.58|41.90|23.07|13.37|
|XSP + ST + anchor + hard consistency|23.36|29.35|41.20|25.13|12.96|

公开基线的统一池化结果来自 `public_baselines_large_corrected_20261003_remote`。其协议字段与本轮一致，但它们不是同一个代码快照中的同轮重跑，因此这里用于方向性比较；正式论文主表前仍应在冻结代码快照下重跑关键基线。

## 5. 结果解释

1. ST-Sinkhorn 是最稳定的低成本改动：三图平均 Final F1 相对 Current 为 +2.00 pp，Old F1 为 +2.35 pp，New F1 基本不变（+0.10 pp）。这说明让 assignment plan 对 X 可微，主要改善了旧类保持和整体可学习性，但没有解决新类塑性瓶颈。
2. coherence anchor 没有带来稳定增益：三图平均仅 +0.97 pp，且 Arxiv 下降 0.28 pp、New F1 略降。仅按 assignment entropy 对低置信 slot 拉向类别中心，容易抹平类内多模态，暂不应单独进入主方法。
3. Full 在三图平均 Final F1 达到 23.36，相对 Current +2.67 pp，Accuracy +3.69 pp，Old F1 +3.23 pp；但 New F1 -0.56 pp，构造时间约为 Current 的 2.70x。它改善的是 hard payload 与 soft assignment 的一致性以及旧类保持，不是新类塑性。
4. Full 的收益并不等于已经证明主方法成立：它在 Arxiv 仍低于 CCTP top-5/class-shared，在 SportsFit 仍低于 class-shared；相对 k-center 和 GECC 则多数数据集有优势。当前更准确的结论是：ST 是低成本有效组件，hard consistency 是有代价的候选组件，Anchor 不值得保留。
5. 本轮 Current 与历史单独 XSP 运行分数不完全一致，因此历史 XSP 不能作为严格复现基线。后续必须冻结代码、缓存、Student 数值设置后重跑 Current 与关键公开基线；本轮变体之间的同轮配对差值仍然有效。

## 6. 审计与实现修正

本轮四个版本均保持 `historical_raw_reads=0`、`old_memory_updated=False`、`topology_updated=False`，文本是按 X slot 的 transport cluster 材料化，三图没有出现单一 class payload 广播塌缩。结果生成后发现 slot audit 中 entropy/effective-source 的统计漏除了 Sinkhorn 列质量；该问题不影响训练或 F1，只影响诊断字段。代码已修正为先归一化列质量，并补充了测试；本轮已有 JSON 的该两列不用于结论。

## 7. 下一步

- 保留 Current、ST、Full 三个版本做一次独立 construction-seed 确认，淘汰 Anchor；不要再叠加新的 selector 或拓扑模块。
- 在同一冻结代码快照下重跑 CCTP、GECC、k-center 和 class-shared 的关键三图结果，先解决 Current 历史分数不复现的问题。
- 对 Full 做成本优化实验：只保留最终一次 hard refresh、缓存 MiniLM 编码，确认收益是否来自 hard consistency 本身，而不是重复编码次数。该实验只改实现成本，不改变目标函数。
- 新类塑性仍是主要未解决问题。下一阶段应直接分析 slot 数量、assignment mass 和新类 margin 的关系，不再依赖 anchor 强行收缩类内文本。
- 在上述确认完成前，不把 Full 冻结为最终主方法；论文候选定位应为“ST-Sinkhorn 为有效组件，Full 为高成本增强候选”。
