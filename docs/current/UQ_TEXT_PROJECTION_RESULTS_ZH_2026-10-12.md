# Unified-Q-EF 文本投影对照与正式确认结果

更新时间：2026-10-13 00:10 快照，Asia/Shanghai。本文件记录已完成结果与实际缺项；新快照另存，既有 sealed 结果不覆盖。

## 当前结论

1. BookHis、WikiCS、SportsFit 的六组文本对照全部完成，共 18 个方法/数据集单元、54 次 Student 训练。
2. 仅改善软硬文本 cosine gap 或候选覆盖率，没有带来跨数据集一致的分类提升。BookHis 原始 Q-top1 优于两个新投影版本；SportsFit 的 coverage 仅增加 0.12 F1，medoid 下降 0.32 F1。
3. 正式多顺序确认中，BookHis 对 GECC-hard 的 Macro-F1 优势已有九个配对单元支撑；ArXiv 对 GECC-hard 接近持平。SportsFit 尚未完成九单元确认。
4. 本轮证据不支持把 Q-medoid/Q-coverage 升级为主方法，也不支持把共享 Q 的精度优越性写成已证实的原因。继续保留冻结的 Unified-Q-EF（Q-top1）。

## 一、六组对照的协议与方法边界

本批固定 order 9101、construction seed 271108，Student seeds 150/151/152。三次 Student 训练复用同一构造 memory；它们只反映训练重复性，不能代替三个独立构造或类别顺序。

- 累计固定凝聚率 5%，分母为排除 report-dev/test 后的合法构造池。
- current-delta-only，各方法仅访问当前原始增量及自己的冻结历史 synthetic prefix。
- 共同 MiniLM、类别 prompts、Prompt GCN Student、2048 训练步、batch 24；Q 优化 30 步，Sinkhorn 上限 4096。
- 全部为无显式合成边的 memory；训练及最终保存的硬文本来源一致，原始文本清理并按 320 字符处理后重新编码。
- 新投影规则只读取 report-dev 进行本轮比较，不读取或改写 sealed 表。
- 本轮仅评估最终阶段；阶段平均 F1 来自另行预注册的正式确认，不由本批推算。

| 名称 | 实际做法 | 对比类别 |
| --- | --- | --- |
| Unified-Q-EF / Q-top1 | 学习共享 Q，在每个同类 slot 选择 Q 权重最大的真实文本；特征保留原始 in-loop 构造 | 冻结主方法 |
| Q-medoid | 选择最接近 Q 加权文本中心的同类真实文本；训练硬前向和保存均采用该规则 | 内部文本投影改动 |
| Q-coverage | 同类候选 shortlist 32，在 slot 兼容性上加入权重 0.35 的语义覆盖收益，确定性贪心无放回选择 | 内部文本投影改动 |
| CaT-CCTP | CaT 风格分布匹配构造 X，graph-nearest top-5 真实文本附着 | 外部论文机制适配基线 |
| Adaptive coherent | 复用该单元完全相同的 CaT X，进行自适应 slot 的 coherent 文本处理 | 历史内部候选方法 |
| GECC-hard | GECC 类内硬分配池化，采用与 CaT 相同的 CCTP 文本附着 | 外部论文机制适配基线 |

Q-medoid/Q-coverage 同时改变构造时的硬文本前向和最终保存规则。因此 Q 与 X 可以随之变化，这不是固定同一 X 后只替换文本的事后对照。Coherent 的构造成本包含复用的 CaT 成本；不把内部候选方法当作外部论文基线。CaT/GECC 也不标为官方原协议 reproduction。

实际累计预算：WikiCS `[247,324,356,379,410]` / 8191 合法节点，最终 5.0055%；BookHis `[579,592,601,1429,1454]` / 29086，最终 4.9990%；SportsFit `[3817,4486,4812,5305,6057]` / 121137，最终 5.0001%。小幅偏差来自整数取整。

## 二、已完成的 report-dev 六组结果

表中 F1、Acc、Old/New 单位为百分比；F1 的 ± 是同一 memory 上三个 Student seeds 的样本标准差。每个方法目前仅一个 order/construction 单元，不能报告独立构造的显著性。

### WikiCS

| 方法 | Macro-F1 ± Student SD | Acc | Old F1 | New F1 | 构造时间 s | 唯一文本比例 |
| --- | --- | --- | --- | --- | --- | --- |
| Unified-Q-EF / Q-top1 | 80.49 ± 0.47 | 83.59 | 81.30 | 73.22 | 18.4 | 1.000 |
| Q-medoid | 81.56 ± 0.57 | 84.13 | 82.47 | 73.46 | 22.3 | 0.454 |
| Q-coverage | 81.18 ± 0.67 | 83.53 | 82.15 | 72.49 | 22.7 | 1.000 |
| CaT-CCTP | 81.15 ± 0.78 | 84.42 | 81.57 | 77.32 | 21.4 | 0.985 |
| Adaptive coherent | 81.53 ± 0.46 | 84.59 | 82.10 | 76.34 | 25.5 | 0.037 |
| GECC-hard | 80.23 ± 0.32 | 82.76 | 80.93 | 73.92 | 1.9 | 1.000 |

Q-medoid 相对原始 top1 为 +1.08 F1 / +0.54 Acc；Q-coverage 为 +0.69 F1 / -0.06 Acc。Q-medoid 与 coherent 的 F1 均值只差约 0.04，不能凭一个单元宣称超过。

### BookHis

| 方法 | Macro-F1 ± Student SD | Acc | Old F1 | New F1 | 构造时间 s | 唯一文本比例 |
| --- | --- | --- | --- | --- | --- | --- |
| Unified-Q-EF / Q-top1 | 48.50 ± 1.91 | 82.33 | 47.02 | 64.80 | 65.7 | 0.996 |
| Q-medoid | 48.29 ± 0.47 | 82.54 | 46.80 | 64.57 | 65.7 | 0.790 |
| Q-coverage | 47.08 ± 1.82 | 82.52 | 45.73 | 61.91 | 76.7 | 0.993 |
| CaT-CCTP | 30.40 ± 0.81 | 81.35 | 31.63 | 16.88 | 28.8 | 0.971 |
| Adaptive coherent | 26.37 ± 1.38 | 80.28 | 27.60 | 12.85 | 42.3 | 0.015 |
| GECC-hard | 44.00 ± 0.62 | 82.24 | 42.63 | 59.08 | 49.9 | 0.999 |

原始 top1 相对 GECC-hard 为 +4.50 F1 / +0.10 Acc。Q-medoid 相对 top1 为 -0.22 F1 / +0.20 Acc；Q-coverage 为 -1.42 F1 / +0.18 Acc。覆盖规则的收益没有转化为本单元分类提升。

### SportsFit

| 方法 | Macro-F1 ± Student SD | Acc | Old F1 | New F1 | 构造时间 s | 唯一文本比例 |
| --- | --- | --- | --- | --- | --- | --- |
| Unified-Q-EF / Q-top1 | 79.48 ± 1.63 | 84.70 | 77.14 | 92.36 | 176.5 | 1.000 |
| Q-medoid | 79.16 ± 1.21 | 85.76 | 76.93 | 91.44 | 180.8 | 0.645 |
| Q-coverage | 79.60 ± 0.84 | 86.02 | 77.18 | 92.86 | 445.4 | 0.975 |
| CaT-CCTP | 73.20 ± 2.10 | 85.28 | 69.76 | 92.11 | 182.0 | 0.863 |
| Adaptive coherent | 72.31 ± 2.39 | 85.29 | 68.63 | 92.57 | 228.5 | 0.004 |
| GECC-hard | 79.88 ± 0.94 | 89.06 | 77.33 | 93.94 | 871.1 | 0.888 |

Q-coverage 相对原始 top1 为 +0.12 F1 / +1.32 Acc，但构造时间为 2.52 倍；Q-medoid 为 -0.32 F1 / +1.06 Acc。coverage 仍比 GECC-hard 低 0.29 F1 / 3.05 Acc。三次 Student 重复不足以认定这些单单元差值稳定。

本单元 Q-top1 与 GECC-hard 的构造峰值分别为 24.08 GiB 和 0.98 GiB。GECC 的 871.1 s 中，阶段 0 用时 821.7 s；该数据与同机并行负载和初始化计算有关，不能据一个单元宣称 Q 全面优于 GECC 的效率。时间、峰值和准确度应分别报告。

## 三、文本诊断：为什么暂不升级

诊断汇总使用明确分母：slot 覆盖、来源熵和软硬 gap 按 slot 数加权；归一化 assignment 行熵按原始节点数加权；语义覆盖为各阶段各类别等权平均；唯一字符串比例在最终 memory 计算。

| 数据集 | 规则 | 硬 slot 覆盖 | 归一化 assignment 熵 | 软硬 cosine gap | 类语义覆盖 | 唯一字符串比例 |
| --- | --- | --- | --- | --- | --- | --- |
| WikiCS | top1 | 1.000 | 0.335 | 0.305 | 0.626 | 1.000 |
| WikiCS | medoid | 1.000 | 0.317 | 0.156 | 0.645 | 0.454 |
| WikiCS | coverage | 1.000 | 0.319 | 0.173 | 0.664 | 1.000 |
| BookHis | top1 | 1.000 | 0.135 | 0.446 | 0.446 | 0.996 |
| BookHis | medoid | 1.000 | 0.134 | 0.233 | 0.524 | 0.790 |
| BookHis | coverage | 1.000 | 0.133 | 0.238 | 0.526 | 0.993 |
| SportsFit | top1 | 0.9998 | 0.391 | 0.213 | 0.749 | 1.000 |
| SportsFit | medoid | 1.000 | 0.394 | 0.110 | 0.802 | 0.645 |
| SportsFit | coverage | 1.000 | 0.392 | 0.121 | 0.806 | 0.975 |

BookHis 的 medoid 将 gap 从 0.446 降到 0.233，但 F1 没有提高。coverage 同时改善 gap、语义覆盖且基本保持字符串多样性，F1 仍下降。因此这两个代理指标不是充分的下游质量目标；不能把材料化误差直接认定为主要瓶颈。

SportsFit 也显示相同限制：medoid 的 gap 从 0.213 降到 0.110，F1 反而下降；coverage 将 gap 降到 0.121、语义覆盖提高到 0.806，F1 仅微增。硬 slot 覆盖的定义是原始节点对 slot 做 argmax 后，至少被选中一次的 slot 比例；SportsFit top1 为 6056/6057，不表示保存 memory 少一行。软分配仍覆盖所有 slot，实际 memory 为 6057 行。

top1 的唯一文本比例已接近 1，也没有出现大量同类 slot 共享一个字符串的现象。Coherent 在相同当前 Student 和 CaT X 下没有重现早期的大幅收益；历史不同 Student/协议的绝对分数不能直接与本表混合。

## 四、独立于文本候选试验的正式 sealed 确认

这一部分为实验前冻结的既有版本，orders 9101/9102/9103 × construction seeds 271006/271108/271109，各单元平均 Student seeds 150/151/152。± 为单元均值的样本标准差，不将 27 次 Student 读数当独立样本。

| 数据集 | 方法 | 完成单元 / 9 | Sealed F1 ± Cell SD | Acc | Old F1 | New F1 | 阶段平均 F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ArXiv | Unified-Q-EF | 9 | 52.91 ± 0.73 | 71.15 | 52.25 | 54.27 | 62.23 |
| ArXiv | CaT-CCTP | 9 | 34.98 ± 1.79 | 60.41 | 33.77 | 37.50 | 47.36 |
| ArXiv | GECC-hard | 9 | 52.68 ± 0.45 | 70.94 | 51.45 | 55.24 | 62.34 |
| BookHis | Unified-Q-EF | 9 | 48.81 ± 1.57 | 81.05 | 48.39 | 53.45 | 51.82 |
| BookHis | CaT-CCTP | 9 | 29.16 ± 1.05 | 79.23 | 29.16 | 29.09 | 34.87 |
| BookHis | GECC-hard | 9 | 45.34 ± 0.74 | 81.26 | 45.48 | 43.86 | 50.43 |
| SportsFit | Unified-Q-EF | 7 | 79.84 ± 0.41 | 85.11 | 78.96 | 84.70 | 81.67 |
| SportsFit | CaT-CCTP | 7 | 74.91 ± 2.21 | 85.71 | 73.53 | 82.51 | 78.01 |
| SportsFit | GECC-hard | 6 | 80.16 ± 1.24 | 89.04 | 78.58 | 88.84 | 81.10 |

SportsFit 行均值来自不同数量的完成单元，不能直接相减解释为配对优势。正确比较只用相同 order/construction 单元：

| 配对 F1 差：EF 减基线 | 单元数 | 均值 pp | 95% paired t CI | 正提升单元 | 最差差值 pp |
| --- | --- | --- | --- | --- | --- |
| ArXiv − CaT-CCTP | 9 | +17.93 | [16.81, 19.05] | 9/9 | +15.92 |
| ArXiv − GECC-hard | 9 | +0.22 | [-0.34, 0.79] | 4/9 | -0.47 |
| BookHis − CaT-CCTP | 9 | +19.66 | [18.38, 20.93] | 9/9 | +17.63 |
| BookHis − GECC-hard | 9 | +3.47 | [2.20, 4.74] | 9/9 | +0.09 |
| SportsFit − CaT-CCTP | 7 | +4.93 | [2.86, 7.01] | 7/7 | +2.22 |
| SportsFit − GECC-hard | 6 | -0.31 | [-1.96, 1.34] | 2/6 | -2.14 |

BookHis 的 GECC 对比支持稳定的 Macro-F1 增益，但 Acc 差为 -0.21 pp，CI [-0.75, 0.33]，不能宣称两种指标均显著改善。SportsFit 对 GECC 的 Acc 差为 -4.14 pp，CI [-5.54, -2.75]，这是必须保留的负结果。ArXiv 与 GECC 的阶段平均 F1 也接近持平。

## 五、共享 Q 的归因边界

此前单顺序、三个构造种子的 report-dev shared−independent F1 差为 ArXiv -0.484、BookHis -2.423、SportsFit -0.580，均 0/3 正提升。去掉 residual 后分别为 +0.036、-1.189、+0.182；BookHis 的负差 CI 为 [-1.831,-0.546]。这些结果不被新的文本试验覆盖。

共享 Q 的可支持作用是统一图文来源、减少分配参数和资源，而不是已证明的精度主因。实测 shared/independent 构造时间与峰值为：ArXiv 241.3/292.5 s、12.11/16.30 GiB；BookHis 66.9/81.6 s、2.83/3.31 GiB；SportsFit 177.9/226.7 s、25.01/32.92 GiB。时间来自共享服务器，不等同独占设备效率基准。

soft-stored 与匹配 hard-stored 的单单元差为 ArXiv +0.506、BookHis +4.846、SportsFit +0.744 F1。这说明软文本存储仍有可研究空间；本轮真实文本投影规则尚未恢复同等收益。soft-stored 不可由参考字符串重新编码精确恢复，不能与可审计硬字符串完全等同。

## 六、完整缺项与结果来源

正式确认的计划网格包含 81 个方法×数据集×顺序×构造单元。00:10 快照完成 74，部分完成 1（SportsFit o9103/c271006 的 GECC），未启动 6（SportsFit o9103/c271108、o9103/c271109，各三方法）。不能仅依据四个部分结果文件声称只剩四项。

三图六组文本对照已经结束；正式剩余确认在 GPU1 的现有顺序队列继续。原生机制基线队列中，GECC-soft SportsFit、TAGSAM-selector-response BookHis、STAD BookHis 尚未完整完成，不纳入已完成强基线结论。官方核心保真度及共同协议适配偏差仍须分别说明。

- 文本原始快照：[uq_q_coverage_snapshot_20261013_000839](../../results/uq_q_coverage_snapshot_20261013_000839/)
- 文本表格、诊断与 PNG/PDF：[summary.md](../../results/uq_q_coverage_report_20261013_000839/summary.md)
- 正式确认原始快照：[snapshot.json](../../results/uq_final_evidence_snapshot_20261013_001014/snapshot.json)
- 正式统计与可视化：[summary.md](../../results/uq_final_evidence_report_20261013_001014/summary.md)
- 完整缺项身份：[confirmation_completion.csv](../../results/uq_final_evidence_report_20261013_001014/confirmation_completion.csv)
- 预注册边界：[UQ_FINAL_EVIDENCE_PREREGISTRATION_ZH_2026-10-12.md](UQ_FINAL_EVIDENCE_PREREGISTRATION_ZH_2026-10-12.md)

验证：文本汇总的 6 项测试已通过；正式汇总新增未启动单元、单侧评分缺失及未注册单元检查后，9 项测试通过。独立用 PowerShell 从原始每 Student 记录复算文本对照的 180 个均值/标准差与正式确认的 960 个最终指标均值，最大误差为 0；12 个主比较的配对均值/CI 最大误差约 6.3e-10。三图 F1/Acc 图、文本诊断图和正式配对图均已视觉核验。

## 七、本轮决策与下一步

本轮不根据 WikiCS 的局部提升改选主方法，不将 Q-coverage 的 SportsFit 单单元微增当作稳定优势。冻结的 Q-top1 继续完成预注册九单元确认；两个文本投影版本作为已完成的机制分析，保留其正负结果。

先完成 SportsFit 剩余确认及高保真相关基线，再决定下一轮单点优化。若继续优化文本，应在 report-dev 上检验节点级图文配对或 Student 判别目标，而不是直接增加通用语义覆盖奖励；本轮尚未证明这种新目标一定有效，也未启动测试集驱动的选型。共享 Q 的资源收益、硬文本可重编码与历史访问限制可以构成方法贡献，但不能替代对强基线的质量证据。

论文可以采用的结果表述是：在当前增量访问和 5% 合法池预算下，Unified-Q-EF 在 BookHis 上相对 GECC-hard 取得稳定 Macro-F1 增益，在 ArXiv 上具有相近效果，相对 CaT-CCTP 有明显增益。不能据此扩展为全图普遍 SOTA，或认定共享 Q 是精度提升的已证实原因。新文本投影的诊断改善与分类效果不一致，应作为机制分析保留。
