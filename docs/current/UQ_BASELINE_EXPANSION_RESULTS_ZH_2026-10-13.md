# 五数据集扩展基线实验：结果与完成状态

最新本地快照采集于 2026-10-14 10:11（北京时间）。本文不改写旧 sealed 结果，不根据新测试结果选择方法。

## 完成范围

- ArXiv、SportsFit、BookHis、Cora、Citeseer 的主扩展面板已全部完成，50/50 个方法/数据集组合，主队列退出码为成功、failures=0。
- 每个数据集都有 10 行配置：Unified-Q-EF、CaT-CCTP、GECC-hard、Random、K-center、Cluster-medoid、Herding、PUMA-inspired、GCond-x core、SFGC bounded core。即主方法加 9 个对照配置，不是 9 篇外部论文复现。
- 包括额外参考分支在内，已完成 57/59 个注册组合。额外七行是 BookHis GECC-soft 一行，以及 ArXiv、SportsFit、BookHis 每图各两行 TAGSAM bounded core 和 STAD local Qwen。ArXiv/SportsFit 的 GECC-soft 未完成，停止重试，不填零或借用 BookHis 分数。
- Unified-Q-EF、CaT-CCTP、GECC-hard 在三个大图上的 3 顺序 × 3 构造种子 × 3 Student seeds 确认矩阵已完整，每个方法/数据集有 27 次 Student 评估。

## 统一协议

主扩展表使用 order=9101、construction seed=271006、Student seeds=150/151/152；Student 为共同 Prompt GCN，MiniLM 文本编码器，2048 steps/batch 24。合法构造池、阶段划分和 report-dev/sealed test 掩码一致。

凝聚比例为累计合法构造池的 5%，允许整数舍入和类别覆盖下限。它是固定比例，而不是固定行数或固定字节内存。只读取当前增量 raw，历史 synthetic prefix 冻结。无边 memory 的 Student 推理仍使用共同原始 transductive 图。Cora/Citeseer 沿用项目 canonical split，不能按 Planetoid 每类 20 标签的论文分数解释。

所有主扩展配置共用 Student 训练预算；构造优化的迭代数、教师数量和原生模块各不相同，见适配说明，不能称为完全相同的构造计算预算。

## 核心结果

下表为主扩展单元的三个 Student seed 均值，单位为百分数。单元稳定性结论使用后面的九单元配对表。

| 数据集 | Unified-Q-EF F1 / Acc | CaT-CCTP F1 / Acc | GECC-hard F1 / Acc | EF−GECC F1 点 |
| --- | --- | --- | --- | --- |
| ArXiv | 53.20 / 71.05 | 36.35 / 61.59 | 52.98 / 71.15 | +0.21 |
| SportsFit | 80.35 / 86.32 | 77.50 / 86.58 | 78.85 / 88.71 | +1.50 |
| BookHis | 49.24 / 82.01 | 31.11 / 79.70 | 45.71 / 81.03 | +3.53 |
| Cora | 87.24 / 87.82 | 85.94 / 87.45 | 85.37 / 86.59 | +1.87 |
| Citeseer | 70.14 / 74.40 | 69.74 / 75.34 | 67.15 / 73.25 | +2.99 |

PUMA-inspired 在 SportsFit、Cora、Citeseer 的 F1 为 81.00、87.60、70.42，高于该主扩展单元的 EF；这些结果已原样保留。它只是项目伪教师适配参考，不能称为完整官方 PUMA。

| 数据集 | 九单元 EF−GECC F1 点 | 95% paired t CI | 正单元比例 | 最差配对差 |
| --- | --- | --- | --- | --- |
| ArXiv | +0.22 | [−0.34, +0.79] | 4/9 | −0.47 |
| SportsFit | −0.49 | [−1.50, +0.52] | 2/9 | −2.14 |
| BookHis | +3.47 | [+2.20, +4.74] | 9/9 | +0.09 |

因此，最可靠的外部强对照收益集中在 BookHis。ArXiv 与 GECC-hard 接近，SportsFit 没有稳定领先且 Accuracy 较低；不能由单一扩展单元的正差值声称普遍 SOTA。区间描述本次九个顺序/构造单元，不是跨数据集保证，也不能用新增对照替代共享 Q 的组件归因证据。

## 外部方法适配与有效性限制

完整报告包含“原方法—改动内容”表，明确区分公开机制适配、固定官方提交的核心适配、选样对照和启发式参考。

- CaT-CCTP、GECC-hard：保留相应特征凝聚机制，并附着同类特征最近 5 个当前文本，合计 320 字符。不是官方原任务分数复现。
- GCond-x：固定官方 SGC/梯度距离函数，1000 次外层特征更新，无 inner training，单位合成邻接，统一 nearest-5 文本。
- SFGC：固定官方 GCN/ReparamModule 与多步教师轨迹匹配，5 experts × 800 epochs、200 outer × 200 unroll；相对官方示例缩减构造预算，标为 bounded core。
- GECC-soft：GraphSlim 原生 FCM，fuzziness=1.3、20 restarts、300 iterations、tolerance=1e-8；与 hard 分支分开列出。
- TAGSAM：保留文本 selector/response，构造预算从 5000×15 缩至 100×10；列为计算预算受限参考。
- STAD：保留拓扑与伪标签流程，10% labels/10% 内部验证/80% pseudo，本地 Qwen2.5-1.5B；监督和拓扑不同，单列参考。

BookHis 最后一个阶段只有一个当前类别。GCond-x/SFGC 的本次代理仅含当前类别，其单输出交叉熵没有判别性特征梯度：两个配置的该阶段特征更新范数均为 0，已记录 11/11 个特征梯度样本为 0。此前多类别阶段有更新。这个适配限制必须随结果一起披露，不能据其低分声称完整官方方法差。原始配置和结果均保留，没有根据测试分数偷偷修改构造目标。

## 未完成的注册组合

| 方法 | SportsFit | ArXiv |
| --- | --- | --- |
| GECC-soft | 未完成，停止重试 | 未完成，停止重试 |
| TAGSAM bounded core | 已完成；受限参考 | 已完成；受限参考 |
| STAD local Qwen | 已完成；受限参考 | 已完成；受限参考 |

57/59 是已完成的结果覆盖率，不是十篇独立论文的比较数。受限参考不能与十配置主面板合并为同构造预算排名；GECC-hard 是公开机制适配，不能称为原生 soft FCM 的官方复现。本轮只整理已存快照，不启动远端构造，也不删除用户文件。

## 核验与产物

独立复算直接读取快照中的 Student 分数，不调用汇总程序的统计函数；最新快照的 `independent_verification.json` 已通过。该核验保证已存指标的汇总算术一致，未重新读取预测或混淆矩阵，因此不声称重新验证底层预测。

F1、Accuracy 和九单元配对区间的 PNG/PDF 已生成并目视检查。表格同时提供 Old/New F1、阶段均值、构造时间、CUDA peak、实际凝聚比例和来源哈希。峰值是 PyTorch allocated peak，包括共同数据张量；并发环境中的 wall time 不能用作精细的隔离吞吐排名。

- [中文分数据集总表与完整适配说明](../../results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/summary_zh.md)
- [结果索引及五份 CSV](../../results/uq_baseline_expansion_20261013/README.md)
- [独立核验记录](../../results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/independent_verification.json)
- [全部已有结果工作簿](../../results/all_results_aggregate_20261013/all_results_aggregate_20261013.xlsx)：首页将五图 50 行、受限参考 7 行、三大图九单元确认及历史 full-history 分区；604 行跨协议详细结果另置各 sheet，不跨协议排名。
- [聚合工作簿核验记录](../../results/all_results_aggregate_20261013/verification.json)：114,737 项检查通过，核对 57 行快照登记哈希、85 个源文件及导出单元格；这是结果与来源一致性检查，不是重新训练。

本轮未混入历史 report-dev full-history 保留率，没有修改论文或推送 Overleaf。论文可以使用这组结果扩大数据集和机制覆盖；关于普遍优越性、官方 reproduction、共享 Q 的精度因果收益仍需按上述限制表述。
