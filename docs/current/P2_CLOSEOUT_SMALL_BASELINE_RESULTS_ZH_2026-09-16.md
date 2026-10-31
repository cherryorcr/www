# P2 收尾与 Cora/Citeseer 对照结果报告

日期：2026-09-16。状态：本轮训练和验收完成；仅小图，未启动大图。

## 1. 最重要的结论

P2 收尾完成：没有新文本模块达到稳定晋升标准，当前主线继续使用字符版 center concat。本次补齐两张小图、18 个实验组、3 个阶段、6 个 student seeds，共 **648 次正式下游训练**，另有 6 次共享 delta teacher 训练。检查点、参数继承、预算、源数据访问、历史记忆与指标审计 **0 error**；7 项新增测试通过。TAGSAM 适配另外包含双层优化内训，不能把 648 当作所有优化计算的总次数。

**v4 在 Citeseer 有较明确优势，但在 Cora 还不是最优方案。** Cora 上，完整数据加 selector-dev 检查点选择达到 51.31，v4 初始化后增量微调达到 53.41，均高于 v4 的 49.85；相关配对区间仍跨零，不能直接晋升微调策略。Citeseer 上 v4 的 60.77 则高于这两个参考的 55.60 和 52.47。

**最终分数与持续过程表现必须分开。** Cora 的 k-center、按类 k-center 和 self-loop 构造在阶段平均上显著高于当前 v4 的数值；因此不能仅凭终阶段领先就宣称解决持续学习问题。

**本轮外部方法结论是适配负结果，不是击败官方 SOTA。** TAGSAM 响应匹配和内部 row/column ASM 在本次有限预算、共同 GraphCLIP 学生、当前 delta teacher 的适配中没有带来优势。没有完成官方完整训练配置及其它最新 TAG/持续凝聚方法的全面复现。

## 2. P2 如何收尾

已完成的 P2 证据包含 2052 次归档 student fit：严格 token 矩阵、独立 seeds、DeepSets、batch/rate/排列扰动和指标诊断。完整性通过，但 `independent_confirmation_pass=[]`、`large_graph_gate_passed=false`。不再追加文本候选来寻找最高 dev 分数。

本轮保留两个独立基线：

- **v4 主线**：历史 `joint_g5_c320` 中心文本拼接，使用真实 v4 response allocation 后的封存合成图。
- **concat64 控制**：每行最多 64 个内容 tokens，保持旧 dispersion 构造的 X 和边不变，仅替换文本。它不是重新运行 v4 selector 得到的新方法。

旧 `comparison_small_cora_2026_09_16.json` 和 `comparison_small_citeseer_2026_09_16.json` 不再作为有效主表。其 seen-prompt 累计指标把任务 F1 平均，且 `sca` 并不等于实际 v4。本轮保存独立结果，没有覆盖旧证据。

详见 [P2 原验收记录](P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md) 和 [本轮冻结协议](P2_CLOSEOUT_SMALL_COMPARISON_PROTOCOL_ZH_2026-09-16.md)。

## 3. 公平性设置

| 项目 | Cora | Citeseer |
|---|---|---|
| 类别阶段 | [0,1] / [2,3] / [4,5,6] | [0,1] / [2,3] / [4,5] |
| 每阶段合法 raw pool 节点 | 501 / 858 / 536 | 778 / 953 / 498 |
| 累计合成预算 | 25 / 67 / 94 | 38 / 86 / 111 |
| 每阶段 canonical student steps | 400 / 1072 / 1504 | 608 / 1376 / 1776 |

共同设置：pool 累计 5% fixed-rate；两层 GCN，hidden/output384；冻结 MiniLM384；可训练文本投影与 logit scale；双向 InfoNCE；Adam1e-3；batch16；student seeds100..105。每步全训练图 forward，采16个图文对计算 loss。合成学生每阶段 scratch，微调组明确例外。

selector-dev 为 original_id%10=1，report-dev 为3，test 为2，其余为凝聚训练 pool。按类合成与按类 k-center 使用 pool 类别标签，因此整条凝聚 pipeline 不能称 label-free；student 不使用节点分类标签训练。

完整数据组的“完整”指 **所有已见类别中的合法训练 pool**，最终为1895/2229行，不包含留出文本训练目标。训练图为这些节点的真实诱导图。评估始终使用完整真实图 X/A，因此包含未来与留出节点的无标签输入，是 transductive setting，不是严格时间隔离。

本轮没有评估 test 标签。不过早期项目文档已记录开发阶段访问过这些小图的 test，不能将其包装成从未使用过的最终盲测。

## 4. 最终阶段结果

下表为 report-dev Macro-F1，均值 ± 6 个 student seeds 的总体标准差，单位为百分点。最后阶段全部类别已经出现，all-class 与 seen-class prompt 的分数完全一致。

| 方法 | Cora | Citeseer |
|---|---:|---:|
| dispersion synthetic append | 49.85 ± 3.00 | 59.17 ± 1.57 |
| **当前 v4** | **49.85 ± 3.00** | **60.77 ± 1.64** |
| P2 concat64 控制 | 50.64 ± 1.45 | 56.03 ± 1.04 |
| 随机真实节点 | 43.47 ± 2.48 | 49.79 ± 2.47 |
| k-center | 46.35 ± 4.36 | 55.34 ± 1.11 |
| 相同按类配额的 k-center | 45.09 ± 2.86 | 53.40 ± 0.82 |
| herding | 37.78 ± 4.36 | 55.85 ± 3.76 |
| k-means medoid | 47.50 ± 2.62 | 49.46 ± 2.49 |
| 旧合成图 + 新 delta 重新聚类 | 49.70 ± 1.21 | 54.24 ± 1.91 |
| 全历史 raw 重凝聚，按类 dispersion | 47.21 ± 2.06 | 48.28 ± 3.48 |
| 完整数据训练，与合成图相同步数 | 47.93 ± 4.10 | 49.89 ± 2.17 |
| 完整数据训练，固定8192步终点 | 46.26 ± 2.73 | 52.33 ± 1.75 |
| 完整数据训练，selector-dev 选检查点 | 51.31 ± 2.89 | 55.60 ± 1.92 |
| raw 初始化，之后仅新 delta 微调 | 48.10 ± 2.78 | 51.13 ± 2.23 |
| v4 初始化，之后仅新 delta 微调 | 53.41 ± 1.85 | 52.47 ± 1.46 |
| 相同 synthetic X/text，仅 self-loop | 49.81 ± 3.01 | 56.14 ± 1.19 |
| TAGSAM self-loop 适配 | 39.81 ± 2.26 | 47.56 ± 4.57 |
| TAGSAM semantic-append 适配 | 32.84 ± 3.13 | 30.49 ± 4.04 |
| 修正后的 row/column ASM 内部控制 | 42.30 ± 1.83 | 46.42 ± 1.93 |

前三个完整数据/全历史组属于放宽历史访问的离线参考，不是符合历史不可回读要求的持续方法。“完整数据训练”也不是数学上界：相同学习率、batch和实例对比目标并不保证完整数据达到该 backbone 的最优分类性能。

微调组继承前一阶段参数、每阶段重置 Adam、只训练当前 raw delta，不回读历史 raw 图，属于合法持续对照。它额外保留跨阶段模型参数，不能把这项持久状态当作零 memory 成本。coreset 组是真实节点对照；本方法、self-loop 和响应匹配组的节点仍为合成节点。

![终阶段对比](../../results/small_comparison_summary_2026_09_16/final_comparison.png)

## 5. 持续表现与旧类/新类差距

本轮累计 F1 在全部已见评价样本上计算。旧类/新类 F1 也使用同一个已见样本集合上的 per-class F1，所以会计入跨任务 false positives；与旧报告中按真实任务切片计算的 old/new 数值不能直接横比。

| 方法 | Cora 阶段平均 all / seen | Citeseer 阶段平均 all / seen |
|---|---:|---:|
| 当前 v4 | 53.08 / 60.72 | 64.82 / 72.85 |
| k-center | 59.86 / 69.75 | 54.18 / 65.56 |
| 按类 k-center | 67.13 / 72.20 | 60.18 / 67.74 |
| self-loop synthetic | 63.31 / 73.16 | 63.21 / 74.39 |
| 完整数据 + selector checkpoint | 59.84 / 69.48 | 60.98 / 68.66 |
| v4 初始化 + 新 delta 微调 | 56.57 / 62.56 | 56.10 / 65.59 |

因此：Cora 的主要问题不只是最终新类差，也包括早期累计表现。Citeseer 的最终优势较清楚，但 seen-prompt 阶段平均仍有 self-loop 对照更高，不能宣称所有持续指标都最优。

| 方法 | Cora 最终旧 / 新类 | Citeseer 最终旧 / 新类 |
|---|---:|---:|
| 当前 v4 | 65.44 / 29.06 | 68.00 / 46.29 |
| 完整数据 + selector checkpoint | 60.85 / 38.59 | 63.90 / 39.01 |
| v4 初始化 + 新 delta 微调 | 68.83 / 32.84 | 60.76 / 35.87 |
| self-loop synthetic | 67.76 / 25.88 | 64.05 / 40.33 |

Cora 的 full-data 参考新类达到38.59，而 v4 只有29.06，这个约9.53点差距值得优先研究。微调组在 Cora 的旧/新类均值都高于 v4，但在 Citeseer 都下降，说明持续模型更新不是普适替代品。

早期 prompt 竞争范围对分数影响很大。本轮同时保留 all-class 和 seen-class 口径，不能在两者之间挑有利结果。不同阶段的 precision 支持集合也会改变，因此阶段下降不应全部解释为纯参数遗忘。

![阶段曲线](../../results/small_comparison_summary_2026_09_16/stage_curves.png)

## 6. 哪些差异比较可靠

以下是同一6个初始化的配对差值，均为 **v4 减去对照**，95% t 区间仅反映 student 初始化不确定性，未覆盖类别顺序、凝聚随机种子或多重比较。

| 对照 | Cora 差值 [95%区间] | Citeseer 差值 [95%区间] |
|---|---:|---:|
| dispersion | 0.00 [0.00, 0.00] | +1.59 [-1.51, 4.69] |
| k-center | +3.51 [-2.24, 9.25] | +5.43 [3.16, 7.70] |
| 按类 k-center | +4.76 [2.84, 6.68] | +7.37 [4.60, 10.13] |
| 完整数据 + selector checkpoint | -1.46 [-6.16, 3.24] | +5.16 [2.34, 7.98] |
| v4 初始化 + 新 delta 微调 | -3.55 [-8.25, 1.14] | +8.30 [5.28, 11.32] |
| self-loop synthetic | +0.04 [-2.61, 2.70] | +4.62 [2.11, 7.13] |

v4 在 Citeseer 上相对简单 dispersion 的额外收益尚无充分统计证据。不能把 v4 对其它较弱基线的优势都归因于 response selector；相当部分优势已经来自基础构造。

## 7. TAGSAM 比较的具体范围

从官方仓库 commit `d812f06fd16db3cd4d497804a8e504d2c738faa8` 直接加载 `wBCELoss`，校验源文件与 git 原文一致。保留合成 X 的双层响应匹配和可学习内层图/文学习率；固定文本，旧合成块不可写。teacher 每阶段只在合法新 delta 上训练4096步，外层在该真实诱导图上的至多256个参考节点匹配响应。

适配后使用共同 GCN/MiniLM、dispersion 初始化、中心拼接、固定100外步和10内步。分别比较 self-loop 与当前 semantic append。最终评价统一 canonical Adam，不采用蒸馏出的学习率。额外 row/column-softmax ASM 是内部机制控制，不能列为另一篇外部 TAG 方法。

与官方仍不同的部分包括 backbone、文本选择器、静态/持续访问规则、参考采样和优化预算。官方默认5000外步、15内epoch，本轮不能代表其充分调优性能。现有结果只说明这种响应优化不能直接迁移为当前主线的优势，不支持宣称 TAGSAM 本身无效或本方法达到 SOTA。

## 8. 计算成本与实现发现

| 成本，两个数据集分别统计 | Cora | Citeseer |
|---|---:|---:|
| v4 本轮18次下游训练总steps | 17,856 | 22,560 |
| v4 已归档 selector fit 次数 | 351 | 297 |
| v4 已归档 selector 总steps | 120,330 | 110,502 |
| full-data long 18次训练总steps | 147,456 | 147,456 |

复用 v4 合成图是为了忠实比较已选定主线，不意味着其 selector 成本为零。response selector 成本明显高于一次小图下游评估，且 Cora 最后仍回退到 dispersion。需要把收益与完整构造成本共同报告。

raw、synthetic 和双层优化的一步所处理节点数不同，steps 不能直接等同 FLOPs。逐fit耗时和峰值显存在 CSV/JSON 中，但远程 GPU 有共享工作负载，本轮 wall time 不用于发表严格加速比。

复现还发现一个数值问题：将 canonical `scale * g @ t.T` 改写成数学等价的 `scale * (g @ t.T)` 会改变浮点舍入并放大到最终分数。首轮新 runner 因回归失败而停止并保留；正式 r2 恢复原操作顺序，经过参数逐位一致性测试和三条历史基线回归。首轮数值不并入本表。

## 9. 后续优先级

1. **先复核下游训练策略与新类塑性。** 在两个小图上使用独立 student seeds、不同类别顺序，固定比较 scratch-memory、delta fine-tune、memory replay + delta fine-tune。检查 Cora 的新类差距与 Citeseer 的反向结果；先不给任何一路自动晋升。
2. **P3 优先做结构控制。** 固定 X/text/预算，比较 self-loop、仅新块内部边、无跨边、当前跨边、质量受控跨边。旧旧边不改并不等于旧节点消息传递不变，跨边仍会改变节点度和归一化；需要验证这一机制，而不是继续盲目增加边。
3. **收紧 response selector。** 将正确的全局支持 old/new 指标用于未来选择目标，加入成本惩罚、早停和预注册候选上限。当前 v4 使用原封存选择策略，未在本轮用 report-dev 重新选择，避免混淆。
4. **文本保持 P2 封存状态。** 只在新机制有明确假设时重开，不能把 concat64 与 legacy 字符版差异当作独立创新收益。
5. **外部方法补强与大图门禁。** 先给 TAGSAM 适配充分的预注册预算/参数敏感性和独立构造验证，再决定是否做官方完整复现及其它 TAG/continual 方法的共同协议适配。至少完成跨顺序、独立 seeds 和旧/新类保持检查后，再考虑大图。

## 10. 可复查交付物

- [完整机器可读汇总](../../results/small_comparison_summary_2026_09_16/summary.json)、[CSV](../../results/small_comparison_summary_2026_09_16/summary.csv)、[所有组的阶段与旧/新类表](../../results/small_comparison_summary_2026_09_16/tables.md)。
- [验收 JSON](../../results/small_comparison_summary_2026_09_16/acceptance.json)：648 fits，0 error。
- [终阶段图 PDF](../../results/small_comparison_summary_2026_09_16/final_comparison.pdf)、[阶段曲线 PDF](../../results/small_comparison_summary_2026_09_16/stage_curves.pdf)。
- [修正版对照 runner](../../code/v2_llm_fusion/run_small_comparison_audited.py)、[TAGSAM 适配 runner](../../code/v2_llm_fusion/run_tagsam_graphclip_comparison.py)、[指标](../../code/v2_llm_fusion/small_comparison_metrics.py)、[测试](../../code/v2_llm_fusion/test_small_comparison.py)、[验收与汇总](../../code/v2_llm_fusion/summarize_small_comparison.py)、[独立绘图脚本](../../code/v2_llm_fusion/render_small_comparison.py)。

逐seed参数 checkpoint 与 teacher checkpoint 完整保留在远程 `TAG_condensation/results/small_comparison_audited_r2_2026_09_16` 和 `tagsam_graphclip_comparison_2026_09_16`。本地同步逐seed JSON、合成/原始支持 artifact 与汇总；不复制数百份大模型参数文件。运行环境为远程 conda `tagsam`，图表在本地 matplotlib 生成。
