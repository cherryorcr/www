# TGC Role-Aware Conserved Assignment 独立顺序确认报告

日期：2026-09-26  
状态：**独立顺序、direct old/new F1 和构造成本已完成；`role_perm` 未通过主方法晋升门禁。**

## 1. 结论

本轮冻结并复用了上一轮的 `role_perm`，没有重新搜索权重、顺序或候选。实验使用未参与开发集的 `order19605`、`order19606`、`order19608`，覆盖 Cora/Citeseer 的 stage 1/2，共 12 个 cell。每个 cell 对 `c0`、`identity` 和 `role_perm` 使用同一 Student、同一 batch、同一步数和 Student seeds `150..155` 重新训练，并额外计算 all-prompt 下的 old-class/new-class F1。

相对 C0，`role_perm` 的独立结果为：

- 累计 Macro-F1 平均 `+1.468 pp`；
- `7/12` 个 cell 为正；
- 最差 cell `-2.042 pp`，最好 `+5.938 pp`；
- old-class F1 平均 `+2.212 pp`；
- new-class F1 平均仅 `+0.558 pp`；Cora 为 `+1.141 pp`，Citeseer 为 `-0.025 pp`；
- stage 1 new-class F1 平均 `+1.318 pp`，stage 2 为 `-0.202 pp`。

因此它没有满足事前门禁“平均正收益、最差 cell 不显著退化、new-class plasticity 不系统下降”。`role_perm` 保留为有解释价值的机制分析和负结果，不升为论文默认主方法；C0 继续作为共同冻结主线。

## 2. 与开发集的关系

开发集使用 `order19600/19609/19610`，当时 `role_perm` 为平均 `+3.579 pp`、`10/12` 正，最差约 `-1.155 pp`。独立顺序改为 `order19605/19606/19608` 后，增益下降到 `+1.468 pp`、`7/12` 正，说明开发集收益仍有类别到达顺序依赖，不能把开发集结果直接写成泛化结论。

这不是 direct 评估实现不一致：direct 评估读取冻结 memory，并按构造阶段相同的 `report_candidate` 训练流程重训；同一 cell 的 C0/role 累计 F1 可以逐项对齐。变化来自独立类别顺序，而不是后验挑选或指标替换。

## 3. 逐 cell 结果

表中 F1 均为 6 个 Student seeds 的均值；差值为 `role_perm - C0`。

| Cell | C0 F1 | Role F1 | F1 差值 | old F1 差值 | new F1 差值 |
|---|---:|---:|---:|---:|---:|
| Cora/order19605/s1 | 50.37 | 49.35 | -1.02 | -1.74 | -0.31 |
| Cora/order19605/s2 | 50.56 | 50.52 | -0.04 | +3.34 | -4.55 |
| Cora/order19606/s1 | 57.25 | 61.01 | +3.76 | +2.63 | +4.89 |
| Cora/order19606/s2 | 44.11 | 45.53 | +1.42 | +2.62 | -0.18 |
| Cora/order19608/s1 | 52.97 | 58.02 | +5.05 | +6.35 | +3.75 |
| Cora/order19608/s2 | 52.65 | 58.34 | +5.68 | +7.50 | +3.26 |
| Citeseer/order19605/s1 | 53.38 | 52.16 | -1.22 | -4.12 | +1.69 |
| Citeseer/order19605/s2 | 55.33 | 53.29 | -2.04 | -1.73 | -2.66 |
| Citeseer/order19606/s1 | 65.69 | 63.79 | -1.90 | -2.54 | -1.26 |
| Citeseer/order19606/s2 | 51.45 | 51.99 | +0.54 | +1.76 | -1.90 |
| Citeseer/order19608/s1 | 54.96 | 60.90 | +5.94 | +12.71 | -0.84 |
| Citeseer/order19608/s2 | 46.95 | 48.41 | +1.45 | -0.23 | +4.83 |

## 4. 聚合结果

| 子集 | 指标 | C0 | Identity | Role | Role-C0 | 正 cell |
|---|---|---:|---:|---:|---:|---:|
| 全部 12 cell | cumulative F1 | 52.974 | 53.604 | 54.442 | +1.468 | 7/12 |
| 全部 12 cell | old F1 | 50.194 | 51.669 | 52.406 | +2.212 | 7/12 |
| 全部 12 cell | new F1 | 53.840 | 53.586 | 54.398 | +0.558 | 5/12 |
| Cora 6 cell | cumulative F1 | 51.321 | 52.295 | 53.795 | +2.474 | 4/6 |
| Cora 6 cell | old F1 | 41.401 | 43.317 | 44.852 | +3.451 | 5/6 |
| Cora 6 cell | new F1 | 61.546 | 61.665 | 62.687 | +1.141 | 3/6 |
| Citeseer 6 cell | cumulative F1 | 54.627 | 54.913 | 55.089 | +0.461 | 3/6 |
| Citeseer 6 cell | old F1 | 58.986 | 60.020 | 59.959 | +0.973 | 2/6 |
| Citeseer 6 cell | new F1 | 46.134 | 45.507 | 46.110 | -0.025 | 2/6 |
| Stage 1 | new F1 | 64.552 | 64.434 | 65.870 | +1.318 | 3/6 |
| Stage 2 | new F1 | 43.129 | 42.738 | 42.927 | -0.202 | 2/6 |

## 5. 成本账

成本只测构造，不包含 Student 训练。计时路径复用冻结实现，分解为共享 two-prototype proposal、role feature/Hungarian assignment 和 payload materialization。

| 部分 | 12-cell 平均 | 中位数 |
|---|---:|---:|
| Two-prototype structural proposal | 33.5 ms | 31.3 ms |
| Role feature + Hungarian assignment | 21.6 ms | 19.2 ms |
| Payload materialization | 3.7 ms | 3.7 ms |
| Role 增量（assignment + materialization） | 25.3 ms | 22.7 ms |
| 总构造路径 | 58.8 ms | 53.3 ms |

role 增量约占这条构造路径的 `41.9%`。绝对成本远小于一次 Student 重训，但在大图上仍需要按候选槽位数报告复杂度，不能仅用小图毫秒数宣称可扩展。

## 6. 审计和协议

- `historical_raw_reads = 0`；
- 构造阶段未使用 report/test 标签；
- `role_perm` 保持 C0 `edge_index` 不变；
- 旧 memory prefix 未变；
- 同类槽位内 payload 是完整双射，source coverage 和 payload multiset 保持；
- direct F1 评估只在 memory 封存之后读取 report-dev 标签；
- C0、Identity、Role 使用相同 Student、batch、steps 和 seeds。

## 7. 方法解释

Role assignment 确实有机制信号：Cora 的独立均值为 `+2.474 pp`，并且部分 cell 同时改善 old/new F1。但它不是顺序无关的增强器。当前 cost 只决定“哪个完整 payload 放入哪个同类 C0 槽位”，不能改变：

1. C0 已经决定的边和归一化传播上下文；
2. two-prototype proposal 的内容误差；
3. 类别 prompt 竞争和 Student 的训练动态；
4. stage 2 中旧类规模、旧类传播和新类塑性的耦合。

因此 role 特征在某些顺序中能修正“内容与传播角色错配”，但在另一些顺序中会把语义 payload 移到对下游 prompt 竞争不利的槽位。`role_perm` 的收益更像条件性的结构-语义匹配，而不是稳定的普适性能提升。

## 8. 决策与后续

本轮后不再对 `role_perm` 做事后权重扫描，也不启动其大图 base+4 扩展；独立顺序已经没有通过小图晋升门禁，继续放大实验会增加选择偏差。论文定位应为：

- C0 是稳定共同基线和默认主方法；
- role-aware conserved assignment 是一个守恒、无需历史 raw 回读的结构-语义匹配机制；
- 它在开发顺序上有明显正信号，但在独立顺序上呈现顺序依赖，new-class plasticity 没有跨数据集稳定改善；
- 该负结果说明“只重排 payload、不改变 C0 拓扑和 proposal 内容”不足以解决 continual TAG condensation 的核心难点。

若后续仍要继续算法创新，应基于这个失败边界重新设计一个事前冻结的联合机制，并先在小图通过独立顺序门禁，再考虑 base+4；不能把 `role_perm` 与后续模块事后叠加成一个未验证的主方法。

## 9. 结果文件

- direct 评估：`results/tgc_role_aware_detailed_eval_20260926_cora/`、`results/tgc_role_aware_detailed_eval_20260926_citeseer/`
- 构造与守恒审计：`results/tgc_role_aware_assignment_independent_20260926/`
- 构造成本：`results/tgc_role_assignment_cost_20260926_cora_payload.json`、`results/tgc_role_assignment_cost_20260926_citeseer_payload.json`
- 计时脚本：`code/v2_llm_fusion/measure_tgc_role_assignment_cost_20260926.py`
