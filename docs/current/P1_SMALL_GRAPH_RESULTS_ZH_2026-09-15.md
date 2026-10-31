# P1 小图验证结果报告

日期：2026-09-15  
数据集：Cora、Citeseer  
协议：unified v3、3-stage、batch=16、student seeds 100--105、报告 dev、test 封存  
主方法：synthetic-only `append_fixed`  
对照：`k_center`、`random_real`

## 1. 先说明结果版本

Claude 交接的原始 53 个结果已经全部完成，但其中 `legacy_15` 会改变新旧 memory 的 rows 配额，不能直接作为纯 allocation 因果结论。因此本报告以新的 matched-quota 受控重跑为主：

- 所有 allocation 在同一阶段使用相同的 `keep_old_rows / new_rows`；
- 只有新块内部的类内 rows 分配不同；
- 所有结果保留 manifest、代码 hash 和 test 封存字段；
- 受控 allocation：Cora/Citeseer 5% × 5 allocation × 3 arms，共 10 个结果；
- 预算策略：Cora/Citeseer 各 fixed-total、hybrid、adaptive-rate，共 6 个结果。

## 2. Matched-quota allocation 结果

最终 s2 cumulative dev Macro-F1：

| 数据集 | allocation | append_fixed | k-center | random_real |
|---|---|---:|---:|---:|
| Cora | `legacy_15` | 45.50 | 46.35 | 43.47 |
| Cora | `equal` | 45.50 | 46.35 | 43.47 |
| Cora | `linear` | 44.52 | 46.35 | 43.47 |
| Cora | `sqrt` | 47.28 | 46.35 | 43.47 |
| Cora | `dispersion` | **49.85** | 46.35 | 43.47 |
| Citeseer | `legacy_15` | 56.70 | 55.34 | 49.79 |
| Citeseer | `equal` | 56.70 | 55.34 | 49.79 |
| Citeseer | `linear` | 58.52 | 55.34 | 49.79 |
| Citeseer | `sqrt` | 57.75 | 55.34 | 49.79 |
| Citeseer | `dispersion` | **59.17** | 55.34 | 49.79 |

阶段配额已经匹配：

- Cora：`s0 0+25, s1 25+42, s2 67+27`；
- Citeseer：`s0 0+38, s1 38+48, s2 86+25`。

### 结果解释

1. 原始报告中 Cora `legacy_15=57.87` 的优势不再出现；受控后为 45.50，与 equal 完全相同。原优势来自 `legacy_15` 在 s2 追加 45 行，而其它策略追加 27 行。
2. `dispersion` 在两个小图上都是最佳 allocation：Cora 49.85，Citeseer 59.17。
3. 这是 matched-quota 下的初步跨数据集支持，但不能直接宣称普适最优；目前只覆盖一个压缩率和一个类别顺序。
4. 在当前协议下，synthetic-only append 的最佳 allocation 均优于真实节点对照：Cora 比 k-center 高 3.50，Citeseer 高 3.83。

## 3. 预算增长协议

`sqrt` allocation 下的最终结果：

| 数据集 | 协议 | budgets | 最终 rows | s2 cumulative F1 |
|---|---|---:|---:|---:|
| Cora | fixed-rate control | 25,67,94 | 94 | 47.28 |
| Cora | fixed-total | 94,94,94 | 94 | 42.79 |
| Cora | hybrid, min_increment=5 | 25,67,94 | 94 | 47.28 |
| Cora | adaptive-rate | 24,64,80 | 80 | **49.16** |
| Citeseer | fixed-rate control | 38,86,111 | 111 | 57.75 |
| Citeseer | fixed-total | 111,111,111 | 111 | 53.60 |
| Citeseer | hybrid, min_increment=5 | 38,86,111 | 111 | 57.75 |
| Citeseer | adaptive-rate | 40,80,103 | 103 | 54.90 |

### 结果解释

- `fixed-total` 在两个小图上都低于 fixed-rate，说明从第一阶段就使用完整 memory 总预算并不能自动带来更好持续效果；它改变了早期/后期记忆容量分布。
- 本次 `hybrid(min_increment=5)` 在两个小图上与 fixed-rate 完全相同，因为 fixed-rate 的阶段增量已经大于 5；该设置没有形成有效干预，不能据此评价 hybrid 的普遍性。
- `adaptive-rate` 在 Cora 用更少 rows 得到 49.16，高于 fixed-rate 47.28；但在 Citeseer 用 103 rows 得到 54.90，低于 fixed-rate 57.75。它有数据集相关的潜力，但还不是稳定优势。
- adaptive utility 完全由 source-only 的 novelty/dispersion 计算，不读取 report-dev 或 test 标签；这满足协议上的 label-free 约束，但 utility 与最终 F1 的一致性尚未证明。

## 4. 小图验收结论

### 已验证

- matched memory 下，allocation 效应仍然存在；
- `dispersion` 在 Cora/Citeseer 的 5% 受控实验中均胜出；
- synthetic-only append 在最佳 allocation 下优于 k-center 和 random-real；
- fixed-total 并非默认优选；
- adaptive-rate 有一个正例（Cora），也有一个负例（Citeseer），因此负结果本身是可靠发现。

### 尚未验证

- allocation 在 0.5%、1%、10% 等 rate 下是否仍稳定；
- 多类别顺序、base+4 和长期遗忘；
- adaptive utility 是否能跨数据集预测最优预算；
- hybrid 在真正产生额外预算的 `min_increment` 下的收益；
- P2 文本压缩、P3 SCA、P4 历史 memory 管理。

## 5. 是否现在上大图

**当前不建议立即把大图结果写入论文主表。** 原因不是小图方法失败，而是小图已经发现两个需要先固定的问题：

1. `legacy_15` 的历史 quota 逻辑会污染 allocation 因果分析；
2. adaptive-rate 的方向跨数据集翻转，不能直接当成 v4 默认机制。

建议先在小图完成一个最小确认批次：

- matched-quota 下补做 1% 和 10% 两个 rate；
- 对 `dispersion` 与 `sqrt` 做一次独立类别顺序或 base+4 确认；
- 将 hybrid 的 `min_increment` 改为确实改变预算的设置；
- 固化当前受控 runner 和 manifest。

若 `dispersion` 在这些小图确认中仍保持优势，再把它带到大图；adaptive-rate 应作为数据集相关的消融，不应直接作为默认主方法。

## 7. 新增三类 response 机制初测

为验证后续方法，新增了三个独立 runner 开关，并仍然只在 Cora/Citeseer、5%、3-stage、6 canonical student seeds 上运行。

### 7.1 类内子簇复杂度分配

方法：每个新类别内部在 joint graph-text space 中划分多个子簇，再按子簇的 mass、dispersion 和 complexity 分配固定总 rows。

| 方法 | Cora | Citeseer |
|---|---:|---:|
| dispersion baseline | 49.85 | 59.17 |
| subcluster complexity | 49.38 | 53.44 |

当前实现可以产生并记录每个 synthetic subcluster 的 `source_mass`、`dispersion`、`complexity` 和 `rows`，但复杂度先验本身没有超过 dispersion。它应作为 candidate generator，而不是最终分配规则。

### 7.2 response-based marginal allocation

方法：固定总新增 rows，以 dispersion 为初始分配，每轮把一行从 donor 类别转移到 recipient 类别，用 selector-dev 的 student response 选择转移方向。

| 方法 | Cora | Citeseer |
|---|---:|---:|
| dispersion baseline | 49.85 | 59.17 |
| response marginal | **52.40** | 57.29 |

Cora 有提升，Citeseer 下降，说明 response selector 有信息，但当前 ranking 还不具有跨数据集稳定性。它暂时不能替代 dispersion。

### 7.3 response-gain extra budget

方法：保留 fixed-rate 基础预算，额外测试最多 8 rows；只有 selector-dev response gain 大于阈值才接受扩容。

| 数据集 | 接受的阶段 | 最终 rows | 最终 F1 | dispersion baseline |
|---|---|---:|---:|---:|
| Cora | s0/s1/s2 | 102 | 46.00 | 49.85 |
| Citeseer | s0 | 111 | 53.43 | 59.17 |

Cora 接受了额外 rows 但结果下降；Citeseer 只在 s0 接受扩容，最终仍下降。这说明当前短程 selector-dev response 对 canonical full student 的预测存在明显 regret，不能直接用作预算控制器。

### 7.4 当前判定

- 三个机制已经完成最小实现和小图首轮运行；
- 子簇复杂度分配没有显示独立优势；
- response marginal 只在 Cora 有效；
- response-gain extra 当前失败，不能作为默认 adaptive budget；
- 后续若继续研究，优先修正 selector：减少 batch 条件混合、增加独立 selection seeds、加入 full-step terminal confirmation 和 gain uncertainty，而不是直接扩大到大图。

### 7.5 六组消融矩阵

当前已完成的五个可解释对照如下，数值为 s2 cumulative dev Macro-F1：

| 对照 | Cora | Citeseer | 结论 |
|---|---:|---:|---|
| fixed-rate + dispersion baseline | 49.85 | 59.17 | 当前最强稳定 baseline |
| complexity-only | 49.38 | 53.44 | 复杂度先验没有独立优势 |
| response-marginal only | **52.40** | 57.29 | Cora 有效，Citeseer 下降 |
| complexity + response-marginal | 48.64 | 55.29 | 复杂度先验没有帮助 response selector |
| fixed base + response-gain extra | 46.00 | 53.43 | 额外 rows 接受规则产生 canonical regret |

第六个“全模块合并”没有直接运行。原因是 response-gain 已经在两个小图上未通过 canonical confirmation，继续把它与复杂度和 marginal selector 合并，只会增加计算成本而不能提供清晰的因果解释。它应在 selector 修正后重新定义为独立实验，而不是当前论文主方法。

当前最合理的研究结论是：

1. 保留 `fixed-rate + matched quota + dispersion` 作为主 baseline；
2. response-marginal 作为有潜力但未稳定的研究分支；
3. complexity-only、complexity+response 和 response-gain 作为当前负结果；
4. 不把全模块合并方案带到大图，先修正 response ranking 与 canonical regret。

## 6. 结果文件

- 受控 allocation：`results/p1_review_2026_09_15/p1_controlled_quota_2026_09_15/`
- 预算策略：`results/p1_review_2026_09_15/p1_budget_policies_2026_09_15/`
- Claude 原始结果快照：`results/p1_review_2026_09_15/p1_results/`
- 交接审查：`docs/current/P1_CLAUDE_HANDOFF_REVIEW_ZH_2026-09-15.md`
