# P1-B 固定预算 Guarded Response 配额转移验收报告

日期：2026-09-18  
状态：Cora/Citeseer 小图、三种类别顺序、三条实验臂全部完成；不晋升为默认主方法，暂不上大图。

## 1. 本轮要回答的问题

P3-A 已经显示：response selector 的短程排序不稳定，而且 selector 成本远高于低成本构造。因此本轮没有继续做 `response-gain extra budget` 或 `adaptive-rate`，而是把问题缩小为一个可审计的固定预算实验：

> 在总 memory rows 和 student 训练步数都不增加的条件下，response 是否值得把一个 row 从某个新类转移给另一个新类？

这样可以把三种因素分开：

- 合成图是否有价值：由 `dispersion` 控制；
- response 选择是否有价值：由 `response_guarded` 测试；
- 任意换配额是否可能带来收益：由 `matched_random_transfer` 测试。

## 2. 冻结协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | canonical、order1701、order1702 |
| 总预算 | 5% fixed-rate，按阶段增长；各臂总 rows 完全相同 |
| 低成本控制 | dispersion allocation + mean-X + `joint_g5_c320` + frozen-old topology |
| response 变化 | 每阶段最多进行一次单 row class-row transfer |
| response 评价 | selector seeds `401,402`，batch `16`，三点 response curve |
| 门禁 | `LCB gain > 0.5`、`horizon gain >= 0`、`LCB >= 0`；失败回退 dispersion |
| 风险设置 | old-class 权重 `0.25`，risk z `1.0` |
| student | 与 v3/v4 相同；六个 report seeds `100--105`，不增加 steps |
| 随机对照 | matched random transfer，固定总 rows、随机移动一个 row，不使用 response |
| 评估 | all-prompt report-dev cumulative Macro-F1；未读取 test labels |

本轮每个数据集完成 `3 orders x 3 arms x 3 stages x 6 student seeds = 162` 次 reload/recheck；两个数据集合计 324 次。两套 `acceptance.json` 均标记 complete，预算、synthetic-only、旧 memory 审计和 checkpoint 重载验收通过。

## 3. 方法臂的具体含义

### 3.1 dispersion

先按 dispersion 规则决定每个新类的 rows，再生成合成节点并 append 到旧 synthetic memory。它是本轮的低成本固定预算参照，不包含 response 训练。

### 3.2 response_guarded

先生成 dispersion allocation，然后枚举一个 row 的 donor/recipient 转移。每个候选使用独立 selector seed 做短 response curve，同时评价新类表现、累计表现和旧类保持。只有候选同时通过 LCB 与 horizon 门禁，才接受这一次转移；否则保留 dispersion allocation。

它不增加：

- 总 memory rows；
- 合成图的最终 payload；
- student steps；
- 历史 memory 的可写范围。

### 3.3 matched_random_transfer

从同一个 dispersion allocation 出发，随机选择 donor/recipient 并转移一个 row。它不运行 response selector，用于判断 response 的提升是否只是由“改变类间配额”造成。

## 4. 最终阶段 Macro-F1

`delta` 是相对于同一数据集、同一类别顺序的 dispersion。

| 数据集 | 顺序 | dispersion | response_guarded | delta | random_transfer | random delta |
|---|---|---:|---:|---:|---:|---:|
| Cora | canonical | 49.85 | 49.85 | +0.00 | 46.48 | -3.37 |
| Cora | order1701 | 52.59 | 54.28 | +1.69 | 52.84 | +0.25 |
| Cora | order1702 | 42.77 | 46.55 | +3.78 | 45.33 | +2.57 |
| Citeseer | canonical | 59.17 | 60.77 | +1.59 | 54.36 | -4.81 |
| Citeseer | order1701 | 53.76 | 53.76 | +0.00 | 54.69 | +0.92 |
| Citeseer | order1702 | 50.94 | 54.20 | +3.26 | 56.55 | +5.61 |

### 4.1 按类别顺序聚合

| 数据集 | 方法 | 平均最终 F1 | 顺序间 SD | 平均阶段 F1 | 最终 old F1 | 最终 new F1 |
|---|---|---:|---:|---:|---:|---:|
| Cora | dispersion | 48.40 | 4.14 | 49.04 | 56.99 | 36.95 |
| Cora | response_guarded | 50.23 | 3.16 | 54.82 | 58.08 | 39.76 |
| Cora | matched_random_transfer | 48.22 | 3.30 | 47.64 | 53.82 | 40.74 |
| Citeseer | dispersion | 54.63 | 3.42 | 60.97 | 58.67 | 46.53 |
| Citeseer | response_guarded | 56.24 | 3.20 | 61.47 | 60.18 | 48.37 |
| Citeseer | matched_random_transfer | 55.20 | 0.97 | 58.74 | 60.15 | 45.30 |

response 相对 dispersion 的聚合结果为：

| 数据集 | 最终 F1 平均 delta | 阶段平均 F1 delta | 最终获胜顺序数 | 阶段平均获胜顺序数 |
|---|---:|---:|---:|---:|
| Cora | +1.82 | +5.78 | 2/3 | 1/3 |
| Citeseer | +1.62 | +0.50 | 2/3 | 2/3 |

## 5. response 实际接受了多少次

这里的“接受”只表示候选通过门禁并真正改变了最终 class-row allocation；门禁失败时，即使 audit 中记录了候选，也计为 fallback。

| 数据集 | 顺序 | 接受阶段 | 转移 rows/阶段 | fallback 阶段 |
|---|---|---:|---|---:|
| Cora | canonical | 0/3 | 0, 0, 0 | 3 |
| Cora | order1701 | 1/3 | 0, 1, 0 | 2 |
| Cora | order1702 | 3/3 | 1, 1, 1 | 0 |
| Citeseer | canonical | 1/3 | 0, 1, 0 | 2 |
| Citeseer | order1701 | 0/3 | 0, 0, 0 | 3 |
| Citeseer | order1702 | 1/3 | 0, 1, 0 | 2 |

合计：

- Cora：9 个阶段接受 4 次，接受率 `4/9 = 44.4%`；
- Citeseer：9 个阶段接受 2 次，接受率 `2/9 = 22.2%`。

这说明 guarded gate 确实阻止了很多不可靠的配额变化，但也说明 response 信号只有少数阶段能够通过当前门禁。它更像一个“有条件的候选”，还不是稳定的默认分配器。

## 6. old/new 与保持-塑性解释

response_guarded 的最终 old/new 平均 F1 在两个数据集都高于 dispersion：

- Cora：old `56.99 -> 58.08`，new `36.95 -> 39.76`；
- Citeseer：old `58.67 -> 60.18`，new `46.53 -> 48.37`。

但这不是每个顺序都成立。例如 Citeseer order1701 的 response 最终结果完全回退到 dispersion；Citeseer order1702 的 random transfer 反而高于 response。因而本轮最多说明：

> 在固定总预算下，受门禁保护的 response 配额转移在小图平均结果上有积极信号，但收益具有顺序依赖，并且不能排除部分收益来自配额变化本身。

task-level forgetting 的审计也已计算。它只能作为辅助指标，因为不同阶段旧任务集合不同，不能把不同旧类集合的 old F1 直接当作同一任务的纵向差值。

## 7. 成本账

成本范围为三阶段 fresh construction 加六个 student seeds；共享 preprocessing 不计入。student steps 完全相同，差异主要来自 selector。

| 数据集 | 方法 | 构造+审计秒数 | selector 秒数 | 六 seed 总秒数 | selector steps |
|---|---|---:|---:|---:|---:|
| Cora | dispersion | 0.44 | 0.00 | 37.12 | 0 |
| Cora | response_guarded | 56.97 | 55.01 | 93.25 | 25,704 |
| Cora | random transfer | 0.45 | 0.00 | 36.69 | 0 |
| Citeseer | dispersion | 0.58 | 0.00 | 49.14 | 0 |
| Citeseer | response_guarded | 43.40 | 41.84 | 91.34 | 18,984 |
| Citeseer | random transfer | 0.56 | 0.00 | 48.50 | 0 |

response 构造时间约为 dispersion 的：

- Cora：`56.97 / 0.44 ~= 130x`；
- Citeseer：`43.40 / 0.58 ~= 75x`。

端到端三阶段链的六 seed 总耗时约为：

- Cora：`93.25 / 37.12 ~= 2.51x`；
- Citeseer：`91.34 / 49.14 ~= 1.86x`。

所以当前 response 不能被描述为计算高效。它只有在性能增益足够稳定时才可能值得保留；本轮证据尚不足以支持这个结论。

## 8. 结合 P3-A 的判定

### 8.1 P3-A 带来的设计约束

P3-A 的核心负结果是：旧版 response-on 的短训 ranking 不能稳定预测 canonical ranking，且 selector 成本远高于低成本构造。因此 P1-B 做了四个收缩：

1. 不再允许 response 同时决定 memory 总规模；
2. 每阶段最多只转移一个 row；
3. 使用独立 selection seeds `401,402` 和固定 batch `16`；
4. 使用 LCB、horizon 和 fallback gate，失败就回到 dispersion。

### 8.2 本轮是否通过晋升

**性能筛选：部分通过。** 两个数据集的平均最终 F1 都提升，且各有 2/3 顺序不低于 dispersion。

**因果筛选：未通过。** Citeseer order1702 中 random transfer `56.55` 高于 response `54.20`，说明 response 不是所有收益的唯一来源。

**稳定性筛选：未通过。** response 仍有大量 fallback，且最终/阶段平均优势不是所有顺序一致。

**成本筛选：未通过。** response 构造时间增加约 75--130 倍，三阶段总链耗时增加约 1.86--2.51 倍。

因此判定为：

> P1-B 固定预算 guarded response transfer 通过“小图有潜力”门禁，但未通过“默认主方法”门禁，也未通过“大图扩展”门禁。

## 9. 后续推进建议

### 保留

- 将 `response_guarded` 作为固定预算的候选消融；
- 在论文中报告其平均提升、fallback 率和成本；
- 保留 `matched_random_transfer`，否则无法判断配额变化本身的作用。

### 暂停

- `response-gain extra budget`；
- `adaptive-rate`；
- 将 response selector 扩大到更多候选和更长搜索；
- 直接上大图。

### 下一步优先级

1. 先在固定 rows 下完成 P1-X：Mean-X、Weighted-X、Moment-X；
2. 对表现最好的 X 构造再做独立类别顺序和 selection seed 确认；
3. 如果 X 构造仍不能稳定缩小与 k-center 的差距，再研究低成本 response 代理；
4. 只有固定预算候选在小图上通过独立确认，才重新考虑大图和 extra budget。

## 10. 复现文件

- 结果根目录：`results/p1b_guarded_small_20260918/`
- 汇总结果：`results/p1b_guarded_small_20260918/summary/summary.json`
- 每顺序表：`results/p1b_guarded_small_20260918/summary/per_order.csv`
- 聚合表：`results/p1b_guarded_small_20260918/summary/aggregate.csv`
- 运行入口：`code/v2_llm_fusion/run_p1b_guarded_small.py`
- 汇总脚本：`code/v2_llm_fusion/summarize_p1b_guarded.py`

