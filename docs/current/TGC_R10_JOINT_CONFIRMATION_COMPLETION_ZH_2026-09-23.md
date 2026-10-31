# TGC-R10 联合 Content-Topology 独立确认完成报告

## 1. 实验目的

R5/R6 与 Phase12 已经观察到：content-only 和 topology-only 可能低于 C0，但两者联合时出现正交互。R10 不再把单组件是否获胜作为联合机制的否定依据，而是在新的 teacher seed 和独立 student seed 下，严格复验四个配对条件：

1. C0；
2. content-only：只使用共享 assignment 生成的新 X/文本，拓扑沿用 C0；
3. topology-only：只使用共享 assignment 投影的新拓扑，X/文本沿用 C0；
4. joint：同时使用共享 assignment 生成的新 X/文本和新拓扑。

本轮不根据 report F1 选择候选，不回读历史完整 raw graph。recursive、maintenance、capacity 作为 R5 runner 的兼容分支生成和审计，但不进入本轮四臂联合主表。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19601`、`order19602`、`order19603` |
| 阶段 | base + 2 incremental，共 3 stages |
| memory | 累计合法 pool 的 5% growing rows |
| 构造 | R5 shared assignment；teacher seeds `9000,9001` |
| student | 两层 GCN/GraphCLIP student，从头训练；batch 16；Adam `1e-3`；固定协议 steps |
| report seeds | `150..155` |
| 主指标 | all-prompt pooled report-dev Macro-F1 |
| 访问范围 | 旧 synthetic memory + 当前 raw delta；不回读历史完整 raw graph |
| 审计 | 每个数据集和顺序均执行独立 checkpoint reload 与 memory 验证 |

R10 新增入口为 `code/v2_llm_fusion/run_tgc_r10_joint_confirmation.py`，它没有改写 R5 构造实现，只替换确认顺序、teacher seeds 和 report seeds。

## 3. 最终阶段结果

| 数据集 | 顺序 | C0 | content | topology | joint | joint-C0 | 交互项 |
|---|---|---:|---:|---:|---:|---:|---:|
| Citeseer | order19601 | 54.54 | 54.41 | 56.84 | 55.52 | +0.98 | -1.19 |
| Citeseer | order19602 | 52.04 | 54.73 | 54.47 | 54.40 | +2.37 | -2.76 |
| Citeseer | order19603 | 52.93 | 51.80 | 50.41 | 50.84 | -2.09 | +1.57 |
| Cora | order19601 | 50.38 | 50.77 | 48.51 | 51.64 | +1.26 | +2.75 |
| Cora | order19602 | 53.05 | 52.45 | 45.42 | 48.01 | -5.04 | +3.18 |
| Cora | order19603 | 46.15 | 47.52 | 48.95 | 48.36 | +2.21 | -1.95 |

汇总：

| 范围 | content-C0 | topology-C0 | joint-C0 | 联合交互项 |
|---|---:|---:|---:|---:|
| 全部最终 cell | +0.43 | -0.75 | -0.05 | +0.27 |
| Cora 最终 cell | +0.38 | -2.23 | -0.52 | +1.33 |
| Citeseer 最终 cell | +0.48 | +0.74 | +0.42 | -0.79 |
| stage 1/2 全部 cell | -1.00 | -1.49 | -1.38 | +1.11 |

## 4. 对“联合优化不能被拆分结果否定”的确认

这轮结果支持用户的判断，但需要精确表述：

- 单组件的负结果不能否定联合方法；联合机制确实在多个 cell 中超过 C0。
- 交互项在最终六个 cell 中有三个为正，平均为 `+0.27 pp`；stage 1/2 平均交互项为 `+1.11 pp`。
- Cora/order19601 是比较清晰的正协同：content-only `+0.39 pp`、topology-only `-1.87 pp`，但 joint `+1.26 pp`，交互项 `+2.75 pp`。
- Cora/order19602 中 joint 仍低于 C0 `5.04 pp`，但相对于两个单组件的加和存在 `+3.18 pp` 的正交互。也就是说，联合构造减轻了两个单组件组合造成的损失，但没有完全恢复到 C0。
- Citeseer/order19602 中两个单组件都高于 C0，但 joint 只高 `+2.37 pp`，低于简单相加的预期，交互项为负。这说明联合机制不是自动产生正协同。

因此当前最准确的结论是：

> content 与 topology 存在条件性的协同构造效应。联合机制不能由单组件消融直接否定，但其绝对收益仍受数据集和类别顺序影响，尚未成为稳定主方法。

## 5. 晋升条件检查

预先写入协议的推进条件及结果如下：

| 条件 | 结果 | 要求 | 是否通过 |
|---|---:|---:|---|
| 最终 joint-C0 平均 | `-0.05 pp` | `>= +0.50 pp` | 否 |
| stage 1/2 joint-C0 平均 | `-1.38 pp` | `>= 0 pp` | 否 |
| 最差最终 joint-C0 | `-5.04 pp` | `>= -2 pp` | 否 |
| 最差最终 old-F1 差 | `+0.10 pp` | `>= -2 pp` | 是 |
| 最终 cell 中 joint-C0 >= -1 pp | `4/6` | `>= 4/6` | 是 |
| 最终 new-F1 平均差 | `-1.92 pp` | `>= 0 pp` | 否 |

整体晋升条件未通过。主要问题不是旧类保持，而是新类塑性和最差顺序表现。joint 在部分 cell 提高了旧类保持，但没有稳定改善新类学习。

## 6. 审计状态

- Cora 和 Citeseer 的 evaluate、audit 均完成；每个数据集包含 3 个顺序。
- 每个数据集/顺序 audit 包含 126 个 fit slots 和 84 个 unique checkpoint reload scores。
- 已读取的 Cora audit 最大重载误差为 `1.42e-14 pp`，Citeseer 为 `2.13e-14 pp`。
- test labels 未参与构造和选择。
- memory、文本重编码、边集合、旧历史字段和 checkpoint 支持 hash 均通过检查。

## 7. 当前方法判断

R10 不能把联合方法判定为失败。它确认了三个有价值的事实：

1. 联合方法确实存在超过 C0 的实际 cell，而不是只有抽象交互项；
2. 单组件失败与联合收益可以同时出现，说明这是非加性构造问题；
3. 当前 joint 的主要瓶颈是顺序稳定性和新类塑性，而不是旧类遗忘。

但 R10 也不能支持“joint 已经稳定优于 C0”。当前应保留 C0 作为可靠控制，把 joint 定位为有实证正交互的候选机制。

## 8. 下一步

下一轮不再重复同样的原始 joint，而做一个最小稳定化实验：

1. 保留 teacher 和共享 P，不改变文本预算和 student；
2. 对 learned P 加 C0 anchor，比较原始 joint、`P_initial`、`0.5P_learned+0.5P_initial`；
3. 保持同一 joint topology materialization，避免同时引入新的文本或预算机制；
4. 报告最终/阶段 F1、old/new F1、交互项、构造时间和 P 的数值漂移；
5. 若 anchor 只能提升 Cora 或单一顺序，不晋升；只有跨两个数据集并降低最差顺序损失，才进入下一版主方法候选。

## 9. 结果文件

- 汇总报告：[R10 自动汇总报告](../../reports/tgc_r10_joint_confirmation_20260923b/report.md)
- 最终 cell：[final_cells.csv](../../reports/tgc_r10_joint_confirmation_20260923b/final_cells.csv)
- 阶段 cell：[stage_cells.csv](../../reports/tgc_r10_joint_confirmation_20260923b/stage_cells.csv)
- 汇总 JSON：[summary.json](../../reports/tgc_r10_joint_confirmation_20260923b/summary.json)
- 运行协议：[protocol.json](../../results/tgc_r10_joint_confirmation_20260923b/meta/protocol.json)
