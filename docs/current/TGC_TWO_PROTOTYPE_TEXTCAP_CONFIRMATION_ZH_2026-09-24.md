# 双原型文本上限确认实验报告

日期：2026-09-24  
状态：完成三格六 seed 复现；未确认稳定提升，暂不晋升为主方法

## 1. 实验目的

此前 Cora `order19601` 的一个旧实验曾报告双原型相对 C0 约 `+2.328 pp`。该结果使用了旧远程依赖快照，部分拼接文本实际超过名义上的 320 字符，且 Student seed 数量较少。本轮重新实现并回答三个问题：

1. 增益是否来自旧文本长度实现漂移；
2. 增益是否能在多个 Student seed 中复现；
3. 增益是否能跨数据集和类别顺序保持。

## 2. 固定方法

双原型候选仍采用此前的局部 shared two-prototype split/merge：每个新类选择一个高离散 source group 拆成两个 prototype，同时合并该类最相近的两个 group，使总 rows 不变；三个 proposal 使用 Hungarian matching 对齐回 C0 的 row slots。X 和文本使用同一 source piece，拓扑完全复制 sealed C0，旧 synthetic prefix 不变。

控制与候选严格成对评估：

| 项目 | 设置 |
|---|---|
| 数据集/顺序 | Cora `order19601`、Cora `order19602`、Citeseer `order19601` |
| 阶段 | stage 1 |
| memory | 5% growing memory，总 rows 与 sealed C0 相同 |
| Student | 原始两层 GCN、trainable text projection、双向 InfoNCE |
| Student seeds | `150..155`，每个 arm 使用相同 seed |
| batch/steps | 与对应 C0 相同，batch=16，steps=protocol v3 |
| 文本来源 | top-5，冻结 MiniLM 重新编码 |
| 文本 cap | 严格 320；显式 324；两种 cap 均同时构造 C0 和双原型 |
| 拓扑 | sealed C0 `edge_index` bitwise copy |
| 历史访问 | 不访问历史完整 raw graph |
| 选择 | 不使用 report/test F1 selector |

`c0_320` 与 `two_320` 使用严格 320 字符上限；`c0_324` 与 `two_324` 使用显式 324 字符上限，用来复现旧实现可能产生的 321--324 字符文本。所有候选的 cap 都是显式参数，不再依赖偶然的拼接越界。

## 3. 主结果

表中差值为相对于同一文本 cap、同一 dataset/order 和同一 Student seed 的 paired Macro-F1 差值，单位为百分点。

| 条件 | two-320 vs c0-320 | 正向 seed | two-324 vs c0-324 | 正向 seed |
|---|---:|---:|---:|---:|
| Cora/order19601 | `+1.520` | 4/6 | `+2.697` | 4/6 |
| Citeseer/order19601 | `-1.970` | 1/6 | `-2.794` | 1/6 |
| Cora/order19602 | `-0.072` | 3/6 | `-1.131` | 2/6 |
| 三格简单平均 | `-0.174` | 8/18 | `-0.409` | 7/18 |

逐 seed 结果：

| 条件 | two-320 vs c0-320 | two-324 vs c0-324 |
|---|---|---|
| Cora/order19601 | `+6.249, +2.318, +1.993, +2.242, -0.686, -2.993` | `+5.335, +4.171, -0.562, +3.210, -2.109, +6.138` |
| Citeseer/order19601 | `-4.501, -3.634, -2.297, -3.755, -3.533, +5.900` | `-4.315, -6.502, -4.240, +4.414, -4.517, -1.602` |
| Cora/order19602 | `+6.673, +2.224, -2.833, -2.478, -8.063, +4.048` | `+2.247, -4.843, -3.375, +0.917, -0.192, -1.541` |

## 4. 文本 cap 与 C0 重构审计

| 条件 | sealed C0 vs c0-320 | c0-324 vs c0-320 |
|---|---:|---:|
| Cora/order19601 | `-0.467` | `-0.550` |
| Citeseer/order19601 | `-2.172` | `+1.747` |
| Cora/order19602 | `-1.453` | `-0.268` |

这说明旧 sealed C0 与由当前 source assignment 重构的 C0 不是同一 memory，不能把旧 sealed C0 与新候选直接拼成同一 paired comparison。正式比较使用同 cap 的 `two_xxx` 与 `c0_xxx`；sealed C0 只作为实现审计参考。

每个 cell 的文本长度审计均符合显式上限：`c0_320/two_320 <= 320`，`c0_324/two_324 <= 324`。候选与控制的拓扑和旧 prefix 审计通过。

## 5. 结论

### 5.1 不是简单的统计错误，但不是稳定方法收益

Cora/order19601 的正向信号在六个 Student seed 中仍然出现，说明旧结果并非完全由错误统计造成；然而它只在一个数据集/顺序上成立。Citeseer 和另一条 Cora 顺序不支持该结论，三格平均为负，且存在较大的负 seed。

因此正确表述是：

> 局部双原型内容改造在特定 Cora 顺序上存在可复现的条件性正信号，但目前不能作为跨数据集、跨顺序稳定提升的方法。

### 5.2 324 字符不是充分解释

Cora/order19601 中 324 cap 比 320 cap 多约 `+1.177 pp` 的平均候选收益；但 Citeseer 324 反而比 320 更差，Cora/order19602 也更差。因此旧文本长度漂移可能放大了单个 cell 的结果，但不是普适增益来源。

### 5.3 当前主要矛盾仍是内容与 C0 row role 的关系

本轮拓扑完全固定，双原型只改变新块 X/text。结果仍然随顺序和 Student seed 大幅变化，说明问题不只是文本 cap，也不只是“是否拆成两个 prototype”。候选内容被写入 C0 的不同 row slots 后，会继承这些 slots 的 degree、旧邻居比例、跨块负荷和消息传播角色。相同的内容改造在不同 slot role 上可能产生相反的 downstream 结果。

## 6. 晋升判定

本轮不满足主方法晋升条件：

- 至少两个独立数据集/顺序稳定正收益：未通过；
- 三格平均收益不低于零：未通过；
- 最差 seed 不低于 `-2 pp`：未通过；
- 旧类保持与新类塑性同时改善：未通过。

C0 继续作为当前可靠主方法和共同比较锚点。双原型保留为条件性内容机制，不进入主结果主线。

## 7. 下一步

不再继续扫描文本 cap、固定 alpha 或更多双原型组合。按原计划转向 topology-role 诊断，但先做低成本的归因实验，不直接构造新的主方法：

1. 固定同一双原型内容、文本、rows 和 C0 topology；
2. 将同一组改变分别写入低/高 degree、低/高 cross-load、不同旧邻居比例的 C0 row slots；
3. 报告 synthetic message drift、old/new message risk、真实新类 margin 和 Student Macro-F1；
4. 不使用 report F1 选择 slot，不改旧 prefix，不访问历史完整 raw graph；
5. 若 role 特征能解释正负差异，再实现受限的 topology-role-aware matching；若不能解释，则停止双原型路线，回到 C0 并将其作为负结果。

## 8. 产物

- Runner：[run_tgc_two_prototype_textcap_confirmation.py](../../code/v2_llm_fusion/run_tgc_two_prototype_textcap_confirmation.py)
- Remote wrapper：[tgc_two_prototype_textcap_confirmation_remote.py](../../tools/tgc_two_prototype_textcap_confirmation_remote.py)
- 结果目录：[tgc_two_prototype_textcap_confirmation_20260924](../../results/tgc_two_prototype_textcap_confirmation_20260924/)
- 远程快照：`/home/pc/research/TAG_condensation/work_runs/tgc_two_prototype_smoke_20260924`
- 本地语法检查：通过
- 三个 cell：远程执行与结果拉回完成
