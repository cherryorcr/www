# TGC-R11 C0-anchored Joint 完成报告

## 1. 结论先行

R11 在 R10 已封存的 joint assignment 上加入 C0 anchor：

\[
P_\alpha=(1-\alpha)P_{\mathrm{initial}}+\alpha P_{\mathrm{joint}}
\]

然后使用同一个 `P_alpha` 同步生成新节点特征、文本和投影拓扑。它不是对已经材料化的 memory 做平均，也没有回读历史完整 raw graph。

本轮结果：

- `anchor025` 是最好的 anchor 候选，但最终六个 cell 平均只比 C0 高 `+0.142 pp`；
- `anchor025` 的阶段平均仍比 C0 低 `-0.856 pp`，新类 F1 低 `-2.059 pp`，最差最终 cell 为 `-2.104 pp`；
- 原始 `joint` 最终平均为 `-0.051 pp`，与 R10 一致；
- anchor 能够把原始 joint 的最差最终退化从 `-5.043 pp` 缩小到 `-2.104 pp`，但主要改善来自旧类保持，不是新类塑性；
- 所有构造和 checkpoint audit 均通过，不能把本轮问题归因于实现错误；
- **R11 不晋升为主方法，C0 继续作为可靠主线和共同控制。**

因此 R11 的准确定位是：

> C0-anchored assignment interpolation 是一个有条件的 joint 稳定化组件，而不是已经成立的性能主方法。

## 2. 实验问题

R10 发现 content 与 topology 的联合构造存在条件性正交互，但 joint 在最差顺序上退化明显。R11 只改变共享 assignment，不改变 student、文本预算、memory budget 和拓扑材料化规则，回答：

1. 将 learned assignment 拉回 C0 初始 assignment，是否能减少 joint 的坏顺序？
2. 哪个 anchor 强度在保持 joint 方向性的同时最稳定？
3. 稳定化是否同时改善新类塑性，还是只保护旧类？

## 3. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19601`、`order19602`、`order19603` |
| 阶段 | base + 2 incremental，共 3 stages |
| memory | 累计合法 pool 的 5% growing rows |
| R10 source | sealed R10 C0/joint memory 与 shared assignment |
| teacher seeds | `9000, 9001`，继承 R10 封存结果 |
| report seeds | `150..155` |
| student | 两层 GCN/GraphCLIP student，从头训练；batch 16；Adam `1e-3` |
| student steps | `max(50, min(2048, 16 * rows))` |
| 候选 | `C0`、`P-initial`、`joint`、`anchor025`、`anchor050`、`anchor075` |
| alpha | `0`、`1`、`0.25`、`0.5`、`0.75` |
| 文本 | 同一 assignment 选最多 5 个来源、320 字符后重新编码 |
| 拓扑 | 同一 assignment 投影后材料化为二值边 |
| 历史访问 | 旧 synthetic memory + 当前 raw delta；历史 raw reads = 0 |
| 选择 | 不根据 report F1 选择候选；所有候选全部报告 |
| test labels | 未使用 |

每个数据集有 `3 × 3 × 6 × 6 = 324` 个 fit slot，两数据集合计 `648` 个 slot；其中实际 student fit 为 `468`，其余为 stage-0 或相同 memory 的合法复用。两套数据集的 evaluate/audit 均通过，最大 reload 误差低于 `3e-14 pp`。

## 4. 算法流程

对于 stage 1/2，R11 先读取 C0 的旧 memory 和当前 delta，构造 C0 parent append 基础块。然后得到：

1. `P_initial`：由当前类别约束和初始 assignment 得到的软 assignment；
2. `P_joint`：R10 teacher joint 产生并封存的 assignment；
3. `P_alpha`：在 assignment 层做线性插值；
4. `X_new = P_alpha^T X_delta / column_mass`；
5. 对每个新行按 `P_alpha` 权重排序，选择最多 5 个文本来源，按 320 字符预算拼接并重新编码；
6. 用同一个 `P_alpha` 投影旧块与新块的支持关系，材料化二值拓扑；
7. 旧历史行的 `X/text/mass/depth/stage` 和旧历史边保持不变；
8. 在相同 student 协议下从头训练并报告 all-prompt pooled Macro-F1、old F1、new F1。

关键点是所有模态共享同一个 `P_alpha`。因此本轮测试的是 assignment anchor，而不是独立的 X、文本或拓扑替换。

## 5. 最终阶段结果

以下是 stage 2 的 all-prompt pooled Macro-F1；差值均相对同一 dataset/order 的 C0。

| 数据集 | 顺序 | C0 | P-initial | joint | anchor025 | anchor050 | anchor075 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | order19601 | 50.382 | 49.397 (-0.985) | 51.644 (+1.262) | 51.736 (+1.354) | 50.031 (-0.352) | 51.433 (+1.050) |
| Cora | order19602 | 53.048 | 47.185 (-5.862) | 48.005 (-5.043) | 52.037 (-1.011) | 48.315 (-4.733) | 48.858 (-4.189) |
| Cora | order19603 | 46.152 | 47.389 (+1.237) | 48.365 (+2.213) | 46.645 (+0.493) | 48.426 (+2.274) | 48.165 (+2.013) |
| Citeseer | order19601 | 54.535 | 57.203 (+2.668) | 55.517 (+0.982) | 54.484 (-0.051) | 54.702 (+0.167) | 53.556 (-0.980) |
| Citeseer | order19602 | 52.035 | 54.076 (+2.040) | 54.405 (+2.369) | 54.205 (+2.169) | 54.205 (+2.169) | 51.958 (-0.077) |
| Citeseer | order19603 | 52.933 | 50.578 (-2.356) | 50.843 (-2.090) | 50.829 (-2.104) | 51.527 (-1.407) | 50.173 (-2.760) |

## 6. 汇总结果

| 方法 | 最终 F1 | 最终相对 C0 | 最终 old F1 | 最终 new F1 | 阶段平均相对 C0 | 最差最终 cell | 正收益 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/6 |
| P-initial | 50.971 | -0.543 | +1.439 | -3.453 | -0.416 | -5.862 | 3/6 |
| joint | 51.463 | -0.051 | +1.342 | -1.917 | -0.920 | -5.043 | 4/6 |
| anchor025 | **51.656** | **+0.142** | +1.629 | -2.059 | -0.856 | **-2.104** | 3/6 |
| anchor050 | 51.201 | -0.313 | +2.193 | -3.945 | -0.737 | -4.733 | 3/6 |
| anchor075 | 50.690 | -0.824 | +1.237 | -3.883 | -1.535 | -4.189 | 2/6 |

最终平均不是事后挑选出的最好顺序，而是六个预先固定顺序/数据集 cell 的平均。`anchor025` 的 F1 最高，但没有达到预注册晋升条件 `+0.5 pp`。

## 7. Cora 与 Citeseer 的差异

| 数据集 | 方法 | 最终 F1 | 相对 C0 | old F1 差 | new F1 差 |
|---|---|---:|---:|---:|---:|
| Cora | C0 | 49.861 | 0.000 | 0.000 | 0.000 |
| Cora | joint | 49.338 | -0.523 | +2.239 | -4.205 |
| Cora | anchor025 | 50.139 | +0.279 | +2.600 | -2.817 |
| Citeseer | C0 | 53.168 | 0.000 | 0.000 | 0.000 |
| Citeseer | joint | 53.588 | +0.420 | +0.445 | +0.371 |
| Citeseer | anchor025 | 53.173 | +0.005 | +0.658 | -1.302 |

anchor025 对 Cora 的主要作用是减轻 order19602 的 joint 崩落；但 Cora 新类仍下降。Citeseer 上原始 joint 的平均收益主要来自 order19602，anchor 反而将这部分收益拉回，说明固定 alpha 不能同时适配两个数据集的 assignment 偏移。

## 8. 结果解释

### 8.1 anchor 确实解决了什么

R10 joint 在 Cora/order19602 的最终差值为 `-5.043 pp`。anchor025 将其改善到 `-1.011 pp`，这是本轮最清楚的稳定化信号。它说明 joint 的坏情况部分来自 `P_joint` 偏离 `P_initial` 过大，而不是所有联合 content-topology 机制都无效。

同时，anchor025 的最差最终 cell 为 `-2.104 pp`，比 joint 的 `-5.043 pp` 好约 `2.94 pp`。old F1 也从 joint 的 `+1.342 pp` 提高到 `+1.629 pp`。

### 8.2 anchor 没有解决什么

anchor025 的 new F1 仍下降 `-2.059 pp`。在 Cora 上，anchor025 的 new F1 差为 `-2.817 pp`；在 Citeseer 上为 `-1.302 pp`。因此它主要是在保护旧类传播和降低 assignment 过度偏移，并没有把新类信息有效转化为 student 可学习信号。

### 8.3 为什么不能直接宣布新方法

固定 alpha 的方向具有明显数据集/顺序依赖：

- `P-initial` 在 Citeseer 的两个顺序上有效，但在 Cora/order19602 和 Citeseer/order19603 退化；
- `anchor025` 在 Cora/order19602 有明显回退保护，但在 Citeseer/order19601/19603 没有收益；
- `anchor050` 获得较好的 old F1，但 new F1 平均下降更多；
- `anchor075` 整体最差。

这不是简单的“alpha 越小越好”或“joint 越强越好”，而是 old retention 与 new plasticity 的真实冲突。当前还没有一个不使用 report F1 的规则能够在这些 cell 上可靠决定 alpha。

## 9. 晋升门禁

| 条件 | 要求 | anchor025 结果 | 判定 |
|---|---:|---:|---|
| 最终平均相对 C0 | `>= +0.50 pp` | `+0.142 pp` | 未通过 |
| 阶段平均不低于 C0 | `>= 0` | `-0.856 pp` | 未通过 |
| 最差最终 cell | `>= -2 pp` | `-2.104 pp` | 未通过 |
| new F1 平均不下降 | `>= 0` | `-2.059 pp` | 未通过 |
| student 成本 | 不显著增加 | 约与 C0 相同 | 通过 |
| memory/历史访问审计 | 通过 | 通过 | 通过 |

结论是：R11 达到了“降低最差 joint 退化”的局部目标，但没有达到“稳定提高总体性能”的目标。

## 10. 当前主线与后续建议

### 当前保留方案

论文和后续比较继续使用 C0 作为冻结主线：C0 的 assignment、均值 X、`g=5/c=320` 文本重编码、C0 拓扑材料化和相同 student 协议。R10 joint 与 R11 anchor025 作为联合构造/稳定化的机制诊断，不放入主方法默认配方。

### 若继续优化，下一轮只做一个问题

不要同时加入动态预算、文本 attention、旧 memory revision 和新的 topology。下一轮应预注册一个“新类塑性约束的 anchor”实验：

1. 保留 `P_alpha` 的 assignment-level 结构；
2. 用不依赖最终 report F1 的新类风险代理，限制 anchor 只作用于高风险旧块/边界块；
3. 将 old retention 和 new plasticity 分开报告；
4. 增加一个 matched-random assignment displacement 对照；
5. 仍要求至少两个数据集、三个顺序、独立 student seeds 后才考虑晋升。

若该受控版本仍不能将新类差值拉回非负，应该停止继续堆叠 joint 模块，把 R10/R11 写成“联合构造的条件性协同与稳定性负结果”，C0 作为可靠方法基线。

## 11. 结果文件

- [R11 汇总报告](../../reports/tgc_r11_c0_anchor_20260923b/report.md)
- [R11 方法汇总 CSV](../../reports/tgc_r11_c0_anchor_20260923b/methods.csv)
- [R11 cell 汇总 CSV](../../reports/tgc_r11_c0_anchor_20260923b/cells.csv)
- [R11 对照差值 CSV](../../reports/tgc_r11_c0_anchor_20260923b/contrasts.csv)
- [R11 构造诊断 CSV](../../reports/tgc_r11_c0_anchor_20260923b/build_diagnostics.csv)
- [R11 原始结果目录](../../results/tgc_r11_c0_anchor_20260923b/)
- [R11 汇总脚本](../../tools/summarize_tgc_r11.py)
