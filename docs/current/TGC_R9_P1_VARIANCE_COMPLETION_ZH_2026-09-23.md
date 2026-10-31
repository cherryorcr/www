# TGC-R9 P1 Teacher-Response 方差分解完成报告

## 1. 结论摘要

P1 已完成并通过独立重载审计。实验没有把任何 arm 按 report-dev F1 选入主方法，而是用于定位 TGC-R 构造响应不稳定的来源。

最重要的结论是：

1. **拓扑材料化是当前最强的可观测因素。** 在固定 `H/P/X/text`、只把拓扑替换为 C0 拓扑时，12/12 个 order-stage cell 都下降，平均下降 `3.66 pp`，新类 F1 平均下降 `6.63 pp`。
2. **teacher group 存在显著方差，但没有稳定的全局优胜 seed。** `g1-g0` 的 12-cell 差值均值接近 0（`+0.02 pp`），标准差约 `3.04 pp`；Cora 平均下降 `1.50 pp`，Citeseer 平均上升 `1.54 pp`。
3. **P assignment 优化目前不能作为可靠增益。** 跳过 P 优化的 `p_initial` 相对 `g0` 平均下降 `0.45 pp`，新类 F1 平均下降 `1.72 pp`；说明 P 会改变构造，但当前目标与 downstream student 结果没有稳定对齐。
4. **文本材料化是数据集依赖因素。** 用 C0 文本替换 g0 文本后，Cora 平均下降 `1.39 pp`，Citeseer 平均上升 `1.04 pp`，不能宣称当前文本构造普适有效。
5. **递归 synthetic parent 目前接近中性。** `recursive_g0` 相对 `g0` 的 pooled 差值为 `-0.10 pp`；Citeseer 上升 `0.86 pp`，Cora 下降 `1.06 pp`，暂不适合作为默认主线。

因此，P1 不直接晋升 gate、动态预算或复杂 selector。下一步应优先分析和改进拓扑材料化，并用多 teacher 的均值与不确定性降低 response variance；P assignment 只能作为待校准组件，不能依据当前结果控制预算。

## 2. 协议与审计状态

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19600`、`order19609`、`order19610` |
| 增量阶段 | stage 1、stage 2；stage 0 作为共同 C0 parent |
| memory rate | 5% |
| teacher | 两层 GCN/GraphCLIP student，128 steps，batch 64，Adam `1e-3` |
| downstream student | 两层 GCN，冻结 MiniLM，双向 InfoNCE，batch 16，Adam `1e-3`，从头训练 |
| report seeds | 310、311、312 |
| 历史访问 | 旧 synthetic memory + 当前 raw delta；不回读历史完整 raw graph |
| 选择方式 | 不用 report F1 选择 arm；所有 arm 全部报告 |
| Cora 审计 | 162 fit rows、54 construction cells，最大 reload error `0.0` |
| Citeseer 审计 | 162 fit rows、54 construction cells，最大 reload error `0.0` |
| test labels | 未使用 |

审计修正说明：初次审计在非 canonical order 上没有恢复该 order 的 class stream，导致重载模型按错误 prompt scope 评分。修正 `audit_tgc_r9_variance.py` 后重新审计通过。该问题属于审计入口 bug，不改变已保存的训练 checkpoint、memory 或实验指标。

## 3. Downstream 结果

以下为 6 个 order-stage cell 的平均 all-prompt pooled Macro-F1（百分比）。`g0` 是方差分解的共同参考，不等同于论文最终主方法。

| Dataset | C0 | g0 | g1 | init swap | batch swap | P initial | text C0 | topology C0 | recursive g0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 42.79 | 44.30 | 42.80 | 42.20 | 44.25 | 43.80 | 42.91 | 39.37 | 43.23 |
| Citeseer | 56.67 | 57.43 | 58.97 | 57.86 | 56.57 | 57.03 | 58.47 | 55.04 | 58.29 |

相对 g0 的平均差值（pp）：

| Arm | pooled F1 | old F1 | new F1 | Cora F1 | Citeseer F1 | F1 正收益 cell |
|---|---:|---:|---:|---:|---:|---:|
| g1 | +0.02 | -0.06 | +0.08 | -1.50 | +1.54 | 8/12 |
| init swap | -0.84 | -1.19 | -0.58 | -2.10 | +0.43 | 6/12 |
| batch swap | -0.45 | -0.31 | -0.85 | -0.04 | -0.86 | 7/12 |
| P initial | -0.45 | +0.42 | -1.72 | -0.50 | -0.40 | 6/12 |
| text C0 | -0.18 | -0.27 | -0.33 | -1.39 | +1.04 | 6/12 |
| topology C0 | -3.66 | -1.16 | -6.63 | -4.92 | -2.39 | 0/12 |
| recursive g0 | -0.10 | -0.39 | -0.08 | -1.06 | +0.86 | 3/12（另有 6 个并列） |

## 4. 方差来源解释

### 4.1 Teacher 初始化与 batch 轨迹

`g0` 和 `g1` 同时改变 teacher 初始化及 batch trajectory。`init_swap`、`batch_swap` 用交叉替换将两者拆开。结果表明：

- 初始化单独变化的 pooled 影响为 `-0.84 pp`；
- batch trajectory 单独变化的 pooled 影响为 `-0.45 pp`；
- 两者都不是稳定的全局主导因素；
- 但 Cora 与 Citeseer 的方向相反，且 order 影响很大。

这说明当前不是简单的“换一个固定 teacher seed 就能解决”的问题。更合理的后续机制应保存 teacher ensemble 的均值、方差或 disagreement，而不是把单个 teacher 结果作为 gate 分数。

### 4.2 Assignment P

`p_initial` 与 g0 使用同一 `H`，只跳过 P 优化。因此它隔离了 assignment 优化本身的影响。P 优化确实产生了不同的 `P/X/text/topology`，但 downstream 结果没有稳定收益：

- pooled F1 `-0.45 pp`；
- new F1 `-1.72 pp`；
- Citeseer old F1 略有上升，但新类塑性下降；
- Cora 和 Citeseer 都没有稳定正收益。

结论是：当前 P 的 teacher 代理目标与 canonical student 的最终训练响应存在错位。下一版不能简单增加 P 的优化步数或把 P 直接接入预算分配，应先做 construction regret 分解和 student-response 对齐。

### 4.3 文本材料化

`text_c0` 保持 g0 的 `H/P/X/topology`，只替换新块文本。结果显示文本影响真实存在，但方向依赖数据集：

- Cora：C0 文本替换使 F1 下降 `1.39 pp`，new F1 下降 `1.05 pp`；
- Citeseer：C0 文本替换反而上升 `1.04 pp`；
- pooled 只有 `-0.18 pp`，不能把当前 concat/re-encode 视为稳定增益。

文本机制可以继续研究，但必须固定 token budget，并把 graph-conditioned selection、text-only attention、shuffle control 和 concat 作为严格对照；目前没有足够证据把它放入主方法。

### 4.4 拓扑材料化

`topology_c0` 保持 g0 的 `H/P/X/text`，只替换拓扑。它在全部 12 个 cell 都下降，是本轮最强的因果信号：

- pooled F1 平均下降 `3.66 pp`；
- new F1 平均下降 `6.63 pp`；
- Cora 下降 `4.92 pp`，Citeseer 下降 `2.39 pp`。

这说明拓扑不是附带实现细节，而是当前凝聚结果能否被 student 使用的关键。后续应优先分解：块内边、跨块边、旧块归一化传播、跨边度数和 edge weight；然后再判断收益来自新类传播、旧类保持，还是仅仅来自 C0 拓扑的结构差异。

### 4.5 Recursive parent

`recursive_g0` 在 stage 2 使用自己 stage 1 的 synthetic memory 作为 parent，而其它 arm 使用 C0 parent。结果近似中性且数据集依赖：Cora 下降、Citeseer 上升。它说明递归 rollout 会传播历史构造误差，但并未在当前规模形成稳定灾难或稳定收益。需要更长链和独立 order 才能判断是否值得设计 refresh/merge。

## 5. 对后续算法设计的影响

### 不应立即做的事情

1. 不把 `g1`、`batch_swap` 或某个 order 上的最佳 arm 直接晋升为主方法。
2. 不用最终 report F1 训练或选择 gate。
3. 不让当前 P assignment 直接决定额外预算。
4. 不在没有拓扑控制实验前把 text attention、动态 budget 和 old-memory revision 同时加入。

### P1 下一阶段优先级

| 优先级 | 实验 | 目的 | 晋升判据 |
|---|---|---|---|
| P1-A | 拓扑因果拆解：C0 topology、g0 topology、无跨边、随机端点、固定边数、weighted/binary | 判断真正有用的拓扑算子 | 至少两个数据集/独立 order 平均正收益，且 new/old trade-off 可解释 |
| P1-B | Teacher ensemble disagreement：多 seed teacher 的均值、方差、最坏分歧 | 把 teacher variance 转化为可用的不确定性 | 能预测 construction regret，且不读取 report F1 |
| P1-C | P/student-response 对齐：固定 H，比较 P initial、teacher proxy P、短 student response P | 判断 P 是否值得保留 | construction regret 与 downstream 增益有稳定相关性 |
| P1-D | 递归 parent 长链：3/5/8 stages，固定 parent 与 order | 判断历史误差是否累积 | 至少两个数据集的长期保持或塑性收益 |

## 6. 当前可采用的方法定位

当前可以可靠描述为：

> 一个在历史 raw graph 不可回读条件下，利用 teacher-response 诊断合成内容、文本和拓扑材料化方差的持续 TAG 凝聚框架。

当前还不能描述为：

> 一个已经证明 teacher-guided gate、动态预算或 student-compatible selector 普遍优于 C0 的最终算法。

P1 的价值不是立即提高主表分数，而是把“为什么复杂 TGC 组件不稳定”拆成了可审计的因素，并明确了下一步应优先处理 topology 和 teacher uncertainty，而不是继续盲目堆叠模块。

## 7. 产物

- 协议：[TGC_R9_P1_VARIANCE_PROTOCOL_ZH_2026-09-23.md](TGC_R9_P1_VARIANCE_PROTOCOL_ZH_2026-09-23.md)
- Cora 结果：[variance_metrics.csv](../../results/tgc_r9_variance_20260923b_cora/variance_metrics.csv)
- Citeseer 结果：[variance_metrics.csv](../../results/tgc_r9_variance_20260923b_citeseer/variance_metrics.csv)
- Cora 审计：[r9_audit.json](../../results/tgc_r9_variance_20260923b_cora/r9_audit.json)
- Citeseer 审计：[r9_audit.json](../../results/tgc_r9_variance_20260923b_citeseer/r9_audit.json)
