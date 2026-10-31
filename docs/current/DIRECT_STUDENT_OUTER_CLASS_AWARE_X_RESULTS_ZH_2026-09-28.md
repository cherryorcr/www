# Class-aware X-v2 外层优化实验报告

日期：2026-09-28  
状态：四个小图 cell 已完成构造、配对评估和不变量审计。

## 1. 实验问题

上一轮直接对合成特征 `X` 做 Student 外层反向优化时，当前到达 delta 中的不同新类共享同一组 `X` 梯度。即使 outer loss 能够反向传播到 `X`，一个新类的更新也可能被另一个新类的梯度抵消。

本轮只改变梯度路由：

```text
普通 X-only：所有新 synthetic rows 接收混合的新类梯度
x-class：每个新类单独计算 support objective，只更新该类对应的 synthetic slots
```

`P`、文本 embedding、拓扑 `A`、旧 synthetic rows、总 memory rows、Student 和评估协议均保持不变。

## 2. 严格协议

| 项目 | 设置 |
|---|---|
| Student | canonical 两层 GCN + 双向图文 InfoNCE |
| outer inner steps | 2 |
| outer steps | 2 |
| outer learning rate | 0.01 |
| support loss | old/new InfoNCE 权重 0.25/1.0 |
| X trust region | old 0.05；new 0.20 |
| 可编辑行 | 仅当前阶段新增 synthetic rows |
| 训练/验证 seeds | 9300 / 9310 |
| 下游 Student seeds | 150--155 |
| 评估 | report-dev，未用于构造或选择 |
| 历史 raw graph | 不读取；仅 previous synthetic memory + current delta support |
| 改变的模块 | X 的 class-wise outer gradient routing |

本轮 outer 构造在远端 `tagsam` 环境执行。由于双卡同时可见时 PyTorch 2.4.1+cu121 出现 CUDA device 初始化错误，最终构造和评估使用 `CUDA_VISIBLE_DEVICES=0` 的单卡配置；GPU 单卡 smoke test 通过。这个环境修正不改变算法参数。

## 3. 类别到 synthetic slot 的合法映射

不能直接假设 synthetic row 的来源类。映射脚本使用当前 delta 的 construction-time class id，按与 C0 相同的 dispersion 配额生成连续新块，并要求每类 synthetic mass 与该类当前 raw support 行数精确相等；不满足就拒绝运行。

| 数据集/顺序 | old rows | 新类 | slot 配额 | 当前 raw mass |
|---|---:|---|---|---:|
| Cora/canonical | 25 | 2, 3 | 25, 17 | 567, 291 |
| Cora/order19102 | 35 | 1, 3 | 14, 15 | 295, 291 |
| Citeseer/canonical | 38 | 2, 3 | 24, 24 | 474, 479 |
| Citeseer/order19102 | 40 | 1, 3 | 21, 22 | 383, 479 |

四个 cell 均通过 `mass_matches_source_count=true`。旧 rows 的 class id 保持 `-1`，因此本轮不会用历史 raw 标签推断旧 memory 来源。

## 4. 结果：相对 C0

指标单位为百分点（pp）。`Final F1` 是全部已见节点的 cumulative Macro-F1；`New F1` 是当前新类；`Old F1` 是旧类平均；Accuracy 是 cumulative accuracy。

| Cell | C0 Final F1 | x-class Final F1 | Delta | C0 Acc. | x-class Acc. | Delta | New F1 Delta | Old F1 Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora/canonical | 52.068 | 47.206 | **-4.861** | 44.253 | 40.640 | -3.612 | -8.492 | +1.587 |
| Cora/order19102 | 53.339 | 56.064 | **+2.725** | 53.917 | 55.417 | +1.500 | **+6.013** | -0.383 |
| Citeseer/canonical | 69.348 | 68.339 | -1.009 | 65.854 | 64.837 | -1.016 | -3.243 | +0.852 |
| Citeseer/order19102 | 59.755 | 61.595 | **+1.840** | 53.626 | 54.324 | +0.697 | +0.513 | +0.811 |
| **四 cell 平均** | - | - | **-0.326** | - | - | **-0.608** | **-1.302** | **+0.717** |

正向 cell 为 2/4。两个 `order19102` cell 正向，两个 canonical cell 负向，说明类别顺序仍是主要影响因素。Cora/order19102 的新类 F1 提升最大，但不能单独作为泛化证据。

## 5. 相对普通 X-only

普通 X-only 的四 cell 平均 Final F1 Delta 为 -0.931 pp；x-class 为 -0.326 pp。因此 class-wise routing 带来约 **+0.604 pp** 的平均改善，且把平均 Old F1 从 -0.744 pp 改为 +0.717 pp。

但它没有稳定改善 New F1：平均 New F1 仍为 -1.302 pp。逐 cell 的 x-class 相对普通 X-only 如下：

| Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora/canonical | -1.906 | +0.657 | -3.814 | +3.601 |
| Cora/order19102 | +2.565 | +2.083 | +5.512 | +0.239 |
| Citeseer/canonical | -0.773 | -1.152 | -1.061 | -0.855 |
| Citeseer/order19102 | +2.530 | +1.534 | +0.069 | +2.858 |
| **平均** | **+0.604** | **+0.781** | **+0.176** | **+1.462** |

这里的平均改善是“相对普通 X-only 的改善”，不是相对 C0 的正收益。它证明了路由错误是一个真实因素，但没有证明 class-aware outer objective 已经解决 Student-compatible condensation。

## 6. 逐 Student seed 配对观察

相对 C0 的 Final F1 配对变化：

| Cell | 六个 seed 的 pp 变化 | 正向 seed 数 |
|---|---|---:|
| Cora/canonical | -3.210, -0.071, -6.074, -4.870, -0.041, -14.902 | 0/6 |
| Cora/order19102 | -1.509, -3.209, +2.271, +12.055, +5.679, +1.064 | 4/6 |
| Citeseer/canonical | +0.725, -2.761, -5.145, -1.189, +4.871, -2.557 | 2/6 |
| Citeseer/order19102 | +7.835, +5.845, -3.759, -3.278, +3.017, +1.377 | 4/6 |

这也说明 class-aware 版本的响应方差仍然较大。当前报告中不把单个 seed 的大幅提升当成确定性结论。

## 7. 构造审计

四个 candidate 的共同审计结果：

| Cell | outer variable | max relative displacement | mean displacement | historical raw reads | changed fields |
|---|---|---:|---:|---:|---|
| Cora/canonical | `x_class_relative` | 0.200 | 0.119 | 0 | new X only |
| Cora/order19102 | `x_class_relative` | 0.196 | 0.089 | 0 | new X only |
| Citeseer/canonical | `x_class_relative` | 0.200 | 0.102 | 0 | new X only |
| Citeseer/order19102 | `x_class_relative` | 0.200 | 0.092 | 0 | new X only |

文本、`P`、拓扑、mass、depth、stage、rows 均不变。class-aware 测试和审计也确认旧 X 行没有被修改。

## 8. 结果解释

### 8.1 已确认的正信号

1. **新类梯度混合确实是问题。** 把新类 support 分开并路由到对应 slots 后，平均 Final F1 相对普通 X-only 提升约 0.604 pp。
2. **类感知路由改善了旧类保持。** 相对普通 X-only，平均 Old F1 提升约 1.462 pp。
3. **梯度路径和类映射都是真实执行的。** 审计显示变量是 `x_class_relative`，位移达到 new trust region，mass 映射逐类守恒。

### 8.2 尚未解决的问题

1. **外层 support objective 仍与最终 Student 泛化错位。** 它优化的是短 unroll 后的 support InfoNCE，而最终评估是独立 Student 在真实 full graph 上的 prompt 分类。
2. **类内 slot 仍然过于粗糙。** 一个类的所有 slots 接收同一个类级梯度，但类内不同子簇的局部结构、文本和传播角色仍未区分。
3. **new trust region 可能过大。** 多个 cell 的最大位移达到 0.20，说明 outer 更新容易撞到边界；这会放大 proxy loss 的误导。
4. **构造映射只解决“属于哪个类”，没有解决“类内哪个 raw 子群对应哪个 slot”。** 这正是下一步应做的 provenance/子簇路由，而不是继续扫 prompt 权重。

## 9. 是否晋升为主方法

目前不能。正确结论是：

> class-aware X-v2 是比普通 X-only 更合理的 Student 外层优化机制，并提供了新类梯度混合问题的正向诊断证据；但在四个小图 cell 中只相对 C0 提升 2/4，canonical 顺序下降，平均 New F1 仍下降 1.302 pp，因此暂不作为主方法。

这不是“梯度优化没有价值”，而是说明仅把总 delta 按类拆开还不够。它应保留为下一版的基础组件和机制证据。

## 10. 下一步建议

优先级应是：

1. **类内 provenance 路由：** 在每个 class block 内按 joint graph-text 子簇建立 raw support 到 synthetic slot 的软分配；每个 slot 只接收其负责子簇的 outer gradient。仍保持 `P`、文本、拓扑固定。
2. **缩小并退火 new trust region：** 先比较 0.05、0.10、0.20；使用固定候选配置，不按 report-dev 选最优，避免把参数调到特定 cell。
3. **改善 outer horizon：** 先固定 class/provenance 路由，再比较 2/4/8 inner steps；同时冻结 text projection 和 logit scale，防止 Student 通过文本分支投机降低代理损失。
4. **联合 old/new constraint：** 保留 C0 anchor，用 old support response 约束旧类保持，以 new support response 约束新类塑性；失败时回退到 C0。
5. **独立顺序确认：** 只有在至少两个数据集、独立类别顺序和多个 seed 上稳定提升后，才考虑 `P/A` 联合优化。

暂不建议：继续增大 prompt CE/margin、直接启用 `X+P+A`、动态预算、无界旧 memory refresh 或同时改变 Student/文本/拓扑。

## 11. 产物

- 实现：[direct_student_outer.py](../../code/v2_llm_fusion/direct_student_outer.py)
- 运行入口：[run_direct_student_outer.py](../../code/v2_llm_fusion/run_direct_student_outer.py)
- 类映射准备：[prepare_class_aware_x_support.py](../../code/v2_llm_fusion/prepare_class_aware_x_support.py)
- 四个 class-aware 构造：`results/direct_student_outer_20260928/class_aware_x_*`
- 四个配对评估：`results/direct_student_outer_20260928/eval_class_aware_x_*.json`
- 类映射审计：`results/direct_student_outer_20260928/class_slot_map_*/audit.json`
- 前一轮 X-only/prompt 结果：[DIRECT_STUDENT_OUTER_CORRECTED_PROMPT_RESULTS_ZH_2026-09-28.md](DIRECT_STUDENT_OUTER_CORRECTED_PROMPT_RESULTS_ZH_2026-09-28.md)
