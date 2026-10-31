# TGC-P1：transfer-aware X 与 student warm-start 小实验

日期：2026-09-24  
状态：smoke 完成，两个方向均未晋升。  
范围：Cora、`order19601`、stage 1、student seeds 150/151。

## 1. 实验目的

P0 归因表明，真实新类 prompt margin 比 synthetic loss 更能解释 new-F1。因此本轮先验证两种低风险修正：

1. 固定 C0 的 rows、文本、mass、旧行和 C0 topology，只对当前新块的 `X` 做 differentiable student-response 优化；目标是合法 current-delta 子图上的无标签图文 InfoNCE。
2. 固定 C0 memory，只将当前 student 从上一阶段 C0 checkpoint warm-start，判断新类塑性是否主要来自 scratch 初始化。

两项都没有改变主协议 C0，也没有使用 report F1 选择构造参数。

## 2. Transfer-aware X

### 2.1 设计

- base：封存 R15 C0 memory；
- 只修改 stage 1 新行 X；
- text、old X、mass、depth、stage、row budget、C0 topology 全部冻结；
- current raw delta 按固定 80/20 划分 construction/probe；
- inner student seeds：7800/7801；
- smoke inner horizon：8 steps，outer steps：4；
- 第一版 radius：0.10；第二版 radius：0.05，并加入 old synthetic response penalty；
- 下游评估仍使用标准 150/151 scratch student 和完整真实图。

### 2.2 结果

| 版本 | radius | old guard | seed 150 final delta | seed 151 final delta | 平均 final delta | 平均 new-F1 delta |
|---|---:|---|---:|---:|---:|---:|
| transfer-X v1 | 0.10 | 无 | -2.297 | -1.652 | -1.974 | -1.644 |
| transfer-X v2 | 0.05 | synthetic old loss | -3.401 | +1.215 | -1.093 | -1.233 |

v1 的 probe transfer loss 约从 `1.475` 降到 `1.469`，但 downstream F1 下降；v2 的 old penalty 在 smoke 中没有激活，两个 seed 的结果仍然方向相反。说明：

- 短 inner-horizon 的 current-delta InfoNCE 改善不能稳定预测长训练 student；
- 只修改 X 会通过固定跨边改变旧节点传播，即使 old X 没动，old-F1 仍可能下降；
- 简单 radius/old-loss 调参不足以把该组件变成可靠主方法。

因此不运行 transfer-X 的完整 6-seed、6-cell 矩阵，不把它晋升为候选版本。

## 3. Student warm-start

同一 Cora/order19601/stage1，C0 memory 不变：

| seed | scratch final F1 | warm-start final F1 | delta | scratch new F1 | warm new F1 | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 58.855 | 56.723 | -2.132 | 70.515 | 67.613 | -2.902 |
| 151 | 59.781 | 51.050 | -8.731 | 67.134 | 59.488 | -7.646 |

warm-start 同时损害 old/new F1，说明当前问题不是简单的 scratch 初始化不足。上一阶段参数会把旧阶段的表示偏置带到新阶段，而当前 memory/text/topology 没有足够机制承载新的类别边界。

## 4. 决策

### 不继续扩展

- 不继续扫描 transfer-X 的 outer lr、radius 和单一 old penalty；
- 不将 warm-start 作为默认 student；
- 不把这两个 smoke 的单个正 seed 写成方法收益。

### 下一版真正需要改变的对象

下一步应从“单独改 X”升级为**共享 assignment 下的内容-文本协同构造**：

1. 一个当前 delta assignment 同时决定 X prototype 和 text source set；
2. 以 transfer probe 作为独立诊断约束，而不是直接反向选择最终 F1；
3. 允许有限的类内多原型，但固定总 rows 和 C0 topology；
4. 把 new-class response 与 old retention 作为成对约束；
5. 先在一个独立顺序上做 smoke，再按晋升条件扩展到三顺序两数据集。

这条路线仍然保留 TGC 的核心创新方向，但避免把一个不稳定的 response selector 或单变量 X 优化直接当作主算法。

## 5. 可复现文件

- P1 transfer-X：[run_tgc_p1_transfer_x.py](../../code/v2_llm_fusion/run_tgc_p1_transfer_x.py)
- P1 remote entry：[tgc_p1_transfer_x_remote.py](../../tools/tgc_p1_transfer_x_remote.py)
- transfer-X v1：[tgc_p1_transfer_x_smoke_20260924c](../../results/tgc_p1_transfer_x_smoke_20260924c/)
- transfer-X v2：[tgc_p1_transfer_x_smoke_20260924d](../../results/tgc_p1_transfer_x_smoke_20260924d/)
- warm-start：[run_tgc_student_protocol_smoke.py](../../code/v2_llm_fusion/run_tgc_student_protocol_smoke.py)
- warm-start 结果：[tgc_student_protocol_smoke_20260924b](../../results/tgc_student_protocol_smoke_20260924b/)
