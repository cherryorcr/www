# TGC-R9 P1：Teacher-Response 方差分解协议

## 1. 目的

R8 表明固定 `lambda` 的排序会随 teacher group、数据集和类别顺序变化。P1 不再添加新的 gate 或文本模块，而是固定同一个 parent、当前 raw delta、预算和 canonical student，定位构造方差的来源：

1. teacher 初始化；
2. teacher batch/训练轨迹；
3. assignment `P` 是否优化；
4. 文本材料化；
5. 拓扑材料化；
6. 递归 parent synthetic memory。

本实验是诊断实验，不从 report-dev F1 选择主方法。

## 2. 固定协议

- 数据集：Cora、Citeseer。
- 类别顺序：`order19600`、`order19609`、`order19610`。
- 阶段：stage 1、stage 2；stage 0 只作为共同 C0 parent。
- memory rate：5%，预算、C0 新块、source enumeration 和 class mask 固定。
- teacher：两层 GraphCLIP/GCN student，128 steps，batch 64，Adam `1e-3`。
- downstream student：canonical 两层 GCN，冻结 MiniLM，双向 InfoNCE，batch 16，Adam `1e-3`，从头训练，独立 report seeds `310,311,312`。
- 构造访问范围：上一阶段 synthetic memory 与当前 raw delta；不回读历史完整 raw graph。
- 评价：report-dev all-prompt pooled Macro-F1，同时报告 old/new F1、construction cost 和 memory hash。

## 3. Factorial arms

| arm | 控制变量 | 目的 |
|---|---|---|
| `g0` | init 与 batch 都使用 teacher group 0 | 参考构造 |
| `g1` | init 与 batch 都使用 teacher group 1 | 总 teacher group 差异 |
| `init_swap` | group 1 初始化 + group 0 batch 轨迹 | 估计初始化贡献 |
| `batch_swap` | group 0 初始化 + group 1 batch 轨迹 | 估计训练轨迹贡献 |
| `p_initial` | 与 `g0` 相同的 `H`，跳过 assignment 优化 | 估计 `P` 优化的作用 |
| `text_c0` | 与 `g0` 相同的 X/topology/P，新块文本替换为 C0 文本 | 估计文本材料化差异 |
| `topology_c0` | 与 `g0` 相同的 X/text/P，拓扑替换为 C0 拓扑 | 估计拓扑材料化差异 |
| `recursive_g0` | g0 机制，但 stage 2 使用自己 stage 1 的 synthetic parent | 估计递归 parent 传播 |

`g0`、`g1`、`init_swap`、`batch_swap`、`p_initial`、`text_c0` 和 `topology_c0` 默认使用同一个 C0 parent；只有 `recursive_g0` 在 stage 2 改用自己的历史 synthetic 输出。

## 4. 保存的诊断

每个 cell 保存：

- teacher init/batch seed、batch sequence hash、最终 teacher state hash；
- `H` hash 与相对 `g0` 的最大绝对差；
- `P` hash、相对 `g0` 的最大绝对差、`P_initial -> P_optimized` 位移；
- X、文本 embedding、文本字符串、边集合的差异；
- parent/base/delta/memory hash；
- canonical student 每个 report seed 的 Macro-F1、old/new F1 和成本。

## 5. 解释规则

- `init_swap` 与 `g0` 差异明显、`batch_swap` 接近 `g0`：主要是 teacher 初始化敏感；
- `batch_swap` 与 `g0` 差异明显、`init_swap` 接近 `g0`：主要是 teacher 训练轨迹敏感；
- `H` 相同而 `P` 或 student 结果变化：assignment 优化是主要来源；
- `P` 相同但 text/topology 变体改变 F1：材料化是主要来源；
- stage 2 的 `recursive_g0` 相对 C0-parent g0 明显改变：递归 parent 是主要来源；
- 构造差异很大但 student F1 不变：该差异可能是不可识别的表示噪声；
- F1 只在一个顺序提升：记录为局部信号，不晋升为 gate。

## 6. 晋升前置条件

P1 本身不晋升算法。只有当方差来源明确后，才设计 bounded gate。任何后续 gate 必须：

- 使用独立 selection seeds；
- 不读取最终 report F1；
- 与固定 lambda、随机 gate、C0 做匹配对照；
- 同时报告 new plasticity、old retention、最差顺序和构造成本；
- 在至少两个数据集和独立类别顺序上重复确认。
