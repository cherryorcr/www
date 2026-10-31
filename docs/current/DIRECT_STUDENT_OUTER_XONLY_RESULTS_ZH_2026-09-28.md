# Direct Student Outer X-only 实验报告

> **校正说明**：本文档第 3 节中的首轮四-cell 表格已被后续审计发现存在 `order19102` 的 `old_rows` 边界错误，不能作为最终结果。请以[修正协议与 Prompt 目标实验报告](DIRECT_STUDENT_OUTER_CORRECTED_PROMPT_RESULTS_ZH_2026-09-28.md)为准；本文档保留首轮运行轨迹，便于追溯错误是如何被发现的。

日期：2026-09-28  
状态：四个小图 cell 已完成构造、六个 Student seed 下游评估和 stop-gradient 负对照。

## 1. 本轮问题

本轮直接验证：能否把 Student 的训练反馈反向传到合成节点特征 `X`，而不是只用启发式的 Mean-X 或距离分配。

当前实验只更新 `X`，暂不更新软分配 `P` 和拓扑 `A`，以便把目标函数问题与联合变量问题分开。

构造阶段使用：

- 上一阶段 synthetic memory + 当前 raw delta；
- 不读取历史完整 raw graph；
- 不使用 report/test 标签；
- canonical 两层 GCN + 可训练 text projection + 双向图文 InfoNCE Student；
- inner Student unroll 2 步；outer Adam 2 步；outer learning rate `0.01`；
- 全部 memory rows 可编辑；旧行半径 `0.05`，新行半径 `0.20`；
- 下游使用 Student seeds `150--155`，保持与 C0 reference 完全一致。

`stage=1` 是脚本的零基第二阶段，即本报告中的 Stage-2。

## 2. 算法流程

1. 固定 candidate memory 和合法 support，并对 old/new support 各自做互斥的 train/validation 切分。
2. 从同一 Student 初始化开始，在 candidate synthetic graph 上进行可微分的 2 步 Adam unroll。
3. 用 unroll 后 Student 在 support old/new rows 上计算双向 InfoNCE：

   `L_outer = 0.25 * L_old + 1.0 * L_new + regularization`

4. 对 `X` 反向传播并更新 outer variable。
5. 用 trust region 材料化候选，保持 memory rows、文本、质量字段和拓扑不变。
6. 用固定的 canonical Student 独立训练 6 个 seed，报告最终 Macro-F1、Accuracy、old-class F1 和 new-class F1。

这证明了 `X` 上确实存在非零的 Student outer gradient，但没有预先假设降低 InfoNCE 就等价于提高下游分类。

## 3. 下游结果

所有差值均为 `X-only - C0/reference candidate`，单位为百分点。Base 和 Candidate 使用同一批 Student seeds、同一 stage 和同一 report-dev mask。

| 数据集/顺序 | Base F1 | X-only F1 | Delta F1 | Base Accuracy | X-only Accuracy | Delta Accuracy | Delta New F1 | Delta Old F1 | Outer objective | 最大相对位移 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora / canonical | 51.441 | 49.151 | -2.291 | 44.663 | 41.379 | -3.284 | -4.032 | +0.047 | 2.6971 -> 2.6851 | 0.20 |
| Cora / order19102 | 53.339 | 54.718 | **+1.379** | 53.917 | 55.000 | **+1.083** | **+3.220** | -0.593 | 2.5906 -> 2.5681 | 0.20 |
| Citeseer / canonical | 69.348 | 68.314 | -1.034 | 65.854 | 66.057 | +0.203 | -4.352 | **+3.363** | 2.9792 -> 2.9672 | 0.20 |
| Citeseer / order19102 | 59.755 | 59.200 | -0.556 | 53.626 | 52.371 | -1.255 | -1.709 | -0.426 | 2.7240 -> 2.7298 | 0.20 |
| **四 cell 平均** | - | - | **-0.626** | - | - | **-0.813** | **-1.718** | **+0.598** | - | - |

补充的 Student-seed 统计：

- Cora/canonical：6 个 seed 中 2 个 F1 为正；
- Cora/order19102：6 个 seed 中 4 个 F1 为正；
- Citeseer/canonical：6 个 seed 中 0 个 F1 为正；
- Citeseer/order19102：6 个 seed 中 4 个 F1 为正，但一个负向 seed 的幅度较大，拉低了均值。

因此不能把 Cora/order19102 的提升解释为稳定晋升。

注意：`selected_validation_objective` 包含 X anchor regularizer，而 `baseline_validation` 是未加该 regularizer 的 support loss；它们可以用于看趋势，不能作为严格同量纲的最终比较。Citeseer/order19102 的选择目标甚至略有变差，这与其下游退化一致。

## 4. 梯度路径与负对照

四个 X-only audit 均满足：

- `x_relative` outer gradient norm 非零；
- 输出最大相对位移为 `0.20`，没有越过 trust region；
- memory row 数不变；
- 历史 raw 读取数为 `0`；
- report/test 标签未用于选择。

另行运行 stop-gradient control：它直接复制 base memory，`max_relative_displacement=0`，`nonzero_outer_gradient=false`。这确认本轮候选差异确实来自 `X` 的外层更新，而不是评估脚本或文件重排造成的差异。

## 5. 结果说明

### 5.1 不是“梯度没有作用”

梯度路径、Adam unroll、trust region 和材料化审计都正常工作。Cora/order19102 的 `new F1 +3.220 pp` 也说明该路径在某些顺序上可以改善新类塑性。

### 5.2 当前 outer objective 与真正下游目标不一致

当前 outer objective 优化的是 support 节点上的图文实例级 InfoNCE，而最终评价是：

`Student graph embedding -> class prompt embedding -> zero-shot classification`

两者并不相同：

- InfoNCE 约束每个 support row 与自己的文本 embedding 匹配；
- 下游分类要求同一类别的节点靠近同一类 prompt，并与其它类 prompt 拉开 margin；
- support graph 的传播结构与最终真实评估图不同；
- `new_weight=1.0` 只是 InfoNCE 权重，不等于最大化 new-class F1；
- X 同时影响旧节点传播、候选节点传播和归一化邻接后的消息混合，局部 loss 下降可能破坏下游分类边界。

### 5.3 新类塑性仍是主矛盾，但不是简单的 old/new 权重问题

四个 cell 平均 old F1 上升 `+0.598 pp`，new F1 下降 `-1.718 pp`。但不同顺序的方向不一致，说明还存在：

- 类别顺序导致的 support 几何差异；
- synthetic rows 与 incoming classes 的对应关系没有进入 outer loss；
- prompt 分类 margin 没有被直接优化；
- 真实下游使用 full graph，而 outer 只看到 candidate/support 的局部传播。

因此单纯继续扫描 `old_weight`、`new_weight` 或增大 outer steps 不足以解决问题。

## 6. 当前决策

本轮 `X-only` 暂不进入主方法，也暂不直接运行 `P-only`、`A-only` 或 `X+P+A`。原因不是否定 Student-compatible optimization，而是先修正目标函数；否则联合变量只会把同一个错配目标放大，并且无法判断退化来源。

## 7. 下一步最小改动

下一轮只替换 outer objective，保留 X trust region、Student、拓扑、文本和预算不变：

1. 保留当前 old/new InfoNCE 作为稳定项；
2. 从当前 raw delta 的类别协议中读取 incoming class identity 和对应 prompt embedding；
3. 在同一 Student unroll 后，用 support graph 上的 incoming raw rows 计算 prompt-aware classification margin；
4. 只使用 construction-time 的当前阶段类别信息，不使用 report/test 标签；
5. 对新增 synthetic slots 使用 class mass/provenance 约束，避免把一个 slot 的梯度平均分给不相关类别；
6. 先做 `X-only + prompt-margin`，再做 `X-only + InfoNCE + prompt-margin`，最后才考虑 P/A。

晋升条件：至少两个数据集、两个独立顺序平均不低于 reference，new F1 平均为正，最差 cell 不低于 `-2 pp`，且构造成本不超过当前 X-only 的 `2x`。达不到则保留为机制诊断，不升为主方法。

## 8. 产物

- 构造代码：[direct_student_outer.py](../../code/v2_llm_fusion/direct_student_outer.py)
- 运行入口：[run_direct_student_outer.py](../../code/v2_llm_fusion/run_direct_student_outer.py)
- 单元测试：[test_direct_student_outer.py](../../code/v2_llm_fusion/test_direct_student_outer.py)
- 本地评估 JSON：`results/direct_student_outer_20260928/eval_xonly_*.json`
- 本地构造审计：`results/direct_student_outer_20260928/audit_xonly_*.json`
