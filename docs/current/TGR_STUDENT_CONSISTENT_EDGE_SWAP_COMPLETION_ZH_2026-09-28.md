# Student-consistent bounded cross-edge swap：小图验证报告

日期：2026-09-28  
状态：实现与审计完成；组件未通过性能晋升条件，不进入 Semantic-Recursive 主方法。

## 1. 本轮回答的问题

本轮验证一个最小的 Student-consistent topology component：只允许修改 Semantic-Recursive
(SR) synthetic memory 中的旧新跨块边，并用一个固定短程 Student 训练响应决定是否接受交换。
它不修改合成特征、文本、预算、初始分配或旧记忆节点内容。

脚本中的比较臂名称仍使用 `c0`，但本轮 `c0` 实际读取的是封存的
`semantic_perm_memory.pt`，因此下文统一称为 `SR`；不能把它误解为另一个原始 C0 实现。

## 2. 方法的完整流程

对阶段 `t`，设 parent memory 的旧行数为 `N_old`，当前新增 synthetic rows 为
`N_new`。

1. 读取 SR parent 和当前 delta 的合法支持信息。历史完整 raw graph 不用于恢复旧节点特征、文本或标签。
2. 固定 SR 的 `X`、`text_emb`、rows、memory budget、初始 `P`、旧旧边和新新边。
3. 从 message-balanced support 构造合法的新旧跨边候选池。当前验证固定 `cross_degree=1`，并对旧端点设置容量上限。
4. 对当前跨边枚举 two-switch：两个新行交换一个旧端点。交换后严格保持：
   - 每个新行的跨边数量；
   - 每个旧端点的跨边容量；
   - old-old 边；
   - new-new 边。
5. 对每一个候选拓扑，在相同初始化下训练 canonical Student 一步：两层 GCN、冻结 MiniLM
   文本表示、双向图文 InfoNCE，selector 使用 seed `17`、`lr=1e-3`、batch `16`。
6. 计算旧块 loss reduction `old_gain`、新块 loss reduction `new_gain` 和一步后的总 loss。
   候选必须同时满足：

   `new_gain >= 0`，`old_gain >= 0`，`total_after <= base_total_after`。

7. 每个 cell 最多接受一次交换。没有候选通过门控时精确回退 SR；不根据 report/test 标签选择。
8. 下游评估仍用原 Student 和相同 Student seeds `150--155`，比较 Macro-F1、Accuracy、new-class F1、
   old-class F1 以及 stage-average F1。

这不是更新 `X` 或旧记忆的算法。它是一个受限跨边交换诊断，用来测试“短程 Student 响应是否能指导
新旧传播关系”。

## 3. 实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 开发顺序 | `order19611`、`order19612` |
| 独立顺序 | `order19613`，不参与参数选择 |
| 阶段 | stage 1、stage 2；每个 cell 使用 SR parent |
| 固定对象 | `X`、文本、预算、初始 `P`、old-old/new-new 边、Student |
| selector | 1 个固定 seed，1 个 Adam step，最多 1 个 two-switch |
| 下游 Student seeds | `150--155` |
| 访问约束 | 不读历史完整 raw feature/text；不使用 report/test 标签 |

共完成 12 个构造/评估 cell：开发顺序 8 个，独立顺序 4 个。

## 4. 逐 cell 结果

下表均为 `Student-consistent - SR` 的百分点差值。`0.000` 表示门控失败后精确回退，
不是“交换后没有影响”。

| 数据集 | 顺序 | 阶段 | Macro-F1 | Accuracy | New F1 | Old F1 | Stage-average F1 | accepted |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Citeseer | order19611 | 1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Citeseer | order19611 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Citeseer | order19612 | 1 | +2.522 | +1.375 | -1.761 | +6.949 | +2.594 | 1 |
| Citeseer | order19612 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Cora | order19611 | 1 | -0.548 | -1.734 | -0.380 | -1.170 | -0.775 | 1 |
| Cora | order19611 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Cora | order19612 | 1 | -3.107 | -5.988 | -1.200 | -3.493 | -2.346 | 1 |
| Cora | order19612 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Citeseer | order19613 | 1 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Citeseer | order19613 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |
| Cora | order19613 | 1 | +1.489 | +0.327 | +0.378 | +4.498 | +2.438 | 1 |
| Cora | order19613 | 2 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0 |

### 4.1 汇总

| 范围 | Final Macro-F1 | Accuracy | New F1 | Old F1 | Stage-average F1 | 正收益 cell |
|---|---:|---:|---:|---:|---:|---:|
| 开发顺序 8 cells | -0.142 | -0.793 | -0.418 | +0.286 | -0.066 | 1/8 |
| 独立顺序 4 cells | +0.372 | +0.082 | +0.094 | +1.125 | +0.609 | 1/4 |

按 paired Student seed 的最终 F1 差值，开发顺序为 `-0.142 +/- 2.503` pp（48 个配对），
独立顺序为 `+0.372 +/- 1.184` pp（24 个配对）。这里的标准差是配对差的离散程度，
不是置信区间，也不支持显著性结论。

## 5. 构造审计

| 审计项 | 结果 |
|---|---:|
| 构造/评估 cell | 12/12 |
| accepted swap 的 cell | 4/12 |
| 门控失败并回退 SR | 8/12 |
| old-old 边保持 | 12/12 |
| new-new 边保持 | 12/12 |
| 新行集合保持 | 12/12 |
| 每个新行跨边度保持 | 12/12 |
| 旧端点容量保持 | 12/12 |
| `X`、文本、预算、初始 `P` 更新 | 0 |
| 历史 raw feature/text 读取 | 0 |
| report/test 标签选择 | 否 |

4 个接受 cell 的 selector 局部信号均为正：一步 `new_gain` 为 `+0.00065` 到 `+0.00399`，
一步 `old_gain` 为 `+0.00074` 到 `+0.00248`，且一步后的总 loss 下降。但最终下游结果中，
4 个接受 cell 只有 2 个提升、2 个下降。这是本轮最重要的诊断结果：局部一步响应与最终
stage-level Macro-F1 尚未可靠对齐。

下游 JSON 中的 `student_seconds` 只计最终评估 Student，不包含 selector 对每个候选重复训练
所花的构造成本。因此本轮不能宣称该组件降低了总计算成本；实际总成本应额外计入 selector。

## 6. 结果说明了什么

### 6.1 实现层面是正确的

跨边交换的守恒约束、回退逻辑和访问约束均通过。结果差异不能归因于 X、文本、预算、旧旧边、
新新边或边数意外改变。

### 6.2 Student response 有局部信号，但当前 selector 不够可靠

它在局部的一步 InfoNCE loss 上能够找到“旧块和新块都不变差”的候选；但该目标有三个限制：

1. selector 只使用一个 seed 和一步更新，噪声较大；
2. selector 只看块级平均 loss，没有直接约束每个新类的 plasticity；
3. 下游评估是更长训练轨迹和累计阶段 F1，一步 loss 改善不保证长期分类边界改善。

因此，Cora `order19612/s1` 的 `-3.107 pp` 说明当前门控仍可能接受对长期 Student 不利的交换；
Citeseer `order19612/s1` 的 `+2.522 pp` 则不能单独视为可复现增益。

### 6.3 当前组件没有解决新类塑性问题

开发顺序 New F1 平均为 `-0.418 pp`，独立顺序仅为 `+0.094 pp`。Old F1 略有提升，说明
当前短程门控更容易保护旧块，而不是稳定改善新块。它没有证明“Student-consistent cross-edge
swap”能够提升新类塑性。

## 7. 当前决策

本轮不晋升为 SR 主方法组件，也不与旧记忆 `X/P` revision、文本 attention、动态预算或其它
拓扑模块联合。应保留它作为一个有审计的诊断性负结果：

> 受限跨边交换可以在不改内容和预算的条件下引入 Student 反馈，但单步、单 seed 的块级
> InfoNCE gate 尚不能预测长期新类塑性；技术守恒成立，性能晋升不成立。

这比直接放弃梯度或旧记忆更新更具体：问题定位在 **selector horizon/variance 与最终 Student
目标之间的错配**，不是“梯度方法原则上无效”。

## 8. 后续最小推进

下一轮只改 selector 可靠性，继续冻结 `X`、文本、预算和拓扑候选池：

1. 用固定的 3 个 selector seeds 和 3--5 个 Student steps，报告候选的均值、标准差和最差 seed；
2. 接受条件改为多 seed 的稳健门控：新块不能在多数 seed 上退化，旧块和总 loss 也不能恶化；
3. 增加按新类而不是仅按 new-block 平均的约束，避免“大多数新类改善、少数新类崩溃”；
4. 记录 selector 总成本，并与“同成本多训练 Student steps”的对照比较；
5. 仍只允许最多一个跨边交换，失败时回退 SR；只有在独立顺序上通过后，才考虑和 `X` 的
   trust-region 更新联合。

不要在下一轮同时修改 `P`、`X`、文本、预算或 Student 架构。否则无法判断收益来自 selector
稳定化还是来自新的合成内容。

## 9. 复现入口

- 实现：[student_consistent_edge_swap.py](../../code/v2_llm_fusion/student_consistent_edge_swap.py)
- 构造入口：[run_tgr_cross_edge_transport.py](../../code/v2_llm_fusion/run_tgr_cross_edge_transport.py)
- 开发构造：[launch_tgr_student_consistent_20260928.sh](../../code/v2_llm_fusion/launch_tgr_student_consistent_20260928.sh)
- 开发评估：[launch_tgr_student_consistent_eval_20260928.sh](../../code/v2_llm_fusion/launch_tgr_student_consistent_eval_20260928.sh)
- 独立构造：[launch_tgr_student_consistent_independent_20260928.sh](../../code/v2_llm_fusion/launch_tgr_student_consistent_independent_20260928.sh)
- 独立评估：[launch_tgr_student_consistent_independent_eval_20260928.sh](../../code/v2_llm_fusion/launch_tgr_student_consistent_independent_eval_20260928.sh)
- 远端开发评估目录：`/home/pc/research/TAG_condensation/results/tgr_student_consistent_eval_20260928/`
- 远端独立评估目录：`/home/pc/research/TAG_condensation/results/tgr_student_consistent_independent_eval_20260928/`

