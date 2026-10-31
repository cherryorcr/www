# 链路预测核机制对照结果

日期：2026-09-20  
状态：已完成远程运行、独立结构审计、汇总和结果拉取

## 1. 结论摘要

这轮实验回答的问题是：链路预测中的提升究竟来自原始的 graph-text-signature 联合核，还是主要来自更基本的拓扑约束，例如度均衡和覆盖约束。

结论如下：

1. **graph signature 不是当前链路预测提升的必要条件。** 只使用文本核的 `T + degree` 已经达到原始 `XTG + degree` 的相近水平；Citeseer 上还明显更高。
2. **X 的加入没有稳定增益。** `XT + degree` 相比 `T + degree` 在 Cora 的 AP 略高，但 AUC 略低；Citeseer 两项略高，增益幅度较小，尚不能证明 X 核是必要组件。
3. **度均衡是最稳定的有效组件。** 它把 C0 中的孤立合成节点消除，并在两个数据集上带来一致的链路预测提升。
4. **coverage 不是无条件增益。** Citeseer 上它有效，Cora 上反而低于 degree-only 版本；它应作为数据集敏感的候选控制，而不是默认主方法。
5. **这不能反向晋升分类主线。** 本报告证明的是合成拓扑对链路预测的影响，C0 仍是跨任务的冻结比较版本。

因此，若只针对链路预测选择一个简单候选，当前优先级是：

> `T + degree-balanced`，即文本相似度核负责选择语义相近的连边，度均衡负责修复合成图的孤立节点和极端度分布。

## 2. 对照方法

所有机制臂使用相同的合成 `X`、文本、mass、row order、edge split、GAE、训练步数和 student seeds，只改变新旧块之间的连边核和拓扑约束。

| 方法 | 核输入 | 拓扑约束 | 用途 |
|---|---|---|---|
| `C0 replay` | C0 原始连边 | C0 原始规则 | 独立复跑控制 |
| `T + degree` | text only | degree-balanced | 判断文本核和度均衡的基本效果 |
| `XT + degree` | X/text，权重 `(4/7, 3/7, 0)` | degree-balanced | 判断 X 是否提供独立贡献 |
| `T + coverage` | text only | coverage | 判断 coverage 是否能替代 degree-only |
| `XT + coverage` | X/text，权重 `(4/7, 3/7, 0)` | coverage | 判断 X 与 coverage 的组合 |
| `XTG + degree` | X/text/graph signature，权重 `(0.4, 0.3, 0.3)` | degree-balanced | 上一轮原始联合核参照 |
| `XTG + coverage` | X/text/graph signature，权重 `(0.4, 0.3, 0.3)` | coverage | 上一轮原始联合核参照 |

其中：

- `T` 只比较文本相似度；
- `XT` 比较合成节点属性与文本的联合相似度；
- `XTG` 是此前的 X、text、graph signature 联合核；
- 本轮新增控制不携带 `graph_signature`，避免控制臂隐式获得被检验的组件。

## 3. 冻结协议和审计

- 数据集：Cora、Citeseer；
- 预算：5% 合成节点；
- 持续协议：canonical 三阶段；
- 边划分：沿用上一轮的 3 个 edge split；
- student seeds：6 个，100--105；
- 每个数据集：5 个机制臂 × 3 split × 6 seeds × 3 stages = 270 fits；
- 总计：540 fits；
- 下游模型：独立 GAE 链路预测模型；
- 指标：最终阶段全体合法边的 ROC-AUC 和 Average Precision；
- 防泄漏：先隐藏真实评估边，再重建合成拓扑；评估边不参与 signature 或新边构造；
- 历史约束：旧 rows 和旧-旧 edges 保持冻结，新控制只改变允许构造的新边。

独立审计结果：

| 数据集 | fits | 阶段 memory | 父实验审计 | 当前审计 |
|---|---:|---:|---|---|
| Cora | 270 | 45 | pass | pass |
| Citeseer | 270 | 45 | pass | pass |

`C0 replay` 与父实验的最终差异小于 0.002 pp，说明本轮结果没有由重跑随机性或汇总口径造成的明显偏移。

这轮沿用旧 edge split，属于**机制开发诊断**，不是独立泛化验证。后续若要晋升候选，必须在新的 edge split 和新的顺序/seed group 上再测。

## 4. 最终阶段结果

### 4.1 新增机制臂与 C0

数值单位为百分比；括号中是 3 个 split 间的标准差。

| 数据集 | 方法 | AUC | AP | 相对父 C0 AUC | 相对父 C0 AP |
|---|---|---:|---:|---:|---:|
| Cora | C0 replay | 85.90 | 84.04 | -0.00 | +0.00 |
| Cora | T + degree | **87.17** | 85.12 | **+1.28** | +1.09 |
| Cora | XT + degree | 87.04 | **85.94** | +1.15 | **+1.90** |
| Cora | T + coverage | 86.51 | 84.44 | +0.61 | +0.40 |
| Cora | XT + coverage | 86.53 | 84.93 | +0.63 | +0.89 |
| Citeseer | C0 replay | 87.01 | 85.15 | +0.00 | -0.00 |
| Citeseer | T + degree | 90.66 | 89.04 | +3.66 | +3.89 |
| Citeseer | XT + degree | 90.76 | 89.20 | +3.76 | +4.05 |
| Citeseer | T + coverage | **91.29** | **89.60** | **+4.28** | **+4.45** |
| Citeseer | XT + coverage | 90.95 | 89.24 | +3.95 | +4.09 |

### 4.2 与上一轮 XTG 参照比较

上一轮的原始联合核结果为：Cora `XTG + degree = 87.11/85.68`、`XTG + coverage = 87.57/86.51`；Citeseer `XTG + degree = 90.31/88.04`、`XTG + coverage = 90.33/88.41`。

| 数据集 | 新方法 | XTG 参照 | AUC 差值 | AP 差值 | 解释 |
|---|---|---|---:|---:|---|
| Cora | T + degree vs XTG + degree | 87.17/85.12 vs 87.11/85.68 | +0.06 | -0.55 | AUC 持平，AP 略低 |
| Cora | XT + degree vs XTG + degree | 87.04/85.94 vs 87.11/85.68 | -0.07 | +0.26 | 没有稳定优势 |
| Cora | T + coverage vs XTG + coverage | 86.51/84.44 vs 87.57/86.51 | -1.07 | -2.06 | coverage 与数据集结构交互明显 |
| Cora | XT + coverage vs XTG + coverage | 86.53/84.93 vs 87.57/86.51 | -1.05 | -1.58 | 同上 |
| Citeseer | T + degree vs XTG + degree | 90.66/89.04 vs 90.31/88.04 | +0.35 | +1.00 | T-only 已超过联合核 |
| Citeseer | XT + degree vs XTG + degree | 90.76/89.20 vs 90.31/88.04 | +0.45 | +1.16 | X 有小幅附加收益，但非决定性 |
| Citeseer | T + coverage vs XTG + coverage | 91.29/89.60 vs 90.33/88.41 | +0.95 | +1.19 | coverage 在 Citeseer 上有效 |
| Citeseer | XT + coverage vs XTG + coverage | 90.95/89.24 vs 90.33/88.41 | +0.62 | +0.84 | X 没有超过 T-only |

## 5. 机制解释

### 5.1 度均衡是主要收益来源

C0 的合成图中存在明显孤立节点：最终阶段 Cora 平均约 3 个、Citeseer 平均约 5 个；在 stage 0，孤立节点更多。`degree-balanced` 版本将这些孤立节点消除，同时不增加 rows 和总边预算。

这解释了为什么 `T + degree` 和 `XT + degree` 在两个数据集上都优于 C0：下游 GAE 至少能对每个合成节点获得一阶结构信号。对于链路预测，拓扑可用性比更复杂的相似度核更直接。

### 5.2 graph signature 当前不是必要组件

如果 graph signature 是提升的主要来源，那么移除它后，`T/XT + degree` 应明显低于 `XTG + degree`。实际结果是：

- Cora：T-only 与 XTG 的 AUC 基本相同，XT-only 的 AP 略高；
- Citeseer：T-only 和 XT-only 都超过 XTG；
- 因此没有证据支持 graph signature 在当前小图链路预测协议下是必要的。

更稳妥的表述是：graph signature 可能包含有用的结构信息，但在当前合成预算、边预算和 GAE 下游中，其信息被文本相似度和度均衡拓扑大部分替代；继续保留它会增加方法复杂度，却没有稳定的独立收益。

### 5.3 X 不是稳定的独立贡献

`XT + degree` 相比 `T + degree`：

- Cora：AUC 下降约 0.13 pp，AP 上升约 0.81 pp；
- Citeseer：AUC 上升约 0.10 pp，AP 上升约 0.16 pp。

方向并不完全一致，提升幅度也小于度均衡带来的提升。因此 X 更适合保留为分析变量，而不是在当前版本中作为不可删的核心模块。

### 5.4 coverage 具有数据集依赖性

在 Citeseer，coverage 进一步提高了 T-only 的 AUC/AP；但在 Cora，coverage 低于 degree-only。可能原因包括：

- Cora 的文本相似性已经足以提供较好的局部连接，coverage 的额外约束会打散有益的高相似边；
- Citeseer 的合成块更需要跨簇覆盖来避免结构塌缩；
- 当前 coverage 规则是确定性启发式，不是针对下游指标学习得到的最优拓扑。

所以 coverage 应作为结构敏感的候选，而不能直接写成普适改进。

## 6. 与完整原始图参考的关系

上一轮完整原始图直接训练的参考为：

| 数据集 | 当前机制中最好候选 | AUC/AP | 完整原始图 | AUC/AP 差距 |
|---|---|---:|---:|---:|
| Cora | T + degree（AUC）/ XT + degree（AP） | 87.17/85.94 | 90.71/90.81 | -3.53/-4.87 |
| Citeseer | T + coverage | 91.29/89.60 | 92.64/91.99 | -1.35/-2.39 |

这不是“当前方法失败”的证据，而是压缩凝聚的正常代价：有限合成 rows 和有限边预算不可能完整保留原图的全部结构。该表真正说明的是：

- 链路预测上，简单拓扑修复可以显著缩小 C0 与完整图之间的差距；
- Citeseer 的差距已经小于 Cora，说明该机制对不同图结构的可压缩性不同；
- 仍不能宣称合成图已经达到完整图上限。

## 7. 当前判断和论文边界

### 可以支持的判断

1. 合成图构造的拓扑质量会显著影响链路预测，而不仅是节点分类；
2. C0 的孤立节点和不均衡度分布是一个可观测的跨任务缺陷；
3. 度均衡是低成本、可解释、跨两个小图稳定有效的修复；
4. 复杂的 graph signature 联合核没有在当前协议中显示稳定独立收益；
5. coverage 的效果依赖数据集，不能替代度均衡成为默认规则。

### 不能支持的判断

- 不能据此宣称新的联合核是分类主线的晋升版本；
- 不能据此宣称超过完整原始图；
- 不能据此宣称超过所有链路预测 baseline，尤其是文本 cosine 无训练基线；
- 不能把沿用旧 edge split 的机制诊断当成独立泛化结果；
- 不能把 Cora/Citeseer 的结果外推到大图。

## 8. 当前最合理的版本选择

如果目标是保留一个简单、可解释、便于扩展的链路预测候选，建议使用：

```text
semantic topology construction = text kernel T
topology repair = degree-balanced edge allocation
memory rule = C0 append-only rows and frozen old-old edges
```

也就是 `T + degree-balanced`。`XT + degree` 可作为低成本敏感性对照；`XTG + coverage` 保留为上一轮复杂联合核参照；coverage 只在新 edge split 上确认有效后才考虑启用。

但对于当前论文的跨任务主线，仍建议保持 `C0` 为冻结共同基线，把 `T + degree-balanced` 表述为：

> 面向链路预测的拓扑诊断候选，而不是已经取代 C0 的统一凝聚方法。

## 9. 后续建议

只有在准备把链路预测作为论文第二任务时，才值得继续做：

1. 新 edge split、新类别顺序和独立 seed group，验证 `T + degree` 是否仍稳定；
2. 1%、5%、10% budget curve，判断提升是否只在 5% 出现；
3. 分关系类型和 stage 的 AUC/AP，检查收益来自孤立节点修复还是边排序；
4. 将 T-only、XT-only、XTG-only 的构造成本与内存成本一并报告；
5. 只在上述通过后再考虑大图。

详细机器证据位于：

- `results/lp_kernel_mechanism_20260920/methods.csv`
- `results/lp_kernel_mechanism_20260920/paired_differences.csv`
- `results/lp_kernel_mechanism_20260920/summary.json`
- `results/lp_kernel_mechanism_20260920/cora/audit.json`
- `results/lp_kernel_mechanism_20260920/citeseer/audit.json`

