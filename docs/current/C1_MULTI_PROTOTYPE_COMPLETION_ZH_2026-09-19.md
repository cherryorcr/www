# C1 类内多原型与复杂度分配小图完成报告

日期：2026-09-19  
协议：`c1-multi-prototype-small-v1`  
状态：**实验完成；候选未晋升为默认主方法**

## 1. 实验目的

C0 使用 `dispersion allocation + Mean-X`。它把每个类内聚类块压缩为一个均值属性原型，容易丢失类内多模态、尾部方向和局部结构角色。C1 的目标是验证：

1. 是否应在类别内部继续按复杂度划分子簇；
2. 是否应为子簇生成多个合成原型，而不是只保留一个均值；
3. Weighted-X 与 Moment-X 是否能在固定 memory rows 下缓解均值的方差收缩；
4. 这些变化能否改善从头训练 student 后的持续任务效用。

## 2. 严格协议

实验固定：

- 数据集：Cora、Citeseer；
- 类别顺序：`canonical`、`order1701`、`order1702`；
- memory：5% fixed-rate growing budget；
- student：两层 GCN，hidden/output 384，双向图文 InfoNCE，batch=16；
- student seeds：100--105，共 6 个；
- 文本：`joint_g5_c320 + concat`，冻结 MiniLM，融合后重新编码；
- 拓扑：`frozen_old`，cross-degree=1；
- 评估：cumulative/stage Macro-F1、old/new F1；未使用 test labels；
- 每个数据集共 3 个顺序、4 个 arm、3 个 stage、6 个 student fits，完成并重载审计。

候选如下：

| Arm | 类内预算/聚类 | X 构造 | 作用 |
|---|---|---|---|
| `dispersion` | 类级 dispersion | Mean-X | C0 控制 |
| `subcluster_mean` | 类内子簇复杂度 | Mean-X | 只改变分配层级与子簇划分，但仍使用均值 |
| `subcluster_weighted` | 类内子簇复杂度 | Weighted-X | 子簇内确定性加权凸组合 |
| `subcluster_moment` | 类内子簇复杂度 | Moment-X | 子簇内均值/方差/排斥约束构造 |

注意：三个候选都同时改变了类内子簇预算分配和 X 构造路径。因此本实验能验证“C1 联合方向是否值得继续”，不能把差异严格归因给某一个 X 算子。

## 3. 结果总表

数值为 3 个类别顺序的均值；`delta` 相对同数据集的 C0 `dispersion`。

| 数据集 | Arm | Final F1 | 阶段平均 F1 | old F1 | new F1 | Final delta | 阶段平均 delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | C0 dispersion + Mean-X | 48.40 | 49.04 | 56.99 | 36.95 | 0.00 | 0.00 |
| Cora | 子簇 Mean-X | 49.55 | 52.93 | 54.64 | 42.75 | +1.14 | +3.89 |
| Cora | 子簇 Weighted-X | 49.21 | 52.26 | 54.69 | 41.91 | +0.81 | +3.22 |
| Cora | 子簇 Moment-X | **50.36** | **55.30** | 55.71 | 43.23 | **+1.96** | **+6.26** |
| Citeseer | C0 dispersion + Mean-X | **54.63** | **60.97** | 58.67 | 46.53 | 0.00 | 0.00 |
| Citeseer | 子簇 Mean-X | 53.55 | 54.31 | 56.18 | 48.29 | -1.07 | -6.66 |
| Citeseer | 子簇 Weighted-X | 53.98 | 53.92 | 57.03 | 47.88 | -0.65 | -7.04 |
| Citeseer | 子簇 Moment-X | 53.79 | 55.77 | 57.03 | 47.32 | -0.83 | -5.19 |

## 4. 顺序稳定性

### Cora

- `canonical`：Moment-X final `50.52`，比 C0 高 `+0.66`；阶段平均高 `+5.27`。
- `order1701`：三个子簇候选 final 均低于 C0，Moment-X 为 `49.34`，低 `-3.25`。
- `order1702`：三个候选均明显高于 C0；Moment-X final `51.23`，高 `+8.46`；阶段平均高 `+14.24`。

### Citeseer

- `canonical`：三个候选 final 均低于 C0，Moment-X 低 `-6.67`。
- `order1701`：final 差异接近 0，但阶段平均下降约 `13--16` 个百分点。
- `order1702`：final 有小幅正差异，但阶段平均仍低于 C0；Moment-X final 高 `+4.01`，阶段平均低 `-0.40`。

因此 C1 的正向结果主要来自 Cora 的部分顺序，无法视为跨数据集、跨顺序稳定增益。

## 5. 成本与正确性

- 每个数据集完成 `216` 次重载 fit 审计，两个数据集合计 `432` 次；
- 每个 arm 每个顺序包含 18 个 student fits；
- Cora 每个顺序总 student steps 为 17,856，Citeseer 为 22,560；
- 平均构造链耗时约为：Cora C0 `0.45s`、子簇候选 `0.50--0.52s`；Citeseer C0 `0.58s`、子簇候选 `0.57--0.61s`；
- 所有 arm 保持目标 rows：Cora 最终 94，Citeseer 最终 111；
- 合成节点仅由源簇统计构造，没有直接复制真实节点；
- canonical C0 的历史回归和 synthetic student 状态审计通过；test labels 未访问。

## 6. 结论

### 6.1 可以确定的内容

1. 类内多子簇方向在 Cora 上有机制信号，尤其 Moment-X 能提高阶段平均表现和部分新类塑性。
2. Citeseer 上三个候选都不如 C0，说明该方向不是数据集无关的默认改进。
3. Weighted-X 没有表现出相对 Mean-X 的稳定额外收益；Moment-X 的优势也不稳定。
4. 不能因为 Cora `order1702` 的高分就把 C1 命名为 v5 或主方法。
5. 与 k-center/medoid 的比较仍然说明“合成 X + 文本 + 拓扑的联合构造”尚未稳定超过保留真实节点的 coreset 参照。

### 6.2 不能由本实验声称的内容

- 不能声称 Moment-X 单独有效，因为分配和 X 同时变化；
- 不能声称多原型普遍优于单均值；
- 不能声称 C1 解决了大图长期记忆问题；
- 不能声称 C1 已经支持 SOTA 或 v5 主线。

## 7. 后续安排

C1 第一轮到此收尾，暂不继续扫描更多手工 X 权重。若后续仍要拆解，应只做一轮小规模因果分解：

1. 固定 `dispersion`，只替换 Mean-X 为子簇 Mean-X；
2. 固定类内子簇分配，只替换 Mean/Weighted/Moment-X；
3. 使用新的独立顺序和 seed group，不复用当前三个开发顺序作为独立确认；
4. 只有在两个数据集均不退化且成本可接受时，才建立新的版本号。

在当前证据下，P5 优先级高于继续扩张 C1：先做 `1%/5%/10%` matched-rate 曲线、可行的更长阶段链、独立顺序，以及小图通过后的大图 base+4。C0 仍是后续共同控制：

```text
5% fixed-rate + dispersion + Mean-X + joint_g5_c320 concat
+ frozen-old topology + current cross-edge + response off
```

## 8. 产物

- 汇总：`results/c1_multi_prototype_20260919/summary/summary.json`
- 聚合表：`results/c1_multi_prototype_20260919/summary/aggregate.csv`
- 顺序表：`results/c1_multi_prototype_20260919/summary/per_order.csv`
- 实现：[run_c1_multi_prototype_small.py](../../code/v2_llm_fusion/run_c1_multi_prototype_small.py)
- 汇总脚本：[summarize_c1_multi_prototype.py](../../code/v2_llm_fusion/summarize_c1_multi_prototype.py)
