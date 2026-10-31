# P1-X 合成节点特征算子实验报告

日期：2026-09-18  
状态：Cora / Citeseer 小图、三种预注册顺序、三种 X 算子全部完成；实验完整性通过，候选方法均未通过主线晋升门禁。

## 1. 这轮要回答什么

此前当前方法与强 coreset 的差距，可能来自合成节点属性 `X` 的构造过于简单：每个 joint graph-text cluster 只取成员属性的均值。P1-X 在严格控制变量下只改变合成节点 `X`，检查：

> Mean-X 是否造成多模态信息和方差收缩，从而限制持续凝聚效果？

三种方法为：

| 算子 | 做法 |
|---|---|
| `Mean-X` | cluster 成员属性的算术均值，当前默认基线 |
| `Weighted-X` | 按 joint graph-text 中心距离给成员分配确定性权重，再与均匀权重混合，形成受限凸组合 |
| `Moment-X` | 以均值为中心，沿 cluster 内主方差方向加入半个主标准差的确定性偏移，并做平均范数校正 |

两种新算子都生成合成向量，不把真实节点直接放入凝聚图。`Moment-X` 不是从均值恢复完整分布，而是一个低成本的第一版方差保持诊断。

## 2. 严格控制变量

本轮固定：

- Cora / Citeseer；
- `canonical`、`order1701`、`order1702` 三种预注册类别顺序；
- P0 `p0-canonical-class-then-source-id-v1` 枚举协议；
- 固定 5% 累计预算、相同 class-level `dispersion` allocation；
- 相同 cluster assignment、rows、文本构造 `joint_g5_c320 + concat`；
- 相同 frozen-old topology、`cross-degree=1`、旧 memory 保持规则；
- 相同两层 GCN384 / MiniLM384 / 双向 InfoNCE student；
- batch 16、student seeds 100--105、协议步数；
- 不启用 response selector、response-gain、extra budget 或 refresh。

唯一变化是 cluster 内合成 `x` 的算子。每个数据集完成 3 个顺序 x 3 个算子 x 3 个阶段 x 6 个 student seeds，即每个数据集 162 次 student fit，两个数据集合计 324 次；所有 checkpoint 均保存并复载核对。构造没有使用 test labels。

## 3. 主结果

最终 Macro-F1 和阶段平均 Macro-F1 均为六个 student seeds 的均值。`delta` 相对于同一数据集和顺序的 `Mean-X`。

### 3.1 三顺序平均

| 数据集 | X 算子 | 最终 F1 | 阶段平均 F1 | 旧类 F1 | 新类 F1 | 最终相对 Mean-X | 阶段平均相对 Mean-X | 最终胜出顺序数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | Mean-X | 48.40 | 49.04 | 56.99 | 36.95 | 0.00 | 0.00 | 0/3 |
| Cora | Weighted-X | 48.42 | 49.42 | 57.46 | 36.37 | **+0.02** | **+0.38** | 1/3 |
| Cora | Moment-X | 47.13 | 49.08 | 56.34 | 34.84 | **-1.27** | +0.04 | 1/3 |
| Citeseer | Mean-X | 54.63 | 60.97 | 58.67 | 46.53 | 0.00 | 0.00 | 0/3 |
| Citeseer | Weighted-X | 54.33 | 60.13 | 58.20 | 46.60 | **-0.29** | **-0.83** | 1/3 |
| Citeseer | Moment-X | 53.76 | 61.41 | 57.55 | 46.19 | **-0.86** | +0.44 | 1/3 |

三顺序平均的结论很明确：

- `Weighted-X` 在 Cora 只有 `+0.02` 个百分点的最终平均差异，不能视为性能提升；在 Citeseer 反而下降 `0.29` 个百分点。
- `Moment-X` 在 Cora 最终下降 `1.27` 个百分点，在 Citeseer 下降 `0.86` 个百分点。
- 两种方法都没有在任一数据集的三个顺序中稳定胜过 Mean-X。
- 因此当前默认仍为 Mean-X；Weighted-X 保留为低成本负结果参照，Moment-X 第一版否决。

### 3.2 每个顺序的详细结果

#### Cora

| 顺序 | Mean-X 最终 | Weighted-X 最终 | delta | Moment-X 最终 | delta | Mean-X 阶段平均 | Weighted-X 阶段平均 | Moment-X 阶段平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 49.85 | 49.77 | -0.08 | 46.25 | -3.61 | 53.08 | 54.16 | 52.02 |
| order1701 | 52.59 | 53.09 | +0.50 | 50.43 | -2.16 | 62.10 | 62.31 | 60.92 |
| order1702 | 42.77 | 42.41 | -0.36 | 44.72 | +1.95 | 31.93 | 31.78 | 34.31 |
| 三顺序平均 | 48.40 | 48.42 | +0.02 | 47.13 | -1.27 | 49.04 | 49.42 | 49.08 |

#### Citeseer

| 顺序 | Mean-X 最终 | Weighted-X 最终 | delta | Moment-X 最终 | delta | Mean-X 阶段平均 | Weighted-X 阶段平均 | Moment-X 阶段平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 59.17 | 58.78 | -0.40 | 57.95 | -1.22 | 64.29 | 63.33 | 64.33 |
| order1701 | 53.76 | 51.78 | -1.98 | 52.38 | -1.38 | 66.82 | 64.89 | 66.69 |
| order1702 | 50.94 | 52.44 | +1.50 | 50.95 | +0.02 | 51.79 | 52.19 | 53.20 |
| 三顺序平均 | 54.63 | 54.33 | -0.29 | 53.76 | -0.86 | 60.97 | 60.13 | 61.41 |

### 3.3 旧类保持与新类塑性

相对 Mean-X 的最终变化：

| 数据集 | 算子 | 旧类变化 | 新类变化 | 解读 |
|---|---|---:|---:|---|
| Cora | Weighted-X | +0.47 | -0.58 | 旧类略稳，但新类略差，整体抵消 |
| Cora | Moment-X | -0.65 | -2.11 | 方差偏移没有改善塑性，反而损伤新类 |
| Citeseer | Weighted-X | -0.48 | +0.07 | 新类几乎不变，旧类保持下降 |
| Citeseer | Moment-X | -1.12 | -0.34 | 旧类和新类都没有稳定收益 |

单个顺序的正结果不能改变结论。例如 Cora order1702 的 Moment-X 最终高 `1.95`，但它在 canonical 低 `3.61`、order1701 低 `2.16`；Citeseer order1702 的 Weighted-X 高 `1.50`，但 order1701 低 `1.98`。这更像顺序与数据几何的交互，而不是普适的 X 构造改进。

## 4. 与强 coreset 的位置

P1-X 没有重新运行 k-center，而是与已经完成的同协议小图比较对照。三顺序平均最终 F1 为：

| 数据集 | Mean-X | Weighted-X | Moment-X | k-center | kmeans-medoid |
|---|---:|---:|---:|---:|---:|
| Cora | 48.40 | 48.42 | 47.13 | 49.40 | 54.24 |
| Citeseer | 54.63 | 54.33 | 53.76 | 57.73 | 51.08 |

这轮没有缩小当前方法与 k-center 的主要差距。因此，问题不能简单归结为“均值造成方差收缩”。更可能还包含：

1. 合成节点属性与冻结文本原型之间的图文对齐不匹配；
2. 合成节点不是实际图中的真实局部结构点，而 topology 仍由文本相似性和受控跨边决定；
3. k-center 直接保留真实属性，天然保留了非线性边界和局部异常模式；
4. 当前 X 算子只在 cluster 内改一个向量，不能恢复 cluster 内的多峰结构或节点级拓扑信息。

因此这轮结果不能支持“只要换一种均值聚合就能超过 coreset”。

## 5. 成本与 memory 控制

| 数据集 | 算子 | 平均构造时间 | 六 seed student 时间 | 最终 logical payload |
|---|---|---:|---:|---:|
| Cora | Mean-X | 0.447 s | 37.07 s | 326.3 KB |
| Cora | Weighted-X | 0.486 s | 36.49 s | 326.3 KB |
| Cora | Moment-X | 0.464 s | 36.62 s | 326.3 KB |
| Citeseer | Mean-X | 0.550 s | 47.82 s | 384.9 KB |
| Citeseer | Weighted-X | 0.585 s | 47.01 s | 384.9 KB |
| Citeseer | Moment-X | 0.619 s | 47.16 s | 384.9 KB |

三种算子的最终 rows、边数、文本字节和 tensor payload 相同。Weighted/Moment 的额外计算很小，不是本轮失败原因；失败原因是性能没有稳定收益。

## 6. 实验完整性验收

| 检查项 | 结果 |
|---|---|
| Cora / Citeseer | 完成 |
| canonical / order1701 / order1702 | 完成 |
| Mean / Weighted / Moment | 完成 |
| 每个数据集 student fits | 162 |
| 两个数据集合计 fits | 324 |
| checkpoint 复载 | 通过，acceptance 完整 |
| P0 enumeration manifest | 已记录 |
| test labels | 未用于本轮构造或选择 |
| rows / text / topology / allocation 控制 | 通过 |
| response selector | 关闭 |

结果入口：

- 汇总表：[summary.csv](../../results/p1x_x_operator_20260918/summary/summary.csv)
- 三顺序聚合：[aggregate.csv](../../results/p1x_x_operator_20260918/summary/aggregate.csv)
- 机器可读结果：[summary.json](../../results/p1x_x_operator_20260918/summary/summary.json)
- 汇总脚本：[summarize_p1x_x_operator.py](../../code/v2_llm_fusion/summarize_p1x_x_operator.py)
- 远程验收：`results/p1x_x_operator_20260918/{cora,citeseer}/acceptance.json`

## 7. P1-X 结论与后续

### 通过的部分

- 已把合成 `X` 从构造函数中独立成可审计的 operator；
- Mean-X 与历史均值定义保持一致；
- Weighted-X 权重非负、和为 1，并且不退化为真实节点复制；
- Moment-X 确定性、尺度受控，不复制真实节点；
- 三种顺序、两数据集的严格配对实验完成。

### 未通过的部分

- Weighted-X 没有跨顺序稳定的最终 F1 或阶段平均增益；
- Moment-X 在两个数据集的最终平均均下降；
- 与 k-center 的差距没有被这两个低成本 X 算子解决；
- 不足以把 P1-X 算子升级为新的默认方法或 v5。

### 后续建议

1. 冻结 `Mean-X + dispersion` 作为当前默认控制，不继续扫描 Weighted 权重温度或 Moment 偏移系数；否则容易变成在小图顺序上过拟合。
2. 将“均值的信息损失”改写为更具体的问题：需要联合建模属性、文本和局部拓扑，而不是只改变 cluster 的一个属性向量。
3. 下一优先级是 P3-B 的拓扑诊断，区分 k-center 的优势来自 X、真实节点局部结构，还是 synthetic topology。
4. 若继续做 P1-X，下一版应直接设计多原型/多节点簇表示，或使用 graph-conditioned X-text consistency；不要只在均值附近做更多手工扰动。
5. 暂不上大图。当前小图已足以否决这两个第一版 X 算子的晋升。

