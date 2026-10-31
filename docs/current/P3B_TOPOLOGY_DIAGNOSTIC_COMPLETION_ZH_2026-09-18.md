# P3-B 拓扑诊断完成报告

日期：2026-09-18  
状态：Cora / Citeseer 小图、三种预注册类别顺序、四种拓扑规则完成；实验通过正确性验收，但没有产生可直接晋升的统一新拓扑。

## 1. 实验目的

本轮回答两个问题：

1. 当前 synthetic memory 的新旧跨边是否真的有助于持续学习；
2. 如果有帮助，收益来自跨边数量，还是来自当前“按文本相似度选择旧端点”的规则。

P1-X 的 `Mean-X + dispersion + joint_g5_c320` 固定为输入。只改变新旧跨边，避免把 X、文本、预算和拓扑变化混在一起。

## 2. 四个拓扑臂

| 臂 | 跨边规则 |
|---|---|
| `current` | 当前实现：每个新合成节点连接文本 embedding 最相似的 1 个旧节点 |
| `none` | 不添加新旧跨边 |
| `random` | 与 `current` 相同数量，使用固定 seed 选择随机旧端点 |
| `quality` | 与 `current` 相同数量，按 `0.5*cos(text)+0.5*cos(X)` 选择旧端点 |

共同约束：旧行、旧旧边、新节点 X、新节点文本、新新边、rows、class allocation、类别顺序、student batch、学习率、步数和 six student seeds 全部相同。P3-B 不做 response 选择、不使用 test labels、不刷新旧节点。

规模为：2 个数据集 × 3 个顺序 × 4 个拓扑臂 × 3 个阶段 × 6 个 student seeds，共 432 次 student fit。远程环境为 `tagsam`，Cora / Citeseer 分别使用两张 RTX 4090。

## 3. 最终结果

最终 Macro-F1 是第三阶段六个 student seed 的平均；`stage-average` 是三个阶段最终 Macro-F1 的平均。

### 3.1 三顺序平均

| 数据集 | 拓扑 | 最终 F1 | 顺序标准差 | 阶段平均 | 旧类 F1 | 新类 F1 | 旧表征漂移 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | current | 48.40 | 4.14 | 49.04 | 56.99 | 36.95 | 0.43 |
| Cora | none | 45.43 | 5.49 | 46.78 | 52.04 | 36.61 | 0.00 |
| Cora | random | 50.16 | 9.02 | 53.03 | 59.20 | 38.09 | 0.51 |
| Cora | quality | 48.05 | 3.53 | 48.96 | 54.39 | 39.59 | 0.43 |
| Citeseer | current | 54.62 | 3.42 | 60.97 | 58.67 | 46.53 | 0.32 |
| Citeseer | none | 52.60 | 1.12 | 58.83 | 56.02 | 45.76 | 0.00 |
| Citeseer | random | 54.69 | 1.79 | 61.61 | 60.89 | 42.30 | 0.38 |
| Citeseer | quality | 55.75 | 1.29 | 59.81 | 59.90 | 47.46 | 0.30 |

相对 `none`：

- Cora `current` 最终 F1 高 2.97，阶段平均高 2.26；
- Citeseer `current` 最终 F1 高 2.02，阶段平均高 2.14。

因此，跨边整体有用，尤其对持续阶段中的新旧信息交互有帮助。

### 3.2 顺序敏感性

| 数据集 | 顺序 | current | none | random | quality |
|---|---|---:|---:|---:|---:|
| Cora | canonical | 49.85 | 44.77 | 52.75 | 49.21 |
| Cora | order1701 | 52.59 | 52.46 | 59.67 | 51.68 |
| Cora | order1702 | 42.77 | 39.05 | 38.04 | 43.26 |
| Citeseer | canonical | 59.17 | 51.40 | 55.70 | 57.10 |
| Citeseer | order1701 | 53.77 | 54.09 | 52.17 | 54.01 |
| Citeseer | order1702 | 50.94 | 52.30 | 56.20 | 56.15 |

`random` 在 Cora 的两个顺序上很高，但在 `order1702` 反而低于 current；Citeseer 也没有稳定超过 current。`quality` 在 Citeseer 平均提升约 1.13，但在 Cora 下降约 0.35。因此不存在跨数据集、跨顺序都可靠的新端点规则。

## 4. 传播漂移结果

旧旧边索引没有被改写，但由于加入跨边后节点度数变化，重新计算的
`D^{-1/2}(A+I)D^{-1/2}` 会改变旧节点的归一化传播。

### 4.1 二、三阶段平均诊断量

| 数据集 | 拓扑 | 旧旧归一化相对漂移 | 旧节点传播相对漂移 | 跨边 X 消息相对量 | 总旧节点传播变化 |
|---|---|---:|---:|---:|---:|
| Cora | current | 0.32 | 0.27 | 0.36 | 0.31 |
| Cora | none | 0.00 | 0.00 | 0.00 | 0.00 |
| Cora | random | 0.35 | 0.32 | 0.35 | 0.39 |
| Cora | quality | 0.32 | 0.27 | 0.37 | 0.31 |
| Citeseer | current | 0.30 | 0.25 | 0.33 | 0.28 |
| Citeseer | none | 0.00 | 0.00 | 0.00 | 0.00 |
| Citeseer | random | 0.32 | 0.28 | 0.32 | 0.37 |
| Citeseer | quality | 0.29 | 0.23 | 0.34 | 0.27 |

这验证了一个重要机制结论：**frozen-old 只保证旧边和旧行不被写入，不保证旧节点的归一化传播不变。** 当前跨边会造成约 0.25--0.32 的旧节点传播相对漂移，并带来约 0.28--0.31 的总传播变化。

### 4.2 训练前后表征漂移

最终阶段训练后的旧节点 graph embedding 漂移与拓扑作用方向一致：无跨边为 0；current 为 Cora 0.43、Citeseer 0.32；random 分别为 0.51、0.38；quality 分别为 0.43、0.30。这里的量包含 student 参数重训影响，不能单独解释为拓扑造成，但与冻结历史模型的传播诊断共同说明跨边会改变历史节点的有效输入。

## 5. 成本与正确性

| 数据集 | 拓扑 | 18 次 student fit 时间 | student steps | 拓扑构造时间 |
|---|---|---:|---:|---:|
| Cora | current | 36.51 s | 16,512 | <0.01 s |
| Cora | none | 36.54 s | 16,512 | <0.01 s |
| Cora | random | 36.32 s | 16,512 | <0.01 s |
| Cora | quality | 36.29 s | 16,512 | <0.01 s |
| Citeseer | current | 48.82 s | 21,440 | <0.01 s |
| Citeseer | none | 48.52 s | 21,440 | <0.01 s |
| Citeseer | random | 48.63 s | 21,440 | <0.01 s |
| Citeseer | quality | 48.57 s | 21,440 | <0.01 s |

拓扑选择本身的成本可以忽略；性能差异不是由额外训练时间造成。所有 arms 使用相同的 18 次 fit、相同 student steps 和相同 memory payload 规模。远程 acceptance、每阶段 JSON、checkpoint 和硬件记录均已生成。

## 6. 结论与方法决策

### 已确认

1. 新旧跨边不是无效装饰：删除跨边会使两个数据集的最终和阶段平均性能整体下降。
2. 当前“文本最近邻跨边”不是唯一可能的有效规则；随机端点有时更强，说明当前规则可能过度集中或引入错误传播。
3. 但 random / quality 都有明显顺序或数据集依赖，不能直接替换 current。
4. 旧边冻结与旧传播保持是两个不同概念，后续 P4 需要考虑分块归一化、跨边权重控制或单向跨块传播。

### 当前不晋升

- 不把 random 设为默认：Cora order1702 和 Citeseer order1701 失败，顺序标准差明显增大。
- 不把 quality 设为默认：Citeseer 有收益，但 Cora 没有跨顺序收益。
- 不将 P3-B 结果包装成 SOTA；它是机制诊断，不是完整新算法。

当前默认继续为：

```text
Mean-X + dispersion allocation + frozen-old rows/old-old edges
       + text-nearest cross edges (cross-degree=1)
```

## 7. 后续优先级

2026-09-18 用户后续决定：有界跨边、单向跨块传播留待后续，先执行 P2-A。P2-A 已完成但未晋升；最新的方向定义与公平算子控制见 [P2-A 报告第 9.1 节](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md#91-后置的拓扑实验有界跨边与单向跨块传播)。下列清单保留原始建议，不代表已经运行，单向传播须区分 old-to-new 与 new-to-old。

1. **P3-B-2：有界跨边机制。** 固定当前跨边端点，比较 cross-edge weight、degree correction、block-normalized adjacency 和单向 new-to-old 消息，目标是减少约 0.25--0.32 的旧传播漂移。
2. **P3-B-3：跨边端点的独立验证。** 使用未参与开发的类别顺序和 seed group；只有 random/quality 在独立条件下稳定胜出，才考虑晋升。
3. **P4：历史管理。** 研究 bounded refresh / merge / split，但必须把拓扑漂移作为约束，而不是只看最终 F1。
4. P2 图条件文本 attention 继续后置；P1 extra budget 和 adaptive-rate 仍依赖选择可靠性与成本证据。

完整 JSON 结果：`results/p3b_topology_diagnostic_20260918/`。汇总脚本为 `code/v2_llm_fusion/summarize_p3b_topology_diagnostic.py`。
