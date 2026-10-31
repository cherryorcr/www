# Semantic-Recursive 小图统一基线比较报告

日期：2026-09-26  
状态：已完成两张小图、两个递归类别顺序、六个配对 Student seeds 的统一比较。  
结果目录：`results/tgc_semantic_recursive_unified_baselines_20260926`  
汇总目录：`results/tgc_semantic_recursive_unified_baselines_20260926_summary`

## 1. 结论摘要

本轮的目标不是把所有方法强行放进同一排行榜，而是回答三个问题：

1. Semantic-Recursive 在自己的递归 continual 协议下是否比共同控制和常用 coreset 更有效；
2. 直接让 Student 跨阶段微调带来的收益，是否只是训练状态带来的，而不是 memory 构造收益；
3. 不同类别顺序、预算和访问权限会怎样影响结论。

在严格匹配的 memory-only 比较中，Semantic-Recursive 相对 C0 的四个 cell 结果为：

| 指标 | 结果 |
|---|---:|
| Final Macro-F1 平均变化 | **+0.925 pp** |
| Final Accuracy 平均变化 | **+0.364 pp** |
| F1 为正的 cell | **3/4** |
| 最差 cell | **-0.528 pp** |
| New-class F1 平均变化 | **+1.418 pp** |
| Old-class F1 平均变化 | **+0.372 pp** |

因此，Semantic-Recursive 已经可以作为当前论文的主方法候选，并且相对 C0、random、k-center/FPS、herding 和 medoid 有明确的统一协议证据。但它还不能被表述为对所有公开 TAG 凝聚方法的 SOTA 证明：公开 TAG 方法的完整静态适配和 continual 访问权限不同，必须单独分层报告。

## 2. 统一递归协议

### 2.1 固定不变量

所有 memory-only 方法共享：

- Cora、Citeseer；
- `order19611`、`order19612` 两个类别顺序；
- stage 0、stage 1、stage 2；
- 每个 cell 内完全相同的 memory budget；
- 当前阶段 C0 的 `edge_index` 位置；
- 两层 GCN Student、384 hidden/output、冻结 MiniLM、可训练 text projection；
- 双向 graph-text InfoNCE；
- batch size 16；
- Student seeds `150..155`；
- Student steps `max(50, min(2048, 16 * memory_rows))`；
- 只在 report-dev 上计算指标；
- construction 不读取 test 标签，不按 report-dev F1 选择 memory。

主指标是节点分类的 all-prompt report-dev Macro-F1，同时报告 Accuracy。Accuracy 定义为 report-dev 中已见类别节点的正确预测比例；Macro-F1 对各已见类别的 F1 做宏平均。

### 2.2 每个 cell 的实际预算

两个顺序会改变每阶段当前 raw 数量，因此 fixed-rate 会导致不同的绝对预算。这个事实本身是协议影响，不能被隐藏：

| 数据集/顺序 | stage 0 | stage 1 | stage 2 |
|---|---:|---:|---:|
| Cora / order19611 | 43 | 60 | 94 |
| Cora / order19612 | 27 | 62 | 94 |
| Citeseer / order19611 | 42 | 78 | 111 |
| Citeseer / order19612 | 32 | 75 | 111 |

所以本报告只做**同一个 dataset/order cell 内的 paired 比较**，不把不同 order 的绝对 F1 直接排序。后续正式论文还应增加 matched-budget order control，固定每个顺序的预算后再单独测序列影响。

### 2.3 Semantic-Recursive 的协议位置

Semantic-Recursive 的 stage 0 与 C0 相同；stage 1、2 使用上一阶段 Semantic-Recursive memory 作为 parent：

```text
M0 = C0
M1 = Append(Semantic-Recursive(M0, delta1))
M2 = Append(Semantic-Recursive(M1, delta2))
```

每个新阶段只访问当前 raw delta，旧 synthetic prefix 保持不变。构造出的 payload 在同类 slot 内做确定性语义守恒匹配，再写入同一阶段 C0 的拓扑位置。

## 3. 比较方法及其访问权限

| 方法 | 类型 | 说明 |
|---|---|---|
| C0 | 内部控制 | 项目内部 `dispersion + append` 控制，不是公开方法；与主方法共享预算、拓扑和 Student。 |
| Semantic-Recursive | 当前主方法 | two-prototype 局部重组 + Mean-X + top-5/320 文本 + semantic conserved payload assignment + recursive append。 |
| Random real-node | coreset 控制 | 当前 delta 中随机选择真实节点；不产生 synthetic node。 |
| k-center / greedy FPS | 强 coreset 控制 | 在图文联合空间中做 farthest-first 真实节点选择。这里两者是同一个贪心算子，不重复计为两个独立方法。 |
| Herding | coreset 控制 | 选择使已选节点均值接近全体均值的真实节点。 |
| K-means medoid | coreset 控制 | 联合空间聚类后选择真实 medoid。 |
| Full raw graph scratch | 离线参考 | 使用截至当前阶段全部合法 raw pool，重新训练 Student；不受 memory 行数限制，不能当作同预算凝聚方法。 |
| Student delta-only fine-tune | 参数状态对照 | stage 0 训练 Student，后续只用当前 raw delta 微调；保留上一阶段 Student 参数。 |
| Student C0-replay + delta fine-tune | 参数状态对照 | 保留 Student 参数，并在当前 C0 memory 与当前 raw delta 上继续训练。 |

最后两种方法不属于 memory-only condensation。它们额外拥有跨阶段模型状态，因此只用于回答“直接持续学习能达到什么效果”，不能直接作为凝聚算法的同资源基线。

## 4. 主表：跨两个小图和两个顺序的汇总

数值为四个 dataset/order cell 的等权平均。`Delta vs C0` 是最终阶段 F1 的 cell-level 平均差；`Paired delta vs Semantic` 是 4 个 cell、6 个 Student seed，共 24 个配对样本的 95% t 置信区间。

| 方法 | Final F1 | Final Acc. | Stage-avg F1 | Stage-avg Acc. | Old F1 | New F1 | Delta vs C0 | 正收益 cell | vs Semantic 的 paired F1 delta (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 internal control | 52.09 | 59.22 | 47.25 | 47.95 | 51.95 | 51.34 | 0.00 | 0/4 | -0.92 [-2.08, +0.23] |
| **Semantic-Recursive** | **53.02** | **59.59** | 48.35 | 48.73 | **52.32** | **52.76** | **+0.92** | **3/4** | 0.00 |
| Random real-node coreset | 48.22 | 55.74 | 46.59 | 46.68 | 48.39 | 47.34 | -3.87 | 1/4 | -4.80 [-7.14, -2.46] |
| k-center / greedy FPS | 48.23 | 55.32 | 46.12 | 47.65 | 48.53 | 46.70 | -3.86 | 0/4 | -4.78 [-6.57, -2.99] |
| Herding real-node coreset | 50.40 | 58.97 | **48.45** | **49.32** | 49.63 | 50.02 | -1.69 | 1/4 | -2.61 [-4.71, -0.52] |
| K-means medoid coreset | 50.04 | 57.73 | 47.85 | 48.23 | 51.28 | 46.95 | -2.05 | 0/4 | -2.98 [-4.81, -1.14] |
| Full raw graph scratch | 49.73 | 54.96 | 45.67 | 42.37 | 48.84 | 49.84 | -2.37 | 1/4 | -3.29 [-4.80, -1.78] |
| Student delta-only fine-tune | 51.26 | 57.57 | 45.59 | 44.92 | 53.57 | 46.72 | -0.83 | 2/4 | -1.76 [-4.23, +0.72] |
| Student C0-replay + delta fine-tune | 53.03 | 58.76 | 47.19 | 46.03 | **54.73** | 49.33 | **+0.94** | 2/4 | +0.02 [-1.98, +2.02] |

### 4.1 逐 cell 最终阶段 F1/Accuracy

格式为 `Macro-F1 / Accuracy`。

| Dataset/order | C0 | Semantic-Recursive | Random | k-center/FPS | Herding | Medoid | Full raw | Delta fine-tune | Replay + delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Citeseer/order19611 | 56.62/62.43 | **57.70/62.33** | 50.67/57.00 | 51.63/56.84 | 49.86/55.38 | 54.05/61.55 | 52.54/57.52 | 51.70/54.60 | 54.55/57.89 |
| Citeseer/order19612 | 54.54/61.91 | 54.01/61.44 | **55.96/59.98** | 54.31/60.24 | 51.76/59.77 | 54.07/62.12 | 49.26/52.77 | 50.97/56.01 | 53.82/58.93 |
| Cora/order19611 | 49.51/55.23 | **50.21/56.40** | 43.28/51.72 | 45.10/49.88 | 46.10/53.94 | 46.62/53.26 | 49.19/54.18 | 52.68/61.44 | **54.76/62.79** |
| Cora/order19612 | 47.70/57.32 | **50.14/58.18** | 42.96/54.24 | 41.89/54.31 | **53.89/66.79** | 45.41/54.00 | 47.91/55.35 | 49.69/58.24 | 49.01/55.41 |

### 4.2 Memory 与训练成本

memory-only 方法的最终行数为：Cora 94、Citeseer 111。Full raw scratch 使用的最终 raw rows 约为 Cora 1895、Citeseer 2229，因此不是等 memory 预算参考。最终阶段 Student wall time 的四-cell 平均约为：

| 方法 | Student 时间/最终 cell |
|---|---:|
| C0 | 20.8 s |
| Semantic-Recursive | 20.7 s |
| Random/k-center/Herding/Medoid | 20.7--20.8 s |
| Full raw scratch | 26.9 s |
| Student delta-only fine-tune | 59.7 s |
| Student C0-replay + delta fine-tune | 60.3 s |

这表明 Semantic-Recursive 的主要额外成本在构造阶段，而不是 Student 重训阶段；直接持续学习获得的收益伴随着约 2.9 倍的 Student 训练时间，并且拥有跨阶段参数状态。

## 5. 已有公开 TAG 方法：单独的静态协议层

以下结果来自此前的 `EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md`，不是本轮递归协议重跑。它们保留各自方法的静态构造方式、访问完整历史 raw 的能力或监督训练设置，因此只能作为公开方法背景对照，不能与上面的 continual memory-only 主表直接排序。

| 方法 | Cora Final F1 | Citeseer Final F1 | 协议说明 |
|---|---:|---:|---|
| TAGSAM-ASM self-loop | 47.73 | 51.09 | 静态适配，self-loop topology |
| TAGSAM-ASM semantic | 52.86 | 42.00 | 静态适配，semantic topology |
| STAD-local-canonical | 33.99 | 28.91 | STAD 的本地适配，不是官方完整复现；监督式 TAG 方法 |

这张表的作用是说明 Semantic-Recursive 需要在后续工作中补齐严格的公开 continual TAG baseline，而不是把静态适配结果伪装成同协议比较。DANCE 等方法当前没有可确认的官方实现，不能虚构分数。

此前还做过一个 STAD 原生监督训练 sanity check：Cora 的 STAD synthetic graph 为 `73.63 F1 / 76.81% Accuracy`，Citeseer 为 `67.46 F1 / 72.10% Accuracy`。它使用监督 GCN 和 STAD 自己的训练任务，不是当前冻结 MiniLM + InfoNCE Student，因此只作为任务上界/训练范式诊断，不能放进上面的 continual 主表。

## 6. 结果说明了什么

### 6.1 Semantic-Recursive 的有效性

它不是依靠保留更多行获得提升：同一 cell 内与 C0 行数、边位置、Student 和 seeds 都相同。其主要收益来自：

1. two-prototype 操作把固定行预算从简单区域转移到图文联合离散度更高的局部区域；
2. X、文本和 mass 作为一个完整 payload 共同移动；
3. 递归阶段使用上一阶段 Semantic memory，而不是每次回到 C0 parent；
4. 同类语义守恒匹配减少结构 proposal 与 slot 语义的错位。

新类 F1 平均提升约 1.42 pp，说明它最明确的作用是改善新类塑性，而不是单纯保守旧类。Citeseer/order19612 的 -0.53 pp 也说明该机制仍有顺序依赖，不能宣称普遍单调提升。

### 6.2 为什么 k-center/FPS 没有更强

k-center/FPS 选择的是图文空间中的真实节点，擅长覆盖几何极端点，但本任务的 Student 训练目标需要 synthetic X、文本和当前拓扑共同可重训。真实节点 coreset 没有完成同样的 payload 凝聚，因此几何覆盖不等于 Student-compatible condensation。它在四个 cell 中均低于 Semantic-Recursive，且 paired F1 CI 明显为负。

### 6.3 为什么直接 Student 持续学习不能替代凝聚方法

Replay + delta fine-tune 的最终 F1 平均为 53.03，几乎与 Semantic-Recursive 的 53.02 相同；但它：

- 保留上一阶段 Student 参数；
- 继续访问当前 raw delta；
- 训练时间约为 memory-only 的 2.9 倍；
- 不产出可独立重训、可传输的 continual synthetic memory。

因此它是很有价值的上界/控制，不是对 Semantic-Recursive 的否定，也不能作为同资源凝聚 baseline。

### 6.4 为什么 Full raw scratch 不能直接叫理想上界

Full raw scratch 使用了远多于 5% memory 的 raw rows，但仍使用当前固定 Student steps 和训练目标；没有充分 checkpoint 搜索，也没有监督标签训练。因此本报告把它称为 offline full-raw reference，而不是理论上界。它在当前协议下反而低于 Semantic-Recursive，说明“完整图”并不自动等于更好的 Student 结果，训练预算和优化路径同样重要。

## 7. 后续正确控制协议影响的方案

### P0：matched-budget order control

当前两个 order 采用真实 fixed-rate，所以 raw 到达量不同会改变预算。下一轮应固定：

- base 类别数和每阶段类别数；
- 每阶段 raw rows；
- 每阶段 memory rows；
- Student steps；
- report-dev 节点集合；
- Student seeds。

只改变类别排列，至少增加 3 个、最好 5 个独立 order。主指标使用同一 cell 内 paired delta，不使用单个 canonical order 宣称普适优势。

### P1：三类协议分层报告

正式论文建议保留三层：

1. **Matched-budget memory-only**：Semantic-Recursive、C0、random、k-center/FPS、herding、medoid、其他可复现 continual condensation；
2. **Realistic fixed-rate continual**：允许预算随数据自然增长，报告真实 rows、cost 和每顺序 paired delta；
3. **Stateful Student controls**：delta fine-tune、replay + delta、full raw scratch，明确访问权限、参数状态和训练成本。

### P2：公开 TAG 方法的正确接入

静态 TAGSAM、STAD 继续单列为静态适配层；只有当某个方法能够在相同 current-delta access、memory budget、Student 和 report-dev 规则下运行，才进入 continual 主表。不能把监督 GCN 的 STAD 结果与 InfoNCE Student 的 F1 直接当作同任务冠军比较。

### P3：额外指标

除 final Macro-F1 外，最终主表应保留：

- Accuracy；
- stage-average Macro-F1/Accuracy；
- old-class F1；
- new-class F1；
- paired 95% CI；
- worst-stage；
- positive-cell ratio；
- memory rows；
- construction time；
- Student training time。

## 8. 当前论文表述建议

可以写：

> Under a matched recursive continual protocol on Cora and Citeseer, Semantic-Recursive improves the internal C0 control by 0.925 percentage points in final Macro-F1 on average across four dataset-order cells, with positive gains in three cells. Its gain is accompanied by a 1.418-point average improvement in new-class F1, while memory size, topology positions, Student architecture, training rule, and Student seeds are fixed within each cell.

不能写：

- Semantic-Recursive 普遍优于所有 TAG 凝聚方法；
- Semantic-Recursive 已经证明 SOTA；
- static TAGSAM/STAD 的分数与 continual 主表是严格同协议；
- order19611/19612 的绝对 F1 可以直接横向排序；
- Student replay + delta 是同 memory 预算的凝聚方法。

## 9. 可复现实验资产

- 统一 runner：[run_semantic_recursive_unified_baselines_20260926.py](../../code/v2_llm_fusion/run_semantic_recursive_unified_baselines_20260926.py)
- 汇总 runner：[summarize_semantic_recursive_unified_baselines_20260926.py](../../code/v2_llm_fusion/summarize_semantic_recursive_unified_baselines_20260926.py)
- 原始结果：[results.json](../../results/tgc_semantic_recursive_unified_baselines_20260926/results.json)
- 主表汇总：[summary.md](../../results/tgc_semantic_recursive_unified_baselines_20260926_summary/summary.md)
- 外部静态协议：[EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md](EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md)
