# 语义拓扑强控制结项报告

本报告总结在 arXiv 和 BookChild 上完成的 semantic-kNN / local-cycle 强控制实验。实验的目的，是判断 T+degree 的收益究竟来自语义块内构图、跨块容量分配，还是“单环/路径保持”这一特定拓扑形式。

## 结论先行

- 本批新增 1,440 个 student fit、80 张合成 memory；每个数据集 720 个 fit，其中 NC 240、LP 480。旧的 C0、internal-only、T+degree、random-ring、degree-shuffle、cross-only 结果复用上一批 2,160 个已审计 fit，没有把复用结果算作本批重跑。
- 语义块内构图有证据支持，但没有出现跨数据集、跨任务统一最优的语义拓扑。BookChild 上 local-cycle 的五阶段平均 NC Macro-F1 为 17.515，明显高于 C0 的 16.189 和 T+degree 的 16.844；arXiv 上 NC 的五阶段平均仍以 C0 的 21.670 最好。
- T+degree 的“均匀跨块容量”没有被证明是普适增益。BookChild 上 local-cycle + 最近邻跨边在最终 NC/LP 上已经接近 T+degree；arXiv 上 cycles-capacity 的最终 NC 较 T+degree 高 0.567 pp，但五阶段平均只高 0.027 pp，不能据此把单环或 capacity 晋升为主方法。
- 当前推荐继续保留 C0 作为共同冻结基线、T+degree 作为机制参照。下一步若只选一个拓扑候选继续开发，应优先考察“semantic local-cycle + nearest cross”，同时把 medoid、独立新顺序和端到端构造成本作为后续确认项。

## 协议与方法臂

所有方法共享同一批合成行、X、文本、质量权重、memory budget、row order、student、batch、steps、报告 seed、类别顺序和 LP split；只替换新块内部边以及预先指定的跨块端点。旧行和旧旧边逐阶段保留。

| 方法臂 | 新块内部拓扑 | 新旧跨块拓扑 | 用途 |
|---|---|---|---|
| C0 | 原冻结构造 | 原最近邻规则 | 共同基线 |
| internal-only | T+degree 的块内拓扑 | C0 跨边 | 分离块内构图 |
| T+degree | T+degree 的块内拓扑 | uniform-capacity | 原研究候选 |
| knn-nearest | 固定 n 条边的 budgeted semantic 2-NN | C0 最近邻 | 常规语义邻居控制 |
| knn-capacity | 同上 | uniform-capacity | kNN 与 capacity 联合控制 |
| cycles-nearest | 文本相似度贪心、节点度上限 2、允许提前闭环的 local-cycle cover | C0 最近邻 | 检验单环约束是否必要 |
| cycles-capacity | 同上 | uniform-capacity | local-cycle 与 capacity 联合控制 |

这里的 budgeted 2-NN 先保留对称化 1-NN 的覆盖边，再从对称化 2-NN 中补齐到新块节点数；它不是严格 degree-matched vanilla kNN。local-cycle 保证新块内部每个节点度为 2、边数为节点数，但允许形成多个局部环，因而不等价于 T+degree 的单一路径/闭环约束。

下游 NC 是真实完整图上的 transductive report-dev Macro-F1；LP 使用固定真实边划分和独立 GAE student，报告全体、old、new、cross、within-new 的 AUC/AP。这里的绝对分数用于观察凝聚 memory 的可重训效用，方法比较优先看同一 student 条件下的配对差。

## 最终阶段分类结果

均值跨两个冻结顺序、六个 student seeds。`all` 是累计 Macro-F1，`old` 和 `new` 是对应类别组 F1，单位为百分比。

### arXiv

| 方法 | all | old | new |
|---|---:|---:|---:|
| C0 | 20.759 | 20.910 | 19.902 |
| T+degree | 20.439 | 20.626 | 19.382 |
| knn-nearest | 20.829 | 20.902 | 20.410 |
| knn-capacity | 20.775 | 20.778 | 20.756 |
| cycles-nearest | 20.336 | 20.337 | 20.325 |
| cycles-capacity | 21.006 | 20.861 | 21.830 |

cycles-capacity 是 arXiv 最终 NC 最好，但它相对 C0 的优势只有 +0.247 pp，五阶段平均为 21.580，低于 C0 的 21.670；因此优势主要出现在最后阶段，不能称为稳定持续收益。knn-nearest 相对 T+degree 的最终 all/new 差值为 +0.389/+1.028 pp，但旧类差值只有 +0.276 pp，且五阶段平均 all 仍低于 C0 0.186 pp。

### BookChild

| 方法 | all | old | new |
|---|---:|---:|---:|
| C0 | 13.050 | 13.316 | 11.190 |
| T+degree | 14.600 | 14.944 | 12.193 |
| knn-nearest | 14.168 | 14.566 | 11.382 |
| knn-capacity | 13.960 | 14.497 | 10.199 |
| cycles-nearest | 14.555 | 15.074 | 10.926 |
| cycles-capacity | 13.757 | 14.195 | 10.693 |

BookChild 的最终分数仍由 T+degree 取得，但 cycles-nearest 只低 0.045 pp；更关键的是五阶段平均 all F1 为：C0 16.189、T+degree 16.844、knn-nearest 17.907、knn-capacity 17.669、cycles-nearest 17.515、cycles-capacity 17.329。说明 local-cycle 的过程效用较好，但最终阶段和过程平均的排序不同，不能只看一个 endpoint。

## 链路预测结果

### 最终阶段 AUC/AP

| 数据集 | 方法 | all AUC | all AP | new AUC | cross AUC | within-new AUC |
|---|---|---:|---:|---:|---:|---:|
| arXiv | C0 | 96.044 | 95.763 | 95.932 | 97.465 | 94.242 |
| arXiv | T+degree | 96.031 | 95.711 | 95.864 | 97.299 | 94.525 |
| arXiv | knn-nearest | 96.272 | 96.028 | 95.965 | 97.473 | 94.543 |
| arXiv | cycles-nearest | 96.124 | 95.828 | 95.731 | 97.284 | 94.395 |
| arXiv | cycles-capacity | 96.268 | 96.002 | 96.033 | 97.396 | 94.697 |
| BookChild | C0 | 92.205 | 91.366 | 92.488 | 92.414 | 92.714 |
| BookChild | T+degree | 92.927 | 92.345 | 93.223 | 93.117 | 93.589 |
| BookChild | knn-nearest | 92.433 | 91.868 | 92.811 | 92.729 | 93.165 |
| BookChild | cycles-nearest | 92.869 | 92.260 | 93.195 | 93.129 | 93.534 |
| BookChild | cycles-capacity | 92.820 | 92.219 | 93.175 | 93.056 | 93.584 |

arXiv 上 kNN-nearest 和 cycles-capacity 的最终 all AUC 分别较 T+degree 高 0.241 和 0.237 pp；BookChild 上 cycles-nearest 较 T+degree 低 0.058 pp，cycles-capacity 低 0.107 pp。五阶段平均则 BookChild 的 cycles-nearest all AUC 为 91.899，高于 T+degree 的 91.773；arXiv 五阶段平均以 knn-nearest 的 96.714 略高于 T+degree 的 96.499。这个结果支持“语义块内拓扑对 LP 有用”，但不支持固定一种拓扑在两个数据集上统一胜出。

## 配额与拓扑机制解释

### 1. 块内语义边确实是主要候选因素

BookChild 上所有四个新臂的五阶段平均 NC all F1 都高于 C0，增幅为 +1.139 到 +1.718 pp；LP all AUC 增幅为 +1.407 到 +1.729 pp。这个信号在最终阶段仍存在，但大小缩小，说明语义块内边可能改善阶段间训练稳定性，而不是保证最后一阶段绝对最优。

### 2. capacity 不是稳定的独立增益

在相同内部拓扑下切换 nearest/capacity，结果方向不一致：arXiv 最终 NC 中 capacity 相对 nearest 对 kNN 为 -0.054 pp、对 local-cycle 为 +0.671 pp；BookChild 则分别为 -0.209 pp 和 -0.798 pp。LP 也随数据集和内部算子改变方向。因此 uniform-capacity 只能保留为控制臂，不能直接写成主算法增益。

### 3. 单环约束没有被证明必要

local-cycle 允许提前闭环和多连通分量，但 BookChild 上 cycles-nearest 已接近 T+degree，五阶段平均还更高；arXiv 上 cycles-capacity 的最终 NC 更高但轨迹平均不高。当前证据更支持“固定边数 + 语义相似性 + 有界节点度”这一较宽泛机制，尚不足以支持“必须保持单环/单路径”。

### 4. 不能把本批结果写成独立泛化确认

本批继续使用 order 49011/49012、student seeds 210--215、LP split 49101/49102。它是冻结条件下的机制强控制，不能算新的类别顺序或新 seed group 确认。semantic medoid 也尚未完成，因此 P1 的“常规语义强控制”只完成了 budgeted kNN 和 local-cycle 两类，不应标记为全部完成。

## 审计与成本

- 两个数据集的矩阵均完整：每个数据集 240 个 NC fit、480 个 LP fit、720 个新增 fit；80 张新增 memory 的边数、节点数、旧行、旧旧边和评价输入均通过审计。
- BookChild `passed=true` 且 `strict_metric_reload_passed=true`。arXiv `passed=true`，但保留两条初始阶段 NC 重载告警：order 49012、seed 213、cycles-nearest/capacity；最大指标差为 0.026873 pp，来自一个接近并列的预测，all F1 未改变，且不影响最终阶段。原始指标没有被替换，也没有放宽审计容差。
- 原始 runner 和 `complete.json` 仍保留错误的计数模板（NC=480、LP=960、总数=720、expected=1440），不能用它判断缺少实验。`comparison_integrity.json` 根据完整逐-fit 矩阵给出每图 NC=240、LP=480、总数=720，并显式标记 `runner_count_metadata_valid=false`。这是计数元数据错误；没有重写 fit JSON、`results.json`、checkpoint 或冻结 runner，也不追改其源码哈希。

### 补充结构检查

arXiv 和 BookChild 的补充检查均已完成，分别为 `passed=true`、40 张 memory、20 次 local-cycle 度检查。四臂全部 80 张 memory 的当前跨块边与预设 nearest/capacity 参照一致；40 张 local-cycle memory 的完整度序列与对应 `internal_only`/`t_degree` 一致，其新块内部每个节点的度为 2。这使 local-cycle 与单环的比较能控制度序列，而不仅是控制总边数。

**kNN 不是等最大度控制。** 其新块最大度在 arXiv 可达 17，在 BookChild 可达 39；local-cycle 为 2。因此 kNN 与 local-cycle 的差异仍混合了端点选择、度分布和连通性，不能仅凭分数断言哪一因素导致提升。相同节点内容和相同边数也不意味着传播算子等价。

### 块内构图微基准

平台为 RTX 4090、PyTorch 2.4.1+cu121、4 个 CPU 线程。每图 2 个顺序 × 5 个阶段，共 10 个新块；每块先预热一次，再轮换算子顺序测量 3 次。下表是“块内 3 次计时中位数”的跨块算术平均，单位秒。新块规模分别为 arXiv 347--3419 行、BookChild 44--1748 行。

| 数据集 | C0 块内 | T+degree 单环块内 | semantic-kNN | local-cycle |
|---|---:|---:|---:|---:|
| arXiv | 0.2753 | 0.5263 | 0.1891 | 0.6243 |
| BookChild | 0.2736 | 0.1181 | 0.0389 | 0.1425 |

这是结果产生后的实现微基准：包含现有块内算子的核计算、选边与设备转换；**不包含原始数据读取、X/文本构造、跨块边、写盘和 student 训练**。kNN 的块内平均耗时最低，单环/local-cycle 相对 C0 的耗时方向则随数据集改变。块大小和实现路径不同，不能据此宣称端到端加速、普遍低于 2× 成本门或更优渐近复杂度。逐块原始计时和范围均保留在结果文件中。

### Student 时间与存储

下表为既有 fit 记录的 `train_seconds` 平均值，每图每臂 NC 为 60 次、LP 为 120 次，单位秒。C0/T+degree 来自上一批训练，四个新臂来自本批；跨批运行环境波动未通过交错重跑排除，因此小幅计时差不能归因于算法加速。NC/LP 的计时边界与任务不同，也不能直接彼此比较。

| 数据集/任务 | C0 | T+degree | kNN-nearest | kNN-capacity | cycles-nearest | cycles-capacity |
|---|---:|---:|---:|---:|---:|---:|
| arXiv NC | 4.924 | 4.903 | 4.678 | 4.688 | 4.673 | 4.674 |
| arXiv LP | 0.657 | 0.655 | 0.589 | 0.593 | 0.588 | 0.588 |
| BookChild NC | 4.557 | 4.557 | 4.463 | 4.451 | 4.448 | 4.459 |
| BookChild LP | 0.585 | 0.585 | 0.588 | 0.584 | 0.582 | 0.584 |

四臂在同一顺序/阶段的张量字节数、文本 UTF-8 字节数一致；最终 memory 文件大小为 arXiv 19.6133--19.6380 MiB、BookChild 8.8970--8.9314 MiB（1 MiB = 2^20 字节）。范围包含不同顺序，序列化文件也含元数据；它不是构造时 RAM/显存峰值，更不等于原图字节数的 5%。当前没有 topology-only 改动节省存储的证据。

两图训练后采集的 MiniLM 文件指纹一致；这是补充 provenance，不追称为训练前预注册。补充收据、矩阵、度/跨边约束、计时样本和单位换算已纳入汇总脚本检查；17 个 semantic-topology 单元测试通过，连同 P3-B 与因果审计/汇总在内的 34 个 topology 回归测试也全部通过（包含前述 17 个，不相加）。另直接从逐-fit 原始记录复核 C0、T+degree、kNN-nearest、cycles-nearest 的 NC 最终/阶段平均与 LP 最终 AUC，与汇总一致；原始结果及补充文件哈希通过复核。**本轮补齐的是结构与局部成本诊断，完整端到端构造/推理成本仍未完成，也没有新增独立确认实验。**

可从本地已回传的原始记录重建汇总，不需要重训：

```powershell
python code/v2_llm_fusion/report_semantic_topology_controls.py --root results/semantic_topology_controls_20260921 --baseline-root results/topology_causal_ablation_20260921_formal
python -m unittest discover -s code/v2_llm_fusion -p 'test_semantic_topology*.py' -v
python -m unittest discover -s code/v2_llm_fusion -p 'test_*topology*.py' -v
```

## 当前决定与后续工作

本批不把任何 semantic control 升为 CTAG-final，也不替换 C0 的冻结身份。最合理的后续顺序是：

1. 在预注册的新顺序/新 report seeds 上复核 `cycles-nearest` 与 `knn-nearest`，并补做 medoid；
2. 单独记录原始数据读取、X/文本构造、拓扑构造、memory 写盘和 student 训练时间；
3. 只选择一个内部算子进入 bounded propagation / bounded refresh 研究，不同时叠加 dynamic budget、response selector、文本 attention；
4. 若 semantic local-cycle 在新条件和 LP/NC 两任务上仍稳定，再将“语义块内拓扑 + 有界跨块负载”组织成 CTAG-1；否则保留为机制分析和负结果。

原始结果、配对表、完整性收据和独立审计见：

- [semantic topology protocol](SEMANTIC_TOPOLOGY_CONTROLS_PROTOCOL_ZH_2026-09-21.md)
- [comparison report](../../results/semantic_topology_controls_20260921/SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md)
- [paired contrasts](../../results/semantic_topology_controls_20260921/paired_semantic_contrasts_pp.csv)
- [comparison integrity](../../results/semantic_topology_controls_20260921/comparison_integrity.json)
- [internal topology cost summary](../../results/semantic_topology_controls_20260921/internal_topology_cost_summary.csv)
- [student training costs](../../results/semantic_topology_controls_20260921/training_costs.csv)
- [memory storage summary](../../results/semantic_topology_controls_20260921/memory_storage_summary.csv)
- [arXiv supplemental checks](../../results/semantic_topology_controls_20260921/arxiv/supplemental_topology_checks.json)
- [BookChild supplemental checks](../../results/semantic_topology_controls_20260921/bookchild/supplemental_topology_checks.json)
- [arXiv audit](../../results/semantic_topology_controls_20260921/arxiv/independent_audit.json)
- [BookChild audit](../../results/semantic_topology_controls_20260921/bookchild/independent_audit.json)
