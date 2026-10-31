# 统一协议公开基线对比报告（大图）

日期：2026-10-01  
状态：第一批 strict-continual mechanism-adapted 基线已完成；TAGSAM、PUMA 和 STAD 分层参考尚未并入本表。

## 1. 结论摘要

本轮在 arXiv、BookChild、SportsFit 三张大图上，使用相同的 `base+4` 类别增量协议、5% cumulative memory、相同 edge-aware Student 和 6 个 Student seeds，对以下方法进行了比较：

- `append-edge`：当前持续追加控制；
- `Bounded X+P/XPA`：当前封存的主方法候选；
- `random-real`：随机真实节点 coreset；
- `k-center`：几何覆盖 coreset；
- `medoid`：类内 medoid coreset；
- `CaT-MMD-TAG`：CaT 的当前 TAG 接口适配；
- `GECC-hard-TAG`：GECC 的当前 TAG 接口适配。

三张大图等权平均的最终阶段 Macro-F1 为：

| 方法 | Final Macro-F1 | Final Accuracy | 阶段平均累计 Macro-F1 | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|
| `Bounded X+P/XPA` | **18.95** | **26.76** | 21.70 | **26.17** | **21.36** |
| `k-center` | 18.18 | 23.99 | **22.34** | 24.67 | 20.45 |
| `GECC-hard-TAG` | 17.59 | 22.47 | 21.29 | 24.31 | 19.89 |
| `random-real` | 17.26 | 21.89 | 20.13 | 23.10 | 19.71 |
| `append-edge` | 17.23 | 23.54 | 20.99 | 23.75 | 19.67 |
| `medoid` | 17.40 | 21.85 | 21.26 | 23.54 | 20.22 |
| `CaT-MMD-TAG` | 17.38 | 23.03 | 20.66 | 23.76 | 19.58 |

这里的三数据集均值是“先对每张图求 Student-seed 均值，再对三张图等权平均”，不是按节点数合并后的 pooled F1。`阶段平均累计 Macro-F1` 是五个阶段的 cumulative Macro-F1 平均，不是五个 current-class F1 的平均。

最重要的配对结果：

| 比较 | arXiv | BookChild | SportsFit | 三图平均 |
|---|---:|---:|---:|---:|
| X+P/XPA - append-edge，Final F1 | +0.62 | +0.33 | +4.22 | **+1.72** |
| X+P/XPA - append-edge，Accuracy | +4.17 | +0.01 | +5.48 | **+3.22** |
| X+P/XPA - k-center，Final F1 | -0.43 | +1.06 | +1.67 | **+0.77** |
| X+P/XPA - k-center，阶段平均累计 F1 | +0.81 | +0.73 | -3.48 | **-0.64** |

因此当前最稳妥的结论是：

> 在本轮固定的三张大图、单类别顺序和 strict-continual 接口下，Bounded X+P/XPA 相比 append-edge 在三个数据集上均有正的最终阶段收益，并且在三图平均 Final F1、Final Accuracy、Old F1 和 New F1 上均为最高；但 k-center 在阶段平均累计 F1 上更强，CaT/GECC 在部分数据集的最终阶段超过 X+P/XPA。当前方法已证明有效，但尚不能声称对所有公开凝聚方法稳定 SOTA。

## 2. 统一协议

### 2.1 任务与访问边界

| 项目 | 冻结设置 |
|---|---|
| 任务 | text-attributed graph 的类别增量，`base+4`，共 5 个阶段 |
| memory | 每阶段累计约 5% 的合法训练池节点数 |
| 构造访问 | 当前阶段 delta + parent synthetic memory |
| 历史 raw | 禁止回读历史完整 raw graph；`historical_raw_reads=0` |
| 标签隔离 | report/test 标签不参与构造和选择 |
| Student | 原有 edge-aware 两层 GCN、冻结 MiniLM 文本编码器、可训练 text projection、双向 graph-text InfoNCE |
| Student 训练 | 每个阶段从头训练 Student；batch=16；lr=0.001；steps=`max(50,min(2048,16*K))` |
| Student seeds | 100、101、102、103、104、105 |
| 指标 | cumulative Macro-F1、Accuracy、Old F1、New F1、阶段轨迹和成本 |
| 拓扑 | edge-aware 方法使用同一 Student 传播算子；CaT/GECC 保留 edge-free 适配语义 |

### 2.2 每张大图的预算和类别组

| 数据集 | 五阶段 memory rows | 类别组规模 |
|---|---|---|
| arXiv | 4962 / 5246 / 5502 / 5734 / 5926 | 16 / 6 / 6 / 6 / 6 |
| BookChild | 2344 / 2458 / 2551 / 2629 / 2690 | 按 frozen base+4 manifest |
| SportsFit | 4491 / 4965 / 5430 / 5802 / 6056 | 5 / 2 / 2 / 2 / 2 |

类别具体编号、stage ids hash、source manifest 和每个 memory 的 SHA256 保存在结果 JSON 与 source protocol 中。不同数据集的类别顺序不同，因此不能把某一张图的绝对 F1 直接解释成方法难度排序。

## 3. 各方法到底做了什么

### 3.1 append-edge

这是当前的持续追加控制：加载已经封存的 append memory，保持旧节点和旧旧边，只为新块补充块内边和有限新旧连接。它不是公开论文方法，而是用来回答“当前 X/P 内容机制相对于普通追加是否有收益”。

### 3.2 Bounded X+P/XPA

本轮按当前封存的大图 artifact 直接评估，不在基线 runner 中重新构造。它保留当前候选主方法的配置：

1. 使用当前 delta 和 parent synthetic memory；
2. 对新合成节点内容 `X` 做受限 Student-compatible 调整；
3. 用 `P` 在候选内容和合法来源之间做有界 payload transport；
4. 使用当前 edge-aware 拓扑配置和已经封存的拓扑策略；
5. 固定 memory rows、文本编码器、Student 和 report-dev 评价。

结果中的 arm 名称是 `x_plus_p`。由于它使用的是已生成的封存 memory，runner 中记录的 `construction_seconds` 只是加载/哈希审计时间，不能拿来宣称 X+P 的完整构造成本低于 coreset、CaT 或 GECC。完整 outer construction 成本应引用 X+P 自己的 source artifact。

### 3.3 random-real

在当前 delta 中随机选择合法真实节点，按相同 rows 形成 memory。为了让真实节点可以进入统一 edge-aware Student，适配器为每个 block 构造 text-kNN 块内边，并补一条文本最近的跨块连接。它是随机真实节点下界，不是“保留原始图诱导子图”的唯一实现。

### 3.4 k-center

在当前 delta 的特征空间内按类别做 farthest-first 几何覆盖，选出的中心对应真实节点；文本仍使用对应原始文本，拓扑使用与 random-real 相同的 text-kNN + 一条跨块边控制。它是本轮最强的低成本 coreset 对照之一。

### 3.5 medoid

在类内特征空间进行聚类/中心选择，最终返回真实节点 medoid，而不是自由合成向量。文本和拓扑适配规则与 k-center 相同。它检验“选择真实代表点”本身是否已经足够。

### 3.6 CaT-MMD-TAG adaptation

这是 CaT 机制的当前 TAG 适配，不是 CaT 官方端到端复现：

1. 对当前 delta 按类别分配合成 rows；
2. 用 propagated feature 初始化合成 `X`；
3. 用类条件 MMD 和随机特征映射优化合成 `X`；
4. 旧 memory 直接保留，禁止历史 raw refresh；
5. 用当前 delta 内的 class-local nearest text 片段构造文本并重新编码；
6. 保留 edge-free memory 语义，由 Student 自带 self-loop 训练。

它保留了 CaT 的核心“类平衡 continual condensation block”，但原论文的监督图学习、TiM 训练和数值图输入被替换为本项目统一的 TAG InfoNCE Student，因此必须称为 `mechanism-adapted`。

### 3.7 GECC-hard-TAG adaptation

这是 GECC 的 hard clustering 适配：

1. 在当前 delta 内计算传播后的特征；
2. 按类别做 hard Lloyd/KMeans 聚类；
3. 使用多次随机重启和类内平衡项选择中心；
4. 旧中心进入 parent memory 后直接继承，不读取历史 raw；
5. 文本从当前 delta 的近邻片段构造；
6. 保留 edge-free memory 语义。

它没有复现 GECC 的累计 raw evolving、fuzzy clustering 或官方下游模型，因此也是 `mechanism-adapted`，不是官方结果。

## 4. 按数据集的最终结果

数值为 6 个 Student seeds 的 `mean +/- std`；Accuracy 为 mean。所有分数单位为百分比。

### 4.1 arXiv

| 方法 | Final Macro-F1 | Final Accuracy | Old F1 | New F1 | 构造时间（秒） |
|---|---:|---:|---:|---:|---:|
| append-edge | 17.06 +/- 1.26 | 25.34 | 24.60 | 18.82 | 0.2 |
| Bounded X+P/XPA | 17.68 +/- 0.97 | 29.51 | 25.58 | 19.39 | 0.2* |
| random-real | 15.95 +/- 1.54 | 22.85 | 22.89 | 17.77 | 23.2 |
| k-center | 18.11 +/- 0.97 | 31.25 | 26.52 | 19.43 | 28.6 |
| medoid | 16.16 +/- 0.55 | 24.99 | 23.01 | 18.29 | 42.1 |
| CaT-MMD-TAG | **19.11 +/- 0.84** | **32.90** | **28.12** | **20.57** | 97.5 |
| GECC-hard-TAG | 17.53 +/- 1.30 | 27.69 | 25.77 | 18.96 | 222.5 |

### 4.2 BookChild

| 方法 | Final Macro-F1 | Final Accuracy | Old F1 | New F1 | 构造时间（秒） |
|---|---:|---:|---:|---:|---:|
| append-edge | 11.45 +/- 1.73 | 13.58 | 16.28 | 13.13 | 0.1 |
| Bounded X+P/XPA | 11.77 +/- 1.33 | 13.59 | 16.70 | 13.32 | 0.1* |
| random-real | 10.58 +/- 1.56 | 11.68 | 14.55 | 12.33 | 4.5 |
| k-center | 10.71 +/- 2.58 | 11.08 | 14.45 | 12.50 | 6.1 |
| medoid | 11.91 +/- 0.82 | 12.77 | 16.43 | **13.78** | 8.8 |
| CaT-MMD-TAG | 11.91 +/- 1.02 | 11.99 | 16.75 | 13.57 | 47.7 |
| GECC-hard-TAG | **13.29 +/- 1.26** | **14.60** | **18.37** | 15.35 | 93.9 |

### 4.3 SportsFit

| 方法 | Final Macro-F1 | Final Accuracy | Old F1 | New F1 | 构造时间（秒） |
|---|---:|---:|---:|---:|---:|
| append-edge | 23.19 +/- 4.63 | 31.70 | 30.38 | 27.07 | 0.2 |
| Bounded X+P/XPA | **27.41 +/- 1.91** | **37.18** | **36.23** | **31.37** | 0.2* |
| random-real | 25.25 +/- 4.71 | 31.13 | 31.85 | 29.04 | 15.7 |
| k-center | 25.73 +/- 3.51 | 29.65 | 33.03 | 29.42 | 22.6 |
| medoid | 24.13 +/- 5.57 | 27.80 | 31.20 | 28.58 | 40.2 |
| CaT-MMD-TAG | 21.13 +/- 3.56 | 24.20 | 26.39 | 24.59 | 229.2 |
| GECC-hard-TAG | 21.93 +/- 3.42 | 25.13 | 28.79 | 25.35 | 718.7 |

`*` 表示 sealed-source runner 的加载/审计耗时，不是 X+P/XPA 完整构造耗时。append-edge 同理。其余四类适配器的构造时间来自本轮运行，包括当前 delta 的选择、合成或聚类和文本/拓扑材料化。

## 5. 阶段平均和成本

三张大图等权平均：

| 方法 | Final F1 | Final Accuracy | 阶段平均累计 F1 | Old F1 | New F1 | 构造时间/轨迹（秒） |
|---|---:|---:|---:|---:|---:|---:|
| append-edge | 17.23 | 23.54 | 20.99 | 23.75 | 19.67 | 0.18* |
| Bounded X+P/XPA | **18.95** | **26.76** | 21.70 | **26.17** | **21.36** | 0.18* |
| random-real | 17.26 | 21.89 | 20.13 | 23.10 | 19.71 | 14.46 |
| k-center | 18.18 | 23.99 | **22.34** | 24.67 | 20.45 | 19.14 |
| medoid | 17.40 | 21.85 | 21.26 | 23.54 | 20.22 | 30.38 |
| CaT-MMD-TAG | 17.38 | 23.03 | 20.66 | 23.76 | 19.58 | 124.80 |
| GECC-hard-TAG | 17.59 | 22.47 | 21.29 | 24.31 | 19.89 | 345.04 |

所有 Student 使用相同的 6 seeds 和 steps 规则，平均 Student 训练时间约 25 秒/seed/阶段量级。CaT 和 GECC 的构造成本明显高于 coreset；SportsFit 上 GECC 的完整五阶段构造约 718.7 秒，而 X+P 的这里仅是 sealed-source 加载时间，不能做不对称成本结论。

### 5.1 五阶段 cumulative F1 轨迹

下面的每个数是该阶段 Student 训练后，在截至当前阶段全部已见类别上的 cumulative Macro-F1；不是只对当阶段新类计算的 F1。

| 数据集 | 阶段 | append-edge | X+P/XPA | random | k-center | medoid | CaT | GECC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arXiv | 0 | 29.62 | 29.62 | 25.77 | 26.02 | 30.09 | 32.71 | 28.65 |
| arXiv | 1 | 25.12 | 25.02 | 21.83 | 23.68 | 21.27 | 25.36 | 26.44 |
| arXiv | 2 | 20.64 | 20.01 | 19.30 | 21.01 | 19.45 | 21.96 | 22.49 |
| arXiv | 3 | 19.71 | 18.01 | 19.79 | 17.46 | 17.58 | 20.26 | 19.14 |
| arXiv | 4 | 17.06 | 17.68 | 15.95 | 18.11 | 16.16 | 19.11 | 17.53 |
| BookChild | 0 | 15.99 | 17.88 | 17.93 | 18.21 | 16.31 | 17.72 | 19.54 |
| BookChild | 1 | 15.62 | 17.17 | 13.38 | 15.19 | 15.47 | 15.04 | 18.09 |
| BookChild | 2 | 11.74 | 13.74 | 13.18 | 13.38 | 13.68 | 13.36 | 15.67 |
| BookChild | 3 | 12.42 | 12.40 | 13.01 | 11.81 | 12.56 | 11.86 | 14.22 |
| BookChild | 4 | 11.45 | 11.77 | 10.58 | 10.71 | 11.91 | 11.91 | 13.29 |
| SportsFit | 0 | 31.40 | 29.56 | 26.69 | 34.30 | 30.48 | 24.91 | 26.26 |
| SportsFit | 1 | 30.50 | 28.68 | 27.02 | 37.06 | 33.02 | 24.74 | 26.46 |
| SportsFit | 2 | 26.26 | 28.06 | 28.13 | 36.26 | 31.03 | 26.37 | 25.66 |
| SportsFit | 3 | 24.13 | 28.48 | 24.10 | 26.21 | 25.73 | 23.45 | 24.02 |
| SportsFit | 4 | 23.19 | 27.41 | 25.25 | 25.73 | 24.13 | 21.13 | 21.93 |

## 6. 结果如何解释

### 6.1 相对 append-edge：当前方法确实有效

X+P/XPA 在三张大图的最终 F1 均高于 append-edge：

- arXiv：`17.68 - 17.06 = +0.62 pp`；
- BookChild：`11.77 - 11.45 = +0.33 pp`；
- SportsFit：`27.41 - 23.19 = +4.22 pp`。

Accuracy 也在三张图上均不下降，三图平均提升约 `+3.22 pp`。这说明当前 X/P 联合构造不是只在某一个数据集的偶然收益，至少相对于普通持续追加控制存在可重复的最终阶段改善。

### 6.2 相对强 coreset：最终阶段有优势，但阶段轨迹不完全占优

k-center 是本轮最强的简单 coreset 参考之一：

- 最终 F1 三图平均为 `18.18`，X+P/XPA 为 `18.95`，X+P 高 `0.77 pp`；
- X+P 在 BookChild 和 SportsFit 的最终 F1 高于 k-center，在 arXiv 低 `0.43 pp`；
- 但 k-center 的阶段平均累计 F1 为 `22.34`，高于 X+P 的 `21.70`。

这说明当前方法的优势主要集中在最终累计状态和 Accuracy/New F1，不应写成“从每一个中间阶段都超过 k-center”。阶段平均更高的 k-center 仍然是必须保留的强基线。

### 6.3 相对 CaT/GECC：结果支持“有效”，不支持“普遍最优”

CaT-MMD-TAG 在 arXiv 最终 F1 达到 `19.11`，高于 X+P 的 `17.68`；GECC-hard-TAG 在 BookChild 达到 `13.29`，高于 X+P 的 `11.77`。但 SportsFit 上 X+P 明显超过 CaT `+6.27 pp`、超过 GECC `+5.47 pp`。

这两个基线是机制适配，不是官方端到端复现；它们还使用 edge-free memory，与 edge-aware X+P/coreset 的传播结构不同。因此正确的论文表述应是：

> 在统一 Student 和访问预算下，X+P/XPA 相对普通追加和简单 coreset 具有正向最终阶段收益，并在不同数据集上表现出更好的最终综合结果；与 CaT/GECC 的适配方法相比，优势具有数据集依赖性。

## 7. 目前已完成和未完成的公开基线

### 已完成

| 方法 | 状态 | 进入哪张表 |
|---|---|---|
| random-real | strict interface control | 主表 |
| k-center | strict interface control | 主表 |
| medoid | strict interface control | 主表 |
| CaT-MMD-TAG | mechanism-adapted | 主表，明确 adapted |
| GECC-hard-TAG | mechanism-adapted | 主表，明确 adapted |
| append-edge | internal continual control | 主表控制列 |
| Bounded X+P/XPA | sealed current candidate | 主表方法列 |

### 尚未完成

以下内容不能因为本轮七个 arms 完成就写成“全部公开基线已经完成”：

1. `TAGSAM-stage-local`：需要保留 TAGSAM 的文本选择和 attribute matching，再接入当前 delta + parent 接口；
2. `TAGSAM-global-oracle`：允许累计 raw 访问，只能放 offline/oracle 表；
3. `PUMA-inspired-TAG`：必须实现 pseudo-label、one-time propagation 和 edge-free memory，不能用 CaT 适配结果代替；
4. `STAD-native`：监督 GCN/静态 TAG 结果只能作为 native static reference，不能与 InfoNCE Student 直接排名；
5. 小图 Cora/Citeseer 的同一 `base+4`、同一 Student、同一七方法矩阵；
6. 三到五个独立类别顺序和 construction seeds 的稳定性确认。

## 8. 审计边界与限制

- 本轮每张大图只有一个冻结类别顺序和一个 construction seed，不能据此证明顺序不变性。
- Student 有 6 个 seeds，但不是 6 个独立 condensation constructions；表中的 `+/- std` 是 Student seed 波动。
- CaT/GECC 是机制保持适配，不是官方论文数字，也没有使用官方监督下游模型。
- 真实节点 coreset 的拓扑是统一控制构造的 text-kNN/cross-edge，不是直接把原始完整图诱导子图带入 Student。
- `append-edge` 和 X+P/XPA 是 sealed source；本 runner 的构造时间只代表加载/审计，不代表完整生成时间。
- 结果还没有包含 TAGSAM/PUMA/STAD，因此当前报告是“第一批严格接口基线”，不是最终论文主表的全部内容。

## 9. 证据和代码位置

本轮 runner：

- `code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py`
- `code/v2_llm_fusion/external_continual_mechanisms.py`

本地结果包：

- `tmp_remote_results/public_baselines_large_20261001_gpu0_r4/arxiv_base4/results.json`
- `tmp_remote_results/public_baselines_large_20261001_gpu0_r4/bookchild_base4/results.json`
- `tmp_remote_results/public_baselines_large_20261001_gpu1_r4/sportsfit_base4/results.json`
- `tmp_remote_results/public_baselines_large_20261001_gpu0_r4/summary.json`
- `tmp_remote_results/public_baselines_large_20261001_gpu1_r4/summary.json`

基线检索与适配矩阵：

- `docs/current/PUBLIC_BASELINE_SEARCH_AND_ADAPTATION_MATRIX_ZH_2026-10-01.md`

