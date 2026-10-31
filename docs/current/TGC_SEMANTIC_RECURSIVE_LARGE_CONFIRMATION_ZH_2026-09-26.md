# TGC Semantic Recursive 大图 `base+4` 确认报告

日期：2026-09-26  
状态：已完成 arXiv、BookChild、SportsFit 三个大图的五阶段实验、C0 配对重评和结构审计。

## 1. 本轮要回答的问题

小图上的递归 semantic conserved payload assignment 已经取得稳定的方向性收益。本轮把完全相同的机制扩展到已有的大图 `base+4` 协议，检验：

1. 收益是否只来自 Cora/Citeseer 的小图条件；
2. 递归 parent 是否能在五阶段类别增量链上继续工作；
3. semantic payload assignment 是否会改变拓扑、memory budget 或 Student 训练协议；
4. 收益主要来自旧类保持还是新类塑性。

本轮没有加入 selector、response gate、dynamic budget、GraphGPS、文本 attention、旧 memory refresh 或新的 Student。

## 2. 完整方法

当前候选为：

`frozen C0 topology + two-prototype structural proposal + semantic conserved payload assignment + recursive append + original Student`

每个阶段的流程如下：

1. **读取当前 delta**：只把当前阶段的 pool 节点作为构造输入；历史阶段的 raw graph 不参与后续 block 构造。
2. **重建 C0 source slots**：按照冻结 P5 `base+4` 的类级 rows，在当前类内用 graph-text joint space 聚类，得到当前新增 rows 的 source assignment。
3. **two-prototype structural proposal**：在每个新类中选择高离散度 C0 slot 做 split，并将最近的两个 slot merge，保持当前新 block 的 rows 不变；新 X 与文本来自同一个 source piece。
4. **semantic conserved payload assignment**：在同一类别内，对 structural proposal 的完整 payload 做一对一 semantic matching。被移动的是完整 `(X, text embedding, text, mass, depth, stage)` 单元，不重新平均、不复制、不删除 source coverage。
5. **递归 append**：stage 0 使用 C0；stage `t>0` 将 semantic stage `t-1` 作为 parent，只追加当前 semantic block。历史 synthetic prefix 保持 bitwise 不变。
6. **固定拓扑**：直接使用同 stage 的冻结 C0 `edge_index`。semantic assignment 不改变边、边数、度数或 old topology。
7. **Student 重训**：使用原始两层 GCN、可训练 text projection 和双向图文 InfoNCE；batch=16，Student seeds=100--105，steps=`protocol_v3.student_steps(rows)`，本轮大图均为 2048 steps。
8. **评估**：用 report-dev 真实图和类别 prompt 计算 cumulative、old-class、new-class Macro-F1。report/test 标签不参与构造或选择。

## 3. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集 | arXiv、BookChild、SportsFit |
| 类别协议 | `base+4`，共 5 stages |
| split | P5 冻结的 `base_incremental`，`base_fraction=0.4` |
| memory budget | fixed-rate 5% |
| Student | 原始两层 GCN + trainable text projection |
| Student objective | bidirectional graph-text InfoNCE |
| Student batch | 16 |
| Student seeds | 100, 101, 102, 103, 104, 105 |
| Student steps | `max(50, min(2048, 16 * roots))`，大图为 2048 |
| C0 arm | P5 已生成的 `append_fixed_s0..s4_memory.pt`，本轮重新评估 |
| candidate arm | recursive semantic |
| construction labels | 只用于类别增量 stream 定义，不用于候选选择 |
| report/test labels | 不用于 construction 或 selection |
| historical raw reads | 0（构造阶段） |

各数据集的 budget 为：

| 数据集 | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|---:|
| arXiv | 4962 | 5246 | 5502 | 5734 | 5926 |
| BookChild | 2344 | 2458 | 2551 | 2629 | 2690 |
| SportsFit | 4491 | 4965 | 5430 | 5802 | 6056 |

## 4. 逐阶段结果

下表为本轮同一进程、同一 Student seeds 下的 paired comparison。单位为 Macro-F1 百分点；`old` 和 `new` 是 semantic 相对 C0 的分解。

### 4.1 arXiv

| stage | C0 cumulative | semantic cumulative | delta | old delta | new delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 29.618 | 29.618 | +0.000 | -- | +0.000 |
| 1 | 26.519 | 26.098 | -0.421 | -1.395 | +4.019 |
| 2 | 22.153 | 23.616 | +1.463 | -0.929 | +5.247 |
| 3 | 21.610 | 22.706 | +1.095 | +1.390 | +0.677 |
| 4 | 18.817 | 19.227 | +0.409 | +0.304 | +1.940 |

增量 stage 1--4 平均 cumulative `+0.637 pp`，old `-0.157 pp`，new `+2.971 pp`；3/4 个增量阶段为正，最差 `-0.421 pp`。

### 4.2 BookChild

| stage | C0 cumulative | semantic cumulative | delta | old delta | new delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 17.093 | 17.093 | +0.000 | -- | +0.000 |
| 1 | 16.960 | 17.728 | +0.768 | +1.426 | +3.132 |
| 2 | 14.746 | 13.195 | -1.552 | -4.100 | -3.272 |
| 3 | 14.389 | 15.281 | +0.892 | +0.782 | +1.468 |
| 4 | 12.607 | 13.313 | +0.706 | +0.446 | +7.635 |

增量 stage 1--4 平均 cumulative `+0.203 pp`，old `-0.361 pp`，new `+2.241 pp`；3/4 个增量阶段为正，最差 `-1.552 pp`。

### 4.3 SportsFit

| stage | C0 cumulative | semantic cumulative | delta | old delta | new delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 26.482 | 26.482 | +0.000 | -- | +0.000 |
| 1 | 26.758 | 32.981 | +6.223 | +7.675 | +1.175 |
| 2 | 31.144 | 28.763 | -2.381 | -0.942 | -4.126 |
| 3 | 27.349 | 27.384 | +0.036 | -0.251 | +0.611 |
| 4 | 25.013 | 28.076 | +3.063 | +2.152 | +2.658 |

增量 stage 1--4 平均 cumulative `+1.735 pp`，old `+2.158 pp`，new `+0.080 pp`；3/4 个增量阶段为正，最差 `-2.381 pp`。

## 5. 汇总判断

| 数据集 | 增量 cell 数 | 正 cell | cumulative 平均 | old 平均 | new 平均 | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|
| arXiv | 4 | 3/4 | +0.637 | -0.157 | +2.971 | -0.421 |
| BookChild | 4 | 3/4 | +0.203 | -0.361 | +2.241 | -1.552 |
| SportsFit | 4 | 3/4 | +1.735 | +2.158 | +0.080 | -2.381 |
| 合计 | 12 | 9/12 | **+0.858** | **+0.547** | **+1.764** | **-2.381** |

stage 0 不存在方法差异，因此汇总表只对增量 stages 统计。若把 stage 0 的零差值也纳入五阶段平均，三组分别为 arXiv `+0.509 pp`、BookChild `+0.163 pp`、SportsFit `+1.388 pp`。

## 6. 机制与审计结果

15 个 stage cells 全部通过结构契约：

- current C0 topology 与 semantic topology 完全相同；
- semantic parent prefix 与上一阶段 semantic memory 完全相同；
- structural payload 经 semantic permutation 后 multiset 完整守恒；
- construction 阶段 historical raw reads=0；
- report/test 标签未用于构造或选择；
- rows、budget、batch、steps、Student seeds 与 C0 成对一致。

每个数据集的 semantic payload move 数量如下，说明 semantic assignment 实际改变了 block 中的一部分角色，而不是只复制 C0：

| 数据集 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|
| arXiv | 11 | 13 | 17 | 15 |
| BookChild | 5 | 10 | 8 | 8 |
| SportsFit | 6 | 5 | 5 | 5 |

## 7. 结果说明什么

1. **大图上存在可复现的正向机制信号。** 三个数据集均有 3/4 增量阶段为正，合计 12 个 cell 平均提升 `+0.858 pp`；这不是只在 Cora/Citeseer 上观察到的现象。
2. **收益主要与 new-class plasticity 有关。** arXiv 和 BookChild 的 new-class 平均分别为 `+2.971`、`+2.241 pp`；SportsFit 的收益更多来自 old-class retention，new-class 近似持平。
3. **递归 parent 是必要的协议组成。** 本轮不是每个阶段从 C0 重新开始，而是把 semantic stage `t-1` 作为 stage `t` 的 parent；这与小图中递归版本改善尾部失败的观察一致。
4. **方法不是跨数据集、跨阶段稳定优于 C0。** BookChild stage2 和 SportsFit stage2 为明显负值，最差 `-2.381 pp`。因此当前证据支持“平均有效、机制可解释”，不支持“统一稳定提升”或 SOTA。
5. **不能把提升归因于更多 rows 或训练成本。** 两个 arm 使用完全相同的 rows、edges、batch、steps 和 seeds；semantic arm 只重排/替换当前新增 block 的 payload。

## 8. 与历史 P5 数值的可比性说明

本轮把 P5 的 `append_fixed` memory 重新加载，并在当前代码中重新跑 C0 Student 作为 paired control。arXiv 的 C0 指标与历史 P5 结果完全一致；BookChild/SportsFit 的部分历史 P5 C0 指标与本轮复跑存在差异。因此：

- 本报告所有 delta 只使用本轮同进程的 C0 与 semantic 配对结果；
- 不把历史 P5 的 C0 数值与本轮 semantic 数值直接拼接；
- 历史 P5 表仍可作为旧运行记录，但不能用于本轮差值计算。

这一处理避免了不同代码状态、运行顺序或数值非确定性带来的跨运行比较偏差。

## 9. 当前结论与论文边界

本轮可以支持如下表述：

> 在三个大图的统一 `base+4` continual protocol 下，recursive semantic conserved payload assignment 相对同一冻结 C0 配对控制，在 12 个增量 stage cells 上平均提升 0.858 个百分点，9/12 个 cell 为正，新类 Macro-F1 平均提升 1.764 个百分点。该收益不改变 memory budget、拓扑、Student 或训练步数，但仍具有数据集和阶段依赖，最差 cell 为 -2.381 个百分点。

当前不能支持：

- 宣称在所有数据集上稳定优于 C0；
- 宣称优于 k-center、medoid、静态 TAGSAM 或全历史重凝聚；
- 宣称 SOTA；
- 把本轮 semantic assignment 单独解释为唯一因果来源，因为 two-prototype proposal 与 semantic assignment 是当前候选的联合构造。

## 10. 下一步

1. 把本轮 recursive semantic 与大图 `k-center/medoid`、完整数据训练、全历史重凝聚放入同一主表，统一使用本轮 paired Student 评估口径。
2. 在 Cora/Citeseer 保留 stage-local 与 recursive 两种结果，作为“递归 parent 必要性”消融；不把 stage-local 的较好 cell 混入主方法。
3. 对三个大图补充 construction time、Student training time、memory rows、old/new F1 和 worst-stage 曲线。
4. 暂不加入 response selector、动态预算、旧 memory refresh、GraphGPS 或文本 attention；先确认强 baseline 对比后再决定是否进入 v5/v6 主线。

## 11. 证据文件

- 运行器：[run_tgc_semantic_recursive_large_20260926.py](../../code/v2_llm_fusion/run_tgc_semantic_recursive_large_20260926.py)
- 结果根目录：[tgc_semantic_recursive_large_20260926](../../results/tgc_semantic_recursive_large_20260926)
- arXiv 结果：[results.json](../../results/tgc_semantic_recursive_large_20260926/arxiv_base4/results.json)
- BookChild 结果：[results.json](../../results/tgc_semantic_recursive_large_20260926/bookchild_base4/results.json)
- SportsFit 结果：[results.json](../../results/tgc_semantic_recursive_large_20260926/sportsfit_base4/results.json)
- 每个 stage 的 `stage_*_audit.json` 位于对应数据集目录。
