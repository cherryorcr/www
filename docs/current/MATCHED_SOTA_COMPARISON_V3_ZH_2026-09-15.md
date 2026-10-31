# v3 严格同协议对比与外部 SOTA 审计

日期：2026-09-15

## 1. 对比口径

当前项目的目标不是普通监督式图凝聚，而是同时满足以下条件的 TAG 持续凝聚：

- 凝聚阶段不读取类别标签；
- 节点文本保留并由冻结 MiniLM 编码；
- student 只在 synthetic memory 上从头训练；
- 评测使用类别 prompt 的 zero-shot 图文匹配；
- 数据以类别增量方式到达；
- 旧阶段 raw 数据在边界后不可重读；
- 每个阶段使用累计预算 5%、batch=16、student seeds=100..105、
  `max(50, min(2048, 16*roots))` steps；
- dev=`original_id%10==1`，test=`%10==2` 且 test 不参与构造、选择和调参。

只要一个外部方法改变了监督目标、student、评测任务、数据流、预算或历史访问权限，
就不能把它的数字直接排进 v3 主表。

## 2. 外部方法资格矩阵

| 方法 | 主要任务 | 与 v3 的关系 | 处理方式 |
|---|---|---|---|
| TAGSAM | Text-Attributed Graph condensation | TAG 领域最直接的静态参考，但公开设定不是本项目的类别增量、zero-shot、不可回读协议 | external reference |
| GCond | 监督式 gradient matching graph condensation | 读取标签并优化监督 GNN 训练轨迹；没有文本 prompt 目标 | 不进入同协议主表 |
| SFGC | 大规模 structure-free graph condensation | 主要面向监督节点分类和无图训练数据；目标与 v3 不同 | 不进入同协议主表 |
| PUMA | continual graph learning with graph condensation | 是最接近的持续凝聚工作，但使用伪标签/监督式持续学习，数据和 student 口径不同 | closest continual external |
| GECC | continual graph condensation for evolving graphs | 使用 class-wise clustering、聚合特征和历史 centroid 继承；没有 v3 的 TAG 文本 prompt、zero-shot 和 frozen-old append 约束 | closest continual external |
| CCC | dynamic graph continual learning with condensation-concatenation | 面向动态图快照与模型持续学习，强调历史结构和标签分布；不是 v3 的文本图 zero-shot student | closest continual external |
| CA-DGCL | dynamic graph continual learning via condensation and attachment | 面向动态快照，使用跨时间 node chain、Tucker 表示和 attachment；任务/表示/评测均不同 | closest continual external |
| OpenGC | open-world graph condensation | 研究结构感知分布漂移下的开放世界凝聚，不等同于类别增量 frozen-memory protocol | adjacent external |
| CaT | balanced continual graph learning with graph condensation | 监督式 continual node classification，使用 labelled incoming graph 和 model replay；不满足 label-free zero-shot | closest continual external |
| GCondenser | graph condensation benchmark/toolkit | 是方法评测基础设施，不是一个可直接替换 v3 的 TAG continual 方法 | benchmark reference |

论文与官方资料：

- [TAGSAM, arXiv:2606.03839](https://arxiv.org/abs/2606.03839)
- [GCond, arXiv:2110.07580](https://arxiv.org/abs/2110.07580)
- [SFGC, arXiv:2306.02664](https://arxiv.org/abs/2306.02664)
- [PUMA, arXiv:2312.14439](https://arxiv.org/abs/2312.14439)
- [CaT, arXiv:2309.09455](https://arxiv.org/abs/2309.09455)
- [OpenGC, arXiv:2405.17003](https://arxiv.org/abs/2405.17003)
- [GECC, arXiv:2502.17614](https://arxiv.org/abs/2502.17614)
- [CCC, arXiv:2512.11317](https://arxiv.org/abs/2512.11317)
- [CA-DGCL, arXiv:2607.11112](https://arxiv.org/abs/2607.11112)
- [GCondenser, arXiv:2405.14246](https://arxiv.org/abs/2405.14246)

本次检索日期为 2026-09-15，检索关键词覆盖 `continual graph condensation`、`dynamic graph continual learning condensation`
和 `text-attributed graph continual condensation`。这里的“最新”指截至该日期可公开检索到的相关预印本；没有把尚未完成
同协议重实现的外部数值伪造为 v3 结果。

## 3. 严格同协议主表的组成

在没有外部方法同时满足上述七项条件前，v3 主表只纳入可以完全重跑的臂：

1. `random_real`、`k_center`、`herding`、`kmeans_medoid`：同预算、同 student 的 label-free real-node baseline；
2. `append_fixed`：旧 memory 保持、当前 delta 单独凝聚；
3. `recluster`：允许旧 synthetic 与当前 delta 混合重聚类，作为压力测试；
4. `full_history`：重新读取历史 raw 的违规上界参照；
5. `SCA`：response-curve selection + topology-preserving append；
6. `raw`：只作为不可压缩的上界参照，不与 synthetic 方法按成本排序。

这些臂都由统一 runner 输出同一个 manifest，包含数据集、protocol、预算、seed、batch、steps、
stage split、test 封存状态和 stagewise F1；每个 stage 的 artifact 另外保存 synthetic rows、mass、
文本和 `edge_index`，因此 rows/edges 可以由原始结果文件复核。当前 manifest 尚未纳入独立的运行时间和
显存峰值，论文表中不报告未经记录的效率数字。

三套大图的 matched stage-2 累计 Macro-F1（%）如下；所有行均为同一 v3 student、split、预算和
六个 student seeds：

| 数据集 | random_real | k_center | herding | kmeans_medoid | append_fixed | SCA |
|---|---:|---:|---:|---:|---:|---:|
| arXiv | 17.54 | 19.68 | 18.52 | 18.66 | 20.43 | 20.03 |
| BookChild | 13.43 | 12.03 | 14.58 | 13.61 | 13.97 | 15.04 |
| SportsFit | 29.36 | 30.03 | 27.06 | 29.07 | 25.90 | 26.22 |

这个 matched 表显示：SCA 在 arXiv 和 BookChild 的最终累计指标高于四个 real-node 构造基线，
但在 SportsFit 明显低于 `k_center`。因此 k-center 不能被简单归结为实现错误，SCA 也不能被写成
跨数据集的普遍最优。

## 4. 当前可下的结论

- 目前没有发现一个已公开、同时满足“TAG + continual class-incremental + label-free condensation + zero-shot prompt evaluation + frozen old-history protocol”的外部方法，因此不能诚实地写成“击败 continual TAG SOTA”。
- PUMA、GECC、CCC 和 CA-DGCL 可以作为最近的持续图凝聚外部参照，但不能与 v3 的 zero-shot Macro-F1 直接横比；它们分别在监督信息、动态图快照、类别聚类、历史记忆或模型训练方式上改变了任务。
- TAGSAM 是最相关的 TAG 静态方法，应在 related work 和 external reference 中对比机制、压缩率、构造成本；若其训练目标或评测 backbone 不同，数值不能直接并入 v3 主表。
- GCond、SFGC 等可用于说明监督式凝聚的研究位置和方法谱系，但不能被包装成 v3 的同协议 baseline。

## 5. 交付状态

- 已完成：Cora、Citeseer 的六 seed 静态与 continual 统一 runner 复核；WikiCS、BookHis 的核心 continual 复核；
  小图 SCA、topology、text operator、dynamic rows、bounded refresh 消融。
- 已完成：arXiv、BookChild、SportsFit 的六 seed canonical continual 主表，包含 `append_fixed`、SCA，及其
  `recluster/full_history` 诊断臂；结果已复制到本地 `results/` 并汇总到七数据集表。
- 已完成：外部方法的严格协议资格审计，并在论文表格中区分 `matched`、`closest external` 和 `upper reference`。
- 尚未完成且不应伪装成已完成：对 PUMA/TAGSAM 等外部方法的完整重实现重跑。由于任务定义不匹配，当前结论是
  “没有可诚实直接横比的公开同协议 continual TAG SOTA”，而不是“已经击败这些方法”。

所有大图 matched baseline 的原始 JSON 已保存于：`results/baseline_v3_2026_09_15_arxiv/`、
`results/baseline_v3_2026_09_15_bookchild/`、`results/baseline_v3_2026_09_15_sportsfit/`。

## 6. 论文写法

主线应写为：**student-response calibrated selection + topology-preserving append**，并同时报告
selection seed 不稳定、frozen memory 过时风险和 full-history/recluster 的负结果。
不使用 “SOTA” 作为结论；更准确的表述是：
“在尚缺乏同设定公开竞争方法的 TAG continual zero-shot 任务中，建立了可审计的 v3 基准，
并系统分析了响应选择、历史拓扑和记忆管理的作用边界。”

## 7. 已完成的多数据集结论

在当前七数据集集合上，SCA 的最终 stage-2 累计 F1 相对 `append_fixed` 为：Cora `0.00`、Citeseer `0.00`、
arXiv `-0.40`、BookChild `+1.07`、SportsFit `+0.31`；WikiCS 和 BookHis 目前没有同一完整 SCA
canonical 结果，不做补推。这个结果支持“response mechanism 可解释、但收益依赖数据集和路径”的结论，
不支持普遍性能优势或 SOTA 声称。

因此论文主表建议分为：

1. matched v3 continual：本项目统一 runner 下的 append、recluster、full-history、SCA 与可获得的 real-node baselines；
2. closest external：PUMA、GECC、CCC、CA-DGCL、CaT、TAGSAM 等，报告任务、监督信息、student、split 和预算差异；
3. upper/reference：raw 或 native/original implementation，只用于定位压缩代价，不参与同协议排序。
