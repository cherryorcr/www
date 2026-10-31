# CTAG 拓扑因果拆分协议

日期：2026-09-21  
状态：结果产生前冻结。该协议只验证拓扑机制，不加入动态预算、多原型 X、文本 attention、response selector 或 memory refresh。

## 1. 研究问题

当前 T+degree 同时改变了块内连边和跨块容量，不能从已有结果判断收益来源。本实验固定合成节点内容，只拆分两个拓扑部件：

1. 新 block 内的 cap-2 语义路径/闭环；
2. 新节点连接旧 memory 时的接收容量均衡。

同时加入保持拓扑度序列的随机控制，判断收益是否只是来自连通性和度分布。

## 2. 冻结协议

| 项目 | 设置 |
|---|---|
| 数据集 | arXiv、BookChild |
| 阶段 | base+4 五阶段；沿用既有已审计阶段规模 |
| 类别顺序 | 49011、49012；全部类别打乱后按 base+4 分块 |
| memory | `floor(0.05 * cumulative legal pool)` |
| 构造 seed | 9001 |
| student seeds | 210--215 |
| LP split seeds | 49101、49102 |
| NC student | 两层 GCN 384/384，双向图文 InfoNCE，2048 steps，batch 16 |
| LP student | GAE GCN 384/128，内积解码，256 steps |
| 节点内容 | 所有 arm 共享同一 stage block 的 X、text、mass、depth、stage 和 row order |
| 总边数 | 所有 arm 每阶段使用同一无向 pair 数 |
| 旧数据访问 | 只访问当前合法 pool 和上一阶段 synthetic memory |
| 旧行/旧旧边 | 本轮沿用 append-only 作为实验条件，不将其写成问题的强制约束 |

## 3. 方法臂

| arm | 块内边 | 跨块边 | 目的 |
|---|---|---|---|
| `c0` | C0 原有文本高相似度边 | C0 最近旧邻居 | 冻结共同控制 |
| `cross_only` | C0 块内边 | T+degree 容量均衡 | 只测跨块负载 |
| `internal_only` | T+degree cap-2 语义路径闭环 | C0 最近旧邻居 | 只测块内连通性/度上限 |
| `t_degree` | T+degree cap-2 语义路径闭环 | T+degree 容量均衡 | 当前完整候选 |
| `random_ring` | 随机重标记的 cap-2 环 | T+degree 容量均衡 | 保留度和连通性，去除块内语义端点匹配 |
| `degree_shuffle` | T+degree 拓扑的随机新节点重标记 | 同步重标记跨块端点 | 保留完整度序列，去除内容-拓扑对应关系 |

`random_ring` 和 `degree_shuffle` 不增加 rows 或 edges。它们只对当前新 block 的节点 ID 做确定性置换，旧旧边保持不变。所有随机控制使用 `construction_seed=9001` 派生的固定种子，并写入 manifest。

## 4. 主要指标

任务指标：

- NC：最终和阶段平均 all-class Macro-F1，old/new F1；
- LP：最终和阶段平均 AUC/AP，old/new/cross/within-new；
- student seed 均值、标准差和配对差。

拓扑机制指标：

- isolated rows、最大度、平均度、degree Gini；
- 新 block 连通分量和跨块 pair 数；
- 旧节点新增接收负载及其最大值；
- 旧块归一化传播变化：旧子块 row-L1 漂移、含跨边的 full row-L1 漂移、跨边归一化权重。

成本指标：

- 构造时间；
- NC/LP student 训练时间；
- memory bytes。

## 5. 事前解释规则

1. `cross_only - c0` 为正而 `internal_only - c0` 接近零：主要收益来自旧节点接收负载均衡。
2. `internal_only - c0` 为正而 `cross_only - c0` 接近零：主要收益来自新 block 连通性和度上限。
3. `t_degree` 明显优于两个单项：两个部件存在交互，才可将联合拓扑追加作为主算法候选。
4. `random_ring` 接近 `t_degree`：收益主要来自连通性/度分布，语义端点选择不是必要因素。
5. `degree_shuffle` 接近 `t_degree`：收益主要来自拓扑度序列，内容-拓扑匹配不是必要因素。
6. 若随机控制全部复现收益，则不宣称语义拓扑算法新颖性，应转为拓扑机制分析。

## 6. 晋升门

T+degree 只有同时满足以下条件，才进入 CTAG-1 主方法：

- 在至少两个数据集或一个数据集加一个独立协议上，完整候选相对 C0 有稳定正收益；
- 相对至少一个度匹配随机控制仍有额外收益，或明确证明拓扑统计量是关键机制；
- 不显著损害另一任务的 old/new 保持；
- 构造成本不超过 C0 的约 2 倍，student 训练成本基本不增加；
- arXiv 的退化可以由传播漂移、低质量闭环或顺序差异解释，而不是被忽略。

本协议不能证明 T+degree 超过 full-raw、k-center、medoid 或静态 TAG 凝聚方法；这些属于下一阶段的性能定位实验。

## 7. 结果后执行核对（2026-09-21 补记）

本节是结果后的执行说明，不改写上述事前设置或晋升门：

- 六臂正式矩阵已完成 2,160 fits，见 [结果与验收报告](TOPOLOGY_CAUSAL_ABLATION_RESULTS_ZH_2026-09-21.md)。
- 本轮沿用上一轮 degree confirmation 的 49011/49012、210--215、49101/49102，因此是机制扩展，不是独立新顺序确认。
- runner 未记录完整构造耗时；本协议成本项仍有缺口，不能宣称已通过 2× 构造成本门。
- NC 使用真实完整图 transductive 传播、report-dev 指标；LP 为分层随机均衡负边。不能称严格 inductive test 或 hard-negative 验证。
- 大路线要求的 semantic kNN/medoid 强拓扑控制未纳入本轮六臂，仍待完成。
- 原 manifest 未存协议哈希及 MiniLM 路径/权重哈希；保留这一追溯限制，不把本补记伪装为事前预注册。
