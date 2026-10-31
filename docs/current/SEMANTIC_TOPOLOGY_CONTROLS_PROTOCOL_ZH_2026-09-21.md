# CTAG 语义邻居强对照：结果前冻结协议

本批只补拓扑强控制，不改内容、预算、训练模型，不加入 degree-aware capacity 或 soft closure。既有六臂实验已知，本批是后续机制开发而不是独立确认。

## 1. 固定输入与矩阵

- arXiv / BookChild；base+4；orders 49011/49012；student seeds 210--215；LP splits 49101/49102。
- 从已审计 `topology_causal_ablation_20260921_formal` 的各阶段 C0 memory 提取当前新 block，严格共享 X、text_emb、texts、mass、depth、stage、row order。
- 每阶段累计 5% 合法 pool 预算；每个新 block 内无向边 `min(n,n(n-1)/2)`，增量跨边每个新节点一条，均与已审计基线相同。
- 各新臂保留自己的旧行/旧旧边；跨边复用原 C0 的最近邻或原 T+degree 的 uniform-capacity 端点。两者仅依赖共享内容和阶段 row 数，与旧边无关。
- 复用原评价张量、NC/LP student 实现及参数，记录哈希。NC 2048 steps/batch16；LP 256 steps/edge batch256；均从头训练。
- 四新增臂 × 两图 × 两顺序 × 五阶段 × 六 seeds ×（NC + 两 LP splits）= **1440 次新训练**。原六臂 2160 fits 只作带来源的复用对照，不伪称本轮重跑。

## 2. 两个块内算子

### A. budgeted semantic 2-NN

以冻结合成文本向量的余弦为相似度。先保留对称化 1-NN 的边并集，保证每个节点有邻居；再从对称化 2-NN 的剩余边中按相似度降序补到 n 条无向边。相似度并列时按节点 ID 决定。没有独立调 k 或基于评价集选图。

这是**适配固定边预算的 2-NN**，不是原封不动的 vanilla kNN：对称化 2-NN 通常超过 n 边，而直接裁剪会打破节点覆盖。该臂不强制最大度为 2，不得称严格 degree-matched。

### B. semantic local cycle cover

对全部候选边按文本余弦降序贪心选取，两端度均小于 2 才能加入，**允许提前闭合局部环**。与 T+degree 的区别在于不要求最终所有节点处于同一环。

若最终剩一个孤立节点或一个二节点路径，删除一个已选环边并插入剩余节点/路径，选择语义总分损失最小的插入位置；并列按端点 ID。该补全不改变最终 n 边、每点度 2。记录补全次数；不宣称求得最优 2-factor。n<3 采用唯一可行简单图。

此臂用于在相同逐节点内部度与边数下检验“强制单环”的必要性。它是语义度约束控制，不包装成标准 kNN 或新主方法。

## 3. 四个新增方法臂

| arm | 新块内部 | 跨块 |
|---|---|---|
| knn_nearest | budgeted 2-NN | 最近旧邻居 |
| knn_capacity | budgeted 2-NN | uniform-capacity |
| cycles_nearest | cap-2 local cycle cover | 最近旧邻居 |
| cycles_capacity | cap-2 local cycle cover | uniform-capacity |

分别对照 `internal_only` 和 `t_degree`；C0、cross_only 和随机端点臂同时保留。对 cycles 检查与匹配跨边基线的**全部 memory 逐节点度**是否完全相同。

## 4. 事前解释和输出

报告最终、五阶段平均、old/new NC Macro-F1；LP AUC/AP 及关系切片；同 order/split/seed 配对差、标准差和正收益次数。报告分顺序而不只总体平均。不以共享 memory 的多 seed 重复冒充独立数据集数量，不作预设显著性结论。

- 若 kNN 接近或优于 T：不足以宣称复杂单环设计不可替代，优先考虑较简单构图。
- 若 local cycles 接近或优于 T：严格单环约束不是必要，后续闭合/连通性目标需要放宽。
- 若 T 相对两个常规语义控制仍稳定获益：才增加其算法特有价值的证据。
- 以正负结果一起决定下一步，不事后改主指标或选择最优顺序。该批不单独晋升 CTAG-final。

记录当前算子计算时间和 memory bytes。共享内容已缓存，**这些耗时仅是拓扑构造，不是端到端凝聚成本**；本批不能补成全流程 2× 成本门。另对 C0/单环/两控制做同输入、同硬件拓扑微基准，不能把重复诊断时间算作正式构造费用。

## 5. 审计与边界

新增代码和本协议在训练前哈希冻结，MiniLM 目录权重/配置哈希、原数据/cache/source/evaluation/memory 来源绑定。逐个 memory 重建审计，全部新 checkpoint 重载推理并用保存预测复算指标；保留数值告警，不修改旧记录。原批没有逐簇来源与完整构造计时的限制继续存在。

NC 仍为真实完整图 transductive report-dev；LP 构造不读取真实边，评价用真实 train edges 和分层均衡随机负边；构造用类别标签。没有新增 full-raw/coreset、hard negatives、时间流、独立新顺序或 bounded refresh 实验。
