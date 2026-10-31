# CTAG 第一批：拓扑因果拆分结果与后续决策

日期：2026-09-21。对应 [冻结协议](TOPOLOGY_CAUSAL_ABLATION_PROTOCOL_ZH_2026-09-21.md) 与 [完整算法工作路线](NEXT_RESEARCH_WORKPLAN_T_DEGREE_ZH_2026-09-21.md)。

## 1. 本批完成了什么

**完成六臂、两大图、五阶段的 2,160 次正式训练及结果汇总；这不是整个 CTAG 路线完成，也不是 CTAG-final 定稿。**

| 范围 | arXiv | BookChild | 合计 |
|---|---:|---:|---:|
| 合成 memory | 60 | 60 | 120 |
| 节点分类 NC fits | 360 | 360 | 720 |
| 链路预测 LP fits | 720 | 720 | 1,440 |
| 总 fits | 1,080 | 1,080 | 2,160 |

两图全部 checkpoint 已重载推理、120 份 memory 已审计。BookChild 无超容差告警；arXiv 有 1 条已定位的初始阶段随机环 NC 舍入告警，保留原标记，见第 10 节。只统计文件数量的旧 `summary_status.json` 不等于模型重载验收。

**重要口径修正：** 本轮的类别顺序 49011/49012、student seeds 210--215、LP splits 49101/49102 已用于上一轮 [跨任务确认](DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md)。本轮是扩展机制消融，不是另一批独立顺序/seed 的泛化确认。所有结论使用本轮共同重跑的 C0，不与旧 NC 分数拼表。

核心结论：

1. BookChild 上，语义块内构图贡献明确，完整 T+degree 的最终两任务指标进一步改善。
2. 单独改跨块容量没有稳定收益；当前均匀容量规则不是已经成立的通用创新。
3. arXiv 上，T+degree 的分类均值略降，LP 的顺序效应一正一负；不能用总体均值掩盖。
4. 随机拓扑不能复现 BookChild 最终 LP 的全部增益，说明内容与边端点的对应关系值得继续研究。
5. 保留 T+degree 作为研究基础、C0 作为共同对照；尚不能无条件晋升最终算法。

## 2. 共同设置与访问边界

| 项目 | 设置 |
|---|---|
| 数据与阶段 | arXiv：40 类，16+6+6+6+6；BookChild：24 类，10+4+4+3+3 |
| 类别顺序 | 49011、49012；按既定阶段类别数重排 |
| 节点预算 | 累计合法 pool 的 5%，向下取整；不是完整图节点数的 5% |
| 最终合成节点 | arXiv 5,926；BookChild 2,690 |
| 内容构造 | dispersion 类别配额、joint X/text 聚类、mean-X、中心文本 concat/re-encode，`joint_g5_c320` |
| 构造随机性 | 固定 construction seed 9001；本轮未增加构造 seed 重复 |
| NC student | 两层 GCN，384/384，双向 InfoNCE；2,048 steps，batch 16，lr 0.001 |
| LP student | GAE，GCN 384/128，内积解码；256 steps，正负边 batch 各 256，lr 0.001 |
| student 重训 | 每阶段每个 seed 从头训练，不继承上一阶段参数 |
| 共享输入 | 各臂同 X、文本表示、文本字符串、mass、depth、birth stage、row order、rows 和无向边总数 |
| memory 管理 | 各臂保留它自己上一阶段的旧行和旧旧边，不要求各臂的旧拓扑相同 |

NC 是已见节点的 Macro-F1：主表允许全部类别 prompt 参与竞争，再对已见类别取宏平均；另存 seen-prompt 结果。最后阶段两者一致。NC 在真实完整图上传播，是 **transductive、report-dev 评价**，不是隐藏未来图结构的严格 inductive test。

LP 先在合法节点的真实边上按到达阶段分层划分约 80/10/10；训练 memory 不读取真实边，评价传播使用真实 train edges，held-out 正边不进入评价传播图。负样本是按阶段关系分层、数量均衡的随机非边，不是 hard negatives。两任务用同一组合成 memory，但各自训练不同 student。

构造使用类别标签分配和分簇。因此只能称“在两个下游任务上的复用验证”，不能称无监督或已证明任务无关。旧行/旧边不变是本批的控制条件，不是用户问题的强制约束。

## 3. 六个方法到底改了哪里

| arm | 新块内部边 | 新旧跨块边 | 要回答的问题 |
|---|---|---|---|
| C0 | 全局选高文本相似度边，无覆盖/度上限 | 每个新节点连接最近旧节点 | 共同构造对照 |
| cross_only | 同 C0 | 旧端点本阶段接收容量均衡 | 只限制跨块接收量是否有效 |
| internal_only | 文本相似度引导的 cap-2 路径，最后闭环 | 同 C0 | 改善新块覆盖、度分布与连通性是否有效 |
| T+degree | 同 internal_only | 同 cross_only | 两组件组合是否更强 |
| random_ring | 对语义环的新节点端点做固定随机置换 | 保留语义 capacity 跨边 | 相同环结构是否还需要块内语义匹配 |
| degree_shuffle | 同一个置换同时作用于新块内部和跨边的新端点 | 随上述置换 | 去掉更多内容-拓扑对应关系是否损失性能 |

当前 capacity 是 `ceil(new_rows / old_rows)` 的均匀上限，依次为每个新节点选可接收的最相似旧节点。**它不是根据旧节点当前度、mass 或传播漂移计算的 degree-aware capacity。** 本轮所有增量阶段 new_rows < old_rows，实际接收上限为 1。

两随机臂只有一个确定性置换 seed，不是所有随机图的期望。审计检查的是每节点度相同，而不只是排序后的度分布相同。2×2 对照比较完整五阶段策略轨迹；不是固定同一个历史图后只干预最后一步。

## 4. 最终阶段主结果

单位均为百分数，差值用百分点 pp。NC 每个方法为 2 orders × 6 seeds；LP 为 2 orders × 2 splits × 6 seeds。以下是等权均值，不把共享 memory 的重复 seed 当成独立数据集。

| 数据集 | 方法 | NC Macro-F1 | LP AUC | LP AP |
|---|---|---:|---:|---:|
| arXiv | C0 | 20.759 | 96.044 | 95.763 |
| arXiv | cross_only | 20.003 | 96.060 | 95.779 |
| arXiv | internal_only | 20.464 | 96.217 | 95.930 |
| arXiv | T+degree | 20.439 | 96.031 | 95.711 |
| arXiv | random_ring | 21.718 | 95.920 | 95.660 |
| arXiv | degree_shuffle | 21.458 | 95.587 | 95.256 |
| BookChild | C0 | 13.050 | 92.205 | 91.366 |
| BookChild | cross_only | 13.305 | 92.125 | 91.220 |
| BookChild | internal_only | 14.038 | 92.802 | 92.245 |
| BookChild | T+degree | **14.600** | **92.927** | **92.345** |
| BookChild | random_ring | 14.060 | 91.831 | 91.125 |
| BookChild | degree_shuffle | 13.703 | 91.353 | 90.481 |

### 4.1 配对增益和波动

以下标准差是逐个同 order/split/seed 配对差的标准差，不是标准误、置信区间或显著性检验。

| 数据集 | T+degree − C0：NC | AUC | AP | 正收益次数 NC / AUC / AP |
|---|---:|---:|---:|---|
| arXiv | -0.319 ± 1.486 | -0.013 ± 0.482 | -0.052 ± 0.604 | 6/12；9/24；9/24 |
| BookChild | +1.549 ± 1.144 | +0.722 ± 0.561 | +0.979 ± 0.651 | 12/12；21/24；23/24 |

BookChild 的优势并非仅由一个 seed 产生。但当前只有两个既有顺序，不能由上述计数推出跨数据集普遍有效。

| 数据集 / 顺序 | NC 差值 | AUC 差值 | AP 差值 |
|---|---:|---:|---:|
| arXiv / 49011 | -0.233 | -0.329 | -0.457 |
| arXiv / 49012 | -0.406 | +0.304 | +0.353 |
| BookChild / 49011 | +1.318 | +0.854 | +1.187 |
| BookChild / 49012 | +1.780 | +0.590 | +0.771 |

### 4.2 块内语义与容量的增量贡献

| 数据集 | 对比 | NC 差值 | AUC 差值 | AP 差值 |
|---|---|---:|---:|---:|
| arXiv | internal_only − C0 | -0.295 | +0.173 | +0.168 |
| arXiv | T+degree − internal_only | -0.024 | -0.185 | -0.220 |
| arXiv | T+degree − random_ring | -1.278 | +0.112 | +0.050 |
| arXiv | T+degree − degree_shuffle | -1.018 | +0.444 | +0.455 |
| BookChild | internal_only − C0 | +0.987 | +0.597 | +0.879 |
| BookChild | T+degree − internal_only | +0.562 | +0.125 | +0.100 |
| BookChild | T+degree − random_ring | +0.540 | +1.097 | +1.220 |
| BookChild | T+degree − degree_shuffle | +0.897 | +1.574 | +1.864 |

BookChild 的主要 LP 改善，单用 internal_only 已经能取得。完整 T 的 capacity 额外收益较小，不应把全部收益归给跨块负载控制。相同度序列的随机控制仍损失 LP，支持内容与拓扑对应关系有用，但尚不能证明当前贪心语义环优于普通 semantic kNN。

交互项定义为 `T − cross_only − internal_only + C0`。arXiv 的 NC/AUC/AP 为 +0.731/-0.202/-0.236 pp；BookChild 为 +0.307/+0.205/+0.246 pp。交互项没有跨数据集、跨任务一致为正，不能声称两个部件具有普遍协同作用。

## 5. 不只看最后一阶段

阶段平均先在每条 matched trajectory 内对五阶段等权平均，再跨重复汇总。

| 数据集 | 方法 | 阶段平均 NC | 阶段平均 AUC | 阶段平均 AP |
|---|---|---:|---:|---:|
| arXiv | C0 | 21.670 | 96.488 | 96.279 |
| arXiv | internal_only | 21.320 | 96.608 | 96.378 |
| arXiv | T+degree | 21.553 | 96.499 | 96.245 |
| arXiv | random_ring | 21.135 | 96.354 | 96.140 |
| BookChild | C0 | 16.189 | 90.171 | 88.789 |
| BookChild | internal_only | 17.233 | 91.898 | 91.150 |
| BookChild | T+degree | 16.844 | 91.773 | 90.994 |
| BookChild | random_ring | 17.433 | 91.479 | 90.648 |

BookChild 的完整 T 最终指标最好，但 internal_only 的三个阶段平均都更高；随机环的分类阶段平均还更高。**“最终最好”不等于“整个持续过程最好”。**

最终分类的旧类/新类分解：

| 数据集 | 方法 | 旧类 F1 | 新类 F1 |
|---|---|---:|---:|
| arXiv | C0 | 20.910 | 19.902 |
| arXiv | internal_only | 20.526 | 20.113 |
| arXiv | T+degree | 20.626 | 19.382 |
| arXiv | random_ring | 22.209 | 18.936 |
| BookChild | C0 | 13.316 | 11.190 |
| BookChild | internal_only | 14.324 | 12.034 |
| BookChild | T+degree | 14.944 | 12.193 |
| BookChild | random_ring | 14.631 | 10.060 |
| BookChild | degree_shuffle | 14.480 | 8.260 |

BookChild 的 T 同时改善旧类和新类；随机控制的平均分类提升可能伴随新类损失。不能只比较总体 F1 而忽略塑性。

BookChild LP 的最终 old/new/cross/within-new AP，C0 为 90.935/91.498/91.536/91.485，T 为 91.900/92.641/92.535/92.966。四个关系切片都有增益；arXiv 则不一致，原始分项和配对统计均已导出。

## 6. 拓扑诊断：哪些解释成立，哪些不成立

以下是最后阶段的部分全 memory 统计；“漂移”是相对各臂自身的上一阶段，不是相对 C0。

| 数据/顺序 | 方法 | 孤立行 | 最大度 | degree Gini | 旧子块 row-L1 漂移 | 含跨边完整旧行漂移 | 最大相对新增负载 |
|---|---|---:|---:|---:|---:|---:|---:|
| arXiv/49011 | C0 | 978 | 32 | 0.5280 | 0.01840 | 0.04064 | 2.333 |
| arXiv/49011 | T | 0 | 6 | 0.1313 | 0.01787 | 0.03676 | 0.333 |
| arXiv/49012 | C0 | 612 | 58 | 0.5111 | 0.01572 | 0.03458 | 3.000 |
| arXiv/49012 | T | 0 | 6 | 0.1049 | 0.01440 | 0.03020 | 0.333 |
| BookChild/49011 | C0 | 880 | 88 | 0.7326 | 0.06791 | 0.14610 | 3.000 |
| BookChild/49011 | internal_only | 0 | 11 | — | 0.05411 | 0.11120 | 1.667 |
| BookChild/49011 | T | 0 | 6 | 0.1262 | 0.05970 | 0.12053 | 0.333 |
| BookChild/49012 | C0 | 161 | 82 | 0.6089 | 0.10155 | 0.25555 | 3.000 |
| BookChild/49012 | internal_only | 0 | 30 | — | 0.09425 | 0.20925 | 4.500 |
| BookChild/49012 | T | 0 | 6 | 0.1120 | 0.11296 | 0.24178 | 0.333 |

BookChild 最后新块在 C0 下分别有 345/755 个连通分量，语义环都为 1。说明 C0 在同等边数下确实存在覆盖不足和高度集中的连接。

但不能把以下命题混为一谈：

- 降低最大接收负载：本轮成立。
- 降低每个节点或平均传播漂移：不保证成立。
- 降低漂移就会提高任务指标：本轮也没有证明。

原因是归一化传播权重为 `1/sqrt((d_i+1)(d_j+1))`。均匀分散新增边可能让更多低度旧节点的归一化权重发生变化。BookChild/49012 的 T 旧子块平均漂移反而高于 C0；T 相对 internal_only 的两种平均漂移在两个顺序均更大。

因此下一步应研究**语义质量、相对负载和传播变化之间的权衡**，而不是把“更均匀”直接作为优化目标。当前环的低质量闭合边也仍是待测假设，不应在未做 soft-closure 对照前当作已确认的退化原因。

## 7. 零训练 LP 对照必须保留

这些对照直接用真实节点特征/文本算余弦，不训练 GAE，也不使用真实图传播；与 GAE 不是同预算训练模型，但可诊断测试任务是否容易被原始相似度解决。

| 数据集 | 对照 | AUC | AP |
|---|---|---:|---:|
| arXiv | X cosine | 96.822 | 96.843 |
| arXiv | text cosine | 96.881 | 96.906 |
| arXiv | X+text cosine | 96.895 | 96.920 |
| arXiv | T+degree GAE | 96.031 | 95.711 |
| BookChild | X cosine | 79.423 | 81.755 |
| BookChild | text cosine | 80.337 | 82.730 |
| BookChild | X+text cosine | 80.111 | 82.500 |
| BookChild | T+degree GAE | 92.927 | 92.345 |

arXiv 的全部本批 GAE 都低于最强零训练相似度对照。BookChild 的 GAE 有更大的下游价值空间，但真实 train graph 传播也贡献了信息，不能把它相对余弦的全部优势归因于凝聚算法。

本轮没有新增 full-raw、k-center/medoid 或静态 TAG 凝聚共同重跑，因此不计算它们的性能复现率，也不宣称超过这些强基线。旧小图主表不能直接拼入本批大图拓扑消融。

## 8. 成本账与缺失项

下表为每次 student fit 的训练秒数，跨全部阶段、顺序及 seeds 平均；是同任务方法间对照，不是不同任务间成本比较。

| 数据集 | C0 NC | T NC | C0 LP | T LP |
|---|---:|---:|---:|---:|
| arXiv | 4.924 | 4.903 | 0.657 | 0.655 |
| BookChild | 4.557 | 4.557 | 0.585 | 0.585 |

所有臂同节点、同边数、同 student 更新数，训练成本基本相同。小于约 1% 的时间差不解释为算法提速。全部 fits 记录的训练时间合计约 4,294.93 秒，不包含构造、推理、审计、I/O，也不等于两 GPU 并行后的墙钟时间。

- memory 张量和文本有效载荷在同一数据/顺序/阶段的六臂完全相同；序列化文件名引起的少量字节差不是表示容量差异。逐个 memory 的 bytes 已写入独立审计。
- NC 峰值统计止于训练结束，LP 峰值在真实图推理之后记录。二者不能直接比较，也不能把 LP 峰值全部算成凝聚训练开销。
- **完整构造耗时缺失。** 原协议要求这一项，但 runner 未保存。旧轮次的成本值不能冒充本轮记录；当前不能验收“构造成本不超过 2× C0”。后续需要共同从内容构造、拓扑选择到保存的端到端计时，而非只测环构图。

## 9. 对完整算法路线的决定

这轮不是“T+degree 没用”，而是把原先捆绑的解释拆开了：

**已得到支持：** BookChild 上，以语义信息构造覆盖更完整的合成拓扑，可以改善两类下游任务；只有相同度分布和连通性不够复制全部 LP 收益。

**尚未成立：** 当前均匀跨块容量是普遍必要部件；两部件在所有任务都协同；方法优于强 semantic kNN；可以只靠这些结果宣称完整持续凝聚算法已达到论文最终标准。

上一轮通过的是“允许作为研究主线继续推进”的标准，本轮检验的是更严格的机制拆分与 CTAG-1 条件，二者不矛盾。保持 CTAG-0/T+degree 的冻结身份，不事后把某个指标更好的 internal_only 或随机臂替换成原主方法。

| 下一步 | 怎么做 | 本轮依据 / 验收要求 |
|---|---|---|
| 先补强拓扑控制 | 同 rows/content/pairs 下补 semantic kNN，必要时加 medoid 拓扑参照 | 六臂尚不能排除收益只是修复 C0 的集中/孤立缺陷；原路线此项未完成 |
| 再做 degree/load-aware capacity | 保留 internal_only、均匀 capacity；另以 `load/(old_degree+1)` 和候选语义损失作受控目标 | 不直接增大容量或叠加复杂模块；固定总跨边数，报告相似度损失与完整旧行漂移 |
| 单独测 soft closure | 固定边预算，比较强制闭环与质量受限的替代连边 | 不能减少边数后把差值归因于语义质量；不预先认定是 arXiv 的退化原因 |
| 计时与独立确认 | 新顺序/seed 在下一轮结果前登记；全臂共同记录完整构造成本 | 当前顺序已用于开发；先定选择规则，再查看独立确认结果 |
| 有证据再做 bounded rewire | 明确可修改旧边/行的预算，只用旧 memory 和新数据 | 用户并未强制旧图永久冻结；将不可回读与不可改写分开测 |
| 内容升级和完整比较 | 多原型 X、预算或文本只选一个候选逐步验证；补 full-raw/coreset/静态 TAG 对照 | 这些均未因本批六臂实验而完成或得到正结果 |

不建议现在同时加入 dynamic budget、student response selector、attention、多原型和 refresh。完整算法可以由可靠基础组件与一个有实验证据的新核心共同组成，不要求每个组件都是新发明；但目前的新核心需要上述强控制和独立确认才能确立。

## 10. 审计、文件与复现

| 审计项 | arXiv | BookChild |
|---|---:|---:|
| 重载并重新推理 checkpoint | 1,080/1,080 | 1,080/1,080 |
| memory 内容/拓扑检查 | 60/60 | 60/60 |
| 保存预测重算指标 | 全部一致 | 全部一致 |
| 原 source 文件哈希 | 152 个匹配 | 152 个匹配 |
| NC 最大重载指标差（pp） | 0.002848 | 0 |
| LP 最大重载指标差（pp） | 0.0000322 | 0.000846 |
| 严格指标容差告警 | 1 条，见下文 | 0 |

arXiv 告警位置为 `order=49012, stage=0, random_ring, seed=210`。三次额外重新推理均定位到 **1 个 all-prompt 预测改变，原 top-2 logits 恰好并列（margin=0）**；seen-prompt 预测不变，accuracy 不变，all-class Macro-F1 下降 0.002847762 pp。这是微小 logit 舍入引起的 argmax 差异，不涉及最终阶段。对 random_ring 跨五阶段、两顺序、六 seeds 平均的影响约 0.0000475 pp，不改变本文三位小数表或方法排序。原始指标不替换，容差不放宽，`strict_metric_reload_passed=false` 保留。诊断记录为 [nc_reload_roundoff_diagnosis.json](../../results/topology_causal_ablation_20260921_formal/arxiv/nc_reload_roundoff_diagnosis.json)。

本批独立审计覆盖：原记录中的 data/text/source hashes；完整 fit 笛卡尔矩阵及每条 JSON 与 `results.json` 的一致性；所有 memory 内容共享与每臂自身旧前缀；无向边双向化、无重复、无自环、等边数；新组件重建、2×2 因子边集合一致性、随机臂每节点度匹配；独立稀疏矩阵重算传播漂移；LP split 重建和 held-out 隔离；所有 checkpoint 的哈希、保存预测重算、重载权重后重新推理。

未覆盖的历史证据：manifest 未保存 MiniLM 路径/权重哈希及协议文件哈希；审计时可检查 prompt 编码一致，但不能事后伪装成结果前冻结的模型指纹。未保存逐簇成员表，故未独立重做完整聚类；源节点池和共享合成内容已核对。这些限制不应被“审计通过”掩盖。

主要证据目录：

- [arXiv 原始与汇总记录](../../results/topology_causal_ablation_20260921_formal/arxiv/summary_status.json)
- [BookChild 原始与汇总记录](../../results/topology_causal_ablation_20260921_formal/bookchild/summary_status.json)
- [arXiv 独立审计](../../results/topology_causal_ablation_20260921_formal/arxiv/independent_audit.json)
- [BookChild 独立审计](../../results/topology_causal_ablation_20260921_formal/bookchild/independent_audit.json)
- 每图 `endpoint_summary_percent.csv`：最终/阶段平均及 SD；`paired_contrasts_pp.csv`：同重复配对差和交互项；`trajectory_metrics_percent.csv`：逐轨迹数据；`cosine_summary_percent.csv`：零训练对照。
- `fit_summary.csv`、`topology_summary.csv` 保留逐阶段详细数据；前者 LP 为原生 0--1，不要与新百分数 CSV 混用。
- 大型 memory/checkpoint `.pt` 仍保存在远程 `/home/pc/research/TAG_condensation/results/topology_causal_ablation_20260921_formal`，文本证据已回传本地。

汇总与审计代码：`code/v2_llm_fusion/summarize_topology_causal_ablation.py`、`audit_topology_causal_ablation.py`。汇总 7 项、审计 6 项单元测试均通过，覆盖缺失/重复矩阵、空指标、百分数换算、配对统计、度统计、传播漂移和非法边。

本地重新汇总：

```powershell
python code/v2_llm_fusion/summarize_topology_causal_ablation.py --root results/topology_causal_ablation_20260921_formal/arxiv --dataset arxiv
python code/v2_llm_fusion/summarize_topology_causal_ablation.py --root results/topology_causal_ablation_20260921_formal/bookchild --dataset bookchild
python -m unittest discover -s code/v2_llm_fusion -p test_topology_causal_reporting.py -v
```

注意：不要运行正式 runner 来“刷新汇总”，也不要把另一个 `construct-only` smoke 目录中的 `complete.json` 当成完成训练。本报告仅使用名称带 `_formal` 的正式目录。
