# TGC-R5 最小执行协议

本轮执行用户批准的四项工作：old-only 独立复验；内容/拓扑共同机制；有界旧记忆维护；固定总新增 rows 的容量调配。C0 是初始化和控制，不预设为最终算法。联合实验不以单项候选先获胜为启动条件。旧历史原始图不可回读；旧合成节点允许有界修改。

## A. old-only 独立复验

- Cora/Citeseer，三阶段，5% 累积预算；保留顺序 order19441/order19442，开跑前审计实际分组的历史重合。
- C0、relation_old、同旧行逐行等位移 random_old、shuffled_old。s0 相同，s1/s2 共享 C0 父记忆，仅作为机制复验。
- R4 配方不变；独立 teacher seeds 8200/8201、inner 8300/8301、validation 8310/8311、probe 8290；report 120..125。mask 仍使用 legacy-uniform teacher utility，不混称 lift teacher 选行。
- 主指标是最终 all-prompt pooled report-dev Macro-F1；同时报告阶段均值、old/new、随机控制、成本。测试标签不参与。
- 原探索性推进数值：final 均值 >= +0.5 pp；至少 3/4 cell >= -1；最差 final/old >= -2；平均 new 和相对 random >= 0。它们是研究推进规则，不是显著性或录用标准。

## B. 最小共同机制与递归运行

在已观察的开发顺序 order19341/order19342 上实现 teacher 引导的共享软来源分配，独立命名和封存快照。仅用旧 synthetic memory 与当前 delta 训练 teacher；共享分配产生加权 synthetic X、原文片段融合后重新编码的文本、投影后的稀疏二值边。最终 student 使用实际落盘内容和边，不使用 soft surrogate。

四臂析因：C0、仅内容、仅拓扑、联合。内容包括 X 和文本，所以不将该析因解释成 X 与文本各自贡献。比较保持相同 rows、块内边数、新旧边数和 student；记录实际字符/token，字符预算相同不等于 token 完全相同。s1/s2 共享 C0 父记忆。

此外始终完成最小递归联合臂、有界历史修订臂、固定新增 rows 的容量调配臂；各臂在下一阶段读取自身输出。容量臂允许类间转移而不改变当前总 rows。s0 共同使用 C0，变体从首个增量开始。开发结果不能当独立泛化确认；详细目标、常数和源代码哈希在该实验开始报告前由 runner 写入 protocol.json 并锁定。

## C. 共同验收

两层共同 GCN、MiniLM 输入、batch=16、Adam lr=0.001、steps=max(50,min(2048,16*rows))；全部候选先构造封存，再进行完整 scratch student 报告训练。构造/teacher/优化不读取 report/test 标签；C0 当前池类别分配仍使用训练池标签。

先运行实现不变量、梯度与数值检查；完成后独立重载全部不同 checkpoint 重算 all/seen 指标。报告配对逐 cell、六 seed 差值、跨 cell 描述性区间、析因交互与成本。递归结果与共享父记忆结果分表。若候选失败，保留结果并定位问题；不按 cell 拼接最优臂，也不事后改变参数重称确认。

机器冻结协议以两轮各自的 `protocol.json`、`selection_seal.json` 和隔离 snapshot 为准；本文后续补充解释不修改已经封存的实现或参数。大图、长链、多 rate、正式新条件晋升验收不属于本轮最小实验。

## D. 共享分配的实际实现

这部分解释 `tgc_shared_assignment.py` 已封存的实现；不表示其已优于控制。

1. 用 C0 的新块作为初始化。合法支持集 U 仅有旧 synthetic memory 和当前 raw delta。将合成新旧边按 C0 来源分配分摊到当前原始节点，旧节点的归一化旧块与当前 compact 图对应。该性质不等于旧输出不变，也不等于恢复历史原始图。
2. 在 U 上训练两个共同架构的 teacher，seed 8400/8401、各 128 步、batch=64；采样保持 compact 图的旧行比例。冻结其图表示与投影文本表示，拼接为 H。teacher 的训练步数和 batch 是构造成本，正式 student 仍沿用共同协议。
3. 当前来源到新合成行的软分配为 P，旧行保持单位映射，完整映射记为 Q=diag(I,P)。当前训练池类别限定 P 的非零位置，初始为 0.9 C0 hard assignment + 0.1 类内均匀分配。softmax 输出混入 0.02 均匀质量，避免直接退化为选取一个真实节点。这里使用训练池标签，不宣称无标签凝聚。
4. 设带自环的合法加权邻接为 A，Z=diag(Q^T 1)^(-1) Q^T H，W=Q^T A Q，B=diag(W1)^(-1)W。最小化 H 与 QZ 的重构误差、D^(-1)AH 与 QBZ 的传播误差，加 0.05 的初始分配 KL 和 0.01 的 log-mass 漂移。旧/新误差分别求均值，再按 compact 旧行比例加权。Adam lr=0.05，共 128 步，固定取最后一步。
5. 输出新 X=P^T X_delta/column_mass。每行按 P 权重选择至多 5 个当前原文来源，固定 320 字符以内拼接，再用冻结 MiniLM 重新编码。旧文本表示不变。所有合成 X 都是来源混合，不是把真实节点直接当作合成节点。
6. 对投影 W 做对称度归一化打分，保留与 C0 相同数量的新新无向边，并给每个新节点选一个旧邻接点；旧旧边不变。最终保存的是二值对称 edge_index，正式 student 不使用 soft edge weights。
7. content/topology/joint 使用同一个 P：分别只替换内容、只替换边、同时替换两者。新 X 与文本在 content 臂中共同改变，不能由本轮推断二者各自贡献。

该目标是 teacher 空间中的压缩近似，不是对完整 student 训练做精确反向传播；图文重构误差下降不自动意味着节点分类 F1 上升。二值化和片段重编码也会引入 soft-to-hard 差距，必须以实际产物的完整训练检验。

## E. 递归、维护与容量的实现边界

`recursive`、`maintenance`、`capacity` 从 s1 开始分别沿自身 memory 运行，不根据报告 F1 进行切换。

- maintenance：先做 joint 构造，再使用该次 P 将新旧边 lift 到当前来源；用 R4 关系响应修订最多 10% 旧行的 X，相对位移上限 0.1，短训旧关系 KL 约束 0.001。文本和边在修订步骤不变。该约束只针对指定 32-step student/probe，不是完整训练的历史保持保证。构造 seed 为 8400/8401，inner 8500/8501，validation 8510/8511，probe 8490。
- capacity：最多转移一行新增配额。对 donor 合并 teacher 表示最接近的两个初始簇，对 receiver 分裂 teacher 重构误差最大的可分簇，其它簇不重聚类。逐个 donor/receiver 对重新优化 P，以重构+传播误差选择，允许保留原配额；不比较含 allocation-dependent regularizer 的总目标。总新 rows、总 memory、边数和 student steps 不变。该版本同时包含对应的局部 merge/split，不能把全部效应归为抽象的“增加一个配额”。

容量臂使用相同 teacher 压缩目标，但不是 student marginal gain，也没有新增总预算。旧边冻结是本轮为归因保留的设计选择，不是用户施加的任务约束。完整 TGC 的可学习文本处理、旧边修订、训练更新匹配尚未由本轮实现。
