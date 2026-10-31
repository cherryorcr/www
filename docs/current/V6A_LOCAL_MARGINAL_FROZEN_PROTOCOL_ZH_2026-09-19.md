# v6-A：局部合成编辑与 student 边际效用诊断

日期：2026-09-19。状态：正式运行前冻结。此文档不随结果修改。

## 1. 研究边界

用户授权推进 v6。v3/SCA、v4/P3-A、P1-B 已实现多步 response、独立 seeds、配额转移和回退；重新命名不能构成新颖性。本轮不是已成立的新主方法，不宣称首次或 SOTA。

新增可检验假设：旧配额转移会重新聚类整类，所谓“一行边际变化”实际上可能改变许多合成行。若把变化局部化，在固定预算下仅合并/拆分当前新块中的三个合成行，是否能提高真实 student 效用及短训可预测性？

这只是有限候选诊断，不声称已实现全局双层最优、可微 student 反传凝聚或动态总预算。

## 2. 构造

底座为 C0：合法 pool 累计 5% growing rows，dispersion，类内 joint 聚类，Mean-X，joint_g5_c320 concat/re-encode，冻结历史行/旧旧边，cross-degree=1。

每个诊断阶段都从同一 C0 历史出发，不把上一阶段选中的候选传递下来。因此本轮为单阶段反事实实验，不能把各阶段独立赢家拼成一条持续方法结果。

只研究 stage 1、2。候选为：

1. C0，不转移。
2. local：枚举不同新类 donor/recipient。donor 中合并 Ward 增量最小的一对簇，recipient 中将 SSE 最大且至少含两个源样本的簇确定性二分。保留合并后的原槽位，空出的槽位放第二个子簇，另一个子簇放原分裂槽位。只有三个新槽位重算 X、文本、mass；其它行逐位保留。
3. recluster：相同 donor/recipient 配额改变，但重聚类两个受影响新类。采用 Hungarian joint-center 匹配，将结果放入与 local 完全相同的类别槽位；未涉及的类保持。它是排除全局行枚举差异的整类重聚类控制，不是原 P1-B 数值复现。

所有 X 均由源簇均值合成，未引入真实节点 coreset；单样本簇是均值构造的退化情况，单独统计。每个候选当前源集合完全相同、总 rows 完全相同、各类至少保留一行。旧 memory 不改写。新文本变化引起的新边/跨边按相同规则重算，因此不能将 local 效果归因于单独的 X 算子。

源成员 ID 仅作为离线审计旁文件保存，不进入下一阶段 memory。

## 3. 数据与条件

Cora/Citeseer；canonical 作为已知开发条件。各另选一个从未记录过的 source-size 近似匹配三阶段顺序，排除 v5 registration 和历史完整划分；只用合法 pool 类规模，不使用任何分数。运行前保存 registration。

canonical 和新顺序均使用 selection groups 6200/6201 与 6300/6301，独立 report seeds 9200..9205。种子组不交叉。每个候选全部重训到 canonical steps，但从同一轨迹读取 1/8、1/4、1/2、全长四个检查点。跨 horizon 不当作独立重复。

预计 56 张候选 memory × 10 seeds = 560 次训练；若无法二分则成对跳过 local/recluster 并记录，不能补选有利候选。每个条件另构造 C0 三阶段链，仅作共同历史，不另报递归候选结果。

student 与 C0 相同：两层 GCN，384 hidden/out，冻结 MiniLM，双向 InfoNCE，Adam 1e-3，batch16，从头训练，steps=max(50,min(2048,16*rows))。

selector/report/test 分别 original_id %10 == 1/3/2，均不进入 source pool。主指标是 all-prompt、pooled seen support 的节点分类 Macro-F1，old/new 按同一 pooled support 的逐类 F1 计算；同时保存 seen-prompt 指标。

评估仍是继承 v3 的 transductive 完整真实图接口，selector 使用已见 selector-dev 标签。因此本轮不是完全无标签选择，也不是不允许历史验证服务/未来无标签图输入的严格在线评估。report 不参与候选选择，test 不评分。新顺序与新 seeds 不能消除反复使用 report-dev 的开发污染。

## 4. 预注册选择器

local 和 recluster 分别与 C0 组成候选集。每 seed 的分数为 1/4、1/2 两个终点相对同 seed C0 的 cumulative F1 增益均值；不用旧版未配对的两个标准差相减。

两组 selection seeds 各自选平均增益最高者，平局先 C0 再名称。仅在以下条件同时满足时接受，否则回退 C0：

- 两组选择同一非 C0 候选；
- 四个 paired gains 的 mean - sample_SE > 0.5 pp；
- 两组各自的 pooled old 平均变化 >= -2 pp；
- 1/4、1/2 各自跨 seeds 的平均增益 >= 0。

SE 仅为固定风险代理，不称为覆盖保证。四个 horizon 中 1/8 仅作早期诊断，全长仅作失配分解，不参与该选择器。

统一保存 geometry-SSE 选择、固定随机非 C0 候选、uniform-candidate 期望、full-selector argmax 和 report-oracle。后两者是诊断，report-oracle 不可部署且有赢家偏差，不作方法性能。

geometry 在包含 C0 的家族候选中选择最小 SSE，平局优先 C0；uniform 对包含 C0 的全部家族候选等权。random 在名称排序后的非 C0 候选中使用 Python Random(6400 + stage + 100*是否新顺序 + 1000*是否Citeseer) 取一项，不看分数。全长 selector/报告 oracle 按对应 seeds 的均值取最大值，平局先 C0。

## 5. 核心诊断

- 候选空间：全长 report 上相对 C0 的 oracle headroom、均匀随机候选表现。
- 排序：短 selector -> 全长同 seed selector（长度）；全长 selector -> 全长同 seed report（split）；同 split 全长 selection seeds -> 独立 report seeds（seed）。
- 选择：Spearman、top1 agreement、regret、回退、old/new 增益。
- 编辑：变化槽位、合成 mass 分布、SSE、字节、边数与旧块一致性。
- 成本：实测全长诊断费用；真实 1/2-prefix 选择所需训练/评估成本；单次部署的构造+selector+一个正式 student，不能拿六次 report 重复摊薄成本。

计时分别保存 student 训练、真实图 forward、selector/report 评分和状态快照审计。部署估算只计所用家族的四个 selection seeds，到 1/2 的训练及 1/4、1/2 两次 selector 评估；不计 report 与法证快照。候选构造暂将两家族全部构造时间记作保守上估计并明确标记，正式 student 用六个 report fits 的平均单次成本。显存峰值包含常驻 encoder/真实图与检查点评估，不称为纯 student 训练显存。

## 6. 下一阶段触发条件

本轮只是进入递归 v6-B 的筛选，不是主方法晋升。local 必须同时达到：

1. 两个新顺序共四个阶段中，至少两个阶段的 report-oracle headroom >= 1 pp；
2. 八个阶段单元中，1/2 selector 与独立全长 report 的平均 Spearman >= 0.3（常数排序记 0 并单列）；
3. local 选择相对 C0 的八单元平均增益 >= 0.5 pp，平均不低于 geometry 和 uniform-candidate 期望；
4. 两个新顺序数据集各自的阶段平均增益 >= 0，且其中至少一次实际接受非 C0；
5. 任一单元累计 F1 不低于 C0 超过 2 pp，pooled old 不低于 C0 超过 3 pp；
6. 源数据、预算、checkpoint、旧 memory、前缀等价审计通过。

失败则停在 v6-A，报告具体失配环节，不放宽阈值或启动 extra-budget。通过则在另留的顺序/seed 上实施递归固定预算候选选择，再考虑总预算。动态 rows 必须从离散候选预算中按效用/成本选择，不能把 F1 数值直接相加当节点数。

## 7. 复现与实现验收

新文件与 C0/v5 源码隔离；训练源快照、协议 hash、registration、候选来源/槽位、memory、四个 horizon 的 selector/report logits、模型 state hash、末端 checkpoint、耗时与选择记录必须保存。

共享前缀须与独立固定步数训练逐位比较；中间评估保存并恢复 RNG、模型模式，不更新优化器。两个数据集各选 canonical stage1 C0、seed6200，检查四个 horizon 与独立训练。末端额外与既有 G.train_student 比较。

最终报告清楚区分研究创新假设、已验证机制、性能证据、成本和未执行的后续阶段。
