# GraphCLIP continual v3 / SCA：协议、流程与证据说明

依据：本地现行代码、formal_unified、canonical_v3、baseline_v3、diagnostic_v3 原始 JSON，以及小图机制实验记录。本文区分现行实现、历史实验和研究设想；不启动新实验。

## 1. 方法到底做什么

输入是分阶段到来的文本属性图节点：节点属性 X、文本 T、原图 A；类别标签用于定义实验阶段与评估。输出是一份小型图文训练记忆 M_t，包含 synthetic X、文本、文本 embedding、边、mass、depth、stage。每阶段在 M_t 上从头训练新的图文 student，再在真实图上用类别文字描述分类。跨阶段继承的是压缩数据，不是 student 参数。

核心可以写为 M_t = Append(Prune(M_(t-1)), Construct(delta_t, recipe_t))。SCA 决定 recipe_t：用候选记忆训练 student，再用验证响应选择候选。核心构造没有对 synthetic X 做梯度优化，也没有使用 teacher trajectory matching；ASM 等属于其他诊断臂。

## 2. 协议层

### 2.1 数据与监督

- pool：original_id % 10 不属于 {1,2}；dev：余数 1；test：余数 2。原始 ID 不存在时使用本地行号，因此不同数据版本的 ID 对应必须核对。
- 候选构造和 InfoNCE student 训练不读取节点类别作为监督目标；阶段划分使用标签，新增预算使用新阶段类别数量，SCA 选择使用有标签 dev Macro-F1。因此整个流程不能称为完全无监督或完全无标签。
- 当前 zero-shot 指 student 不用节点分类标签训练，通过 graph embedding 与 class-prompt embedding 匹配预测；它不表示类别名称未知，也不表示 dev 标签未参与方法选择。
- 预测与 SCA dev 评估使用全量真实图 X/A，预测类别空间包含全部 prompts，评分 mask 才限制到已见阶段。因此这是 transductive 实验，不能当作未来图不可见的严格在线 inductive 实验。
- 统一 runner 加载全量数据并缓存全量文本 embedding。pool 切片限制候选来源，但 test_accessed=false 是运行声明，不是系统级访问隔离证明。应分别记录 test label 未评分、test source 未构造、全图特征/结构用于推理。
- 主表是 dev 结果，且同一 dev 参与 SCA 选择；尚不能当作封存 test 上的最终泛化结论。

### 2.2 静态、持续与诊断

静态：所有允许的 pool 一次凝聚；持续：delta 分阶段凝聚、压缩记忆跨阶段保留。两者的来源和优化机会不同，不能混用指标。Cora 类别阶段为 [0,1]/[2,3]/[4,5,6]；Citeseer 为 [0,1]/[2,3]/[4,5]；arXiv 使用冻结 classincr3；其他数据按 pool 类别规模排序后轮转分配为三阶段。classincr3 与自然时间流不同。

| 臂 | 构造可访问的数据 | 行为与含义 |
|---|---|---|
| append_fixed | 旧 synthetic + 当前 delta | 固定 joint_g5 配方，保留旧行后追加 |
| SCA | 旧 synthetic + 当前 delta，另有 dev 评估 | 在若干候选配方间选择 |
| real-node baselines | 旧选中记忆 + 当前 delta | random/k-center/herding/medoid 选真实行，再使用同一 append |
| recluster | 旧 synthetic + 当前 delta | 混合重聚类，允许改写旧记忆；不回读历史 raw |
| full_history | 全部已见历史 raw | 违反不回读历史 raw 的部署限制，仅作诊断 |
| raw reference | 未压缩数据 | 压缩代价参照，计算预算可能不同，不是数学性能上界 |

不回读历史 raw 是任务约束；冻结旧 synthetic 是待比较的方法选择。不能因为外部方法更新旧 synthetic 就认定它不符合持续任务。full_history、recluster 的对照同时可能涉及构造后端和边数差异，不能从均值直接归因某个单独机制。

### 2.3 预算与 student

K_t = floor(0.05 * 累计 pool 节点数)。这是随历史增长的节点预算，不是固定总内存，不是 5% 字节/文本 token/边/计算预算。

K_new = max(K_t - |M_old|, 15 * 当前新增类别数)，K_keep = K_t - K_new。15 是当前 delta 的总配额下限系数，不保证每类恰好 15 行；构造没有逐类分配。若需腾出预算，删除低 mass 旧行，保留行维持顺序并重映射边。

Cora 预算 28/78/108：stage2 原可新增 30，但 3 个新类要求 delta 至少 45，因此从 78 行旧记忆保留 63 行，再追加 45 行。冻结不代表永不淘汰。

student 为两层 GCN，hidden/out 均为 384，另有可训练的文本线性投影和温度。MiniLM 文本编码器冻结。Adam lr=1e-3，batch=16，steps=max(50,min(2048,16*roots))，最终 seeds=100..105。每步全记忆图传播，然后抽取 batch 行计算双向 InfoNCE；batch 不是独立子图大小。mass 用于记忆管理/重聚类，现行 train_student 没有以 mass 加权损失。

## 3. 每阶段完整算法

1. 根据冻结的 stage 和 pool 掩码取当前 delta，计算累计总预算、新增和保留配额。
2. 对文本用冻结 MiniLM 编码为归一化的 384 维向量 e_i。构造 joint_i=[normalize(x_i);e_i]；xonly 配方仅在聚类时使用 normalize(x_i)。
3. 使用 max-min 初始化的 Lloyd 类聚类；小池为现行 mass-weighted 实现，大池候选使用 GPU 分块后端（阈值 20000，5 次迭代）。后端是算法实现的一部分，不能未经等价性验证只称为无影响加速。
4. 每个簇生成一个 synthetic row：X 为簇内原属性均值；在 joint 空间取靠近簇中心的 g 条文本，截断后拼接并重新编码；记录簇大小 mass、出生 stage 和 fusion depth。字符配额近似控制，空格等会使实际总长与参数略有差异。
5. 新块内部按文本相似度选高分边；小块是全局 top-E 无向边再双向展开，E≈rows*degree。degree=1 不代表每个节点恰好一个邻居。大块为分块候选边近似。
6. 如旧记忆超出保留配额，按 mass 裁剪。frozen_old 保留剩余旧旧边，新块内部单独连边，每个新节点默认连接一个最相似旧节点并双向化。global 消融对合并记忆重新选边。
7. append_fixed 直接提交固定配方；SCA 对各配方得到的完整候选记忆执行选择，提交被选中的记忆。
8. 最终用 seeds=100..105 重新初始化 student，在该记忆上训练，报告真实图的 stagewise dev 结果；保存 artifact 和 JSON。

冻结旧旧边只保证所保留边的 identity 不变。新增跨边会改变旧节点度、归一化邻接和信息传播，student 参数也重新训练，所以旧节点表示和预测仍会变化。frozen/global 总边数可能不同，需 edge-budget-matched 对照才能分离结构保持与额外边的贡献。

## 4. SCA 的两种现存实现

### 4.1 统一 runner 中的大图简化版

候选为 joint_g3_c192、joint_g5_c320、joint_g8_c512、xonly_g5_c320。xonly 仍有文本、文本连边和图文损失，不是去掉文本的消融。g 与 chars 同时改变，不能把差异只解释为 g 的效果。

每个完整候选记忆在 selection seeds=300/301/302、batch=8/16/32、多 horizon 上从头训练。horizon 以 max(128,final_steps//4) 为基准乘 0.5/1/2 并截断到最终步数；大图为 256/512/1024，最终为 2048，不是最终步数的 0.5/1/2 倍。

每个 horizon 都重新训练，不是一次训练保存多个 checkpoint。记录新类、旧阶段、累计 dev F1 及方差。response_auc 为当前阶段 F1 的归一化梯形面积；tail 为最后两个 horizon 的均值。

统一 selector 按 (cumulative_tail_mean, response_auc, -tail_std, candidate_name) 字典序排序。累计尾部 F1 是主排序量，AUC 与方差主要用于打破前项相同的排序；这不是联合风险效用，更不是 Pareto/rank-consistency 选择。每阶段 4*3*3*3=108 次 selection student 训练，最终另有 6 次 canonical 训练，因此不能只报告最终 student 成本。

### 4.2 小图增强实验版

run_student_compatible_append.py 另有：跨 step/batch ranking consistency、保持相对旧记忆参考的 old_delta、Pareto 集合、风险效用、beam2、terminal full-step confirmation。其效用包含尾部新类表现、AUC、方差、old_delta 和一致性奖励。保持阈值与一致性约束有 fallback，不能当作不可违反的数学保证。

beam 保留多条完整记忆路径，减轻早期误删；terminal 用 selection seeds 在 full steps 复评，最终再用 canonical seeds 测试。只有实际运行这些选项的结果才能归因这些机制。多 horizon、独立 seed、冻结拓扑等已经实现，但排序稳定性仍未解决。

## 5. 当前性能

以下为三个大图 stage2 累计 dev Macro-F1（%），六个最终 student seeds 的均值。

| 方法 | arXiv | BookChild | SportsFit |
|---|---:|---:|---:|
| random_real | 17.54 | 13.43 | 29.36 |
| k_center | 19.68 | 12.03 | 30.03 |
| herding | 18.52 | 14.58 | 27.06 |
| kmeans_medoid | 18.66 | 13.61 | 29.07 |
| append_fixed | 20.43 | 13.97 | 25.90 |
| SCA 简化版 | 20.03 | 15.04 | 26.22 |
| recluster | 19.39 | 12.92 | 28.13 |
| full_history，历史 raw 诊断 | 18.99 | 12.08 | 29.21 |

SCA vs append 为 -0.40/+1.07/+0.31 pp。SCA std 分别 1.23/1.57/3.38；append std 为 1.03/1.50/3.47。这些是固定构造/选择路径下 student 波动，不是独立 selector 重复实验的方差，也不是置信区间。不能仅凭均值正差宣称显著提升。SportsFit 上 SCA 比 k-center 低 3.81 pp，是应正面解释的反例。

Cora/Citeseer formal_unified 的 SCA 只配置 joint_g5 一个候选，最终与 append 同为 51.82/58.89；这是退化为固定配方的运行，不是多候选 SCA 无效的证据。历史增强版某组路径达到 56.95/64.13，但 Cora 改变 selection seed 组后曾降至 49.72；不能选择最好路径代替完整稳定性分布。WikiCS/BookHis 当前正式结果仅有 append，最终为 53.08/17.41。

old_mean 是旧阶段各自 Macro-F1 的平均，current 是当前阶段子集 F1，cumulative 在合并样本上重新计算；跨阶段误报及类别集合使它们不可线性平均。特别地，现有 JSON 没有保存足以独立重算所有指标的逐节点预测，代码版本与 labels 参数也应冻结核对。

本次发现旧主表文字有几处与原始 JSON 不符：WikiCS/BookHis 实际保存了 old/current；arXiv recluster/full_history stage2 old 应为 32.13/32.00，BookChild 应为 15.70/14.85。本文采用 JSON，原表应另行纠正。大图 baseline 已统一评估超参数，但这不足以证明历史各臂使用完全相同的构造代码版本；部分 stage0 本应相同臂也有数值差异，需要 artifact/hash 审计后再作精细因果归因。

## 6. 已做优化及其证据

- 统一 runner、split、student seeds/batch/steps、来源排除：降低显式配置混杂；尚缺完整 source hash、访问追踪、每臂总成本记录。
- 多 horizon + 独立 selection seeds：减少单终点和同 seed 筛选偏差；同一 dev 仍被重复使用。
- frozen_old + artifact edge_index + shrink remap：避免训练时隐式重建旧旧边；不是免遗忘保证。
- rank consistency/Pareto/beam/terminal：增强版已实现并做小图诊断；没有整合为大图统一 selector。
- text mean/weighted/set_pool：现行 set_pool 是距离权重函数，不是学习的 Set Transformer；小图未显示跨数据集一致优势。
- dynamic rows：用当前 delta 的属性/文本离散度决定新增配额，受总预算和刷新上限约束；不是逐类别学习分配，也没有显式历史漂移估计。
- bounded_refresh：低 mass 旧行朝相似新行做凸组合；不等于带稳定性保证的记忆校正。rescore_only 目前没有实质重评分行为。
- GPU 分块、文本缓存：支持大图运行；构造近似、不同后端和多候选训练成本仍需单独计量。

## 7. 贡献、创新候选与 challenge

| Challenge | 当前应对 | 可主张程度 |
|---|---|---|
| 历史 raw 不可回读 | 新 delta 独立构造，旧 synthetic 记忆跨阶段传递 | 构造操作已实现；完整进程访问隔离未建立 |
| 几何相似不等于可训练 | 用真实 student dev response 选择 | 是最主要的方法候选；跨数据集增益不稳定 |
| 旧结构改写影响保持 | 显式保存并保留 old-old topology | 有机制依据和小图证据；边数及传播混杂未充分分离 |
| 新类塑性与历史保持冲突 | delta 配额，增强版 retention/Pareto | 仅缓解，没有稳定兼得结论 |
| 小记忆对 batch/seed 敏感 | 多条件响应、独立 seeds、beam | 已诊断；未解决路径不稳定 |
| 图文压缩造成语义错配 | 簇均值 + 代表文本重编码 | 可运行基线；文本混合未必优于真实原型 |
| 实验排名受协议影响 | 固定 v3 设置并保留原始结果 | 实验基础设施贡献，不能替代算法创新 |

论文最有潜力的中心问题是：怎样让持续压缩得到的图文记忆，在新增数据后仍然可被一个新 student 可靠学习？相应贡献候选是可重训响应选择、历史拓扑保留、以及响应排序/路径风险的系统分析。

max-min、Lloyd、文本拼接、InfoNCE、从头训练、append 各自不能独立当作原创性结论。此前相关工作已经涉及记忆凝聚和从头训练；当前需要证明的是具体组合的必要性、机制和同条件增益。外部方法检索/任务资格审查不等于外部算法重跑，更不证明该领域不存在可比较方法。

## 8. 下一步优先级

### A. 先修复证据口径

冻结 dataset IDs、代码 hash、构造后端、prompt、指标 label 集合、候选集合、selector 版本；区分无标签构造与 dev 有监督选择。为每阶段导出预测和混淆矩阵，重算 current/old/cumulative/BWT；将 selection-dev 与报告-dev 或最终封存 test 分离。把单候选运行明确标成控制组；补齐真正多候选小图运行与 WikiCS/BookHis SCA。

### B. 先解释 SportsFit，再增加复杂度

固定同一簇划分、边数、student 和随机初始化，对比 real medoid/mean X，以及单条文本/拼接文本。检查簇内语义多模态、属性均值是否偏离真实节点分布、重复文本形成假负例。把 real-node 原型纳入有限 SCA 候选集作为待验证方案，同时计算增加候选后的搜索成本与 dev 过拟合。失败原因目前是假设，不是已证实结论。

### C. 改进选择与计算效率

统一简化版与增强版实现，以相同选择计算预算比较。探索阶段相关 horizon、自适应延长训练、逐步淘汰置信不足候选、接近时回退稳健基线；一次训练保存多个 checkpoint 减少重复训练。使用多个互不重叠 selector seed groups，报告路径频率、top1 agreement、full-step regret、最终 F1 分布。

### D. 改进拓扑和记忆管理

先做相同总边数/相同跨边数的 topology 对照；区分 old-old edge 保存与旧节点传播算子稳定。再用旧记忆影响度、覆盖和不确定性指导有限刷新，比较固定总内存和随数据增长预算，扩展至更多阶段。任何稳定性约束都需覆盖 fallback 行为。

### E. 文本与预算创新

固定 token/byte 总预算比较代表文本、提取式证据、双原型、真正可训练的集合编码；不能把手工距离池化的负结果推广到全部集合建模。先用 dev 校准固定有限网格，再考虑学习式配置；逐样本或簇的预算分配应结合覆盖、冗余和 student 边际效用，避免直接使用类别标签分配却宣称 label-free。

### F. 外部比较

给出 external 原设定参照与明确命名的 v3-adapted 版本；共同限制应该是任务信息权限、预算和评估，而不是强制对手采用我们的 frozen_old 算法。保持原方法机制、记录适配变化并重跑后，才能作性能比较。暂不宣称 SOTA，也不能承诺某个新模块必然达到 WWW 水平。

## 9. 可复核入口

- `code/v2_llm_fusion/protocol_v3.py`：划分、预算、student 超参数。
- `code/v2_llm_fusion/run_unified_v3_continual.py`：统一数据、简化 selector、大图主表 runner。
- `code/frozen_v1_2026_09_09/run_student_compatible_append.py`：候选、append、响应、增强选择与 beam。
- `code/frozen_v1_2026_09_09/run_gate1_continual.py`：student、指标、mass 管理、重聚类。
- `results/canonical_v3_2026_09_15_{arxiv,bookchild,sportsfit}/results.json`：大图 append/SCA。
- `results/baseline_v3_2026_09_15_{arxiv,bookchild,sportsfit}/results.json`：大图 real-node 基线。
- `results/formal_unified_*/results.json`：小图和中图运行的真实候选配置。
- `docs/current/SCA_V3_RESPONSE_TOPOLOGY_OPT_RESULTS_ZH_2026-09-14.md`：历史小图增强实验及后续否定性复核。
