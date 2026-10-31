# 块保持持续文本属性图凝聚：从 Student-Compatible Append 到度均衡拓扑的跨任务分析

**论文草稿参考版本：2026-09-16；结果更新：2026-09-20**
**研究状态：小图基线与大图跨任务确认汇总；T+degree 作为后续主方法基础，C0 保留冻结对照；不宣称 SOTA，不作为最终投稿定稿**

**本版证据范围：** 保留[外部 TAG 基线报告](../current/EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)、[官方 GraphCLIP checkpoint 报告](../current/OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md)及 [P5 指标口径审计](../current/P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)，新增联合核探索、小图链路预测、大图筛选和 [T+degree 跨任务确认](../current/DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md)。小图完整数据保留率、大图节点分类与链路预测分开报告，不将不同顺序、seed 或 student 的分数拼表。本次仅整理已完成实验，没有新增训练或修改冻结协议。

| 名称 | 本文含义 | 当前定位 |
|---|---|---|
| 历史 v4 / SCA response-on | 合成追加加短程 student response 配额选择 | 历史分支；不将其数字改名为 C0 |
| C0 / response-off | dispersion、mean-X、center concat/re-encode、文本相似度追加 | 冻结共同对照；旧小图完整数据保留率属于此版本 |
| v5 / v6-A / v7-A 候选 | 保持约束、局部边际编辑、梯度兼容性选择 | 各自未通过晋升；不是连续替换 C0 的已成立主方法 |
| 原始联合核及 V2 | X、text、graph signature 与度/coverage 等控制 | 分类门禁未通过；保留探索证据，不等同于 T+degree |
| T+degree | C0 节点构造不变，改为文本引导的块内环与容量受限跨块边 | 通过最新预定研究推进标准，作为后续主方法基础 |
| XT+degree | 同样连接约束，评分加入 X 核 | 预设对照；不按任务或数据集事后替换默认 T |

## 摘要

文本属性图（Text-Attributed Graph, TAG）凝聚的目标不是简单地保存少量节点，而是在受限记忆和历史原图不可回读的条件下，持续构造一份能够被下游图文学生可靠重新训练的合成图文记忆。这个问题同时受到图结构传播、节点属性压缩、文本语义压缩、类别增量、新类塑性和旧类保持的共同影响。现有静态凝聚结果难以直接回答：当新数据到达后，怎样追加记忆而不破坏旧结构，怎样判断一个候选合成块是否真的适合重新训练 student，以及几何上更均匀的节点是否一定带来更好的持续性能。

本文整理并审计一个面向持续 TAG 的合成追加框架。方法在随数据量增长的固定压缩率预算下，根据类别规模和图文离散度分配新增合成行，在类内联合空间聚类，并生成均值属性及拼接后重新编码的文本。最新 T+degree 保留 C0 的节点构造，只替换两类选边规则：新阶段块内以文本余弦引导度为 2 的连通环，跨块以旧节点接收容量约束追加边。文本核是两类边共用的已有评分规则，不是新增可训练模块。旧合成行和旧旧边冻结；历史 response 选择、refresh、graph signature 和注意力均未启用。

在 Cora 和 Citeseer 的历史三阶段协议中，C0 分别使用 94/111 个合成节点，最终 report-dev Macro-F1 为 49.85/59.17，相对同 student 完整合法 pool 长训所选检查点的经验性能保留率为 97.16%/106.42%；历史 v4 的 49.85/60.77 单列。在 arXiv/BookChild 的新顺序 base+4 确认中，同一合成 memory 分别重训分类与链路预测 student，共完成 1080 次训练及全量 checkpoint 重载。T+degree 相对 C0 的最终分类变化为 -0.662/+1.546 pp，链路预测 AUC/AP 变化分别为 -0.011/-0.050 pp 与 +0.723/+0.980 pp。BookChild 两顺序、两任务均改善；arXiv 分类小幅下降，链路预测存在顺序依赖。

外部 TAGSAM/STAD 适配和官方 GraphCLIP checkpoint 仍按任务与数据权限分层报告，不作为新大图实验的匹配分母。T+degree 构造耗时在 arXiv/BookChild 约为 C0 的 1.42/1.15 倍，student 训练基本不变；arXiv 零训练余弦链路预测仍超过本轮 GAE。结果支持度与连接容量受控的合成拓扑作为下一步主线，而非普遍优于 C0、统计显著非劣性或 SOTA。内部环与跨块容量的独立作用、强基线及完整图性能保留率仍需在新协议下检验。

**关键词：** 文本属性图；持续图凝聚；块保持追加；度均衡；合成图记忆；图文对齐；链路预测；跨任务效用

## 1 引言

图凝聚通常将大图压缩为少量节点和边，再用压缩图训练下游模型。对于文本属性图，压缩对象不只是节点特征和拓扑，还包括与每个节点对应的文本描述、图文对齐关系以及文本驱动的合成图边。静态设定中，方法可以反复访问完整原图并为最终目标共同优化；持续设定则要求系统按阶段接收新数据，旧原始图在阶段结束后不可回读，只能依赖已经保存的压缩记忆。

这个 setting 带来四个相互牵制的挑战。第一，压缩记忆必须同时支持旧类保持和新类塑性。固定总预算会导致新增信息挤压旧记忆，固定每类行数又无法适应类别规模和类内多模态。第二，图文压缩的几何相似性不等于 student 可重训性。一个候选节点可能接近原始簇中心，却因为文本过短、拓扑不合适或局部负样本结构改变而难以重新训练。第三，冻结旧边并不等于旧表示冻结。新增跨边会改变节点度和归一化邻接矩阵，从而改变旧节点的信息传播。第四，持续实验很容易被协议混杂污染：训练步数、batch、seed、prompt 范围、真实节点与合成节点、历史 raw 访问和指标聚合方式都可能改变结论。

本文研究的问题是：在不回读历史原图的前提下，能否构造一份随着新数据追加而增长、同时保留历史结构并保持 student 可重训性的合成 TAG 记忆？我们将研究重点从“生成最像原图的节点”转向“生成能被下游 student 稳定重新学习的记忆”。

本文的主要工作包括：

1. **统一并审计持续 TAG 协议。** 固定类别阶段、数据掩码、5% cumulative fixed-rate 预算和共同 student；在两大图上增加新顺序、新 student seeds 与新 LP splits，明确节点分类的 transductive 条件和链路预测的隐藏边边界。最新确认含 1080 次训练、全量重载与 60 份 memory 审计。
2. **构建并检验度均衡块保持追加。** 以 C0 的合成节点为共同输入，只改变块内环形连接及跨块接收容量；在同节点数、同无向边数下比较分类与关系预测效用。文本余弦是已有评分器，不把“使用文本核”单独宣称为创新；两个约束的因果拆分和文献新颖性仍待补充。
3. **以同 student 性能保留率评价凝聚。** 比较完整合法 pool、静态 TAG 凝聚适配、持续合成追加和真实节点 coreset，报告最终及阶段平均性能保留率，并结合旧类/新类、计算成本和配对 student seeds 分析差距。官方预训练模型和监督任务仅作分层参照。
4. **报告跨任务收益和失败边界。** 文本 set/dual prototype、response、历史 refresh 和复杂联合核未形成稳定默认升级；度均衡的收益在 BookChild 两任务上得到确认，但 arXiv 分类退化、LP 顺序依赖和余弦强对照仍限制结论。不把任务条件性收益外推为通用最优，也不以本地适配负结果否定官方方法。

## 2 相关工作

### 2.1 文本属性图与图文对齐

GraphCLIP 类方法通过图编码器和文本编码器将节点属性、局部图结构与文本表示映射到共同空间，使类别 prompt 可以用于 zero-shot 节点分类。对于 TAG 凝聚，这种图文对齐既是压缩依据，也是下游 student 的训练目标。本文主评估器为简化的 GraphCLIP-style 两层 GCN，而不是官方 GPS 架构；训练图编码器、文本投影和温度参数，冻结 MiniLM。官方 12 层 GPS checkpoint 的纯推理结果另设外部参照，不替换共同 student。

### 2.2 静态图凝聚

GCond、SFGC、OpenGC 和 GCondenser 等工作研究了通过梯度匹配、特征统计或结构保持生成小型图。它们主要关注静态训练图或固定任务。静态方法的 full-history 访问假设不能直接转化为持续凝聚：持续系统必须决定哪些历史信息可保存、哪些旧节点可改写以及新数据如何进入记忆。

针对 TAG，本文已接入 TAGSAM 响应匹配静态适配和 STAD-local-canonical 静态适配。它们允许访问累计合法历史 pool，与持续 append 的数据访问权限不同；下游统一为本项目 student 后，结果反映其凝聚记忆在该图文任务中的效用，不等价于官方原生任务分数复现。

### 2.3 持续图学习与持续图凝聚

PUMA、CaT、GECC 和相关持续图凝聚工作涉及类别增量、动态图快照或模型 replay，但任务监督、student、memory 访问和评估口径与本文的 zero-shot TAG 协议不同。它们是重要的外部参照，但不能在 setting 不同的情况下直接合并为同一主表。本文的早期 TAGSAM 持续适配与后来新增的全历史静态适配分别命名和报告，均不冒充官方完整复现。

### 2.4 与本文的区别

本文不把“节点更接近原始中心”作为唯一评价目标，而是将合成记忆视为下游 student 的训练接口。历史 response 分支直接估计短训收益，但排序与成本问题使 C0/T+degree 均关闭该选择器。当前研究转向固定节点内容后的拓扑干预：在不能回读原始历史的追加场景，连接覆盖及跨块负载分布是否影响同一 memory 的跨任务可重训效用。环、度约束和容量分配本身是已有图算法思想，本文尚未完成证明其组合具有文献新颖性的系统核验。

## 3 问题定义与统一协议

### 3.1 持续 TAG

令原始 TAG 为 (G=(V,A,X,T,Y))，其中 (A) 为边，(X) 为节点属性，(T) 为文本。合法 pool 的标签 (Y) 可用于阶段划分、按类构造和预算分配；selector-dev 标签用于候选选择，report-dev 标签仅用于报告。本方法 student 的 InfoNCE 不使用分类交叉熵，但整套 pipeline 不是 label-free。数据按阶段到达，当前增量为 (Delta_t)。系统保存的不是原始子图，而是合成记忆：

\[
M_t=(X_t^{syn},T_t^{syn},E_t^{syn},w_t,d_t,s_t),
\]

其中 (w_t) 是 source mass，(d_t) 是文本或重聚类深度，(s_t) 是出生阶段。持续算子写为：

\[
M_t=\operatorname{Append}(M_{t-1},\operatorname{Construct}(\Delta_t;\theta_t)).
\]

阶段结束后，不允许读取 (V_{<t}) 的原始节点属性和文本。允许读取 (M_{t-1}) 以及当前 (Delta_t) 的合法训练 pool。

### 3.2 类别阶段与数据掩码

Cora 使用三阶段 ([0,1]/[2,3]/[4,5,6])，Citeseer 使用 ([0,1]/[2,3]/[4,5])。原始节点 ID 的末位决定掩码：selector-dev 为 1，report-dev 为 3，test 为 2，其余为 condensation pool。构造来源排除 selector-dev、report-dev 和 test。

节点分类 student 推理使用完整真实图的无标签 (X/A)，再用 prompt 计算类别预测，因此属于 transductive setting；它不是未来节点和留出节点都不可见的严格 inductive setting。本文节点分类不评价 test 标签，但项目历史开发文档存在早期 test 访问记录，不能包装成最终盲测。新增链路预测使用独立的 train/val/test 边划分并报告留出 test edges，其任务与“节点分类 test 标签未读”必须区分，具体设置见第 6.6 节。

### 3.3 预算

主协议使用 cumulative fixed-rate：

\[
K_t=\lfloor \rho N_t^{pool}\rfloor,\qquad \rho=0.05.
\]

Cora 的历史累计预算为 25/67/94，Citeseer 为 38/86/111。新增行数为 (K_t-|M_{t-1}|)，C0/T+degree 不因每类固定 15 行下限而裁剪旧行。预算增长随合法 pool 增大，因此“固定 5%”是固定压缩率，不是固定总内存；新大图预算链另列于第 6.6 节。

### 3.4 评价指标

Macro-F1 评价的是**节点分类**：在真实图上计算节点图表示，与类别文本 prompt 表示匹配，取最大相似度对应的类别。新增链路预测另用 ROC-AUC/AP，不能与 F1 混为同一指标。对已见类别集合 (C_t)，定义：

\[
F1_c=\frac{2TP_c}{2TP_c+FP_c+FN_c},\qquad
\operatorname{MacroF1}_t=\frac{100}{|C_t|}\sum_{c\in C_t}F1_c.
\]

分母为零时按实现记为 0。Macro-F1 对类别等权；Accuracy 则是分类正确的节点比例，二者不能互换。

报告 report-dev 上的累计 Macro-F1、当前新类 F1、旧类 F1、阶段平均、准确率和训练成本。累计 F1 在全部已见 report-dev 样本上一次计算，不能用各阶段任务 F1 的简单平均替代。历史五层比较的旧类/新类使用同一已见样本支持集上的 per-class F1（pooled），会计入跨任务 false positives，旧类按类别等权。该比较同时报告 all-class prompt 和 seen-class prompt；最终阶段两者一致。

**P5 口径例外：** 冻结 unified runner 的 old/new 为 task-slice：先按真实标签筛选每个任务的样本再计算 F1，old 对旧任务等权，其他任务样本产生的误报不计入该切片。它不等于上述 pooled old/new，不能直接拼表。P5 cumulative 仍在全部已见样本上计算，但使用目标中实际出现的类别；五层 scorer 使用协议指定的已见类别，缺失类别时两者可能不同。当前 Cora/Citeseer canonical 的累计一致性已逐 seed 验证。P5 只保存全类别 prompt 口径，不虚构其中间 seen-prompt 指标。

凝聚性能保留率定义为 `100 × 方法最终 F1 的六种子均值 / 同 student 完整 pool 长训所选 checkpoint F1 的六种子均值`。阶段平均保留率是双方三个阶段累计 F1 等权均值之比，默认全部类别 prompt 竞争。它们不是逐种子比值的平均、预测一致率或理论信息保留率；官方预训练模型和监督任务参考均不进入该分母。

最新跨任务确认的 NC 使用 global-support old/new：在同一已见 report-dev 集合计算逐类 F1，再按旧类/新类分组平均，不使用 P5 的 task-slice。LP 在已见 pool 的留出正负边上计算 ROC-AUC/AP，正负例 1:1，并按 old/new/cross/within-new 关系切片。分类、LP 的最终均值分别汇总 2 orders × 6 seeds、2 orders × 2 splits × 6 seeds；五阶段平均等权。共享 memory 的重复训练不视为独立数据集。新确认未补匹配 full-raw，故不计算新的完整图性能保留率。

## 4 方法：T+degree 块保持合成追加

### 4.1 方法总览

Student-compatible append 研究的是合成记忆能否被重新初始化的 student 有效利用；它不是必须启用 response selector 的同义词。当前 T+degree 的流程为：

1. 固定合法 pool、类别顺序和累计 5% 预算，计算本阶段新增行数，不裁剪旧行。
2. 用冻结 MiniLM 编码当前文本，根据类别规模与图文离散度分配新增行数。
3. 在每个类别内部按所分配的行数做确定性联合聚类，生成 mean-X、中心文本拼接及重编码表示。
4. 在整个新阶段块内，以合成文本余弦为评分，在度和连通约束下选内部边。
5. 每个新节点向仍有本阶段接收容量的旧节点追加一条文本相似跨块边。
6. 保留旧行与旧旧边，保存当前 memory；分别从头训练分类和链路预测 student，再在各自的真实图评估条件下报告结果。

步骤 1--3 与 C0 相同；步骤 4--5 替换 C0 的两类选边规则。文本核只是这两步共用的相似度评分，不是先生成一张“文本核图”，再额外生成两张图。构造器只读取当前 pool 的 X/text/y 和旧合成 memory，不读取真实边或历史 raw；真实边用于下游评估传播及 LP 切分，不参与当前合成边生成。历史 v4 的 response 仅在第 4.6 节说明，未混入上述流程。

### 4.2 图文联合表示与聚类

对当前 pool 节点 (i)，冻结 MiniLM 得到 (e_i\in\mathbb{R}^{384})，将图属性归一化后组成：

\[
z_i=[\operatorname{norm}(x_i);\operatorname{norm}(e_i)].
\]

先按第 4.3 节分配类别行数，再采用确定性的 max-min 初始化和 Lloyd 更新形成对应数量的簇。source label 用于类别分块和预算统计，但最终 synthetic node 不是原始真实节点；student 的 InfoNCE 训练不使用分类标签。C0/T+degree 的整套 pipeline 因此不能称为完全 label-free。

### 4.3 新增行分配

对阶段内类别 (c)，令 (n_c) 为合法 pool 规模，(d_c) 为图文联合离散度，当前基础分配权重为：

\[
w_c=\sqrt{n_c}(1+d_c).
\]

其中，令 (\bar u_c)、(\bar v_c) 分别为该类逐行归一化 X、文本 embedding 的均值，实际离散度为：

\[
d_c=\operatorname{clip}\left(\frac{(1-\|\bar u_c\|^2)+(1-\|\bar v_c\|^2)}{2},0,1\right).
\]

总新增行数通过 largest-remainder 分配，预算可行时每类至少一行。这个设计缓解固定 15 rows 的规模失配，但仍是启发式先验，不是通用最优 allocator。历史 v4 启用类间 response marginal 校准（第 4.6 节），C0/T+degree 均关闭；C1 类内多原型与 P1-B guarded transfer 已做小图诊断但未晋升，不能写成默认方法。

### 4.4 Synthetic node 与文本

对于簇 (C_j)，合成图属性为：

\[
\tilde{x}_j=\frac{1}{|C_j|}\sum_{i\in C_j}x_i.
\]

文本从靠近 joint center 的代表文本中选择最多 (g=5) 条，将约 320 字符内容预算按实际代表条数分配后截断拼接；选满 5 条时每条最多 64 字符，连接空格另计，最多可达 324 字符，而非每条 320 字符或严格 320 tokens。拼接文本重新输入冻结 MiniLM 得到 (\tilde{e}_j)。节点保存 (mass_j=|C_j|)、stage 和 depth；当前分类 student 不按 mass 加权 InfoNCE。P2 严格 token 实验进一步测试了 32/64/128 内容 token、mean/weighted/set/dual prototype 和 DeepSets，但没有发现跨 Cora/Citeseer、独立 seeds 和阶段均值都稳定更好的算子，因此主线保留 center concat。

### 4.5 两类选边规则：内部环与跨块容量

设旧记忆有 (K_o) 行，新阶段块有 (K_n) 行。C0 和 T+degree 共用重编码后的合成文本评分：

\[
s_{ij}^{T}=\cos(\tilde e_i,\tilde e_j).
\]

它只用于排序，输出为单位权重无向边，存储时双向化，不把余弦值当边权。当前 T 不使用 X 核或 graph signature；对照 XT 使用 (4/7)s^X+(3/7)s^T，其余约束相同。

| 环节 | 冻结 C0 | 当前 T+degree |
|---|---|---|
| 节点内容与预算 | dispersion、mean-X、concat/re-encode、5% growing | 完全相同，行顺序也相同 |
| 新块内部边 | 按文本高相似度取固定数量边，无度或覆盖约束 | 相同边数，度上限 2 且避免提前成环，最后闭合路径 |
| 新旧跨块边 | 每个新节点接最近旧节点，无接收容量限制 | 每个新节点接最相似的尚有容量旧节点 |
| 旧行与旧旧边 | 冻结 | 冻结 |
| response、refresh、额外预算 | 关闭 | 关闭 |

**块内选边。** 对整个新阶段块的所有节点对按文本余弦降序、固定 tie-break 排序。只有两端当前度均小于 2 且属于不同连通分量时才接受一条边；用并查集禁止提前成环。取到 (K_n-1) 条边时形成覆盖所有节点的路径，再连接两个端点。对于本轮均满足的 (K_n\ge3)，得到 (K_n) 条无向边的环。它跨越该阶段的多个类别，不是每类单独一环；闭环边也不保证有高语义相似度。C0 大块使用近邻候选加速后取高分边，不应将其描述为所有规模下均精确穷举全部节点对。

**跨块选边。** 旧记忆非空时，固定新行处理顺序；每个新节点从仍有容量的旧节点中选择最高分者。每个旧节点在本次追加可接收的上限为：

\[
b_t=\left\lceil K_n/K_o\right\rceil.
\]

每次分配后递增该旧节点计数。这是逐行贪心，不是全局最优匹配；上限只约束本阶段新增连接，不是整个生命周期的总度上限。由于总接收容量不小于 (K_n)，每个新节点都能获得一条跨边。首阶段没有跨块边。

最后拼接三类边：

\[
E_t=E_{old}\cup E_{new}\cup E_{cross}.
\]

旧旧 edge identity 不被重建，但新增跨边仍改变旧节点度和归一化邻接，所以“块保持”不等于旧传播算子不变或免遗忘。环保证新块覆盖，容量限制缓解跨边集中，这是设计动机；是否改善分类或 LP 必须由实验决定。本轮将两种约束作为整体比较，尚未通过 2×2 实验分离各自贡献。

### 4.6 Student response selection（历史 v4 分支）

候选评估使用 selection seeds 300/301/302、batch 8/16/32 和多个 response horizons。响应分数由累计尾部 F1 与旧阶段保持组成：

\[
S=\operatorname{tailMean}_{cum}+0.25\operatorname{tailMean}_{old}.
\]

风险分数使用经验标准差惩罚：

\[
S_{risk}=S-\max(\operatorname{std}_{cum},\operatorname{std}_{tail}).
\]

这不是经过统计校准的置信下界。历史 v4 还启用了 response marginal：在固定总新增行数下，将一行从 donor 类转移给 recipient 类，只有多 horizon 的风险增益达到阈值才接受。C0/T+degree 均跳过本节，不产生选择训练成本。response-gain extra reserve 也默认关闭，因为此前额外 rows 的短训收益不能可靠预测 canonical F1。

### 4.7 Downstream student

Student 为两层 GCN，hidden/output dimension 均为 384；文本编码器 MiniLM 输出 384 维并冻结，新增一个可训练文本线性投影和 logit scale。图文对齐使用双向 InfoNCE：

\[
\mathcal{L}=\frac{1}{2}(\operatorname{CE}(S,S_{id})+\operatorname{CE}(S^\top,S_{id})).
\]

其中 (S_{ij}) 为 batch 内节点 (i) 与文本 (j) 的缩放相似度，(S_{id}) 是配对索引而非节点类别标签。正样本是节点与其自身文本，其余 batch 配对作为负样本；训练目标是实例级图文匹配，推理目标则是与短类别 prompt 匹配。本文使用共同 student 来隔离凝聚记忆差异，不将其当作已充分优化的分类模型。

每步对完整 synthetic graph 做传播，再抽取最多 16 行计算损失，并非只在 16 节点子图上传播。Adam 学习率为 (10^{-3})。历史小图使用 (\max(50,\min(2048,16K_t))) 步和 seeds 100--105；最新大图确认固定 2048 步、seeds 210--215。每个阶段均 scratch 训练，不继承上一阶段权重。

**链路预测另用 GAE。** 两层 GCN hidden/output 为 384/128，输入为 ([\operatorname{norm}(X);\operatorname{norm}(T)]/\sqrt{2})，内积解码并以边二分类损失训练。正样本来自合成边、负样本来自合成非边，每步正负边 batch 各至多 256，Adam lr=0.001、256 步。真实图评估只在已见合法 pool 和真实 train edges 上传播，对留出真实边计算 AUC/AP。NC 与 LP 共享同一份 memory，不共享 student 参数；两种模型分别用于固定任务内比较。

### 4.8 算法伪代码

```text
Input: current delta D_t, previous synthetic memory M_{t-1}, rate rho
K_t <- floor(rho * cumulative_pool_size)
K_new <- K_t - |M_{t-1}|; keep <- |M_{t-1}|
H_t <- frozen MiniLM(D_t.text)
Z_t <- concat(normalize(D_t.x), normalize(H_t))
allocation <- largest_remainder(K_new, sqrt(class_mass) * (1 + dispersion))
clusters <- classwise_joint_clustering(Z_t, allocation)
block <- synthetic_mean_features_and_center_text(clusters)
score_new <- cosine(block.reencoded_text, block.reencoded_text)
E_new <- greedy_cap2_path_then_close(score_new)  # block size >= 3
E_cross <- empty
if keep > 0:
    capacity <- ceil(K_new / keep)
    score_cross <- cosine(block.reencoded_text, M_{t-1}.reencoded_text)
    E_cross <- greedy_one_old_neighbor_per_new_row(score_cross, capacity)
M <- concatenate_frozen_old_and_new(M_{t-1}, block, E_new, E_cross)
assert old_rows_unchanged and old_old_edges_unchanged
return M
```

伪代码覆盖本轮块大小；一行/两行块的边界由实现单独处理，不能形成上述简单环。当前方法没有额外 student 搜索循环，最终评估训练不反向修改 memory。

## 5 机制分析与可证伪边界

### 5.1 凝聚记忆是 student training interface

传统几何凝聚可将目标近似为：合成节点尽量覆盖原始 joint space。本文的核心观察是，持续 TAG 的目标还必须满足：合成节点、文本和拓扑组成的训练接口能够被另一个随机初始化的 student 重新优化。因而候选质量至少包含三部分：表示覆盖、图文对齐和训练可达性。student response 不是最终性能的无偏估计；本轮已经观察到 selection ranking 与 canonical ranking 不完全一致，所以 response 只能作为有风险的选择信号。

### 5.2 旧记忆保持与跨边干扰

frozen-old 保持旧行和旧旧边的 identity，但新跨边改变归一化传播。对称归一化邻接 (\hat{A}=D^{-1/2}(A+I)D^{-1/2}) 中，D 包含自环度；增加跨边会同时改变 D 和旧节点的消息聚合系数。例如旧节点 i、j 原有边的权重由 (d_i d_j)^{-1/2} 变为 ((d_i+\delta_i)(d_j+\delta_j))^{-1/2}，即使该边从未删除，旧传播也发生改变。本阶段接收容量限制了单点新增度，但不直接保证 F1/AUC 改善。

P3-B/C3 的早期诊断未形成稳定替代；最新 T+degree 在 BookChild 两任务上给出条件性正证据，而 arXiv 的退化表明连接均衡并非越强越好。特别地，LP 中拓扑既决定消息传播，也决定合成正边和负边集合；当前差异不能全归因于传播更稳定或孤立点减少。

### 5.3 Budget growth 与信息价值

固定 rate 使 memory 随合法数据增长，避免固定总预算不断降低保留比例；但它没有回答每个新阶段的信息量是否相同。adaptive-rate with response gain 是按 student 边际收益扩容的候选思路，不具有必然改善性能的理论保证。此前短训 gain 在 canonical student 上出现 regret，因此它尚未成为经过验证的默认机制。T+degree 固定 5%，不让该信号控制总预算；后续仍需同时报告 memory size、F1、old retention、new plasticity 和完整成本。

### 5.4 类内子簇复杂度

按类别规模和 dispersion 分配能够处理大类与高离散类，但不能完全表示类内多模态。类内复杂度、多原型与 response 调配此前做过候选诊断，未得到足够稳定的晋升证据。当前每类仍可生成多个聚类合成节点，但未启用额外的子簇复杂度预算器；不能将未来对半径、coverage gap 或 redundancy 的组合设想写成已有效的模块。最新跨任务收益来自固定节点内容后的拓扑替换，不是这些节点算子的升级。

## 6 实验设置

### 6.1 历史小图数据集与协议

| 项目 | Cora | Citeseer |
|---|---:|---:|
| 原始图节点数 | 2708 | 3186 |
| report-dev 节点数 | 271 | 319 |
| 阶段类别 | [0,1]/[2,3]/[4,5,6] | [0,1]/[2,3]/[4,5] |
| 合法 pool 节点 | 501/858/536 | 778/953/498 |
| 累计 memory rows | 25/67/94 | 38/86/111 |
| student steps | 400/1072/1504 | 608/1376/1776 |
| student seeds | 100--105 | 100--105 |

### 6.2 P2 文本矩阵

严格 token 轮共归档 2052 次 student fit，包含 center concat、diversity concat、mean/weighted/set pooling、dual prototype、no-fuse、no-text、DeepSets、rate 2.5% 扩展、排列和 batch 扰动。完整性验收通过，但 `independent_confirmation_pass=[]`，因此文本升级不进入主线。

### 6.3 小图对照组

2026-09-16 修正版包含 dispersion、v4、concat64、random real、k-center、按类 k-center、herding、k-means medoid、recluster、full-history synthetic、raw equal-step、raw 8192-step、selector checkpoint、raw fine-tune、v4-init fine-tune、self-loop synthetic，以及持续 TAGSAM/ASM 适配，共 648 次正式下游训练。这是该批次的历史计数，不是后续所有实验总数。2026-09-17 又补充静态 TAGSAM、STAD 及监督诊断；官方 checkpoint 则只增加推理，不计为训练。

完整 pool 参考每阶段读取所有已见合法原始训练节点及其原文，训练图为这些节点的原始诱导子图，不做凝聚；selector/report/test 节点不进入训练图。`raw_equal` 与合成记忆使用相同 optimizer 步数；`raw_long_selected` 每阶段从头训练 8192 次 optimizer update，每 512 步由 selector-dev 选择检查点，report-dev 不参与选点。因此 8192 不是 epoch，所报结果也不一定来自第 8192 步。两者评估均使用完整真实图，仍未证明长训参考已经充分收敛或最优。

### 6.4 外部方法适配

早期 TAGSAM 持续适配使用官方仓库 commit `d812f06fd16db3cd4d497804a8e504d2c738faa8` 的 `wBCELoss`，保留双层 response matching、可学习 inner graph/text learning rate 和正负概率平衡；使用共同 GCN/MiniLM、dispersion 初始化、冻结旧 synthetic rows、当前 delta teacher、100 外层步和 10 内层步。它是历史受限的机制诊断，不能代表强静态方法。

新增静态 TAGSAM-ASM self-loop/semantic 两臂允许使用全部已见合法 pool 并改写整份 memory，在相同节点预算和共同 student 下评估。尚未覆盖官方完整 NPMI 文本选择和全部构造优化预算，因此仍称“静态适配”。两种拓扑均完整报告，不按数据集挑最高值拼接。

STAD-local-canonical 基于作者仓库 commit `5aaf1a06e44638499373b1358c85d1546187edae`，保留协同自训练、fused logits 聚类、合成均值属性与加权拓扑、关键词和文本候选排序机制；采用本地 Qwen3-32B-AWQ、canonical MiniLM、合法 pool 划分及共同 InfoNCE student。每个阶段可回读累计历史 pool，因此不是持续 append。另将同一最终 STAD 合成图用于原生风格监督 GCN 诊断，但不与主表混排。详细适配和成本见[外部 TAG 基线报告](../current/EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)。

### 6.5 官方 GraphCLIP checkpoint 外部参照

使用已发布 checkpoint 的官方 GPS 图塔（12 层、hidden 1024、输出 384）及其 MiniLM 文本塔，图/文本参数均严格加载并冻结；训练步数为 0，不微调、不搜索 prompt、不选择 checkpoint。对当前 report-dev 节点，从完整原始 TAG 提取完整二跳 ego 图，使用 32 步 random-walk PE 和官方“ego 均值 + 中心节点”读出，与相同类别 prompt 做归一化相似度匹配。不裁剪子图，也不采用旧脚本“大于 256 节点退回一跳”的规则。

这是官方权重在本地推理适配下的测量，不是官方论文协议的完整复现。推理允许完整图的无标签特征和边；预训练语料与目标数据的重合尚未审计。它既不是“在完整 Cora/Citeseer 上重新训练官方模型”，也不是对合成图训练官方模型的实验。其结果仅作为外部模型参照，不进入同 student 的凝聚主表或保留率分母。

### 6.6 最新 T+degree 跨任务确认协议

本轮在结果产生前冻结[确认协议](../current/DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md)，与此前大图筛选分开归档。新顺序必须重新构造 memory，不是仅换 student seed。

| 项目 | arXiv | BookChild |
|---|---|---|
| 类别数 / 五阶段类别数 | 40 / 16,6,6,6,6 | 24 / 10,4,4,3,3 |
| 类别顺序 seeds | 49011、49012 | 49011、49012 |
| 构造 seed / student seeds | 9001 / 210--215 | 9001 / 210--215 |
| LP split seeds | 49101、49102 | 49101、49102 |
| 顺序 49011 累计 rows | 3419,3766,4413,5475,5926 | 1748,1849,2098,2142,2690 |
| 顺序 49012 累计 rows | 2611,3943,5148,5528,5926 | 602,886,1608,1686,2690 |
| 分类 / LP 训练次数 | 180 / 360 | 180 / 360 |

顺序生成对全部类别进行 permutation，再按表中数量分块、块内排序，不固定最大类在 base。三种方法为 C0、T+degree、XT+degree；每个顺序/阶段的 X、文本、mass、depth、stage、行顺序、节点数及无向边数完全匹配，只有边端点不同。“同边数”不表示边集合相同；同一方法/顺序/阶段的 memory SHA256 在两个任务及两个 LP splits 间相同。

分类按第 4.7 节训练 2048 步，以终点模型报告 all-class prompt 的 report-dev Macro-F1，不以 report-dev 选检查点；seen-class 单列。全真实图用于 transductive 推理，含未来节点的无标签 X/边，不报告分类 test 标签。

LP 在合法 pool 内按边端点的到达阶段组合分层，正边 train/val/test=80/10/10，匹配 1:1 负边，负例排除全部观测正边，各划分互斥。推理传播只用已见 pool 和真实 train edges，报告留出 test edges，val 不参与调参；固定 256 步，不选最佳 seed/checkpoint。另加入无需训练的原始 X/text/XT 余弦对照。

每图为 2 orders × 5 stages × 3 arms × 6 seeds = 180 NC，加两个 splits 的 360 LP，共计 1080 fits。两类 student 独立训练，不继承上阶段或另一任务权重。本轮确认的是新的随机配置，不是新数据集或全新节点划分，也未扫描新的 construction seeds。

## 7 实验结果

### 7.1 历史小图最终阶段基线表

下表为同一图文 student 下的 report-dev Macro-F1，均值 ± 6 个 student seeds 的总体标准差。最终阶段 all-class 和 seen-class prompt 口径一致。这是历史 canonical 三阶段对照，不是 T+degree 的新大图主表；其分数和完整数据分母不能挪用到第 7.12 节。

| 方法 | Cora | Citeseer |
|---|---:|---:|
| **冻结 C0，response off** | **49.85 ± 3.00** | **59.17 ± 1.57** |
| 历史 v4，response on | 49.85 ± 3.00 | 60.77 ± 1.64 |
| STAD-local-canonical，静态适配 | 33.99 ± 4.67 | 28.91 ± 2.55 |
| TAGSAM-ASM self-loop，静态适配 | 47.73 ± 4.00 | 51.09 ± 2.85 |
| TAGSAM-ASM semantic，静态适配 | 52.86 ± 3.46 | 42.00 ± 5.85 |
| random real nodes | 43.47 ± 2.48 | 49.79 ± 2.47 |
| k-center | 46.35 ± 4.36 | 55.34 ± 1.11 |
| class-wise k-center | 45.09 ± 2.86 | 53.40 ± 0.82 |
| herding | 37.78 ± 4.36 | 55.85 ± 3.76 |
| k-means medoid | 47.50 ± 2.62 | 49.46 ± 2.49 |
| full-history synthetic | 47.21 ± 2.06 | 48.28 ± 3.48 |
| raw equal-step reference | 47.93 ± 4.10 | 49.89 ± 2.17 |
| raw long, selector checkpoint | 51.31 ± 2.89 | 55.60 ± 1.92 |

除 raw 两行外，上表各方法使用相同阶段节点预算和 canonical student 步数。真实节点方法是 coreset，不是合成凝聚；静态 TAGSAM、STAD 和 full-history synthetic 可回读历史 pool 并改写旧 memory，不满足 C0/v4 的持续约束。统一下游并不消除构造阶段的标签、LLM 和历史访问差异。来源为 [C0 审计](../current/P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)、[五层汇总](../../results/five_level_summary_2026_09_17/summary.csv)与[外部 TAG 汇总（含 STAD）](../../results/external_tag_comparison_2026_09_17/summary.csv)。

以下保留 2026-09-16 已有诊断，不是本轮新增 v4 消融：

| 历史诊断设置 | Cora | Citeseer |
|---|---:|---:|
| dispersion synthetic append | 49.85 ± 3.00 | 59.17 ± 1.57 |
| P2 concat64 control | 50.64 ± 1.45 | 56.03 ± 1.04 |
| continual recluster | 49.70 ± 1.21 | 54.24 ± 1.91 |
| raw long, terminal | 46.26 ± 2.73 | 52.33 ± 1.75 |
| raw delta fine-tune | 48.10 ± 2.78 | 51.13 ± 2.23 |
| v4 init + raw delta fine-tune | 53.41 ± 1.85 | 52.47 ± 1.46 |
| same synthetic, self-loop | 49.81 ± 3.01 | 56.14 ± 1.19 |
| TAGSAM self-loop，早期持续适配 | 39.81 ± 2.26 | 47.56 ± 4.57 |
| TAGSAM semantic，早期持续适配 | 32.84 ± 3.13 | 30.49 ± 4.04 |
| corrected row/column ASM | 42.30 ± 1.83 | 46.42 ± 1.93 |

### 7.2 v4 的跨数据集差异

v4 在 Citeseer 相对 dispersion 提升 1.59 个百分点，但配对 95% 区间为 [-1.51, 4.69]，因此不能称为稳定显著收益。Cora 的 v4 与 dispersion 最终完全相同，说明 response selector 在该数据集最终回退或没有改善基础 allocation。更稳妥的结论是：response selection 有信息，但当前实现尚未证明可跨数据集稳定带来收益。

### 7.3 完整数据和参数继承参考

凝聚有效性的主要证据是固定 student 后，用多少合成节点保留了多少完整训练数据参考性能：

| 指标 | Cora | Citeseer |
|---|---:|---:|
| 最终合成节点 / 完整合法 pool | 94 / 1895 | 111 / 2229 |
| 节点数保留比例 | 4.96% | 4.98% |
| C0 最终 F1 / 完整 pool 长训所选 F1 | 49.85 / 51.31 | 59.17 / 55.60 |
| C0 最终 F1 保留率 | **97.16%** | **106.42%** |
| C0 阶段平均 F1 保留率，全部类别 prompt | **88.71%** | **105.43%** |
| 历史 v4 最终 F1 保留率 | 97.16% | 109.28% |
| 历史 v4 阶段平均 F1 保留率 | 88.71% | 106.31% |

这些比例使用未舍入数值计算；“约 5%”只表示合法 pool 中的节点比例，不表示文本字节、边数或端到端计算成本也是 5%。超过 100% 仅表示超过当前经验完整数据参考，不表示超过理论最优。Cora 的最终保留率较高，仍不能掩盖阶段平均和新类塑性的差距。

Cora 上，raw selector checkpoint 为 51.31，v4-init fine-tune 为 53.41，均值高于 C0/v4；Citeseer 上分别为 55.60 和 52.47，均低于 C0 的 59.17 和 v4 的 60.77。v4-init 仍是历史 v4 初始化，不改名为 C0 微调。这种反向结果说明：

1. 完整 raw 训练不是数学意义上的性能上界；训练步数、图结构、student 初始化和 prompt 竞争仍会影响结果。
2. 参数继承微调可能改善 Cora 的新类塑性，但在 Citeseer 牺牲了旧类保持和最终累计 F1。
3. 不能把 scratch synthetic student 或 delta fine-tune 直接定义为唯一正确的持续下游协议。

### 7.4 真实节点 coreset 与 k-center

Cora canonical 的 k-center 阶段平均 all/seen 为 59.86/69.75，最终为 46.35；Citeseer 最终为 55.34，均低于该单顺序的 C0/v4。但后续[三顺序扩展](../current/SMALL_ORDER_COST_COMPARISON_REPORT_ZH_2026-09-17.md)中，k-center 平均最终为 49.40/57.73，高于历史 v4 的 47.47/55.48。因此不能概括为 k-center 不强，也不能将单顺序优势解释为普遍优于 coreset。该扩展的 v4 结果不改标为 C0。coreset 保留真实 X/文本，仍是重要强效果参照，但不属于 synthetic condensation。

### 7.5 拓扑与响应优化

历史相同 synthetic X/text 的 self-loop 对照在 Cora 最终为 49.81、Citeseer 为 56.14。它在两图早期 seen-prompt 阶段都较强，但没有稳定超过历史 v4。早期持续 TAGSAM semantic 适配和 corrected ASM 未恢复该版本性能；新增的全历史静态 semantic 适配却在 Cora 达到 52.86，比 v4 高 3.01 点，在 Citeseer 为 42.00。Cora 的对应配对 student-seed 95% 区间跨零，因此不宣称稳定胜出，也不能再概括为“TAGSAM 适配全部为负结果”。结构边和 response objective 必须结合历史访问、student、边数和预算解释。

### 7.6 持续阶段、旧类和新类

| 方法 | Cora 阶段平均 all/seen | Citeseer 阶段平均 all/seen |
|---|---:|---:|
| C0 同构造归档 dispersion 参照 | 53.08/60.72 | 64.29/72.33 |
| 历史 v4 | 53.08/60.72 | 64.82/72.85 |
| k-center | 59.86/69.75 | 54.18/65.56 |
| class-wise k-center | 67.13/72.20 | 60.18/67.74 |
| self-loop synthetic | 63.31/73.16 | 63.21/74.39 |
| raw selector checkpoint | 59.84/69.48 | 60.98/68.66 |
| v4 init fine-tune | 56.57/62.56 | 56.10/65.59 |

上表 C0 的 seen-prompt 参照来自历史 dispersion，并非 P5 新增评估。两图共 6 对 memory 完全一致、36 个逐 seed 累计分数一致，但 P5 未保存预测，不能宣称逐节点预测复算相同。

在共同 pooled 口径下，同构造历史 dispersion 的 old/new 为 Cora 65.44/29.06、Citeseer 66.46/44.61；历史 v4 分别为 65.44/29.06、68.00/46.29；raw selector checkpoint 分别为 60.85/38.59、63.90/39.01。这提示 canonical Cora 的新类缺口仍明显。P5 原始 old/new 为 75.05/40.91、76.67/55.48，属于 task-slice，不能与上述 pooled 数值比较或用于相同归因。详见[指标审计](../current/P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。

![](../../results/small_comparison_summary_2026_09_16/final_comparison.png)

图 1：2026-09-16 历史小图对照的最终阶段 Macro-F1。误差条为 6 个配对 student seeds 的标准差；尚不包含新增静态 TAGSAM、STAD 和官方 checkpoint，最新数值以正文表格为准。

![](../../results/small_comparison_summary_2026_09_16/stage_curves.png)

图 2：2026-09-16 历史记忆构造对照的阶段曲线，不包含后来新增的外部方法。阶段曲线用于解释最终分数；类别集合随阶段变化，曲线下降不能全部解释为遗忘。

### 7.7 负结果

P2 的严格 64-token 结果为：Cora center concat 50.64、mean 53.06、weighted 53.28、set 52.95；Citeseer center concat 56.03、mean 57.34、weighted 56.42、set 58.40。独立 seed 后，mean 的阶段平均在 Cora 下降，set 在 Cora 下降，因此没有文本算子满足跨数据集、独立 seeds、阶段保持的晋升门禁。DeepSets 也没有稳定超过 center concat。

这不是“文本无用”的结论：no-text diagnostic 在两图显著下降，说明文本是必要输入；真正的结论是，在固定预算下，简单 center concat 已经是当前最可靠的压缩文本操作，复杂集合编码的额外收益尚未被证明。

### 7.8 官方预训练 GraphCLIP 的直接推理

以下参照使用同一 report-dev 节点集合和类别 prompt，但模型、预训练信息和图读出不同，不是凝聚算法单因素对照。官方 checkpoint 是固定权重的一次评估，不报告虚构的六训练种子标准差。

| 外部模型参照与同 student 参考 | Cora Macro-F1 | Citeseer Macro-F1 |
|---|---:|---:|
| 官方 GPS-12L-1024 + MiniLM checkpoint，0 步训练 | **58.98** | **57.76** |
| 共同 student，完整 pool 等步数训练 | 47.93 ± 4.10 | 49.89 ± 2.17 |
| 共同 student，完整 pool 长训 selector checkpoint | 51.31 ± 2.89 | 55.60 ± 1.92 |
| 冻结 C0，约 5% 合成节点训练共同 student | 49.85 ± 3.00 | 59.17 ± 1.57 |
| 历史 v4，约 5% 合成节点训练共同 student | 49.85 ± 3.00 | 60.77 ± 1.64 |

| 官方 checkpoint 指标或差值 | Cora | Citeseer |
|---|---:|---:|
| Accuracy | 67.53% | 64.58% |
| 正确分类节点数 / 报告节点数 | 183 / 271 | 206 / 319 |
| 历史 v4 减官方 checkpoint，百分点 | -9.13 | +3.01 |
| 完整 pool 长训 student 减官方 checkpoint，百分点 | -7.67 | -2.15 |

Cora 存在较明显的外部模型性能差距，而 Citeseer 的差距较小，v4 均值还高于该 checkpoint；不能据此笼统判断当前 student 在两图都很弱或已达到上限。官方 checkpoint 的逐类 F1 也不均衡，例如 Cora 的 theory 为 0.00，Citeseer 的 artificial intelligence 为 11.76。它说明固定 checkpoint、prompt 和推理适配仍有分类弱项，尚不能将原因定位到某个组件。

本次完整二跳推理没有一跳回退，Cora 最大子图含 426 个节点。两个图均将推理 batch 从 8 改为 1 重测，全部节点预测和阶段指标一致；最大 logit 差低于 `1.5e-7`。图、文本参数和 buffers 的哈希在推理前后不变，最终指标从保存的预测独立复算通过。报告和预测见[官方 checkpoint 证据](../current/OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md)。

旧记录中的 57.47/62.90 来自 `%10==1` 的旧 dev，且存在大子图退回一跳规则，不能与当前 v4 分数直接相减。当前结果也不能支持“约 58 分就是任务水位”的说法。阶段切片仅是同一个冻结模型在不同已见类别集合上的评分，不是持续训练结果或遗忘测量。本次没有在完整 TAG 上重新训练官方 GraphCLIP。

### 7.9 STAD 的任务迁移与监督诊断

STAD-local-canonical 在共同图文 student 上为 33.99/28.91；这是该静态 TAG 凝聚适配对当前任务的迁移结果，不是对原生监督任务的否定。为检查这一点，在不改动最终合成图的前提下，另用两层 GCN64、dropout 0.5、600 轮监督训练，selector-dev accuracy 选点，seeds 100..105 评估：

| 独立监督任务中的配对参照 | Cora Macro-F1 | Citeseer Macro-F1 |
|---|---:|---:|
| STAD 合成图，监督 GCN64 | 73.63 ± 2.42 | 67.46 ± 0.63 |
| 完整合法 pool，同一 GCN64/600 轮 | 84.54 ± 0.75 | 69.72 ± 0.79 |
| STAD / 对应监督完整数据参考 | 87.10% | 96.77% |

STAD 使用构造器产生的合成伪标签，raw 使用合法 pool 真标签，标签信号并不相同；本表尚未包含同设置下的 v4，所以不能判断两者的监督凝聚优劣。低图文迁移分数与较高监督效用并存，进一步说明凝聚效用依赖下游任务接口，不能只看不同模型间的绝对分数。

此前有限网格搜索的完整图监督 GCN384 参考为 84.10 ± 0.33 / 69.26 ± 0.82，属于另一组设置，不能与本表拼接为统一上界。两组监督结果均改变了损失、输出头或训练传播及检查点规则；它们只说明分类任务还有经验性能空间，不能把与 v4 的全部差值解释成 InfoNCE 损失或凝聚损失。

### 7.10 P5：压缩率、长链、顺序与大图外推

P5 使用冻结 C0（response off）补充了 rate curve、5-stage/base+4、新 frequency 类别顺序诊断和大图长链。新顺序不是预注册独立确认。小图最终结果如下：

| 数据集 | 1% | 5% | 10% |
|---|---:|---:|---:|
| Cora | 42.00 | **49.85** | 45.38 |
| Citeseer | 52.20 | **59.17** | 55.82 |

5% 是当前两个小图较稳的工作点，但 rate 曲线非单调，不能把更大的 memory 直接解释为更强方法。5-stage/base+4 的最终累计 F1 为 Cora 40.71、Citeseer 54.78；一组新的 frequency 类别分组最终为 Cora 51.61、Citeseer 52.79，而 canonical 对照为 49.85/59.17，说明类别到达顺序会显著改变结果。

在 arXiv、BookChild、SportsFit 的 5-stage/base+4 大图首轮中，C0 最终累计 F1 分别为 18.82、13.22、27.90，最终 task-slice old/new F1 分别为 33.63/21.30、30.86/44.54、49.47/58.42。切片指标排除了跨任务误报，不能作为累计分数的 old/new 分解，尤其不能据 BookChild new=44.54 就断言新类已学好。三者均可完成 frozen-old append，但长链累计表现存在回落，且阶段类别集合变化，不能将回落直接全部归因于遗忘。结果支持“协议可运行和边界可测量”，不支持“fixed-rate 在所有长链上稳定”。

大图每个数据集执行 5 个阶段、6 个 student seeds、每阶段最多 2048 步，共 61,440 个 student optimizer steps。student 重训约 146.80--159.44 秒，阶段构造约 31.21--61.16 秒；全池 MiniLM 编码在旧运行中未单独计入阶段构造时间，新 runner 已增加 `text_embedding_seconds` 字段，但不能补回旧运行耗时。缺失字段按未知而非零处理；当前成本是已记录部分的下界，不是端到端计算优于完整图的证据。

P5 同时暴露了一个比较口径问题：此前五层表中的 `v4` 含 response 分配，不能直接当作当前 C0。当前 C0 相对于全历史启发式重凝聚在 Cora/Citeseer 分别高 2.64/10.89 个百分点，相对于旧 synthetic + 当前 delta 重聚类高 0.15/4.93 个百分点；Cora 仍低于完整 raw 长训最佳检查点和静态 TAGSAM semantic 适配，Citeseer 则高于这些已审计参照。该结果支持“有效但有边界”的定位，不支持普遍 SOTA。

### 7.11 从保持约束、候选选择到跨任务拓扑

以下是候选形成过程，不与新确认混为一个实验。不同轮次的顺序、seed、split 和核不同，不能将最好值拼成一条虚构主方法曲线。

| 轮次 | 实际发现 | 如何使用证据 |
|---|---|---|
| v5 保持约束追加 | 288 fits；core/topology/full 的四条件平均最终 F1 差分别 +0.30/-2.36/-2.91 pp | 三候选均未晋升；小图门槛未过，条件性大图扩展未启动 |
| v6-A 局部 merge/split 边际编辑 | 560 fits；8 个 cell 的 local 选择全回退，平均排序 Spearman -0.018 | 局部编辑及昂贵短训排序未形成稳定收益，未进入递归 v6-B |
| v7-A 梯度兼容性预筛选 | 8 个 cell 全回退，Spearman 0.125；成本均值 1.163× | 成本改善但排序失败；多原型 stage 1 局部高收益不能拼为持续轨迹 |
| 原始 X+T+graph-signature 联合核分类 | 36 条件平均最终 F1 +0.952 pp，20/36 为正，最差 -6.408 pp | 平均有收益但退化过大，分类晋升未通过 |
| Safe joint-kernel V2 开发 | 16 条件 coverage 平均 +2.77、最差 -4.13；degree 平均 +2.26、最差 -5.04 pp | 两者均未通过；不能因均值为正直接替换 C0 |
| 原始联合核 coverage 的小图 LP | Cora AUC/AP +1.68/+2.47；Citeseer +3.33/+3.26 pp | 开始出现结构任务收益，但此时仍不是 T-only |
| 小图 LP 核与约束拆解 | T+degree 的 Cora AUC/AP 为 87.17/85.12，Citeseer 为 90.66/89.04 | graph signature 非必要；沿用旧 splits，属于开发诊断 |
| 大图 LP 首轮筛选 | T+degree 的 arXiv 97.742/97.579；BookChild 93.329/92.832 | 相对该轮 C0 分别 +0.058/+0.003 与 +1.268/+1.749 pp，随后冻结新确认协议 |

这些结果不等于证明所有候选没有潜力：例如 v6 的整类重聚类控制和 v7 的多原型有局部正收益，但选择规则未可靠识别收益，且单阶段反事实不等于递归持续轨迹。原始联合核、V2、小图 LP 机制与大图筛选也分别归档，训练计数不能替代独立任务数量。尤其 V2 的 signature 存储开销不能记到无 signature 的 T+degree 上。最新确认不沿用筛选轮绝对分数，其结果如下。

### 7.12 最新确认：节点分类与链路预测

所有值为百分数，差值为百分点。分类最终值平均 2 orders × 6 seeds，LP 平均 2 orders × 2 splits × 6 seeds；阶段均值对五阶段等权。不挑最佳 seed 或事后切换默认候选。

| 数据集 | 方法 | 最终分类 F1 | 阶段平均 F1 | 最终旧类 F1 | 最终新类 F1 |
|---|---|---:|---:|---:|---:|
| arXiv | C0 | 21.101 | 21.761 | 21.210 | 20.483 |
| arXiv | T+degree | 20.439 | 21.553 | 20.626 | 19.382 |
| arXiv | XT+degree | 20.122 | 20.982 | 20.380 | 18.660 |
| BookChild | C0 | 13.054 | 15.596 | 13.497 | 9.956 |
| BookChild | T+degree | **14.600** | **16.844** | **14.944** | **12.193** |
| BookChild | XT+degree | 14.133 | 16.145 | 14.588 | 10.945 |

旧/新类按 global-support 计算。BookChild 的最终分类 +1.546 pp、阶段均值 +1.248 pp，旧新类同时改善；arXiv 最终 -0.662 pp、阶段均值 -0.208 pp，新类约 -1.10 pp。分类绝对分数仍不高，不能仅凭相对提升宣称强分类性能。

| 数据集 | 方法 | 最终 LP AUC | 最终 LP AP | 阶段平均 AUC | 阶段平均 AP |
|---|---|---:|---:|---:|---:|
| arXiv | C0 | 96.043 | 95.762 | 96.487 | 96.278 |
| arXiv | T+degree | 96.033 | 95.712 | 96.500 | 96.245 |
| arXiv | XT+degree | 96.044 | 95.720 | 96.453 | 96.198 |
| BookChild | C0 | 92.204 | 91.365 | 90.171 | 88.789 |
| BookChild | T+degree | 92.928 | 92.345 | 91.774 | 90.994 |
| BookChild | XT+degree | **93.203** | **92.567** | **92.329** | **91.564** |

T+degree 在 BookChild 最终 AUC/AP +0.723/+0.980 pp，阶段均值 +1.603/+2.205 pp；arXiv 最终 -0.011/-0.050 pp，平均接近持平。差值按未舍入值计算。XT 在 BookChild LP 上更强但分类弱于 T，不能事后按任务选择 T/XT 并报告为同一个默认方法。

### 7.13 顺序依赖与零训练强对照

下表为 T+degree 减 C0，LP 在每个顺序内平均两个 splits：

| 数据集 | 顺序 | 最终分类 ΔF1 | 阶段分类 ΔF1 | 最终 LP ΔAUC / ΔAP | 阶段 LP ΔAUC / ΔAP |
|---|---:|---:|---:|---:|---:|
| arXiv | 49011 | -0.813 | -0.235 | -0.327 / -0.454 | -0.208 / -0.309 |
| arXiv | 49012 | -0.511 | -0.181 | +0.306 / +0.355 | +0.232 / +0.244 |
| BookChild | 49011 | +2.613 | +1.770 | +0.856 / +1.188 | +1.864 / +2.380 |
| BookChild | 49012 | +0.478 | +0.726 | +0.590 / +0.773 | +1.342 / +2.031 |

arXiv 并非每个条件都不退化：49011/split49101 的最终 AP 差为 -0.578 pp，超过 -0.5 pp；预定门槛针对的是每个顺序平均两个 splits 后的差值，不能改写成每个 split 都通过。最终 LP 逐 seed 配对胜率为 BookChild AUC 21/24、AP 23/24，arXiv 均为 9/24；这些不是 24 个独立任务。

| 数据集 | 零训练余弦输入 | 最终 AUC | 最终 AP |
|---|---|---:|---:|
| arXiv | X | 96.822 | 96.843 |
| arXiv | text | 96.881 | 96.906 |
| arXiv | XT | **96.895** | **96.920** |
| BookChild | X | 79.423 | 81.755 |
| BookChild | text | **80.337** | **82.730** |
| BookChild | XT | 80.111 | 82.500 |

arXiv 上原始表征的直接余弦已超过本轮所有 GAE，约 96% AUC 不能单独证明凝聚学到更强结构。BookChild 的 GAE 超过余弦，但 T 相对 C0 的净收益仍是 +0.723/+0.980 pp，不能把 GAE 与余弦约 10 pp 的差距计为新拓扑增益。随机负边的容易程度及上游特征也限制 LP 分数的外推。

### 7.14 拓扑、成本与验收

| 数据集 / 顺序 | 最终 rows / 无向边数（三臂相同） | C0 孤立点 / 最大度 | T+degree 孤立点 / 最大度 | memory 文件大小（三臂近似相同） |
|---|---:|---:|---:|---:|
| arXiv / 49011 | 5926 / 8433 | 978 / 32 | 0 / 6 | 19.613 MiB |
| arXiv / 49012 | 5926 / 9241 | 612 / 58 | 0 / 6 | 19.638 MiB |
| BookChild / 49011 | 2690 / 3632 | 880 / 88 | 0 / 6 | 8.897 MiB |
| BookChild / 49012 | 2690 / 4778 | 161 / 82 | 0 / 6 | 8.931 MiB |

这些结果确认收益并非增加节点或边数造成，也显示了连接覆盖和集中度变化；但内部环与跨块容量同时修改，孤立点减少不能单独视为已证明的因果解释。表中最大度 6 是本轮实际结果，不是算法对任意阶段流的全局定理。

| 数据集 | C0 五阶段构造秒 | T+degree 五阶段构造秒 | 构造倍率 | NC 每 fit 秒 C0 / T | LP 每 fit 秒 C0 / T |
|---|---:|---:|---:|---:|---:|
| arXiv | 77.076 | 109.306 | 1.42× | 4.967 / 4.934 | 0.641 / 0.642 |
| BookChild | 34.852 | 39.922 | 1.15× | 4.583 / 4.537 | 0.581 / 0.582 |

构造成本包含一条五阶段链的共同 X/text 构造与各自拓扑，再平均两个顺序；共同构造虽在实验中复用，仍给每个独立方法计一次。未包含复用缓存的原始文本编码、数据加载、最终推理及独立审计，所以不是端到端总成本。新增代价集中在构图，没有 response 搜索训练；每 fit 的小差异不足以宣称加速。进程 GPU 峰值还包含 encoder/推理缓冲区，不能等同于隔离的 synthetic 训练显存。

**事前推进标准与验收。** T+degree 满足：BookChild 每顺序最终及阶段 LP AUC/AP 均为正增益；arXiv 每顺序对应 LP 差值均不低于 -0.5 pp；分类每数据集平均最终/阶段差值不低于 -1 pp、每顺序不低于 -2 pp、最终旧类每顺序不低于 -2 pp；工件审计通过。它们是研究推进容忍值，不是统计非劣性检验。

1080/1080 fits、1080/1080 checkpoint 重载及 60/60 memory 审计通过，9 项相关单元测试通过；本地独立核算核对 624 个汇总单元。分类重载最大 F1 差为 0，LP 最大指标差在 arXiv/BookChild 分别为 5.71e-7/3.95e-6（0--1 单位），没有超出预定重载容差的告警。审计没有逐簇独立重凝聚全部 X/text，也不等于操作系统级历史访问隔离或上游特征无泄漏证明。

完整证据见[确认结果报告](../current/DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md)、[原始汇总](../../results/degree_confirmation_20260920/overview.json)和[配对 CSV](../../results/degree_confirmation_20260920/comparison_rows.csv)。本轮未新跑同顺序 full-raw、k-center/medoid 或静态 TAG 强基线，因此尚不能给 T+degree 报告新协议下的完整图性能保留率。

## 8 讨论

### 8.1 当前主线与冻结对照

基于最新确认，后续主方法基础升级为 T+degree，C0 保留为冻结对照：

> 5% growing memory + dispersion allocation + mean-X + center concat/re-encode + 文本引导块内环 + 容量受限跨块边 + frozen-old，response off。

这不是将全部优化模块叠加，也不是宣称 T 在所有条件下最优。它在 BookChild 两个新顺序、两种下游任务上改善，在 arXiv 的退化落于预定研究容忍范围；相对成本可量化，因而值得作为新基础继续推进。C0 的历史小图有效性和外部基线比较仍成立，但不能追溯记为 T 的成绩。

历史 v4 response-on、联合核 coverage、XT+degree 均保留原名。response、extra-budget、复杂度分配、refresh 和注意力不属于新默认，不能从拓扑的成功倒推出此前失败模块有效。

### 8.2 解决了哪些 challenge

| Challenge | 当前处理 | 当前证据边界 |
|---|---|---|
| 历史 raw 不可回读 | 合成 memory 跨阶段 append | 访问声明已审计，完整系统级隔离仍需加强 |
| 固定每类 rows 不合理 | sqrt(mass) × dispersion 分配 | 比例启发式，非通用最优 |
| 文本和结构压缩错配 | joint clustering + center text re-encode | 当前采用的可靠控制，未证明普适最优 |
| 旧拓扑被重写 | frozen-old old-old edge identity | 新跨边仍改变传播，不保证免遗忘 |
| 边集中与节点覆盖不足 | 文本引导内部环、跨块接收容量 | 本轮同边数消除孤立点，但两算子尚未因果拆分 |
| 几何 proxy 与 student 性能脱钩 | 历史 short student response；当前跨任务直接检验 | selector 排序有限，C0/T 均关闭 |
| 新类塑性与旧类保持冲突 | growing budget、global-support old/new 分开诊断 | BookChild 两者改善，arXiv 两者下降 |
| 结果受协议影响 | 新顺序/seed/split、同 memory、hash 与重载审计 | 大图已补 global-support 与分项成本，完整成本及强基线仍缺 |

### 8.3 为什么不能宣称 SOTA

证据已从 Cora/Citeseer 扩展到 arXiv/BookChild 的新顺序跨任务确认，但 T 的最新配对表只有 C0 和 XT，缺同协议强 coreset、完整图 student 和静态 TAG 强基线。分类仍为 report-dev、transductive 推理，历史项目存在早期 test 开发访问；LP 虽报告留出 test edges，后续据此调参即成为开发条件。两数据集、两顺序和固定构造 seed 不能支持普适结论。

TAGSAM/STAD 是明确标注的适配，GraphCLIP 仅做 checkpoint 推理且预训练重合未审计；PUMA/CaT/GECC 的任务定义不同。环、容量约束和文本相似度本身也不是新发明，尚需说明其在持续合成拓扑上的必要性、相关文献差异及可验证分析。因此“通过本轮推进标准”不等于已确立 WWW 长论文的新颖性或 SOTA。

### 8.4 Student 容量、训练目标与凝聚效果

共同 student 的双向 InfoNCE 优化实例图文对应，而非类别监督交叉熵。同类节点可能在 batch 中互为负样本，训练节点文本与推理类别 prompt 也存在分布差异；这些是合理的待验证机制，不是本次纯推理实验已经证明的性能瓶颈。两层 GCN、预训练、文本投影、图上下文和训练目标尚未被单独控制，不能仅凭分数将问题归因于“模型太浅”或“InfoNCE 不适合分类”。

本文优先衡量固定 student 后的凝聚效用，因此历史保留率以相同 student 的完整合法 pool 训练为分母。C0 的 97.16%/106.42%（历史 v4 为 97.16%/109.28%）仅适用于相应小图经验参考，不保证任意更强模型仍成立，也不用于最新 T 的分母。官方 checkpoint 只校准外部模型水平；若校准共同 student，应对所有凝聚方法统一重测，不能只给某方法更换模型。

最新 LP 是同一 memory 转交另一模型的效用检验，并非 NC 模型直接零训练迁移。BookChild 的收益说明合成拓扑不只影响类别 prompt；arXiv 余弦强于 GAE 又说明高 AUC 可能主要来自冻结表征。下一步应先用匹配 full-raw/coreset 确定性能位置，再分离传播和正负边监督的作用。

## 9 局限性与后续工作

P1-X、P1-B、P2-A/B、P3-A/B、C1/C3、P4 和 P5 早期探索没有产生稳定默认升级；后续联合核分类仍有明显失败条件，但 LP 拆解最终形成 T+degree，并通过第 7.14 节的新确认。不能继续概括为“所有优化都失败”，也不能将最新整体收益归功于尚未拆解的单个算子。接下来按以下顺序推进，以下任务尚未执行：

1. **拆分两个拓扑算子。** 固定 T、节点内容、行序和边数，做“C0/环形内部边 × C0/容量跨边”的 2×2；加入固定度序列端点打乱，记录语义质量、旧归一化传播块漂移、NC old/new 和 LP old/new/cross。先判断覆盖、语义及跨块负载各起什么作用，再改算子。
2. **诊断 arXiv 退化。** 检查强制闭环的低相似边、跨类混合和跨块传播，区分 LP 训练边改变与传播改变。当前只是待验证解释；若用此次确认条件优化，它们转为开发条件，另留新顺序/seeds/splits 确认，不能重复测试后称独立泛化。
3. **补新主线的效用定位。** 固定 T 后做 1%/5%/10%，配套同顺序同 student 的 k-center/medoid、完整合法图训练及全历史重凝聚；NC/LP 各用自己的匹配参考，报告最终/阶段性能、保留率与训练成本。旧小图五层表不替代这一步。
4. **补推广与成本。** 增加 construction seed、更多顺序和适合的第三数据集（SportsFit 旧 C0 已有，T 尚未完成本轮同级确认），必要时延长任务链；补缓存编码、构造、训练、推理和存储的完整账，不把缺失时间记为零。当前全对候选评分/排序的成本也需评估可扩展性。
5. **完成创新论证。** 文献核验度约束、结构覆盖和持续追加的已有做法；分析每阶段接收容量对旧归一化传播扰动的约束及适用假设，不从算子界直接推出 F1/AUC 提升。将正结果与适用边界共同组织为论文，不以协议工程替代算法贡献。

外部 TAGSAM/STAD 补强按此前决定暂停；全局 response、全局梯度/特征匹配等设想仍不能标为完成。注意力、动态总预算和 old-row refresh 不在本轮重启；若重新加入，应有明确新假设及单项证据，避免把此前失败模块直接组合。官方 GraphCLIP 仍只有 checkpoint 推理，没有重新训练。

## 10 结论

本文构建并审计持续 TAG 合成追加框架，以随合法数据增长的预算、类内联合聚类、mean-X 和文本重编码形成可重训 memory。历史 C0 在 Cora/Citeseer 约 5% 节点预算下取得 49.85/59.17 Macro-F1，保留同 student 完整 pool 长训参考的 97.16%/106.42%；这些是特定经验参考下的有效性证据，不是理论上界或新版本的成绩。

最新 T+degree 在不改变节点内容和边数量的情况下，以文本引导内部环和跨块接收容量替换 C0 选边规则。在 arXiv/BookChild 的两个新顺序中完成 1080 次训练和全量重载。BookChild 最终分类 +1.546 pp、LP AUC/AP +0.723/+0.980 pp；arXiv 分别 -0.662 pp、-0.011/-0.050 pp，并存在顺序依赖。构造成本约 1.42/1.15 倍、student 训练基本不变，支持以 T+degree 作为下一轮主线基础、C0 作为冻结对照。

论文可以围绕“历史不可回读下的合成拓扑可重训效用、度与容量控制、跨任务收益及边界”组织，而不是只报告复杂模块的负结果。但两个拓扑约束的独立贡献、文献新颖性、新协议强基线、完整图保留率和端到端成本仍未补齐；arXiv 直接余弦超过 GAE 也限制结构收益的解释。当前不宣称 SOTA、普遍优于完整数据或已达到 WWW 长论文的充分贡献标准。

## 可复现性声明

历史 audited small-comparison 保存了逐 seed checkpoint、memory artifact、selector 统计和验收 JSON，完整 checkpoint 主要保留在远程；P5 unified runner 只保存 memory 和标量结果，未保存 student checkpoint/逐节点预测，不能声称其预测可直接重放。当前小图汇总位于 `results/small_comparison_summary_2026_09_16/` 与 `results/p5_rate_curve_small_20260919/`。新指标审计 `results/p5_metric_scope_audit_20260919/` 的 1,558 项检查全部通过，但范围是 artifact 相等与聚合复核，不是重做历史模型评估。P5 总报告位于 `docs/current/P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md`。运行环境为远程 conda `tagsam`，PyTorch 2.4.1+cu121，双 RTX 4090。

2026-09-17 外部方法表与逐 seed 数据位于 `results/external_tag_comparison_2026_09_17/`，STAD 监督诊断位于 `results/stad_supervised_sanity_2026_09_17/`。官方 checkpoint 推理证据位于 `results/official_graphclip_report_2026_09_17/`，包括两个数据集的逐节点 logits、标签、预测、子图规模、源码及输入 SHA256，以及 batch=1 复测。图/文本权重严格加载、冻结状态校验和指标复算已完成；相应 7 项本地测试通过。checkpoint SHA256 为 `e93b19a565446c3d62f53eb31fee08570f94c558de216e1aec805d05532721aa`。本次论文更新不产生新的训练结果。

最新确认的本地文本证据位于 `results/degree_confirmation_20260920/`，含每图 `results.json`、`audit.json`、`summary.json`、manifest、日志及逐任务预测核算记录；全量 `.pt` memory/checkpoints 主要在远程 `/home/pc/research/TAG_condensation/results/degree_confirmation_20260920/{arxiv,bookchild}`，不声称已下载全部模型。冻结源码在远程 `/home/pc/research/TAG_condensation/work_runs/degree_confirmation_20260920_snapshot`。文本证据归档 SHA256 为 `117599c2d1abb7b7a15965b71a7a71c451c0192acbd086a6a9e9c84d4f4892ac`。

运行器为 `code/v2_llm_fusion/run_degree_cross_task_confirmation.py`，独立审计为 `audit_degree_confirmation.py`，聚合为 `summarize_degree_confirmation.py`，本地核算为 `tools/report_degree_confirmation.py`。最新确认保存模型和预测并完成重载，不能据此倒推旧 P5 也具备同样工件。跨任务 memory 相同、合法 source IDs 和旧块冻结已审计，但上游特征预处理与系统级历史隔离仍是边界。

## 参考文献（草稿）

[1] GraphCLIP: Enhancing Transferability in Graph Foundation Models.

[2] TAGSAM: Text-Attributed Graph Condensation. arXiv:2606.03839, 2026.

[3] Jin, W. et al. GCond: Gradient Matching for Condensing Graphs. ICLR, 2022. arXiv:2110.07580.

[4] SFGC: Simple and Fast Graph Condensation. arXiv:2306.02664, 2023.

[5] PUMA: Efficient Continual Graph Learning with Graph Condensation. arXiv:2312.14439, 2023.

[6] CaT: Balanced Continual Graph Learning with Graph Condensation. arXiv:2309.09455, 2023.

[7] OpenGC: Open Graph Condensation. arXiv:2405.17003, 2024.

[8] GECC: Continual Graph Condensation for Evolving Graphs. arXiv:2502.17614, 2025.

[9] GCondenser: A Graph Condensation Benchmark and Toolkit. arXiv:2405.14246, 2024.

[10] Devlin, J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.

[11] Reimers, N. and Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP, 2019.

[12] STAD: Semi-Supervised Text-Attributed Graph Distillation. arXiv:2607.20477, 2026. [论文](https://arxiv.org/abs/2607.20477)，[作者代码](https://github.com/laiyurui/STAD-Semi-Supervised-Text-Attributed-Graph-Distillation)。本文使用明确标注的本地适配，而非官方分数复现。

## 附录 A：分版本配置卡

**A.1 历史小图与外部模型参照**

| 配置 | 值 |
|---|---|
| Protocol | tagsam-unified-v3/v4 audited small-graph protocol |
| Rate | 0.05 cumulative pool |
| Text encoder | frozen MiniLM, 384 dimensions |
| Student | two-layer GCN, hidden/output 384 |
| Student loss | bidirectional graph-text InfoNCE |
| Optimizer | Adam, lr 1e-3 |
| Batch | 16 |
| Student seeds | 100, 101, 102, 103, 104, 105 |
| Selector seeds | C0 不使用；历史 v4 为 300, 301, 302 |
| Text recipe | center concat, g=5, legacy ~320 character budget |
| Topology | frozen-old, semantic new-new, cross-degree 1 |
| Node-classification test labels | not evaluated in these tables |
| Evaluation graph | full real transductive graph |
| Full-pool reference | same student, induced legal-pool graph, 8192 updates, selector every 512 |
| Official checkpoint reference | frozen GPS-12L-1024 + MiniLM; 0 training updates; separate table |
| Official inference adapter | complete 2-hop ego, RW-PE(32), no subgraph cap |
| Report-dev | original_node_id % 10 == 3; 271 Cora / 319 Citeseer nodes |

**A.2 最新 T+degree 跨任务确认**

| 配置 | 值 |
|---|---|
| 数据集与链长 | arXiv / BookChild；base+4 |
| 方法 | C0、T+degree、XT+degree；默认候选为 T |
| 构造 | dispersion、mean-X、joint_g5_c320 concat/re-encode |
| 总预算 | floor(0.05 × 累计合法 pool)；旧行不裁剪 |
| T 拓扑 | 文本余弦排序；内部 cap-2 路径闭环；每新节点一跨边、旧接收容量 ceil(new/old) |
| XT 评分 | X 权重 4/7、T 权重 3/7；无 graph signature |
| 关闭的模块 | response、extra-budget、refresh、coverage penalty、可训练 attention |
| 类别顺序 / 构造 seed | 49011、49012 / 9001 |
| Student seeds / LP split seeds | 210--215 / 49101、49102 |
| NC | GCN 384/384、双向 InfoNCE、Adam 0.001、batch 16、2048 steps |
| LP | GAE GCN 384/128、内积解码、Adam 0.001、正负边 batch 各 ≤256、256 steps |
| NC 推理 / 报告 | 完整真实图 transductive / report-dev all-class F1、global-support old/new |
| LP 推理 / 报告 | 已见 pool + real train edges / held-out test AUC/AP；val 不调参 |
| Memory 复用 | 同一 arm/order/stage 的 SHA256 在 NC/LP 与 LP splits 间相同 |
| 训练与审计 | 1080 fits、1080 checkpoint reloads、60 memories、624 汇总核对 |
| 新协议 full-raw 性能保留率 | 尚未测量，不沿用旧顺序分母 |

## 附录 B：P2 文本消融摘要

| Operator | Cora final | Citeseer final |
|---|---:|---:|
| center concat, strict 64 tokens | 50.64 | 56.03 |
| mean pooling | 53.06 | 57.34 |
| weighted pooling | 53.28 | 56.42 |
| set attention pooling | 52.95 | 58.40 |
| diversity concat, lambda=.75 | 52.20 | 56.82 |
| no-fuse | 41.77 | 54.52 |
| no-text | 4.30 | 5.49 |

这些数值是严格 token 控制下的 text-only intervention，不是完整重新运行 v4 response allocation 的替代主方法。

## 附录 C：审计与结果入口

- P2 收尾结果：`docs/current/P2_CLOSEOUT_SMALL_BASELINE_RESULTS_ZH_2026-09-16.md`
- P2 验收：`results/p2_acceptance_2026_09_16/acceptance.json`
- 小图汇总：`results/small_comparison_summary_2026_09_16/summary.json`
- 小图验收：`results/small_comparison_summary_2026_09_16/acceptance.json`
- P5 指标与版本审计：[报告](../current/P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)、[验收 JSON](../../results/p5_metric_scope_audit_20260919/audit.json)
- 最终对比图：`results/small_comparison_summary_2026_09_16/final_comparison.png`
- 阶段曲线：`results/small_comparison_summary_2026_09_16/stage_curves.png`
- 修正版 runner：`code/v2_llm_fusion/run_small_comparison_audited.py`
- TAGSAM 适配 runner：`code/v2_llm_fusion/run_tagsam_graphclip_comparison.py`
- 最新外部 TAG 比较：[报告](../current/EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)、[数值表](../../results/external_tag_comparison_2026_09_17/summary.csv)
- 官方 checkpoint 当前报告划分：[报告](../current/OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md)、[Cora 预测与指标](../../results/official_graphclip_report_2026_09_17/cora.json)、[Citeseer 预测与指标](../../results/official_graphclip_report_2026_09_17/citeseer.json)
- 官方推理代码：[eval_official_graphclip_report.py](../../code/v2_llm_fusion/eval_official_graphclip_report.py)
- 官方推理验收：[test_official_graphclip_report.py](../../code/v2_llm_fusion/test_official_graphclip_report.py)
- STAD 独立监督诊断：`results/stad_supervised_sanity_2026_09_17/`
- 保持与选择分支：[v5](../current/V5_RETENTION_EXPERIMENT_COMPLETION_ZH_2026-09-19.md)、[v6-A](../current/V6A_LOCAL_MARGINAL_COMPLETION_ZH_2026-09-19.md)、[v7-A](../current/V7A_GRADIENT_COMPATIBLE_APPEND_COMPLETION_ZH_2026-09-19.md)
- 原始联合核分类：[第一轮报告](../current/PHASE12_C0_JOINT_KERNEL_COMPLETION_ZH_2026-09-20.md)、[V2 报告](../current/SAFE_JOINT_KERNEL_V2_COMPLETION_ZH_2026-09-20.md)
- 小图 LP：[初始比较](../current/LINK_PREDICTION_COMPARISON_ZH_2026-09-20.md)、[核与拓扑机制](../current/LP_KERNEL_MECHANISM_RESULTS_ZH_2026-09-20.md)
- 大图 LP 开发筛选：[报告](../current/LARGE_LP_TOPOLOGY_VALIDATION_ZH_2026-09-20.md)
- 最新 T+degree：[冻结协议](../current/DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md)、[确认报告](../current/DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md)
- 最新配对表：[overview](../../results/degree_confirmation_20260920/overview.json)、[CSV](../../results/degree_confirmation_20260920/comparison_rows.csv)、[全表](../../results/degree_confirmation_20260920/comparison_tables.md)
- arXiv 确认：[逐 fit](../../results/degree_confirmation_20260920/arxiv/results.json)、[审计](../../results/degree_confirmation_20260920/arxiv/audit.json)、[汇总](../../results/degree_confirmation_20260920/arxiv/summary.json)
- BookChild 确认：[逐 fit](../../results/degree_confirmation_20260920/bookchild/results.json)、[审计](../../results/degree_confirmation_20260920/bookchild/audit.json)、[汇总](../../results/degree_confirmation_20260920/bookchild/summary.json)
- T+degree 实现：[runner](../../code/v2_llm_fusion/run_degree_cross_task_confirmation.py)、[块内环](../../code/v2_llm_fusion/large_lp_support.py)、[核与跨块容量](../../code/v2_llm_fusion/safe_joint_kernel_append.py)

## 附录 D：大语言模型使用声明

大语言模型用于整理实验记录、辅助代码审查、统一术语、生成论文草稿初稿和检查表格引用。研究问题、协议设计、实验执行、数值核验、失败分析、文献取舍和最终结论由作者审核并承担责任。模型没有被列为作者，也没有替代实验测量或统计判断。

此外，外部 STAD-local-canonical 基线在算法构造阶段使用本地 Qwen3-32B-AWQ 生成合成文本候选，与论文写作辅助是不同用途；该使用方式及成本已在外部基线报告中披露。历史 v4、冻结 C0 和当前 T+degree 都使用冻结 MiniLM 与代表文本拼接，不以该生成模型构造主方法文本。
