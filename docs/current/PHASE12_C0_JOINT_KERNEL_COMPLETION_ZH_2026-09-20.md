# C0 基线底座与块保持联合核：两阶段完成报告

日期：2026-09-20。对应本次批准的“阶段一：冻结 C0、完成基线底座”和“阶段二：只验证块保持联合核追加”，不沿用历史文本 P2 的编号。

**两阶段已完成，实验验收通过；联合核的性能晋升门禁未通过，继续保留 C0。** 完成不等于候选有效。本轮没有把联合核替换成主方法，也没有启动大图、response selector、动态预算、旧行 refresh 或注意力组合。

## 1. 完成范围与决定

| 项目 | 完成情况 |
|---|---|
| 阶段一基线表 | 两小图、canonical 三阶段、约 5% 节点预算；14 种方法或参考，包含新增 random synthetic、持续 coreset、recluster、全历史重凝聚、静态 TAGSAM 适配与完整数据训练 |
| 阶段二矩阵 | 2 数据集 × 3 类别顺序 × 3 压缩率 × 2 阶段数 = 36 个条件；每条件 C0/k-center/recluster/joint-kernel 四臂、6 个 report seeds |
| 新增构造与训练 | 矩阵 576 份阶段 memory、3456 次 student fit；random synthetic 另有 6 份 memory、36 次 fit；合计 582 份 memory、3492 次 fit |
| 独立模型验收 | 两数据集各 1746 次 checkpoint 重载重评，合计 3492 次，0 error；不是仅检查 JSON 是否存在 |
| 历史基线来源验收 | 468 对 checkpoint/support 检查，0 error；复用此前 108 次静态/监督扩展模型重评，不计为本轮新增训练 |
| 候选晋升 | 20/36 个条件最终 F1 改善，均值 +0.9516 pp，最差 -6.4082 pp；未通过事前门槛 |
| 主方法 | C0 保持冻结；旧 v4(response on)、v5/v6/v7 候选及本轮联合核均单列 |

阶段一不是将 14 种方法扩展到全部 36 个条件。完整扩展矩阵只有事前指定的四臂。阶段一复用已审计的历史结果和成本，同时对 C0、k-center、recluster 重跑复现，并补齐 random synthetic。TAGSAM 是现有静态适配，未宣称完整官方复现。

## 2. 共同协议

| 项目 | 固定设置 |
|---|---|
| 数据与预算 | Cora、Citeseer；构造只读取合法训练 pool；累计预算为已见 pool 节点数乘压缩率后向下取整，持续增加，不剪除旧行 |
| 类别顺序 | canonical、Python random.Random(19101) 与 random.Random(19102) 的排列；阶段内 class ID/source ID 升序 |
| 三阶段 | Cora 2/2/3 类；Citeseer 2/2/2 类 |
| 五阶段 | Cora 3/1/1/1/1 类；Citeseer 2/1/1/1/1 类；canonical 按类别 ID，不与旧频率排序 base+4 混表 |
| 压缩率 | 1%、5%、10%；指节点数，不指总字节或总计算量 |
| Student | 两层 GCN，hidden/out=384；冻结 MiniLM 文本编码器、可训练文本投影；双向图文 InfoNCE；Adam，学习率 0.001 |
| 训练 | 每阶段从头训练；report seeds 100..105；batch=16，实际取 min(16,rows)；步数 max(50,min(2048,16×rows)) |
| 推理 | 在真实全图上做 transductive 推理，与全部类别 prompt 匹配；构造素材与推理可见结构分别限定 |
| 主指标 | 全部类别 prompt 竞争下，已见目标节点的 pooled report-dev Macro-F1；同时报告最终、阶段均值、pooled old/new |
| 汇总 | 每条件先平均 6 个 student seeds，再对条件等权；不得把同一 memory 的 6 个 seeds 视为 6 个独立数据集 |
| 例外参考 | 完整 pool 等步数分支匹配相应合成图训练步数；8192 步长训及其选检查点分支明确单列，不冒充等计算预算 |

这里的 Macro-F1 是节点分类评价，不是训练损失。Student 不使用节点类别标签做监督分类训练，但 C0 的按类构造使用合法 pool 类别标签；k-center 的普通版本不使用这些标签，因此同时展示 class-aware k-center。不能称为整个方法完全无监督或完全不使用类别信息。

本批未评价 test 标签；项目历史开发曾接触 test，不能宣称整个项目拥有未污染的最终盲测集。canonical 3-stage/5% 是既有开发条件；新顺序在本批冻结前未用于调参，但 report seeds 仍为已有 seeds。old/new 均使用本轮 pooled 口径，不与历史 task-slice old/new 混比。

## 3. 本轮算法

### 3.1 C0：冻结控制

每阶段按固定压缩率计算累计 rows，保留全部旧合成记忆，把新增配额按 dispersion 分配到当前类别；每类在图属性与文本的 joint space 中聚类。每簇构造 Mean-X 合成特征，按 joint_g5_c320 选择最多 5 篇代表文本、约 320 字符预算拼接，再由冻结 MiniLM 重编码。

追加时冻结旧特征、旧文本、旧行及旧旧边，按既有 text-nearest 规则加入当前块与跨块边，跨块度为 1。没有 response 选优、动态 extra budget、可训练文本 attention 或历史 refresh。提出的方法使用合成节点；真实节点只用于 coreset 对照。

当前 append-only 就是 C0，不再列一个名称不同、算法相同的“当前方法 append-only”来增加表格行。C0 canonical 最终结果为 Cora 49.8518、Citeseer 59.1747；旧 v4 Citeseer 60.7653 不能换入本表。

### 3.2 唯一新机制：块保持联合核

候选逐元素复用 C0 的 X、文本 embedding、原文、mass、depth、stage、rows 及行顺序。重建 C0 源簇，并核对每簇的 Mean-X 与 mass 完全一致；“多原型”指 C0 已有的多个合成簇，不额外添加 C1 配额或新的 X 算子。

固定关系核为：

\[
K=0.4K_X+0.3K_T+0.3K_G.
\]

其中 K_X 与 K_T 分别是归一化合成 X 和重编码文本的 Gram 矩阵。K_G 的签名只从当前 delta 的合法 pool 诱导图构造：

\[
J=[\operatorname{norm}(X),T],\quad
H=\operatorname{norm}(0.5J+0.5\widehat A_{\Delta}J),\quad
g_j=\operatorname{norm}\!\left(\frac{1}{|G_j|}\sum_{i\in G_j}H_i\right).
\]

\(\widehat A_{\Delta}\) 为带自环的对称归一化邻接矩阵，G_j 为 C0 的源簇。新签名随合成记忆保存；后续阶段不回读历史原始节点。源节点分组文件仅作离线审计证据，不计作算法可访问的持续记忆。

候选只替换两类边的端点：

1. 新块内部选择核值最高的无向节点对，并补反向边；条数与 C0 匹配。
2. 每个新合成行连接核值最高的一个旧行，并补反向边；并列值按行 ID 决定。

旧行、旧文本、旧签名、旧旧边和稠密核旧块保持不变；严格 append 路径拒绝 prune。权重 0.4/0.3/0.3 为事前工程假设，本轮没有按 report 结果调权。

“旧旧边不变”不等于“旧传播算子不变”：新跨边改变旧节点的度，对称归一化后旧块仍会变化。正式记录中，联合核每数据集 54 次追加的归一化旧块相对漂移均值约为 Cora 0.14、Citeseer 0.20。这证明存在算子变化，不证明它就是 F1 下降的原因。

稠密 K 是非负加权 Gram，具有 PSD 性质；取最高分边后的稀疏无权邻接矩阵并无相同保证，更不提供性能下界。签名与新构造成本必须计入，不能将其视为免费机制。

## 4. 阶段一：canonical 基线主表

以下均为 canonical 三阶段。所有压缩分支匹配同一节点轨迹：Cora 25/67/94，Citeseer 38/86/111；最终合法 pool 分别为 1895、2229 个节点。原始数据参考不受压缩预算限制。F1 单位为百分数，差值单位为百分点。

最终列为 6 个 report seeds 的均值 ± 标准差；标准差按现有汇总程序 np.std、ddof=0。阶段均值是三个阶段 F1 的算术平均。seed 波动仅反映给定构造下的 student 随机性。

| 方法或参考 | 数据访问/管理 | Cora 最终 F1 | Citeseer 最终 F1 | Cora 阶段均值 | Citeseer 阶段均值 |
|---|---|---:|---:|---:|---:|
| **C0** | 旧合成记忆 + delta，冻结追加 | **49.85 ± 3.00** | **59.17 ± 1.57** | 53.08 | 64.29 |
| Random synthetic | 同类配额内随机分组后合成，追加 | 50.49 ± 2.84 | 38.51 ± 1.91 | 51.95 | 33.53 |
| Random coreset | 旧真实样本 + delta，追加 | 43.47 ± 2.48 | 49.79 ± 2.47 | 55.50 | 48.51 |
| k-center | 真实节点覆盖选择，追加 | 46.35 ± 4.36 | 55.34 ± 1.11 | 59.86 | 54.18 |
| Class-aware k-center | 按类真实节点选择，追加 | 45.09 ± 2.86 | 53.40 ± 0.82 | 67.13 | 60.18 |
| K-means medoid | 真实代表节点，追加 | 47.50 ± 2.62 | 49.46 ± 2.49 | 55.56 | 50.08 |
| Herding | 真实节点均值逼近，追加 | 37.78 ± 4.36 | 55.85 ± 3.76 | 48.01 | 56.77 |
| Recluster | 旧合成记忆 + delta，可重写旧行 | 49.70 ± 1.21 | 54.24 ± 1.91 | 48.99 | 61.28 |
| Full-history synthetic | 每阶段回读全部已见 pool，全局重凝聚 | 47.21 ± 2.06 | 48.28 ± 3.48 | 53.86 | 58.37 |
| TAGSAM self-loop 静态适配 | 全部已见 pool，全局重凝聚 | 47.73 ± 4.00 | 51.09 ± 2.85 | 65.67 | 55.92 |
| TAGSAM semantic 静态适配 | 全部已见 pool，全局重凝聚 | 52.86 ± 3.46 | 42.00 ± 5.85 | 56.21 | 43.97 |
| 完整原始训练图，等步数 | 全部已见合法 pool | 47.93 ± 4.10 | 49.89 ± 2.17 | 51.38 | 51.74 |
| 完整原始训练图，8192 步末点 | 全部已见合法 pool | 46.26 ± 2.73 | 52.33 ± 1.75 | 48.24 | 52.39 |
| 完整原始训练图，8192 步选检查点 | 全部已见合法 pool，selector-dev 选点 | 51.31 ± 2.89 | 55.60 ± 1.92 | 59.84 | 60.98 |

完整逐 seed、逐阶段和来源见 [phase1_summary.csv](../../results/phase12_frozen_20260920/summary/phase1_summary.csv)。

### 4.1 这些结果能支持什么

| C0 对比完整 pool 8192 步选点参考 | Cora | Citeseer |
|---|---:|---:|
| 最终节点比例 | 4.96% | 4.98% |
| 最终 F1 保留率 | 97.16% | 106.42% |
| 阶段平均 F1 保留率 | 88.71% | 105.43% |
| 最终 F1 差值 | -1.46 pp | +3.57 pp |
| C0 最终 pooled old / new | 65.44 / 29.06 | 66.46 / 44.61 |
| 完整 pool 选点参考 pooled old / new | 60.85 / 38.59 | 63.90 / 39.01 |

保留率定义为 C0 的 F1 除以指定完整数据参考的 F1，不是准确率，也不是理论信息保留率。高于 100% 只表示超过当前模型、训练目标和选点策略下的实测参考；8192 步不构成充分收敛证明或理论上界。

C0 在该 canonical 条件下有明确的压缩任务效用，最终优于本表的 k-center、medoid 和全历史启发式重凝聚，但不能推出整个训练过程都更优。例如 Cora 的 k-center 阶段均值 59.86 高于 C0 的 53.08，TAGSAM semantic 最终结果也高于 C0。

Random synthetic 保持 C0 的 dispersion 类配额、Mean-X、concat/按类批次重编码与追加拓扑，仅将类内分簇换成确定性、均衡随机分组，构造 seed 为 6400+101×stage。它在 Cora 最终高 0.64 pp、阶段均值低 1.13 pp；Citeseer 最终低 20.66 pp。这支持分组方式在部分条件下很重要，不能证明聚类在所有图上最优；随机构造这里只覆盖一个 seed family。

全历史重凝聚与 recluster 的数据访问、历史可改写性及构造细节不同。特别是现有 recluster 不沿用 C0 的全部按类分配规则，故这些对照不是“只改变一个约束”的严格因果损失分解。不能把差值全部归因于历史不可回读或旧行不可改写。

### 4.2 静态适配与监督参考

TAGSAM 两臂完整展示，未按数据集挑选有利变体；它们是本项目共同 student 下的静态适配，不代表完整官方 TAGSAM 的最佳性能。共同 student 对照回答压缩产物的可重训效用，不证明本方法全面超越原论文。

STAD 的共同 student 适配及原生风格监督诊断继续保留在 [外部 TAG 基线报告](EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)，本轮未补强。监督 GCN 使用分类标签和不同训练目标，只能单列为监督任务参考，不是共同 InfoNCE 主表的直接排名对象；也不把其称为数学上界。相关历史结果见 [五层对照报告](FIVE_LEVEL_GLOBAL_VS_CONTINUAL_EXPERIMENT_REPORT_ZH_2026-09-17.md)。

## 5. 阶段二：完整 36 条件结果

每数据集 18 个条件，每条件 6 个 seeds；下表按条件等权。此表不是 canonical 单点，也不与上表的数字混用。

| 数据集 | 方法 | 最终 F1 | 阶段平均 F1 | 最终 pooled old | 最终 pooled new |
|---|---|---:|---:|---:|---:|
| Cora | C0 | 46.39 | 53.68 | 53.93 | 27.29 |
| Cora | k-center | 44.78 | 51.99 | 52.37 | 24.46 |
| Cora | Recluster | 45.45 | 51.61 | 52.39 | 27.67 |
| Cora | Joint-kernel | 48.07 | 57.11 | 55.68 | 30.35 |
| Citeseer | C0 | 52.28 | 57.24 | 56.15 | 38.24 |
| Citeseer | k-center | 53.59 | 59.75 | 56.71 | 40.87 |
| Citeseer | Recluster | 51.52 | 54.66 | 54.89 | 38.31 |
| Citeseer | Joint-kernel | 52.51 | 52.77 | 55.81 | 40.21 |

联合核相对 C0，Cora 最终 +1.67 pp、阶段均值 +3.44 pp；Citeseer 最终 +0.23 pp、阶段均值 -4.46 pp。Citeseer 上 k-center 的最终与阶段均值均超过 C0 和联合核。两图平均稍有改善不能掩盖这项弱点。

### 5.1 压缩率与阶段数

每行平均三个类别顺序；“正收益数”指最终 F1 高于 C0 的顺序数。所有差值单位均为 pp。

| 数据集 | 阶段 | rate | 正收益数 | Joint-C0 最终 | Joint-C0 old | Joint-C0 new | Joint-k-center 最终 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 3 | 1% | 3/3 | +5.35 | +5.87 | +4.66 | +21.52 |
| Cora | 3 | 5% | 0/3 | -2.46 | -0.32 | -5.32 | +0.68 |
| Cora | 3 | 10% | 2/3 | +0.84 | +0.59 | +1.16 | -4.91 |
| Cora | 5 | 1% | 3/3 | +7.48 | +8.35 | +2.23 | +8.53 |
| Cora | 5 | 5% | 1/3 | +0.02 | -1.82 | +11.02 | -5.18 |
| Cora | 5 | 10% | 2/3 | -1.19 | -2.17 | +4.65 | -0.93 |
| Citeseer | 3 | 1% | 1/3 | -1.17 | -3.19 | +2.87 | -0.60 |
| Citeseer | 3 | 5% | 2/3 | +1.99 | +0.98 | +4.00 | -1.83 |
| Citeseer | 3 | 10% | 0/3 | -2.28 | -1.45 | -3.92 | -5.25 |
| Citeseer | 5 | 1% | 3/3 | +2.66 | +2.83 | +1.81 | +4.81 |
| Citeseer | 5 | 5% | 0/3 | -2.84 | -3.56 | +0.71 | +0.20 |
| Citeseer | 5 | 10% | 3/3 | +3.04 | +2.38 | +6.31 | -3.78 |

Cora 1% 是明确值得解释的收益区域，但该观察来自本次结果，不能事后据此将“只在低 rate 启用联合核”当成已验证的新方法。Citeseer 3-stage/1% 并不复现相同趋势。

### 5.2 不能忽略的条件

| 条件 | 观察 |
|---|---|
| Cora / 3-stage / canonical / 5% | C0 49.85 → Joint 47.68，最终 -2.17 pp；阶段均值却 +9.45 pp，新类 -7.91 pp，前期改善未延续到最终 |
| Citeseer / 3-stage / canonical / 5% | C0 59.17 → Joint 54.01，最终 -5.16 pp |
| Citeseer / 3-stage / order19102 / 1% | 最差最终 -6.41 pp，旧类 -8.08 pp |
| Cora / 5-stage / canonical / 1% | 最终 +9.33 pp、旧类 +12.58 pp，但新类 -10.19 pp，不能把此收益解释为塑性改善 |
| Cora / 5-stage / canonical / 5% | 最终 +6.98 pp、新类 +27.19 pp，仍低于 k-center 3.89 pp |
| Cora / 5-stage / canonical / 10% | 最终 -5.67 pp、旧类 -7.55 pp |
| Cora / 3-stage / order19102 / 5% | 最终 -4.99 pp，六个配对 student seeds 全部为负 |

36 个条件和配对 seed 差值见 [matrix_cells.csv](../../results/phase12_frozen_20260920/summary/matrix_cells.csv)，逐阶段见 [matrix_stages.csv](../../results/phase12_frozen_20260920/summary/matrix_stages.csv)。JSON 中提供条件内的 paired-seed 95% 区间。不同条件共享数据，不能将这些区间或 36 个条件解释为 36 个独立数据集的显著性检验。

## 6. 事前晋升门禁

“至少 3/4 条件正收益”在冻结协议中落实为完整 36 条件的 75%；性能、保持、成本和对照门槛需同时满足。以下使用未四舍五入的值判定。

| 门槛 | 实际 | 通过 |
|---|---:|---|
| 完整 36 条件并通过独立验收 | 36/36，0 error | 是 |
| 至少 75% 条件最终改善 | 20/36 = 55.56% | 否 |
| 平均最终增益至少 +1 pp | +0.951599 pp | 否 |
| 最差最终差值不低于 -2 pp | -6.408180 pp | 否 |
| 最差 pooled old 差值不低于 -2 pp | -8.084636 pp | 否 |
| 平均 pooled new 正增益 | +2.515224 pp | 是 |
| 至少半数条件新类改善 | 20/36 | 是 |
| 每数据集构造 + 单次 student 成本 ≤2×C0 | Cora 1.0158×；Citeseer 0.9937× | 是 |
| 全矩阵平均不劣于固定 k-center 对照 | +1.104777 pp | 是 |
| canonical 两图均值不劣于预定 TAGSAM self-loop 适配 | +1.436282 pp | 是 |

**结论：promote=false。** +0.9516 pp 不能四舍五入成 +1 pp 后宣称过线。即使放宽这一项，正收益覆盖率、最差最终和最差旧类仍不满足条件。最后两项是事前定义的聚合门槛，不表示逐数据集、逐条件都胜过强基线。

## 7. 成本与记忆

下表构造时间和六 seed 训练时间是每数据集 18 条完整阶段链的总和，单位秒；记忆是 18 个条件最终持久 payload 的平均字节数，不是单次峰值显存。

| 数据集 | 方法 | 构造总秒 | 六 seed 训练总秒 | 平均最终 memory 字节 |
|---|---|---:|---:|---:|
| Cora | C0 | 7.62 | 1021.20 | 347435 |
| Cora | k-center | 0.31 | 1023.14 | 399463 |
| Cora | Recluster | 7.95 | 1022.21 | 351810 |
| Cora | Joint-kernel | 10.34 | 1021.71 | 655659 |
| Citeseer | C0 | 9.33 | 1177.32 | 409935 |
| Citeseer | k-center | 0.38 | 1169.91 | 481600 |
| Citeseer | Recluster | 10.25 | 1165.03 | 418061 |
| Citeseer | Joint-kernel | 12.68 | 1149.49 | 773455 |

门禁使用的成本为“构造 + 六 seed 训练时间 / 6”，再跨 18 条链求和。Cora C0 为 177.82 秒、联合核 180.62 秒；Citeseer 分别为 205.55、204.26 秒。小幅计时波动不构成加速结论。各臂同数据集总训练步数相同：Cora 461856，Citeseer 478368，均为六 seed 合计。

联合核包含共享 C0 rows 的完整构造成本及重建分组/签名成本，不只记选边增量。其持久 memory 约为 C0 的 **1.89 倍**，主要来自 graph_signature；相同节点数不等于相同字节预算。签名不输入 student，因此训练时间接近不能消除这项存储代价。当前结论只覆盖等节点率，不覆盖等总字节率。

共同预处理、原始 pool 文本缓存首次编码、离线审计、磁盘 I/O 未计入上述构造/训练时间，不能宣称全生命周期计算高效。canonical C0 的峰值 CUDA allocated 显存为 Cora 115.92 MiB、Citeseer 116.64 MiB，与持久 payload 是不同指标。

阶段一历史成本在 CSV 中标记 historical；C0、k-center、recluster、random synthetic 标记 fresh。例：canonical 三阶段 C0 构造/单 seed 训练分别为 Cora 0.42/6.65 秒、Citeseer 0.49/9.22 秒；完整 pool 8192 步参考的历史单 seed 训练为 76.71/69.74 秒。历史 TAGSAM 构造时间也保留，但不把跨批计时拼成严格同时段端到端加速比。详见 [phase1_costs.csv](../../results/phase12_frozen_20260920/summary/phase1_costs.csv)、[phase1_summary.csv](../../results/phase12_frozen_20260920/summary/phase1_summary.csv) 和 [matrix_costs.csv](../../results/phase12_frozen_20260920/summary/matrix_costs.csv)。

## 8. 实现与审计证据

正式 runner 为 [run_frozen_baseline_joint_kernel.py](../../code/v2_llm_fusion/run_frozen_baseline_joint_kernel.py)，算法实现为 [joint_kernel_append.py](../../code/v2_llm_fusion/joint_kernel_append.py)。早期窄矩阵 smoke runner 仅作开发材料，不用于本报告结论。

| 验收项 | 结果 |
|---|---|
| 每图完整矩阵 | 18 条件、1728 fits；两个 completion 标记均存在 |
| 每图新增 random synthetic | 1 条件、18 fits；两个 completion 标记均存在 |
| 每图独立重载审核 | 1746 fits、19 条件（包括 baseline），partial=false，0 error |
| 历史 canonical 复现 | C0/k-center/recluster × 2 图 × 3 阶段，18 个对照；输入张量与文本相同，全部 108 个逐 seed F1 差值为 0 |
| 基线文件与 support | 468 对检查通过，0 error；包含历史来源指纹及支持集合检查 |
| 实现/汇总测试 | 6 个联合核单元测试、3 个报告判定测试，本地和远端均通过 |
| 本地回传完整性 | 4365 个文本证据文件，其中 4351 个 JSON、12 个 CSV、2 个日志；36 份正式 results.json 的 SHA256 与汇总来源逐一相符 |

审计包括 checkpoint 重新推理与指标重算、旧行和旧旧边保持、源池合法性、源簇与 C0 Mean-X/mass 对应、X/文本/预算/边数受控及来源指纹。模型重载审核、历史 support 审核与训练次数是不同计数，不能相加宣称更多独立实验。

正式源快照：/home/pc/research/TAG_condensation/work_runs/phase12_20260920_snapshot。
正式远端工件：/home/pc/research/TAG_condensation/results/phase12_frozen_20260920。
本地证据：D:/Aprojects/continue_tag_condensation/results/phase12_frozen_20260920。

memory 和 model 的二进制 .pt 文件仍保留在远端，本地未下载；本地有结果、预测/support、指纹、日志、审计与汇总，不能称为本地已保存完整二进制实验包。

| 核心文件 | SHA256 |
|---|---|
| 正式 runner | c71520001ed6c73dcb463cb0e75f88210bc5129ca36447138f7082f5062072ee |
| 独立 auditor | b600d134b26259cf8944333eb9db22d0941989f2c5acb35c598bb5ae60055057 |
| 阶段一汇总程序 | c5e6a14bd0217c87aa767d4e81a26ed293a1e74815a87db34b4850386f561964 |
| 阶段二汇总程序 | e3aac931df0729e77f154e5897d0afce9b05cbe6658fdecf75dd177d8d7ccc34 |

直接验收入口：[Cora 审计](../../results/phase12_frozen_20260920/audit_cora.json)、[Citeseer 审计](../../results/phase12_frozen_20260920/audit_citeseer.json)、[阶段一验收](../../results/phase12_frozen_20260920/summary/phase1_acceptance.json)、[阶段二完整判定](../../results/phase12_frozen_20260920/summary/phase2_summary.json)。

## 9. 研究解释与下一步

本轮能确定：C0 是具有可重训压缩效用的共同控制，当前联合核改边可以影响下游性能，也在部分条件明显改善；但收益对数据集、阶段数和 rate 敏感，尚未形成满足主方法标准的稳定机制。相同 X/文本与边数排除了“更多节点、更多文本或更多边”这一类解释，但未分离新块内部边、跨块端点、度分布和归一化扰动各自的作用。

下列只是待验证解释，不是已证明原因：固定语义相似度未必匹配 InfoNCE 所需的可区分关系；最高分选边可能集中到少数旧行并改变传播；低预算时原拓扑瓶颈可能更明显；当前 delta 内的一跳签名可能不足以表达跨阶段结构。正负条件均能提供线索，不能只选择支持其中一种解释的结果。

后续建议按以下顺序决策，本轮未执行：

1. **先做机制分离，再设计新版本。** 固定本轮 X/文本/rows，分开比较只改新块内部、只改跨块、两者同时改；加入同边数且尽量匹配度分布的随机端点控制，回答收益具体来自哪里。
2. **只有定位后才处理权重或传播。** 权重消融必须独立开发与确认，不能用本轮 36 条件挑“最佳权重”后重复宣称泛化。旧块传播保护需结合此前 C3/v5 已有负结果，不能当成未经尝试的新概念重新包装。
3. **补字节公平性和外部确认再考虑扩展。** 如签名机制继续保留，需要等总字节预算对照、其他数据集或新预注册顺序，并保留 k-center 强参照。主方法仍冻结 C0，暂不上大图，也不叠加 response/dynamic budget 以掩盖不稳定性。

论文可以使用本轮统一比较、约 5% 节点的任务效用、块保持边界和联合核失效条件作为证据；这些实验本身尚不足以确立文献层面的新颖性、SOTA 或 WWW 长论文所需的核心算法贡献。阶段一完成提高了结论可信度，阶段二的否决避免把局部收益包装为普遍有效的新方法。

协议原文：[两阶段冻结协议](PHASE12_C0_JOINT_KERNEL_FROZEN_PROTOCOL_ZH_2026-09-20.md)。历史谱系与已有候选见 [文档索引](../INDEX.md)。
