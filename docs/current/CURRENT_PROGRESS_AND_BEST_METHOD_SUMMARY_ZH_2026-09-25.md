# 当前科研进度与最优方法总览

日期：2026-09-25  
状态：已完成 semantic permutation、P-guided two-prototype 以及最新 role-aware conserved assignment 的小图实验与守恒审计；本文汇总截至当前已经完成并通过审计的本地/远程结果

## 2026-09-26 独立确认后的修订结论

`role_perm` 已完成预留类别顺序 `order19605/19606/19608` 的 direct old/new F1 和构造成本确认。12 个独立 stage-cell 上相对 C0 的累计 F1 平均 `+1.468 pp`，`7/12` 正，最差 `-2.042 pp`；old F1 `+2.212 pp`，new F1 `+0.558 pp`，Citeseer new F1 `-0.025 pp`，stage 2 new F1 `-0.202 pp`。因此开发集的 `+3.579 pp、10/12 正` 没有完全复现，`role_perm` 仍未满足“至少 8/12 正、最差不低于 -2 pp、new-class 稳定改善”的晋升门槛，也暂不启动其 base+4/大图扩展。C0 继续作为冻结共同基线；role-aware assignment 保留为有机制解释但顺序依赖的候选。详见 [TGC Role-Aware 独立顺序确认报告](TGC_ROLE_AWARE_INDEPENDENT_CONFIRMATION_ZH_2026-09-26.md)。

## 1. 结论先行

需要区分三个“最优”：

| 含义 | 当前结论 | 定位 |
|---|---|---|
| 最可靠、可复现的默认主方法 | **C0** | 论文共同控制和当前默认方法 |
| 已有结果中最好的条件性候选 | **R15：R11 content + C0 topology** | 有正向信号，但新类和阶段平均仍不稳定 |
| 最新最值得继续验证的机制 | **守恒 semantic permutation** | 开发集和独立集有平均正收益，但未通过最差 cell/新类塑性晋升门槛 |
| 最新小图内部候选（开发集） | **Topology-Role-Aware Conserved Payload Assignment (`role_perm`)** | 开发集平均 `+3.579 pp`，但独立集为 `+1.468 pp`、7/12 正、最差 `-2.042 pp`；未通过晋升门槛 |

因此现在不能写成“某个新版本已经稳定超过 C0”。最稳妥的论文主线仍是：

> 在历史原始图不可回读的 continual TAG condensation setting 下，C0 提供可靠的确定性构造基线；teacher/joint、文本、拓扑、响应选择和 payload 置换实验揭示了保持-塑性、槽位传播和目标错位的边界。

## 2. 当前完整方法：C0

### 2.1 输入和访问约束

阶段 `t` 到达时，构造器只读取：

1. 上一阶段合成 memory `M_{t-1}`；
2. 当前合法原始增量 `Delta_t`；
3. 当前阶段允许使用的图、节点特征、文本和类别信息。

它不回读历史完整 raw graph。memory 使用合成节点，不把真实节点 coreset 作为主方法。当前主协议采用 growing memory，预算约为已见合法 pool 的 5%，旧行不因新增阶段被整体剪除。

### 2.2 预算和类别内分组

新增预算按当前 C0 的确定性 dispersion 规则分配到当前类别。每个类别内部在图文联合空间中分簇：

\[
J_i=[\operatorname{norm}(x_i),t_i].
\]

每个簇对应一行合成 memory。当前版本没有使用 response selector、动态 extra budget、可训练 attention 或复杂 learned allocator。

### 2.3 合成特征 X

对簇 `G_j` 使用 Mean-X：

\[
x_j^{syn}=\frac{1}{|G_j|}\sum_{i\in G_j}x_i.
\]

这保证特征构造简单、确定、可审计，但也带来类内多模态被平均掉、新类塑性不足等问题。此前 weighted-X、multi-prototype、student-guided X 等尝试没有形成跨数据集稳定收益。

### 2.4 文本

每个簇按 joint center 选择最多 5 个代表文本，使用约 320 字符 cap 拼接，再通过冻结 MiniLM 重新编码得到 `text_emb`。文本不是生成式摘要，不引入无法追溯的文本；文本来源可以审计到当前合法 raw delta。

### 2.5 拓扑

C0 复制历史合成拓扑，旧旧边保持不变；新块内部和新旧跨块边使用简单的相似度规则材料化：

- 新块内部按文本/联合相似度选择固定数量边；
- 新行连接相似的旧行；
- 保持与 C0 相同的 rows 和大致边预算；
- 不使用 GraphGPS，也不把复杂 weighted topology 作为默认配置。

注意：旧旧边不变不代表归一化 GCN 传播完全不变。新增跨边仍会改变节点度，从而改变归一化旧传播块。

### 2.6 下游 Student 和指标

- 两层 GCN/GraphCLIP-style student；
- hidden/output dimension 384；
- MiniLM 编码器冻结，文本投影层可训练；
- 双向图文 InfoNCE；
- 每个阶段从头训练，不继承上一阶段 student 参数；
- Adam，学习率 `1e-3`，batch size `16`；
- steps 使用 `max(50, min(2048, 16 * rows))`；
- 推理时使用真实图结构和类别 prompt 做节点分类；
- 主要指标是 all-prompt competition 下的 report-dev Macro-F1，并单列 old/new F1。

这里的 Macro-F1 是节点分类结果，不是 condensation loss，也不是监督 GCN 的分类训练准确率。Student 不使用节点标签做监督分类损失，但构造阶段的类别分配使用合法 pool 的类别信息，因此不能称为完全无监督凝聚。

## 3. C0 效果

在 canonical、三阶段、约 5% memory 的冻结基线中：

| 方法 | Cora 最终 F1 | Citeseer 最终 F1 | Cora 阶段均值 | Citeseer 阶段均值 |
|---|---:|---:|---:|---:|
| **C0** | **49.85** | **59.17** | **53.08** | **64.29** |
| k-center | 46.35 | 55.34 | 59.86 | 54.18 |
| Recluster | 49.70 | 54.24 | 48.99 | 61.28 |
| Full-history synthetic | 47.21 | 48.28 | 53.86 | 58.37 |
| 完整 pool、8192 步选点 | 51.31 | 55.60 | 59.84 | 60.98 |

C0 相对完整 pool 8192 步选点参考：

- Cora 最终低 `1.46 pp`；
- Citeseer 最终高 `3.57 pp`；
- Cora 节点压缩率约 `4.96%`，Citeseer 约 `4.98%`。

这说明 C0 在受限 memory 下有较强任务效用，但不是理论上界，也不能宣称全面超过完整数据训练。k-center 在某些阶段均值更强，说明 C0 的优势依赖阶段、顺序和 student 训练条件。

## 4. 重要路线和当前判定

| 路线 | 主要结果 | 当前判断 |
|---|---|---|
| P1 预算分配 | dispersion 在部分 cell 优于 k-center，但数据集/顺序相关；fixed-total 不适合作为默认 | 保留 C0 dispersion，动态预算未晋升 |
| P2 文本压缩/attention | concat + re-encode 最可靠；attention、图条件文本和 student-guided attention 未稳定提升 | 文本 attention 暂不进入主方法 |
| P3 response selector | 短训 ranking 与最终 F1 不一致，成本高，selector 常常安全回退但召回不到正收益 | 不作为默认组件 |
| P4 memory maintenance | C0-M、旧行 refresh、简单 revision 未稳定改善新类 | C0 继续冻结控制 |
| P5 多阶段/大图 | 顺序依赖明显；大图上 T+degree 对 BookChild 有益，但 arXiv 不稳定 | 只作为任务相关候选 |
| R10 joint | content-topology 有条件正交互；最终平均 `-0.05 pp`，4/6 cell 正 | 不晋升 |
| R11 anchor | `anchor025` 最终平均 `+0.142 pp`，阶段平均 `-0.856 pp`，new F1 `-2.059 pp` | 稳定化组件，不晋升 |
| R15 content + C0 topology | 最终平均约 `+0.723 pp`，但阶段平均 `-0.314 pp`、new F1 `-1.314 pp` | 当前最好的旧候选，但仍不稳定 |
| T+degree | BookChild 分类和 LP 有益；arXiv 分类下降 | 跨任务研究候选，不是统一主方法 |
| 旧 role-match | payload/source coverage 不守恒 | 旧正收益数字作废 |
| semantic permutation | 开发集 12 cell 平均 `+1.414 pp`；独立集 12 cell 平均 `+1.878 pp`，均为 8/12 cell 正 | 有真实机制信号，但未晋升；C0 仍是默认主方法 |
| P-guided two-prototype | 独立确认相对 C0 平均 `+0.35 pp`，7/12 cell 正，最差 `-6.13 pp`，new F1 `-1.01 pp` | 开发集正收益未稳定复现，不晋升 |
| role-aware conserved assignment | Cora/Citeseer、3 顺序、stage 1/2 共 12 cell 平均 `+3.579 pp`，10/12 正，最差 `-1.155 pp`；相对 identity 额外 `+3.066 pp` | 当前小图内部最强候选；先进入下一轮独立确认，不立即替换冻结 C0 |

## 5. 最新 semantic permutation 方法

这是当前最值得继续验证、但还没有晋升的机制。

### 5.1 做法

1. 先用双原型 split/merge 生成一组 proposal；
2. 不改变 proposal 的 X、文本、mass 或来源集合；
3. 在同类 C0 slots 中根据 joint graph-text center 计算 semantic cost；
4. 用 Hungarian assignment 将 proposal 分配到目标 slots；
5. 被挤出的完整 payload 通过剩余双射迁移；
6. 固定 C0 edge_index、旧 prefix、总 rows 和 Student；
7. 同时加入 random permutation 作为控制。

与旧 role-match 最大区别是：每行完整 payload 必须守恒，不能把 proposal 写到目标槽位后遗失原有内容。

### 5.2 stage-1 结果

相对于同一 `c0_320`：

| 方法 | 六 cell 平均 | 正 cell |
|---|---:|---:|
| Identity 双原型 | `-0.806 pp` | 1/6 |
| **Semantic permutation** | **`+1.166 pp`** | 3/6 |
| Role permutation | `-0.009 pp` | 2/6 |
| Random permutation | `+0.892 pp` | 3/6 |

代表性 cell：

- Cora/order19603：semantic `+7.184 pp`，六个 seed 全部为正；
- Citeseer/order19601：`+1.502 pp`；
- Citeseer/order19602：`-1.040 pp`；
- Citeseer/order19603：`-1.524 pp`。

因此目前只能说：

> 内容 proposal 与合成槽位的重新匹配存在真实正向信号，但 semantic cost 尚未成为稳定、跨顺序的 student utility 代理。

random permutation 也有收益，说明部分收益可能来自传播槽位重排，而不是 semantic cost 已经正确预测了 student response。

### 5.3 stage-2 与合并结果

stage-2 使用相同的 C0 拓扑、文本、Student 和六个 report seeds，只切换到第二个增量阶段：

| 比较 | 平均相对 C0 | 正 cell | 最差 cell |
|---|---:|---:|---:|
| Identity 双原型 | `+0.780 pp` | 4/6 | `-2.052 pp` |
| **Semantic permutation** | **`+1.663 pp`** | **5/6** | `-1.155 pp` |
| Role permutation | `+0.780 pp` | 4/6 | `-2.052 pp` |
| Random permutation | `-0.014 pp` | 3/6 | `-4.267 pp` |

stage-1 与 stage-2 合并 12 个 cell 后：

| 方法 | 平均相对 C0 | 正 cell | 最差 cell |
|---|---:|---:|---:|
| **Semantic permutation** | **`+1.414 pp`** | **8/12** | `-1.524 pp` |
| Random permutation | `+0.439 pp` | 6/12 | `-4.267 pp` |

按数据集，Cora 平均 `+2.416 pp`，Citeseer 平均 `+0.412 pp`。semantic permutation 的 old-class F1 平均变化为 `+0.674 pp`，new-class F1 平均变化为 `+2.621 pp`；其中 Cora new-class 平均 `+4.491 pp`，Citeseer 仅为 `+0.751 pp`。这支持“槽位匹配可改善部分新类塑性”的解释，但不支持已经跨数据集稳定泛化的结论。

stage-2 的 18 个置换 arm 全部通过 `payload_multiset_preserved`、`source_coverage_preserved`、`same_c0_topology` 和 `same_old_prefix` 审计，且没有使用 report/test F1 选择候选。

### 5.4 独立顺序确认

使用未参与前面开发集的 `order19605/19606/19608`，在 Cora/Citeseer 上执行 stage-1/2：

| 数据集/顺序 | stage-1 | stage-2 |
|---|---:|---:|
| Cora/order19605 | `-0.210 pp` | `+0.592 pp` |
| Cora/order19606 | `+5.411 pp` | `-1.264 pp` |
| Cora/order19608 | `+6.334 pp` | `+5.684 pp` |
| Citeseer/order19605 | `+1.720 pp` | `-3.631 pp` |
| Citeseer/order19606 | `-3.105 pp` | `+2.081 pp` |
| Citeseer/order19608 | `+4.782 pp` | `+4.144 pp` |

独立集平均 `+1.878 pp`，8/12 cell 为正，random permutation 平均 `-0.339 pp`；但 semantic 最差 cell 为 `-3.630 pp`，且 Citeseer new-class F1 平均 `-0.698 pp`。因此它相对 random 有更强的平均信号，却没有通过“最差 cell 约不低于 `-2 pp`、新类塑性稳定”的晋升标准。全部独立置换 arm 的 payload、source coverage、C0 topology 和 old-prefix 审计均通过。

### 5.5 P-guided two-prototype 独立确认

随后对开发集表现较好的 P-guided two-prototype 做了预留顺序 `order19605/19606/19608` 的独立确认。组合 runner 同时重评估 C0、R11 content、semantic/random permutation 和 P-content arms，保持原始 Student、C0 topology、top-5/320 文本预算以及 report seeds `150..155`。

统一组合 runner 的 12 个 stage-cell 结果为：

| 方法 | 平均相对 C0 | 正 cell | 最差 cell | new F1 平均变化 |
|---|---:|---:|---:|---:|
| `semantic_perm` | `+1.59 pp` | 9/12 | `-3.43 pp` | `-0.41 pp` |
| `p_content_semantic_recomputed` | `+0.71 pp` | 8/12 | `-5.24 pp` | `+0.96 pp` |
| `p_content_two_proto_identity` | `+0.35 pp` | 7/12 | `-6.13 pp` | `-1.01 pp` |
| `r11_content_c0_topology` | `-0.25 pp` | 7/12 | `-6.74 pp` | `-1.50 pp` |

P identity 的数据集均值为 Cora `+0.63 pp`、Citeseer `+0.08 pp`；阶段均值为 stage-1 `+0.56 pp`、stage-2 `+0.15 pp`。它没有通过“平均至少 +0.5、最差不低于约 -2、new-class 不系统下降”的晋升门槛。完整报告见：[P 内容独立确认报告](TGC_P_CONTENT_TWO_PROTOTYPE_INDEPENDENT_CONFIRMATION_ZH_2026-09-25.md)。

独立结果目录：`results/tgc_two_prototype_role_permutation_independent_20260925/`。

## 6. 当前真正的瓶颈

### 新类塑性

多数复杂版本 old F1 可以保持甚至提高，但 new F1 下降。问题集中在：

- Mean-X 压缩类内多模态；
- 新行接入的旧传播角色不一定适合新类；
- graph/text assignment 与最终 GCN message passing 不完全一致；
- student 是从头训练的零样本图文匹配模型，teacher proxy 或短训 margin 不能稳定预测最终 Macro-F1。

### 顺序稳定性

同一机制在 Cora/Citeseer 或不同类别顺序上可能方向相反。当前不能用单一最好顺序代表泛化，也不能事后从多个 candidate 中挑最好结果。

### 目标错位

teacher loss、synthetic margin、局部 message loss 和最终 all-class Macro-F1 不是同一个目标。此前 selector 主要失败在“可以过滤坏候选，但召回不了真正好的候选”。

### 任务泛化

T+degree 在 BookChild 的 LP/分类上有收益，但在 arXiv 上不稳定；分类和 LP 对拓扑的偏好也不完全一致。因此不能用单一下游任务确认一个通用拓扑。

## 7. 当前推荐方法层级

### 论文默认主方法：C0

使用 C0 作为主表和共同控制，诚实报告它相对完整 pool、k-center、recluster、全历史重凝聚和静态 TAG 方法的差距。

### 研究候选：C0 + conserved semantic slot reassignment

semantic permutation 已完成 stage-1/stage-2 复现和独立顺序确认，可作为 C0 的受控增强研究候选，但还不能替代默认主方法。P-guided two-prototype 的独立确认也已完成，但不满足晋升条件。下一步只做：

1. 冻结当前 semantic cost，不再按已有 cell 调权重；
2. 将本轮 P 内容生成结果作为完整负结果，说明 P-weighted X/text 重构没有稳定解决新类塑性；
3. 若继续 semantic placement，只补做构造成本、移动行数和任务结构指标，不再扫描 P 权重；
4. semantic permutation 仍未满足最差 cell 和 new-class 稳定性门槛，因此当前不晋升；后续只能作为机制候选，或重新设计不依赖最终 F1 的 assignment 约束。

### 暂不继续的方向

- 不再扫描 role 权重；
- 不再扩大短训 selector；
- 不把 GraphGPS 加入当前 Student；
- 不把 fixed alpha、简单 degree 重连或 text attention 直接写入主方法；
- 不因为单个大图或单个顺序的正收益宣称 SOTA。

## 8. 论文可写的贡献边界

当前最稳妥的贡献不是“提出一个已经稳定 SOTA 的新算子”，而是：

1. 明确并实现 continual TAG condensation 的访问约束、growing memory 和统一 student 协议；
2. 给出可审计的确定性 synthetic append 构造；
3. 系统分析预算、X、文本、拓扑、teacher、response selector 和 memory revision 的有效边界；
4. 发现 content/topology 的条件性协同、新类塑性瓶颈以及 payload-slot 传播匹配的重要性；
5. 用守恒置换实验区分真实机制收益与实现覆盖量混淆。

如果 semantic permutation 在后续结构任务和重新设计的独立确认中继续成立，可以再增加一个算法贡献：

> constrained semantic slot reassignment：在不访问历史完整 raw graph、保持 memory payload 守恒的条件下，将新 proposal 分配到与其图文内容相容的历史传播槽位。

目前 stage-2 和独立顺序均已完成；独立集最差 cell 与 Citeseer new-class 结果未达晋升标准，且链路预测仍未完成，因此暂不把它写成已确认的主算法创新。

## 9. 2026-09-25 最新修订：role-aware conserved assignment

最新完成的 12-cell 小图实验改变了“当前候选只能停留在 semantic permutation”的判断，但没有改变 C0 作为冻结共同控制的地位。

### 9.1 方法

该方法先沿用固定 two-prototype structural proposal，然后不改 proposal 的 `X`、文本、mass 或 C0 `edge_index`，只在同类 slots 内做完整 payload 的确定性双射匹配。目标 cost 为：

\[
C=z(C_{joint})+0.50z(C_{role})+0.10z(C_{mass}),
\]

其中 `C_role` 由 degree、cross-degree、new-new degree、old-neighbor ratio 和 old-endpoint load 构成。Hungarian assignment 后，`x`、`text_emb`、文本来源、mass、depth 和 stage 整体搬移；旧 prefix、边集合和 source coverage 保持不变。

### 9.2 最新结果

在 Cora/Citeseer、`order19600/19609/19610`、stage 1/2、report seeds `150..155` 上：

| 方法 | 累计 F1 相对 C0 | 正 cell | 最差 cell | 当前阶段 F1 变化 | 旧阶段 F1 变化 |
|---|---:|---:|---:|---:|---:|
| Identity proposal | +0.513 pp | 6/12 | -3.509 pp | +0.595 pp | +1.210 pp |
| Semantic permutation | +2.541 pp | 7/12 | -1.539 pp | +3.484 pp | +2.858 pp |
| **Role permutation** | **+3.579 pp** | **10/12** | **-1.155 pp** | **+5.958 pp** | **+2.945 pp** |

Role assignment 相对 Identity 仍额外提升 `+3.066 pp`，说明信号主要来自“payload 与传播角色的匹配”，而不是只来自 two-prototype 内容变化。Cora 数据集 6/6 cell 为正，Citeseer 为 4/6；stage 1 平均 `+5.373 pp`，stage 2 平均 `+1.785 pp`。

### 9.3 当前定位修订

`role_perm` 通过了本轮小图内部晋升门槛，成为下一轮候选主线，可暂命名为 **Topology-Role-Aware Conserved Payload Assignment**。但本轮没有直接保存 new-class F1，也没有独立测量 assignment 构造成本，且只覆盖两个小图和两阶段。因此：

- C0 继续作为论文默认主方法和共同控制；
- role-aware assignment 进入独立顺序和长链确认；
- message/plasticity/adaptive proxy 先保留为负结果，不继续叠加；
- 只有在直接 new/old F1、构造成本和 base+4/更长链确认后，才决定是否把 role-aware 机制升为论文主算法。

完整结果与审计见：[TGC Role-Aware Assignment 完成报告](TGC_ROLE_AWARE_ASSIGNMENT_COMPLETION_ZH_2026-09-25.md)。

## 10. 2026-09-26：Message-Conditioned X 独立确认

为直接验证“让合成 `X` 匹配当前增量一/二跳 message target 是否能改善新类塑性”，完成了 Cora/Citeseer、`order19605/19606/19608`、stage 1/2、12-cell 的 `C0 / Identity / Message-X` 重训对照。Message-X 只修改 changed synthetic rows 的 `X`，保持 C0 topology、two-prototype text、旧 prefix、mass/depth/stage 不变；构造阶段 `historical_raw_reads=0`。

构造审计显示 81/81 个 changed rows 的全局 message error 下降，均值由 `1.374` 降到 `0.968`，但 Student 结果相对 C0 为：累计 F1 `-0.824 pp`、正 cell `5/12`、old F1 `+1.476 pp`、new F1 `-3.415 pp`。因此 raw induced message proxy 与当前 GCN+InfoNCE Student utility 不一致；不能把 Message-X 晋升为主方法，也不能继续用它单独控制预算或 selector。

完整报告见：[TGC Message-Conditioned X 完成报告](TGC_MESSAGE_CONDITIONED_X_COMPLETION_ZH_2026-09-26.md)。下一步若继续算法创新，应转向真正共享 `P+X` 的联合构造：同一 `P` 生成 `X`、文本和拓扑，并用 bounded Student-facing objective 约束，而不是在固定 C0 拓扑上单独追逐 raw message error。

## 11. 2026-09-26：Role-aware 全 arm 统一 Student 重载

为避免此前只重载 `C0 / identity / role_perm` 造成的候选不完整，本轮对 Cora/Citeseer、`order19605/19606/19608`、stage 1/2 的全部 9 个 arm 统一重训 Student。协议固定为原始两层 GCN、双向图文 InfoNCE、batch 16、Student seeds `150..155`，指标为 all-prompt report-dev 的累计/old/new Macro-F1。

完整 12-cell 结果为：

| 方法 | 累计 F1 相对 C0 | old F1 变化 | new F1 变化 | 正 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|
| semantic_perm | **+1.690 pp** | +2.983 pp | -0.027 pp | **9/12** | **-1.904 pp** |
| role_perm | +1.468 pp | +2.212 pp | +0.558 pp | 7/12 | -2.042 pp |
| anchor_role_perm | +1.027 pp | +2.286 pp | -0.530 pp | 7/12 | -1.904 pp |
| message_role_perm | +0.710 pp | +1.747 pp | -0.417 pp | 8/12 | -3.479 pp |
| plasticity/adaptive role | +1.029 pp | +2.323 pp | -0.568 pp | 8/12 | -3.825 pp |
| random_perm | -0.319 pp | +1.452 pp | -2.340 pp | 5/12 | -7.088 pp |

因此，当前最稳的候选从 `role_perm` 修订为 `semantic_perm`。它满足本轮预先设定的 `+0.5 pp` 平均提升、至少 `8/12` 正 cell、最差不低于约 `-2.5 pp`、new-class 平均下降不超过 `1 pp` 的小图晋升门槛。`role_perm` 虽然 new-class 平均更好，但正 cell 不足，保留为机制解释性 ablation，不直接升为主方法。

这个结果不推翻 C0：C0 仍是共同冻结基线；它说明在保持 C0 topology、旧 prefix、Student 和训练协议不变时，类内 graph-text semantic payload assignment 是目前最值得做新顺序及大图确认的增量机制。完整表和约束审计见：[TGC Role-aware 全 arm 统一评估报告](TGC_ROLE_AWARE_FULL_ARM_EVALUATION_ZH_2026-09-26.md)。
