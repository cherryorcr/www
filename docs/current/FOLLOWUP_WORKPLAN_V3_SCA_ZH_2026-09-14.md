# GraphCLIP Continual v3 / SCA 后续完整工作计划

> 编号说明（2026-09-17）：本文 P0-P6 是旧版工作编号，不能与 2026-09-15 的 v4 P0-P5 直接对应。后续统一采用 v4 编号，映射与状态见 [v4 状态与后续优化路线](V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md)；本文作为历史记录保留。

更新时间：2026-09-14  
项目主线：GraphCLIP continual v3，不再回到 D2/A2 协议  
当前论文定位：机制分析 + 负结果 + 简单可靠构造，暂不宣称 SOTA

## 执行状态（2026-09-15）

本轮已完成第一批代码落地和小图正式复核：

- 新增统一入口 `code/v2_llm_fusion/run_unified_v3_continual.py`，支持 GraphCLIP、TAGBench 和 arXiv 字典 schema，并统一阶段划分、预算、student seeds、manifest 和 stagewise 指标。
- `run_student_compatible_append.py` 新增显式 `text_operator`、`refresh_mode`、`refresh_ratio`、`refresh_alpha` 和 `dynamic_rows` 开关，默认行为仍是原 frozen-old、concat、固定预算。
- `run_multiscale_asm.py` 与 external baseline runner 已扩展到 WikiCS、BookHis、BookChild、SportsFit 数据入口。
- WikiCS smoke 已通过：统一 continual append、weighted text pooling、bounded refresh 和 dynamic rows 均能完成三阶段运行。
- Cora/Citeseer canonical v3 unified continual 已完成，结果见 `results/formal_unified_cora/` 和 `results/formal_unified_citeseer/`。
- Cora/Citeseer canonical construction baseline 已完成，结果见 `results/formal_baseline_cora/` 和 `results/formal_baseline_citeseer/`。

截至 2026-09-15，大图 canonical continual、七数据集统一主表和外部 continual 方法的严格协议审计已经完成。
仍未完成的是对任务定义不匹配的外部方法进行完整重实现；这不影响 v3 的同协议内部主表，但意味着不能写成
“已经击败 continual TAG SOTA”。下面保留已验证与未验证边界，避免把诊断结果包装成普遍优势。

### 小图正式复核摘要

Continual cumulative dev Macro-F1，按 stage 0/1/2 报告：

| 数据集 | append_fixed | recluster | full_history |
|---|---:|---:|---:|
| Cora | 65.77 / 60.69 / 51.82 | 65.77 / 68.33 / 52.44 | 65.77 / 54.30 / 53.47 |
| Citeseer | 61.25 / 73.97 / 58.89 | 61.25 / 63.94 / 56.75 | 61.25 / 68.94 / 57.06 |
| WikiCS | 65.36 / 63.77 / 53.08 | 尚未跑 | 尚未跑 |
| BookHis | 18.71 / 17.49 / 17.41 | 尚未跑 | 尚未跑 |

Static construction baseline，Macro-F1 均值 ± std：

| 数据集 | random | k-center | herding | kmeans-medoid | ours | no-fuse | no-reencode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 49.31±3.99 | 51.41±3.36 | 44.69±1.08 | 50.00±2.98 | 50.75±2.17 | 40.52±3.21 | 44.58±1.89 |
| Citeseer | 49.60±4.51 | 56.14±3.10 | 53.53±2.68 | 58.82±3.79 | 55.65±2.32 | 60.62±1.87 | 50.02±3.19 |

这批结果修正了两条表述：k-center/kmeans-medoid 在统一 static 设置下仍然很强；重新编码比“拼接多个文本”更稳定，no-fuse 在 Citeseer 的提升是明确反例。

### 新机制小图复核

在固定 append、相同 canonical student seeds 下，s2 结果如下：

| 数据集 | fixed append | dynamic rows | bounded refresh | dynamic + refresh |
|---|---:|---:|---:|---:|
| Cora | 51.82 | 51.74 | 51.90 | 48.47 |
| Citeseer | 58.89 | 61.42 | 61.36 | 59.09 |

动态 rows 的预算实际变化为 Cora `28/50/45`、Citeseer `44/64/43`；bounded refresh 在保留旧预算的同时，分别刷新了 Cora s1/s2 的 `3/6` 行和 Citeseer s1/s2 的 `4/10` 行。结果支持“动态预算”和“受限刷新”作为候选增强，但不支持未经选择地组合两者。

文本 operator 的 s2 cumulative dev Macro-F1：

| 数据集 | concat | mean_pool | weighted_pool | set_pool |
|---|---:|---:|---:|---:|
| Cora | 51.82 | 51.02 | 47.49 | 46.90 |
| Citeseer | 58.89 | 53.48 | 54.25 | 54.08 |

当前结论是：没有证据支持 pooling/set operator 作为普适主创新；后续应优先研究动态预算与 bounded refresh 的独立稳定性，并把 text pooling 作为数据特性/负结果分析。

### 多数据集覆盖状态

统一 runner 已在七个数据集上完成至少一个 continual 结果；Cora、Citeseer、arXiv、BookChild、SportsFit 还完成了
SCA 与多种 memory/构造对照。当前仍不能把“多数据集覆盖”误写成“多数据集 SCA 普遍有效”或“已击败 SOTA”。

正式 static baseline 已统一完成 Cora、Citeseer、WikiCS：

| 数据集 | random | k-center | kmeans-medoid | ours | no-fuse | no-reencode |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 49.31±3.99 | 51.41±3.36 | 50.00±2.98 | 50.75±2.17 | 40.52±3.21 | 44.58±1.89 |
| Citeseer | 49.60±4.51 | 56.14±3.10 | 58.82±3.79 | 55.65±2.32 | 60.62±1.87 | 50.02±3.19 |
| WikiCS | 56.09±1.57 | 43.21±5.91 | 56.18±3.84 | 55.09±4.71 | 58.79±2.90 | 53.05±6.27 |

该表的正确读法是：k-center 不是普遍最强，ours 也不是普遍最强；kmeans-medoid 和 no-fuse 在部分数据集更强，说明数据特性与 runner 协议对结果影响很大。大图 matched continual real-node baseline 已另行纳入 v3 主表，外部方法因协议不匹配仍列为 reference。

## 1. 总体判断

当前项目已经有一条可以继续推进的研究主线，但还没有形成可以直接声称“普遍有效”的最终方法：

> 在严格 v3 持续协议下，用 student response curve 评价新增 synthetic block，并冻结历史 synthetic topology，可以改善新类塑性与历史保持之间的折衷；但候选路径对 selection seed 仍然敏感，且跨数据集收益并不稳定。

当前最重要的工作不是继续堆叠手工模块，而是依次完成：

1. 把协议、数据划分、训练预算、选择预算和报告指标彻底固定。
2. 在更大数据集上建立可信的固定 append、recluster、k-center、herding、random 和 SCA 对照。
3. 将 SCA 的选择稳定性从“单次 ranking”提升为“风险感知的路径决策”。
4. 解决历史记忆过时、文本集合利用不足、每类预算固定这三个真正的建模缺口。
5. 只有当新模块在多个数据集和独立 seeds 上成立，才把它提升为论文主方法。

## 2. 已有证据与边界

### 2.1 已经有实验支持的结论

- v3 canonical student protocol 已统一：student seeds `100..105`、batch `16`、steps=`max(50,min(2048,16*roots))`，dev/test 由原始 id 的模 `10` 划分，selection 不接触 test。
- `frozen-old` topology 与 response-based selection 的作用已经在 Cora/Citeseer 分开做过归因，不能把两者的收益混成一个黑箱增益。
- response curve 比单一短训终点更有解释性，但并没有证明它能够稳定预测 full-step canonical ranking。
- Cora 的 selection seed 改变会导致不同父路径，说明 greedy selector 存在真实路径不稳定，不是单纯的偶然日志问题。
- beam2 和 terminal full-step lookahead 可以缓解 premature pruning，但不能宣称已经解决路径不稳定。
- `no_reencode` 在多数已有数据集上变差，说明融合文本后重新编码是有效组件；这不等于“简单拼接就是最优文本建模”。
- 小规模 closed continual 结果中，fixed append 在 Cora/Citeseer 上有竞争力，但在 arXiv 上并不普遍优于 full-history 或 raw reference。

### 2.2 当前不能写成论文强结论的内容

- v3 的 `rate=0.05`、`G=5`、每类 `15 rows` 是否跨数据集最优。
- bounded refresh、动态每类预算、文本 set encoder 是否优于当前实现。
- SCA 是否能在 WikiCS、BookHis 以及更多独立 selection seed group 上稳定有效。
- beam 是否比 greedy 普遍更强。
- 当前方法是否优于最新持续图凝聚 SOTA。外部方法已完成协议资格审计，但没有公开方法同时满足 v3 的 TAG、student、zero-shot 和 frozen-old continual 条件，因此不能直接排名。

## 3. 方法主线的建议升级

建议把最终方法组织成三层，而不是一次性把所有候选都塞进主方法：

### Layer A：已验证核心

**Response-calibrated frozen-old append**：

- 用多个 horizon 的 student response curve 评价候选 synthetic block。
- 用 step、batch 和 selection seed 的 ranking consistency 估计候选风险。
- 保持旧 memory 的旧边不被重写，只构造新块内部边和受限的跨块边。
- 用独立 canonical student seeds 做最终确认。

这一层是目前最适合写成核心机制的部分。

### Layer B：高价值创新候选

**Bounded-refresh memory**：

- 旧 memory 不再永久不可变。
- 每个旧 row 带有 `birth_stage`、`mass`、`last_supported_stage`、稳定性和漂移统计。
- memory 分为 protected core 与 elastic reservoir。
- core 保留历史覆盖，reservoir 允许在预算不变时被低频替换或轻量修正。
- 旧旧拓扑默认仍保持，只有被批准的 reservoir row 及其局部边可以更新。

这个方向直接解决 frozen-old 在分布漂移下可能过时的问题，同时保留历史保持的可解释性。

### Layer C：自适应构造

- 用文本 set encoder 或确定性 attention pooling 处理一个 cluster 中的多条文本，而不是只把若干文本直接拼接。
- 根据类内 dispersion、文本/结构漂移、历史冗余度和类不平衡动态分配新 rows。
- 将 append block 与旧 memory 分成不同管理区，而不是所有 rows 使用同一种保留规则。

Layer C 必须在 Layer A 的统一基线之上逐个加入，避免无法解释增益来源。

## 4. 六个关键问题的处理方案

### 4.1 v3 超参数是否跨数据集可靠

当前结论：**协议可靠，数值最优性未证实**。`batch=16`、student seed 集合和 steps 规则是复现协议，不应在每个数据集上随意改动；但 `rate`、`G`、字符预算、跨块 degree 和 rows/class 可能随图规模、类内冗余和文本长度变化。

采用两套设置：

1. **Universal v3**：所有数据集使用同一组默认值，作为公平主表。
2. **Calibrated v3**：只在 dev/pool 上做有限候选网格选择，冻结后在 test 上一次运行。

建议校准网格保持很小，例如：

- `rate`: `{0.02, 0.05, 0.10}`
- `G`: `{3, 5, 8}`
- text char budget: `{192, 320, 512}`
- topology cross degree: `{1, 2}`
- rows floor: `{8, 15, 24}`

校准报告必须包括搜索空间大小、选择 split、选择 seed、最终固定配置和 test 一次性结果。若 calibrated v3 只在一个数据集有效，不升级为通用方法，而作为 dataset-aware calibration 分析。

### 4.2 历史 memory 不更新是否有问题

有问题的情形主要有三类：新阶段语义漂移、旧类内部结构改变、早期 synthetic rows 被后续 rows 覆盖。永久 frozen 能保护历史，但可能使旧表示逐渐失配。

按以下顺序验证：

1. `frozen-old`：当前主基线。
2. `re-score-only`：不改旧 x 和边，只更新旧 row 的重要性、置信度和保留优先级。
3. `bounded-refresh`：只允许 elastic reservoir 中不稳定且高冗余的 rows 被替换，core 和旧旧边保持不变。
4. `global/recluster`：作为压力测试和负对照，不作为默认方法。

刷新必须有预算上限，例如每阶段最多替换旧 rows 的 `10%` 或 `20%`，并记录被替换 row 的 birth stage、原因和边变化。主指标除 current F1、cumulative F1 外，增加 old-class retention、old row survival ratio、edge churn 和 refresh ratio。

### 4.3 文本不应只做简单拼接

现有流程是：cluster 选出代表文本，按顺序融合成一段文本，再用 frozen MiniLM 重新编码。已有证据支持 re-encode，但尚未支持 concatenation 本身最优。

建议按风险从低到高比较：

1. **Weighted text pooling**：分别编码 cluster 内文本，再按离中心距离、mass 或置信度加权平均，最后 L2 normalize。
2. **Set attention pooling**：在 frozen MiniLM embedding 上使用无训练或 dev-only 轻量 attention/set pooling，保持输入顺序不敏感。
3. **Dual prototype**：graph prototype 与 text prototype 分开生成，student 通过两个对齐项学习，而不是先在输入侧拼成一个构造向量。
4. **Learnable text encoder**：只有当前面三项在多数据集上稳定后再考虑，否则容易把 student capacity 和 text construction 增益混在一起。

每种文本方式都必须保留相同的 cluster、rows、student、steps 和 topology，单独做 `concat / pool / set / dual` 消融。

### 4.4 每类 15 rows 是否应动态调整

`15` 应被视为当前 floor 和协议默认值，而不是理论常数。动态分配的依据应来自可观测数据统计，而不是 test F1：

- 类内结构 dispersion。
- 类内文本 dispersion。
- 当前 delta 与历史 memory 的漂移距离。
- synthetic coverage 或近邻冗余度。
- 类频率和严重不平衡程度。

建议总预算固定为 `B_t`，先为每类保留最小 floor，再把剩余预算按 priority 分配：

`priority_c = dispersion_c + drift_c + imbalance_c - redundancy_c`

第一版只实现确定性统计分配，并设置上下限，避免动态规则变成隐藏调参。需要对比：

- fixed `15/class`。
- proportional to class size。
- dispersion-aware allocation。
- drift-aware allocation。
- dispersion + drift + redundancy。

### 4.5 后续 SCA 优化是否可靠

当前答案是：**部分可靠，部分仍是负结果**。

- 已支持：response curve 有机制解释力；frozen-old topology 有独立归因价值。
- 部分支持：beam2、terminal lookahead 能保留或找回部分较好路径。
- 未支持：selection seed 稳定性；短训 ranking 对 canonical ranking 的普遍预测能力。
- 已观察负结果：仅切换 batch-16 不能解决 Cora 的 ranking mismatch；换 selection seed 仍会改变最终路径。

因此主方法应报告 selection uncertainty，而不是隐藏它。至少报告不同 selection seed group 的 path frequency、rank correlation、top-1 agreement、canonical regret 和最终 F1 均值/标准差。

### 4.6 多数据集、SOTA 和 baseline 对比

已有数据集覆盖 Cora、Citeseer、arXiv、WikiCS、BookHis、BookChild、SportsFit；完整 SCA continual 已在 Cora、Citeseer、arXiv、BookChild、SportsFit 完成，WikiCS/BookHis 当前仍只有核心 append continual 结果。

现有 baseline 包括 random real、k-center、herding、kmeans-medoid、append_fixed、recluster、full-history、raw、no_fuse、no_reencode、random-init 和部分 official GraphCLIP 结果。k-center 在当前旧表中看起来较强，但已经发现 batch、steps 和 runner 不一致会改变排名，因此仍只能作为强基线，不能直接宣称普遍优于 ours。

最终对比至少分为三组：

1. **Synthetic construction baseline**：random、herding、kmeans-medoid、k-center、当前 deterministic construct。
2. **Continual memory baseline**：fixed append、recluster、full-history、bounded refresh、global rebuild。
3. **SCA components**：single endpoint、response curve、rank consistency、frozen-old、beam/lookahead、dynamic rows、text pooling。

最新 TAG continual 方法和 SOTA 必须用同一数据划分、同一 student、同一 memory budget、同一训练步数和同一指标重跑；若无法做到，应单独标为 external reference，不与主表直接排序。official GraphCLIP 若使用不同 backbone、GPS/ego-subgraph 或训练目标，也只能作为背景参考。

## 5. 分阶段执行计划

### P0：协议与实验基础设施冻结

目标：让任何后续增益都能被归因。

- 固定 v3 配置对象、数据 split、pool exclusion、student seeds、selection seeds、batch、steps 和 memory budget。
- 所有 runner 统一输出 manifest：dataset、protocol hash、method、seed、batch、steps、rows、edges、runtime、git state。
- 统一输出 current F1、cumulative F1、old F1、macro/weighted 指标、均值和标准差。
- 统一记录 test 是否被访问，selection 是否只使用 dev/pool。
- 将 k-center、herding、random、fixed append、recluster 接入同一 runner。

通过标准：同一方法重复运行结果可复现；不同 runner 在相同设置下差异低于日志允许误差；旧结果可以被 manifest 解释。

### P1：Universal v3 与 Calibrated v3

目标：回答“v3 默认值是否跨数据集可靠”。

- 先在 Cora/Citeseer 复核，再扩展到 arXiv、WikiCS、BookHis、BookChild、SportsFit。
- 每个数据集先跑 Universal v3，再只用 dev 做有限 calibrated v3。
- 单独报告校准收益、搜索成本和跨 seed 稳定性。

通过标准：Universal v3 作为公平默认；Calibrated v3 只有在至少两个数据集获得稳定收益，且没有明显 old retention 损失时，才进入方法扩展。

### P2：历史 memory 管理

目标：检验 frozen-old 是否因长期不更新而过时。

实验顺序：`frozen-old -> re-score-only -> bounded-refresh-10% -> bounded-refresh-20% -> global/recluster`。

重点分析：漂移阶段、旧类 retention、row survival、edge churn、memory 总规模和运行时间。

通过标准：bounded refresh 至少在两个数据集上提高 cumulative F1 或 old retention，且 refresh ratio 和 edge churn 受控；否则保留 frozen-old，并把 stale-memory 作为负结果。

### P3：文本集合建模

目标：从“拼接文本”升级为可解释的 cluster-level text prototype。

固定 graph construction，只替换 text operator：`concat -> weighted pooling -> set pooling -> dual prototype`。

通过标准：至少三个数据集上平均提升或在相同 F1 下显著降低 std；若只在某个数据集提升，则作为数据特性分析，不作为主创新。

### P4：动态 rows 与 memory budget

目标：验证每类固定 15 rows 是否浪费预算或无法覆盖困难类别。

- 总预算固定，比较 fixed、class-size、dispersion、drift、dispersion+drift-redundancy。
- 输出每阶段每类实际 rows、coverage、redundancy、class-wise F1 和 budget utilization。
- 动态分配不能读取 test labels 或 test F1。

通过标准：在相同总 rows 下，多数据集平均 cumulative F1 提升，同时不牺牲 minority/old classes。

### P5：SCA 风险感知选择

目标：解决 response proxy 与 canonical ranking 失配。

优先级：

1. 独立 selection split 或 selection-only calibration pool。
2. 多 horizon trajectory 特征，而不是一个短训终点。
3. bootstrap/seed ensemble 估计 ranking uncertainty。
4. 风险惩罚：`utility = mean - lambda * std - gamma * canonical-regret-proxy`。
5. 低宽度 beam 或 terminal confirmation 作为可选计算预算。

主表必须同时给 greedy、beam2 和 oracle/full confirmation；oracle 只用于机制分析，不能伪装成可部署 selector。

### P6：完整多数据集与论文对比

目标：形成可审计的主表、消融表和负结果表。

- 7 个数据集分为小图、中图、大图，报告规模、类数、文本长度和类内冗余。
- 所有方法使用相同 student 和 memory budget。
- 最新 continual TAG 方法和公开 SOTA 只在协议真正匹配时进入主表。
- 不匹配的方法进入 external reference 表，并明确 backbone、目标、预算和 split 差异。

通过标准：至少 3 个数据集有完整 continual 对比；至少 2 个数据集有独立 selection seed confirmation；主结论不依赖单个数据集或单个 seed。

## 6. 推荐实验矩阵

| 组别 | 方法 | 目的 |
|---|---|---|
| Construct | random / herding / kmeans-medoid / k-center | 判断构造本身是否真正有益 |
| Memory | fixed append / recluster / full-history / global rebuild | 判断 append 与历史重建的差异 |
| Core ours | deterministic construct + frozen-old | 建立可靠主基线 |
| Selection | endpoint / response curve / rank consistency | 归因 SCA 机制 |
| Path | greedy / beam2 / terminal confirmation | 测试路径风险 |
| Memory upgrade | re-score-only / bounded refresh | 测试历史过时问题 |
| Text | concat / re-encode / weighted pool / set pool / dual | 测试文本利用方式 |
| Budget | fixed 15 / class-size / dispersion / drift / joint adaptive | 测试动态 rows |
| Reference | raw / official GraphCLIP / matched continual SOTA | 给出上界和外部位置 |

每个单元至少记录：当前阶段 F1、累计 F1、旧类 F1、新类 F1、均值、标准差、selection path、rows、edges、训练步数、运行时间和显存峰值。

## 7. 论文创新点的最终筛选规则

可以进入主贡献的条件：

- 有清晰机制，不只是额外模块。
- 在至少两个数据集上独立 seeds 成立。
- 与最接近 baseline 使用同一协议。
- 有反例、消融和失败边界。
- 不依赖 test 调参。

优先级建议：

1. **主贡献候选**：student-response-calibrated selection + topology-preserving append。
2. **第二贡献候选**：bounded-refresh memory，前提是多数据集有效。
3. **第三贡献候选**：动态 rows 或 cluster-level text set prototype，二选一优先，不要同时引入过多自由度。
4. **分析贡献**：response/canonical ranking mismatch、path instability、k-center 配置敏感性和 frozen-memory 的负结果。

如果 P2、P3、P4 都不能稳定提升，论文仍可围绕“为什么简单 append 在某些设置下可靠、哪些机制会失败”形成机制型工作；这比把多个不稳定组件包装成 SOTA 更可信。

## 8. 近期执行顺序

接下来按以下顺序推进：

1. 先完成 P0：统一 runner、manifest、指标和 k-center 配置。
2. 运行 Cora/Citeseer 的 Universal v3 与全部关键 continual baseline。
3. 扩展 fixed append、k-center、herding、random 到 arXiv、WikiCS、BookHis、BookChild、SportsFit；当前已完成大图 canonical append/SCA 主表，WikiCS/BookHis 的额外 memory arms 仍属于后续扩展。
4. 在基线可信后，运行 P1 的 dev-only hyperparameter calibration。
5. 再实现 bounded-refresh，并先做 Cora/Citeseer 的小矩阵。
6. 之后只选择一个最有希望的方向做深：text set prototype 或 dynamic rows。
7. 已完成独立 selection seed、matched continual baseline 和论文主表；外部方法因协议不匹配保留为 reference，不做伪同协议排名。

## 11. 目前的交付结论

已完成的是一套可运行、可审计的 v3 统一实验基础设施，以及 Cora/Citeseer/WikiCS/BookHis 的核心 continual 复核、
Cora/Citeseer/WikiCS 的 static baseline、小图机制消融和 arXiv/BookChild/SportsFit 的 canonical continual
主表。三套大图还完成了 `recluster/full_history` 诊断，外部方法已完成严格协议资格审计。

计划中“可在当前项目内完成”的内容已完成；剩余工作是论文增强项：外部方法的协议适配重实现、更多独立
selection seed group，以及 bounded refresh、动态 rows、cluster-level text prototype 在七数据集上的扩展。
这些内容不应被写成当前已经验证的性能优势。

## 12. 2026-09-15 执行补充

为完成大图 canonical continual，统一 runner 增加了两项只影响执行效率、不改变 v3 任务定义的实现：

- 全量文本 embedding 在每个不可变 run 目录中缓存，stage 和 arm 共享同一份缓存；
- 当 raw pool 达到大图阈值时，候选 block 使用 GPU 分块 max-min/Lloyd，memory topology 使用分块 top-k，避免 CPU 全量 `cdist`；小图仍使用原有 deterministic mass-weighted Lloyd。

arXiv 不使用统一 runner 临时生成的类别顺序，而是优先读取已经冻结的 `classincr3` 三阶段划分（45,199 / 45,122 / 45,152 pool nodes，budget 2,259 / 4,516 / 6,773）。这一步保证新结果能和既有 arXiv 类别增量证据对齐。

2026-09-15 已完成三套大图 canonical：arXiv、BookChild、SportsFit，均使用六个 canonical student seeds、独立
selection seeds `300..302`、batch grid `{8,16,32}`、四个候选 recipe，并同时记录 `append_fixed` 与 SCA；随后
串行完成大图 `recluster/full_history` 诊断臂，结果已汇总到 `MATCHED_SOTA_COMPARISON_V3_ZH_2026-09-15.md`。

随后已将 `random_real`、`k_center`、`herding`、`kmeans_medoid` 接入同一 continual runner，并完成三套大图
的六 seed 重跑。所有 matched baseline 与机制臂都使用同一 stage split、累计预算、student batch/steps、
student seeds 和 `test_accessed=false` manifest；原始结果分别位于 `results/baseline_v3_2026_09_15_{arxiv,bookchild,sportsfit}/`。

## 13. 2026-09-15 收口结果

- canonical 大图主表已完成：arXiv、BookChild、SportsFit。
- 七数据集结果已汇总：Cora、Citeseer、WikiCS、BookHis、arXiv、BookChild、SportsFit。
- SCA stage-2 相对 `append_fixed`：arXiv `-0.40`、BookChild `+1.07`、SportsFit `+0.31`；小图 Cora/Citeseer
  在当前 formal runner 下与 append 持平，因此当前最稳妥结论是“机制有条件收益”，不是“普遍提升”。
- `recluster` 和 `full_history` 已作为历史管理诊断完成；它们的结果说明历史重建既可能改善保持，也可能牺牲
  当前类塑性，不能直接替代 frozen-old append。
- 外部 SOTA 已完成严格同协议资格审计；PUMA、GECC、CCC、CA-DGCL、CaT、OpenGC 和 TAGSAM 等应留在
  closest external/reference，当前没有公开方法同时满足 v3 的 TAG、label-free、zero-shot、class-incremental
  和 frozen-old 条件。

## 9. 当前决策

- 不把 `15 rows/class`、`G=5` 或 `rate=0.05` 宣称为普适最优。
- 不把 frozen-old 当作最终记忆机制，先验证 bounded refresh 的必要性。
- 不把文本拼接当作最终创新，先比较 set/pooling/dual prototype。
- 不把 beam2 或 lookahead 写成已经解决 selector instability。
- 不把旧版 k-center 表直接当作最终结论；arXiv、BookChild、SportsFit 已在统一 runner 下重跑，结果显示其优势具有数据集依赖。
- 即使内部 matched 主表已完成，也不宣称 SOTA；外部方法没有同时满足 v3 的 TAG、zero-shot 和 frozen-old continual 条件。

## 10. 相关材料

- `docs/current/SCA_V3_RESPONSE_TOPOLOGY_OPT_RESULTS_ZH_2026-09-14.md`
- `docs/current/V3_CLOSED_CONTINUAL_RERUN_RESULTS_ZH_2026-09-14.md`
- `docs/current/LOCAL_CURRENT_V3_AUDIT_ZH_2026-09-14.md`
- `docs/current/FINAL_TEST_AND_BASELINES_ZH_2026-09-11.md`
- `docs/current/NEW_BENCHMARKS_RESULTS_ZH_2026-09-11.md`
- `code/v2_llm_fusion/protocol_v3.py`
- `code/frozen_v1_2026_09_09/run_student_compatible_append.py`
