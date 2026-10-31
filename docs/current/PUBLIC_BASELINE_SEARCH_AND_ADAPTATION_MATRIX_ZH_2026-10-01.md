# 公开图凝聚基线检索与统一协议适配矩阵

日期：2026-10-01  
状态：检索与适配设计完成；第一批大图 strict-continual 对比已完成，完整数字见 [统一协议公开基线对比报告](PUBLIC_CONTINUAL_BASELINE_COMPARISON_LARGE_ZH_2026-10-01.md)。TAGSAM、PUMA 和 STAD 仍按本文的 adapted/oracle/native 分层推进。

## 1. 先回答协议问题

当前 `Bounded Student-Compatible X+P` 使用的是一个组合协议：

- text-attributed graph；
- 逐阶段类别增量 `base+4`；
- 每阶段只允许访问当前合法 delta 和上一步 synthetic memory；
- 历史完整 raw graph 不可回读；
- 固定约 5% cumulative memory rate；
- 两层 GCN、可训练 text projection、双向 graph-text InfoNCE Student；
- 每阶段从头训练 Student，而不是继承上一阶段 Student 参数；
- 真实完整图上的 prompt-based report-dev 评估。

目前没有找到一个公开方法同时原生满足上述全部条件。因此，公开方法必须分成三类，而不能声称“已有同协议官方 baseline”：

| 类别 | 含义 | 是否进入严格主表 |
|---|---|---|
| `continual-adapted` | 保留公开方法的凝聚/持续机制，只替换输入接口、文本处理和统一 Student | 可以，单独标明 adapted |
| `static/offline reference` | 每阶段可以读取累计 raw，或一次性凝聚全图 | 可以作为离线参考，不能作为合法 continual 主比较 |
| `native/reference-only` | 监督 Student、不同下游模型、不同标签或不同任务定义 | 只能单列参考，不与当前主方法直接排序 |

因此论文主表应写成“same-protocol mechanism-adapted comparison”，而不是“官方方法原样复现”。

## 2. 当前主方法和统一评价协议

本轮基线适配围绕 `Bounded Student-Compatible X+P`，不把可选拓扑修正 A 默认并入主方法。

| 项目 | 冻结设置 |
|---|---|
| 数据集 | arXiv、BookChild、SportsFit；小图 Cora、Citeseer 用于机制和适配 smoke |
| 任务 | base+4，五个阶段；小图先用既有三阶段顺序做适配验证 |
| memory | 5% fixed-rate，使用主 runner 的累计 rows 轨迹 |
| 构造输入 | 当前 delta + parent synthetic memory；strict continual 禁止历史 raw |
| Student | 原两层 GCN、冻结 MiniLM、trainable text projection、双向 InfoNCE |
| Student 训练 | 每阶段从头训练；batch=16；lr=0.001；`max(50,min(2048,16K))` steps |
| Student seeds | 100--105；新增 construction seed 单独记录 |
| 指标 | Final Macro-F1、Accuracy、stage-average、old F1、new F1、worst-stage |
| 成本 | 构造时间、Student 时间、rows、tensor bytes、文本 bytes/tokens、edge entries、峰值 GPU memory |
| 审计 | source locality、预算、旧 prefix、边端点、checkpoint reload、report/test 标签隔离 |

同一协议下，所有 baseline 只改变 memory 构造机制；如果改变 Student、标签权限、历史 raw 权限或评估任务，则换表。

## 3. 检索到的最相关公开方法

### 3.1 TAGSAM：静态 TAG 凝聚主基线

- 论文：[Text-attributed Graph Condensation via Text Selection and Attribute Matching](https://arxiv.org/abs/2606.03839)
- 代码：[SundayVHan/TAGSAM](https://github.com/SundayVHan/TAGSAM)，检索到的 HEAD：`d812f06fd16db3cd4d497804a8e504d2c738faa8`
- 核心机制：
  - `Subgraph Text Selection`：从相关节点文本中选择/合并文本片段；
  - `Attribute Similarity Matching`：在稳定的属性相似度空间中匹配合成节点与拓扑；
  - 使用 teacher/Student 和可微 synthetic feature 优化，但原任务是静态 TAG 凝聚。

需要做的两个版本：

1. `TAGSAM-stage-local`：每个阶段只把当前 delta 传给 TAGSAM 机制，输出新 block，再追加到 parent。它最接近 strict continual，但并非官方原实验。
2. `TAGSAM-global-oracle`：每阶段读取截至当前阶段的全部合法 raw，再重新凝聚相同总 rows。它用于测量“全局重凝聚能力”，不进入 strict continual 排名。

适配注意：文本选择可保留原始片段选择逻辑；下游改为统一 InfoNCE Student 时，必须标注为 `mechanism-adapted`。不能把 TAGSAM 原论文中的监督/模型数字直接与当前方法比较。

### 3.2 CaT：最直接的持续图凝聚机制

- 论文：[CaT: Balanced Continual Graph Learning with Graph Condensation](https://arxiv.org/abs/2309.09455)
- 代码：[superallen13/CaT-CGL](https://github.com/superallen13/CaT-CGL)，固定 commit：`43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb`
- 核心机制：当前到达图先凝聚成 synthetic replay block，写入 Condensed Graph Memory；后续在 memory 中训练，避免新 raw 图与小 memory 的规模失衡。
- 原方法差异：数值特征、监督图学习、TiM 训练，不包含文本和 InfoNCE。

当前已有 `CaT-MMD-TAG` 适配器。它保留类条件 MMD 更新 X、按类配额和逐阶段 memory 追加；文本采用合法当前 delta 中的近邻片段拼接并重新编码；Student 换成当前统一 InfoNCE Student。结果只能命名为 `CaT-MMD-TAG adaptation`。

### 3.3 PUMA：CaT 的更强持续机制候选

- 论文：[PUMA: Efficient Continual Graph Learning for Node Classification with Graph Condensation](https://arxiv.org/abs/2312.14439)
- 代码：[superallen13/PUMA](https://github.com/superallen13/PUMA)，固定 HEAD：`9e4e87f53db0da6aacad4a0a9d08f1c2970413d0`
- 核心机制：在 CaT 基础上使用伪标签、一次传播、较宽的随机编码器/特征匹配和 edge-free synthetic memory，配合 continual retraining。

PUMA 不能用 CaT 改名代替。若适配，至少必须保留并记录：

- pseudo-label 生成规则；
- one-time propagation；
- edge-free memory；
- 原方法的重训/训练 in memory 语义；
- 伪标签是否允许使用当前真实标签和 report-dev。

第一阶段只做 `PUMA-inspired-TAG` smoke 和机制卡；通过 source/budget audit 后再跑正式矩阵。伪标签若改变了标签权限，必须单独分层。

### 3.4 GECC：演化图的聚类凝聚

- 论文：[Scalable Graph Condensation with Evolving Capabilities](https://arxiv.org/abs/2502.17614)
- 代码：[rockcor/GECC](https://github.com/rockcor/GECC)，固定 commit：`d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f`
- 核心机制：传播后的特征空间中进行类内聚类；扩展时继承旧 condensation centers 作为聚类初始化；强调可追踪性和大规模效率。

当前已有 `GECC-hard-TAG` 适配。strict continual 版本只能对当前 delta 做新类聚类并继承旧中心；不能声称复现官方累计 raw、fuzzy clustering 或全部 evolving 优势。global raw evolving 版本应另列 offline/reference。

### 3.5 STAD：静态 TAG 强参考，但不是同协议 continual baseline

- 论文：[Semi-Supervised Text-Attributed Graph Distillation](https://arxiv.org/abs/2607.20477)
- 代码：[HKBU-LAGAS/STAD](https://github.com/HKBU-LAGAS/STAD)，固定 HEAD：`769141b521714e75bac3b02ad0e34604af454b7a`
- 核心机制：图感知/图无关双路径编码、协同自训练伪标签、Wasserstein distance 图 sketching、关键词/LLM 文本合成。
- 原方法差异：半监督静态 TAG distillation；代码包含监督 GNN/多模型训练路径。

STAD native 不能放进当前 InfoNCE Student 主表。可报告两个参考：

1. `STAD-native`：论文/官方 Student 和监督标签口径，作为静态 TAG 能力参考；
2. 只有在实现统一 InfoNCE adapter 后，才可命名 `STAD-mechanism-adapted`。

不得把 `STAD-native` 的监督 GCN 结果解释为当前主方法的直接优劣。

### 3.6 其它持续/动态图凝聚相关工作

这些工作说明持续/演化图凝聚不是空白，但任务定义不同，优先作为 related work 和第二阶段参考：

| 方法 | 论文 | 机制 | 当前比较角色 |
|---|---|---|---|
| OpenGC | [Graph Condensation for Open-World Graph Learning](https://arxiv.org/abs/2405.17003) | open-world 分布变化和时间不变凝聚 | offline/open-world reference |
| CCC | [Condensation-Concatenation Framework for Dynamic Graph Continual Learning](https://arxiv.org/abs/2512.11317) | 动态快照凝聚、历史 embedding 拼接、变化区域 replay | dynamic snapshot reference |
| CA-DGCL | [Dynamic Graph Continual Learning via Condensation and Attachment](https://arxiv.org/abs/2607.11112) | 跨时间节点链、Tucker 分解和新节点 attachment | dynamic graph reference |
| DyGC | [Dynamic Graph Condensation](https://arxiv.org/abs/2506.13099) | 时空状态匹配和动态结构生成 | dynamic-GNN reference |
| DANCE | [Dynamic, Available, Neighbor-gated Condensation for Federated TAGs](https://arxiv.org/abs/2601.16519) | round-wise model-in-the-loop refresh、neighbor gate、provenance | federated TAG related work；暂不主表 |

它们不能直接被改成一个“看起来相同”的数字：CCC/CA-DGCL/DyGC 保存或更新的是动态图快照、节点链或动态模型状态，DANCE 还引入联邦 round 和全局模型。若后续适配，必须另设 `dynamic/federated` 表。

## 4. 统一比较矩阵

### 4.1 第一批严格 continual 主表

第一批只纳入能在 `delta_t + parent synthetic` 接口下定义清楚的方法：

| 方法 | 原机制 | 当前输入 | 历史 raw | 文本 | 拓扑 | 角色 |
|---|---|---|---:|---|---|---|
| random real coreset | 随机选择 | 当前 delta | 否 | 原文 | 统一/原生各一版 | 下界 |
| k-center / medoid | 几何覆盖 | 当前 delta | 否 | 原文 | 原生或统一 edge-aware | 强 coreset |
| CaT-MMD-TAG | 类条件 MMD X | 当前 delta + parent | 否 | delta 片段适配 | edge-free | mechanism-adapted |
| GECC-hard-TAG | propagated-feature clustering | 当前 delta + parent | 否 | delta 片段适配 | edge-free | mechanism-adapted |
| PUMA-inspired-TAG | pseudo-label + one-time propagation | 当前 delta + parent | 否 | 待定义 | edge-free/原生 | mechanism-adapted |
| TAGSAM-stage-local | text selection + attribute matching | 当前 delta + parent | 否 | 原机制 | 原机制适配 | mechanism-adapted |
| Bounded Student-Compatible X+P | outer Student-compatible X/P | 当前 delta + parent | 否 | 当前固定文本 | edge-aware C0 topology | 主方法候选 |

### 4.2 静态/离线参考表

| 方法 | 允许访问 | 是否改变 Student | 作用 |
|---|---|---:|---|
| TAGSAM-global-oracle | 累计 raw | 统一 Student | 全局 TAG 凝聚上限参考 |
| full raw equal-step | 全量 raw | 统一 Student | 相同训练预算的数据参考 |
| full raw long-train | 全量 raw | 统一 Student | 充分训练数据参考 |
| STAD-native | 全量静态图/监督标签 | 是 | 静态 TAG 能力参考 |
| GCond/SFGC/SimGC/GEOM/CTRL | 全量 raw、通常监督标签 | 通常是 | 经典静态 GC 参考，不与主表直接排序 |
| direct Student continual | 当前 delta + Student state | 保留状态 | 非 memory-only 的训练控制 |

## 5. 适配规则

每个 baseline 必须生成一个 `baseline_card.json`，并记录：

```text
paper_title / paper_url / code_url / commit
native_input / native_labels / native_student / native_memory
historical_raw_access
adapted_input / adapted_text / adapted_topology / adapted_student
budget / construction_seed / student_seeds
source_locality_audit / prefix_audit / topology_audit / reload_audit
status: native | mechanism-adapted | controlled-content | oracle
```

适配不得发生以下情况：

1. 用最终 report/test F1 选择构造参数；
2. 把历史 raw 访问隐藏在缓存或全图 embedding 中；
3. 只替换方法名称而不实现 PUMA/TAGSAM/STAD 的独有机制；
4. 用监督 GCN 数字与 InfoNCE Student 数字直接排冠军；
5. 让某个 baseline 使用不同 memory rows、batch、steps 或 report mask；
6. 把同 C0 topology 的内容控制实验与方法自行构图的端到端实验混在一起。

## 6. 运行顺序

### 阶段 0：检索和 smoke

- 完成 TAGSAM、PUMA、STAD、CaT、GECC 的 `baseline_card.json`；
- 运行官方代码 smoke 或读取固定 commit；
- 确认输出 memory 的字段、文本和边是否能转换为当前 Student；
- 不跑完整矩阵，不根据 smoke 分数选配置。

### 阶段 1：先跑最容易公平的 strict adapters

1. 当前 `CaT-MMD-TAG` 和 `GECC-hard-TAG` 在当前大图 base+4 协议迁移；
2. 同时跑 `k-center/medoid/random` 的 edge-aware 版本；
3. 与 `append-edge`、`Bounded X+P` 做同进程 paired evaluation；
4. 汇总 Final F1、Accuracy、old/new F1、stage-average 和成本。

### 阶段 2：TAGSAM

1. 先做 `TAGSAM-stage-local` 单阶段 smoke；
2. 审计文本来源只来自当前 delta；
3. 通过后跑两张小图和一个大图；
4. 最后做 `TAGSAM-global-oracle`，单列 offline 表。

### 阶段 3：PUMA 和 STAD

- PUMA：先实现伪标签、one-time propagation、edge-free memory，再进入主表；
- STAD：先跑 native static reference；统一 InfoNCE 适配成本较高，暂不把 native 结果当同协议主比较。

## 7. 当前结论和论文写法

当前能客观写的是：

> 由于现有公开方法分别研究静态 TAG 凝聚、监督持续图凝聚或动态图凝聚，没有一个方法与本文的 text-attributed、historical-raw-inaccessible、memory-only InfoNCE Student 协议完全重合。本文因此对 CaT、GECC 和 TAGSAM 进行机制保持的接口适配，并把 full-history TAGSAM 与 STAD native 作为离线/静态参考单独报告。

不能写：

- “没有任何持续图凝聚工作”；
- “TAGSAM 官方结果与本文直接可比”；
- “CaT/GECC 适配分数就是其论文结果”；
- “STAD 监督 GCN 超过/低于本文证明了 TAG Student 的优劣”。

现有本地旧矩阵中的 `CaT-MMD-TAG`、`GECC-hard-TAG`、k-center 和 medoid 结果仍然有效，但它们属于 Cora/Citeseer、旧三阶段协议和单构造 seed。迁移到当前 arXiv/BookChild/SportsFit 的 edge-aware base+4 协议后，必须生成新结果目录，不能与旧数字直接拼接。

## 8. 参考代码和证据位置

- 现有适配器：[run_external_continual_adapters_20260927.py](../../code/v2_llm_fusion/run_external_continual_adapters_20260927.py)
- 机制重实现：[external_continual_mechanisms.py](../../code/v2_llm_fusion/external_continual_mechanisms.py)
- 现有适配协议：[EXTERNAL_CONTINUAL_ADAPTER_PROTOCOL_ZH_2026-09-27.md](EXTERNAL_CONTINUAL_ADAPTER_PROTOCOL_ZH_2026-09-27.md)
- 现有外部矩阵结果：`results/external_continual_adapters_20260927_v2/`
- 当前主方法报告：[XPA_EDGEAWARE_DECOMPOSITION_GATE_REPORT_ZH_2026-10-01.md](XPA_EDGEAWARE_DECOMPOSITION_GATE_REPORT_ZH_2026-10-01.md)

