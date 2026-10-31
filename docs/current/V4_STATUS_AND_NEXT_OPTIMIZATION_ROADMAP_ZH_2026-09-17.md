# v4 版本、P0-P5 状态与后续优化路线

> 2026-09-19 状态指引：下文保留早期路线和当时决策，不再作为最新完成状态。P4 最低闭环及 P5 小图/大图首轮已执行，候选未晋升；当前共同控制是 C0（response off），旧 v4 保持归档。最新见 [P5 总进展](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)和 [指标口径审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。TAGSAM/STAD 补强继续暂停。

日期：2026-09-18。本文是当前计划和状态的统一入口；历史报告、代码入口和结果目录不重命名、不覆盖。已按同日后续要求补完冻结 v4 的小图顺序确认与成本账，没有将新候选模块合并到 v4。

> 最新验收：[P2-A 图条件文本](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md) 已完成 864 fits，未通过晋升；真实邻域 query 未稳定优于打乱邻域。此前 [P3-B 拓扑诊断](P3B_TOPOLOGY_DIAGNOSTIC_COMPLETION_ZH_2026-09-18.md) 支持跨边有效但无稳定替代规则。有界跨边和单向跨块传播按用户要求后置，不在 P2-A 中修改 student 算子。

此前的小图扩展与成本报告仍有效：两图 x 三顺序 x 13 臂 x 三阶段 x 六种子，共 1404 次报告训练；v4 平均最终 F1 为 47.47/55.48，k-center 为 49.40/57.73。凝聚效用存在，但稳定优于 coreset 与一次使用加速均不成立。按用户决定继续暂停 TAGSAM 补强和 STAD 检查，不启动大图。

## 1. 先给结论

不是“只做到了 P2，后面完全没做”，而是：**P0 基础设施已完成；P1 已形成可运行的小图主线，但稳健性验收不完整；P2 实验收尾但新文本机制没有晋升；P3 部分机制提前进入 v4；P4 尚未完成当前协议的系统验证；P5 已提前做了一部分小图外部比较。**

后续保留 2026-09-15 的 P0-P5 编号，不再另起一套同名阶段。近期执行顺序建议为：

> 小图扩展与成本报告已完成 -> 统一行枚举及数值复现边界 -> P3 选择可靠性、成本与受控拓扑诊断 -> P1/P2 单项候选回访 -> P4 有界历史管理及 P5 独立稳健性验证。

P1 的自适应预算与复杂度升级，要依赖 P3 的收益判断能力再重启；P2 暂不重复扩大文本候选搜索。P5 中的小图稳健性检查应贯穿各阶段，不必等所有模块结束才做。

## 2. 版本为什么会乱

### 2.1 四种编号混在了一起

| 名称 | 实际含义 | 今后如何引用 |
|---|---|---|
| frozen v1 / v1.2、早期文本 v2/v3 | 旧方法或旧实验分支 | 只作历史谱系，附日期和协议，不与当前分数拼表 |
| unified v3 | 统一 runner/student/预算等协议基础 | 写“unified v3 基础 + 2026-09-16 audited 四路划分及指标口径” |
| SCA v3、method v4 | 方法机制的演进 | 当前比较对象明确为“冻结 v4”，不是官方 GraphCLIP 的版本 |
| complexity-v2、response-v2 | 单个组件的修订方案 | 标明组件名及实验状态，不能据此宣布整套方法升级 |
| P0-P5 | 工作阶段 | 采用本文下表；不是代码或算法版本号 |
| `code/v2_llm_fusion/` | 历史形成的代码目录 | 目录名不是当前方法版本，不为统一名称移动代码 |

`--v4-mainline` 是 preset，不是不可覆盖的完整实验定义。实际身份还取决于预算、文本、划分、拓扑、seed、代码与 artifact hash；不能只凭开关或文件夹名字认定两次实验同协议。

### 2.2 工作阶段也曾重新编号

| 文档时期 | 当时的编号 | 与本轮 v4 的对应关系 |
|---|---|---|
| 2026-09-14 v3/SCA 工作计划 | P2 历史管理、P3 文本、P4 预算、P5 SCA、P6 外部比较 | 分别对应当前 P4、P2、P1、P3、P5 |
| 2026-09-15 v4 两份总计划 | P1 预算、P2 文本、P3 SCA、P4 历史管理、P5 多阶段与对比 | 作为今后固定编号 |
| 2026-09-16 P2 收尾、论文中的“P3 结构控制” | 将下一步重点写成拓扑实验 | 归入 P3-B 的选择前诊断，同时是 P4 的前置证据；不是把原 P3 全部改名 |

`FUTURE_METHOD_OPTIMIZATION_ROADMAP` 还混有两段不同时期内容：前半已谈 P2 结论，后半仍写 P2 未开始。其旧待办不是当前执行状态。旧报告中的“test 首次读取”“完全 label-free”“只追加必然最好”等强表述也不能沿用为当前事实。

## 3. 六个方向与 P0-P5 对应

| 原优化方向 | 解决的问题 | 对应阶段 | 当前决定 |
|---|---|---|---|
| 方向一：增长/信息自适应预算 | 新数据来了，总共应增加多少合成节点 | P1；收益估计依赖 P3 | fixed-rate 为默认；adaptive-rate、response extra 暂不晋升 |
| 方向二：规模与复杂度分配 | 给定新增预算，该分给哪些类、哪些类内模式 | P1；边际收益依赖 P3 | dispersion 先验 + 保守 class-row transfer；复杂子簇机制仍待验证 |
| 方向三：固定预算文本压缩 | 每个合成节点如何保留有效文本语义 | P2 | 严格实验已收尾，保留 center concat |
| 方向四：历史 synthetic memory 管理 | 旧表示是否更新、何时合并/拆分、如何连接新旧块 | P4；拓扑诊断前置到 P3-B | 默认冻结旧行/旧旧边；refresh/split/merge 不作当前贡献 |
| 方向五：SCA 选择可靠性 | 哪些候选确实能让重新训练的 student 更好 | P3；部分提前进入 P1/v4 | 已有多 horizon/独立 seeds/回退；跨配置可靠性尚不足 |
| 方向六：多阶段持续协议 | 优势是否依赖一个类别顺序和三阶段设定 | P0 提供能力，P5 验证 | 已完成两小图、三个三阶段顺序；暴露顺序敏感，尚缺长链/rate/独立确认 |

官方模型参照、外部 TAG 凝聚及完整数据参考属于 P5 的评价支线，不应作为第七个算法模块。

## 4. 当前冻结 v4 究竟包含什么

1. 在合法累计 pool 上取 `K_t = floor(0.05 * N_t)`，默认随数据增长；新增 rows 为 `K_t - |M_(t-1)|`，当前默认不裁剪旧行。
2. 当前 delta 按类分块，以规模平方根与离散度确定类配额，再在 graph-text joint space 内聚类。构造允许读取 pool 标签；不是整条流程无标签。
3. 输出簇均值合成特征，不将选出的真实节点直接作为主方法凝聚节点。
4. 新合成节点使用 `joint_g5_c320`：至多 5 篇代表文本，在总计约 320 字符额度下拼接，再用冻结 MiniLM 编码。它不是 P2 的 64-token control。
5. 保留旧合成特征、文本表示与旧旧边；增加新块内部边及受控跨边，当前 cross-degree 为 1。
6. 在固定预算内评估 class-row transfer，使用独立 selection seeds、多 batch、多 horizon、风险代理下界和 worst-horizon 检查；不可靠时回退到 dispersion。reserve extra 默认关闭。
7. 每阶段重新训练同一两层 GCN384 图文 student：MiniLM384、双向 InfoNCE、batch16、seeds100..105，步数为 `max(50,min(2048,16*K_t))`。

这里的“安全回退”是实现行为，不是统计安全保证。horizon/batch 分数并非相互独立样本，代码中的 LCB 不能直接解释为带严格覆盖保证的置信下界。当前 Citeseer 的约 +1.59 点还不足以证明 selector 普适有效。

### 原始顺序归档参照

下表保留此前归档值，不是新三顺序均值。Cora raw_long 的本轮同顺序重跑为 50.50，而非旧归档 51.31；GPU 数值重复诊断及验收修订见新报告。不得把旧分母与新均值拼算保留率。

| 指标 | Cora | Citeseer |
|---|---:|---:|
| 最终合成节点数 / 合法 pool | 94 / 1895 | 111 / 2229 |
| v4 最终 Macro-F1 | 49.85 | 60.77 |
| 同 student 完整 pool 长训所选检查点 F1 | 51.31 | 55.60 |
| 最终性能保留率 | 97.16% | 109.28% |
| 阶段平均性能保留率 | 88.71% | 106.31% |
| v4 最终旧类 / 新类 F1 | 65.44 / 29.06 | 68.00 / 46.29 |
| 完整数据参考最终旧类 / 新类 F1 | 60.85 / 38.59 | 63.90 / 39.01 |

保留率是两种方案 F1 均值之比，不是准确率或信息保留量；分母为 8192 步内由 selector-dev 选检查点的经验参考，不是理想最优模型。阶段平均沿用全部类别 prompt 竞争，另有 seen-only 诊断。

### 新三顺序结果对优化的影响

| 指标，三顺序均值 | Cora | Citeseer |
|---|---:|---:|
| v4 / dispersion 最终 F1 | 47.47 / 47.89 | 55.48 / 54.31 |
| k-center / kmeans-medoid 最终 F1 | 49.40 / 54.24 | 57.73 / 51.08 |
| v4 / 完整数据长训最终 F1 | 47.47 / 52.20 | 55.48 / 55.41 |
| v4 最终 / 阶段平均保留率 | 90.93% / 92.95% | 100.13% / 108.20% |
| v4 一次构造加训练 / 完整数据长训耗时 | 264.55 / 55.30 s | 239.26 / 56.47 s |

**优先问题扩大为顺序稳健性、保持-塑性和 selector 性价比。** Cora 1702 的最终保留率仅 74.86%，旧类也出现明显缺口，不能只说新类塑性不足。response 占构造时间约 99%；18 个阶段仅 5 个阶段最终配额改变。普通 dispersion 与 v4 的类块排列不同，新增顺序下两者差值不是严格 selector 消融，须先统一枚举。新结果不支持继续沿用“v4 稳定强于 k-center”的结论。

## 5. 各阶段真实完成度

| 阶段 | 已完成/已有证据 | 未完成或未通过 | 当前状态 |
|---|---|---|---|
| P0 协议与审计 | unified runner、四路划分、synthetic-only、预算/成本/hash；新增 canonical class/source enumeration、stage row hash、seed policy 和顺序不变性审计 | 不能消除项目早期的 test 接触；尚不覆盖全部未来协议 | **完成** |
| P1 预算与 allocation | 两小图 5% matched-quota；多种 allocator；fixed-total/hybrid/adaptive 首轮；子簇/response/extra 首轮；v4 三顺序确认 | 1%/10%、多 selection seed group、更多独立顺序/阶段未齐；complexity-v2 完整目标未验收；reserve 仅有实现/smoke；三顺序未证明稳定优势 | 主线已形成，须先解决比较控制与收益判断 |
| P2 文本 | 严格 token 控制、抽取/dual/pooling/DeepSets、独立 seeds、低 rate、排列与指标审计；2052 次归档 fit；P2-A 另完成 864 fits、144 memory 审计 | 图条件选片未优于 shuffled；Citeseer 阶段表现下降；未触发独立确认 | **P2-A 第一版完成；新机制不晋升** |
| P3 SCA | 多 horizon、独立 selection seeds、LCB/worst-horizon 与 fallback 已进 v4；P0 后完成两小图三顺序 response off/on 严格配对、成本账和 tensor hash 审计；P3-B 完成四臂拓扑诊断 | 跨顺序没有稳定最终收益；Cora 平均退化；selector 约为低成本构造 400--600 倍；当前/随机/质量跨边规则均未形成稳定晋升方案 | **P3-A 完成但未通过晋升；P3-B 完成机制诊断；默认回到 dispersion + current topology** |
| P4 历史管理 | 有更早 refresh/recluster 代码或诊断；当前 frozen-old 控制已存在 | 当前 v4 下 bounded refresh、预算化 split/merge、importance eviction 与跨边更新未系统验收 | 没有可晋升的完整 manager |
| P5 多阶段/外部比较 | 两小图完整 pool、微调、coreset、三顺序与实测成本；静态 TAGSAM/STAD 单顺序适配及监督诊断；官方 checkpoint 纯推理 | 外部方法未做三顺序；更多构造 seeds/独立顺序、严格三方同构造比较、大图正式主表仍缺 | 本批小图确认完成，不是整阶段完成 |

不能用旧协议的大图结果补齐当前 v4 的验收；也不能把 P2 的 2.5% 文本实验当作 P1 的完整预算扫描。代码有参数开关、smoke 通过、正式性能通过是三种不同状态。

## 6. 后续如何推进

### 第一批：保持 v4 冻结，扩展小图与成本账（P5-A，已完成）

遵守此前“先与其它方法比较，最好版本消融后做”的决策，不为了新路线立刻修改 v4。

- 已完成 canonical、1701、1702 三个预固定顺序的 13 臂比较，记录最终/阶段平均保留率、old/new、forgetting、预算与逐 seed 结果。
- 已补一次构造、selector、一次 student 轨迹、输入逻辑字节和 allocated 显存；构造不除以六个报告 seeds，明确单次与复用情景。
- 已完成独立重建、checkpoint 复载、数据/代码指纹及原始顺序回归。Cora raw GPU 端点非完全可复现已公开诊断，只调整验证规则，没有改算法或挑选好分。
- TAGSAM/STAD 继续只引用既有单顺序适配及其边界，不把它们写成三顺序已跑；不再补强或继续检查。DANCE 缺口也不阻塞主线。
- 官方 GraphCLIP checkpoint 的 58.98/57.76 仍是不同资源的纯推理旁证，不是保留率分母，也不重新训练。

交付：[完整比较报告](SMALL_ORDER_COST_COMPARISON_REPORT_ZH_2026-09-17.md)。新发现的负面证据同样保留；结果不构成直接推进大图或宣布主方法稳定占优的验收。

### 第二批：先让 SCA 的收益判断可信（P3-A）

核心问题不是“再增加候选”，而是“在未用于选择的运行条件下，选中的候选仍然值得选吗”。

前置控制：先让 dispersion 与显式 class-row 构造采用一致的类块/节点枚举，保证相同配额时产物相同；对所有对照共同应用，另存新协议结果，不覆盖本轮。继续隔离 GPU 数值变化；不能因 raw 路径的重复误差而放宽合成产物检查。Cora 1702 与 Citeseer 新顺序已成为已知失败条件，优化后的确认需另留预注册顺序。

1. 锁定一个小候选集：dispersion 原配额、有限固定预算 transfer，以及预注册的结构候选。先分开研究 allocation 与 topology，不直接跑二者笛卡尔积。
2. 对同一候选收集短、中、canonical 终点 response。horizon 按 stage 的 canonical steps 比例定义，避免后期仍以过短训练选优。
3. 使用按 seed 配对的增益和 old/new 分项；batch 与 horizon 当压力条件，不当作扩充独立样本量。独立 selection seed group 与报告 student seeds 均完整保留。
4. 检查短训排序与 canonical 排序的一致性、选中后相对 dispersion 的负增益频率、最差阶段退化、fallback 率和 selector 训练成本。
5. 只对少数入围候选作 canonical-step confirmation，再决定是否接受。确定阈值后固定，不看 report-dev 反复调阈值；独立初始化也不能消除反复使用同一数据划分的开发偏差。

失败分流：短训排序无信息时缩小搜索或直接采用固定构造；不要因 selector 复杂就坚持启用。若稳定选择依赖昂贵全训，应报告成本，比较它是否优于直接多训练共同 student。

潜在贡献是“可重训收益的可靠选择及其失效边界”，不是仅给现有短训分数加一个新权重。

#### P3-A 实际验收结果（2026-09-18）

新的 P0 协议下已完成 Cora/Citeseer 各三个类别顺序的 `dispersion` 与 `v4 response-on`
严格配对。最终 F1 的 `v4 - dispersion` 平均变化为：Cora `-0.93` 个百分点，
Citeseer `+0.85` 个百分点；Cora 最差顺序下降 `-2.80`，Citeseer 最差顺序下降
`-2.29`。response selector 平均耗时分别为约 `259 s` 和 `230 s`，而 dispersion
构造不足 `1 s`，成本比约 `565x` 和 `416x`。因此 response-on 未通过晋升门禁，
不再作为默认主方法或 response-gain extra budget 的依据。详细结果见
[P3-A 验收报告](P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md)。

当前执行含义是：保留 response 作为有限诊断工具，默认控制改为 `dispersion + response off`；
下一优先级已完成 P1-X 小图验收。`Weighted-X` 与第一版 `Moment-X` 均未通过跨顺序晋升门禁，
默认继续使用 `Mean-X + dispersion`，后续不再对这两个手工算子做无界参数扫描。详细结果见
[P1-X 合成节点特征报告](P1X_SYNTHETIC_X_COMPLETION_ZH_2026-09-18.md)。

### 第三批：把旧边冻结与旧传播保持区分开（P3-B，已完成）

固定 old/new synthetic X、文本、rows 与 row order，只改变拓扑。先做单阶段、同一旧 memory 起点的干预，再对通过的方案运行完整递归链，避免把历史路径差异误归因于本次边变化。

建议最小组：当前跨边规则、保留旧边但不加跨边、相同跨边数的随机/打乱端点控制、相同跨边数的质量筛选。全 self-loop 可作大幅删边诊断，但它改变旧旧边与边预算，不属于严格 frozen-old 公平主比较。

关键依据是归一化传播：

\[
\widehat A=D^{-1/2}(A+I)D^{-1/2}.
\]

即使 `A_old,old` 不变，新增跨边仍可能改变旧节点度与归一化系数。因此应测 old-old 归一化块漂移、跨边消息量、旧表征漂移和 old/new F1；不能仅检查旧边 hash 就宣称不会遗忘。

只有诊断支持该机制后，再试分块归一化、单向跨块信息流或有界跨边权重。它们是候选而非已证实改进。若改变 GCN 传播定义，则应单列为 operator adaptation，并给基线/完整数据相同能力，不能混入“只改凝聚图、student 不变”的主表。

交付已完成：[P3-B 拓扑诊断完成报告](P3B_TOPOLOGY_DIAGNOSTIC_COMPLETION_ZH_2026-09-18.md)。结果支持“跨边有用”，但不支持 random/quality 稳定替换 current。按用户最新决定，有界跨边和单向传播后置，先完成 P2-A；后续拓扑对照设计见 [P2-A 报告第 9.1 节](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md#91-后置的拓扑实验有界跨边与单向跨块传播)，未执行、未并入 v4。

### 第四批：有界管理历史合成记忆（P4）

先做小动作，再考虑全套 manager。输入必须只有旧 synthetic memory、其合法保存的统计和当前 delta，不回读历史真实文本/节点。

| 动作 | 最小实现 | 必须控制的风险 |
|---|---|---|
| re-score only | 仅更新覆盖/质量统计，不改 X/text/edges | 作为统计更新无性能影响的控制 |
| bounded refresh | 在可匹配到新证据时，对少量旧合成节点做有界更新 | 当前是类别增量，新类不等于旧类的新证据，不能强行刷新所有旧类 |
| merge + reallocate | 合并冗余旧合成节点，将空出的 rows 交给新模式 | 与不合并但相同新 rows 的预算控制分开，测历史覆盖损失 |
| synthetic split | 基于保留的合成子模式或充分统计拆分 | 一个历史均值无法还原已丢失的多峰分布；辅助统计必须计入 memory |
| importance eviction | 只在明确的预算压力对照中淘汰 | 默认 growing fixed-rate 不因任务编号变化就裁剪旧 memory |

所有动作保留 fallback，记录行级来源、birth stage、改写比例与边变化。先比较 keep 与单个动作；只有单动作可靠，再组合 manager。不能同时刷新特征、重写文本和重建全图后将收益归到一个模块。

潜在贡献是“可审计的合成历史更新与保持-塑性权衡”。当前没有证据支持把它写成已完成创新。

### 第五批：P1 进阶回访与 P5 正式稳健性

**预算回访依赖 P3，不独立盲目扩容。** 先验证给定总 rows 的边际调配，再问是否需要更多 rows。subcluster complexity 只作候选生成先验，用固定预算与 dispersion 对照；不能假定多簇就一定更好。

response-gain extra 必须与“在相同阶段额外加相同 rows 的固定规则”对照，同时记录训练步数。canonical 步数随 K 变化，所以扩容还可能增加 student 更新次数；需要等步数诊断才能把 rows 收益与训练量分开。信息量决定压缩率仍是合理研究问题，但不是当前已验证机制。

正式稳健性分两步：

1. 小图已有 5% 三顺序结果，继续补 matched-quota rate 及额外独立顺序。原计划的 1%/5%/10% 可作节点预算曲线，但极低 rate 下阶段 rows 可能少于 batch16；需报告真实 batch/负例数，并另作固定有效 batch 控制，不能重犯 batch 混淆。新增顺序预注册，通过后扩至 5 个顺序；比较必须成对使用相同顺序。
2. 小图验收报告后再决定大图。大图采用 base+4 主协议，3-stage 快速消融，类别数足够时再做 8/10-stage 长链。Cora/Citeseer 只有 7/6 类，不能硬拆成 8/10 个非空 class-disjoint 阶段；若改为同类分批到达，必须另立 data-incremental 协议。

每阶段报告最终/累计 F1、阶段平均、old/new、按固定历史任务定义的 forgetting、rows、实际 rate、edges、文本/统计字节、selector/student steps 和耗时。不同阶段竞争类别变化引起的 F1 下降不能一律叫遗忘。

只有发现明确文本失配机制时再回访 P2，例如图结构相关的文本片段选择。仍需固定 token/rows/topology，且与当前字符版 v4 分开，不因一个终点好分重开无界搜索。

### P2-A 图条件文本结果（2026-09-18）

已完成 Cora/Citeseer、三个预注册顺序、八个文本组和六个 student seeds 的小图矩阵，共 864 fits。所有组共享 source clusters、Mean-X、rows、row order 和拓扑；文本预算违规为 0，两个数据集的独立 artifact audit 均通过。

`graph_selection` 相对 `center_concat` 在 Cora 最终 +3.05、Citeseer -1.85；相对 `shuffled_graph` 在 Cora -0.81、Citeseer -0.82。说明当前“属性 query + 当前 delta 邻域文本均值”的图条件接口没有稳定的真实图优势，不能晋升为主方法。Cora 的 `text_attention` 较好但 Citeseer 低于 center concat，也不支持直接替换默认文本处理。

当前决定：继续使用 `Mean-X + dispersion + joint_g5_c320 center concat + current topology`；P2-A 保留为负结果和接口诊断，不启动独立确认，不命名 v5。P3-B 有界跨边和单向跨块传播继续后置。详见 [P2-A 完成报告](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md) 和 [P2-A 协议](P2A_GRAPH_CONDITIONED_TEXT_PROTOCOL_ZH_2026-09-18.md)。

## 7. 统一验收规则与命名

- 正确性门禁：训练源与选择/报告隔离、预算和历史访问可审计、synthetic-only、冻结版本回归、所有预注册运行完整；任何一项失败先修实现，不读性能结论。
- 晋升门禁：固定主指标及容许 old/new 退化阈值，在独立初始化组和预固定额外类别顺序上确认；报告每个数据集，不用平均收益掩盖一个数据集的明显退化。阈值应在下一轮运行前写进 manifest，不在本次整理中伪装成已有规则。
- 主问题仍是凝聚效果：共同 student、相同节点预算下的性能，以及相对完整数据参考的保留率；监督任务及官方 checkpoint 独立成表。
- 一次只晋升一个通过门禁的改变；其它候选保留负结果。筛过多个候选应承认多重比较，报告初始化区间不能当跨数据集的普适证明。
- 当前名字继续用“冻结 v4”。后续结果用 `v4_p3_selector_<date>`、`v4_p3_topology_<date>`、`v4_p4_refresh_<date>` 等说明实验，不直接命名 v5。只有配置冻结、回归和独立确认通过后再决定正式方法版本。
- 固定保存 method/组件配置、protocol/指标版本、数据与划分 hash、构造与选择/报告 seeds、代码与 artifact hash。正文版本名不能替代这些字段。

若后续均不晋升，冻结 v4 仍作为简单可复现基线，论文保留可靠构造、可重训性诊断和负结果；若 P3/P4 成功，再据实形成“可靠选择 + 受控历史交互”的方法贡献，而不是把六个方向都包装成已完成模块。

## 8. 证据入口

- [原六方向及 P0-P5 总路线](FUTURE_METHOD_OPTIMIZATION_ROADMAP_ZH_2026-09-15.md)：保留历史计划，当前状态以本文和后续验收为准。
- [v4 目标设计](V4_TARGET_METHOD_AND_OPTIMIZATION_PLAN_ZH_2026-09-15.md)：目标机制不等于当前默认配置。
- [P1 受控小图报告](P1_SMALL_GRAPH_RESULTS_ZH_2026-09-15.md)、[v4 小图验收](V4_MAINLINE_SMALL_RESULTS_ZH_2026-09-15.md)。
- [P2 严格实验与审计](P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md)、[P2 收尾协议](P2_CLOSEOUT_SMALL_COMPARISON_PROTOCOL_ZH_2026-09-16.md)。
- [最新外部 TAG 比较](EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)、[官方 checkpoint 参照](OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md)。
- [当前小图比较代码](../../code/v2_llm_fusion/run_small_comparison_audited.py)、[v4 preset 与候选逻辑](../../code/v2_llm_fusion/run_unified_v3_continual.py)。
- [冻结 v4 Cora 结果](../../results/v4_mainline_safe_2026_09_15/cora_results.json)、[冻结 v4 Citeseer 结果](../../results/v4_mainline_safe_2026_09_15/citeseer_results.json)、[最新统一比较数据](../../results/external_tag_comparison_2026_09_17/summary.json)。
- [P2-A 图条件文本完成报告](P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md)、[P2-A 结果汇总](../../results/p2a_graph_conditioned_text_20260918/summary/summary.json)。

本页原为状态梳理，2026-09-18 已补入 P0/P3-A/P1-X/P3-B/P2-A 的实际验收。P2-A 新增 864 fits，其完整证据与成本见完成报告；缺少当前协议验收的其它事项不按旧目录或旧分数自动判为完成。
