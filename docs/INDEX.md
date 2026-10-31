# 2026-10-06 Adaptive slot coherent routing 已完成：[自适应 slot 与边界感知文本预算结果](current/X_COHERENT_ADAPTIVE_SLOTS_RESULTS_ZH_2026-10-06.md)。Arxiv/SportsFit 的 base+4、6 个 Student seeds 已完成；adaptive coherent 相对同批次 CCTP 的 Final F1 分别提升 +6.11/+21.63 pp，anchor 分别提升 +2.52/+10.27 pp。slot 数量由当前 delta 图文离散度在 1--3 之间自适应决定，320 字符预算保留但改为句子/词边界优先截取。

# 2026-10-06 X-conditioned coherent payload routing 的 CCTP anchor 小图确认已完成：[小图三顺序确认报告](current/X_COHERENT_PAYLOAD_ROUTING_ANCHOR_SMALL_ZH_2026-10-06.md)。Cora/Citeseer、order19611--19613、6 个 Student seeds 全部完成；anchor 相对 `cat_nearest` 的 pooled Final F1 仅 +0.029 pp，615 个当前 delta 行只接受 4 个替换，说明当前严格代价过度保守，anchor 暂定位为安全回退组件而非最终主方法。

# 2026-10-06 X-conditioned coherent payload routing 的 CCTP anchor 大图确认已完成：[大图正式确认报告](current/X_COHERENT_PAYLOAD_ROUTING_ANCHOR_LARGE_ZH_2026-10-06.md)。Arxiv、BookChild、SportsFit 的 base+4、5% memory、6 个 Student seeds 全部完成；无约束 coherent 在 BookChild 上退化到 2.963 F1，而 anchor 恢复到 13.371。三图平均 Final F1 为 21.751，相对 CCTP top-5 的 19.727 提升 +2.024 pp，New F1 提升 +2.231 pp；94.16% 候选回退到 CCTP，说明安全约束有效但当前候选代价仍较保守，下一步做小图三顺序和独立 construction seed 确认。

# 2026-10-05 XSP 四臂归因实验已完成：[XSP 文本分配与 X 更新归因报告](current/XSP_ATTRIBUTION_ABLATION_ZH_2026-10-05.md)。Cora/Citeseer、order19611--19613、四个共同 CaT 锚点版本和六个 Student seeds 全部完成；Fixed-X XSP 主要带来新类方向收益但总体持平，XSP-X+CaT-text 的最终 pooled F1 为 +0.341 pp 但仅 2/6 cell 为正，Full XSP 为 -0.711 pp 且 Old F1 为 -1.715 pp。结论是当前 XSP 联合目标不能晋升，下一步优先修正 X 更新与旧类保持，而不是继续堆叠文本或 selector 模块。

# 2026-10-02 CaT Student-response bridge 已完成：[CaT Student-Compatible Response Bridge 实验报告](current/CAT_STUDENT_RESPONSE_BRIDGE_2026-10-02.md)。support-X response 在 Cora/Citeseer 上仅基本不劣；加入 MMD 后 Cora +0.301 pp、Citeseer -0.139 pp；新类 margin 单独优化为 Cora -0.207 pp、Citeseer -0.127 pp。梯度路径真实存在，但单一 response 目标与最终长程 Student 不一致；下一步只做受限 support + small-margin + old-retention 混合目标，不再盲目扫参数。

# 2026-10-01 CaT-TAG CCTP 四顺序公开基线已完成：[CaT-TAG Class-Central Text Prototype 基线矩阵](current/CAT_TAG_TEXT_CONSTRUCTION_STAGE1_ZH_2026-10-01.md)。Cora/Citeseer 四个类别顺序、六个 paired Student seeds 和统一 5% current-delta-only 协议全部通过审计；CCTP 相对 CaT-nearest 的 Final F1 在 8/8 cell 为正，Cora/Citeseer 平均分别 +11.11/+12.47 pp，并高于本轮 GECC、k-center、medoid 适配参照。CaT/GECC 仍标记为机制适配，不宣称官方端到端 SOTA；下一步转向 top-k/soft prototype 消融和大图 base+4 验证。
# 2026-10-01 CCTP top-k/soft prototype 控制已完成：[CaT-TAG 文本构造第一阶段实验报告](current/CAT_TAG_TEXT_CONSTRUCTION_STAGE1_ZH_2026-10-01.md)。固定同一 CaT X 的四顺序消融显示 top-1 明显不足，top-3/top-5 接近，soft prototype 有较好阶段平均但 Final F1 略低；八个 cell 审计通过。大图阶段预注册保留 top-3 与 top-5，不在小图上事后选唯一版本。

# 2026-10-01 Bounded X+P 后续优化已完成：[Operator auxiliary 与 class-constrained P 实验报告](current/BOUNDED_XP_OPERATOR_AUX_OPTIMIZATION_ZH_2026-10-01.md)。修正了 class-level duplicate positive 的 outer loss；hard class-constrained P 不晋升，保留原 support-response 并加入固定 0.25 class-level operator auxiliary 的版本在 6 个可比 cell 中 5 个为正，平均 +0.46 pp，并完成 order19614 的 construction-seed 稳定性验证；相对 CaT/k-center 仍不占优，暂定为候选主方法而非最终 SOTA。

# 2026-09-29 Direct outer topology/joint 实验已完成：[受限拓扑与 X+拓扑联合实验报告](current/DIRECT_TOPOLOGY_JOINT_RESPONSE_RESULTS_ZH_2026-09-29.md)。四个小图 cell 完成 topology-only 与 X+topology 的 v1/v2 评估；cross-edge 守恒实现正确，但短程 Student response gate 不能稳定预测长期新类塑性，联合版本不晋升。
# 2026-09-28 直接 Student 外层梯度实验已完成：[修正协议与 Prompt 目标实验报告](current/DIRECT_STUDENT_OUTER_CORRECTED_PROMPT_RESULTS_ZH_2026-09-28.md)。修正了两个 `order19102` cell 的 `old_rows` 边界后，X-only、prompt CE 和 prompt margin 均未稳定改善新类 F1；确认梯度路径真实存在，但 outer proxy 与最终 Student prompt 分类仍错位，暂不推进 P/A 联合优化。
# 2026-09-28 Class-aware X-v2 已完成：[类感知外层梯度路由实验报告](current/DIRECT_STUDENT_OUTER_CLASS_AWARE_X_RESULTS_ZH_2026-09-28.md)。四个小图 cell 均完成 synthetic-slot mass 守恒审计和六个 Student seed 配对评估；相对普通 X-only 平均改善 0.604 pp，但相对 C0 仅 2/4 cell 为正，canonical 顺序下降，暂不晋升主方法。
# 2026-09-29 Provenance-aware X 已完成：[类内 provenance-aware X 外层优化实验报告](current/DIRECT_STUDENT_OUTER_PROVENANCE_X_RESULTS_ZH_2026-09-29.md)。在相同协议下完成 `r=0.05/0.10` 两组四-cell 评估：较小半径偏旧类保持，较大半径偏新类塑性，但尚未稳定超过 C0；下一步转向有界旧响应约束。
# 2026-09-29 Provenance X 后续受控实验已完成：传播算子一致、旧响应 barrier、construction-time 类 prompt CE 和冻结文本 inner 均已验证；没有形成稳定跨顺序增益，后续转向多 checkpoint Student trajectory response 估计，不再继续堆叠 X-only loss。
# 2026-09-29 Student trajectory response 诊断已完成：[多 checkpoint response 可靠性报告](current/STUDENT_TRAJECTORY_RESPONSE_DIAGNOSTIC_ZH_2026-09-29.md)。四个 cell、24 个配对 Student seed、0/2/4/8/16/32/64 steps 已完成；64-step pooled Spearman 仅 +0.237、符号一致率 54.2%，不能作为 X 更新或 candidate selector，下一步先校准无标签目标，不再盲目增加 response 损失。

# 2026-09-27 Teacher--P--X 完整复现已完成：[Teacher-Guided Payload Revision 报告](current/TEACHER_GUIDED_PAYLOAD_REVISION_ZH_2026-09-27.md)。修正版四个小图 cell 已完成构造、6-seed Student 评估和访问/位移审计；平均 F1 相对 Semantic-Recursive 为 -0.946 pp，未晋升，主要问题定位为 P 未进入结构传播以及 Teacher/Candidate 拓扑与训练 horizon 失配。

# 2026-09-27 外部持续凝聚适配已完成：[360-fit 小图结项报告](current/EXTERNAL_CONTINUAL_ADAPTER_COMPLETION_ZH_2026-09-27.md)。SR 冻结且逐 seed 复现，Final F1/Accuracy 为 53.02/59.59；CaT-MMD-TAG 54.62/59.96、GECC-hard-TAG 48.70/52.53、k-center 独立构图 54.18/59.43、medoid 47.32/53.11。SR 对 GECC/medoid 四条件均较高，对 CaT/k-center 各 1/4 较高。360 个重载、60 个 memory hash 验证通过。此表允许外部方法自行构造 stage0/拓扑，不与旧共享 C0 控制表混称；原生完整复现、PUMA、新独立顺序仍未完成。

# 2026-09-27 并行基线交接：[统一外部基线 Setting 与同学交接协议](current/EXTERNAL_BASELINE_HANDOFF_PROTOCOL_ZH_2026-09-27.md)。冻结数据、预算、访问权限、原始 Student、指标和审计规则；CaT/GECC/k-center/medoid 不重复旧矩阵，优先补 PUMA 与静态 TAG 方法可比性审查。

# 2026-09-26 当前主方法完整评估已整理：[Semantic-Recursive 完整方法、实验与论文评估](current/TGC_SEMANTIC_RECURSIVE_COMPLETE_ASSESSMENT_ZH_2026-09-26.md)。semantic-recursive 固定为论文主方法候选，C0 保留为冻结共同控制；大图五层基线已完成，协议审计通过。当前大图相对 C0 平均 +0.858 pp、new-class +1.764 pp、9/12 增量 cell 为正，但仍有数据集/阶段依赖，暂不宣称 SOTA。

# 2026-09-26 两小图统一递归基线比较已完成：[Semantic-Recursive 小图统一基线比较报告](current/TGC_SEMANTIC_RECURSIVE_UNIFIED_BASELINE_COMPARISON_ZH_2026-09-26.md)。Cora/Citeseer、order19611/19612、6 个配对 Student seeds、9 个方法臂共 108 条 stage 记录完成；Semantic-Recursive 相对 C0 平均 +0.925 pp F1、+0.364 pp Accuracy、3/4 cell 为正。报告区分 memory-only、full-raw offline reference、stateful Student fine-tune 和静态 TAGSAM/STAD 协议，并明确 fixed-rate order confound。

# 2026-09-26 P0 独立顺序与 construction seed 验证已完成：[P0 完整报告](current/TGC_P0_INDEPENDENT_ORDER_CONFIRMATION_ZH_2026-09-26.md)。arXiv、BookChild、SportsFit 三个大图的 construction seeds 9001/9002/9003 均 bitwise 一致，历史 raw reads=0；独立顺序下 semantic-recursive 增量平均约 +0.020 pp、5/12 正，arXiv 为 -0.373 pp，说明构造确定性通过但跨顺序泛化未通过，不能宣称稳定提升。

# 2026-09-26 role-aware 独立确认已完成：[独立顺序、old/new F1 与成本报告](current/TGC_ROLE_AWARE_INDEPENDENT_CONFIRMATION_ZH_2026-09-26.md)。`role_perm` 独立集平均 +1.468 pp、7/12 正、最差 -2.042 pp，new-class plasticity 仍不够稳定；C0 继续作为默认主线，暂不启动 role-aware 的大图扩展。

# 2026-09-26 recursive semantic 大图确认已完成：[arXiv、BookChild、SportsFit 的 base+4 结果](current/TGC_SEMANTIC_RECURSIVE_LARGE_CONFIRMATION_ZH_2026-09-26.md)。三个大图共 12 个增量 cell，9/12 为正，cumulative 平均 +0.858 pp，new-class 平均 +1.764 pp，最差 -2.381 pp；结构审计全部通过。结果支持大图上的方向性机制信号，但仍有阶段/数据集依赖，暂不宣称 SOTA。

# 文档索引

# 2026-10-01 `order19614` 独立顺序验证已完成：[Bounded X+P 独立顺序基线比较](current/BOUNDED_XP_ORDER19614_BLIND_ORDER_COMPARISON_ZH_2026-10-01.md)。Cora/Citeseer 的 X+P、C0、k-center、medoid、CaT/GECC 适配均使用同一三阶段、5% memory、6 个 Student seeds；X+P 在该顺序上低于 C0 和多数强 coreset，结果作为未调参的负泛化证据保留，不并入原两顺序主表。
# 2026-10-01 四顺序 pooled 汇总已完成：[Bounded X+P 四顺序基线汇总](current/BOUNDED_XP_FOUR_ORDER_POOLED_SUMMARY_ZH_2026-10-01.md)。统一汇总 order19611–19614 的 Cora/Citeseer、X+P、CaT/GECC 适配、k-center/FPS 和 medoid；X+P 相对 GECC 在 8 个数据集-顺序 cell 中 5 个为正，但相对 CaT 全部为负、相对 k-center/FPS 仅 3/8 为正，结论为方向性收益而非普遍 SOTA。

# 2026-10-01 公开基线检索与适配矩阵：[公开图凝聚基线检索与统一协议适配矩阵](current/PUBLIC_BASELINE_SEARCH_AND_ADAPTATION_MATRIX_ZH_2026-10-01.md)。确认当前没有与 text-attributed、historical-raw-inaccessible、InfoNCE Student 和 base+4 协议完全重合的官方 baseline；冻结 CaT、PUMA、GECC、TAGSAM、STAD 及动态图方法的 adapted/oracle/reference 分层，规定后续同协议比较与审计要求。
# 2026-10-01 第一批大图公开基线对比已完成：[统一协议公开基线对比报告](current/PUBLIC_CONTINUAL_BASELINE_COMPARISON_LARGE_ZH_2026-10-01.md)。arXiv、BookChild、SportsFit 的 base+4、5% memory、6 Student seeds 和七个方法臂已完成；X+P/XPA 相对 append-edge 三图最终 F1 均为正，三图平均 +1.72 pp、Accuracy +3.22 pp，平均 Final F1 高于 k-center +0.77 pp；同时 k-center 阶段平均更强，CaT/GECC 在部分数据集胜出，因此暂不宣称普遍 SOTA。TAGSAM/PUMA/STAD 仍按 adapted/oracle/native 分层待补。

# 2026-09-27 形式化与持续基线复核：[Semantic-Recursive 受约束传输实现规范](current/SEMANTIC_RECURSIVE_CONSTRAINED_TRANSPORT_SPEC_ZH_2026-09-27.md)；[持续图凝聚相关工作与基线推进](current/CONTINUAL_GRAPH_BASELINE_REVIEW_ZH_2026-09-27.md)。前者严格描述当前 split/merge 后的 payload assignment，不改变冻结输出；后者核对 CaT、PUMA、GECC、OpenGC，并将外部 continual baseline 与内部 C0、静态 TAG 参照分层。

# 2026-09-27 Semantic-Recursive 方法审查与下一轮路线：[方法审查与 BASR 计划](current/SEMANTIC_RECURSIVE_METHOD_REVIEW_AND_NEXT_PLAN_ZH_2026-09-26.md)。结论是保留 SR-base 和 C0，先用预算守恒的目标驱动 split/merge 替代当前启发式规则，再做独立顺序和大图确认；不隐藏 C0，也暂不叠加 response、attention、refresh 或 GraphGPS。
# 2026-09-28 TGR 跨边传输全局 assignment 诊断已完成：[严格实验报告](current/TGR_CROSS_EDGE_TRANSPORT_COMPLETION_ZH_2026-09-28.md)。全局容量 assignment 在开发集相对 SR 平均 +0.402 pp、独立 order19613 平均 -0.951 pp，不能晋升；它确认容量最优不等于 Student 任务最优，后续转向受限 Student-consistent edge swap。

# 2026-09-27 Semantic-Recursive 论文主稿已更新：[Semantic-Recursive 持续 TAG 凝聚论文草稿](paper/TAG_CONTINUAL_CONDENSATION_SEMANTIC_RECURSIVE_DRAFT_ZH_2026-09-26.md)。该稿将 Semantic-Recursive 作为主方法，新增文本、X、response/预算、拓扑与 memory maintenance 的组件级负结果表，以及 TAGSAM、STAD、完整 raw、stateful Student 和官方 GraphCLIP 的访问权限分层对照；C0 仅作为共同控制与回退版本。

# 2026-09-26 大图五层基线已完成：[完整比较报告](current/TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md)。C0、semantic-recursive、coreset、full-history synthetic、complete raw equal-step 和 complete raw 8192-step 均完成五阶段；协议审计 `all_pass=True`。raw 方法属于离线参考，不是合法 continual constructor。

# 2026-09-25 当前总览：[当前科研进度与最优方法总览](current/CURRENT_PROGRESS_AND_BEST_METHOD_SUMMARY_ZH_2026-09-25.md)。C0 仍是可靠默认主方法；R15/T+degree 是条件性候选；role-aware conserved assignment 已完成独立确认但未通过晋升门槛。

# 2026-09-24 双原型文本上限确认已完成：[文本上限确认报告](current/TGC_TWO_PROTOTYPE_TEXTCAP_CONFIRMATION_ZH_2026-09-24.md)。Cora/order19601 在 6 个 Student seeds 下仍有条件性正信号，但 Citeseer 和另一条 Cora 顺序为负，三格平均不支持稳定提升；旧 `+2.328 pp` 不能作为主方法证据。

# 2026-09-24 topology-role 诊断与 role-aware matching 已完成：[Role-aware matching 六顺序报告](current/TGC_ROLE_AWARE_MATCHING_CONFIRMATION_ZH_2026-09-24.md)。固定 role cost 后六格平均 +1.581 pp、相对 semantic matched +1.417 pp，但最差 seed -7.713 pp，不能晋升；synthetic-margin gate 只能部分回退，仍有负尾部，详见 [Gate 验收报告](current/TGC_ROLE_MATCH_SYNTHETIC_MARGIN_GATE_ZH_2026-09-24.md)。C0 继续作为主线。

# 2026-09-24 TGC per-class prompt profile 独立确认已完成：[Profile-LCB独立报告](current/TGC_PROMPT_PROFILE_INDEPENDENT_COMPLETION_ZH_2026-09-24.md)。在独立 order19602 的 Cora/Citeseer、stage 1/2 中，profile-LCB 四格均正确回退 fallback，平均 regret=0，但没有捕获正收益候选；结合 V3 和 prompt-paired 结果，自动 selector 呈现“安全但无收益召回”，暂不启动递归 rollout，主线保持 C0/R11。

# 2026-09-24 TGC prompt-paired selector 已完成：[Prompt-Paired结项报告](current/TGC_SHARED_ASSIGNMENT_PROMPT_PAIRED_COMPLETION_ZH_2026-09-24.md)。class-prompt margin 加 paired candidate-vs-fallback response 使排序相关由 -0.25 略升至 -0.10，但平均 selection regret 由 1.540 pp 升至 2.182 pp，实际捕获收益降至 +0.242 pp；主要问题是 prompt 量纲下固定硬门限过严，造成误拒绝。该版本不进入主方法，下一步只做逐类别 margin、置信区间/Pareto 校准与独立顺序验证。

# 2026-09-24 TGC shared-assignment V3 selector 已完成：[V3结项报告](current/TGC_SHARED_ASSIGNMENT_GCN_RESPONSE_V3_COMPLETION_ZH_2026-09-24.md)。无标签 InfoNCE/margin response 使平均 selection regret 从 V2 的 2.425 pp 降至 1.540 pp，并在 Cora stage 1 选中最优 anchor，但跨数据集排序仍不稳定（平均相关 -0.25）。共享 assignment 候选空间存在条件性收益，当前瓶颈是 response 与 zero-shot class Macro-F1 的目标错位；C0/R11 继续作为稳定主线，下一步只改 class-prompt response 与 paired response，不叠加新构造模块。

# 2026-09-24 TGC-R16 拓扑安全实验已完成：[R16结项报告](current/TGC_R16_TOPOLOGY_SAFE_COMPLETION_ZH_2026-09-24.md)。R11 content + C0 topology 仍为当前最强研究候选，最终平均 +0.723 pp，但阶段平均 -0.314 pp、新类 -1.314 pp，不能晋升。保持新块度数的内容重连最终 -0.251 pp；cross guard 在 12 个增量阶段全部回退 C0，未形成独立增益。当前结论是 R11 内容有正信号，coarse topology 会损害新类传播，下一步应做 teacher-message-compatible topology gate，不再继续扫描 alpha 或简单重连。

# 2026-09-23 TGC-R11 C0-anchor 稳定化完成：[R11结项报告](current/TGC_R11_C0_ANCHOR_COMPLETION_ZH_2026-09-23.md)。C0-anchor assignment interpolation 已完成 Cora/Citeseer、三顺序、六 seeds、648 slots 和全量 audit。`anchor025` 将 joint 最差最终退化从 -5.043 pp 缩小到 -2.104 pp，但最终平均仅 +0.142 pp，阶段平均 -0.856 pp，新类 -2.059 pp，未通过晋升门禁。C0 继续作为主线；R11 定位为条件性稳定化组件，后续若继续只做新类塑性约束的单项实验。

# 2026-09-23 TGC-R10 联合 content-topology 独立确认已完成：[R10结项报告](current/TGC_R10_JOINT_CONFIRMATION_COMPLETION_ZH_2026-09-23.md)。Cora/Citeseer 三个保留顺序、四个核心臂、72 条核心记录和独立重载审计全部完成。joint 最终相对 C0 平均 -0.05 pp，但 4/6 个最终 cell 为正；最终交互项平均 +0.27 pp、stage 1/2 平均 +1.11 pp，确认联合机制存在条件性正协同，但新类平均 -1.92 pp、最差 -5.04 pp，未通过稳定晋升条件。下一步做 C0-anchored joint 稳定化，不把单组件负结果当作联合机制失败。

# 2026-09-23 TGC-R7 已完成稳定性诊断、目标校准、完整递归开发和独立确认：[R7结项与下一步决策](current/TGC_R7_COMPLETION_AND_NEXT_DECISION_ZH_2026-09-23.md)。共1584个槽位、1248次实际student训练，全部重载审计通过。稠密小图构造消除了R5构造重复中的稀疏数值分叉，但response proxy不能稳定预测完整student。新顺序确认中原始/稠密/有界TGC-R最终相对C0分别+1.320/+1.427/+1.055 pp；old F1提升约+2.85--3.02 pp，new F1下降约-0.82--2.06 pp，阶段平均有界版本为负。当前不宣布稳定主方法，下一步为C0-anchored new-class plasticity与可扩展确定性构造。

2026-09-23 TGC-R6两批实验已完成：[独立确认与机制诊断报告](current/TGC_R6_CONFIRMATION_AND_MECHANISM_RESULTS_ZH_2026-09-23.md)。1056槽位、1020次实际student训练及重载、84张修改后memory额外审计全部通过。原样R5 recursive在三个新顺序/新seed确认中，相对C0最终-2.65 pp、相对k-center -4.42 pp，未晋升。开发机制中，训练teacher较随机teacher平均+1.90 pp且4/4为正，文本×拓扑交互+2.48 pp且4/4为正；这些是方向性机制信号，不是独立确认优势。加权边降低teacher误差却使F1下降4.33 pp。固定memory的72对重训F1完全复现，但同配方重构的微小X变化可使单格均值改变4.10 pp。下一步优先构造稳定性及目标有效性校准；没有直接启动大图或晋升对称目标。冻结依据见[协议](current/TGC_R6_FROZEN_EXPERIMENTS_ZH_2026-09-23.md)及[评分前数值修订](current/TGC_R6_MECHANISM_NUMERICAL_AMENDMENT_ZH_2026-09-23.md)。

2026-09-23 TGC-R5 最小实验已完成：[独立确认、联合构造与递归结果](current/TGC_R5_RESULTS_AND_NEXT_DECISION_ZH_2026-09-23.md)。两轮共 792 个正式槽位、552 次完整 student 训练/不同 checkpoint 重载，全部审计通过。old-only 独立确认最终 -0.461 pp、低于等旧行位移 random 1.055 pp；共享 assignment 的 content/topology 最终均值单项为负，但 joint 交互 +2.801 pp，递归 joint 平均 +1.204 pp、最差 -3.112 pp。maintenance/capacity 相对递归分别 -0.997/-1.513 pp。优先研究递归联合构造的顺序稳定性，尚未晋升；旧边不可改写不是用户任务硬约束。详细机制、预算、成本与未完成边界见[执行协议](current/TGC_R5_EXECUTION_PROTOCOL_ZH_2026-09-23.md)。

2026-09-23 R4 结果解释复核：[门槛、失败原因与下一轮决策](current/TGC_R4_INTERPRETATION_AND_NEXT_DECISION_ZH_2026-09-23.md)。纠正了 relation_old“数值未过门槛”的表述：最终 +1.224 pp 等效果指标满足原数值条件，但未注册为主候选、缺少旧行匹配随机对照、阶段均值约持平，尚无晋升结论。shared-parent 不能解释递归累计，support 2x2 没有四条件正式下游对照。建议 C0 保持初始化/控制，优先 old-only 独立小型复验及 teacher/训练阶段诊断，再决定训练更新匹配；本次没有新增训练或改变封存分数。

2026-09-23 最新正式结果：[TGC-R4 关系响应凝聚结果](current/TGC_R4_RELATIONAL_RESULTS_ZH_2026-09-23.md)已完成修正版 `20260923c` 的 Cora/Citeseer、两个开发顺序、三阶段、11 臂、六 student seed 矩阵，共 792 个结果槽位、504 次实际训练；prepare/seal/evaluate/audit 全部通过。lifted support 将旧归一化传播块误差降至约 `6e-7`，但主候选 `relation_new` 最终相对 C0 平均 `-0.040 pp`、相对 matched-random `-0.341 pp`，未通过晋升门槛，不进入独立 confirmation、递归、大图或 LP。`relation_old` 描述性平均 `+1.224 pp`，但阶段和顺序不稳定，只保留为待独立验证假设。C0 继续作为当前主线参照。

2026-09-23 最新设计审查：[TGC-R4 证据审查与修正](current/TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md)。综合预算、X、文本、拓扑、维护、response 与 TGC 历史证据，核对 TAGSAM/STAD/DANCE/CaT 的适用边界；撤回旧 MB-LRR 直接执行建议。R3 当前候选在已观察四个最终 cell 上，即使事后二选一也只有 +0.373 pp 平均增益；建议先验证新块优先的关系响应凝聚，分别诊断 teacher 采样与传播上下文，再检验有限旧修订。本文是待验证设计，不是新实验完成或主方法晋升。

2026-09-22 最新 TGC 结果：[TGC-R3 独立 selector 确认](current/TGC_R3_SELECTOR_CONFIRMATION_RESULTS_ZH_2026-09-22.md)完成两小图、两个新顺序、新种子的 648 条记录与独立审计。固定 w64-k1 候选在 s2 相对 C0 平均 -0.64 pp，预注册 scratch response-curve 策略 -0.50 pp；teacher 对照最终 +0.60 pp，但 s1 -0.64 pp。当前瓶颈不仅是 gate，候选正收益也未稳定复现。不晋升，不启动该配方的递归 rollout 或大图；协议、成本、误接受/误拒绝及后续边界见报告。

2026-09-22 本轮更新：[TGC-R2B2 student 训练阶段对齐](current/TGC_R2B2_RESPONSE_ALIGNMENT_RESULTS_ZH_2026-09-22.md)已完成 Cora/Citeseer、两种顺序、s1/s2 的 1,248 条记录与独立重载审计。warm-up=64、suffix=1 的未加 gate student 在 s2 相对 C0 平均 +0.85 pp、相对匹配 random +1.52 pp，但冻结 gate 全部拒绝该配置，guarded 仅 0.00 pp；所有 warm-up/suffix 组合均未通过晋升条件。该结果支持“中期响应有方向性但选择器失配”的机制诊断，不升级主方法，不启动递归 rollout。

2026-09-22 本轮更新：[TGC-R2B student 反向修订历史 X 验收](current/TGC_R2B_STUDENT_REVISION_RESULTS_ZH_2026-09-22.md)。624 个新增 student fit + 48 个 C0 引用，672 条记录与 378 个不同状态重载审计通过。k1 未加 gate 在第二增量阶段平均 +0.459 pp、新类 +0.946 pp，但 guarded k1 仅 +0.060 pp，三个 horizon 均未过递归 rollout 触发条件；旧端保护项在 768 次优化探测中均未激活。保留方向性信号，不升级主方法；文本/拓扑联合修订及递归 rollout 未启动。完整计时、复测边界及后续训练阶段/旧新竞争对齐建议见报告。

2026-09-22 最新执行结果：[TGC-R2 容量/response/gate 诊断](current/TGC_R2_OPTIMIZATION_RESULTS_ZH_2026-09-22.md)。新增 336 次 canonical 训练、144 条独立 seed 短训轨迹；重载 408 个不同 checkpoint 覆盖 1008 条记录，0 errors。容量扫描未过门槛；64-step response 的 +0.41 pp 仅为离线信号。固定同一 teacher 后原 gate/no-preserve 在 12/12 阶段完全相同，撤回对原两臂分差的 gate 因果解释。文本/拓扑组合按条件未启动。下方为各分支历史摘要，“最新”均以对应日期为准。

当前最新结论：[语义拓扑强控制结项](current/SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md)已完成两大图四个新增拓扑臂、1,440 个新 fit、80 张新 memory 和独立重载审计；旧六臂 2,160 fits 作为配对参照复用。BookChild 的 semantic local-cycle 在五阶段平均 NC/LP 上优于 C0，且接近 T+degree；arXiv 的收益主要出现在部分最终或 LP 条件。uniform capacity 方向不稳定，尚不能把单环或 capacity 升为统一主方法。该批沿用旧顺序、seed 和 LP split，属于机制强控制，不是独立泛化确认；semantic medoid、完整端到端构造成本和新条件确认仍未完成。C0 保持共同冻结基线，T+degree 保留研究参照，不宣布 CTAG-final 或 SOTA。后续见 [完整算法路线及状态](current/NEXT_RESEARCH_WORKPLAN_T_DEGREE_ZH_2026-09-21.md)。

下一版算法候选：[TGC-R teacher-guided continual TAG condensation 设计](current/TEACHER_GUIDED_CONTINUAL_TAG_PROPOSAL_ZH_2026-09-22.md)：最小旧 X revision 已实现并完成 Phase A/R2；完整 budget、text、topology 统一目标仍是候选设计，不能混称为已验证模块，C0 保留共同基线。

最新 memory maintenance 结果：[C0-M core-buffer 结项](current/C0M_CORE_BUFFER_MAINTENANCE_COMPLETION_ZH_2026-09-22.md)完成 Cora/Citeseer、三种顺序、324 个正式 fit 和独立结构/checkpoint 审计。C0-M 只在 2/6 cell 不低于 C0-1 pp，平均 final F1 -0.34 pp，new F1 -0.93 pp，未晋升；C0 继续作为主方法，不能把简单几何历史刷新包装成稳定收益。

新增内容归因：[C0 与 k-center 内容差距归因报告](current/CONTENT_GAP_ATTRIBUTION_RESULTS_ZH_2026-09-22.md)完成 Cora/Citeseer canonical 三阶段、5% rate 的 5 臂受控实验与 180 次 checkpoint 独立重载。固定 C0 topology 后，k-center 的内容优势仍在部分阶段存在，但最终并不普遍超过 C0；anchor-X、anchor-text 及联合 synthetic 版本均未通过稳定晋升判断。C0 继续冻结，下一步优先验证共享 prototype assignment 的多原型 X-text 构造。

上一轮结论：[T+degree 跨任务独立确认结果](current/DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md)已完成 1080 次训练、1080 次 checkpoint 重载、60 份 memory 审计，0 超容差告警；另核对 624 个汇总单元。T+degree 通过[预定推进标准](current/DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md)，可以作为新的研究主线基础，C0 保留对照。BookChild 平均最终分类 +1.546 pp、LP AUC/AP +0.723/+0.980 pp；arXiv 分类 -0.662 pp，LP 平均基本持平但顺序依赖明显。构造成本约 1.42×/1.15×；这些是上一轮记录，不能替代新六臂成本。下方摘要保留各轮次当时结论。

新增大图链路预测拓扑验证：[大图链路预测拓扑验证报告](current/LARGE_LP_TOPOLOGY_VALIDATION_ZH_2026-09-20.md)。arXiv 与 BookChild 均完成 base+4、2 个 edge split、6 个方法臂、4 个 student seeds，共 240 fits/数据集。BookChild 上 degree-balanced 拓扑相对 C0 有约 +1.27--1.73 AUC、+1.75--2.14 AP；arXiv 基本持平，不能据此把新拓扑升为跨数据集统一主方法。当前推荐 `T+degree-balanced` 作为大图链路预测候选，C0 仍是跨任务共同控制。

新增机制对照：[链路预测核机制对照结果](current/LP_KERNEL_MECHANISM_RESULTS_ZH_2026-09-20.md)已完成 540 fits，Cora/Citeseer 各 270 fits，独立结构审计通过。结果表明 `T + degree-balanced` 已接近或超过原始 `XTG` 联合核，graph signature 不是当前链路预测提升的必要条件；度均衡是最稳定的收益来源，coverage 有数据集依赖性。本轮沿用旧 edge split，属于机制诊断，不是独立泛化验证，也不改变 C0 作为跨任务冻结基线的地位。

新增跨任务结果：[C0 与联合核链路预测比较](current/LINK_PREDICTION_COMPARISON_ZH_2026-09-20.md)已完成 540 fits。5% 节点、canonical 三阶段下，coverage 相对 C0 的最终 AUC/AP：Cora +1.68/+2.47 pp，Citeseer +3.33/+3.26 pp，三个边划分均正收益；原始联合核未稳定改善，度均衡贡献占主导。文本余弦对照仍很强，尤其 Citeseer 超过所有 GAE 版本；此为新下游开发验证，不改变下述分类晋升结论。独立重载存在小幅浮点排序告警，双精度复核不影响最终差值。

当前版本与任务状态优先查阅 [C0 基线底座与块保持联合核两阶段结项](current/PHASE12_C0_JOINT_KERNEL_COMPLETION_ZH_2026-09-20.md)：canonical 基线补齐、36 条件四臂矩阵及 3492 次正式 student fit 的独立重载验收完成，0 error；联合核 20/36 条件正收益、平均 +0.9516 pp、最差 -6.4082 pp，未通过晋升门禁，持久记忆约为 C0 的 1.89 倍。其后续的 [块保持联合核 V2 结项](current/SAFE_JOINT_KERNEL_V2_COMPLETION_ZH_2026-09-20.md)也已完成 3012 次正式 fit 和 0-error 独立重载审计；coverage 虽达到平均 +2.77 pp，但最差条件 -4.13 pp，仍未通过晋升门禁。当前共同控制仍为 C0（response off），不启动后续叠加模块或大图。此前 [v7-A 结项](current/V7A_GRADIENT_COMPATIBLE_APPEND_COMPLETION_ZH_2026-09-19.md)、[P5 总进展](current/P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)与 [P5 指标口径审计](current/P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)保留历史证据。旧 [v4 优化路线](current/V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md)用于版本谱系。本索引保留历史报告标题及当时摘要，其中的“最新”“定稿”和旧 P 编号不表示仍是现行结论。

## 当前主结论

| 文档 | 用途 |
|---|---|
| [TGC_R11_C0_ANCHOR_COMPLETION_ZH_2026-09-23.md](current/TGC_R11_C0_ANCHOR_COMPLETION_ZH_2026-09-23.md) | R10 joint 的 C0-anchor assignment 稳定化：最差顺序改善但新类塑性和阶段均值未过门禁，C0 继续主线 |
| [TGC_R12_CLASSFLOOR_COMPLETION_ZH_2026-09-24.md](current/TGC_R12_CLASSFLOOR_COMPLETION_ZH_2026-09-24.md) | R12 类别 coverage floor：未改善新类塑性，floor 增强反而伤害 old/new 平衡，C0继续冻结 |
| [TGC_R13_PLASTICITY_GATE_COMPLETION_ZH_2026-09-24.md](current/TGC_R13_PLASTICITY_GATE_COMPLETION_ZH_2026-09-24.md) | R13 bounded student-response gate：可过滤部分 anchor 坏 cell，但 new F1 仍未改善，C0继续冻结 |
| [TGC_R10_JOINT_CONFIRMATION_COMPLETION_ZH_2026-09-23.md](current/TGC_R10_JOINT_CONFIRMATION_COMPLETION_ZH_2026-09-23.md) | 独立确认联合 content-topology：4/6 最终 cell 高于 C0、交互项有正信号，但平均与新类塑性未过晋升条件；下一步为 C0 anchor 稳定化 |
| [TGC_R6_CONFIRMATION_AND_MECHANISM_RESULTS_ZH_2026-09-23.md](current/TGC_R6_CONFIRMATION_AND_MECHANISM_RESULTS_ZH_2026-09-23.md) | 两批1056槽位全部完成；原样递归独立确认未过，teacher与状态一致性有方向性证据；完整表、成本、数值敏感性及下一步 |
| [TGC_R6_FROZEN_EXPERIMENTS_ZH_2026-09-23.md](current/TGC_R6_FROZEN_EXPERIMENTS_ZH_2026-09-23.md) | 原样R5新顺序/新seeds确认，以及父状态、teacher、P、归一化和材料化诊断的事前协议 |
| [TGC_R6_MECHANISM_NUMERICAL_AMENDMENT_ZH_2026-09-23.md](current/TGC_R6_MECHANISM_NUMERICAL_AMENDMENT_ZH_2026-09-23.md) | 机制评分前处理逐字节构造复现失败；保留失败批，新增原R5封存产物控制，b版624槽位 |
| [TGC_R5_RESULTS_AND_NEXT_DECISION_ZH_2026-09-23.md](current/TGC_R5_RESULTS_AND_NEXT_DECISION_ZH_2026-09-23.md) | R5历史信号：递归joint +1.20 pp、联合交互+2.80 pp；后续泛化以R6确认结果为准 |
| [TGC_R4_INTERPRETATION_AND_NEXT_DECISION_ZH_2026-09-23.md](current/TGC_R4_INTERPRETATION_AND_NEXT_DECISION_ZH_2026-09-23.md) | R4 解释纠错与下一轮依据：研究探索/论文晋升分开，old-only 数值信号、目标与训练阶段错位、匹配对照、独立复验及候选训练更新匹配 |
| [TGC_R4_RELATIONAL_RESULTS_ZH_2026-09-23.md](current/TGC_R4_RELATIONAL_RESULTS_ZH_2026-09-23.md) | 修正版 R4 正式结果：792 槽位、504 次实际 student、四个 audit 通过；lifted support 机制成立但 relation_new 未通过晋升，relation_old 仅为不稳定描述性信号，C0 保持主线 |
| [TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md](current/TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md) | 现行 R4 设计审查：历史证据、外部 TAG 方法、旧配方纠错、新块优先关系响应、支持图尺度诊断、实现约束及待运行实验 |
| [TGC_R3_SELECTOR_CONFIRMATION_RESULTS_ZH_2026-09-22.md](current/TGC_R3_SELECTOR_CONFIRMATION_RESULTS_ZH_2026-09-22.md) | 独立顺序/种子确认：648 记录、216 次正式 fit、144 个 selector checkpoint 重放；候选与选择器均未过门槛 |
| [TGC_R3_SELECTOR_CONFIRMATION_PROTOCOL_ZH_2026-09-22.md](current/TGC_R3_SELECTOR_CONFIRMATION_PROTOCOL_ZH_2026-09-22.md) | 新 C0 构造、先选择后封存再报告、从头 32/64/128 响应曲线、唯一预注册主策略 |
| [TGC_R2B_STUDENT_REVISION_RESULTS_ZH_2026-09-22.md](current/TGC_R2B_STUDENT_REVISION_RESULTS_ZH_2026-09-22.md) | 可微 student-to-old-X：完整小图诊断、672 记录审计、stop/random/guard 对照、gate 与保护项失效分析、成本及后续条件 |
| [TGC_R2B_STUDENT_REVISION_PROTOCOL_ZH_2026-09-22.md](current/TGC_R2B_STUDENT_REVISION_PROTOCOL_ZH_2026-09-22.md) | 单阶段同父 memory 的可微 Adam/独立 seed/有界修订冻结协议及实际重复 fit 的记账补记 |
| [TGC_R2_OPTIMIZATION_RESULTS_ZH_2026-09-22.md](current/TGC_R2_OPTIMIZATION_RESULTS_ZH_2026-09-22.md) | TGC-R2 容量扫描、独立 seed response 离线诊断、共享 teacher gate 回放及全记录预测审计；无配置通过，文本/拓扑按条件未启动 |
| [TGC_R2_FROZEN_OPTIMIZATION_PROTOCOL_ZH_2026-09-22.md](current/TGC_R2_FROZEN_OPTIMIZATION_PROTOCOL_ZH_2026-09-22.md) | 结果前冻结的容量配置、保护与 quota 配对控制、合法 probe、后续轮次触发条件和审计边界 |
| [CONTENT_GAP_ATTRIBUTION_RESULTS_ZH_2026-09-22.md](current/CONTENT_GAP_ATTRIBUTION_RESULTS_ZH_2026-09-22.md) | C0/k-center 内容差距归因：固定 C0 topology 的 anchor-X、anchor-text、联合 synthetic 与真实 k-center 参照；180 次 checkpoint 重载，anchor 方向未晋升 |
| [TOPOLOGY_CAUSAL_ABLATION_RESULTS_ZH_2026-09-21.md](current/TOPOLOGY_CAUSAL_ABLATION_RESULTS_ZH_2026-09-21.md) | 两大图六臂因果拆分：2160 fits、全重载、配对差/交互/传播/成本；语义块内有条件有效，uniform capacity 未证明普适 |
| [TOPOLOGY_CAUSAL_ABLATION_PROTOCOL_ZH_2026-09-21.md](current/TOPOLOGY_CAUSAL_ABLATION_PROTOCOL_ZH_2026-09-21.md) | 首批冻结协议及显式标记的结果后执行补记，不混淆独立确认与机制消融 |
| [SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md](current/SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md) | semantic 2-NN 与 local-cycle：1440 个新 fit、NC/LP、80 张 memory 补充检查、等度边界、局部成本/存储与计数元数据更正 |
| [SEMANTIC_TOPOLOGY_CONTROLS_PROTOCOL_ZH_2026-09-21.md](current/SEMANTIC_TOPOLOGY_CONTROLS_PROTOCOL_ZH_2026-09-21.md) | 四个语义拓扑控制臂的结果前冻结协议、边数/度数约束、复用评价输入和审计边界 |
| [NEXT_RESEARCH_WORKPLAN_T_DEGREE_ZH_2026-09-21.md](current/NEXT_RESEARCH_WORKPLAN_T_DEGREE_ZH_2026-09-21.md) | 完整 CTAG 算法路线；第15节记录本批完成范围和未完成的强控制、degree-aware、维护及成本任务 |
| [DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md](current/DEGREE_CROSS_TASK_CONFIRMATION_RESULTS_ZH_2026-09-20.md) | 新顺序/新 seeds 的 1080 fits 与全重载确认；T+degree 通过研究推进标准，BookChild 两任务获益，arXiv 小幅退化/顺序依赖，含成本与零训练强对照 |
| [DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md](current/DEGREE_CROSS_TASK_CONFIRMATION_PROTOCOL_ZH_2026-09-20.md) | 结果前冻结：两大图、两个新顺序、三臂、同一 memory 跨任务、事前容忍值和完整性审计 |
| [LARGE_LP_TOPOLOGY_VALIDATION_ZH_2026-09-20.md](current/LARGE_LP_TOPOLOGY_VALIDATION_ZH_2026-09-20.md) | 大图 arXiv/BookChild 拓扑筛选：240 fits/数据集；BookChild 的 degree-balanced 有效，arXiv 基本持平；C0 仍为跨任务共同控制 |
| [LP_KERNEL_MECHANISM_RESULTS_ZH_2026-09-20.md](current/LP_KERNEL_MECHANISM_RESULTS_ZH_2026-09-20.md) | 540 fits 的机制对照：T/XT/XTG 核与 degree/coverage 约束；graph signature 必要性、度均衡主效应、数据集依赖和论文边界 |
| [LINK_PREDICTION_COMPARISON_ZH_2026-09-20.md](current/LINK_PREDICTION_COMPARISON_ZH_2026-09-20.md) | 540 fits 的链路预测比较：C0/联合核/度均衡/coverage/完整图，AUC/AP、关系切片、性能保留率、余弦强对照和成本 |
| [LINK_PREDICTION_FROZEN_PROTOCOL_ZH_2026-09-20.md](current/LINK_PREDICTION_FROZEN_PROTOCOL_ZH_2026-09-20.md) | 先隐藏真实边再重建 signature/topology，3 splits × 6 seeds；独立 GAE 下游任务，不混入原分类主表 |
| [LINK_PREDICTION_NUMERICAL_AUDIT_ZH_2026-09-20.md](current/LINK_PREDICTION_NUMERICAL_AUDIT_ZH_2026-09-20.md) | 保留近似并列分数的 GPU 重载排序告警，补全部 checkpoint 的 CPU 双精度推理，不更改原始主表 |
| [PHASE12_C0_JOINT_KERNEL_COMPLETION_ZH_2026-09-20.md](current/PHASE12_C0_JOINT_KERNEL_COMPLETION_ZH_2026-09-20.md) | 最新结项：14 种 canonical 方法/参考、36 条件矩阵、3492 fits 与 0 error 审计；联合核未通过，保留 C0，附完整性能/成本/证据边界 |
| [PHASE12_C0_JOINT_KERNEL_FROZEN_PROTOCOL_ZH_2026-09-20.md](current/PHASE12_C0_JOINT_KERNEL_FROZEN_PROTOCOL_ZH_2026-09-20.md) | 两阶段冻结协议：C0 基线复用审计、随机合成控制、1%/5%/10% 与 3/5-stage 的联合核改边矩阵及晋升门禁 |
| [SAFE_JOINT_KERNEL_V2_COMPLETION_ZH_2026-09-20.md](current/SAFE_JOINT_KERNEL_V2_COMPLETION_ZH_2026-09-20.md) | 块保持联合核 V2：2 数据集、2 顺序、1%/5%、3/5-stage，3012 fits 与 0 error；coverage/degree-balanced 有条件性平均收益但均未通过晋升门禁，C0 保持 |
| [SAFE_JOINT_KERNEL_V2_FROZEN_PROTOCOL_ZH_2026-09-20.md](current/SAFE_JOINT_KERNEL_V2_FROZEN_PROTOCOL_ZH_2026-09-20.md) | V2 冻结协议：度/覆盖/单向传播/漂移 gate/签名压缩的受控矩阵，不叠加 response、dynamic budget、refresh 或 attention |
| [POST_V2_RESEARCH_ROUTE_ZH_2026-09-20.md](current/POST_V2_RESEARCH_ROUTE_ZH_2026-09-20.md) | V2 后路线：冻结 C0，封存局部启发式负结果；下一次创新必须直接检验 student 可重训性与凝聚质量的联系 |
| [V7A_GRADIENT_COMPATIBLE_APPEND_COMPLETION_ZH_2026-09-19.md](current/V7A_GRADIENT_COMPATIBLE_APPEND_COMPLETION_ZH_2026-09-19.md) | 最新结项：144 fits、8 个 cell、候选局部有正收益但 selector 失败；C0 保留，v7-B 不启动 |
| [V7A_GRADIENT_COMPATIBLE_APPEND_FROZEN_PROTOCOL_ZH_2026-09-19.md](current/V7A_GRADIENT_COMPATIBLE_APPEND_FROZEN_PROTOCOL_ZH_2026-09-19.md) | v7-A 冻结协议：C0/recluster/multi-prototype、0/1/2-step 完整梯度、独立 selection/report seeds 与晋升门禁 |
| [V7A_AUDIT_NUMERICAL_AMENDMENT_ZH_2026-09-19.md](current/V7A_AUDIT_NUMERICAL_AMENDMENT_ZH_2026-09-19.md) | v7-A 浮点 Q 审计修订：与 frozen selector 对齐 float32 重算，选择不变，reload audit 通过 |
| [V6_RESEARCH_SYNTHESIS_AND_NEXT_DIRECTIONS_ZH_2026-09-19.md](current/V6_RESEARCH_SYNTHESIS_AND_NEXT_DIRECTIONS_ZH_2026-09-19.md) | v3-v6-A 的研究综合与 v7 设计依据；v7-A 已验证未通过，本文中的 gradient-compatible append 仍是历史待验证假设 |
| [V6A_LOCAL_MARGINAL_COMPLETION_ZH_2026-09-19.md](current/V6A_LOCAL_MARGINAL_COMPLETION_ZH_2026-09-19.md) | 最新结项：固定预算下 local merge/split 未形成稳定收益；recluster 控制局部有效但未晋升；v6-B 不启动 |
| [V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md](current/V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md) | v6-A 冻结协议：counterfactual local/recluster、独立 selection/report seeds、四个 horizon、预注册门禁 |
| [V6A_AUDIT_NUMERICAL_AMENDMENT_ZH_2026-09-19.md](current/V6A_AUDIT_NUMERICAL_AMENDMENT_ZH_2026-09-19.md) | checkpoint 重载时 GPU 浮点数值等价的审计补充，不改变冻结实验协议 |
| [C0M_CORE_BUFFER_MAINTENANCE_COMPLETION_ZH_2026-09-22.md](current/C0M_CORE_BUFFER_MAINTENANCE_COMPLETION_ZH_2026-09-22.md) | C0-M core-buffer memory maintenance：324 fits、6 cells、结构审计通过；平均 final -0.34 pp，未晋升，C0 保留为主方法 |
| [C0M_CORE_BUFFER_MAINTENANCE_FROZEN_PROTOCOL_ZH_2026-09-22.md](current/C0M_CORE_BUFFER_MAINTENANCE_FROZEN_PROTOCOL_ZH_2026-09-22.md) | C0-M 冻结协议、core/buffer 规则、审计项与晋升门禁 |
| [V5_RETENTION_EXPERIMENT_COMPLETION_ZH_2026-09-19.md](current/V5_RETENTION_EXPERIMENT_COMPLETION_ZH_2026-09-19.md) | 最新结项：288 fits、48 memory、预测审计通过；core 局部有效但未稳定晋升，topology/full 未通过，继续保留 C0 |
| [V5_RETENTION_FROZEN_PROTOCOL_ZH_2026-09-19.md](current/V5_RETENTION_FROZEN_PROTOCOL_ZH_2026-09-19.md) | v5 冻结协议：永久 core/buffer、历史传播漂移约束、四臂、新顺序/seed group、预注册晋升门槛 |
| [P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md](current/P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md) | 最新指标审计：1,558 checks / 0 errors；P5 task-slice 与历史 pooled old/new 不混比；C0 性能保留率 97.16%/106.42% |
| [P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md](current/P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md) | 当前状态入口：rate、小图长链、新顺序诊断、大图 base+4 首轮已完成；稳健性能未过门禁，成本部分缺失 |
| [P5_CURRENT_C0_BASELINE_AUDIT_ZH_2026-09-19.md](current/P5_CURRENT_C0_BASELINE_AUDIT_ZH_2026-09-19.md) | 当前 C0 与历史 v4、完整数据、全历史重凝聚及静态适配分列 |
| [SMALL_ORDER_COST_COMPARISON_REPORT_ZH_2026-09-17.md](current/SMALL_ORDER_COST_COMPARISON_REPORT_ZH_2026-09-17.md) | 新完成：1404 次报告训练与独立验收；三顺序 v4 保留率 90.93%/100.13%，但平均不敌 k-center、response 成本高；完整基线表、成本与优化次序 |
| [SMALL_ORDER_COST_PROTOCOL_ZH_2026-09-17.md](current/SMALL_ORDER_COST_PROTOCOL_ZH_2026-09-17.md) | 本轮已执行协议：冻结 v4 的两小图、三顺序、13 臂及成本测量，含 raw 数值复现验收修订；暂停 TAGSAM/STAD 补强 |
| [P0_COMPARISON_CONTROL_COMPLETION_ZH_2026-09-18.md](current/P0_COMPARISON_CONTROL_COMPLETION_ZH_2026-09-18.md) | P0 比较控制完成：统一类别/source 枚举、manifest 指纹和顺序不变性测试 |
| [P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md](current/P3A_RESPONSE_RELIABILITY_COMPLETION_ZH_2026-09-18.md) | P3-A 小图严格配对完成：response-on 未稳定优于 dispersion，且 selector 成本约高 400--600 倍 |
| [P1B_GUARDED_RESPONSE_COMPLETION_ZH_2026-09-18.md](current/P1B_GUARDED_RESPONSE_COMPLETION_ZH_2026-09-18.md) | P1-B 固定预算 guarded response transfer：平均 F1 有提升但接受率低、成本高，暂不晋升或上大图 |
| [P2_GRAPH_CONDITIONED_ATTENTION_FOLLOWUP_ZH_2026-09-17.md](current/P2_GRAPH_CONDITIONED_ATTENTION_FOLLOWUP_ZH_2026-09-17.md) | 图条件注意力原计划：P2-A 第一版已完成但未晋升；可训练投影/残差仍是后续候选 |
| [P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md](current/P2A_GRAPH_CONDITIONED_TEXT_COMPLETION_ZH_2026-09-18.md) | P2-A 小图正式结果：864 fits、图条件 query 未通过晋升门禁，保留为负结果和机制诊断 |
| [P2A_GRAPH_CONDITIONED_TEXT_PROTOCOL_ZH_2026-09-18.md](current/P2A_GRAPH_CONDITIONED_TEXT_PROTOCOL_ZH_2026-09-18.md) | P2-A 冻结协议：固定 X/rows/topology，真实邻域与 shuffled graph 反事实，64-token 文本预算 |
| [P2B_STUDENT_GUIDED_ATTENTION_COMPLETION_ZH_2026-09-18.md](current/P2B_STUDENT_GUIDED_ATTENTION_COMPLETION_ZH_2026-09-18.md) | P2-B 正式结果：648 fits、真实 student 反向 attention 梯度与审计通过，但性能门禁未通过，成本显著增加 |
| [P2B_STUDENT_GUIDED_ATTENTION_PROTOCOL_ZH_2026-09-18.md](current/P2B_STUDENT_GUIDED_ATTENTION_PROTOCOL_ZH_2026-09-18.md) | P2-B 预注册协议：固定片段/X/拓扑，semantic 与 student-guided 对照，独立 construction/report seeds |
| [P2B_FAILURE_DIAGNOSIS_AND_NEXT_EXPERIMENTS_ZH_2026-09-18.md](current/P2B_FAILURE_DIAGNOSIS_AND_NEXT_EXPERIMENTS_ZH_2026-09-18.md) | P2-B 失败原因：目标错位、残差受限、固定证据及 X/拓扑瓶颈；下一轮最小诊断矩阵 |
| [V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md](current/V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md) | 当前路线入口：版本与旧编号映射、六方向对应、P0-P5 状态；小图扩展与成本已完成，先控制行枚举再做选择器/拓扑，预算及注意力候选随后 |
| [OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md](current/OFFICIAL_GRAPHCLIP_REPORT_SPLIT_RESULTS_ZH_2026-09-17.md) | 官方 checkpoint 纯推理：当前报告划分 Cora 58.98 / Citeseer 57.76；与 v4 和完整 pool student 分层比较、逐节点与 batch 一致性核验 |
| [EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md](current/EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md) | 冻结v4对比：约5%节点的最终/阶段平均性能保留率；STAD直接TAG适配、原生风格监督诊断、成本与审计边界 |
| [EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md](current/EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md) | 最新优先级：冻结v4，先做外部TAG基线；STAD本地文本生成、加权图和统一student，消融延后 |
| [FIVE_LEVEL_GLOBAL_VS_CONTINUAL_EXPERIMENT_REPORT_ZH_2026-09-17.md](current/FIVE_LEVEL_GLOBAL_VS_CONTINUAL_EXPERIMENT_REPORT_ZH_2026-09-17.md) | 五层小图对比：新增静态TAGSAM和监督参考，660次记录审计；明确强基线及严格三方分解未完成 |
| [FIVE_LEVEL_FROZEN_PROTOCOL_ZH_2026-09-17.md](current/FIVE_LEVEL_FROZEN_PROTOCOL_ZH_2026-09-17.md) | 冻结配置：全历史响应匹配、独立种子监督网格搜索及验收边界 |
| [P2_CLOSEOUT_SMALL_BASELINE_RESULTS_ZH_2026-09-16.md](current/P2_CLOSEOUT_SMALL_BASELINE_RESULTS_ZH_2026-09-16.md) | 最新小图对照：648次下游训练、full-data/微调/coreset/TAGSAM适配；Citeseer优势与Cora差距，0 error验收 |
| [P2_CLOSEOUT_SMALL_COMPARISON_PROTOCOL_ZH_2026-09-16.md](current/P2_CLOSEOUT_SMALL_COMPARISON_PROTOCOL_ZH_2026-09-16.md) | 当前 P2 封存决定与小图对照协议；累计 F1 修正、真实 v4、完整数据/微调与 TAGSAM 适配边界 |
| [SCA_V3_RESPONSE_TOPOLOGY_OPT_RESULTS_ZH_2026-09-14.md](current/SCA_V3_RESPONSE_TOPOLOGY_OPT_RESULTS_ZH_2026-09-14.md) | **最新方法优化**：response curve + ranking consistency + frozen-old topology；Cora/Citeseer canonical 三阶段均选中第1名 |
| [STUDENT_COMPATIBLE_APPEND_RESULTS_ZH_2026-09-14.md](current/STUDENT_COMPATIBLE_APPEND_RESULTS_ZH_2026-09-14.md) | **最新方法实验**：student-compatible candidate append；Cora stage1 改善，Citeseer s2 暂无稳定收益 |
| [V3_CLOSED_CONTINUAL_RERUN_RESULTS_ZH_2026-09-14.md](current/V3_CLOSED_CONTINUAL_RERUN_RESULTS_ZH_2026-09-14.md) | **最新**：v3 封闭 test-source 重跑；统一 seeds/batch/steps，Cora/Citeseer 持续结果与边界 |
| [LOCAL_CURRENT_V3_AUDIT_ZH_2026-09-14.md](current/LOCAL_CURRENT_V3_AUDIT_ZH_2026-09-14.md) | 当前主线边界、v3 协议与远程执行审计 |
| [ARXIV_CLASSINCR_CONTINUAL_RESULTS_ZH_2026-09-10.md](current/ARXIV_CLASSINCR_CONTINUAL_RESULTS_ZH_2026-09-10.md) | **④结果**：类别增量下full_history在大图上首次落后append(s2 −0.75)，预期验证 |
| [ARXIV_CLASSINCR_SPLIT_ZH_2026-09-10.md](current/ARXIV_CLASSINCR_SPLIT_ZH_2026-09-10.md) | **④已执行**：arxiv类别增量切分已冻结(13/14/13类,池45k/阶段,test未读) |
| [ARXIV_CLASSINCR_SPLIT_DESIGN_ZH_2026-09-10.md](current/ARXIV_CLASSINCR_SPLIT_DESIGN_ZH_2026-09-10.md) | ④设计方案（预注册）：长尾处理、协议统一、陷阱预防 |
| [**BATCH_SIZE_CONFOUND_ZH_2026-09-11.md**](current/BATCH_SIZE_CONFOUND_ZH_2026-09-11.md) | **★重大缺陷**：batch是20+分混淆因素，污染此前的arm比较 |
| [KCENTER_DIAGNOSIS_RESULTS_ZH_2026-09-11.md](current/KCENTER_DIAGNOSIS_RESULTS_ZH_2026-09-11.md) | k-center诊断：矛盾定位到batch；student过拟合 |
| [**BATCH_CONTROLLED_RERUN_ZH_2026-09-11.md**](current/BATCH_CONTROLLED_RERUN_ZH_2026-09-11.md) | **★结论反转**：最优batch下ours领先(+3.0/+2.7)，但batch敏感度高(23分vs3分) |
| [FINAL_TEST_AND_BASELINES_ZH_2026-09-11.md](current/FINAL_TEST_AND_BASELINES_ZH_2026-09-11.md) | **test读取+基线**：4数据集test无偏移；但k-center在4/6上超过本方法 |
| [FUSE_CHARS_ROOT_CAUSE_ZH_2026-09-11.md](current/FUSE_CHARS_ROOT_CAUSE_ZH_2026-09-11.md) | 根因诊断（文本截断假设，后被证伪） |
| [KCENTER_GAP_DIAGNOSIS_ZH_2026-09-11.md](current/KCENTER_GAP_DIAGNOSIS_ZH_2026-09-11.md) | **k-center差距归因**：三个假设(文本/合成X/覆盖性)全部证伪，未归因 |
| [**P1_HANDOFF_TO_CHATGPT_ZH_2026-09-15.md**](current/P1_HANDOFF_TO_CHATGPT_ZH_2026-09-15.md) | **★交接报告**：P1全部数据+判定(组合规则失败/sportsfit劣势)+剩余待办+资源索引+历史坑 |
| [**P1_ROUND1_BUDGET_ALLOCATION_ZH_2026-09-15.md**](current/P1_ROUND1_BUDGET_ALLOCATION_ZH_2026-09-15.md) | **★P1第一轮**：修正协议下ours全面胜过真实节点基线；citeseer上dispersion是Pareto改进(+5.4)但cora上legacy_15最优——allocation方向数据集相关 |
| [**P1_CLAUDE_HANDOFF_REVIEW_ZH_2026-09-15.md**](current/P1_CLAUDE_HANDOFF_REVIEW_ZH_2026-09-15.md) | **★审查修正**：发现 legacy_15 的新旧 rows quota 混淆，完成 matched-quota 受控重跑 |
| [**P1_SMALL_GRAPH_RESULTS_ZH_2026-09-15.md**](current/P1_SMALL_GRAPH_RESULTS_ZH_2026-09-15.md) | **★小图验收**：Cora/Citeseer allocation 与 fixed-total/hybrid/adaptive-rate 结果；暂不上大图主表 |
| [V4_TARGET_METHOD_AND_OPTIMIZATION_PLAN_ZH_2026-09-15.md](current/V4_TARGET_METHOD_AND_OPTIMIZATION_PLAN_ZH_2026-09-15.md) | **★当前优化总设计**：两个负组件的 v2 优化、最终凝聚流程、P1-P5 分阶段实现与验收边界 |
| [V4_MAINLINE_SMALL_RESULTS_ZH_2026-09-15.md](current/V4_MAINLINE_SMALL_RESULTS_ZH_2026-09-15.md) | **★v4主线小图验收**：LCB + worst-horizon 安全 response selector；Cora 回退、Citeseer 小幅提升 |
| [P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md](current/P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md) | **P2完成**：固定 token 文本矩阵、DeepSets、独立 seeds、batch/rate/排列/指标审计；完整性通过，性能门禁未通过 |
| [P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md](current/P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md) | **P5小图扩展**：1%/5%/10% rate curve、5-stage/base+4、实际 batch 和成本账 |
| [P5_INDEPENDENT_ORDER_COMPLETION_ZH_2026-09-19.md](current/P5_INDEPENDENT_ORDER_COMPLETION_ZH_2026-09-19.md) | **P5独立顺序**：Cora/Citeseer frequency 顺序诊断，确认结果存在顺序依赖 |
| [P5_LARGE_BASE4_COMPLETION_ZH_2026-09-19.md](current/P5_LARGE_BASE4_COMPLETION_ZH_2026-09-19.md) | **P5大图首轮**：arXiv、BookChild、SportsFit 的冻结 C0 base+4 长链、F1 与成本结果 |
| [P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md](current/P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md) | **P5总报告**：rate、长链、独立顺序、大图外推、成本边界与后续决策 |
| [P5_CURRENT_C0_BASELINE_AUDIT_ZH_2026-09-19.md](current/P5_CURRENT_C0_BASELINE_AUDIT_ZH_2026-09-19.md) | **当前 C0 基线审计**：拆分 v4(response on) 与 C0(response off)，核对五层对照可比性 |
| [TGC_R4_REDESIGN_AND_NEXT_EXPERIMENTS_ZH_2026-09-22.md](current/TGC_R4_REDESIGN_AND_NEXT_EXPERIMENTS_ZH_2026-09-22.md) | 历史 MB-LRR 草案，已被 2026-09-23 设计审查取代；原因果归因、默认配方和决策树不可直接执行 |
| [TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md](current/TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md) | **最新**：修复 role-match 的 payload 不守恒缺陷；六 cell 守恒置换对照，semantic permutation 局部有效但尚未稳定晋升，role cost 不成立 |
| [**FINAL_METHOD_AND_PAPER_READINESS_ZH_2026-09-11.md**](current/FINAL_METHOD_AND_PAPER_READINESS_ZH_2026-09-11.md) | **★定稿**：最终方法细节+四条主张+论文可行性评估+缺口清单 |
| [**METHOD_FINAL_DETAILED_SPEC_ZH_2026-09-10.md**](current/METHOD_FINAL_DETAILED_SPEC_ZH_2026-09-10.md) | **★超参数与设计理由全书**：逐参数给出取值+代码位置+为何这样设计+已证伪的替代方案 |
| [**METHOD_V12_FULL_SPEC_ZH_2026-09-10.md**](current/METHOD_V12_FULL_SPEC_ZH_2026-09-10.md) | **★主文档**：v1.2完整流程+代码讲解+全数据集结果(含Accuracy)+理论+创新点 |
| [GATE_THIRD_FALSIFICATION_ZH_2026-09-11.md](current/GATE_THIRD_FALSIFICATION_ZH_2026-09-11.md) | **判据第三次证伪**：k相同结论相反(sfit_k6000 vs sportsfit)→扰动模块不成立，应从论文删除 |
| [REDUNDANCY_CRITERION_RESULTS_ZH_2026-09-11.md](current/REDUNDANCY_CRITERION_RESULTS_ZH_2026-09-11.md) | **判据修正**：n_pool与文本冗余度均被证伪，正确判据是簇数k≥4000(5/5) |
| [NEW_BENCHMARKS_RESULTS_ZH_2026-09-11.md](current/NEW_BENCHMARKS_RESULTS_ZH_2026-09-11.md) | **新数据集验证**：4个全新TAG benchmark；规模门槛被证伪，改为"每类节点数≥5k"；ASM在6/7数据集有害 |
| [ADAPTIVITY_TESTS_RESULTS_ZH_2026-09-10.md](current/ADAPTIVITY_TESTS_RESULTS_ZH_2026-09-10.md) | **两线证伪**：η平滑度统计量与规模共线；k失真肘部与F1平台差10倍。方法形态收敛 |
| [FOUR_ADAPTIVITY_QUESTIONS_ZH_2026-09-10.md](current/FOUR_ADAPTIVITY_QUESTIONS_ZH_2026-09-10.md) | **四问分析**：k动态化/扰动动态化=真创新；c仅弱形式；arxiv须改类别增量 |
| [SCALE_GATE_RESULTS_ZH_2026-09-10.md](current/SCALE_GATE_RESULTS_ZH_2026-09-10.md) | **门槛定位**：扰动收益在池规模2.7万→5.4万由负转正，S≈3–5万 |
| [MULTISCALE_ASM_RESULTS_ZH_2026-09-10.md](current/MULTISCALE_ASM_RESULTS_ZH_2026-09-10.md) | **重要修正**：三尺度ASM——loss不降是规模鲁棒的，但ASM收益规模依赖(小图持平/大图+3.3) |
| [ARXIV_XONLY_AND_WEAK_PERTURBATION_ZH_2026-09-10.md](current/ARXIV_XONLY_AND_WEAK_PERTURBATION_ZH_2026-09-10.md) | **关键翻案**：X-only对照推翻弱扰动假说，ASM大图上是方向有效的真优化 |
| [ARXIV_ANCHORED_OPT_ZH_2026-09-10.md](current/ARXIV_ANCHORED_OPT_ZH_2026-09-10.md) | **方法最终形态**：构造式+~50步无锚ASM(18.5→22.0)；锚定有害；ASM loss无诊断性 |
| [ARXIV_ASM_5SEED_ZH_2026-09-10.md](current/ARXIV_ASM_5SEED_ZH_2026-09-10.md) | **判定通过+两处修正**：ASM-frozen +3.29(CI不含0,5/5)但随机扰动也+2.00，ASM仅+1.30方向特有 |
| [ARXIV_DEBIAS_ZH_2026-09-10.md](current/ARXIV_DEBIAS_ZH_2026-09-10.md) | **去偏假设证伪**：白化/尺度对齐/Procrustes全失败，仅各向同性随机扰动有效；ASM≈随机 |
| [ARXIV_JITTER_SWEEP_ZH_2026-09-10.md](current/ARXIV_JITTER_SWEEP_ZH_2026-09-10.md) | **v1.2模块定型**：jitter幅度倒U峰值η≈1.79(=ASM自发漂移)、pooled+1.64稳健、小图有害→大图专属 |
| [ARXIV_CONTINUAL_AND_RAW_ZH_2026-09-10.md](current/ARXIV_CONTINUAL_AND_RAW_ZH_2026-09-10.md) | **最新**：arxiv raw参照+10%+temporal3持续——5%凝聚=raw的94.6%，append追平full_history，类别增量优势边界确立 |
| [ARXIV_SCALE_GATE_ZH_2026-09-09.md](current/ARXIV_SCALE_GATE_ZH_2026-09-09.md) | **最新**：ogbn-arxiv 169k规模gate——全程212s/凝聚30s，5%质量通过，scalability无障碍 |
| [THEORY_DRAFT_V01_ZH_2026-09-09.md](current/THEORY_DRAFT_V01_ZH_2026-09-09.md) | **理论草稿v0.1**：三定理(换序性零代价/组件性质/无代理分离)+证明骨架+实验对应 |
| [T1_NO_PROXY_RESULT_ZH_2026-09-09.md](current/T1_NO_PROXY_RESULT_ZH_2026-09-09.md) | **最新**：11 artifact×5预测器全部失效——"可重训性无代理"命题，学习组件失败的结构性解释 |
| [HEADROOM_GATE_DIRECTION_B_ZH_2026-09-09.md](current/HEADROOM_GATE_DIRECTION_B_ZH_2026-09-09.md) | **最新**：方向B headroom gate——oracle纯度100%不超过基线，可学习聚类关闭，五组件定律闭环 |
| [V3_DISCRIMINATIVE_FUSION_RESULTS_ZH_2026-09-09.md](current/V3_DISCRIMINATIVE_FUSION_RESULTS_ZH_2026-09-09.md) | **最新**：判别性感知融合4修复全败——"分布同构性>判别性"定律确立，方向A关闭 |
| [V2_LLM_FUSION_FULL_RESULTS_ZH_2026-09-09.md](current/V2_LLM_FUSION_FULL_RESULTS_ZH_2026-09-09.md) | **最新**：本地Qwen3 LLM融合完整对照——同预算下判别性>语义质量，v1保持冻结 |
| [RESEARCH_STORY_FROM_TAGSAM_TO_FROZEN_V1_PLAIN_ZH_2026-09-09.md](current/RESEARCH_STORY_FROM_TAGSAM_TO_FROZEN_V1_PLAIN_ZH_2026-09-09.md) | **从头读的通俗版**：基础概念、22步研究因果链、每步做法与结果、最终算法算例、代码导航和结论边界 |
| [FULL_RETROSPECTIVE_AND_CODE_WALKTHROUGH_ZH_2026-09-09.md](current/FULL_RETROSPECTIVE_AND_CODE_WALKTHROUGH_ZH_2026-09-09.md) | **完整复盘**：12步决策台账+冻结方法逐段代码讲解+取舍总模式 |
| [GATE3_TEST_CONFIRMATION_ZH_2026-09-09.md](current/GATE3_TEST_CONFIRMATION_ZH_2026-09-09.md) | **frozen_v1 最终确认表**：test一次性读取，全部主张确认，方法两数据集均居首 |
| [OFFICIAL_GRAPHCLIP_COMPARISON_AND_CORRECTIONS_ZH_2026-09-09.md](current/OFFICIAL_GRAPHCLIP_COMPARISON_AND_CORRECTIONS_ZH_2026-09-09.md) | **最新**：官方GraphCLIP对比（差距<3pp）+append≡分层配额等价审计+c规则冻结，Gate 3''前置条件齐备 |
| [GATE2_CITESEER_RESULTS_ZH_2026-09-08.md](current/GATE2_CITESEER_RESULTS_ZH_2026-09-08.md) | **最新**：Citeseer 第二数据集验证通过——核心结论迁移，保持-塑性前沿框架确立 |
| [BUDGET_SWEEP_AND_RAW_UPPER_ZH_2026-09-08.md](current/BUDGET_SWEEP_AND_RAW_UPPER_ZH_2026-09-08.md) | **最新**：预算分配消融+raw直训上界——compact超过raw，确定性分配规则冻结 |
| [GATE1_CONTINUAL_RESULTS_ZH_2026-09-08.md](current/GATE1_CONTINUAL_RESULTS_ZH_2026-09-08.md) | **最新**：构造式持续凝聚 Gate 1'' 通过——append-构造式追平 full-history，"只追加不重组"确立 |
| [GATE_S2_ABLATION_RESULTS_ZH_2026-09-08.md](current/GATE_S2_ABLATION_RESULTS_ZH_2026-09-08.md) | **最新**：TAGSAM组件回收消融——三组件全部降级，v1构造器冻结为现行方法 |
| [GRAPHCLIP_CONSTRUCTIVE_GATE_RESULTS_ZH_2026-09-08.md](current/GRAPHCLIP_CONSTRUCTIVE_GATE_RESULTS_ZH_2026-09-08.md) | **最新**：GraphCLIP 下游 + 合成X + 融合文本的构造式 gate，通过 |
| [CONSTRUCTIVE_GATE_RESULTS_ZH_2026-09-08.md](current/CONSTRUCTIVE_GATE_RESULTS_ZH_2026-09-08.md) | **最新**：确定性构造式静态凝聚方案，gate 通过（77.2，零 condensation-seed 方差） |
| [STAGE1_RETENTION_GATE_RESULTS_ZH_2026-09-08.md](current/STAGE1_RETENTION_GATE_RESULTS_ZH_2026-09-08.md) | **最新**：response-retention gate 失败判定与"响应约束≠可重训性"机制结论 |
| [STATIC_DIAG_AND_BUDGET_ALLOC_RESULTS_ZH_2026-09-08.md](current/STATIC_DIAG_AND_BUDGET_ALLOC_RESULTS_ZH_2026-09-08.md) | **最新**：5-seed 静态优化诊断（ASM loss 不降）与预算分配扫描 |
| [TAGSAM_PAPER_CALIBRATION_ZH_2026-09-08.md](current/TAGSAM_PAPER_CALIBRATION_ZH_2026-09-08.md) | **最新**：论文报告值 vs 本地复现值校准（差距 21–35 pp 未决） |
| [EXPERIMENT_WALKTHROUGH_ZH_2026-09-08.md](current/EXPERIMENT_WALKTHROUGH_ZH_2026-09-08.md) | 从任务、指标到十组实验的通俗逐项解释 |
| [CURRENT_PROGRESS_SUMMARY_ZH_2026-09-08.md](current/CURRENT_PROGRESS_SUMMARY_ZH_2026-09-08.md) | 当前选题、协议、证据和下一步的完整总结 |
| [TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md](current/TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md) | 文本漂移、方法定义、同 outer 与严格递归结果 |
| [DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md](current/DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md) | centroid-X、outer 轨迹、文本置换和 alignment 归因 |
| [FINE_GRAINED_CONTINUAL_TAG_ATTRIBUTION_2026-09-07.md](current/FINE_GRAINED_CONTINUAL_TAG_ATTRIBUTION_2026-09-07.md) | old-X/new-X、冻结与 joint 更新的单因素实验 |
| [STAGE2_SOURCE_COMPOSITION_ATTRIBUTION_2026-09-07.md](current/STAGE2_SOURCE_COMPOSITION_ATTRIBUTION_2026-09-07.md) | 旧 memory、新 raw、新 compact 的组成归因 |
| [OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md](current/OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md) | 官方 master、add-adj、minimal repair 的静态复核 |
| [STATIC_TAGSAM_AND_CONTINUAL_PROTOCOL_CLARIFICATION_ZH_2026-09-08.md](current/STATIC_TAGSAM_AND_CONTINUAL_PROTOCOL_CLARIFICATION_ZH_2026-09-08.md) | 静态与持续实验的关系、评测口径说明 |
| [NATIVE_TAG_V1_FEASIBILITY_ATTRIBUTION_AND_PROGRESS_ZH_2026-09-07.md](current/NATIVE_TAG_V1_FEASIBILITY_ATTRIBUTION_AND_PROGRESS_ZH_2026-09-07.md) | native v1 可行性、早期归因与阶段性结论的历史审计 |

## 协议和研究设计

| 文档 | 用途 |
|---|---|
| [DETFUSE_V2_AND_CONTINUAL_PLAN_ZH_2026-09-08.md](protocol/DETFUSE_V2_AND_CONTINUAL_PLAN_ZH_2026-09-08.md) | **最新**：静态升级（det-CMI/粗化拓扑/一阶精炼）与持续凝聚（加权再聚类）设计预注册 |
| [PROTOCOL_V2_ZH_2026-09-08.md](protocol/PROTOCOL_V2_ZH_2026-09-08.md) | **现行冻结协议**：预算二元组、response 档A、统计判定规则、gate 序列 |
| [EXECUTION_PROTOCOL_2026-09-07.md](protocol/EXECUTION_PROTOCOL_2026-09-07.md) | native 实验的固定执行配置 |
| [ATTRIBUTION_ADDENDUM_2026-09-07.md](protocol/ATTRIBUTION_ADDENDUM_2026-09-07.md) | 后续归因、teacher 共享修正和解释边界 |
| [RESEARCH_PROPOSAL_PROTOCOL_AND_EXPERIMENTS_ZH_2026-09-06.md](protocol/RESEARCH_PROPOSAL_PROTOCOL_AND_EXPERIMENTS_ZH_2026-09-06.md) | 选题、setting 和实验计划的初始版本 |
| [CAT_DANCE_BASED_CONTINUAL_TAG_RESEARCH_DESIGN_ZH_2026-09-06.md](protocol/CAT_DANCE_BASED_CONTINUAL_TAG_RESEARCH_DESIGN_ZH_2026-09-06.md) | 与 CaT、DANCE、TAGSAM 的差异和创新边界 |
| [STARTING_ROUTE_SETTING_AND_MODEL_DECISION_ZH_2026-09-06.md](protocol/STARTING_ROUTE_SETTING_AND_MODEL_DECISION_ZH_2026-09-06.md) | 为什么先使用 TAGSAM 原生下游、暂缓 GraphCLIP |
| [NEXT_STEP_RESPONSE_RETENTION_GATE_ZH_2026-09-08.md](protocol/NEXT_STEP_RESPONSE_RETENTION_GATE_ZH_2026-09-08.md) | 下一版 response-retention 的最小实现范围、对照和验收门槛 |

## 研究谱系

`lineage/` 中的文件用于解释方向如何从旧分支重置到经典持续 synthetic TAG。它们不覆盖 `current/` 中日期更晚的结论。

- `NEW_CONVERSATION_HANDOFF_CLASSICAL_CONTINUAL_TAG_ZH_2026-09-05.md`：权威交接；
- `FROZEN_RESEARCH_SNAPSHOT_BEFORE_CLASSICAL_RESET_2026-09-04.md`：旧方向冻结边界；
- `CLASSICAL_CONTINUAL_TAG_GRAPH_CONDENSATION_ROADMAP_ZH_2026-09-04.md`：经典路线起点；
- `CURRENT_TOPIC_METHOD_AND_FORWARD_PLAN_ZH_2026-09-05.md`：早期方法雏形；
- `GRAPHCLIP_ONLY_CONTINUAL_TAG_SCOPE_ZH_2026-09-06.md`：已被晚于它的原生下游决策取代，只作历史记录。
- `NEW_CONVERSATION_START_ZH_2026-09-05.txt`：旧交接时使用的简短启动提示。
# 2026-09-24 TGC 双原型 C0-anchor 小图验证已完成：[实验报告](current/TGC_TWO_PROTOTYPE_C0_ANCHOR_SMOKE_COMPLETION_ZH_2026-09-24.md)。严格 320 字符同步协议下，degree-gate 只在 Cora/order19601 提升 +0.998 pp，在 Citeseer/order19601 和 Cora/order19602 分别为 -4.012/-0.716 pp；anchor 可缓解部分 old-F1 退化但不能稳定晋升，C0 继续作为主方法。此前 +2.328 pp 结果来自旧远程依赖快照，不再作为可比数字。

# 2026-09-26 TGC Message-Conditioned X 独立确认已完成：[实验报告](current/TGC_MESSAGE_CONDITIONED_X_COMPLETION_ZH_2026-09-26.md)。12 个 cell 中 81/81 个 changed rows 的局部 message error 下降，但 Student 累计 F1 相对 C0 为 -0.824 pp、new-class F1 为 -3.415 pp；该代理目标与新类塑性失配，转向 bounded shared-P+X 联合设计。
# 2026-09-26 Shared-P + bounded-X 真冻结拓扑验收：[协议修复与真结果报告](current/TGC_SHARED_P_BOUNDED_X_FROZEN_TOPOLOGY_COMPLETION_ZH_2026-09-26.md)。发现并修复旧版 freeze-topology 只替换最终 edge、未冻结内层 rollout 的协议问题。修正版 12-cell 结果 shared-PX 相对 C0 平均 -0.154 pp、5/12 正 cell、new-class -2.295 pp，未晋升；C0 继续作为主线。
# 2026-09-26 Role-aware 全 arm 统一 Student 重载已完成：[完整评估报告](current/TGC_ROLE_AWARE_FULL_ARM_EVALUATION_ZH_2026-09-26.md)。在 12 个独立 cell 上，semantic permutation 相对 C0 平均 +1.690 pp、9/12 正 cell、最差 -1.904 pp，暂列下一轮新顺序/大图确认候选；role permutation 平均 +1.468 pp 但仅 7/12 正 cell，保留为机制消融。
# 2026-10-04 XSP-Sinkhorn 独立 construction seed 确认已完成：[独立确认报告](current/XSP_ST_SINKHORN_INDEPENDENT_CONFIRMATION_ZH_2026-10-04.md)。在 Arxiv/BookChild/SportsFit 的 base+4、5% memory、6 个 Student seeds 下，ST-Sinkhorn 相对 Current XSP 的三图等权 Final F1 为 +0.62 pp，New F1 为 +2.73 pp；Arxiv 与 SportsFit 为正、BookChild 为负，说明梯度组件有可复现方向性信号但跨数据集增益尚未稳定。原始两层 GCN 与单层 GCN 的冻结 memory 泛化验证已启动。
# 2026-10-04 XSP Student 泛化验证已完成：[Student 泛化报告](current/XSP_ST_STUDENT_GENERALIZATION_ZH_2026-10-04.md)。在冻结的 Current XSP/ST-Sinkhorn memory 上替换原始两层 GCN为单层 GCN，保持 InfoNCE、batch、steps 和 6 个 seeds 不变；ST 相对 Current 的 Final F1 在两种 Student、三张大图的 6/6 配对 cell 均为正，两层平均 +0.36 pp，单层平均 +1.58 pp，New F1 平均分别 +1.08/+1.45 pp。支持 ST-Sinkhorn 作为主方法核心，但仍需同快照公开基线重跑和多顺序确认。
# 2026-10-09 Unified-Q 3×3 独立确认已完成：[完整实验报告](current/UNIFIED_Q_EXTENDED_CONFIRMATION_ZH_2026-10-09.md)。27 个构造 cell、27 个 GraphSAGE readout 全部完成，统一汇总 Final F1/Accuracy、Old/New F1、配对差值、成本和可视化；结果支持部分数据集收益，但不支持稳定 SOTA 或普遍超过 GECC。
