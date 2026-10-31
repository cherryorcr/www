# 新对话交接报告：经典持续文本属性图凝聚

更新：2026-09-05。用途：让新对话直接理解用户目标、接续已有项目，并避开过去反复出现的问题。

**当前状态：用户已决定冻结原 prototype-memory 研究分支，回到经典 synthetic graph condensation。新路线已有设计草案，本轮没有执行新的凝聚实验，不能报告新 C0 或 N0 已通过。**

## 1. 用户真正要做的课题

用户最近明确纠正了方向：希望做经典图凝聚，让小型 synthetic graph 上训练的模型尽量复现真实图上的训练效果。差异点聚焦两个方面：持续/演化场景，以及目标图是文本属性图 TAG。

工作题目可写为：**面向演化文本属性图的持续图凝聚**。

目标形式为：

\[
\widetilde G_t=(\widetilde X_t,\widetilde A_t,\widetilde T_t,\widetilde Y_t),
\qquad K_t\ll N_{\le t}.
\]

Synthetic feature、结构、文本状态和监督目标可以成为优化变量。核心判据是压缩数据能否支持有效训练，而非每个 synthetic node 是否对应某个真实 medoid。

用户的稳定偏好：

- 目标是有实质算法贡献的 CCF A 类会议长论文，不能把负结果研究设为默认最终成果。
- TAG 的文本应真正参与算法，用户重视真实可读文本片段；仅附带一个文本字段不够。
- 接受针对一种主要下游模型，不要求所有 backbone 都提高。GraphCLIP 是此前偏好的 TAG-native 候选，GraphSAGE 可作监督对照。
- 有明确压缩率。固定比例不等于永久固定总字节；允许合理放宽协议以获得研究空间。
- 历史完整原图不能作为免费隐藏 replay，但“绝对禁止任何历史样本”不是不可讨论的教条；显式保留的合成图或受预算约束的文本可以 replay。
- 先验证基础模型、基线和可改善空间，再开发新算法。一个 gate 失败后先定位机制，不要立即换选题、换名字或堆模块。
- 协议相同时复用冻结结果；更换模型、切分或训练接口时重新建立匹配基线，不能跨协议借用绝对数值。
- 中文报告、直白判断、明确已完成/未完成；避免反复请求已经给过的授权。

**D2 已退出主方法定位。它是已有正结果，可作为基线或初始化，用户并未要求下一版必须继承 D2 的双角色、融合或锚定方式。**

## 2. 接手时需要纠正的概念

1. 经典 graph condensation 本身通常也是为训练而构造小图。它不必同时完成“仅用压缩图进行所有原图查询”的部署闭合；后者是另一项更强目标。此前报告曾把两者混为判定标准。
2. 经典凝聚并不要求每篇方法都同时自由优化 X、A、Y、T；固定标签或无边凝聚也有文献先例。用户希望回到可优化的合成数据主线，不代表第一版必须一次放开全部变量。
3. “持续 TAG”是研究定位，尚不是已经证实独创的算法。动态重凝聚、transport、soft prompt、trajectory matching 都已有邻近研究，不能因组合后换名就称原创。
4. 冻结模型权重与让梯度穿过模型是两件事；如果优化 soft text token，文本 encoder 虽冻结参数，仍须支持输入梯度和可复现导出。
5. Soft prompt/连续文本向量不是可读文本。若采用它们，必须另行定义文本片段的保存、使用与导出，不能绕过用户的 TAG 文本诉求。
6. GraphCLIP 的无标签 InfoNCE 与监督 soft-label CE 是不同目标。若主任务是 label-free 对齐或 unseen-class zero-shot，不能默默用类别标签优化 synthetic Y；监督与无监督分支应分别定义。

## 3. 工作区与资产位置

本地项目根目录：

`D:/Adocs/简历/research/graph-condensation/TAGSAM`

本地结果：

`D:/Adocs/简历/research/graph-condensation/TAGSAM/problem_validation_results/`

旧强基线镜像：

`D:/Adocs/简历/research/graph-condensation/TAGSAM/frozen_baselines/`

远程历史位置：

`pc@REMOTE_SERVER_HOST:/home/pc/research/TAGSAM/`

历史资源：远程双 RTX 4090，本地也曾运行实验。用户后期已授权使用远程 GPU，并允许释放 vLLM。**本交接没有查询远程当前状态，也没有启动/停止进程；接手后先核查资源，不依据历史空闲状态直接杀进程。**

长期记忆：

`D:/Aprojects/AI-Memory/Projects/Continual-TAG-Condensation.md`

旧长期记忆包含许多历史默认方案；最新重置说明见同目录 `Classical-Continual-TAG-Condensation-Handoff.md`。

## 4. 应先读的资料与优先级

先读本报告，再按问题读取以下文件。旧文档中“下一步”“最终方法”是当时决策，不能覆盖用户最新的方向修正。

| 目的 | 文件 |
|---|---|
| 全部主要实验家族、历史数据和相关工作 | [完整历史报告](D:/Adocs/简历/research/graph-condensation/TAGSAM/CONTINUAL_TAG_CONDENSATION_FULL_RESEARCH_REPORT_ZH_2026-09-04.md) |
| 冻结范围与九份报告哈希 | [冻结快照](D:/Adocs/简历/research/graph-condensation/TAGSAM/FROZEN_RESEARCH_SNAPSHOT_BEFORE_CLASSICAL_RESET_2026-09-04.md) |
| 最新方向草案，阅读时结合本报告第 8 节 | [经典凝聚路线草案](D:/Adocs/简历/research/graph-condensation/TAGSAM/CLASSICAL_CONTINUAL_TAG_GRAPH_CONDENSATION_ROADMAP_ZH_2026-09-04.md) |
| D2 的原始 probe 结果与限制 | [D2 正式报告](D:/Adocs/简历/research/graph-condensation/TAGSAM/D2_ROLE_AWARE_CONTINUAL_UPDATE_REPORT_2026-08-31.md) |
| 真实 SAGEConv 结果 | [A2 正式报告](D:/Adocs/简历/research/graph-condensation/TAGSAM/A2_GRAPHSAGE_FORMAL_GATE_REPORT_2026-08-31.md) |
| 撤回 union/shared loss 的早期错误正结果 | [A3 校正报告](D:/Adocs/简历/research/graph-condensation/TAGSAM/A3_CHECKPOINT_SELECTION_CORRECTION_REPORT_2026-08-31.md) |
| 自由 synthetic X/Z/Q 失败 | [O2 决策](D:/Adocs/简历/research/graph-condensation/TAGSAM/O2_FULL_SET_BILEVEL_SYNTHETIC_ORACLE_DECISION_ZH_2026-09-01.md) |
| 多步轨迹已经尝试过，不能遗漏 | [Trajectory oracle 决策](D:/Adocs/简历/research/graph-condensation/TAGSAM/TRAJECTORY_LEVEL_SYNTHETIC_ORACLE_DECISION_ZH_2026-09-01.md) |
| GraphCLIP 接口有效、unit 无稳定收益 | [G0/G1 审计](D:/Adocs/简历/research/graph-condensation/TAGSAM/GRAPHCLIP_G0_G1_GATE_AUDIT_2026-09-03.md) |
| 早期 t3 已经评测的直接证据 | [早期冻结 t3 报告](D:/Adocs/简历/research/graph-condensation/TAGSAM/FINAL_FROZEN_T3_REPORT.md) |
| 文献边界 | [相关工作与研究复盘](D:/Adocs/简历/research/graph-condensation/TAGSAM/RESEARCH_DIRECTION_LITERATURE_AND_EXPERIMENT_PLAN_ZH_2026-09-02.md) |
| 静态 TAGSAM 复现历史与分支问题 | [长期记忆中的 TAGSAM 复现记录](D:/Aprojects/AI-Memory/Projects/TAGSAM-Baseline-Reproduction.md) |

## 5. D2 到底做了什么、效果如何

Moment 分支按类别保存 `(count, feature_sum, feature_sum_sq, label_sum)`，旧统计与当前统计相加，再由均值/逐维方差构造 sigma prototypes。Coverage 分支在旧序列化代表与当前候选之间做加权聚类和替换，保存 support、soft label、代表 ID 与 distortion。Coverage feature 可以是聚合 centroid，不应全称为未经改变的真实 medoid feature。

A2 训练两个半宽 GraphSAGE 专家，使用 self-loop synthetic training topology，再以 `0.25 moment + 0.75 coverage` 融合预测。alpha 是开发集校准后固定的全局权重。

| 协议 | ArXiv Macro-F1 | PubMed Macro-F1 | 对比含义 |
|---|---:|---:|---|
| D2 原始角色记忆 probe | 47.58% | 83.78% | 对其预选基线 +1.21/+1.25 pp |
| A2 真实 SAGEConv，Continual | 46.98% | 78.09% | 对 controller 预选 Append/PUMA-style 分别 +3.50/+4.35 pp |
| A2 在同张 qualification 表中的最佳非 D2 单角色基线 | PUMA-style 44.08% | Append 77.80% | D2 对这两个表内最高基线仅 +2.90/+0.29 pp |

必须同时保留“预选基线”与“表内最高基线”两种口径。PubMed 的 +4.35 pp 不能写成超过所有强基线的优势；它相对 Append 的均值优势只有 +0.29 pp，而且 Accuracy 77.30% 略低于 Append 77.49%。

A2 报告的五个 seeds 是 `100..104` 下游模型 seeds，构造使用冻结确定性 memory。不能将它写成五个独立凝聚种子。D2/A2 的 old-stage 指标来自历史可见节点，是 retention diagnostic，不是独立旧阶段测试集。

D2 的正结果支持递归状态与压缩训练数据有价值；state shuffle 变差提供对应关系证据，但不能单独证明收益完全来自新递推算法，也不能排除集成多样性的作用。

冻结字节上限 ArXiv/PubMed 为 4,648,093/676,030 bytes，D2 为 3,088,375/399,605 bytes。准确表述是相同名义 root 数、处于相同字节上限内，不是所有 arm 完全相同有效字节。

D2 可保留为初始化、同协议基线和可选稳定约束；不要求新经典凝聚继承两个专家。

## 6. 过去实验提供了什么证据

下面是交接用索引。详细数值和逐 seed 记录按第 4 节及原报告查询，不能把跨协议结果拼成统一排行榜。

| 家族 | 做过什么 | 可保留的判断 |
|---|---|---|
| 静态 TAGSAM 复现 | Cora teacher、released master、batch 修正与 topology branch | Teacher 曾接近论文；condensation 不稳定，存在学习步长/feature 范数膨胀和代码分支差异。未形成可靠复现起点 |
| P0-P2 | k-medoids、K2/K3 有界 feature 更新、full text/title/centroid/evidence | Joint selection 有用；full text 不应随意替换为 title；K3 对强 online baseline 没有跨数据集独立优势 |
| P3-P4 | provenance 边、learned utility、unit/count weight、时间及 action 预算 | 真 endpoint 含信息，算子/边强度影响大；几何/novelty utility 不稳定转为训练收益。固定率资源曲线保留 |
| V5-V9、RAM、message | 统计递推、drift、response、多算子、消息重构、rollback | 部分 proxy 变好但任务不提高；no-op 可减损，不能凭空制造收益 |
| 强 coreset/gradient sketch | Reservoir、online k-center/k-medoids、K2/K3、recursive sketch | GraphSAGE 旧协议下 K3 相对 online k-medoids 为 +0.41/-2.64 pp，递归梯度为 +0.05/-0.37 pp |
| STPM/O1/actions | pointwise influence、full artifact reward、conditional bundle | 单点收益和集合收益不等价；短程 reward 的排名不一定迁移到完整训练 |
| D0-D2/A2 | 确定性 moment、coverage、递推和 fixed fusion | 有正式正结果，但仍缺 TAG 特有因果机制 |
| A3 | union one-head、global moment KL、role shuffle | 校正 checkpoint 选择泄漏后 union不稳定，shared明显退化；早期“union更优”撤回 |
| TD0-TD3 | semantic refresh、retention penalty、0/1/2 action 和 no-op | ArXiv 有局部 refresh 空间，PubMed近零；停止将其视为通用更新规则 |
| Q0/Q0-R | synthetic centroid 与 quotient 对照 | 合成表示有局部静态正信号，可作初始状态；拓扑不稳定，详见 Q0/Q0R 决策 |
| O2、trajectory oracle | X/Z/Q bilevel、后续多步 trajectory loss | 独立 student 重训仍退化；这是新 synthetic 设计必须解释的直接反证 |
| E1/E2、BER/RBCM | local block、关系 evidence、receiver-conditioned interface | 某些 full-history 局部信息有效，但受限状态重建和当前训练接口无法兑现 |
| deployment quotient/query carrier | 压缩图查询/邻域闭合 | fidelity 与任务不一致；不是经典训练凝聚必须额外解决的前置要求 |
| DANCE 式文本 evidence | root + 邻居片段、真实/打乱对应 | paired 比 shuffle好，但增加 evidence 后可能低于 root-only；文本选择和消费接口要区分 |
| GraphCLIP E0/G0/O1/C0/G1/rate | paired/shuffle、influence、atomic替换、多尺度、比例扫描 | graph-text 对应可被使用；现有 root/ego selector与unit没有稳定改进，压缩率曲线非单调 |
| 最新文本 ABC、H/I/L/C | D2 text归因、cross-moment、full-history concat/adapter/fusion/router | 文本包含信息；GraphSAGE 的消费机制跨数据集不稳定，不能将 D2 增益归因于 TAG 文本 |

### 6.1 必须读懂的 O2 与后续 trajectory 失败

O2：ArXiv gate5k/PubMed internal t2、2% roots、固定 evidence-induced adjacency、两个初始化、3 inner steps、20 outer steps，优化 X/XZ/XZQ；最终用独立 GraphSAGE 重新完整训练。ArXiv X-only 41.70% -> 29.10%，PubMed XZQ 85.96% -> 77.09%。

后续 trajectory oracle 已增加：两个初始化、step 0..4 的平均轨迹 loss、controller/selection 分离、step 0 no-op、8 outer steps。结果：

| 数据集 | Frozen Q0 | Optimize X | Optimize XZ |
|---|---:|---:|---:|
| ArXiv | 41.699% | 40.321% | 41.699%，选中 no-op |
| PubMed | 85.963% | 79.441% | 78.079% |

所以“改成多初始化、多 checkpoint 就能解决”没有证据。该实验是短轨迹各状态上的 controller loss，并非充分验证了所有专家参数轨迹匹配算法；不能由它断言经典凝聚不可行，也不能假装 trajectory 从未尝试。

新版本需要说明具体改变了哪个目标、训练分布或优化器路径。student 在 synthetic 上训练、在真实图上评测，以及重新初始化，都是正常凝聚评测；这些差异本身不自动构成实现错误。

### 6.2 GraphCLIP 与最近文本结果

GraphCLIP G0：paired-shuffled Macro-F1 为 ArXiv +12.25 pp、PubMed +8.20 pp，五个凝聚 seeds 同向。它验证接口的配对敏感性。部分 retrieval 指标接近饱和，不能仅用近 100% R@1 判断训练成功。

多尺度 unit 相对 atomic 为 ArXiv -3.261 pp、PubMed +0.265 pp。1%/不同 root 比例下性能非单调；两 epoch full-history与小 atomic 接近。训练收敛、负例池、ego overlap、训练步数与任务定义尚须核查。

最新 GraphSAGE 文本接口相对各自 graph-only 对照：

- full-history直接拼接：-0.19/+0.33 pp；
- train-only adapter早期融合：-0.34/+4.39 pp；
- 全局 late fusion：-0.25/+2.26 pp；
- 条件路由：-0.07/+2.27 pp，但 PubMed old-stage -1.98 pp。

这说明文本有数据集条件的价值，不支持“GraphSAGE原则上无法使用文本”或“GraphCLIP必然解决问题”。

## 7. 测试隔离与报告可信度

这是此前总报告的补正，不改写旧结果。

- `FINAL_FROZEN_T3_REPORT.md` 已报告早期 arXiv 全部 56,447 个 t3 节点结果；长期记忆也记录了该测试。因此整个项目的 t3 不能笼统称“从未打开”。D2/A2 等后续特定运行未读取 t3，是更窄的事实。
- 新对话应核对阶段和 node-ID fingerprint。若新 t3 与已看过的集合相同，只能报告为已暴露的历史评测；严格 final test 需另有未使用的数据/时间 holdout。
- 反复根据 qualification 的失败结果设计新方法，会产生研究层面的适应性选择偏差，即使每个脚本都未直接用 qualification 求梯度。新静态开发应安排内部开发/确认划分，并保留真正未用的最终测试。
- Old-stage retention 应新建独立历史留出集；不能继续用可见训练节点保持率替代泛化保持率。
- “Full-history oracle”在本项目往往只是某一种算法的宽松访问对照，不是数学上的性能上界。有限 action pool、短训练 oracle 失败只能否定该设置的改进空间。
- PubMed temporal 划分、年份来源、缺失年份、原始文本和 embedding fallback 均需按数据构造脚本核验；不能把所有阶段都默认视为天然真实时间事件。

## 8. 最近新路线：哪些已定，哪些待审

已确定的是经典持续 TAG 凝聚方向及旧成果冻结。9 月 4 日草案提出：

\[
U_t=\widetilde G_{t-1}\cup\Delta G_t,
\qquad
\widetilde G_t=\operatorname{Condense}(U_t;K_t).
\]

候选机制包括专家轨迹匹配、synthetic文本、可学习邻接和前后阶段 synthetic transport。它们尚未实现为经过验证的新方法。

在采用草案前需解决：

1. **先有可靠静态基线。** 现有 TAGSAM 复现尚不稳定；应该先核验一个公开经典凝聚实现，再谈新 optimizer。此前本地 TAGSAM `dataset.py` 主要优化 node feature及学习率，text embedding冻结，邻接为固定索引，不等于已经实现 X/Y/A/T 联合优化。
2. **Contrastive训练不是简单可加。** InfoNCE负例分母随集合变化，旧 gradient sketch + 新 gradient 一般不等于并集 gradient；若继续使用递归梯度，必须定义相同负例参考池或清楚标出近似。
3. **Synthetic/real union需要接口定义。** 旧合成节点如何与当前真实节点连接？跨阶段边若依赖 mapping，其来源、成本和可访问性是什么？Disjoint union、边界映射和完整原图是不同协议。
4. **Synthetic标签服从监督协议。** 若选择 label-free GraphCLIP，起步就做 X+Y 与目标不一致；应使用图文配对目标，将监督Y实验单列。
5. **Transport不是已证实创新。** 必须定义质量归一化、unbalanced mass、新类与旧类处理。强迫所有旧 feature/文本/邻接保持不变可能阻止有益重凝聚；先看误差是否真实累积，再引入运输约束。
6. **文本接口不能重复编码。** 草案写 T~=E_text(P) 后又写 E_text(T~)，应在实现前统一 token、embedding和projection的符号；soft prompt可导也不意味着其必然处于自然语言流形。
7. **冻结不等于不可改进。** 旧失败约束具体实验；新的目标和有证据支撑的修改可以做，但不得把已有短轨迹换名复跑。

## 9. 下一对话建议如何推进

接手后的首个交付物应是**静态凝聚起点与最小实验协议**，然后才进入正式方法实现。

1. 阅读本报告和 O2/trajectory 原始代码，核对静态 TAGSAM 复现；选择一个有公开实现的经典凝聚基线，记录论文、代码版本、可优化变量和原始指标。
2. 固定一套任务：GraphCLIP未来图文检索/分类优先作为候选；在相同步数和各自收敛条件下建立 full-real、random-small、经典synthetic三方对照。必要时复用已设计的 N0，但不声称已经运行。
3. 对新静态凝聚 C0，只增加最必要变量。先验证普通 student独立初始化训练仍有效、outer loss改善与确认集收益一致；仅开发阶段调参。Text、adjacency、transport逐项加入。
4. 静态成立后，用“旧synthetic + 当前增量”对比 full-history、old synthetic only、增量 only、Append独立阶段synthetic，判定历史训练信息损失、重凝聚损失和遗忘分别有多大。
5. 根据第4步的具体缺口选择一项持续机制。只有证据显示跨模态状态或跨阶段对应是瓶颈，才发展相应新算法。
6. 完整方法冻结后，在多阶段和独立最终holdout验证，再扩大数据集及迁移模型。

旧草案 C0-C4 可以作为阶段标签，但尚无新正式结果。C0与旧 GraphCLIP C0 重名，文件及 protocol 必须使用完整前缀，例如 `classical_tag_static_c0_v1`，避免再次混淆。

固定压缩率例子：`K_t = round(rho * N_seen_train,t)`；分母必须预注册，不能交替使用全部可见节点、训练节点或 ego root 数。固定rho时 K_t一般随历史增长；固定总K是另一条 memory-cap协议。真实文字、embedding、边、label及必要映射另报字节，blank padding不算有效payload。

判断应报告均值、方差/区间、配对种子差值、当前及历史指标。0.5 pp/4-of-5可作为预设研发阈值，但不是创新的定义，也不因单个初步seed失败自动否定整个选题。

## 10. 相关工作与基线边界

旧文献报告包含 CaT/PUMA、GECC、TAGSAM、DANCE、GraphCLIP、TACO、OpenGC、MCond 等。重点重新阅读：

- CaT/PUMA：持续凝聚、Training in Memory、edge-free replay；已覆盖持续凝聚的大类。
- GECC：evolving condensation、centroid inheritance；已覆盖演化聚类压缩。
- TAGSAM：静态TAG文本选择与匹配；是TAG方向直接近邻，需处理论文/代码分支差异。
- DANCE：旧调研记录为联邦/动态TAG evidence机制；集中式持续协议是否构成差异需再核验原文。
- GraphCLIP：任务模型；使用它本身不能算凝聚算法创新。
- 经典合成图优化路线：在新对话中补查 GCond、DosCond、SFGC、GEOM 等适用公开实现和演化扩展，选一个可验证起点。

本交接没有新做网络检索。旧报告里的 arXiv链接与机制可作检索入口，不能用“此前未找到完全同样组合”替代投稿前的新颖性核验。`PUMA-style/GECC-style/DANCE-style` 是本地适配对照，不是已经完整复现原文。

## 11. 可复用代码和原始结果入口

以下均相对于项目根目录，本轮已确认主要代码文件存在。

| 文件/目录 | 用途 |
|---|---|
| `distill.py`, `dataset.py`, `model.py`, `selection.py` | TAGSAM局部代码与合成训练接口；已有修改，不能假定等于官方master |
| `role_aware_continual_d2.py` | D2状态merge、sigma重构、coverage与账本 |
| `run_strict_d2_graphsage_gate.py` | A2 GraphSAGE训练与评测 |
| `run_o2_full_set_bilevel_synthetic_oracle.py` | O2可微inner-loop与失败目标 |
| `run_trajectory_level_synthetic_oracle.py` | 已失败短轨迹版本，接手必须阅读 |
| `run_graphclip_rate_curve.py` | GraphCLIP比例评测，核查step/epoch与ego重复 |
| `run_graphclip_multiscale_oracle.py` | 多尺度unit失败对照 |
| `problem_validation_results/strict_stage_local_a2/arxiv_a2.json` | A2 ArXiv原始结果 |
| `problem_validation_results/strict_stage_local_a2/pubmed_a2.json` | A2 PubMed原始结果 |
| `problem_validation_results/replay_tag_ideal_evidence/` | Q0/O2/trajectory等结果，按protocol区分 |
| `FROZEN_BASELINE_REGISTRY_2026-08-27.md` | 旧协议reuse条件与基线哈希 |

新实验建议目录 `problem_validation_results/classical_continual_tag_condensation/` 在本轮本地查询中未发现。没有据此断言远程也不存在；新对话启动前应检查远程。

## 12. 本次交接实际完成了什么

- 核验最新方向草案、D2/A2、O2、后续trajectory、早期t3以及长期记忆。
- 校验冻结快照列出的九份报告，SHA-256全部与登记一致。
- 创建本交接报告和可供新对话使用的启动提示，并补充长期记忆的最新方向。
- 没有改写冻结报告、运行新实验、打开新测试集或改变服务器进程。

“冻结快照”是带哈希的文档清单，不是整个数据仓库的只读备份。大artifact完整性、远程文件是否存在和环境版本由新对话按需要核验。

## 13. 新对话启动提示

> 请先阅读 `D:/Adocs/简历/research/graph-condensation/TAGSAM/NEW_CONVERSATION_HANDOFF_CLASSICAL_CONTINUAL_TAG_ZH_2026-09-05.md`，然后在同一项目继续研究。我的目标是经典持续TAG图凝聚：优化小型synthetic graph，使其训练效果接近累计真实TAG，并体现文本与持续机制，目标是有算法贡献的CCF A长论文。D2/A2及旧实验已冻结，只作基线/初始化，不能自动回到prototype buffer主线。先核查旧O2与短轨迹synthetic失败、静态TAGSAM复现和真实经典凝聚基线，明确主任务、监督范围、压缩率分母、文本产物与独立holdout，再形成并实现最小静态可行性实验。请特别留意：多步trajectory已经试过；A2的+4.35 pp不是超过PubMed所有基线；旧arXiv t3已经使用；新C0/N0尚无可报告结果。避免只换名字或同时堆transport、prompt、learned adjacency等模块。
