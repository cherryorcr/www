**经典持续 TAG 图凝聚：当前选题、方法雏形与推进计划**

整理日期：2026-09-05。依据：最新交接、历史实验报告、关键原始结果及部分实现。本文区分已验证结论、候选解释和建议实验；未运行新的凝聚实验，未产生新的 C0/N0 结果。旧报告的方向决策以 9 月 5 日交接修正为准。本轮复核的九份冻结报告 SHA-256 均与快照一致。

**1．当前课题应该怎样表述**

建议工作题目：**面向演化文本属性图的持续图凝聚**（Continual Graph Condensation for Evolving Text-Attributed Graphs）。

研究目标是：当带文本的图随时间增长时，在明确的压缩预算内持续维护一张小型合成图，使模型仅用这张合成图训练后，尽量接近使用累计真实图训练的任务效果，并保持对历史阶段的泛化能力。

核心产物为可训练的合成数据：

\[
\widetilde G_t=(\widetilde X_t,\widetilde A_t,\mathcal S_t,\widetilde Y_t),
\qquad K_t=\operatorname{round}(\rho N_{\mathrm{seen,train},t}).
\]

其中，\(\mathcal S_t\) 是实际保存和使用的可读文本片段及其对应关系；\(\widetilde Y_t\) 仅在监督协议允许时存在。文本连续状态、样本权重等可以按需增加。上述压缩率分母是本文建议的新协议，不能用于重解释旧实验的 root rate。

目前已经确定的是三个方向要求：**合成数据直接服务训练、文本在算法中有实际作用、合成图能够随阶段更新**。尚未确定的是具体主模型、训练匹配目标、文本优化方式和持续约束。D2 是已有基线或可选初始化；双角色、双专家和固定融合不属于新方法的必选结构。[S1]

课题的中心问题可以进一步收窄为：

> 在只能使用上一阶段压缩状态和当前增量的条件下，如何让重新凝聚后的 synthetic TAG 继续保留历史与当前数据对图文模型训练的作用？

这里需要同时解决“单次压缩能否保持训练效果”和“反复压缩是否积累信息损失”。持续训练模型权重与持续更新合成数据应分别评测。经典凝聚允许用训练所得模型在真实评测图上推理；完全依赖压缩图回答所有原图查询，是额外研究目标，暂不作为前置要求。

**2．历史工作留下了哪些可以使用的结论**

历史路线大致经历了：静态 TAGSAM 复现与时序协议建立 → prototype/coreset、边和文本接口探索 → D2/A2 正向记忆结果 → 文本、集合优化与持续机制验证 → 冻结旧分支并回到经典合成图凝聚。

| 实验线 | 已有证据 | 对新主线的含义 |
|---|---|---|
| 静态 TAGSAM 复现 | Teacher 曾接近论文；凝聚训练不稳定，存在步长、feature 范数及代码分支差异 | 还缺一个可以放心扩展的静态经典凝聚起点 |
| P0–P4、K2/K3、强 coreset | joint selection、完整文本及某些真实边对应有局部价值；固定 residual、几何/novelty utility 不稳定胜过强在线对照 | 保留公平基线和接口经验；不能继续将手工 proxy 改善当作训练收益 |
| V5–V9、RAM、message、STPM/actions | 部分统计或短程 reward 改善，独立完整训练却不改善；单点价值与集合价值不等价 | 新凝聚目标必须用独立 student 重训验证，不能仅看 outer loss |
| D2/A2 | 递归 moment 与 coverage 状态在既定下游协议下有正结果 | 压缩历史训练数据可行；可以复用为基线或初始化 |
| A3 | 修复 checkpoint 选择后，早期 union 更优的结论撤回；shared objective 两数据集均退化 | 固定融合不能简单换成共享 loss 就认定升级成功；也不能推断所有单模型方法均失败 |
| TD、BER/RBCM、局部 evidence | 更新和文本证据有局部空间，受限状态与消费接口难以稳定兑现 | 先区分信息是否存在、是否保留下来、模型是否用得上 |
| O2 与后续短轨迹 oracle | 自由 X/Z/Q 优化及多步版本均未形成独立重训收益 | 新静态方法必须实质改变目标或训练路径，并先核对实现 |
| GraphCLIP G0/G1/rate | 正确图文配对有效；多尺度 unit 与现有 selector 没有稳定优势，比例曲线非单调 | GraphCLIP 是有根据的候选主接口，尚不是已经验证的凝聚方案 |
| 最近文本 ABC、H/I/L/C | 文本收益依赖数据集和接口，PubMed 存在正信号，ArXiv 多数不改善 | 不能把 D2 增益归因于文本，也不能声称 GraphSAGE 原则上无法使用文本 |

这批实验最有价值的共同结论是：**压缩状态可递推、图文对应可被模型使用，都只是必要环节；最终仍需证明压缩数据能训练出更好的独立模型。** [S2–S10]

**3．最重要的数值与解释边界**

下表各行属于各自协议，不能纵向拼成统一排行榜。单位为 Macro-F1 百分数或百分点（pp）。

| 结果 | ArXiv | PubMed | 应怎样解释 |
|---|---:|---:|---|
| D2 原始 probe | 47.58%，对预选基线 +1.21 pp | 83.78%，对预选基线 +1.25 pp | 既定角色记忆 probe 的正结果 |
| A2 真实 SAGEConv，Continual | 46.98% | 78.09% | self-loop 合成训练拓扑、两个半宽专家的固定融合 |
| A2 对 controller 预选基线 | 对 Append +3.50 pp | 对 PUMA-style +4.35 pp | 原先的正式研发判据 |
| A2 对同表最高非 D2 单角色基线 | 对 PUMA-style 44.08%，+2.90 pp | 对 Append 77.80%，+0.29 pp | 防止把预选基线优势夸大为超过所有基线的优势 |
| GraphCLIP G0：paired − shuffled | +12.25 pp | +8.20 pp | 5 个凝聚种子；图文配对敏感性，不能充当新凝聚算法增益 |
| GraphCLIP G1：multiscale − atomic | −3.261 pp | +0.265 pp | 2 个凝聚种子；当前多尺度 unit 没有稳定跨数据集收益 |
| 后续短轨迹：frozen Q0 → X | 41.699% → 40.321% | 85.963% → 79.441% | 多步短轨迹优化已经尝试，仍失败 |
| 后续短轨迹：frozen Q0 → XZ | 41.699% → 41.699%，选中 no-op | 85.963% → 78.079% | ArXiv 持平来自不更新，不能写成优化成功 |

A2 五个 seeds 是模型种子 100–104，使用冻结确定性 memory，不是五次独立凝聚。A2 的 PubMed Accuracy 为 77.30%，略低于 Append 的 77.49%；对表内最强基线的 +0.29 pp 也不能自动解释为显著优势。旧 old-stage 指标主要来自历史可见节点，属于保持率诊断。[S3–S5]

资源上，D2 的 ArXiv/PubMed 持久字节为 3,088,375/399,605，对应上限为 4,648,093/676,030。准确说法是相同名义 root 数、同一字节上限内，实际有效字节并不逐项相等。[S4]

最近 GraphSAGE 文本实验相对各自 graph-only 对照：直接拼接 −0.19/+0.33 pp；train-only adapter −0.34/+4.39 pp；late fusion −0.25/+2.26 pp；条件路由 −0.07/+2.27 pp，且后者 PubMed old-stage −1.98 pp。这些值属于不同接口诊断，不能替换 A2 主结果。[S1, S10]

**4．O2 与短轨迹失败，具体约束了什么**

O2 使用两个初始化、3 步 inner-loop、20 步 outer-loop，在固定 evidence-induced adjacency 上优化 X/XZ/XZQ。最终独立 GraphSAGE 重训时，ArXiv X-only 从 41.70% 降至 29.10%，PubMed XZQ 从 85.96% 降至 77.09%。后来已经加入两个初始化、step 0–4 平均 loss、controller/selection 分离及 step 0 no-op，结果仍未改善。[S6, S7]

代码审查表明，后者的“trajectory”实际是：在合成数据上展开短程手工 SGD，在这些模型状态上计算真实图 controller CE 并取平均。它没有直接记录真实图专家的参数轨迹并匹配参数位移；独立评估又使用正常完整训练。因此，**这些结果否定了已测的短轨迹代理目标，尚未充分检验所有经典专家轨迹匹配算法**。同样，独立初始化、在小图上训练并在真实图上评估本来就是正常凝聚测试，不能把这种评估本身称为错误。[S7, S12]

本轮还有一个新的只读实现发现：O2 用 `Parameter(graph_initial)` 和 `Parameter(text_initial)` 创建优化变量，再用同一 initial tensor 做 anchor MSE，存在共享存储导致参考点随参数更新的问题。已保存的 10 个优化 arm、每个 20 步的 graph/text anchor 值均为 0，与此判断一致。后续 trajectory 代码已使用 `.clone()`，所以这一问题不能解释后续失败。它应成为新静态实验前的核验项；本文没有修复或重跑，不能声称“找到失败主因”或“修好就会成功”。[S12]

新版本与旧失败实验的对照清单应至少回答：

- 目标到底是 controller CE、梯度、参数位移还是函数输出匹配？目标为什么更接近最终训练用途？
- 训练状态如何采样，内外层的 optimizer、训练长度及随机性如何对应？
- 有无固定参考的真实 expert trajectory？有无独立确认集和独立 student？
- 相比已有短轨迹版本，究竟改变了一项什么假设？

增加 checkpoint 数量、可学习邻接、transport 三项一起做，无法回答上述问题。

**5．目前可以落地的方法雏形**

建议用描述性名称“持续合成 TAG 重凝聚”组织工作，先不命名新算法。其骨架是：

\[
U_t=\operatorname{BuildUnion}(\widetilde G_{t-1},\Delta G_t),
\qquad
\widetilde G_t=\operatorname{Condense}(U_t;K_t).
\]

旧 synthetic 保存历史训练信息，当前真实增量提供新信息，凝聚器将两者重新压缩。下一阶段只接收明确序列化的状态。任何额外 teacher、sketch、文本缓存或映射若跨阶段保留，都需说明访问规则及成本。这一骨架是待验证的研究框架，本身尚不构成原创机制。[S1, S11]

**主任务建议。** 优先核验 GraphCLIP 图文对齐及非饱和的未来检索接口，分类作为同一协议下的辅助指标；最终主指标在看新结果前固定。GraphSAGE 可用于监督基础验证和迁移诊断。不能在每次结果出来后切换主任务，也不要求所有 backbone 都提高。

若选择 label-free GraphCLIP，凝聚过程中不用类别标签优化 soft Y、做类别均衡选样或类别条件匹配。类别名称用于冻结的 zero-shot 评估，与训练时使用节点类别标签是两种权限。监督 GraphSAGE/监督 TAG 凝聚另立协议，数值不混用。

**静态最小版本。** 先审核一个公开经典凝聚实现，在其原始设置下得到可用基线，再只做必要的 TAG 接口适配。第一版优先优化 synthetic X，固定标签（仅监督设置）、拓扑及可读文本；只有收益稳定后才逐项放开新的变量。固定 A 或固定文本不妨碍它属于经典凝聚，但最终论文需要进一步证明文本参与方法的价值。

GCond、DosCond 可作为较直接的经典起点审核对象；SFGC、GEOM 可作为专家轨迹路线的审核对象。本文未新做联网核验，尚未确定具体论文版本、仓库 commit 和最佳起点，也不声称这些实现可直接运行 GraphCLIP。[S1]

如果采用专家参数轨迹，可用下式表达候选目标：

\[
\mathcal L_{\mathrm{match}}=
\mathbb E_{r,j}
\frac{\|\operatorname{Train}_{m}(\theta^{r,j}_{U},\widetilde G)
-\theta^{r,j+h}_{U}\|_2^2}
{\|\theta^{r,j+h}_{U}-\theta^{r,j}_{U}\|_2^2+\epsilon}.
\]

这里 \(\theta_U\) 来自真实参考数据或允许访问的 union 的训练轨迹，\(m,h\) 和训练优化器需明确注册。这只是可验证的候选实现，不是新颖性声明，也不是已经通过的目标。优先按选定公开基线的定义复现，再解释适配差异。

**可读文本的处理。** 最小版本保留来源明确、受 token/字节预算约束的真实文本片段，实际经 text encoder 进入训练。后续可研究片段选择或分配如何随重凝聚更新。文本来源锚点不要求 synthetic X 等同某个真实节点，也不要求两代节点保持固定身份。

若确有证据需要连续文本优化，分别定义 token/soft prompt P、编码向量 Z 和 projection，避免重复编码。优化 soft prompt 时，冻结 encoder 参数仍须允许输入梯度；导出后也必须可重现。Soft prompt 不等于可读文本，另需交代真实片段怎样保留、使用以及是否承担训练功能。

**union 的处理。** 第一版可以明确使用 disjoint union，并将其与原始跨阶段边缺失的影响分别记录。若要连接旧 synthetic 与新真实节点，必须定义 mapping 的来源和预算，不能隐式查回旧 raw graph。旧节点和新节点的抽样质量/权重也应固定规则，避免“旧历史只有 K 个点”导致其训练占比无意消失。

GraphCLIP 的 InfoNCE 负例分母依赖整个 batch/参考集合，因此一般有：

\[
\nabla L(\widetilde G_{t-1})+\nabla L(\Delta G_t)
\ne\nabla L(U_t).
\]

不能将旧梯度 sketch 与当前梯度直接相加后称为精确累计目标。最小实现应在允许访问的 union 上共同采样并计算 loss；如采用固定负例参考池，需明确其语义与成本。[S1]

**持续机制的候选位置。** 最值得验证的假设是：反复重凝聚会丢失历史图文对应在新训练上下文中的作用，导致保持率下降。若实验支持这一假设，再研究图文训练响应的保持目标；若主要损失来自跨阶段边或负例分布，则优先改相应接口。Transport 是其中一个可选工具，只有证明旧新对应本身存在可利用的瓶颈后才引入，并明确质量归一化、新类和非平衡质量的处理。

这比立即确定“trajectory + soft prompt + learned adjacency + transport”的总损失更容易形成可解释的算法贡献。上述中心假设仍待验证。

**6．后续实验应按什么顺序推进**

各阶段按完成条件推进，不承诺尚未核查资源情况下的运行日期。本表为建议计划，均非已执行结果。

| 阶段 | 要回答的问题 | 最小工作与对照 | 交付物与决策 |
|---|---|---|---|
| P0：协议与实现起点 | 什么数据、任务和代码可以作为可靠起点？ | 核对 TAGSAM 分支与 O2；选一项公开经典基线；建立标签权限、split fingerprint、文本接口和预算定义 | 一份静态最小实验协议、代码版本表、可复现基线说明 |
| P1：训练与压缩空间 | 当前 TAG 模型是否使用文本，压缩是否留下值得研究的差距？ | full-real、random-small、k-medoids-small；paired/shuffled/无新增文本通道；同 optimizer steps 与各自收敛两种口径 | 学习曲线、配对敏感性、full–small 差距、重复 ego 和负例审计；无空间则诊断任务或压缩率 |
| P2：静态经典凝聚 | 优化 synthetic 数据能否改善独立重训？ | 公开经典凝聚、frozen initialization、random/coreset；先只优化 X；D2/Q0 为可选初始化对照 | 新静态 C0 报告：独立模型重训、确认集、outer loss 与任务关系；失败先定位原因 |
| P3：持续损失定位 | 损失发生在历史压缩、合并还是再次凝聚？ | full-history、旧 synthetic + delta、重凝聚结果、old-only、delta-only、分阶段 synthetic Append、full-history 重新凝聚 | 逐阶段当前/历史指标，定位具体差距；此时不预设 transport 必需 |
| P4：一项核心机制 | 哪项更新机制能够稳定缩小已测缺口？ | 根据 P3 选择文本对应、负例上下文、边界连接或历史响应保持中的一项；与无该机制及打乱对照比较 | 同预算、同训练接口的独立因果收益及失败边界 |
| P5：正式验证与论文 | 方法是否有持续效果、资源价值和充分差异？ | 多个时间阶段、第三数据集、完整强基线、独立最终 holdout、主模型及有限迁移 | 主表、保持率与遗忘曲线、文本/结构消融、性能–字节–时间曲线、文献差异表 |

P0 的代码审核与 P1 的任务协议整理可以并行；P2 的方法判断依赖可靠静态基线；持续机制开发依赖 P3 的缺口证据。新实验命名使用 `classical_tag_static_c0_v1` 等完整前缀，避免与旧 GraphCLIP C0 重名。

P3 中最关键的三个比较是：

- **full-history vs 旧 synthetic + delta**：诊断历史状态替代及 union 接口带来的差距，包括旧新边和抽样权重的影响。
- **旧 synthetic + delta vs 新 synthetic**：诊断这一次重新凝聚的损失。
- **持续重凝聚 vs full-history 重新凝聚**：诊断受限历史访问与反复压缩带来的整体差距。

这些是匹配条件下的对照关系，不应把非线性的 F1 差值当成严格可加的误差分解。Full-history 是宽松访问参考；某个 full-history 算法的分数不是数学上界。联合数据训练很差时也不能断言所有压缩器必然失败，只能说明当前参考目标不可靠。

**7．新协议必须提前固定的事项**

| 项目 | 建议定义 |
|---|---|
| 数据演化 | 第一阶段聚焦节点/边到达的 TAG；静态文本随新节点到达不等于旧文本内容发生漂移；旧节点文本更新可后续另测 |
| 历史访问 | 只读显式保留状态和当前增量；历史样本若获准保留，统一计入预算，不能作为隐藏 replay |
| 压缩量 | 主报固定 rho，另报实际 K、边、文本、embedding、labels、mapping、state bytes；固定总 K 是另一协议 |
| 文本权限 | 列清训练与推理可见的文本及来源；若 X 本身已由文本编码，zero text 只能称移除新增文本通道，不能称完全去文本语义 |
| 图可见性 | 固定训练、checkpoint、未来推理可见的节点/边与文本；尤其说明 transductive 或 inductive 条件 |
| student 初始化 | 每个 artifact 用独立 student 训练；预训练模型从同一冻结 base 重启并改变随机种子，与从零训练单独标记 |
| 模型持续状态 | 先用 fresh student 隔离合成数据质量，再做 warm-start 模型的持续学习；保存的模型状态按协议单独核算 |
| 内外层选择 | outer training、内部开发、确认集、最终测试职责分开；确认集不求梯度，也不因反复反馈自动保持独立 |
| 种子和统计 | screening 可先 2 个凝聚种子；正式扩至至少 5 个凝聚种子并配置模型种子；报告配对差值和区间，避免把同一 artifact 多个模型当独立凝聚样本 |
| 计算口径 | 记录凝聚成本、teacher 生成成本、下游训练成本；同时报告同预算和各自合理收敛条件 |

测试隔离尤其需要重建：早期 ArXiv 全部 56,447 个 t3 节点已经评测。旧集合可以作为已暴露的历史 benchmark，但不能重新标为密封最终测试。须核对两数据集所有阶段的 node-ID 与结果暴露记录；仅重新命名或重切已用节点不恢复独立性。旧阶段保持率也应使用未参与凝聚训练的历史留出节点。[S1, S13]

PubMed 构造代码支持按 NCBI ESummary 提取发表年份，并对少量缺失年份用中位数填补，默认最低覆盖要求 99.5%；这只是代码设计。实际运行的年份覆盖、缺失数、排序/同年切分、文本与节点映射和 embedding fallback 仍须对照 artifact 验证，不能仅因文件名含 temporal 就认定所有阶段都是自然时间边界。[S14]

**8．判断成功、失败和创新的方式**

第一步的成功是“经典凝聚基础实现可靠，并且 synthetic 优化在独立训练和确认集上有收益”。它不等于已经完成持续 TAG 算法。后续要逐步补齐：文本机制因果贡献、持续机制因果贡献、预算内强基线优势和最终泛化。

| 观察到的结果 | 合理判断与下一步 |
|---|---|
| full 与 small 接近，学习曲线未收敛或 retrieval 饱和 | 先核对训练步数、重复样本、负例与评测难度；有依据地调整接口或比例 |
| outer loss 下降、独立 student 退化 | 检查参考目标、实现及训练状态覆盖；在开发集诊断，暂停当前目标的扩大实验 |
| union 有效，重凝聚后明显下降 | 优先改本次凝聚器，不把问题全部归为缺历史数据 |
| union 已明显丢失历史性能 | 检查旧 synthetic 信息、采样质量和边界接口；单纯多优化 outer steps 缺乏依据 |
| 静态有效，多阶段逐步退化 | 持续误差确有证据，可研究一项保持机制 |
| paired 优于 shuffle，却低于不加该文本组件 | 只能说明错配有害，不能证明新增文本机制值得保留 |
| 文本学习无增益、固定真实文本有效 | 保留固定文本并研究其训练作用，无需为形式完整而优化文本 |
| 某个数据集失败 | 先判断失效机制和不确定性，不能单个 seed 后立刻换题，也不能宣称通用胜出 |

0.5 pp、4/5 同向或约 2 pp 的 full–small 差距，可以预设为研发资源门槛，须结合方差和任务价值解释；它们不是统计定理或创新定义。最终应报告均值、标准差/区间、配对凝聚种子差值，以及当前和历史任务是否相互牺牲。

“持续 + TAG + trajectory + transport”这一组合本身不足以支持 CCF A 长论文。更有力的贡献叙述应是：**识别持续 TAG 重凝聚中的一项具体信息损失，提出直接处理该损失的算法，并证明它比已有方法在相同预算下更好地保持训练效果。** 不需要人为凑够若干模块，也不能仅凭系统实现和复杂损失认定创新成立。

**9．投稿前必须补齐的相关工作核验**

本轮依据的是项目已有文献整理，没有进行新一轮联网检索。下表是核验清单，不是已完成的新颖性结论。

| 近邻 | 需要核对的重合点 |
|---|---|
| CaT/PUMA | 持续 condensed replay、训练于 memory、分布匹配；区分原论文与本地 PUMA-style 适配 |
| GECC | 演化凝聚、centroid inheritance、状态更新；区分原算法与本地 coverage 对照 |
| CCC | 旧综述列为 condensation-concatenation；优先核验与“分阶段凝聚/拼接/再凝聚”的重合 |
| TAGSAM | 静态 TAG 凝聚、文本片段选择、attribute matching；核对论文、released master 和 topology 分支 |
| DANCE | 文本 evidence、动态更新及联邦/集中式条件；需读原文后确认差异 |
| TACO、OpenGC、MCond | 持续 coarsening、开放世界偏移、原图到合成图映射，避免将邻近问题误作空白 |
| GCond、DosCond、SFGC、GEOM | 梯度/轨迹路线、可优化变量和开源复现；同时查演化或持续扩展 |
| GraphCLIP | 是目标模型与任务接口；使用它不构成凝聚算法创新 |

需要交付“已有工作已经做了什么 → 本课题具体缺口 → 新机制改变什么 → 哪个实验能排除已有解释”的四列表。旧文献中“未找到完全相同组合”不能替代此项工作。[S2, S15]

**10．近期最明确的交付物**

接下来优先形成一份可运行的**静态凝聚起点与最小实验协议**，具体包括：公开基线的论文/commit/原始指标，旧代码审计结论，唯一主任务及标签权限，数据与 holdout 指纹，压缩率分母，可读文本如何进入训练，以及 full-real/random-small/classic-synthetic 的公平对照。

随后才运行任务空间与新静态凝聚实验。静态可行后，用 P3 的对照定位持续缺口，再决定算法机制。当前不启动新的正式全量实验、不预先承诺 transport、不默认返回 D2 主线。本轮文档整理没有改变上述实验状态。

可用于和导师交流的表述：

> 我目前聚焦演化文本属性图上的持续图凝聚：维护一张可训练的小型 synthetic TAG，使其随新数据到达更新，并尽量保持累计真实图的训练效果。已有 D2/A2 实验证明递归压缩训练记忆有价值，GraphCLIP 也证明正确图文配对会影响任务，但经典合成图优化尚未跑通，已有 O2 和短轨迹目标均未带来可靠独立重训收益。下一步先建立可靠静态经典凝聚基线和文本任务接口，再拆开测量历史状态替代、union 接口和再次凝聚各自的损失，根据真实瓶颈发展一项持续图文机制。预期贡献来自对这一瓶颈的算法解决及多阶段证据，而不是继续叠加 prototype、融合或正则模块。

**资料索引**

- [S1：最新交接与补正](D:/Adocs/简历/research/graph-condensation/TAGSAM/NEW_CONVERSATION_HANDOFF_CLASSICAL_CONTINUAL_TAG_ZH_2026-09-05.md)
- [S2：完整历史研究报告，方向与测试状态须服从 S1 补正](D:/Adocs/简历/research/graph-condensation/TAGSAM/CONTINUAL_TAG_CONDENSATION_FULL_RESEARCH_REPORT_ZH_2026-09-04.md)
- [S3：D2 正式结果与限制](D:/Adocs/简历/research/graph-condensation/TAGSAM/D2_ROLE_AWARE_CONTINUAL_UPDATE_REPORT_2026-08-31.md)
- [S4：A2 GraphSAGE 协议与完整结果表](D:/Adocs/简历/research/graph-condensation/TAGSAM/A2_GRAPHSAGE_FORMAL_GATE_REPORT_2026-08-31.md:40)
- [S5：A3 checkpoint 选择校正](D:/Adocs/简历/research/graph-condensation/TAGSAM/A3_CHECKPOINT_SELECTION_CORRECTION_REPORT_2026-08-31.md)
- [S6：O2 决策](D:/Adocs/简历/research/graph-condensation/TAGSAM/O2_FULL_SET_BILEVEL_SYNTHETIC_ORACLE_DECISION_ZH_2026-09-01.md)
- [S7：后续短轨迹决策](D:/Adocs/简历/research/graph-condensation/TAGSAM/TRAJECTORY_LEVEL_SYNTHETIC_ORACLE_DECISION_ZH_2026-09-01.md)
- [S8：GraphCLIP G0/G1 审计](D:/Adocs/简历/research/graph-condensation/TAGSAM/GRAPHCLIP_G0_G1_GATE_AUDIT_2026-09-03.md)
- [S9：GraphCLIP 比例曲线，旧方向决定已被覆盖](D:/Adocs/简历/research/graph-condensation/TAGSAM/GRAPHCLIP_RATE_CURVE_DECISION_ZH_2026-09-03.md)
- [S10a：固定 D2 文本及 cross-moment](D:/Adocs/简历/research/graph-condensation/TAGSAM/D2_TAG_TEXT_CROSS_MOMENT_ABC_DECISION_ZH_2026-09-03.md)
- [S10b：full-history 条件路由](D:/Adocs/简历/research/graph-condensation/TAGSAM/FULL_HISTORY_TEXT_CONDITIONAL_GATE_DECISION_ZH_2026-09-04.md)
- [S11：经典凝聚路线草案，须结合 S1 修正](D:/Adocs/简历/research/graph-condensation/TAGSAM/CLASSICAL_CONTINUAL_TAG_GRAPH_CONDENSATION_ROADMAP_ZH_2026-09-04.md)
- [S12a：O2 参数与 anchor 实现](D:/Adocs/简历/research/graph-condensation/TAGSAM/run_o2_full_set_bilevel_synthetic_oracle.py:120)
- [S12b：O2 结果日志](D:/Adocs/简历/research/graph-condensation/TAGSAM/problem_validation_results/replay_tag_ideal_evidence/o2_bilevel_synthetic_screening_seed100.json)
- [S12c：短轨迹实现，已 clone anchor](D:/Adocs/简历/research/graph-condensation/TAGSAM/run_trajectory_level_synthetic_oracle.py:202)
- [S13：早期已经评测的 ArXiv t3](D:/Adocs/简历/research/graph-condensation/TAGSAM/FINAL_FROZEN_T3_REPORT.md:3)
- [S14：PubMed 年份和文本构造](D:/Adocs/简历/research/graph-condensation/TAGSAM/prepare_taglas_pubmed_temporal.py:121)
- [S15：相关工作与实验复盘](D:/Adocs/简历/research/graph-condensation/TAGSAM/RESEARCH_DIRECTION_LITERATURE_AND_EXPERIMENT_PLAN_ZH_2026-09-02.md)
- [冻结快照与九份报告哈希](D:/Adocs/简历/research/graph-condensation/TAGSAM/FROZEN_RESEARCH_SNAPSHOT_BEFORE_CLASSICAL_RESET_2026-09-04.md)
- [TAGSAM 静态复现历史记忆](D:/Aprojects/AI-Memory/Projects/TAGSAM-Baseline-Reproduction.md)
