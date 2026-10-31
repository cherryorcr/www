# GraphCLIP Continual v4：优化方案与最终凝聚方法设计

> 2026-09-17 状态说明：这是目标设计及历史分阶段计划，不是所有组件都已实现或进入默认 v4。P2 已实验收尾但未晋升新文本机制；当前配置、编号和后续优先级见 [v4 状态与后续优化路线](V4_STATUS_AND_NEXT_OPTIMIZATION_ROADMAP_ZH_2026-09-17.md)。下文“当前阶段”“后续 P2-P5”等保留为当时记录。

日期：2026-09-15  
当前基础：unified v3 + synthetic-only Student-Compatible Append

当前实现状态：v4 主线核心已实现为 `--v4-mainline` preset，默认使用 fixed-rate、dispersion、独立 selection seeds、LCB 和 worst-horizon response gate；reserve extra budget 已实现但默认关闭。文本压缩升级、复杂度 v2、历史 memory manager 和大图多阶段验证仍属于后续 P2-P5。

## 1. 当前结论和方法定位

当前最稳妥的稳定基线是：

> fixed-rate + matched-quota + dispersion allocation + frozen-old topology + synthetic-only append。

首轮小图结果：

| 方法 | Cora | Citeseer | 判定 |
|---|---:|---:|---|
| fixed-rate + dispersion | 49.85 | 59.17 | 当前稳定基线 |
| subcluster complexity | 49.38 | 53.44 | 复杂度先验尚未证明有效 |
| response marginal | 52.40 | 57.29 | Cora 有效，跨数据集不稳定 |
| complexity + response marginal | 48.64 | 55.29 | 复杂度先验没有帮助 selector |
| response-gain extra | 46.00 | 53.43 | 产生 canonical regret |

这意味着：复杂度模块和 response 模块不能直接全部合并。正确路线是：

> 复杂度模块负责产生结构候选，response selector 负责在固定预算内校准，额外预算默认关闭，只有经过风险控制和 full-step confirmation 才允许启用。

本文中的“最终方法”是优化完成后的目标规范，不把尚未通过门禁的组件写成已证实优势。

## 2. 两个负组件的优化

### 2.1 类内多子簇复杂度分配 v2

首轮没有超过 dispersion，原因是子簇数量过于启发式，复杂度不等于 student 边际收益，没有考虑历史覆盖缺口和冗余，也没有经过 response 校准。

v2 将复杂度定义为候选生成先验，而不是最终裁决器。对每个 joint graph-text 子簇 \(g\) 计算：

\[
w_g =
\frac{\sqrt{m_g}(1+\lambda_d d_g)(1+\lambda_m mlt_g)(1+\lambda_u u_g)}
{1+\lambda_r r_g}
\]

含义：

- \(m_g\)：source mass；
- \(d_g\)：joint dispersion；
- \(mlt_g\)：多模态程度；
- \(u_g\)：相对历史 memory 的 coverage gap 或 novelty；
- \(r_g\)：与旧 synthetic nodes 的 redundancy。

改进点：

1. 动态产生 \(q_{min},q_{mid},q_{max}\) 三种子簇数候选；
2. 用 cluster stability、silhouette、子簇半径和图文局部模式判断是否确实需要拆分；
3. 子簇不稳定时回退到少量子簇，不能强行切分；
4. 用带下限和上限的 largest-remainder 在固定总预算 \(B_t\) 内分配；
5. 复杂度只决定初始 rows 和 split/merge candidates；
6. 最终 rows 由 response marginal 在固定总预算内微调；
7. 明确禁止真实节点作为最终凝聚图节点，所有输出仍是 synthetic nodes。

最终形式：

\[
b_g^{prior}=Allocate(B_t,w_g,b_{min},b_{max}),\qquad
b_g^{final}=b_g^{prior}+\delta_g,\quad\sum_g\delta_g=0
\]

验收：Cora/Citeseer 的 1%、5%、10% matched-quota，独立 selection seeds，一个额外类别顺序或 base+4，并报告 rows、coverage、redundancy、F1 和训练成本。只有跨 rate 和阶段不劣于 dispersion，才升级为默认 allocation。

### 2.2 Response marginal v2

首轮 Cora 提升、Citeseer 下降，说明 response 有信息但 ranking 与 canonical student 不一致。v2 在固定总 rows 内搜索 transfer：

\[
\delta^*=\arg\max_{g_i,g_j}
\widehat J(b+e_i-e_j)-\widehat J(b)
\]

改进：

- 同时评估转移 1、2、4 行；
- 使用 0.5x、1x、2x 等多个 response horizons；
- selection seeds 与 canonical seeds 完全分离；
- 加入 old retention、new plasticity、coverage、cost 和 uncertainty；
- 只接受多 seed、多 horizon 下置信下界为正的 transfer；
- 连续两轮无稳定收益就停止。

建议目标：

\[
J=\Delta F1_{new}+\lambda_o\Delta Retention_{old}
+\lambda_c\Delta Coverage-\mu Cost-\nu Uncertainty
\]

### 2.3 Response-gain extra budget v2

首轮最多一次增加 8 rows，selector-dev 的正 gain 没有预测 canonical F1，因此 v2 必须保守：

1. 保留 reserve pool，候选为 0、2、4、8 rows；
2. 每个 extra candidate 用多 selection seeds、多个 horizons 和多个 batch 评估；
3. 用风险下界而非零阈值接受：

\[
LCB(\Delta J)>\tau
\]

4. 同时约束 old retention；
5. 对基础方案和扩容方案做 full-step terminal confirmation；
6. terminal ranking 被推翻时回退基础 budget；
7. 分开报告固定 rows、extra rows、F1/rows、F1/steps 和 selector 成本。

因此 response-gain 默认关闭，只作为经过验证的可选预算机制。

## 3. 文本模块优化

当前文本流程是冻结 MiniLM、代表文本选择、截断、拼接后重新编码。已有证据支持重新编码，但没有证明固定的文本数量和字符预算普适最优。

按以下顺序验证：

1. center concat：强控制；
2. diversity-aware extractive：在固定 token budget 下同时优化中心代表性和多样性；
3. dual prototype：保存典型语义和差异语义两个 prototype；
4. set encoder：将多个 text embeddings 作为集合，用 DeepSets 或 attention pooling。

公平条件：rows、总 token budget、text encoder、graph feature、student steps、batch 和 seeds 全部相同。优先选择跨数据集不劣于 center concat 且成本更低的方案。set encoder 未经 canonical 验证前不进入主线。

## 4. 完整目标方法流程

### Step 0：冻结 protocol

为每个数据集写 manifest，包含 base/incremental sessions、类别顺序、pool、selector-dev、report-dev、test mask、prompt、graph preprocessing、text encoder、batch、steps、seeds 和 memory rule。构造只读 pool，test 只在最终确认读取。

### Step 1：读取当前 delta

对新到达数据计算 graph representation、冻结 MiniLM text embedding 和 normalized joint representation，同时计算 mass、dispersion、multimodality、novelty、coverage gap 和 redundancy。真实节点只作统计来源。

### Step 2：计算总 memory budget

主协议使用 fixed-rate：

\[
K_t=\lfloor\rho N_t^{pool}\rfloor
\]

adaptive-rate 只有在跨数据集门禁通过后才进入默认候选，否则只作消融：

\[
\rho_t=clip(\rho_0[1+\eta(U_t-\bar U)],\rho_{min},\rho_{max})
\]

### Step 3：确定性 joint-space clustering

在当前 delta 中形成多个子簇。类别增量实验可用类别做分块统计；通用任务使用无标签 joint-space clustering。每个子簇保存 mass、中心、半径、文本集合、结构统计和质量 metadata。

### Step 4：复杂度先验生成候选

用 sqrt + dispersion 或 complexity-v2 生成初始 rows，带上下限且总新增 rows 固定。复杂度只产生 rows prior、split/merge candidates 和 coverage metadata。

### Step 5：构造 synthetic nodes

对每个子簇构造：

- synthetic feature：均值、截尾均值或稳健中心；
- synthetic text：固定 token budget 下的中心/多样性文本表示或经验证的 set representation；
- metadata：mass、dispersion、radius、coverage、redundancy、text budget、stage 和 quality；
- synthetic topology：只连接 synthetic nodes，不把真实节点写入凝聚图。

### Step 6：拓扑候选

默认 frozen-old：保留旧 old-old topology，只新增 new-new 和受控 old-new edges。global rebuild 和 controlled cross-edge 作为对照，必须报告 edge count、degree 分布和 old-edge retention。

### Step 7：SCA response selection

对候选执行多 horizon、多 batch、多 selection seed 的短程 student response。先做 Pareto filtering，再以 new plasticity、old retention、coverage、cost 和 uncertainty 选择。beam 只作为路径稳定性诊断，不能挑最好路径后直接报告。

### Step 8：固定预算内 response marginal

在不改变新增 rows 的前提下，从低收益 donor 转移到高收益 recipient。只有置信下界稳定为正才接受，否则保留 dispersion 或 complexity prior。

### Step 9：保守扩容

只有 response-gain 的 reserve candidate 通过 LCB 和 full-step terminal confirmation 才增加 memory。否则不扩容。

### Step 10：历史 memory 管理

P4 前默认 Keep 高质量旧 nodes 和 frozen-old topology。后续可对低质量 synthetic nodes 进行 bounded refresh、split、merge、evict，但每个动作必须记录 before/after metadata，并同时评估 old retention 和 new plasticity。

### Step 11：canonical student

统一使用两层 GCN，hidden/output 384，冻结 MiniLM 384，student 从头训练，batch 16，v3 steps，student seeds 100--105，双向图文 InfoNCE，推理时使用真实图结构和类别 prompt zero-shot matching。

### Step 12：结果报告

每阶段报告 current F1、old F1、new F1、cumulative F1、forgetting、backward/forward transfer、synthetic rows、压缩率、每类/子簇 rows、text token budget、student steps、selector runs、wall time、edge count、old-edge retention、selector agreement 和 full-step regret。

## 5. 目标算法摘要

输入 base data 和增量数据流，输出每阶段 synthetic memory：

1. 初始化 \(M_0=Condense(D_0)\)；
2. 对 \(D_t\) 限制 construction pool；
3. 计算信息量和历史覆盖统计；
4. 计算 \(K_t\)；
5. 在 graph-text joint space 聚类；
6. 用 complexity prior 生成固定总 budget 的 rows；
7. 构造 synthetic features、text prototypes 和 synthetic topology；
8. 生成 frozen-old 等 topology candidates；
9. 用多 horizon response curve 和独立 selection seeds 选择；
10. 用 response marginal 做固定预算内转移；
11. 用 LCB 和 terminal confirmation 决定是否使用 reserve budget；
12. 用 frozen-old 或已验证的 bounded refresh 更新 memory；
13. 用 canonical seeds 完整训练并保存 manifest。

## 6. P1-P5 分阶段计划

### P0：协议和审计基础设施，已完成

统一 mask、预算、阶段、student、文本缓存、synthetic-only 和 manifest。P0 只证明可比较，不证明性能。

### P1：预算和 allocation，当前阶段

依次完成 1%/10% matched-quota、真实有效的 hybrid、complexity-v2、response-marginal-v2、response-gain-v2，并汇总 rows、F1、old retention、new plasticity、coverage 和训练成本。只有跨两个 rate 和一个额外阶段/顺序通过才上大图。

### P2：文本压缩

固定 rows 和 token budget，验证 center concat、diversity-aware extractive、dual prototype、set encoder，先冻结文本方案。

### P3：SCA 稳定性

统一 runner，加入独立 selection seeds、ranking consistency、Pareto retention、terminal confirmation、path frequency 和 full-step regret。

### P4：历史 memory manager

验证 bounded refresh、importance eviction、synthetic split/merge 和 controlled cross-edge update，分析保持-塑性 Pareto。

### P5：正式多阶段和外部对比

主协议采用 base+4，3-stage 作为快速消融，5/8-stage 作为长期遗忘；补充多类别顺序、官方 GraphCLIP 三锚点和统一 continual baselines。

## 7. 最终贡献边界

如果所有门禁通过，论文可以主张：

1. 可审计的 continual TAG condensation protocol；
2. synthetic-only、complexity-aware 的 memory construction；
3. response-calibrated append and allocation；
4. retention-aware historical memory management；
5. 对固定总预算、盲目扩容、简单复杂度先验和短程 response 失配的机制解释。

在实验完成前不能声称 adaptive-rate、complexity-v2、response-gain 或 frozen-old 跨数据集普遍最优，也不能宣称当前方法已是 continual TAG SOTA。

## 8. 当前默认决策

1. fixed-rate + matched-quota + dispersion 为主控制；
2. complexity-v2 只作 candidate generator；
3. response marginal 只在固定总 rows 中运行；
4. response-gain 默认关闭，必须 reserve + LCB + terminal confirmation；
5. topology 使用 frozen-old；
6. 文本优先验证 diversity-aware extractive，center concat 为 control；
7. P4 前不做全量历史重聚类；
8. canonical seeds 100--105，selection seeds 独立，test 最终一次读取。

一句话概括：

> 一个完全由合成节点构成、预算随数据和信息量增长、节点容量按结构复杂度生成候选、由 student response 在固定预算内校准，并以 retention-aware memory manager 控制旧类保持与新类塑性的持续图文凝聚系统。
