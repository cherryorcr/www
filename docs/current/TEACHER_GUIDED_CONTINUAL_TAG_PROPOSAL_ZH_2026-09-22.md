# TGC-R：Teacher-Guided Continual Revision for TAG Condensation

日期：2026-09-22  
状态：主线候选。Phase A 已完成 648 fits，未晋升；R2 已完成 336 次新 canonical 训练、144 条短训轨迹及 20 组共享 teacher 回放，仍未形成稳定正收益。详见 [R2 结果与执行边界](TGC_R2_OPTIMIZATION_RESULTS_ZH_2026-09-22.md)。下方完整目标是候选设计，不代表预算、文本、拓扑组件均已实现。

## 1. 为什么需要新版本

当前 C0 是可靠的冻结底座，但其预算、X、文本和拓扑主要由确定性几何规则决定。此前几条优化路线没有形成稳定主方法：

- response selector 使用短训 student 排序，成本高且 canonical ranking 与最终 ranking 不稳定；
- C0-M 使用 joint geometry 的 redundancy/novelty 管理历史 rows，平均 final F1 下降 0.34 pp，new F1 下降 0.93 pp；
- 文本 attention、复杂 X、联合核和简单 refresh 没有跨数据集稳定收益。

因此下一版不再把 student 当作昂贵的候选排序器，也不再只用几何距离判断哪些内容有用。候选方法应在构造阶段直接匹配“student 需要的训练信号”。

## 2. 核心想法

方法名称：**TGC-R**（Teacher-Guided Continual Revision）。

给定上一阶段合成 memory `M_{t-1}` 和当前合法原始增量 `Delta_t`，训练一个只访问 `M_{t-1} union Delta_t` 的流式 teacher，利用 teacher 的：

1. 图文表示；
2. 图消息聚合统计；
3. student 参数梯度签名；
4. 跨模态不确定性和覆盖缺口；

共同指导当前阶段的有限 memory revision、后续的预算分配、合成属性、文本选择和新旧连接。

与 C0-M 的区别是：TGC-R 不根据 redundancy/novelty 直接刷新旧行，而是先由合法 teacher 计算训练信号，再执行 revise-or-freeze gate。旧 synthetic X 允许在有界范围内修改；旧文本、旧旧边、mass、depth 和 birth stage 保持不变。当前最小实现只开启旧 X revision，其余构造完全复用 C0。

候选论文主张应表述为：

> 在历史原始图不可回读的 continual TAG condensation 中，使用只访问旧 synthetic memory 与当前 raw delta 的流式 teacher，对历史 synthetic rows 执行带功能保持证书的选择性 revision，而不是无条件重凝聚或永久冻结。

这是候选创新主张，必须经过相关工作核对和跨数据集实验后才能使用。

## 3. Teacher 与 student 的角色

### 3.1 流式 teacher

阶段 `t` 的 teacher 只允许读取：

- 当前原始增量 `Delta_t`；
- 旧 synthetic memory `M_{t-1}`；
- 当前阶段的合法训练图和文本。

它不能回读历史 raw graph，不能读取 report-dev/test 标签，也不参与最终推理。Phase A 不把 teacher 参数传递给下游 student，而是在当前访问范围内拟合临时 teacher，并用旧 memory 的图文响应和梯度证据产生 revision proposal：

`L_teacher = L_InfoNCE(M_{t-1} union Delta_t)`。

teacher 不是理想上界，而是当前访问约束下的构造监督器。另设 full-history teacher 只作为 oracle diagnostic，不能进入主方法。

### 3.2 Downstream student

最终 student 与 C0 完全相同：从头在 `M_t` 上训练双向图文 InfoNCE，使用真实图推理节点和类别 prompt。teacher 参数不传给 student，避免把 teacher 的容量或预训练优势误计为凝聚收益。

## 4. 统一 teacher-guided 目标

新 block 的构造变量包括合成属性 `X_syn`、固定文本候选的选择权重和新增边 logits。核心目标为：

`L_TGC = lambda_e L_embed + lambda_g L_grad + lambda_m L_message + lambda_txt L_text + lambda_reg L_budget/topology`。

### 4.1 Phase A：teacher-guided X revision

对旧行 `i`，teacher 计算输入梯度 `g_i = d L_teacher / d x_i`、旧行图文响应和当前 delta affinity。候选更新沿 `-g_i` 移动，并投影到：

`||x_i' - x_i|| / max(||x_i||, eps) <= r`。

只允许非 protected core 行进入候选；每阶段 revision quota 独立于总 memory budget，实际更新数可以为零。候选更新只有在 teacher 总损失不增、旧行平均响应不超过 preservation tolerance 的条件下才接受：

`benefit_i - lambda * risk_i > threshold`。

这一步是当前最小实验主线，直接检验“允许受约束修改旧 synthetic X”是否比 C0 的 frozen-old 更有用。

### 4.2 Teacher embedding matching

对每个类别/局部簇匹配 teacher 的图表示、文本表示和图文联合表示，替代只匹配原始 X/text 几何中心：

`L_embed = || mean(Phi_syn) - mean(Phi_teacher) ||^2`。

它保留 C0 的确定性聚类初始化，但允许 teacher 纠正 mean-X 的过度收缩。

### 4.3 Student-gradient matching

在固定 student 初始化 `theta_0` 上，对相同 InfoNCE 目标计算：

`L_grad = || grad_theta L_InfoNCE(M_t; theta_0) - grad_theta L_InfoNCE(M_{t-1} union Delta_t; theta_0) ||^2`。

这借鉴 GCond/trajectory-style condensation 的训练信号思想，但本项目的区别不应只写“使用梯度匹配”，而应强调：目标是 continual、图文、多阶段且历史 raw 不可回读的合法流式目标。

### 4.4 Message matching topology

teacher 记录每个候选簇的图消息统计：

`m_i^T = sum_j A^T_ij h_j^T`。

新旧连接和新块内部边通过固定边预算下的 edge logits 选择，使合成图的消息聚合接近 teacher message：

`L_message = sum_i ||m_i^syn - m_i^T||^2`。

旧旧边保持 C0 不变；只优化 new-new 和 new-old cross edges，并投影到 degree/cross-capacity 约束中。

### 4.5 Teacher-conditioned text selection

对每个合成簇，将 teacher graph prototype 作为 query，将候选文本片段 embedding 作为 keys/values：

`alpha_j = softmax(q_T^T W k_j - lambda_red redundancy_j)`。

在固定字符/token 预算内选择 top-g 且尽量不重复的片段，拼接后重新用冻结 MiniLM 编码。第一版不生成幻觉文本，不直接存无法还原的连续 text embedding；可学习的 attention 只负责选择合法源片段。

## 5. 后续组件：Teacher-guided budget

总 memory budget 仍由 C0 的 fixed-rate growing rule 决定，避免把总行数增长误当成算法收益。teacher 只重新分配当前新增 rows：

`u_{t,c} = a * uncertainty_c + b * gradient_energy_c + c * coverage_gap_c + d * message_residual_c`。

在每类最小预算约束下，根据 `u_{t,c}` 做确定性整数分配。这里的 marginal utility 是 teacher-side 一次前向/反向统计，不再对每个候选重复训练多个 student。

需要设置 teacher-free 对照：

- C0 dispersion allocation；
- 只用 mass/size；
- 只用 uncertainty；
- 只用 gradient energy；
- 完整组合 utility。

否则无法判断收益来自 teacher 还是又一个预算启发式。

## 6. TGC-R 完整目标流程

1. 初始化 `M_0`：按 C0 构造 synthetic block。
2. 阶段 `t` 到达，读取且只读取 `M_{t-1}` 和当前 `Delta_t`。
3. 用二者拟合临时流式 teacher，计算 embedding、message、gradient 和 uncertainty 统计。
4. 对旧 memory 建立 protected core；对剩余 rows 运行 revise-or-freeze gate。
5. 在固定总预算下按 teacher utility 分配当前新增 rows。
6. 用 C0 joint clustering 初始化簇，再用 teacher embedding/gradient loss 优化合成 X。
7. 用 teacher-conditioned attention 在固定文本预算内选择片段，重新编码文本。
8. 用 message matching 优化 new-new/cross edge logits，并投影到固定边数、度上限和跨块容量约束。
9. 将 revised old memory 与新 block 合并；旧文本、旧旧边和受保护 rows 不改写。
10. 丢弃 teacher，仅从头训练 downstream student，报告 cumulative/old/new F1、LP 指标、成本和 memory 字节。

## 7. 与已有工作的区别

| 方法类别 | 典型思想 | 与 TGC-Append 的区别 |
|---|---|---|
| GCond/gradient matching | 用真实图和合成图的 GNN 梯度匹配优化静态凝聚图 | TGC-R 将梯度作为合法流式 revision evidence，而不是访问完整历史 raw 图做静态优化 |
| trajectory/teacher-style condensation | 让合成数据复现模型训练轨迹或参数更新 | TGC 的 teacher 只能使用当前增量和旧 synthetic memory，不能使用完整历史 raw |
| TAGSAM 类响应匹配 | 用 student/teacher response 优化合成表示和拓扑 | TGC-R 用可审计的 revise-or-freeze gate 和 preservation certificate，避免逐候选短训排序 |
| STAD/蒸馏类 TAG 方法 | 用 teacher logits/文本/结构蒸馏学生 | TGC 的重点是持续 memory 构造、预算分配和新增拓扑，而不是单次静态蒸馏 |

不能仅凭“用了 teacher”宣称创新。真正需要验证的是：同一 teacher 信号能否同时改善预算、X/text 和 topology，并在不增加总 memory 的情况下提高 student retention/plasticity。

## 8. 分阶段实验

### Phase A：teacher-gated revision（当前最小路线）

固定 C0 的新增 block、文本、topology、总预算和 downstream student，只允许 teacher 修订旧 synthetic X。对照包括 C0、naive refresh、C0-M、teacher gate、去 preservation gate 和 shuffled teacher control。若 teacher gate 不能稳定优于 C0，关闭后续复杂组件。

### Phase B：teacher-guided X/text

固定预算和 topology，只优化 synthetic X 与 teacher-conditioned text selection，加入 text-only、gradient-only 和 embedding-only 对照。

### Phase C：teacher-guided topology

固定 X/text，只优化 new-new/cross edge logits，比较 message matching、T+degree、C0 topology 和 shuffled-teacher control。

### Phase D：完整 TGC-R

只有 A--C 至少有两个组件通过门禁，才运行组合矩阵：C0、A、B、C、A+B、A+C、B+C、A+B+C。组合实验必须保持总 rows、边数、student steps 和 seeds 相同。

初始小图协议：Cora/Citeseer、canonical + 两个独立顺序、5% rate、3 stages、student seeds 100--105。晋升建议：平均 final F1 至少 +1 pp，至少 4/6 cell 不低于 C0-1 pp，最差不低于 -2 pp，new F1 平均为正，old F1 最差不低于 -2 pp，构造成本不超过 2x，logical memory 不超过 1.5x。

通过后再做 arXiv/BookChild/SportsFit base+4；未通过则 C0 保留可靠基线，TGC-R 作为“受约束历史 revision”负结果诊断。

## 9. 论文创新潜力与风险

### 潜在贡献

1. 提出新的 continual TAG condensation setting：历史 raw 不可回读、memory 持续增长、下游 student 从头重训。
2. 提出合法流式 teacher，把历史 synthetic revision、预算、图文内容和新增拓扑统一到 teacher sufficient-statistics matching。
3. 给出带 preservation certificate 的 revise-or-freeze 操作，而不是无约束刷新或永久冻结。
4. 给出旧类保持、新类塑性、memory cost 和构造成本的统一评估。

### 主要风险

- 如果 teacher 只是更长训练的 proxy，收益可能来自 teacher capacity，而不是凝聚算法；必须有 teacher-free 和 oracle teacher 对照。
- 梯度匹配本身已有大量工作，创新不能只写“加入 gradient matching”。
- 同时优化 X、文本和 topology 容易造成成本爆炸；必须先做单组件门禁。
- teacher 可能继承 C0 的错误，尤其在长链和顺序变化下；需要独立顺序和 teacher drift 审计。

## 10. 当前决策与实现状态

当前 all-in 主线为 TGC-R，但严格按最小路线推进：

1. 已完成代码骨架：`tgc_revision_gate.py`、`run_tgc_revision_gate.py`、`audit_tgc_revision_gate.py`、`summarize_tgc_revision_gate.py` 和 CPU contract tests。
2. Phase A 首轮只打开旧 X revision；预算、文本、拓扑和 student 全部锁定 C0。
3. Phase A 已完成：6 cells、648 fits，final 相对 C0 平均 -0.88 pp。R2 targeted 容量扫描也已完成：4 个暴露开发 cells、7 个新配置、336 次新训练，均未过开发门槛。
4. R2 的 144 条独立 seed response 轨迹仅属离线诊断；64-step 四候选选择有 +0.41 pp 信号，但不是已运行的持续 selector。文本/拓扑组合与大图未触发。
5. 重要解释修正：固定 teacher 与相同父 memory 回放时，原配置 gate/no-preserve 在 12/12 增量阶段产生逐位相同的 X；原独立 teacher 两臂的 F1 分差不能作为 preservation 有效性的证据。teacher-response 约束不是历史原始任务风险的理论保证。
