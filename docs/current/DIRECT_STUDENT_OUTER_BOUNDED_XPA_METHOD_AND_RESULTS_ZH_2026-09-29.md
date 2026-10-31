# 受限 Student-Compatible X-P-A 持续 TAG 凝聚方法

日期：2026-09-29

## 1. 当前结论

当前方法不是“再做一次静态聚类”，而是在已有持续合成记忆上增加一个受限的 Student-compatible revision 层：

1. 用可微 Student 训练轨迹直接反向优化新合成节点特征 `X`；
2. 用平衡软分配 `P` 将当前到达数据的图特征和文本表示传输到新合成节点；
3. 用候选池和度守恒的 `A` 只修改新旧块之间的少量边；
4. 对每个新合成节点限制最多替换一条原有跨块边；
5. 可选地要求新候选边相对原边具有最小 Student 得分优势；
6. 用同一候选图进行 Student 内层训练、外层梯度计算和最终二值拓扑落地。

因此，当前主线应表述为：

> Bounded Student-Compatible Semantic Revision for Continual TAG Condensation

它是建立在 Semantic-Recursive/C0 初始合成记忆上的增量修正算子，不是完全从原始历史图重新凝聚，也不是已经证明在所有数据集上最优的最终算法。

本轮不再继续寻找小图上的唯一最优参数。应冻结上述算法结构，在大图和更多顺序上重新标定 `X` 半径、`P` 混合比例、拓扑温度和 revision margin。

## 2. 要解决的问题

在阶段 `t`，算法只能访问：

\[
M_{t-1}^{syn}, \qquad \Delta_t^{raw},
\]

其中 `M` 是上一阶段合成记忆，`Delta` 是当前新到达的合法原始数据。历史原始图不能重新读取。

已有简单 append 方法有三个缺陷：

- 新合成节点只追加，和旧合成节点没有 Student 反馈下的协同；
- `X`、文本和拓扑都由启发式规则产生，训练 Student 的目标不直接作用到它们；
- 如果允许完全自由地改边，少量新类收益可能以大范围旧类传播破坏为代价。

本方法的核心取舍是：允许必要的协同更新，但把可修改范围、传播质量和结构变化量显式限制住。

## 3. 输入、输出和符号

### 3.1 Candidate memory

候选合成记忆写为：

\[
M^{cand}=(X,T,E),
\]

其中：

- `X`：合成节点图特征；
- `T`：冻结 MiniLM 空间中的文本 embedding；
- `E`：合成图边集合；
- `old_rows`：旧合成节点前缀长度。

`0,...,old_rows-1` 是旧 synthetic rows，后缀是当前阶段待修正的新 synthetic rows。

### 3.2 Legal support

外层优化的 support 是：

\[
S=(X^s,T^s,E^s),
\]

由上一阶段合法合成前缀和当前 raw delta 组成。support 被预先切分为 train/validation，切分后不会因为类别或 slot 过滤而重新分配样本。

### 3.3 访问约束

本实现审计并保证：

- 不读取历史完整 raw graph；
- 不使用 report-dev/test 标签选择 outer checkpoint；
- 不修改输入 `base` 或 `support`；
- 输出目录不可覆盖已有结果；
- 输出 audit 保存输入哈希、support split、梯度变量和拓扑修改数量。

## 4. 完整方法流程

### Step 0：准备初始候选图

先由已有 Semantic-Recursive/C0 构造得到 `base`。这一阶段负责提供：

- 新旧 synthetic rows；
- 初始 `X` 和冻结文本 embedding；
- 初始边集合；
- 新 synthetic rows 对应的 construction-time class id。

本方法不重新读取历史原始图，也不在 direct outer 层重新生成原始文本。

代码入口：

`code/v2_llm_fusion/run_direct_student_outer.py:69-225`

### Step 1：建立 sealed support train/validation plan

support 被分成 old/new 两部分，并使用固定随机种子形成 train/validation batches：

\[
\mathcal{S}_{train}\cap \mathcal{S}_{val}=\varnothing.
\]

当前实现中的 support plan 在 outer 优化开始前只创建一次；后续的 class 过滤只筛选已有 split，不会把 validation row 移到 train。

代码：

`direct_student_outer.py:426-536`

### Step 2：建立可学习的 X 参数

对选中的新 synthetic rows 建立相对扰动参数：

\[
R_X\in\mathbb{R}^{n_{new}\times d_x},
\]

初始值为零。第 `i` 个新节点的候选特征为：

\[
x_i^{cand}=x_i^{base}+\|x_i^{base}\|_2 r_i.
\]

这样优化的是相对移动而不是不受尺度控制的绝对向量。每个 outer step 后投影到半径约束：

\[
\|r_i\|_2\leq r_{new}.
\]

当前小图代表性设置为 `x_new_radius=0.10`。旧 rows 在当前 `x-class+p+a` 主线中不直接更新；旧记忆通过候选拓扑的跨块边参与新旧协同。

代码：

- 变量创建：`direct_student_outer.py:1516-1575`
- 候选 X 合成：`direct_student_outer.py:1460-1498`
- 半径投影：`direct_student_outer.py:1504-1514`

### Step 3：初始化并学习 P

`P` 是当前 raw delta 到新 synthetic rows 的软传输矩阵，不是标签分类器。

对候选新节点 `i` 和当前 support 新行 `j`，先计算图文联合相似度：

\[
L_{ij}=
\frac{\operatorname{cos}(x_i^{base},x_j^{raw})+
\operatorname{cos}(t_i^{base},t_j^{raw})}
{\tau_P}.
\]

之后用 Sinkhorn 归一化得到：

\[
P_{ij}\geq 0,
\qquad
\sum_jP_{ij}=1,
\qquad
\sum_iP_{ij}\approx\frac{n_{new}}{n_{raw}}.
\]

当前候选 X 和文本为：

\[
\tilde X_{new}=(1-\mu)X_{new}^{base}+\mu P X_{raw},
\]

\[
\tilde T_{new}=(1-\mu)T_{new}^{base}+\mu P T_{raw}.
\]

当前小图设置使用 `p_mix=0.25`。`P` 的 logits 也属于 outer variables，梯度来自 Student 训练轨迹。

class-constrained P 曾经测试过，但在 Cora/Citeseer 上降低性能，因此当前主线使用跨类共享的 balanced P；类别信息通过 candidate class margin 进入，而不是把 P 强行限制在类内。

代码：

- P 初始化：`direct_student_outer.py:1131-1160`
- Sinkhorn：`direct_student_outer.py:1150-1160`
- P 应用到 X/T：`direct_student_outer.py:1439-1460`
- class-constrained 负结果实现：`direct_student_outer.py:1158-1190`

### Step 4：构造 A 的有限候选池

`A` 只负责新节点到旧节点的跨块边。对每个新节点 `i`，根据图特征和文本 embedding 的联合相似度，从旧 rows 中选最多 `a_pool=4` 个候选旧节点，并强制把原来已经存在的旧邻居加入候选池。

\[
C_i=\operatorname{TopK}_{old}
\left[
\operatorname{cos}(x_i,x_o)+
\operatorname{cos}(t_i,t_o)
\right]
\cup N_{old}^{base}(i).
\]

同时记录原始跨块度数：

\[
d_i^{old}=|N_{old}^{base}(i)|.
\]

最终不会凭空增加或删除该新行的跨块边总数。

代码：

`direct_student_outer.py:1194-1243`

### Step 5：对 A 做可微软选择

每个候选边有一个可学习 logit `a_logits`。普通 soft A 使用：

\[
\tilde a_{ij}=d_i^{old}\cdot
\operatorname{softmax}(l_{ij}/\tau_A).
\]

训练时使用 straight-through 形式：

\[
a^{ST}=a^{hard}-\operatorname{stopgrad}(\tilde a)+\tilde a,
\]

因此 forward 使用离散边，backward 仍然能够回传到 `a_logits`。

### Step 6：限制每个新节点的 topology revision

当前主线加入：

\[
r_i\leq 1,
\]

即每个新 synthetic row 最多替换一条原有旧/新跨边。当前 Cora/Citeseer 小图里大多数新行原始跨块度为 1，因此该约束的实际含义是：

> 每个新节点只能“保留原边”或“用一个候选新边替换原边”，不能同时重写多条旧关系。

如果 `revision cap=0`，A 成为冻结拓扑控制；此时 `a_logits` 没有梯度是预期行为，不再被当作异常。

代码：

`direct_student_outer.py:1244-1338`

### Step 7：可选的 revision confidence gate

为避免仅因 logit 略高就替换旧边，当前实现支持 revision margin `tau_R`。对于一条原跨边和若干新候选边：

\[
g_i=
\sigma\left(
\frac{l_{i,new^*}-l_{i,old^*}-\tau_R}{\tau_A}
\right).
\]

soft 阶段使用：

\[
\tilde a_{i,old}=1-g_i,
\qquad
\tilde a_{i,new}=g_i\cdot
\operatorname{softmax}(l_{i,new}/\tau_A).
\]

hard 阶段使用同一个条件：

\[
l_{i,new^*}-l_{i,old^*}>\tau_R
\Rightarrow \text{replace},
\]

否则保留原边。这样训练图和最终 materialized 二值图的拓扑决策规则一致。

代码：

`direct_student_outer.py:1270-1318`

### Step 8：重建归一化候选图

只替换跨块边，旧块内部边、新块内部边和其它固定边保持不变。加入 self-loop 后使用标准 GCN 归一化：

\[
\hat A=D^{-1/2}(A+I)D^{-1/2}.
\]

代码：

- 归一化：`direct_student_outer.py:1113-1129`
- 候选图：`direct_student_outer.py:1345-1390`
- 最终二值 materialization：`direct_student_outer.py:1391-1424`

### Step 9：Student 内层训练

Student 是与下游评估一致的两层 GCN + text projection，而不是额外换成 GraphGPS 或监督 GCN。

给定候选 `X/T/A`，对每个 Student seed 从同样的模型初始化开始，执行 `K=8` 个可微 Adam 更新：

\[
\theta_{k+1}=\operatorname{Adam}
\left(\theta_k,
\nabla_{\theta_k}\mathcal L_{in}(M^{cand};\theta_k)
\right).
\]

图编码器为：

\[
H_1=\operatorname{ReLU}(\hat A XW_1+b_1),
\]

\[
H_2=\operatorname{Normalize}(\hat A H_1W_2+b_2).
\]

文本编码器在 direct outer 阶段不是重新运行 MiniLM，而是对冻结 MiniLM embedding 做可学习线性投影：

\[
Z_T=\operatorname{Normalize}(TW_T+b_T).
\]

代码：

`direct_student_outer.py:271-338`

### Step 10：双向图文 InfoNCE

对一个 batch 的目标 rows `B`，使用对称图到文和文到图的 InfoNCE：

\[
\mathcal L_{NCE}=
\frac12\left[
CE(sH_B Z_T^\top,I)+
CE(sZ_{T,B}H^\top,I)
\right].
\]

旧 support 和当前新 support 分别计算：

\[
\mathcal L_{support}=
\lambda_{old}\mathcal L_{old}+
\lambda_{new}\mathcal L_{new},
\]

当前设置为 `old_weight=0.25`、`new_weight=1.0`，让新类塑性得到更强的直接信号，同时保留旧类约束。

代码：

- Student NCE：`direct_student_outer.py:300-338`
- support loss：`direct_student_outer.py:641-760`

### Step 11：candidate class margin

为了让新类目标真正作用到 candidate graph，而不是只在 support raw graph 上计算，当前 `x-class+p+a` 还在同一个 candidate graph 上加入 class-prompt margin：

\[
\mathcal L_{margin}=
\frac1{|B_{new}|}
\sum_i
\max\left(0,
m-s_{y_i}+\max_{c\neq y_i}s_c
\right).
\]

这里的 `y_i` 只是 construction-time class metadata，用于构造阶段区分新类，不是 report/test label，也不参与最终 checkpoint 选择。

代码：

`direct_student_outer.py:901-940`

### Step 12：完整 outer objective

当前核心 outer objective 可以写成：

\[
\min_{R_X,L_P,L_A}
\mathbb E_{s\in\mathcal S_{train}}
\left[
\lambda_o\mathcal L_o(\theta_K^s)
 +\lambda_n\mathcal L_n(\theta_K^s)
 +\lambda_m\mathcal L_{margin}(\theta_K^s)
 +\lambda_P\mathcal R_P
\right],
\]

其中 `theta_K` 是在 candidate graph 上经过 K 步内层 Student 训练得到的参数，`R_P` 是 P 的列平衡正则项。因为 inner Adam 使用 `create_graph=True`，外层反向传播能够计算：

\[
\nabla_{X}\mathcal L_{outer},
\qquad
\nabla_{P}\mathcal L_{outer},
\qquad
\nabla_{A}\mathcal L_{outer}.
\]

代码：

- outer objective：`direct_student_outer.py:1662-1865`
- outer loop：`direct_student_outer.py:1929-2210`
- 可微 Adam：`direct_student_outer.py:338-405`

### Step 13：validation checkpoint 和 materialization

每个 outer step 后使用 sealed validation support loss 选择 outer checkpoint。这个 validation 是 construction split，不是下游 report-dev/test。

选出最优 outer state 后：

1. 固化 X；
2. 固化 P 传输后的新行 X/T；
3. 用同一 A gate 产生最终边集合；
4. 检查 old rows、不可编辑字段、输入哈希和 payload 有限移动；
5. 保存 memory、audit、protocol。

代码：

`direct_student_outer.py:2140-2260`

## 5. 当前实验结果

所有结果都与对应 `control_*` 使用相同的下游 Student、相同 6 个 Student seeds、相同 report-dev 规则；表中的数值是 candidate 相对 C0/control 的百分点变化。

### 5.1 受限拓扑的必要性

| 方法 | Final Macro-F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| cap=0，A 完全冻结 | -0.57 | -1.17 | -1.27 | +0.01 |
| cap=1，margin=0，第一 construction seed | +2.41 | +1.55 | +2.85 | +0.75 |
| cap=1，margin=0，第二 construction seed | +4.74 | +2.82 | +8.50 | -1.09 |
| cap=1，margin=0，8-cell pooled | **+3.57** | **+2.18** | **+5.68** | -0.17 |

解释：只优化 X/P 而完全冻结 A 不足以复现拓扑协同收益；真正的增益来自有限的 Student-guided cross-edge revision，但不受限的 revision 会产生旧类风险。

### 5.2 revision margin 的作用

| 方法 | Final Macro-F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| cap=1，margin=0.05，8-cell pooled | +1.53 | +1.26 | +2.05 | **+1.20** |
| cap=1，margin=0.02，8-cell pooled | **+2.64** | **+1.93** | **+3.60** | **+0.24** |

这表明 margin 是稳定性组件，不应被解释成在所有图上固定最优的数字。`0.05` 更保守，`0.02` 在本轮 pooled 结果中保留了更多新类收益。

### 5.3 逐 cell 的 margin=0.02 结果

| Construction seed | Cell | Final F1 | Accuracy | New F1 | Old F1 |
|---|---|---:|---:|---:|---:|
| seed1 | Cora canonical | +6.55 | +3.86 | +9.81 | -1.24 |
| seed1 | Cora order19102 | -0.34 | -1.17 | -0.10 | -0.57 |
| seed1 | Citeseer canonical | -0.10 | +0.14 | -0.82 | +1.09 |
| seed1 | Citeseer order19102 | +5.39 | +5.72 | +1.34 | +7.01 |
| seed2 | Cora canonical | +0.18 | +1.15 | +0.61 | -3.88 |
| seed2 | Cora order19102 | +9.86 | +6.17 | +16.54 | +1.04 |
| seed2 | Citeseer canonical | -0.94 | -0.88 | -0.72 | +0.68 |
| seed2 | Citeseer order19102 | +0.53 | +0.49 | +2.15 | -2.24 |

当前证据支持“平均有效、存在顺序和 construction seed 敏感性”，不支持“每个 cell 都稳定超过 C0”。这也是为什么下一步应停止小图调参，转向大图和更多预注册顺序验证。

## 6. 做过但暂不进入主线的机制

### 6.1 完全冻结 A

作为 cap=0 控制，它的 New F1 平均下降，说明只优化 X/P 不能替代拓扑协同。

### 6.2 统一提高旧边初始 logit

`a_existing_logit=0.25` 会使 A 几乎不迁移，结果接近 cap=0。它把所有 rows 用同样的偏置处理，无法区分“应保护”和“值得替换”的局部结构。

### 6.3 old-response barrier

加入统一 old-response barrier 后，旧类保护没有稳定转化为最终 F1，且会压制有效的新类迁移。因此它保留为诊断模块，不进入当前核心流程。

### 6.4 class-constrained P

强制 P 只能在同类内传输会丢失跨类共享结构；它在早期小图实验中下降，因此当前 P 保持跨类 balanced transport。

## 7. 当前方法的正确定位

当前方法的可解释创新不是某一个神奇的 `0.02`，而是把持续 TAG 凝聚中的三个以前分离的对象放入同一个 Student-compatible outer loop：

\[
\boxed{X\;\text{content revision}}
\quad+
\boxed{P\;\text{payload transport}}
\quad+
\boxed{A\;\text{bounded topology revision}}
\]

并且通过：

- differentiable Student unroll；
- candidate/support 双图一致性；
- degree conservation；
- per-row revision cap；
- optional confidence margin；
- no historical raw replay；

把“新类塑性”和“旧类保持”放在同一个可审计的持续凝聚协议中。

## 8. 后续工作边界

本轮之后不再在 Cora/Citeseer 上继续寻找更优 margin。后续只做：

1. 把当前结构迁移到 Arxiv 等大图，重新标定 radius、P mix 和 topology temperature；
2. 使用 base+4、多顺序和 construction seed 做正式确认；
3. 报告 Final Macro-F1、Accuracy、Old F1、New F1、revision count、construction time 和 Student time；
4. 与公开静态 TAG 凝聚、强 coreset、raw reference 和直接 Student continual baseline 统一协议比较；
5. 只有当大图上仍显示相同的“有限拓扑迁移带来塑性、margin 控制旧类风险”现象时，才把该结构冻结为论文最终算法。

### 8.2 下一轮最小优化：新类响应权重

当前新类塑性的主要可疑瓶颈是：outer objective 中的 new support InfoNCE 虽然权重高于 old support，但其梯度仍可能被旧响应、candidate margin、P 平衡项和 A 的拓扑变化共同稀释。下一轮先不改 X/P/A 的参数化，只增加一个显式的 `new_loss_scale`：

\[
L_{outer}=\lambda_o L_{old}
 + \lambda_n\,\gamma_{new}L_{new}
 + \lambda_mL_{margin}+\lambda_PR_P.
\]

默认 `gamma_new=1`，因此不改变现有方法；实验版本只比较固定的少量值，例如 `1.5` 和 `2.0`，并在同一 construction seed、Student seed、memory budget 和报告协议下运行。该参数不是根据 test F1 选择，而是预先固定，并且必须同时报告 New F1、Old F1、Macro-F1、Accuracy 及 construction cost。

这项改动的目的不是简单“把新类分数调高”，而是验证一个明确假设：当前 X-P-A 的负收益是否来自新类 Student response 在联合目标中信号不足。如果提高 new response 权重只提升 construction loss、但不提升 New F1，说明瓶颈不在损失权重，应转向 X 的表示初始化或 A 的消息算子；如果 New F1 上升而 Old F1 受控，再继续做大图确认。

本地回归测试现为 `33 passed`。下一步先做 Cora/Citeseer、两个顺序、一个 construction seed 的小规模 smoke，再决定是否进入完整六 seed 矩阵。

### 8.3 当前继续优化：旧图表示 anchor

大图 X-P-A 结果显示，arXiv 和 BookChild 在长链后期退化，而 SportsFit 保持较好。这表明问题可能不是新类目标不足，而是新阶段的 X/P/A 更新通过归一化消息传播改变了旧 synthetic rows 的 Student 表示。下一轮加入已有的 `old_graph_anchor_loss`，只约束候选图中旧 rows 的表示接近 base C0 图：

\[
L_{anchor}=\frac{1}{|V_{old}|}
\sum_{i\in V_{old}}
\left(1-\cos(h_i^{cand},h_i^{base})\right).
\]

该项不访问历史 raw graph，不更新旧节点本身，不改变 memory budget 或 edge budget；它只通过 Student outer gradient 约束新 X、P 和 A 不要过度破坏旧节点的传播响应。实验只先比较 `anchor=0.25`，并和当前 `anchor=0` 配对，避免再做大范围参数扫描。默认值仍为 `0`，因此原方法行为不变。

上一轮全量旧节点 anchor 在 arXiv/BookChild stage 1 分别下降 `1.25` 和 `2.60 pp`，因此本轮改为稀疏 anchor：每个 outer step 只选候选图中漂移最大的旧 rows 的比例 `old_graph_anchor_fraction`，用它们计算 anchor loss。这样约束的是实际受到新旧消息扰动的风险节点，而不是把全部旧 memory 固定住。默认 fraction 仍为 `1.0`；下一次 smoke 只测试 `fraction=0.25`，并保持 weight=`0.25`。

## 8.1 按类 Student-response P gate 补充实验（2026-09-30）

为检验 old-context payload 是否只在部分新类上有益，本轮增加了按新类独立决策的 gate。它不改变 X、A、预算或 Student；只将 P 的 old-context ratio 在每个新类对应的合成行上分别决定是否保留。

对每个新类 `c`，使用同一 Student 初始化、相同 inner steps、相同 support split，比较：

\[
\Delta_{new,c}=L_{new,c}^{mixed}-L_{new,c}^{pure},
\qquad
\Delta_{old,c}=L_{old,c}^{mixed}-L_{old,c}^{pure}.
\]

只有 `Delta_new,c < -p_gate_margin` 且 `Delta_old,c <= p_gate_old_tolerance` 时，才保留该类的 old-context P；否则该类回退到 pure-new-P。最终 materialization 复用最佳 validation checkpoint 的逐行决策，不读取 report/test 标签。

### 实现校验

- `_class_plan()` 现在保留 sealed train/validation 成员，只过滤 target rows；同类 target 的 context 可以保留同一 split 中其它新类作为 InfoNCE negative；
- 允许某类 validation 只有一个 target，只要 context 至少包含两个 rows；
- `x-class+p+a` 也会正确触发 class-stratified support plan；
- class representation loss 和 slot 相关路径统一兼容 target/context batch；
- 本地 `test_direct_student_outer.py`：`32 passed`；
- Cora 单 seed smoke：每类 gate 行数与新 synthetic rows 对齐，最终 materialization 复用 gate，拓扑 cap 和 memory rows 保持不变。

### 正式结果

协议为 Cora/Citeseer、`order19611/order19612`、6 个 report seeds、`p_old_context_ratio=0.25`、`p_old_context_cap=0.25`，只改变 P gate。

| 数据集 | 顺序 | Final Macro-F1 | Final Accuracy | 阶段 1 keep rate | 阶段 2 keep rate |
|---|---|---:|---:|---:|---:|
| Cora | order19611 | 52.135 | 59.717 | 52.9% | 0.0%
| Cora | order19612 | 50.468 | 58.303 | 34.3% | 0.0%
| Citeseer | order19611 | 55.884 | 61.546 | 100.0% | 0.0%
| Citeseer | order19612 | 54.911 | 60.920 | 0.0% | 47.2%

按类 gate 能识别局部阶段收益，但最终阶段只有 Citeseer `order19612` 保留了部分 old-context rows；Cora 两个顺序的最终阶段和 Citeseer `order19611` 最终阶段均全部回退。最终阶段的 gate 响应均值为：

| 数据集/顺序 | Final-stage `Delta_new` | Final-stage `Delta_old` | 解释 |
|---|---:|---:|---|
| Cora/order19611 | -0.00455 | +0.00368 | 新类略有改善，但旧类响应恶化，回退 |
| Cora/order19612 | -0.00520 | +0.00267 | 同上，回退 |
| Citeseer/order19611 | -0.00008 | +0.00050 | 新类收益不足且旧类变差，回退 |
| Citeseer/order19612 | -0.00254 | +0.00048 | 部分类满足 gate，保留率 47.2% |

结论：按类 gate 修复了全局 gate 过粗的问题，但没有证明 old-context transport 在当前 Student、P mix 和预算下具有稳定的跨阶段收益。因此它暂不晋升为主方法，也不继续扫描更多 gate ratio；应作为受控机制和负结果记录。当前证据更支持 pure-new-P 或基础 X-P-A 作为后续对照，而不是强行加入 old-context P。

## 9. 关键文件

- `code/v2_llm_fusion/direct_student_outer.py`
- `code/v2_llm_fusion/run_direct_student_outer.py`
- `code/v2_llm_fusion/run_gc_sr_stage_eval.py`
- `code/v2_llm_fusion/test_direct_student_outer.py`
- `results/direct_student_outer_20260929/`
- 远程按类 gate 结果：`/home/pc/research/TAG_condensation/results/xpa_perclass_gate_p2_20260930_cora/`
- 远程按类 gate 结果：`/home/pc/research/TAG_condensation/results/xpa_perclass_gate_p2_20260930_citeseer/`

## 10. 方法与参数的区分

本轮冻结的是**方法结构**，不是一组声称在所有图上最优的数字。方法结构固定为：

```text
Semantic-Recursive/C0 初始 memory
        ↓
合法 support = 旧 synthetic 前缀 + 当前 raw delta
        ↓
Student 可微 unroll
        ↓
联合更新 X、P、A
        ↓
半径约束、Sinkhorn 平衡、度守恒、每行 revision cap
        ↓
同一规则 materialize 为最终 synthetic memory
```

其中 `x_new_radius`、`p_mix`、`p_temperature`、`a_temperature` 和
`a_revision_margin` 是**数据集相关的控制量**。它们在 Arxiv 或更长的
base+4 任务上可以重新标定，但不改变上述算法的访问边界、梯度路径和
结构约束。本轮 pooled 小图结果中的 `margin=0.02` 只说明该控制在当前
小图上有正向信号，不能写成全局最优参数。

## 11. 代码级执行顺序

### 11.1 入口与合法数据边界

命令行入口只接收 `base`、`support` 和 `old_rows`。其中 `support` 已经由
调用方准备成“上一阶段 synthetic 前缀 + 当前 raw delta”，direct outer
本身不会重新读取历史 raw graph：

```python
base, support = load(args.base), load(args.support)
result, audit = DSO.optimize(
    base, support, args.old_rows, selected, args.mode, args.device, cfg,
    ...)
```

对应代码：
`code/v2_llm_fusion/run_direct_student_outer.py:141-218`。

`optimize()` 首先验证字段、建立输入哈希并创建 sealed train/validation
plan；完成后 validation 只从既定 split 中筛选，不会把 validation 行重新
放入 train。对应代码：
`code/v2_llm_fusion/direct_student_outer.py:1929-2038` 和
`direct_student_outer.py:426-536`。

### 11.2 可学习变量

当前主线的外层变量是：

```python
variables["x_class_relative"] = Parameter(zeros(...))
variables["p_logits"] = Parameter(_init_p_logits(...))
variables["a_logits"] = Parameter(_init_a_logits(...)[0])
```

代码中真正创建参数的位置是
`direct_student_outer.py:1516-1576`。因此：

- `X` 不是直接无界修改，而是每个可编辑新 synthetic row 的相对位移；
- `P` 不是分类器，而是 raw delta 到新 synthetic slots 的软传输 logits；
- `A` 不是重新生成整张图，而是候选跨块边上的选择 logits。

### 11.3 候选 memory 的生成

每次 outer objective 被调用时，`_candidate()` 按同一顺序生成候选图：

1. 由 `p_logits` 经过 Sinkhorn 得到 `P`，把当前 delta 的 `X` 和文本
   embedding 传输到新 synthetic rows；
2. 对可编辑新 rows 加上有界相对 `X` 位移；
3. 由 `a_logits` 生成带 straight-through 梯度的候选邻接矩阵；
4. 用候选 `X/T/A` 生成 Student 看到的归一化图。

对应代码：
`direct_student_outer.py:1426-1501`。

关键的 `P` 应用逻辑是：

```python
assignment = sinkhorn_assignment(p_logits, ...)
transported_x = assignment @ source_x
transported_text = assignment @ source_text
candidate_x[new] = (1 - p_mix) * base_x[new] + p_mix * transported_x
candidate_text[new] = (1 - p_mix) * base_text[new] + p_mix * transported_text
```

所以 `P` 只影响新 synthetic 后缀；旧 synthetic 前缀的 payload 不被
`P` 直接重写。

### 11.4 Student 内层和外层梯度

`unroll_student()` 每次从固定 seed 创建与下游一致的两层 GCN/InfoNCE
Student，并用可微 Adam 展开 `K` 步：

```python
loss = _nce(graph, encoded, rows, params["logit_scale"])
grads = torch.autograd.grad(
    loss, tuple(params.values()), create_graph=create_graph, ...)
params, first, second = _adam_step(params, grads, ...)
```

对应代码：
`direct_student_outer.py:271-389`。

`create_graph=True` 是关键。它保留了

\[
\frac{\partial \theta_K}{\partial X},\qquad
\frac{\partial \theta_K}{\partial P},\qquad
\frac{\partial \theta_K}{\partial A},
\]

从而 outer loss 可以继续得到：

\[
\nabla_X\mathcal L_{outer},\quad
\nabla_P\mathcal L_{outer},\quad
\nabla_A\mathcal L_{outer}.
\]

`_objective()` 的实际调用顺序是先生成 candidate，再 unroll Student，
最后在 sealed support 上计算旧/新响应和正则项：
`direct_student_outer.py:1662-1875`。

### 11.5 旧类保持与新类塑性

support loss 不是监督分类 loss，而是与当前任务一致的双向图文
InfoNCE：

\[
\mathcal L_{support}
=\lambda_o\mathcal L_{old}
+\lambda_n\mathcal L_{new}.
\]

实现位置为 `direct_student_outer.py:641-738`。当前结构让新 support
获得更大权重，同时通过旧 support 保留旧响应。新类额外可以在**同一
candidate graph**上使用 class-prompt margin：

\[
\mathcal L_{margin}
=\frac1{|B_{new}|}\sum_i
\max(0,m-s_{y_i}+\max_{c\ne y_i}s_c),
\]

实现位置为 `direct_student_outer.py:901-942`，并在
`direct_student_outer.py:1768-1782` 接入 outer objective。这里的 class id
是构造阶段 metadata，不是用来挑选 report/test checkpoint 的标签。

### 11.6 A 的候选池、软选择和最终二值图

对每个新节点，只从旧节点中建立有限候选池。候选池由图特征和文本的
联合相似度 Top-K 加上原有跨边构成，因此不需要遍历全部旧节点：

```python
score = cosine(new_x, old_x) + cosine(new_t, old_t)
candidate = topk(score, a_pool)
candidate += original_cross_neighbors
```

对应代码：`direct_student_outer.py:1196-1241`。

训练时 `A` 采用 straight-through 选择：forward 使用 hard top-k 的
二值边，backward 使用 softmax 权重的梯度。若设新节点原跨块度数为
`d_i`，则软权重总质量仍为 `d_i`；加入 self-loop 后统一使用

\[
\hat A=D^{-1/2}(A+I)D^{-1/2}.
\]

对应代码：`direct_student_outer.py:1244-1388`。

当前受限主线使用：

\[
r_i\le 1,
\]

即每个新 synthetic row 最多替换一条原跨边，并且保持该行原有跨块度数。
最终边由同一个 gate 再次 materialize，避免训练图和落地图使用不同规则：
`direct_student_outer.py:1391-1423`。

### 11.7 外层更新、验证和审计

外层使用 Adam 更新 `X/P/A` 参数；每一步都检查 loss 和梯度是否有限，
然后对 `X` 应用半径投影：

```python
loss.backward()
optimizer.step()
_project_x(...)
```

对应代码：`direct_student_outer.py:2067-2137`。

外层 checkpoint 只由 sealed construction validation loss 选择，随后使用
最佳变量重新 materialize candidate，并执行不可变字段、输入哈希、预算
和有限位移检查：`direct_student_outer.py:2193-2260`。

最终输出至少包括：

- `memory.pt`：新的 synthetic memory；
- `audit.json`：输入哈希、外层变量、梯度范数、revision 数和位移；
- `protocol.json`：Student、访问边界和所有结构配置。

## 12. 当前方法的一句话定义

在不回读历史完整 raw graph 的条件下，以旧 synthetic memory 为锚点，
用同一个可微 Student 训练轨迹联合优化新 synthetic payload、当前 delta
到新 slots 的语义传输以及少量新旧跨块关系，并通过半径、质量、度数和
revision cap 将连续优化结果约束为可审计的离散持续 memory。

这一定义比“某个参数设置最好”更重要；后续大图只需重新标定控制量，
不需要重新发明方法结构。
