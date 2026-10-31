# 经典持续文本属性图凝聚：方向重置与实验路线

日期：2026-09-04  
状态：新主线设计冻结；尚未宣称新方法已经成立

## 1. 方向修正

此前研究逐步收敛成了“持续 prototype/compressed replay memory”，这与最初预期的经典 graph condensation 存在偏差。新的主线明确回到：

> **直接优化一个小型 synthetic text-attributed graph，使在该 synthetic TAG 上训练的模型尽量复现累计真实 TAG 上的训练轨迹和下游性能，并在时间阶段之间持续更新 synthetic graph。**

目标 synthetic graph 为：

\[
\widetilde G_t=
(\widetilde X_t,\widetilde A_t,\widetilde Y_t,\widetilde T_t,\widetilde W_t),
\]

其中：

- `X~`：可学习 synthetic node/graph feature；
- `A~`：可学习稀疏 synthetic adjacency；
- `Y~`：可学习 soft label/class mass；
- `T~`：可学习 synthetic text state；
- `W~`：sample/support weight。

这与 D2 的根本差别是：D2 主要选择或递推真实锚定的训练 memory；新方法的 `X~/A~/Y~/T~` 都是凝聚变量，目标由训练等价性决定，而不是要求每一行对应一个真实节点。

## 2. 选题表述

中文：

> **面向演化文本属性图的经典持续图凝聚**

更完整的研究问题：

> 在历史原始 TAG 不可持续回放、仅保留上一阶段 synthetic TAG 的条件下，如何联合优化节点特征、软标签、文本语义和图结构，使固定压缩率的小图持续保持累计图文训练轨迹、当前任务性能和历史知识？

英文工作表述：

> **Continual Graph Condensation for Evolving Text-Attributed Graphs with Jointly Optimized Synthetic Graph-Text States**

当前不提前命名算法。只有静态经典凝聚和持续重凝聚均通过后，再确定方法名。

## 3. 与已有工作的区别

### 已有近邻

- PUMA/CaT：已经研究 continual graph condensation 和 condensed replay；
- GECC：已经研究 evolving graph condensation、centroid inheritance 和 assignment；
- TAGSAM：已经研究静态 TAG condensation 和 text selection/attribute matching；
- DANCE：已经研究动态 TAG 中的 neighbor gate、text evidence pack 和 provenance；
- GraphCLIP/ConGraT：已经研究 graph-text contrastive representation；
- MCond、TACO、OpenGC：分别覆盖 mapping、continual coarsening 和 open-world condensation。

因此不能把“持续”“TAG”“保存文本”或“学习 synthetic nodes”任一单项写成独立创新。

### 预期差异

新主线的候选贡献必须由以下完整组合形成：

1. 经典 synthetic TAG 变量联合优化，而非 prototype selection；
2. 每阶段将“旧 synthetic TAG + 当前真实增量”重新凝聚成同预算的新 synthetic TAG；
3. 用 temporal transport 显式连接前后两代 synthetic nodes，控制递归误差和遗忘；
4. synthetic text 不只是附带字段，而是进入图文训练轨迹与配对损失；
5. synthetic adjacency 是优化变量，并通过 matched self-loop/shuffled/frozen controls 验证；
6. 固定压缩率与真实 feature/text/label/edge/state bytes 同时报告；
7. 最终在 GraphCLIP/TAG-native 任务上验证，在 GraphSAGE 等模型上做迁移诊断。

如果 temporal transport、graph-text trajectory 和结构变量中没有至少两个获得独立因果支持，算法创新仍然不够。

## 4. 为什么此前 O2 不等于这条新路线

O2 已经尝试过自由 synthetic 优化，但它不是可复用的成功实现：

- 只在 full-history 静态 screening 中运行，不是阶段式 continual re-condensation；
- synthetic topology 固定为 evidence-induced，没有联合学习 `A~`；
- 使用两个 initialization、3 个 inner steps、20 个 outer steps，训练轨迹过短；
- outer objective 只针对一个 checkpoint partition；
- 最终 qualification 使用重新初始化、完整训练的 student，与 outer 优化目标错位；
- 放开 `X/Z/Q` 后自由度增加，但没有跨 checkpoint、跨初始化的稳定约束。

结果表现为严重过拟合：

- ArXiv `X-only` 相对 frozen centroid `-12.60 pp`；
- PubMed 最好 O2 arm 仍相对 frozen centroid约 `-8.87 pp`；
- ArXiv checkpoint Macro-F1 可升到 `81%+`，独立 qualification 却下降到 `10%--29%`。

新路线不是重跑 O2，而是同时改变四件事：

1. 从单 checkpoint response 改为多初始化、多 checkpoint 的训练轨迹匹配；
2. 从静态 full-history 优化改为 `old synthetic + current delta -> new synthetic`；
3. 从固定 topology 改为可学习、受稀疏约束的 synthetic adjacency；
4. 加入前后阶段 synthetic state 的 transport/retention 约束。

## 5. 新算法的理论骨架

### 5.1 递归重凝聚目标

阶段 `t` 只读取：

\[
\widetilde G_{t-1},\quad \Delta G_t.
\]

先构造允许访问的 teacher replay union：

\[
U_t=\widetilde G_{t-1}\cup\Delta G_t.
\]

再将它重新凝聚为固定预算的新 synthetic TAG：

\[
\widetilde G_t
=\operatorname{Condense}(U_t;K_t).
\]

这使 synthetic graph 本身成为唯一历史数据状态，符合持续图凝聚而不是普通 replay buffer 扩容。

### 5.2 多轨迹任务保持

对多个模型初始化 `r` 和训练 checkpoint `j`，分别在 `U_t` 与 `G~_t` 上展开相同优化器，匹配参数更新或预测轨迹：

\[
\mathcal L_{traj}
=\sum_{r,j}
d\left(
\theta_{U_t}^{r,j},
\theta_{\widetilde G_t}^{r,j}
\right).
\]

第一版优先采用归一化参数更新与 held-out training state prediction，而不是再次只匹配单步梯度。多个 checkpoint 用来约束 synthetic graph 对完整训练过程的影响，多个初始化用来降低对某一个 student 的过拟合。

### 5.3 图文联合状态

Synthetic text 第一版不直接生成自然语言，而优化固定 text encoder 可消费的 soft prompt/token embedding：

\[
\widetilde T_t
=E_{text}(P_t),
\]

其中 `P_t` 是受长度和范数约束的 synthetic soft prompt。与任意自由 latent 相比，它保持在冻结 text encoder 的可消费接口内。

图文对应由对比目标约束：

\[
\mathcal L_{pair}
=\operatorname{InfoNCE}
\left(
E_g(\widetilde X_t,\widetilde A_t),
E_t(\widetilde T_t)
\right).
\]

必须保留 shuffled text、zero text 和 frozen medoid text 对照，只有 true synthetic pairing 带来独立任务收益时才能声称 TAG-specific 贡献。

### 5.4 可学习稀疏邻接

不直接优化一个稠密 `K x K` 矩阵。使用 graph-text conditioned edge logits：

\[
s_{ij}=g_\phi(\widetilde x_i,\widetilde x_j,
\widetilde t_i,\widetilde t_j),
\]

再通过固定 edge budget 的 differentiable top-k/Gumbel mask 得到 `A~`。结构正则只承担稀疏性、连通性和度分布约束，最终是否有效由 downstream 与 shuffled/frozen topology 对照决定。

### 5.5 Temporal transport

为防止每一阶段自由优化导致历史 synthetic identity 漂移，引入旧新 synthetic nodes 的软运输矩阵：

\[
\Pi_t\in\mathbb R^{K_{t-1}\times K_t},
\]

并约束 feature、label、text 和局部结构质量的运输误差：

\[
\mathcal L_{transport}
=D_X(\Pi_t\widetilde X_t,\widetilde X_{t-1})
+D_Y(\Pi_t\widetilde Y_t,\widetilde Y_{t-1})
+D_T(\Pi_t\widetilde T_t,\widetilde T_{t-1})
+D_A(\Pi_t\widetilde A_t\Pi_t^\top,\widetilde A_{t-1}).
\]

该项不是要求旧图完全不变，而是在当前增量适应与历史保持之间形成可优化的连续约束。其必要性必须通过 no-transport、identity transport 和 shuffled transport 消融验证。

### 5.6 总目标

\[
\mathcal L
=\mathcal L_{traj}
+\lambda_p\mathcal L_{pair}
+\lambda_t\mathcal L_{transport}
+\lambda_s\mathcal L_{sparse}
+\lambda_a\mathcal L_{anchor}.
\]

`L_anchor` 只在早期用于限制优化发散，可由 D2 medoid/moment 初始化提供。最终方法不能依赖人工固定 5%/2% residual 或固定 role fusion。

## 6. D2 如何进入新方法

D2 的有效部分可以复用，但不再是算法主体。

### 初始化

- coverage medoid 初始化 `X~` 和真实文本/soft prompt；
- moment statistics 初始化 `Y~`、sample weight 和类别质量；
- provenance 初始化旧新 synthetic transport 的候选对应。

### 稳定对照

- D2/A2 作为最强 compressed-memory baseline；
- D2 initialized but frozen 作为“初始化本身”的控制；
- random/TAGSAM/Q0 initialization 用于判断最终收益是否只来自 D2 起点。

### Rollback

若某阶段重新凝聚后 controller 指标下降，允许 no-op，保留上一阶段 synthetic graph 或 D2 anchor。该机制必须预注册，不能在 qualification 上事后选择。

### 不继承的部分

- 固定 `alpha=0.25` 双专家融合；
- 固定 moment/coverage 50/50 role；
- self-loop-only 作为默认最终拓扑；
- 将完整 medoid text 仅作为未参与优化的 payload。

## 7. 实验推进顺序

### Gate C0：静态经典 TAG 凝聚能否成立

目的：先验证优化器能够在单一 t1 静态 TAG 上产生有泛化能力的 synthetic `X/Y/A/T`，而不是重复 O2 的 checkpoint 过拟合。

固定：

- 数据先用 ArXiv、PubMed t1；
- 压缩率 `1%`，补充 `0.5%/2%`；
- 同一 train/checkpoint/qualification 切分；
- condensation seeds 44、45 做 screening，通过后扩到 44--48；
- qualification 只在 synthetic graph 和 checkpoint 完全冻结后读取。

比较：

1. Random/reservoir；
2. online/static k-medoids；
3. Q0 synthetic centroid；
4. released TAGSAM；
5. D2 initialized frozen；
6. trajectory optimize `X+Y`；
7. trajectory optimize `X+Y+T`；
8. trajectory optimize `X+Y+A+T`；
9. shuffled text、shuffled adjacency、random initialization controls。

判断重点：synthetic optimization 必须同时改善独立 qualification 与正常重训练结果；outer/checkpoint 上升但 qualification 下降立即判为 O2 型失败。

### Gate C1：递归 teacher union 是否可用

目的：在正式实现 continual optimizer 前，验证：

\[
\widetilde G_{t-1}\cup\Delta G_t
\]

是否能近似 full cumulative `G_(<=t)` 的训练信号。

比较 full cumulative、old synthetic + delta、D2 + delta、old prototype + delta。若 teacher union 本身远低于 full history，后续凝聚器不可能修复丢失的信息；应先改历史 synthetic state，而不是调 outer optimizer。

### Gate C2：持续重凝聚

只在 C0/C1 通过后执行：

1. 以上一阶段 `G~_(t-1)` warm start；
2. 为新类别或漂移区域临时增加候选 slots；
3. 在 `G~_(t-1) + Delta G_t` 上生成 multi-trajectory teacher；
4. 联合优化 `X~/Y~/A~/T~` 回到固定 `K_t`；
5. 用 transport 保持历史质量；
6. qualification 前冻结 synthetic graph。

比较 Append、PUMA/CaT、GECC、online k-medoids、D2/A2、无 transport、无 text、固定 adjacency 和 full-history static condensation oracle。

### Gate C3：TAG 与图结构因果归因

在相同 root、parameter、bytes 和 compute 下比较：

- learned text vs frozen medoid text vs zero text vs shuffled text；
- learned adjacency vs self-loop vs frozen projected adjacency vs degree-matched shuffled adjacency；
- learned soft label vs hard label；
- temporal transport vs no transport vs shuffled transport。

必须至少证明 graph-text pairing 和 synthetic adjacency 中的一项对主任务有稳定独立贡献；否则只能称 feature/label dataset condensation，不能称完整 TAG graph condensation。

### Gate C4：持续与泛化

- ArXiv、PubMed 加第三个 temporal TAG，优先 ogbn-mag paper-citation 子图；
- 至少 5 个时间阶段，避免只在 t1/t2 上拟合更新规则；
- 方法冻结后打开 t3；
- GraphCLIP/TAGSAM-style dual encoder作为 TAG-native 主任务；
- GraphSAGE、GCN 作为 transfer diagnostic，不要求全部提高；
- 报告 current、retention、forgetting、MRR、Recall@K、Macro-F1、BA、Accuracy 和真实成本。

## 8. 主模型选择

新方法第一阶段优先针对一个明确 backbone，而不是要求所有 backbone 同时提升。

建议主接口为 GraphCLIP 或 TAGSAM-style graph-text dual encoder，原因是 paired/shuffled 实验已经证明它们会消费 graph-text correspondence。GraphSAGE 继续作为监督迁移和分类诊断，但当前 direct text concat/adapter 在两个数据集上不稳定，不适合单独承担 TAG-specific 主 claim。

如果 GraphCLIP 的收敛受控 full-to-memory gap 仍然不存在，则应：

1. 先换为非饱和 future retrieval/class-held-out 任务；
2. 再换用 ogbn-mag 等更自然的 TAG；
3. 仍无 gap 时，选择 TAGSAM-style contrastive encoder作为 condensation target，而不是回到纯 graph-only GraphSAGE。

## 9. 成功标准与研发停止规则

这些标准用于决定是否继续投入，不是要求每个模型、每个 seed 绝对提高。

### C0 可行

- 至少一个完整 synthetic arm 在 ArXiv、PubMed 的独立 qualification 上均不低于最强 static prototype baseline；
- 至少一个数据集有清楚的 `>=0.5 pp` task gain；
- checkpoint 与 qualification 方向一致；
- true text 优于 shuffled/zero，或 learned adjacency 优于 shuffled/self-loop。

### C2 方法成立

- 主 TAG-native backbone 在两个数据集均不低于 D2/PUMA/GECC 中的最强基线；
- 多数 condensation seeds 同方向；
- old-stage retention 不出现系统性下降；
- 历史 raw graph/text 读取为 0；
- fixed-rate 与实际 bytes 均可审计。

### 停止或切换

- C0 仍出现 O2 型 checkpoint/qualification 反转：停止当前 trajectory objective，先修静态凝聚；
- C1 teacher union 明显低于 full cumulative：先改历史 synthetic state，不进入 C2；
- text 和 adjacency 消融均无贡献：方法不再声称 TAG/full graph condensation；
- 两个数据集方向完全相反：不训练通用 controller，先诊断任务或数据饱和；
- t3 失败：不得打开 t3 重新调参。

## 10. 第一轮具体工程任务

1. 新建独立结果目录和 protocol schema，保持旧 artifact 只读；
2. 将现有 O2 runner 拆成可复用的 differentiable inner-loop、trajectory recorder 和 independent evaluator；
3. 实现 multi-initialization、multi-checkpoint trajectory objective；
4. 先只优化 `X+Y`，验证不再发生 qualification 崩溃；
5. 通过后加入 soft-prompt `T`，再加入 sparse `A`，每次只增加一个变量；
6. 添加 D2/Q0/TAGSAM/random 四种初始化和 matched controls；
7. 完成 C0 seed 44/45 screening 报告；
8. C0 通过后再实现 temporal transport 和 C1/C2。

第一轮不做：动态预算、更多 role、复杂 selector、自然语言生成、所有 backbone 联合优化和 t3。

## 11. 当前方向的一句话定义

> **冻结 D2 训练记忆成果，以它作为初始化和强基线，重新回到直接联合优化 synthetic feature、soft label、text state 与 sparse adjacency 的经典图凝聚，并通过“旧 synthetic TAG + 当前增量 -> 新 synthetic TAG”的递归重凝聚和 temporal transport 形成持续 TAG 特有的算法贡献。**
