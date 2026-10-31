# Edge-free In-loop CaT-Q：实际实现、外部比较和组件证据

日期：2026-10-08。主方法配置仍是 `uq_dm_edgefree`；本轮不更换 Student、不扫描超参数、不使用封存 test 选方法。

完整数值、方差、配对区间、Old/New F1、成本和可视化由可复核脚本生成：

- [统一结果表](D:/Aprojects/continue_tag_condensation/results/unified_q_matrix_20261008/summary.md)
- [结构化汇总](D:/Aprojects/continue_tag_condensation/results/unified_q_matrix_20261008/summary.json)
- [逐 cell 指标](D:/Aprojects/continue_tag_condensation/results/unified_q_matrix_20261008/cells.csv)
- [预注册范围](D:/Aprojects/continue_tag_condensation/docs/current/UNIFIED_Q_BASELINE_ABLATION_PLAN_ZH_2026-10-08.md)

## 1. 先澄清两个实现问题

### 1.1 跨类别的 -10000 不会进入加权平均

要区分可训练分数矩阵 $Z$ 与实际分配矩阵 $Q$：

$$
Q=\operatorname{ClassMaskedSinkhorn}(Z).
$$

`_initial_logits` 把不同类别位置的 **Z** 初始化为 `-1e4`。真正计算时，`_masked_sinkhorn` 先建立全零 Q，只分别读取每个类别的合法子矩阵进行 Sinkhorn，再填回这些类内块。

因此不是把负数 -10000 当成聚合权重，也不是依赖整个矩阵 softmax 把它“变得很小”：

$$
y_i\ne y_j\Longrightarrow Q_{ij}=0,
\qquad \frac{\partial L}{\partial Z_{ij}}=0.
$$

测试还把跨类 Z 人为改成正的 `+1e4`，验证 Q 和对应梯度仍然为零。安全性来自类内取块，而不是哨兵值大小。

对类别 $c$，当前 raw 数为 $n_c$、slot 数为 $m_c$。Sinkhorn 约束为行和 1、列和 $n_c/m_c$，误差按浮点容差审计。合成特征只聚合同类节点：

$$
d_j=\sum_iQ_{ij},\qquad
X_j^{pool}=\frac{\sum_iQ_{ij}x_i}{d_j},\qquad
X_j=X_j^{pool}+R_j.
$$

相关实现：[类内 Sinkhorn](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py:192)、[初始化分数](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py:312)。

### 1.2 文本“硬前向、软反向”到底优化什么

每个合法当前 raw 文本先清理、最多保留 320 字符，再由冻结 MiniLM 编码，得到可真正保存的候选 $e_i$。不是任意生成一个无对应字符串的 embedding，也不是每个类别共用一条文本。

对 slot $j$：

$$
T_j^{soft}=\operatorname{Normalize}\left(\frac{\sum_iQ_{ij}e_i}{d_j}\right),
\quad i_j^*=\arg\max_i Q_{ij},\quad T_j^{hard}=e_{i_j^*}.
$$

训练中使用：

$$
T_j^{ST}=T_j^{hard}+T_j^{soft}-\operatorname{stopgrad}(T_j^{soft}).
$$

前向时后两项抵消，模型实际接收真实候选文本的 embedding。反向时 `stopgrad` 不求导，所以采用软聚合的导数去更新 Q 的 logits Z。它不是对字符串字符求导，不更新 MiniLM，也不是对离散 argmax 求得了精确导数。

在当前带 GM 的主方法中，反向链更准确地写成：

$$
L_{GM}\to g_s=\nabla_\theta L_{mm}\to T^{ST}
\overset{\text{代理梯度}}{\longrightarrow}T^{soft}\to Q\to Z.
$$

同时还存在 $L_{GM}\to X\to(Q,R)$ 和 $L_{DM},L_{prop}\to X,Q$ 的路径。优化器更新的是 Z 与 R，而非用于计算匹配目标的代理模型参数。合成侧 `autograd.grad(..., create_graph=True)` 保留二阶/混合导数图，之后 outer loss 的 `backward()` 才能回传到数据参数。

这是 straight-through 有偏梯度估计，不保证软目标下降一定改善硬选择或最终分类。因此必须保留 stop-text-gradient 对照，不能仅凭“用了梯度”声称有效。

相关实现：[候选重编码](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py:671)、[ST 前向](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py:795)、[GM 与反向](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py:833)。

## 2. 当前主方法真正用了什么

| 内容 | 实际状态 | 准确表述 |
|---|---|---|
| CaT 先优化好 X 再做修补 | 未使用 | 没有 CaT warm start，也没有预训练 CaT X anchor |
| CaT 风格 class-wise 分布匹配 DM | 启用，系数 1 | 在 Q 生成 X 的同一个循环内优化，不是两段分离训练 |
| 多模态梯度匹配 GM | 启用，系数 1 | 对固定随机初始化的图文 proxy 做梯度方向及范数匹配 |
| 可学习 Q | 启用 | class mask + capacity-balanced Sinkhorn；Adam 更新 Z |
| 特征 residual R | 启用 | Adam 更新；每行限制在当前 pooled X 范数的 5% |
| 真实文本 ST routing | 启用 | 一 slot 一次 top-1 硬选择；不同 slot 可以选不同文本，不保证绝不重复 |
| 传播一致性 | 启用，系数 0.5 | 当前 raw 的局部消息聚合与合成 X 方向对齐，使用 cosine |
| Q-anchor | 启用，系数 0.1 | 限制 Q 偏离 Q0 |
| R penalty | 启用，系数 0.1 | 归一化残差平方惩罚，与硬 5% 投影约束不是同一个操作 |
| 旧 memory 更新、old anchor | 未使用 | 旧 X/text/labels/边前缀冻结；保留系数不代表该项非零 |
| 新旧交互 | 仅损失层面 | GM 参考中包含旧 synthetic 与当前 raw；没有新旧拓扑边，也不更新旧 X |
| Q 诱导拓扑、boundary sketch | 未使用 | 当前构造与保存均无新增非自环边 |
| EM、交替优化 | 未使用 | 当前是 joint Adam 更新 Z/R，不应称为 EM |

实际总目标为：

$$
L=L_{GM}+L_{DM}+0.5L_{prop}+0.1L_{Qanchor}+0.1L_{Rpenalty}.
$$

DM 匹配的是当前 raw 的一跳传播表示与合成表示经固定随机 Encoder bank 后的逐类均值。GM 的图文任务损失包含多正样本图文对齐、类别 prompt CE 与 prompt margin。DM 是分布/矩匹配；GM 才是显式模型梯度匹配；两者不能混称。

edge-free 不等于没有利用原图结构：初始化使用邻域消息，DM/propagation 使用当前 delta 的 raw 邻接。它意味着最终合成 memory 不存储额外的非自环边，Student 在 memory 上训练后，评价时在完整原图上传播。

配置入口：[固定主方法配置](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_unified_q_inloop_cat_20261007.py:18)、[本批消融映射](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_unified_q_matrix_20261008.py:56)。

## 3. 基线与消融必须分开

### 3.1 外部方法适配

| 原方法或来源 | 本实验名称 | 保留与改动 | 不能声称什么 |
|---|---|---|---|
| CaT | CaT-CCTP adapter | 保留 class-wise CGM/MMD 优化自由 X；改为当前 delta，旧 synthetic 冻结；补 top-5 真实文本与公共 Student | 不是原论文的 TAG 结果；不同 CCTP 文本版本不是不同论文基线 |
| GECC | GECC-hard TAG adapter | 保留多跳传播、hard Lloyd/restarts；当前 delta 聚类、补真实文本、统一预算 | 不是完整 fuzzy GECC 或原累计 raw 协议 |
| PUMA | PUMA-inspired TAG adapter | 一跳传播、teacher 伪标签、置信过滤与凝聚；在合法 pool 内留出子集模拟未标注数据，再附着文本 | 监督分配有改动，不能当官方 PUMA 原样复现 |
| TAGSAM | TAGSAM-response adapter | 调用固定官方 commit 的 wBCELoss；response unroll；当前 delta 与 synthetic parent 训练 teacher；冻结前缀，统一最终 Prompt Student | 构造 proxy/初始化/预算/Student 改动较大，不是官方完整流水线 |
| 通用子集方法 | Random coreset | 同类别配额，随机保留真实 X/text | 不是梯度凝聚方法 |
| 通用覆盖方法 | Graph-text k-center | 同类别配额，联合图文空间 farthest-first 选择真实节点 | 不是一篇独立 TAG 持续凝聚论文 |

适配比较回答的是：**在当前受限协议与当前 Student 下，这些明确说明过的构造方案表现如何。** 不能把结果写成“超过原论文所有官方版本”。尤其 GECC-hard 与 PUMA-inspired 后续仍需更高保真的官方机制确认。

不能公平适配的官方静态方法可以另做 native/offline reference，但要标明其历史 raw 访问、编码器、Student、预算与评价差异。没有成功完成的官方运行不填数、不用 0 代替、不混入持续排名。

### 3.2 内部组件消融

本批十行：full、去 DM、去传播项、去 R、固定 Q0 文本、切断文本反传、去文本初始化、去 Q-anchor、去 R penalty、DM-only。每个配置在三图采用同一规则，不按数据集各挑最优。

固定 Q0 文本会同时改变前向选择；切断文本反传仍允许当前 Q 改变硬文本选择。二者分别回答“动态 routing 是否有用”与“文本反向路径是否有用”，不可混为同一个消融。

DM-only 同时删多个损失，只能证明整体简化方案的表现，不能单独归因某一项。已有 no-GM 三顺序乘三构造 seeds 结果单列复用；它仍有 DM/传播目标对 Q/R 的梯度，不是“无梯度方法”。

### 3.3 哪些功能可能重叠，哪些并不等价

DM 对类别内 slot 排列基本不敏感，它约束的是每个类别在随机非线性映射后的总体均值；传播项则要求每个 slot 对齐其自身 Q 所聚合的 raw 消息方向。二者有共同的信息来源，但约束粒度不同，不能在数学上直接说是同一损失。

Q-anchor 限制分配关系相对初始化的改变；R penalty 和 5% trust region 都限制残差，但前者是可被其它目标权衡的软代价，后者是每步投影的硬边界。去 R penalty 的实验仍保留硬边界，不能说完全取消了残差约束。

GM 与 DM 都可能鼓励类别可分表示，但 GM 额外经过固定随机 proxy 的模型梯度，还受图文代理任务和 ST 梯度误差影响。即使 GM 数值下降，也不能推出最终 Student 的泛化必然变好。

因此应按证据做减法：组件已启用、优化目标下降、最终 F1 提升是三个不同判断。独立删项的影响也不能直接相加；组合简化必须实际重跑，再以独立顺序确认。

## 4. 配对协议与证据边界

- 三个数据集：WikiCS、SportsFit、BookHis；均为 base+4、相同 pool/report-dev/stage IDs。
- 主对比固定参考预算轨迹：WikiCS `[247,324,356,379,410]`；SportsFit `[3817,4486,4812,5305,6057]`；BookHis `[579,592,601,1429,1454]`。
- 顺序改变而预算固定，因此不能称为每个顺序都严格 5% fixed-rate。
- 外部主要表：三个顺序 9101/9102/9103，构造 seed 271006，每 cell 三个 Student seeds 150/151/152。
- 消融：参考顺序和一个构造 seed，每个 arm 重建完整五阶段链。与本批 full 比较，不拿旧同 seed 结果当完全确定的重放。
- 同一 Prompt Student、2048 steps、batch 24、MiniLM；每阶段 Student 重新训练。构造方法自身所需计算量不强行相同，另报成本。
- 预测采用类别 prompt，不训练测试分类器；这些结果不是严格未见类别或跨数据集 zero-shot。
- 全部为 report-dev。类别 labels 在构造/训练时已使用；报告中不称封存 test，也不据这些结果反复调参。
- 构造函数只接收当前 delta 与旧 synthetic；整个评估进程仍持有原图用于 transductive 推理。访问声明不是 OS 级隔离追踪证明。
- 外部表 SD 来自顺序均值；消融 SD 来自条件于某次构造的 Student seeds。两者不可混用，CI 跨零也不代表严格等效。

## 5. 归档与已知限制

远程磁盘紧张，本批在 RAM 中保留 memory，只落盘指标、配置、来源 hash 和审计，不删除旧结果。hash 不能恢复 memory，也不能重放任意新 Student；论文最终封存需完整 checkpoint、候选来源与环境信息。

首次 WikiCS 去 Q-anchor 在 assignment 误差检查处停止，原错误日志保留，在新目录补齐未完成 arm。全类内求解容差仍为 1e-4；只对全矩阵零填充归约允许 8 个 float32 epsilon，并记录实际误差。原失败未保存数值，不能断言其根因已被精确测定。不是按 F1 重试。

同 seed CUDA 构造还存在未彻底消除的重放差异，所以新 full 与旧 full 不当作同一个确定性样本，也不把新重放加入旧多 seed 均值。

官方 TAGSAM/STAD 原生流水线尚未完成。TAGSAM native 需要对应输入/teacher 与不同评估协议；STAD 涉及外部 LLM 生成配置，未向外部 API 发送数据。本批 TAGSAM-response 成功也不能填补官方完整原生证据。
