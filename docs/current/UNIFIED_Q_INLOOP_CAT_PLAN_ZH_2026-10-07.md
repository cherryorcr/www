# Unified-Q 同循环 CaT 分布匹配：实现与受控验证

## 范围

用户要求借鉴 CaT 的优化目标，直接在 Q 生成 X 的过程中优化，而不是先求 CaT-X 再做 anchor。本轮仅实现这个机制，旧实验结果和服务器数据不删除、不覆盖。

本轮不是最终方法冻结，不改变 Student、类别到达协议、5% fixed-rate、Student seeds 150/151/152、2048 steps、batch 24、MiniLM、320 字符、top-1 文本硬前向。所有实验仍为 report-dev，不是最终封存测试，也不是 unseen-class zero-shot 证据。

## 同一个优化循环

每个阶段仅给构造器当前 raw delta 和本方法的旧 synthetic memory。Q 仍由类别掩码 Sinkhorn 产生，X 仍为：

\[
X_s=D_Q^{-1}Q^\top X_\Delta+R.
\]

R 保留原来的每行相对当前聚合特征 5% 半径，Q 初始化、学习率、30 个外层步骤及数值容差均不改变。并未凭空恢复自由 X 的全部表达能力。

新增真实侧目标是合法参考图的当前节点一跳消息：

\[
H_r=(\widehat A_{ref}X_{ref})_{current}.
\]

合成侧也使用当前合成节点一跳消息：

\[
H_s=(\widehat A_sX_s^{full})_{current}.
\]

edge-free 时合成节点只有自环，H_s 等于 X_s，与 CaT 的“传播后真实特征对自由合成 X”分布目标对应。有拓扑时匹配传播后的 H_s，避免既把传播后的语义压到 X 又额外传播一次。旧节点冻结；当前不添加没有梯度作用的旧记忆 MMD 项。

随机映射沿用 CaT 适配实现：phi_W(H)=normalize(ReLU(HW))。使用本阶段固定的 4 个 512 维随机 Encoder，在每个映射空间内对类别均值作平方距离，类别与特征维求和，Encoder bank 求平均：

\[
L_{DM}=\frac{1}{4}\sum_b\sum_c
\left\|\operatorname{mean}_{i\in\Delta_c}\phi_{W_b}(H_{r,i})-
\operatorname{mean}_{j\in S_c}\phi_{W_b}(H_{s,j})\right\|_2^2.
\]

W 不训练；真实侧类别统计不保留梯度，只由合法输入计算。随机 bank 使用独立 Generator，不改变旧 GM proxy 的初始化随机数。另一组独立 bank 仅记录起终点误差，不参与优化、选择或提前停止。真实侧按 4096 行分块计算统计量，不保存 n-by-n kernel。

在原目标上只增加 lambda_distribution=1 的 L_DM。梯度沿 H_s -> X_s -> Q logits/R 回传；有图时同时有 H_s -> A_s(Q) -> Q 的路径。没有调用 cat_block，也不读取预训练 CaT-X。传入 cat_anchor_x 会明确报错。

需注意：原始特征的线性类别均值在均衡 Q 下本来就守恒，不能作为有效的 Q 学习目标。非线性 Encoder 是必要部分。Sinkhorn 只保证质量/容量，不保证各 slot 内容不同；本轮不额外叠加多样性损失，以免混杂归因。

## 对照

| 臂 | 拓扑 | 相对匹配对照唯一改变 | 用途 |
|---|---|---|---|
| uq_v3 | 原 Q 诱导与保度换边 | 无 | 完整旧对照 |
| uq_dm_joint | 同 uq_v3 | 增加 L_DM | 完整拓扑条件下的单因素实验 |
| uq_edgefree384 | 自环 | 无新增 DM，384 维 proxy | 无拓扑匹配对照 |
| uq_dm_edgefree | 自环 | 增加 L_DM | 本轮最小候选 |
| uq_dm_edgefree_no_gm | 自环 | 相对候选去掉 GM | GM 必要性；仍保留传播与原正则 |
| uq_dm_edgefree_fixed_q | 自环 | 相对候选冻结 Q | 检查分配学习作用 |
| CaT-CCTP | 自环 | 原 CaT 自由 X + top-5 文本 | 外部适配对照，不称官方 TAG 方法 |

有拓扑对照与无拓扑对照分开比较，不能将多个改动的收益全部归给 DM。只有通过预算、划分、顺序、Student 和基线实现哈希核对的历史结果才能复用。

## 验证顺序

1. 本地和远程单元测试：CaT 公式等价、Q/R 非零梯度、一二阶有限、无预训练调用、原模块禁用行为、旧前缀和硬材料化一致。
2. WikiCS 参考顺序、单构造种子、三个 Student seeds、五阶段初验。
3. 同一冻结配置扩展 SportsFit、BookHis，并补无 DM 与固定 Q 控制。
4. 先报告全体预定臂，不按每图最好结果重新命名方法；只有出现明确正信号后再规划独立顺序和构造种子确认。

本轮外层步数、残差半径、损失权重不扫参。若分类不改善但 DM 下降，应如实报告目标与下游性能的差距，不能只用构造 loss 宣称优化成功。

## 代码

- code/v2_llm_fusion/unified_q_multimodal.py：distribution target/loss 与循环内联合梯度。
- code/v2_llm_fusion/run_unified_q_inloop_cat_20261007.py：明确的单因素实验配置。
- code/v2_llm_fusion/test_unified_q_distribution.py：新增机制、梯度与不变量测试。

新增参数默认关闭，历史实验没有被重新解释成已使用 DM。远程使用独立 experiments/unified_q_inloop_cat_20261007 快照与 results/unified_q_inloop_cat_20261007 输出目录。
