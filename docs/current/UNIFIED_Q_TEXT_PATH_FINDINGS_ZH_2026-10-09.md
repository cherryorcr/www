# In-loop CaT-Q：文本梯度诊断、单一硬投影修改与独立确认

实验开始于 2026-10-08，2026-10-09 完成。诊断与 12/12 个独立确认组合均已结束，48 份最终 memory 归档审计通过。结论：不将 Q-medoid 升级为主方法；保留原 In-loop CaT-Q 梯度主线，保留本次实现和混合/负结果，不按数据集或 Student 选择不同版本。

## 1. 范围与决策边界

遵循预注册顺序：先观察原主方法，不以新分类成绩选择修改；再只替换一个硬文本投影规则；最后与原主方法、CaT、GECC 做同协议比较，并将相同最终 memory 交给两种真实不同架构。

原主方法保留 learned Q、受限特征残差、in-loop DM、GM 和传播目标。没有新增联合统计损失、EM、合成拓扑、旧 memory 更新或动态预算。历史 shared/split-Q、no-GM 和旧 InfoNCE 负结果不被本批覆盖。

## 2. 已完成的机制诊断

WikiCS、SportsFit、BookHis，参考顺序 9101，构造种子 271006，五阶段。每阶段选三个优化时点记录梯度，总计 45 个探针；每步记录文本来源切换。这里没有训练下游 Student。

| 数据集 | 原硬文本失真 | Q-medoid 失真 | 原覆盖余弦 | Q-medoid 覆盖余弦 | medoid 唯一来源比例 | 直接文本/DM 梯度范数比 |
|---|---:|---:|---:|---:|---:|---:|
| WikiCS | 0.5073 | 0.3991 | 0.6343 | 0.6455 | 45.61% | 1.5047 |
| SportsFit | 0.3356 | 0.2478 | 0.7499 | 0.8020 | 77.88% | 0.1018 |
| BookHis | 0.6569 | 0.5332 | 0.4385 | 0.5189 | 86.73% | 0.0650 |

表中失真/覆盖先在阶段内按类别等权，再按五阶段等权。唯一来源比例按 slot 计，各阶段等权。梯度比平均 15 个探针，不是 Adam 更新量或贡献百分比。

| 数据集 | 原硬文本与归一化软文本的余弦 | 相邻优化步硬来源切换率 | 原 Q-max 唯一来源比例 |
|---|---:|---:|---:|
| WikiCS | 0.6968 | 10.47% | 100% |
| SportsFit | 0.8009 | 1.70% | 100% |
| BookHis | 0.5456 | 0.31% | 100% |

切换率是每步发生切换的 slot 比例的平均，不是整个训练中曾经切换过的累计比例。这里的失真为 Q 加权的原文到硬文本的余弦距离；覆盖为每个 raw 文本到同类已选 payload 的最大余弦再平均。它们不是分类指标，也不是语义信息保留率。上述 medoid 指标在原方法最终 Q 上做只读投影，不能与重新优化候选后的唯一文本数量混为一谈。

- 文本直接梯度不是缺失。WikiCS 的范数甚至高于 DM，不能一律增大文本权重。
- 文本与 DM 梯度夹角有正有负：负余弦探针分别为 8/15、7/15、6/15。不能将其概括成所有阶段都发生梯度冲突。
- 硬文本与 Q 聚合文本中心的偏差在三图都存在。相同 Q 下改为 medoid 能降低定义中的失真，并提高阶段平均覆盖，但这不证明分类成绩必然提高。
- 更低失真伴随重复来源增加，WikiCS 风险尤其明显。不能把它重新包装成无缺点的语义增强。

### 梯度测量的含义

测量的是总外层 GM 经 `text_new` 返回 Q logits 的直接向量-雅可比路径，并与总 GM、DM、传播路径比较；并不是一个独立的可加性“文本损失收益”。诊断调用 `autograd.grad`，不向参数 `.grad` 累加，并保存/恢复 Python、NumPy、Torch 和 CUDA RNG。

CPU 开关测试逐位一致。GPU WikiCS 正式开关对照 X 最大误差约 9.19e-6，来源一致；off/off 重复自身最大误差约 4.90e-3，且来源不完全一致。开关差异处于本次重复波动包络内，但这不等于证明 GPU 全流程逐位确定。最初更严格 smoke 失败记录保留。

## 3. 唯一的方法修改

令原文 embedding 经过归一化，slot j 的质量为 m_j，Q 加权文本均值为：

\[
\bar e_j=\frac{1}{m_j}\sum_i Q_{ij}e_i,\qquad m_j=\sum_iQ_{ij}.
\]

原方法选择 `argmax_i Q_ij`。候选改为当前 delta 同类真实文本中的加权余弦 medoid：

\[
r_j^*=\arg\min_{r:y_r=y_j}\frac{1}{m_j}\sum_iQ_{ij}(1-e_i^\top e_r)
=\arg\max_{r:y_r=y_j}\bar e_j^\top e_r.
\]

这个等价式给出的是固定 Q、固定合法文本集合下的硬投影最优性，不是全局凝聚最优性或分类保证。它无需新损失系数，不生成新文本，不访问历史 raw，不强制同类 slot 使用同一文本，也不保证各 slot 选择互异。

每次优化前向与最终保存均使用同一函数。反向仍使用原 ST 代理：

\[
T_j^{ST}=e_{r_j^*}+\widetilde T_j-\operatorname{stopgrad}(\widetilde T_j),
\quad \widetilde T_j=\operatorname{Normalize}(\bar e_j).
\]

硬选择变化也会改变后续 GM 与 Q/X 的优化轨迹，所以正式候选重新进行五阶段闭环构造，不能用一次最终文本替换冒充新方法结果。ST 仍是有偏代理，不宣称是离散 argmax 的真实梯度。

实现：`unified_q_payload_projection.py:q_text_medoid`；`unified_q_multimodal.py:_text_rows/optimize_stage`。默认 `q_max` 保留原行为，候选配置只改 `text_materialization` 一项。

代码入口：

- [硬文本投影](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_payload_projection.py:7)
- [训练前向与最终保存的共同选择接口](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py:612)
- [只读梯度探针](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_text_diagnostics.py:58)
- [SGC 架构迁移](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/graphtext_sgc_transfer.py:11)
- [正式闭环确认入口](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_unified_q_text_confirmation_20261008.py)

## 4. 独立确认协议

已完成 3 数据集 × 2 顺序 × 2 构造种子 × 4 方法，每条五阶段构造链。每份最终 memory 使用 GCN/SGC 两种架构、各三个 Student seeds 评价，共 48 条构造链、240 次阶段构造、96 条 readout 汇总、288 次正式 Student 训练。诊断、smoke、不可行顺序和已完成组合的恢复校验不计为新增独立样本。

构造种子 271108/271109；Student seeds 150/151/152；2048 steps，batch 24；MiniLM、prompt、损失和 report-dev 划分固定。此批只评估最终 memory，不报告未测的 stage-average 或 worst-stage。

| 类别 | 名称 | 实际做法与改动 |
|---|---|---|
| 候选单项修改 | Q-medoid | 原 CaT-Q，仅替换训练前向与保存时的硬文本投影 |
| 内部主方法 anchor | 原 In-loop CaT-Q | 同类受限 Q + X 残差 + DM/GM 等原目标，Q-max 原文 |
| 外部论文机制适配 | CaT + nearest5 | class-wise 随机非线性表示 DM，3000 步自由 X 优化；当前 delta 凝聚、冻结旧前缀、graph-nearest5 文本附着 |
| 外部论文机制适配 | GECC-hard + TAG | 0/1/2-hop 特征的权重分别为 0/.3/.9；20 次 hard Lloyd 重启与平衡罚项；sklearn 实现、当前 delta、同配额和 nearest5 文本附着 |

后两项不是官方完整流水线复现；CCTP 文本附着不是新增的一篇外部论文。所有方法为 edge-free memory，访问接口统一，rows 与 320 字符上限相同，但实际 token/字节信息量不严格相同。

### 真实架构迁移

GCN 沿用原两层非线性模型；SGC 使用 `normalize(W(A_hat^2 X)+b)`，两跳传播后仅一个线性图映射。冻结文本编码器、可训练文本投影、pair/prompt/margin 损失及训练预算不变。不针对 SGC 重新凝聚或调参。预测采用类别 prompt 接口，但不是严格未见类别或跨数据集 zero-shot。

### 预算可行性与顺序范围

预算轨迹固定，类别顺序改变。BookHis/9201 第 3 阶段有 659 个当前 raw，却要求新增 828 个 slot，在任何训练前被拒绝。按只看 pool 类别数的规则从 9201 递增，选择前两个可行顺序：WikiCS/SportsFit 为 9201/9202，BookHis 为 9202/9203。完整失败计数保留在 order_feasibility.json。

因此结论限定于可行 fixed-schedule 顺序，不代表任意自然到达流。真实 fixed-rate 仍是后续单独协议，不混入本表。

## 5. 结果与判读

下表是最终 Macro-F1 / Accuracy，单位为百分比。每个 cell 先平均三个 Student seeds，再对两个顺序与两个构造种子的四个 cell 等权平均。完整表另外报告 cell 标准差、Old/New F1、配对区间和成本。

| 数据集 / Student | 原 In-loop CaT-Q | Q-medoid 单项修改 | CaT + nearest5 适配 | GECC-hard + TAG 适配 |
|---|---:|---:|---:|---:|
| WikiCS / GCN | 79.49 / 81.40 | 79.84 / 81.89 | 80.62 / 83.35 | 79.54 / 82.25 |
| SportsFit / GCN | 78.94 / 84.01 | 78.87 / 83.53 | 77.49 / 84.76 | 78.44 / 83.92 |
| BookHis / GCN | 40.51 / 76.63 | 39.55 / 76.59 | 19.23 / 67.24 | 39.69 / 76.18 |
| WikiCS / SGC | 77.78 / 80.64 | 78.18 / 81.37 | 72.32 / 75.98 | 78.57 / 81.25 |
| SportsFit / SGC | 72.30 / 79.43 | 73.03 / 80.01 | 56.64 / 61.35 | 75.62 / 81.33 |
| BookHis / SGC | 36.97 / 73.17 | 36.61 / 73.93 | 17.42 / 60.94 | 37.70 / 74.63 |

### 5.1 单项修改是否有效

| 数据集 / Student | Q-medoid 减原方法 F1，pp | 探索性 95% 配对 CI | 正收益 cell | 最差 cell 差值，pp |
|---|---:|---:|---:|---:|
| WikiCS / GCN | +0.35 | [-0.45, +0.79] | 3/4 | -0.45 |
| SportsFit / GCN | -0.07 | [-1.57, +1.35] | 2/4 | -1.57 |
| BookHis / GCN | -0.96 | [-2.06, +0.32] | 1/4 | -2.06 |
| WikiCS / SGC | +0.40 | [-0.33, +1.36] | 2/4 | -0.33 |
| SportsFit / SGC | +0.73 | [+0.11, +1.07] | 4/4 | +0.11 |
| BookHis / SGC | -0.36 | [-2.12, +0.81] | 2/4 | -2.12 |

区间采用顺序和构造种子两个交叉因素的配对 bootstrap。每个因素只有两个水平，区间仅作探索性描述，不构成强显著性结论，也没有把同一 memory 的三个 Student seeds 当作三个独立凝聚样本。

结论是不升级，而不是“文本梯度没有用”。SportsFit/SGC 的四个 cell 均改善，值得保留；但 GCN 下没有一致增益，BookHis 两种架构均降低平均 F1。BookHis/GCN 的 New F1 从 35.74 降至 32.17，不能只看几乎不变的 Accuracy。BookHis/SGC 则出现 Accuracy 提升而 Macro-F1 下降，也应如实报告。

### 5.2 原主方法是否优于外部基线

- GCN 下，相对 CaT 适配版的 F1 差值为 WikiCS -1.13、SportsFit +1.46、BookHis +21.28 pp。WikiCS 四个 cell 均低于 CaT；SportsFit 的收益依赖顺序，配对区间跨零；BookHis 四个 cell 均明显更好。
- GCN 下，相对 GECC 的差值只有 -0.06、+0.51、+0.82 pp，三个数据集的配对区间均跨零。不支持“稳定优于强基线”的表述。
- SGC 下，原方法在三图均优于 CaT 适配版，但仍分别低于 GECC 0.78、3.32、0.73 pp。SportsFit 的四个 cell 都低于 GECC；不能用换架构后对 CaT 的大幅优势掩盖这个结果。
- 这证明需要把模型架构和凝聚方法拆开比较；不证明共享 Q 是跨架构收益的唯一来源。GECC 同样具有较强迁移表现，历史共享/独立 Q 和 no-GM 的负结果继续有效。

本表使用新顺序、新 construction seeds，与此前 handoff 或 9101/9102/9103 表的协议实例不同。不可将跨表绝对数值变化当作某个方法组件的因果收益。所有外部结果均标为机制适配，不写成官方完整结果或普遍 SOTA。

### 5.3 成本

| 数据集 | 原 CaT-Q 构造秒 | Q-medoid 构造秒 | CaT 适配构造秒 | GECC 适配构造秒 |
|---|---:|---:|---:|---:|
| WikiCS | 19.7 | 22.2 | 21.0 | 1.9 |
| SportsFit | 136.9 | 132.9 | 81.0 | 378.3 |
| BookHis | 71.8 | 71.0 | 28.7 | 46.7 |

每个数是五阶段构造时间之和再跨四个 cell 平均。它只计构造函数，不含首次数据/模型加载、额外审计重编码、归档与下游训练；共享服务器上几秒的差值不能解释为可靠加速比。完整 Student 用时、进程 PyTorch 峰值显存与存储大小在 summary.json 中。相同 rows 和字符上限不代表相同实际 token 数或字节数。

## 6. 机制结论与下一步边界

本轮回答了三个问题：文本到 Q 的直接梯度存在；Q-max 的硬材料化存在可测失真；仅降低这种失真不足以稳定提升分类。文本均值更有代表性，不等于保留了类内对分类有用的差异。

medoid 重复来源增加是值得继续排查的机制，但本轮没有隔离重复率作为唯一因果变量，不能据此断言“只要去重就会变好”。改变硬前向还会改变后续 X/Q 轨迹；本次是闭环算法比较，不是固定 X 的最终文本替换实验。ST 代理偏差也没有被这项修改消除。

因此保持 `text_materialization=q_max` 为原主方法默认，保留可选 medoid 实现和完整负结果。没有删除文本梯度，没有撤回 learned Q，也没有在中途新增损失、调整权重或改 Student 挑有利分数。

后续可检验的单一假设是：现有目标是否缺少类内图文配对分布，而不仅是各自的类别均值。若进入下一轮，应预注册一个联合分布匹配版本，明确与现有 DM/GM 的重复关系，比较替换或增量效应，使用新的确认顺序/种子；不能在本轮反复加组件后仍称为独立确认。该方向尚未实现或验证，不作为现有方法贡献。

对论文的判断也维持审慎：当前有价值的持续访问约束与部分数据集增益仍在，但这次单项修改没有补足稳定的文本收益、共享 Q 独有收益或强基线普遍优势。它不能被包装成新的充分 novelty，也不足以据此宣称 WWW 长文已经稳妥。至少还需更多独立顺序/种子、更高保真的公开方法核对、封闭 test，以及与所声称贡献相匹配的机制实验。

## 7. 审计、复现与产物

- 完成 12 个预定可行组合，无缺失；代码/Student 配置、同顺序数据、跨顺序固定预算与划分检查通过。
- 240 个阶段记录的预算、冻结前缀、合法当前 delta 同类文本来源与数值检查通过；96 条 readout 对应 48 份相同归档 memory 的双架构评价。
- 48 份压缩 memory 在远端逐份加载并核对内容哈希、edge-free 拓扑、rows 与文本合同。所有 readout 的三种子聚合和 Old/New Macro-F1 重建检查通过。
- 110 项相关测试通过。不可行 BookHis/9201、早期诊断 smoke 和历史负结果保留，未计入正式确认。
- 本轮仅评估 report-dev，未评估 sealed test 或按其分数选择方法；构造访问审计是代码接口与来源检查，不是操作系统级隔离证明。没有删除历史文件或提交远程仓库。

完整 memory 保留于远端 `/home/pc/research/TAG_condensation/results/uq_text_path_20261008`；本地同步了完整指标、协议与归档哈希清单，不宣称本地已有全部 tensor 归档。

- [完整表与配对统计](D:/Aprojects/continue_tag_condensation/results/uq_text_path_20261008/summary.md)
- [结构化指标与成本](D:/Aprojects/continue_tag_condensation/results/uq_text_path_20261008/summary.json)
- [48 份归档审计](D:/Aprojects/continue_tag_condensation/results/uq_text_path_20261008/archive_audit.json)
- [确认计划及不可行顺序规则](D:/Aprojects/continue_tag_condensation/docs/current/UNIFIED_Q_TEXT_PATH_PLAN_ZH_2026-10-08.md)
- [冻结确认实现快照](D:/Aprojects/continue_tag_condensation/tmp_remote_results/uq_text_path_confirmation_20261008.tar.gz)

![Final Macro-F1](D:/Aprojects/continue_tag_condensation/results/uq_text_path_20261008/final_f1.png)

![Final Accuracy](D:/Aprojects/continue_tag_condensation/results/uq_text_path_20261008/accuracy.png)
