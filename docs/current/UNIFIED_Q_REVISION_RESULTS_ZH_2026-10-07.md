# Unified-Q 受控优化：实现、验证与大图结果

## 1. 本轮范围

本轮实现和冻结的是同一个 Unified-Q 框架，而不是重新切回 CaT 初始化，也不是把每张图的最好组件组合成不同方法。预先指定的候选为 `uq_semantic`：一致性修复、新旧组及组内类别均衡、特征/消息/文本多样性初始化、Q 与特征残差联合优化。

本轮已经全部完成：20 个实验配置、54 个方法 cell、270 个阶段、810 次 Student 训练评估。18 个追加确认任务退出码均为 0，两张 GPU 已空闲。完成方案不等于所有优化都有效，更不等于已经达到最终论文方法晋升条件。

实验分为三层，不能混合计算一个总分：

| 层次 | 数据集 | 顺序/构造种子 | 比较内容 |
|---|---|---|---|
| 机制归因 | WikiCS、BookHis | order 9101；construction 271006 | 8 个方法、每个 5 阶段 |
| 固定预算确认 | WikiCS、BookHis | orders 9101/9102/9103；seeds 271006/271007/271008 | CaT-CCTP 与预先指定 Unified-Q |
| 真实 fixed-rate | WikiCS、BookHis | order 9101；construction 271006 | CaT-CCTP、均衡版、语义初始化版 |

每次评估均使用同一个 GraphTextPromptStudent、Student seeds 150/151/152、2048 步训练。固定预算确认共有每数据集 9 个 order-construction cell，每个 cell 内先平均三个 Student seed。order 9101 是已有参考顺序，9102/9103 是本轮新增独立顺序，不应说成三个全新顺序。

所有结果是 report-dev，而不是独立最终 test。当前实验检验的是文本 prompt 分类接口，不是未见类别或跨数据集泛化。训练使用标签和全部类别 prompt，阶段评分的节点限于已经到达类别；推理候选仍包括全部 prompt。测试时没有另外拟合类别分类头，但这不能表述为训练从未见过测试类别。

## 2. 现在实际优化什么

令当前新增数据为 $\Delta_t=(X_\Delta,E_\Delta,A_\Delta,y_\Delta)$，上一阶段合成记忆为 $S_{t-1}$。构造器只接收当前 delta 和本方法自己的合成 parent。

### 2.1 预算和初始化

固定预算模式在所有类别顺序下保持每阶段 memory rows 一致：WikiCS 为 `[285,330,368,391,409]`，BookHis 为 `[1395,1423,1436,1445,1454]`。本阶段新增 slot 数是累计预算减去 parent rows。类内按当前节点比例分配 slot，并为每个到达类别至少保留一个 slot。

fixed-rate 使用 $B_t=\max(\operatorname{round}(0.05N_{seen}),B_{t-1}+|C_t|)$。因此它不是固定预算实验的另一个随机种子，不能混在一起报告。若 class floor 生效，预算高于严格 5% 的事实必须保留。

新初始化把归一化特征、当前子图邻域消息和文本组成联合签名；先选最靠近类中心的真实节点，再维护到已选种子的最近距离，依次选择最远节点。每个种子同时提供特征、真实邻域消息、文本 prototype。它是覆盖初始化，不是可证明的最优聚类，也不是每类固定几个文本。

### 2.2 同一个 Q 生成三种对象

可训练参数是分配 logits $Z$ 和特征残差 $R$。先在每个类别内部进行 Sinkhorn：$Q=\operatorname{Sinkhorn}(Z/\tau)$，不同类别间分配为零，目标行质量为 1、类内目标列质量为 $n_c/m_c$。当前固定 30 次迭代，只近似满足这些约束。

记 $D=\operatorname{diag}(Q^\top\mathbf1)$，则新特征为：

$$X_{new}=D^{-1}Q^\top X_\Delta+R.$$

每行残差投影回 $\|R_j\|\le0.05\|(D^{-1}Q^\top X_\Delta)_j\|$。这个约束限制的是残差，不限制 Q 改变后整个合成特征的总位移。

文本候选是当前 delta 的真实字符串，统一清理空格、按词边界截至最多 320 字符，再用冻结 MiniLM 编码。每个 slot 的软表示为 $\tilde T_j=\operatorname{Normalize}(Q_{:j}^\top E_\Delta)$，硬文本则是 $i_j^*=\arg\max_iQ_{ij}$ 对应的一条真实文本。训练采用 straight-through：前向使用真实硬文本，反向使用软加权表示的导数。最终重新编码同一字符串并核对误差。它不是语言生成模型，也不保证每个 slot 的硬文本一定不同。

新拓扑得分为 $W=D^{-1/2}Q^\top A_\Delta QD^{-1/2}$，去除对角项，再在给定边数预算内材料化为二值无向图。硬图前向和最终 Student 使用同一去重、自环及对称归一化规则；反向经过软图。硬文本和硬边的 straight-through 都是有偏梯度估计，不应宣称离散问题的精确可微最优解。

### 2.3 多模态梯度匹配

同一个代理图文网络分别读取合法真实侧 $S_{t-1}\cup\Delta_t$ 与合成侧 $S_t$。代理损失为：

$$L_{mm}=L_{prompt\text{-}CE}+0.5L_{pair}+0.1L_{margin}.$$

CE 是类别 prompt 交叉熵，pair 是把同类视为多个正样本的图文对比损失，margin 针对最强错误 prompt。均衡版在新旧两组存在时各给 1/2 损失质量，各组内部再均分到类别和节点；首阶段新组质量为 1。图文损失同时调整 anchor 和候选质量，而不只是最后乘一个类别权重。

$$g_r=\nabla_\theta L_{mm}(S_{t-1}\cup\Delta_t),\quad
g_s=\nabla_\theta L_{mm}(S_t),$$

$$L_{GM}=1-\cos(g_r,g_s)+0.25\left|\log\|g_r\|-\log\|g_s\|\right|.$$

总目标为 $L_{GM}+0.5L_{prop}+0.1L_{Q-anchor}+0.1L_{R-anchor}$。其中 propagation 项匹配当前 raw 邻域消息池化后的方向与合成图传播方向。代码虽保留可选 CaT anchor 接口，本轮没有传入 CaT anchor，因此该项实际为零。

代理参数保持在预先指定的一次初始化，不在每个 outer step 训练 Student。通过 `create_graph=True` 对 $g_s$ 再求导，更新 Q 的 logits 和 R，共 30 次 outer backward。这是单初始化梯度匹配，不是完整 Student unroll 或轨迹匹配。最终评估重新训练独立的 384 维 Student；凝聚代理为 128 维，二者算子一致不等于模型容量、采样和学习轨迹完全一致。

### 2.4 新旧关系的真实边界

旧合成节点参加真实侧和合成侧损失，因而会影响新节点的优化方向；但旧 X、旧文本、旧旧边全部冻结。本轮没有 boundary sketch，所以没有新增跨阶段边。不能把“损失考虑旧记忆”写成“旧记忆已联合更新”，也不能宣称已解决新旧结构连接。

## 3. 与 CaT-CCTP 的比较含义

本轮对照采用本地 CaT-MMD 机制适配：对当前 delta 做一次特征传播，以随机特征空间的 class-wise MMD 直接优化合成 X，3000 步；再按 graph-nearest 选择 top-5 文本并在相同 320 字符预算内附着。合成训练图没有额外边，只使用 Student 自环。历史合成前缀冻结，当前 raw 访问范围、memory rows、Student 和评价节点均与 Unified-Q 相同。

它不是官方 CaT 的完整 TAG 原生实现。与 Unified-Q 的差异不止文本路由，还包括 X 参数化、构造目标和合成拓扑。因此胜负不能单独证明文本、拓扑或新旧均衡某一个模块有效。

## 4. 已完成的主消融

下表是单参考顺序、一个构造种子、三个 Student seeds 的均值，单位为百分比。不要把它与多顺序均值当成同一统计层次。

| 方法 | WikiCS F1 | WikiCS Acc | BookHis F1 | BookHis Acc |
|---|---:|---:|---:|---:|
| 预先指定 Unified-Q：均衡 + 语义初始化 | 72.37 | 77.66 | 39.45 | 75.49 |
| CaT-MMD + CCTP top-5 | 80.65 | 83.36 | 20.03 | 31.70 |
| Unified-Q 一致性修复 | 72.35 | 76.95 | 40.25 | 73.39 |
| 再加新旧/类别均衡 | 72.79 | 77.81 | 37.33 | 69.31 |
| 文本分配固定在 Q0 | 71.08 | 75.24 | 39.50 | 74.65 |
| 新拓扑固定在 Q0 | 69.72 | 75.13 | 35.00 | 65.46 |
| 不生成新拓扑 | 74.68 | 79.80 | 38.60 | 71.28 |
| 交替更新 Q/R | 72.08 | 76.87 | 35.82 | 68.99 |

文本学习相对固定文本：WikiCS +1.29 F1，BookHis -0.04 F1，尚不支持普遍文本收益。学拓扑相对固定 Q0 拓扑，两图均更好；但相对无新边，WikiCS -2.31 F1，BookHis +0.85 F1，因此“能优化已有拓扑”不等于“合成拓扑一定优于不加边”。

交替更新在相同 outer backward 次数下各只更新 Q/R 15 次，联合版各更新 30 次。当前结果不支持保留交替版，但不能外推成“任何 EM 都无效”；本实现没有生成式概率模型，也不应叫严格 EM。

类别均衡修正了两侧损失质量的语义差异，但并未稳定改善指标。语义多样性初始化也没有稳定优于简单初始化。它们是受控实验组件，而不是已被证实的独立贡献。

## 5. 独立确认

完整运行及每个 cell 的结果见 [统一结果表](D:/Aprojects/continue_tag_condensation/results/unified_q_revision_verified_20261007/summary.md) 和 [聚合 CSV](D:/Aprojects/continue_tag_condensation/results/unified_q_revision_verified_20261007/aggregate_cells.csv)。下表均为 3 个顺序乘 3 个构造种子的 9 个完整 cell；标准差是 cell 间标准差，不是把 27 个 Student 重复当成 27 个独立顺序。

| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|
| WikiCS | 预先指定 Unified-Q | 72.76 ± 1.65 | 76.57 ± 2.40 | 76.56 | 73.02 | 70.41 |
| WikiCS | CaT-CCTP | 80.17 ± 1.11 | 83.14 ± 1.02 | 83.91 | 80.45 | 77.66 |
| BookHis | 预先指定 Unified-Q | 38.45 ± 1.36 | 75.56 ± 1.61 | 43.17 | 38.11 | 42.19 |
| BookHis | CaT-CCTP | 17.39 ± 3.88 | 53.04 ± 17.32 | 28.63 | 17.88 | 12.01 |

| 数据集 | 配对 F1 差 | order 聚合的描述性 95% CI | 正收益 cell | 最差 cell 的 F1 差 |
|---|---:|---|---:|---:|
| WikiCS | -7.41 pp | [-10.51,-4.31] | 0/9 | -9.76 pp |
| BookHis | +21.06 pp | [+9.51,+32.61] | 9/9 | +16.37 pp |

仅三个顺序，区间很粗；已有参考顺序也包含其中，不能解释为充分的未见顺序泛化证明。WikiCS 的差距不是“只是一个种子不好”，BookHis 的优势也不是只来自一个最好 cell。

### 5.1 真实 fixed-rate：单独报告

这里仅完成参考顺序、一个构造种子、三个 Student seeds，不能借用上面九个固定预算 cell 的 CI。

| 数据集 | 方法 | Final F1 | Accuracy | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|
| WikiCS | 预先指定 Unified-Q | 75.42 | 78.83 | 76.87 | 62.36 |
| WikiCS | CaT-CCTP | 80.79 | 83.96 | 81.81 | 71.68 |
| WikiCS | UQ 均衡版，不加语义初始化 | 73.65 | 77.04 | 74.79 | 63.34 |
| BookHis | 预先指定 Unified-Q | 40.45 | 77.68 | 39.17 | 54.48 |
| BookHis | CaT-CCTP | 32.92 | 80.76 | 34.70 | 13.42 |
| BookHis | UQ 均衡版，不加语义初始化 | 43.22 | 79.09 | 41.48 | 62.39 |

BookHis 的预先指定候选仍提高 F1 7.52 pp，但 Accuracy 下降 3.08 pp。它更像类别均衡性能与节点总体准确率的权衡，不是所有指标同时改善。均衡版在这个 cell 更好，但没有相同独立确认，不能据此直接替换主候选。WikiCS 在实际 5% 协议仍落后 5.38 F1，固定预算错配不是其全部原因。

## 6. 数值与实现审计

### 6.1 必须单独报告预算协议影响

BookHis 的参考固定预算继承了历史表格，和当前 order 9101 的到达量不匹配。它仍是方法间相同预算的合法压力测试，但不代表每个阶段均按当前数据量保留 5%：

| 阶段（从 1 开始） | 当前 raw rows | 固定表新增 slot | 真正 5% 新增 slot |
|---|---:|---:|---:|
| 1 | 11584 | 1395 | 579 |
| 2 | 248 | 28 | 13 |
| 3 | 179 | 13 | 9 |
| 4 | 16562 | 9 | 828 |
| 5 | 513 | 9 | 25 |

两个协议最终都是 1454 rows，但 frozen-prefix 禁止历史重分配，阶段预算不同会显著改变最终记忆。这解释了为什么只核对最终 rows 不足以控制协议。CaT-CCTP 的 F1 从固定表 20.03 变为真实 5% 的 32.92、Accuracy 从 31.70 变为 80.76；这不是换 Student，而是预算的阶段配置改变。

后续自然增长主实验应以真正 fixed-rate 为主要实际场景，固定表保留为极端阶段容量约束压力测试。不能把固定表上的更大差距选择性作为唯一主结果，也不能据此归因所有差距都是算法问题。

### 6.2 实现与数值检查

已修复原图重复边导致的传播/计数不一致、硬前向丢失旧旧边，以及文本 argmax 并列时训练与材料化选择不一致。原始参考图改为稀疏邻接；图文对比损失按 anchor 分块并 checkpoint，仍使用完整候选集合。分块前后的值、一阶梯度及二阶梯度均有测试。

本地测试 25 项通过，其中核心实现 16 项在服务器 tagsam 环境也通过。新增结果审计重算 Student 种子均值与标准差、old/new F1 的按类别分解，核对代码哈希、数据 mask、阶段 IDs、预算、冻结前缀、梯度路径、残差半径和实际更新次数。该审计未重新保存/重算所有节点预测，历史 raw 的限制来自构造接口与调用链审查，不是独立 OS 沙箱证明。

注意：固定 30 次 Sinkhorn 不保证数值收敛到严格容量约束。主消融中最大行质量残差约 0.019894，约 1.99%；加入全部确认后最大为 0.066129，约 6.61%。170 个 UQ 阶段中 72 个超过诊断阈值 $10^{-3}$，全部保留在 `audit_report.json` 的 `numerical_warnings`，与统计或工件损坏分开列示。统计、协议哈希与工件检查通过，但严格 transport 数值容差未通过，不能表述为全部约束零违反。

## 7. 开销与剩余问题

主消融五阶段合计构造时间：WikiCS CaT 21.13 秒、预先指定 UQ 10.47 秒；BookHis CaT 26.19 秒、UQ 22.61 秒。对应峰值 allocated memory 分别约 376/416 MiB 与 467/2533 MiB。它们不包括数据加载/模型启动等全部端到端开销，也不是相同优化步数的比较。多顺序确认任务每 GPU 共享运行，不用于独占吞吐结论。

稀疏原图和分块损失降低了峰值内存，但 Q 仍为稠密 $n_\Delta\times m$，对比损失全候选计算仍有二次时间成本。不能宣称算法已经线性扩展。rows 相同也不等于严格字节预算相同：支持来源 ID、文本和拓扑都有额外存储。

目前较明确的风险是：同一个 Q 同时承担特征聚合、硬文本选择和拓扑分组，三条梯度可能冲突；小残差半径不能纠正过度平均；二值 top-k 拓扑与软代理梯度不同；单个随机代理的梯度好不等于完整 Student 训练后的边界好。现有消融支持“拓扑存在条件依赖”，尚不能在这些可能原因之间作最终因果判断。

## 8. 下一轮范围建议

本轮确认已全部完成。下一轮不按数据集选择不同赢家，不扩大 temperature/margin 搜索，只做可解释的单因素试验：

1. 先修数值可行性：给 Sinkhorn 增加收敛残差和列质量审计，或单独验证边际修正；固定数值标准，不按 F1 选迭代次数。保留本轮代码和结果，新版本单开目录，禁止混写。
2. 拓扑传输质量：保留共享 Q，先审计材料化前后消息失真、边预算及过平滑，再测试一个带传播保持约束的稀疏化方案；不直接叠加无依据的边选择规则。
3. 若无边版本也持续落后，做 Q 固定/R 学习与 Q 学习/R 固定的最小归因，检查特征平均和代理梯度是否为主瓶颈，然后再决定是否采用轻量多代理/分块梯度匹配。
4. 真实 fixed-rate 仍需独立顺序/构造种子确认。跨阶段边、旧记忆小幅更新和未见类别任务另开协议，不与上述归因同时修改。

当前应保留 Unified-Q 作为研究候选，不应冻结“已稳定优于 CaT”或“已达到最终论文方法”的结论。BookHis 的大幅正信号有价值，WikiCS 的系统性差距也必须保留。

## 9. 工件与复现

- [冻结计划](D:/Aprojects/continue_tag_condensation/docs/current/UNIFIED_Q_REVISION_PLAN_ZH_2026-10-07.md)
- [构造器](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/unified_q_multimodal.py)
- [实验入口](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_unified_q_revision_20261007.py)
- [结果审计](D:/Aprojects/continue_tag_condensation/tools/audit_unified_q_revision_20261007.py)
- [汇总程序](D:/Aprojects/continue_tag_condensation/tools/summarize_unified_q_revision_20261007.py)
- [WikiCS 主消融图](D:/Aprojects/continue_tag_condensation/results/unified_q_revision_verified_20261007/wikics_ablation.png)
- [BookHis 主消融图](D:/Aprojects/continue_tag_condensation/results/unified_q_revision_verified_20261007/bookhis_ablation.png)

所有新输出放入独立目录。本轮没有删除远程数据，没有提交或推送仓库。
