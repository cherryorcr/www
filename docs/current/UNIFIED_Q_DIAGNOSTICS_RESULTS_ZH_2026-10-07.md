# Unified-Q：受控诊断、多顺序确认与 SportsFit 迁移结果

日期：2026-10-07。状态：本轮预注册实验及必要重试全部结束。

## 1. 结论先行

**Unified-Q 的可微统一构造能够运行，BookHis 上的 Macro-F1 优势可以复现，但目前不能作为“稳定优于 CaT-CCTP”的最终性能主方法。**

- WikiCS：3 个顺序 × 3 个构造种子，9/9 cell 的 F1 均落后于 CaT-CCTP，平均差值 −6.21 pp。
- BookHis：同样 9 个 cell，9/9 的 F1 都更好，平均 +13.54 pp；但 Accuracy 9/9 都更低，平均 −1.60 pp。
- SportsFit：新增单顺序迁移，完整 Unified-Q 为 69.55 F1 / 74.74 Accuracy，CaT-CCTP 为 77.23 / 87.23。未证明第三图泛化优势。
- BookHis 的新类收益不稳定：参考顺序 New F1 很高；两个独立顺序的 New F1 差值平均 −1.09 pp，Old F1 平均 +14.68 pp。不能继续概括为“普遍改善新类塑性”。
- 本轮修复了数值可行性和大矩阵反向传播的显存问题；这些是正确性、可运行性改进，不等于分类性能提升。

完整数据入口：[原始汇总](../../results/unified_q_diagnostics_20261007/summary.md)、[配对结果与统计表](../../results/unified_q_diagnostics_20261007/assets/summary.md)。本轮没有删除历史文件，没有提交 Git，也没有改写已经完成的两图实验。

## 2. 实际完成了什么

| 实验 | 范围 | 状态 |
|---|---|---|
| 同 memory 的 Student A/B | WikiCS / BookHis，两种预算协议、两种 memory、两种 Student | 完成 |
| 评估算子与重复性审计 | 同一个训练模型切换 raw 图归一化；同 seed 重复训练 | 完成 |
| 数值及组件诊断 | 两图各 15 个方法设置，5 阶段，Student seeds 150/151/152 | 完成 |
| 多顺序、多构造种子 | 两图各 3 × 3 cell；CaT-CCTP 对预注册 Unified-Q v3 | 完成 |
| 第三图迁移 | SportsFit：CaT-CCTP、基础 UQ、完整 UQ | 完成；两种 UQ 经显存实现修复重试 |

主诊断、确认和迁移共有 **65 个完整 method cell、325 个完整阶段、975 次阶段级 Student 训练/评价**；此计数不含额外 Student A/B 与评估算子对照。另保留了 6 个旧数值失败尝试中已经产生的阶段记录，它们没有被并入完整 cell 均值。未完成任务没有按零分计入。

本地相关测试 37 项通过；显存修复快照在远程 `tagsam` 环境的核心测试 30 项通过。

## 3. 协议与方法定义

### 3.1 公平比较的边界

- 主比较采用真实累计 **5% fixed-rate**，预算依据当前实际到达的构造池节点数计算，记录 class-floor 例外。
- 每图 5 个阶段；更换顺序保持各阶段类别数量不变，但 fixed-rate 下中间阶段的节点预算随到达量改变。不能把不同顺序的绝对分数当成纯粹的顺序因果效应。
- 参考顺序 `9101`；独立顺序 `9102/9103`。构造 seeds 为 `271006/271007/271008`。这不是 3 个完全未用于开发的新顺序。
- 每个配对 cell 使用相同 Student、种子 `150/151/152`、2048 steps、batch 24、MiniLM、report-dev mask、类别顺序和每阶段 memory rows。
- 构造接口仅接收当前 raw delta 与该方法自己的旧 synthetic memory。主比较冻结旧 X、文本和旧旧边，不启用 boundary sketch。
- 完整 raw graph 仍由独立评价器持有并用于 transductive 评价，不能把“构造器不回读历史 raw”写成“整个训练/评价系统从不访问历史 raw”。本轮检查了接口、快照和记录，不是 OS 级文件读取追踪。
- 全部数值是 **report-dev**，不是封存最终测试集。预测使用类别 prompt 相似度，无额外训练的分类头；但构造和 Student 训练使用了已见类别标签，且本轮评价已见类，不是严格 unseen-class zero-shot 验证。
- 历史 fixed-schedule 只保留为容量压力协议。它与当前 fixed-rate 的中间预算不同，不能合并汇总。

### 3.2 名字与实现对应

| 名称 | 实际执行内容 |
|---|---|
| CaT-CCTP / `cat_cctp` | CaT 的类别级分布匹配凝聚 X，3000 次构造更新；按合成特征的 graph-nearest 附着当前 delta 的 top-5 文本；保存 edge-free memory。属于 CaT 的 TAG 适配，不是官方原论文 TAG 实现。 |
| 基础 Unified-Q / `uq_30` | 结构化同类初始化；一个可学习 Q 同时生成聚合 X、真实 top-1 文本与诱导拓扑；联合梯度匹配更新 Q logits 和受限特征残差 R；128 维代理；固定 30 次 Sinkhorn。 |
| 严格 Unified-Q / `uq_strict` | 基础方法改为检查边际残差，容差 `1e-4`，失败明确报错；不按 F1 回退。 |
| 384 维代理 / `uq_proxy384` | 在严格方法上将代理 hidden/out 对齐至 384；其它目标不变。 |
| 传播尺度修正 / `uq_prop384` | 384 维代理再将单位向量的逐维平方均值改为平均 cosine 距离，避免传播项被维度除弱。 |
| 完整预注册候选 / `uq_v3` | 严格求解 + 384 维代理 + 传播尺度修正 + 受限保度换边。每次二值化至多接受 2 次 two-switch，并使用同样的硬图前向/材料化流程。 |

所有本轮梯度版本的外层更新次数是 **30**。`uq_30/uq_64/uq_128` 的数字指 **Sinkhorn 迭代次数**，不是外层优化步数。

当前 Unified-Q 没有先运行 CaT 初始化：`cat_anchor_x` 未提供，代码中的 CaT anchor 项为零。CaT 只在对照臂中执行。`R` 是额外的 X 残差；冻结 R 不等于冻结 X，因为 Q 仍然改变聚合特征。

文本来自当前 delta，清理空格、限制为 320 字符、同一 MiniLM 重编码。训练前向和最终保存使用同一个最高权重真实文本；反向通过软聚合文本传梯度。它不是 LLM 生成文本，也没有直接优化离散词语。

度守恒是相对于 **Q 诱导 top-k 合成图的换边前后**，不是相对于原始 raw 图。主版本没有新旧跨阶段边，也没有更新旧 memory；旧 X 更新是单独协议消融。

## 4. 多顺序主结果

下表先在每个 cell 内平均三个 Student seeds，再对 9 个“顺序 × 构造种子”cell 汇总。± 为 cell 均值的样本标准差，不是 27 次独立实验的标准差或置信区间。

| 数据集 | 方法 | Final Macro-F1 | Accuracy | Stage F1 | Stage Acc | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| WikiCS | Unified-Q v3 | 75.16 ± 1.04 | 78.98 ± 1.11 | 78.76 | 82.41 | 75.57 | 71.47 |
| WikiCS | CaT-CCTP | 81.37 ± 0.69 | 84.24 ± 0.49 | 84.65 | 87.34 | 81.62 | 79.17 |
| BookHis | Unified-Q v3 | 43.53 ± 1.39 | 78.71 ± 0.71 | 47.55 | 82.07 | 42.92 | 50.25 |
| BookHis | CaT-CCTP | 29.99 ± 1.52 | 80.31 ± 0.32 | 35.92 | 83.56 | 29.57 | 34.66 |

![多顺序 F1 与 Accuracy](../../results/unified_q_diagnostics_20261007/assets/confirmation_main.png)

| 数据集 | ΔF1 | 描述性 95% order-level CI | F1 正收益 cell | 最差 ΔF1 | ΔAccuracy |
|---|---:|---|---:|---:|---:|
| WikiCS | −6.21 | [−7.18, −5.24] | 0/9 | −7.98 | −5.26 |
| BookHis | +13.54 | [+12.11, +14.97] | 9/9 | +9.99 | −1.60 |

CI 以 3 个顺序的平均配对差值为单位，使用自由度 2 的 t 区间。顺序数量很少，且含开发参考顺序；这是描述性不确定性，不是对任意顺序泛化的强证明。不同 Student seeds 不作为额外独立顺序。

仅看两个独立顺序，WikiCS 的 ΔF1 为 −6.40，BookHis 为 +13.36，正负方向并没有因为排除参考顺序而改变。

### 4.1 新旧类别拆开后，结论有所变化

Old/New F1 是最后阶段评价集合上，按旧类/本阶段新类分组平均的 per-class F1；不是把预测类别限制在对应子集后重新分类。不同类别顺序的 New F1 对应不同实际类别。

| 数据集 | 顺序 | 总体 ΔF1 | Old ΔF1 | New ΔF1 |
|---|---:|---:|---:|---:|
| BookHis | 9101，参考 | +13.90 | +10.71 | +48.95 |
| BookHis | 9102，独立 | +13.85 | +15.28 | −1.83 |
| BookHis | 9103，独立 | +12.88 | +14.08 | −0.35 |
| WikiCS | 9101，参考 | −5.83 | −5.39 | −9.84 |
| WikiCS | 9102，独立 | −6.61 | −6.47 | −7.96 |
| WikiCS | 9103，独立 | −6.18 | −6.28 | −5.27 |

![按顺序观察新旧类差值](../../results/unified_q_diagnostics_20261007/assets/order_old_new.png)

因此，BookHis 的总体收益不是虚假的，但“收益主要来自新类塑性增强”的原解释不够稳健。是否改善少数类、牺牲头部类，还需要逐类混淆矩阵与类别频率相关性分析，不能仅凭 F1/Accuracy 差异断言原因。

## 5. SportsFit 第三图迁移

本节为 1 个参考顺序、1 个构造 seed、3 个 Student seeds；不能与上一节的 9-cell 标准差混用。所有方法最终 memory 为 6057 rows。

本轮 UQ/CaT 运行器按当前 pool 节点数重新计算的 fixed-rate budgets 是 `3817, 4486, 4812, 5305, 6057`。旧的 SportsFit public-baseline manifest 中另有一组 `4491, 4965, 5430, 5802, 6056`；那是历史 baseline 的 manifest，不属于本轮 UQ/CaT 配对协议。两种本轮方法彼此严格共用前一组预算，不能把两组预算的绝对分数直接合并比较。

| 方法 | Final Macro-F1 | Accuracy | Stage F1 | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|
| Unified-Q v3 | 69.55 ± 2.54 | 74.74 ± 4.02 | 66.38 | 66.95 | 83.88 |
| 基础 Unified-Q | 68.85 ± 1.19 | 76.60 ± 2.65 | 66.62 | 66.06 | 84.17 |
| CaT-CCTP | 77.23 ± 2.98 | 87.23 ± 0.75 | 76.24 | 74.57 | 91.89 |

完整版本相对基础 UQ 仅 +0.70 F1，同时 Accuracy −1.86；相对 CaT-CCTP 为 −7.68 F1 / −12.49 Accuracy。不能把“能跑通”当作“迁移成功”。

### 5.1 显存故障与修复

原始两种 UQ 都在首阶段构造时 OOM。远程设备实测为两张 RTX 4090，每张驱动报告 49140 MiB，总量并非标准 24 GiB。失败时单进程使用约 47 GiB。

定位到 `_log_sinkhorn` 原来仅在迭代次数超过 64 时启用激活重计算。大矩阵即使只迭代 30 次，也会保留大量反向中间量。修复为：同类块元素数超过 100 万时，用 8 次迭代为一段进行 checkpoint/recompute。

- 不改变 Q 的可行域、数据、5% 预算、损失、梯度目标、迭代次数、精度、种子或 Student。
- 新增测试验证前向输出及一、二阶梯度与未 checkpoint 版本一致；远程核心测试通过。
- 使用独立 `unified_q_diagnostics_20261007_memscale` 快照和 `memretry_` 目录，不覆盖原失败记录或两图完整结果。
- SportsFit 两种 UQ 的构造峰值降为约 **22,490 MiB，即 21.96 GiB**，完成全部五阶段。
- 原/重试 protocol 对比没有字段变化，只有 Unified-Q 实现文件哈希改变。OOM 记录保留，任务被标记为“已通过重试解决”，不是忽略失败。

这解决的是当前实现的显存冗余，**没有消除稠密 Q 的规模复杂度**。更大的图仍需要分块或稀疏的等价实现。

## 6. 诊断与消融说明了什么

以下为参考顺序、单构造 seed、相同 Prompt Student 的 Final Macro-F1。不能把每图表现最好的行重新命名为同一个预注册主方法。

| 设置 | WikiCS | BookHis | 可支持的解释 |
|---|---:|---:|---|
| CaT-CCTP | 81.01 | 31.24 | 当前强对照 |
| 基础 UQ，Sinkhorn 30 | 74.31 | 40.30 | 基础完整方法 |
| Sinkhorn 64 | 72.76 | 40.51 | 增加数值迭代不保证性能收益 |
| Sinkhorn 128 | 74.01 | 40.22 | 同上，外层仍为 30 步 |
| 严格 Sinkhorn | 74.62 | 40.10 | 正确性修复，没有明显性能改善 |
| 仅 Q 初始化、不做外层更新 | 71.16 | 40.30 | WikiCS 中梯度更新有正向作用；BookHis 单个 cell 的最终 F1 差异很小 |
| 固定 Q，只更新 R | 70.90 | 40.41 | WikiCS 需要学习分配，而不只是调整残差 |
| 更新 Q、固定 R | 74.24 | 41.94 | BookHis 上额外残差不是必然有益；不能说冻结了所有 X |
| feature-only 控制 | 74.91 | 39.49 | 同时去文本梯度与新拓扑，不能单独归因给文本 |
| 去新拓扑、保留文本学习 | 76.33 | 37.72 | 拓扑对 WikiCS 有负影响，对 BookHis 有帮助 |
| 严格 + 384 维代理 | 76.09 | 42.90 | 两图都有正信号，尚未单独跨顺序确认 |
| 再修正传播损失尺度 | 74.75 | 42.19 | 尺度纠正不自动等于更好性能 |
| 完整 v3：再加保度换边 | 74.76 | 42.69 | 换边没有证明在两图都带来明显提升 |
| v3 + 旧 X 每阶段 1% 半径 | 73.94 | 42.13 | 单独旧记忆更新协议；相对同设置 frozen v3 两图都未更好 |
| v3 + 旧 X 无硬半径 | 71.22 | 38.96 | 无界旧特征更新更不稳定；仍有软 anchor，不是完全无正则 |

尤其不能用 `42.13 > 40.30` 来宣称旧记忆更新有效：这两行还改变了代理维度、传播尺度和拓扑。正确单因素参照是 v3 的 42.69。

当前较明确的瓶颈是：拓扑质量依赖数据集；代理损失的改善不稳定转化为最终分类改善；受限 Q 聚合与有限残差未必具备 CaT 自由优化 X 的表达能力。后一点仍是待验证假设，不能当作已经证实的失败原因。

## 7. CCTP 数值为什么变高

控制同一份 memory、Student 训练预算和评价集合，统一 full-raw 评价算子后：

| 数据集 | Memory | 旧 Student F1 | Prompt Student F1 |
|---|---|---:|---:|
| WikiCS | CaT-CCTP | 52.73 | 80.79 |
| BookHis | CaT-CCTP | 16.63 | 32.92 |
| WikiCS | Unified-Q 旧确认 memory | 42.49 | 75.39 |
| BookHis | Unified-Q 旧确认 memory | 14.20 | 40.41 |

这是 **Student 模型/目标组合** 的对照，不是单个 loss 的消融。Prompt Student 使用多正样本图文对齐、prompt CE 和 margin，更直接对应最终 prompt 分类。

另外，旧评价保留重复边，新评价去重并对称化。换算子对 WikiCS 影响很小，对 BookHis 某些行约 1 pp，但不足以解释上述大幅跳升。同 seed 重复的 UQ 有 GPU 数值非确定性，Prompt Student 此次重复差值最多约 0.16 pp；不能声称所有运行位级一致。

**所以，CCTP 约 50% 到约 80% 不能写成凝聚算法本身的改进。** 历史旧 Student 的公开基线也不能不经审计直接并入当前主表。

## 8. 正确性、成本与完成审计

- 18 个确认配对、180 个阶段的预算、冻结前缀、有限数值、传播算子一致性检查通过；从 per-Student 原始值重新计算了 684 个有效均值，无不一致。
- 确认阶段使用的核心算法、Student、CaT、数据加载器、稀疏化模块哈希一致。初始严格求解上限 512 不够时提高到 2048，容差不变；该修复基于数值失败，而不是基于 F1。
- 全部已记录的 201 个严格求解阶段中，残差违反 `1e-4` 的记录为 0；其中包含旧失败尝试前已成功产生的阶段。固定迭代控制不承诺达到严格容差。
- 确认实验 final memory：WikiCS 410、BookHis 1454；第三图 SportsFit 6057。文本字符串、embedding、来源以及图均以合法阶段 memory 保存。
- 所有预注册任务均有完整结果或明确对应的成功重试；两图保留的旧 partial JSON 不是当前仍在运行的任务。
- 运行结束，两张 GPU 空闲，远程仍余约 5.09 GiB；没有自动删除文件。

| 数据集/汇总范围 | 方法 | 总构造秒数 | 三 Student、五阶段总训练秒数 | memory rows |
|---|---|---:|---:|---:|
| WikiCS，9-cell 平均 | Unified-Q v3 | 19.53 | 142.78 | 410 |
| WikiCS，9-cell 平均 | CaT-CCTP | 30.78 | 141.49 | 410 |
| BookHis，9-cell 平均 | Unified-Q v3 | 43.80 | 155.93 | 1454 |
| BookHis，9-cell 平均 | CaT-CCTP | 52.10 | 151.65 | 1454 |
| SportsFit，单 cell | Unified-Q v3 | 156.86 | 123.30 | 6057 |
| SportsFit，单 cell | CaT-CCTP | 182.93 | 114.75 | 6057 |

这些是队列实际测得的构造/训练耗时，不含数据加载等全部端到端开销；两图确认使用共享 GPU，不是独占吞吐基准。CaT 构造步数为 3000、UQ 外层为 30，不能直接把构造秒数差解释为理论效率优越。SportsFit 构造峰值：UQ 约 21.96 GiB，CaT 约 1.05 GiB，显存仍明显更高。

## 9. 下一步只做有针对性的调整

1. **先定位 WikiCS/SportsFit 的表示损失，不再把数值求解次数当作性能调参。** 在固定 384 维代理下比较完整图与去新拓扑，并记录 Q 聚合前后的类间 margin、类内方差、硬化前后消息误差。当前去拓扑只收回部分 WikiCS 差距，不能认定全部问题都在 A。
2. **统一检查代理权重与最终评价目标。** 当前 GM 对 old/new 组和类做均衡加权，最终 Student 采样按 memory 行分布；Macro-F1 与 Accuracy 的 trade-off 可能与此有关。先做逐类贡献及梯度块分析，再只修改一个目标；不再叠加 EM、selector 或新 Student。
3. **把硬拓扑的传播质量作为一个明确问题。** 两次局部保度换边不足以证明解决了全局稀疏化问题。保留 Q 统一生成 X/文本/图的主线，研究更直接的传播保持目标，而不是根据评价 F1 在有图/无图版本间选择。
4. **旧记忆更新暂保留消融，不作为已有效组件。** 当前同设置对照不支持晋升，但也不能据此证明所有新旧联合优化都无效。
5. **性能修正之后再冻结最终方法。** 必须保留本轮负结果；随后用新独立顺序与统一 Student 重跑强公开基线，并补最终封存测试与跨代理模型验证。

目前可以写问题定义、可微统一构造、访问/预算约束与失败机制，但不能写“稳定 SOTA”“普遍改善新类塑性”“已证明对任意 Student 泛化”。只完成更多表格不会自动解决这些证据缺口。

## 10. 文件索引

- [实验原始总表](../../results/unified_q_diagnostics_20261007/summary.md)
- [完整逐方法 cell](../../results/unified_q_diagnostics_20261007/method_cells.csv)
- [18 个配对 cell](../../results/unified_q_diagnostics_20261007/assets/paired_cells.csv)
- [顺序级区间与独立顺序差值](../../results/unified_q_diagnostics_20261007/assets/confirmation_uncertainty.csv)
- [完成及重试协议审计](../../results/unified_q_diagnostics_20261007/assets/completion_audit.json)
- [配对与原始均值复算审计](../../results/unified_q_diagnostics_20261007/assets/confirmation_audit.json)
- [Student 算子归因](../../results/unified_q_diagnostics_20261007/operator_attribution.csv)
- [数值检查](../../results/unified_q_diagnostics_20261007/numerical_audit.csv)
- [Q 联合构造实现](../../code/v2_llm_fusion/unified_q_multimodal.py)
- [本轮方法配置与执行器](../../code/v2_llm_fusion/run_unified_q_diagnostics_20261007.py)
- [保度稀疏化](../../code/v2_llm_fusion/unified_q_sparsification.py)
- [表格、可视化及完成审计生成器](../../tools/build_unified_q_diagnostics_assets_20261007.py)
