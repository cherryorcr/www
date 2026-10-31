# In-loop CaT-Q：确认实验、基线与机制诊断

日期：2026-10-07。本报告由逐阶段结果自动汇总，不按最好 cell 选择参数或方法。

## 一、完成范围

| 任务 | 已完成方法 cell | 计划 | 说明 |
|---|---:|---:|---|
| 独立顺序与构造 seed | 81 | 81 | 三图 × 三顺序 × 三构造 seed × 三方法 |
| fixed-Q 控制 | 3 | 3 | 三图参考顺序/构造 seed |
| 新增同协议基线 | 12 | 12 | 三图 × 四种适配/coreset 方法 |
| 固定内容文本/拓扑诊断 | 21 | 21 | 三图 × 七种诊断；每图只构造一条参考链 |

共 117 个完整方法 cell；每个有 5 个阶段和 3 个 Student seeds。
SportsFit、BookHis 的 no-GM 已纳入独立确认，不仅是单 cell 补测。

完成的是本轮预先列明的机制适配矩阵，不是所有公开算法的官方复现。官方静态 TAG 强基线和最强外部适配版的多顺序确认仍未补齐。

## 二、协议与统计

- 构造器只接收当前 raw delta 和本方法旧 synthetic memory；旧前缀不改写，不使用 boundary sketch。
- 三个顺序为参考顺序 9101 和独立顺序 9102/9103；构造 seeds 为 271006/271007/271008。
- 固定 base/各阶段类别数量和参考逐阶段 memory rows。独立顺序采用 fixed-schedule，不把它误写成各顺序都严格 5% fixed-rate。
- 相同 Prompt Student、MiniLM、2048 Student steps、batch 24、Student seeds 150/151/152；每阶段从累计 memory 重新训练，不继承 Student 权重。
- 构造允许使用当前类别标签，Student 也使用类别 prompt 监督。预测没有单独训练的类别线性头，但不是严格未见类别 zero-shot。
- 所有指标为 report-dev；推理使用完整原图的传导式传播。历史 raw 访问限制针对构造器，不针对该评估器。封存 test 未用于本轮。
- 先对三个 Student seeds 求均值，再对九个顺序/构造 cell 求 mean±SD。95% paired CI 对顺序轴和构造 seed 轴交叉 bootstrap；只有三个顺序，属探索性稳定性证据。

| 数据集 | 五阶段累计 rows | 最终 rows |
|---|---|---:|
| WikiCS | [247, 324, 356, 379, 410] | 410 |
| SportsFit | [3817, 4486, 4812, 5305, 6057] | 6057 |
| BookHis | [579, 592, 601, 1429, 1454] | 1454 |

## 三、独立确认主表

下表不是单次最佳结果。F1 指 Macro-F1，单位均为百分数。

| 数据集 | 方法 | Final F1±SD | Accuracy±SD | Stage F1 | Stage Acc | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| WikiCS | In-loop CaT-Q | 80.13±0.99 | 82.99±1.59 | 83.67 | 86.41 | 80.19 | 79.60 |
| WikiCS | In-loop CaT-Q, no GM | 80.10±1.29 | 82.73±1.87 | 83.52 | 86.20 | 80.11 | 79.98 |
| WikiCS | CaT-CCTP adapter | 80.70±0.48 | 83.68±0.42 | 84.21 | 87.03 | 80.92 | 78.72 |
| SportsFit | In-loop CaT-Q | 78.73±1.61 | 84.60±2.06 | 81.94 | 87.89 | 78.64 | 79.20 |
| SportsFit | In-loop CaT-Q, no GM | 78.90±0.92 | 84.73±1.48 | 82.15 | 88.02 | 78.88 | 79.00 |
| SportsFit | CaT-CCTP adapter | 75.32±1.66 | 83.77±2.32 | 79.00 | 88.06 | 75.57 | 73.91 |
| BookHis | In-loop CaT-Q | 42.31±4.15 | 79.97±1.82 | 48.37 | 84.46 | 41.60 | 50.14 |
| BookHis | In-loop CaT-Q, no GM | 43.16±3.15 | 80.47±1.46 | 48.79 | 84.60 | 42.44 | 51.13 |
| BookHis | CaT-CCTP adapter | 24.71±4.74 | 75.43±4.33 | 30.51 | 80.41 | 24.13 | 31.10 |

![独立确认](../../results/unified_q_inloop_validation_20261007/figures/confirmation.png)

| 数据集 | 配对比较 | F1 差值 [95% CI] | 正收益 cell | 最差 cell 差值 |
|---|---|---:|---:|---:|
| WikiCS | In-loop CaT-Q − CaT-CCTP adapter | -0.57 [-1.70, +0.22] | 3/9 | -2.43 |
| WikiCS | In-loop CaT-Q, no GM − CaT-CCTP adapter | -0.60 [-2.01, +0.75] | 3/9 | -2.89 |
| WikiCS | In-loop CaT-Q, no GM − In-loop CaT-Q | -0.03 [-0.59, +0.53] | 3/9 | -0.94 |
| SportsFit | In-loop CaT-Q − CaT-CCTP adapter | +3.41 [+0.51, +6.07] | 8/9 | -0.00 |
| SportsFit | In-loop CaT-Q, no GM − CaT-CCTP adapter | +3.58 [+2.30, +4.78] | 9/9 | +2.05 |
| SportsFit | In-loop CaT-Q, no GM − In-loop CaT-Q | +0.17 [-1.30, +1.80] | 4/9 | -1.52 |
| BookHis | In-loop CaT-Q − CaT-CCTP adapter | +17.60 [+13.81, +21.43] | 9/9 | +10.71 |
| BookHis | In-loop CaT-Q, no GM − CaT-CCTP adapter | +18.45 [+15.11, +23.01] | 9/9 | +13.86 |
| BookHis | In-loop CaT-Q, no GM − In-loop CaT-Q | +0.85 [-0.31, +2.12] | 7/9 | -1.08 |

结论：SportsFit、BookHis 相对 CaT-CCTP 的收益跨顺序仍存在，WikiCS 没有明确优势。no-GM 与带 GM 的配对 CI 三图均跨零；这不证明两者严格等效，但没有支持保留 GM 的稳定性能证据。不能按每图挑一个赢家拼成主方法。

## 四、外部方法名字与适配边界

| 原方法/类别 | 本轮名字 | 改动 | 不能声称的内容 |
|---|---|---|---|
| CaT，普通属性图持续凝聚 | CaT-CCTP adapter | 当前 delta 上 class-wise 随机 Encoder MMD，3000 次优化自由 X；按 graph-nearest 取 top-5 文本拼接、320 字符重编码；edge-free、冻结前缀 | 不是原论文原生 TAG 结果；CCTP 是文本附着适配名 |
| GECC，图凝聚机制 | GECC-hard TAG adapter | 当前 delta 的 hop 传播、hard Lloyd 聚类与 restart 选择，附着 top-5 文本；不累计访问历史 raw | 不等于官方 fuzzy/完整 GECC 复现 |
| PUMA，图凝聚机制 | PUMA-inspired TAG adapter | 当前 delta 一次传播、线性伪标签 teacher、0.8 置信过滤和 CaT matching；附着文本 | 在标注 pool 内留出一部分模拟伪标签候选，监督使用方式有差异；不是官方完整 PUMA |
| 真实节点子集基线 | Stratified random coreset | 相同类别配额随机选节点，真实 X/文本，320 字符重编码、自环 | 不属于学习式凝聚 |
| 真实节点覆盖基线 | Stratified graph-text k-center | 同类归一化 X/text 联合空间 exact farthest-first，真实文本和相同行数 | 不属于学习式凝聚 |

它们统一了外层数据顺序、rows、Student 和评估器；不能据此说原算法、监督分配和内部优化预算完全相同。静态 TAG 官方方法没有在本批新协议下完成。

### WikiCS：参考顺序同协议比较

仅一个顺序、一个构造 seed，SD 来自三个 Student seeds，不与九-cell 均值混用。

| 方法 | Final F1±SD | Accuracy±SD | Stage F1 | Old F1 | New F1 | 构造秒 | Student秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| In-loop CaT-Q | 81.06±0.49 | 84.07±0.18 | 84.86 | 81.76 | 74.69 | 18.9 | 110.9 |
| In-loop CaT-Q, no GM | 81.84±0.31 | 84.47±0.13 | 85.48 | 82.60 | 75.00 | 19.8 | 109.5 |
| CaT-CCTP adapter | 80.79±0.22 | 83.82±0.10 | 85.04 | 81.76 | 72.11 | 21.0 | 107.0 |
| Stratified random coreset | 74.93±0.38 | 80.48±0.35 | 80.20 | 75.75 | 67.56 | 0.2 | 131.5 |
| Stratified graph-text k-center | 67.90±0.44 | 68.66±0.47 | 74.67 | 68.91 | 58.81 | 0.2 | 108.5 |
| GECC-hard TAG adapter | 80.79±0.54 | 83.22±0.26 | 83.69 | 81.44 | 74.95 | 1.8 | 109.5 |
| PUMA-inspired TAG adapter | 82.04±0.25 | 84.47±0.42 | 85.29 | 83.07 | 72.68 | 22.8 | 110.5 |

### SportsFit：参考顺序同协议比较

仅一个顺序、一个构造 seed，SD 来自三个 Student seeds，不与九-cell 均值混用。

| 方法 | Final F1±SD | Accuracy±SD | Stage F1 | Old F1 | New F1 | 构造秒 | Student秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| In-loop CaT-Q | 79.10±1.88 | 85.69±2.62 | 78.48 | 76.71 | 92.22 | 179.4 | 128.3 |
| In-loop CaT-Q, no GM | 79.79±0.96 | 86.51±1.99 | 78.33 | 77.49 | 92.47 | 157.3 | 114.1 |
| CaT-CCTP adapter | 77.54±2.95 | 86.46±1.82 | 76.03 | 74.80 | 92.59 | 180.9 | 126.4 |
| Stratified random coreset | 75.95±2.16 | 86.48±1.55 | 75.83 | 73.14 | 91.39 | 1.1 | 114.4 |
| Stratified graph-text k-center | 58.30±1.09 | 77.22±1.42 | 55.13 | 52.83 | 88.39 | 3.9 | 114.6 |
| GECC-hard TAG adapter | 77.81±4.04 | 88.81±0.66 | 78.88 | 74.92 | 93.72 | 702.5 | 114.1 |
| PUMA-inspired TAG adapter | 80.93±1.58 | 89.44±0.65 | 79.41 | 78.43 | 94.70 | 183.8 | 113.8 |

### BookHis：参考顺序同协议比较

仅一个顺序、一个构造 seed，SD 来自三个 Student seeds，不与九-cell 均值混用。

| 方法 | Final F1±SD | Accuracy±SD | Stage F1 | Old F1 | New F1 | 构造秒 | Student秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| In-loop CaT-Q | 46.18±1.12 | 81.70±1.06 | 51.75 | 44.72 | 62.27 | 65.5 | 111.8 |
| In-loop CaT-Q, no GM | 46.53±1.34 | 82.21±0.30 | 51.52 | 45.54 | 57.37 | 64.3 | 109.4 |
| CaT-CCTP adapter | 31.02±5.14 | 80.96±0.83 | 41.77 | 32.88 | 10.51 | 36.7 | 109.0 |
| Stratified random coreset | 37.31±2.11 | 81.24±0.22 | 42.13 | 36.47 | 46.49 | 0.4 | 109.6 |
| Stratified graph-text k-center | 23.00±2.98 | 76.50±1.08 | 30.69 | 24.71 | 4.20 | 0.8 | 109.4 |
| GECC-hard TAG adapter | 45.04±1.71 | 81.98±0.40 | 51.88 | 44.43 | 51.74 | 57.1 | 109.4 |
| PUMA-inspired TAG adapter | 34.56±6.27 | 81.45±1.08 | 43.38 | 34.54 | 34.74 | 31.1 | 109.0 |

![同协议基线](../../results/unified_q_inloop_validation_20261007/figures/baselines.png)

构造和 Student 耗时是该次运行实测，部分时段共享过资源，不能据此宣称严格的独占 GPU 加速比。九-cell 基线均值目前只有 CaT-CCTP；GECC/PUMA 单次结果不足以支持对它们的稳定优越性结论。

## 五、哪些组件真的有作用

### Q 和 GM

| 数据集 | 可学习 Q 的 F1 | 固定 Q 的 F1 | 差值 |
|---|---:|---:|---:|
| WikiCS | 81.06 | 72.79 | +8.26 |
| SportsFit | 79.10 | 58.10 | +21.00 |
| BookHis | 46.18 | 38.40 | +7.78 |

固定 Q 时仍保留允许的 residual 优化，所以这个对照支持在当前受限表示下学习 Q 的作用，而不是泛化地证明所有 transport 一定优于自由 X。

no-GM 只把 lambda_gradient 设为 0，仍通过分布匹配、传播一致性和 anchor 对 Q logits/R 反向传播。它不是不用梯度，也不是 CaT 先优化完成再加 anchor。

必须注意：当前直接依赖合成文本 T 的优化项位于 GM 分支。删除 GM 后，文本影响 Q 的初始化与 q0 anchor，并随着 Q 的改变重新材料化，但不再有直接的文本/图文损失梯度。因此 no-GM 可以是性能简化候选，不能仍宣称它显式联合优化了多模态梯度匹配。

### 固定内容文本/拓扑

以下是与同一次构造的 self-loop 对照的 F1 差值。拓扑臂的 X/文本完全相同；文本臂的 X 完全相同。

| 数据集 | 空文本 | 换 CCTP 文本 | 类内置换文本 | 二值拓扑 | 加权拓扑 | 置换拓扑 |
|---|---:|---:|---:|---:|---:|---:|
| WikiCS | -0.46 | +0.37 | +0.84 | -6.90 | -7.19 | -27.34 |
| SportsFit | +0.18 | +1.04 | +0.58 | -4.05 | -9.37 | -6.21 |
| BookHis | -0.08 | +0.14 | -4.13 | +0.35 | -3.95 | -1.94 |

![固定内容诊断](../../results/unified_q_inloop_validation_20261007/figures/attribution.png)

1. 当前拓扑未达到至少两个数据集稳定增益的门槛；暂不并入主方法。BookHis 的小幅 F1 增益还伴随 Accuracy 下降。这里只否定当前固定内容加边方案，不能证明所有联合拓扑学习都无效。
2. 空文本几乎不损失最终 F1，而换为 CCTP 文本三图都有小幅增加。当前真实文本 payload 的额外贡献证据不足。
3. 类内置换在 BookHis 降低 F1，但在另两图没有降低，说明 slot-text 对应的作用有数据集依赖。
4. 空文本只在构造完成后替换 payload；原 Q 初始化、构造过程和类别 prompt 仍可利用文本，不能称为整个算法的无文本消融。

## 六、可复现性审计

诊断重建的 memory 并非旧 pilot 的按位复现，三图 final F1 的偏移分别约为 WikiCS −0.36、SportsFit +0.38、BookHis +0.14 pp。这些诊断没有混入主表的独立确认均值；配对仅发生在同次构造内部。

代码审计和额外真实 WikiCS 首阶段 probe 表明：拓扑回调没有改变 Q、memory、Python/NumPy/Torch/CUDA RNG。两个没有回调的相同-seed 重跑本身也出现差异，故此前的‘回调消耗 RNG 导致漂移’猜测未获支持。新增 RNG 保存/恢复只是预防性隔离，不是已证实的根因修复。

另拟的全量 callback-RNG 重跑在启动阶段失败，没有成功训练结果；不计入 117 个 cell，也不作为修复验证。启动日志保留，队列已停用。固定内容结论来自已完成且内容哈希一致的同次构造内对照。

正常 CUDA 设置下，probe 的重复 X 相对差约 1.68%，247 个 slot 中有 41 个文本来源变化；开启确定性运算后，该对重复的 X 差降到约 9.27e-7，文本来源不变，但另一次重复仍有约 1.05e-3 的 X 差。该证据符合数值误差被硬选择/优化放大的风险，但尚未定位所有来源，也不能保证大图所有阶段已确定性复现。

SportsFit 的加权诊断算子存在约 6.14e-6 的最大非对称残差。保留实际用过的算子和 hash，不事后替换或重算 F1；审计以明确的 float32 容差记录警告，而不是宣称其严格对称。

no-GM 的确定性设置下额外重跑也不是按位相同：

| 对照 | X 相对误差 | 文本来源变化数 |
|---|---:|---:|
| plain_b vs plain_a | 3.636e-05 | 1 |
| with_callback vs plain_a | 2.022e-04 | 0 |

因此删除 GM 不能当作数值可复现性的完整修复；仍需独立定位归约/硬选择等数值路径。

存档审计状态：`passed_with_numerical_warning`；核对 109 份方法 memory；未完成项 0。
包含 protocol 一致性、checkpoint 与结果 hash、有限张量、rows、前缀、320 字符以及固定内容检查。访问隔离为构造器接口/代码审计，不是 OS 文件读取追踪。

## 七、当前建议与未完成证据

- 保留 edge-free In-loop CaT-Q 为主方法候选；可学习 Q、in-loop 非线性类别分布匹配与受限 R 是当前最有证据的核心。
- no-GM 是有根据的简化候选，不按数据集分别选择带/不带 GM。未预设等效界，不能将跨零 CI 写成证明 GM 无用。若删除 GM，必须相应收缩文本联合优化的论文主张。
- 拓扑不进入主方法；当前的 hard/weighted/置换图作为负结果和机制边界保留。
- 下一优先级是数值可复现性与真正的构造端文本消融，然后对当前最强外部适配版补独立顺序，补官方静态 TAG 基线，最后一次性评估封存 test。
- 不能据此宣布全数据集 SOTA、严格未见类别 zero-shot、公开官方算法全面胜出、图文拓扑联合学习的全部贡献均已证实，或 WWW 长文录用已获保证。

## 八、完整数据与实现

- [完整表格与统计](../../results/unified_q_inloop_validation_20261007/summary.md)
- [逐 cell CSV](../../results/unified_q_inloop_validation_20261007/cells.csv)
- [预注册范围](UNIFIED_Q_VALIDATION_SCOPE_ZH_2026-10-07.md)
- 核心构造：`code/v2_llm_fusion/unified_q_multimodal.py`。本次没有改动训练目标或冻结实验配置。
- 基线适配：`code/v2_llm_fusion/run_unified_q_matched_baselines_20261007.py`。
- 固定内容：`code/v2_llm_fusion/run_unified_q_fixed_content_20261007.py`。
- 汇总/审计/重跑 probe 位于 `tools/`，结果目录保留 JSON 证据和源代码 hash。
