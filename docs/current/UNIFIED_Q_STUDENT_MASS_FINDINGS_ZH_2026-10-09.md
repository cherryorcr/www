# Unified-Q student_mass 单因素实验报告

日期：2026-10-09  
状态：development，未封存测试，不用于冻结主方法结论

## 1. 实验目的

本轮只检验一个构造端假设：当前 GM proxy 使用 group_class 权重，可能与下游 Student 对 synthetic memory 的均匀采样不一致。student_mass 保持 synthetic slot 等权，并按 synthetic memory 中每个 (stage, class) 组的 slot 质量，把 reference raw rows 的权重重新分配到对应组。

本轮没有改变：

- Q 的初始化、class mask、Sinkhorn 和容量约束；
- X residual、CaT-style distribution matching、GM、propagation 和 anchor；
- 文本候选、straight-through hard materialization 和 320 字符限制；
- 拓扑、历史 raw 访问边界和 frozen prefix；
- 下游 Prompt-GCN、Prompt-SGC、训练步数、batch 和 Student seeds。

因此它是一个单因素归因实验，而不是新主方法。

## 2. 实现

核心实现位于 [unified_q_multimodal.py](../../code/v2_llm_fusion/unified_q_multimodal.py)：

- _proxy_weights()：为 reference 和 synthetic proxy 生成权重；
- _loss()：让 pair loss、prompt CE 和 margin 使用同一套 reference sample weights；
- optimize_stage()：只在构造端 GM 的真实侧使用 student_mass，最终 Student 训练协议不变；
- audit["loss_weighting"]、real_new_loss_mass 和 synthetic_new_loss_mass 记录审计信息。

边界处理为：synthetic 中必须出现的 (stage, class) 组必须在 reference 中存在；reference 中有、但预算没有分配 synthetic slot 的额外组权重为零，不再错误抛出集合不一致异常。测试覆盖见 [test_unified_q_multimodal.py](../../code/v2_llm_fusion/test_unified_q_multimodal.py)。

## 3. 协议

覆盖 WikiCS、SportsFit、BookHis；共 2 个 construction seeds：

- WikiCS：order 9201，seeds 271108/271109；
- SportsFit：order 9201，seeds 271108/271109；
- BookHis：order 9202，seeds 271108/271109。

每个 cell 使用 Prompt-GCN 和 Prompt-SGC，Student seeds 为 150/151/152，2048 steps，batch 24。source memory 来自已经审计的 uq_text_path_20261008 主方法归档；新输出只改变 loss_weighting。所有 6 个 cell 的 changed_constructor_fields 均为 ["loss_weighting"]。

所有分数是 final-memory report-dev、transductive readout，不是 sealed test，也不是严格 unseen-class zero-shot。

## 4. 逐 cell 结果

数值为 Final Macro-F1；括号内是 student_mass - group_class，单位为百分点。

| 数据集 | seed | Prompt-GCN 原方法 | Prompt-GCN student_mass | Delta | Prompt-SGC 原方法 | Prompt-SGC student_mass | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| WikiCS | 271108 | 79.57 | 80.01 | +0.44 | 77.46 | 76.90 | -0.56 |
| WikiCS | 271109 | 79.96 | 80.25 | +0.29 | 78.98 | 77.70 | -1.27 |
| SportsFit | 271108 | 78.73 | 76.05 | -2.68 | 72.85 | 72.49 | -0.36 |
| SportsFit | 271109 | 77.00 | 77.30 | +0.29 | 73.82 | 74.90 | +1.09 |
| BookHis | 271108 | 41.94 | 41.63 | -0.32 | 35.62 | 35.49 | -0.13 |
| BookHis | 271109 | 41.21 | 41.31 | +0.09 | 38.14 | 37.14 | -1.00 |
| **6-cell mean** |  | **66.41** | **66.09** | **-0.31** | **62.81** | **62.44** | **-0.37** |

均值行已按六份原始 JSON 的 `rows[].score.final_f1` 重新核算，更正此前汇总抄录错误；差值从未舍入的逐 cell 分数计算。逐 cell 原始结果与候选不升级的结论不变。

Accuracy 的平均变化：

| Student | 平均 Accuracy Delta | 正向 cells |
|---|---:|---:|
| Prompt-GCN | -0.37 pp | 5/6 |
| Prompt-SGC | -0.24 pp | 3/6 |

Old/New F1 的平均变化：

| Student | Old F1 Delta | New F1 Delta |
|---|---:|---:|
| Prompt-GCN | -0.31 pp | +0.02 pp |
| Prompt-SGC | -0.44 pp | +0.66 pp |

构造成本没有得到系统改善。6 个 cell 中 candidate 相对原方法的构造时间约为 0.96x--1.14x，差异主要来自 GPU/稀疏计算波动；student_mass 没有形成可报告的效率收益。

## 5. 结论

1. student_mass 没有稳定提升 Final Macro-F1。Prompt-GCN 和 Prompt-SGC 的跨数据集平均变化分别为 -0.31 pp 和 -0.37 pp。
2. 它可能在少数条件下改善新类或 Accuracy，但收益不稳定，并伴随旧类保持下降；不能把局部正向 cell 解释为通用改进。
3. 当前 GM 代理与 Student 采样分布不一致的诊断仍然成立，但“简单地按 synthetic slot mass 重加权”不足以解决这个问题。问题可能在更深层的 proxy/readout mismatch、multi-positive pair 结构或 full-batch proxy 对实际 minibatch trajectory 的近似方式。
4. 不升级 student_mass，继续保留为负结果/诊断结果。当前暂定主线仍是 edge-free In-loop CaT-Q，默认 group_class、raw feature pooling、原有文本材料化和已审计配置。

## 6. 后续安排

不再继续扫描 student_mass 的参数。下一步优先级：

1. 独立类别顺序、construction seed 和 sealed-test 证据；
2. 真正不同的 Student architecture readout，而不是只改变 loss family；
3. 成本、预算曲线、旧类重新到达和新旧混合到达；
4. 若仍要研究 GM proxy，只做一个更有针对性的改动：让 proxy 保留类内图文配对/覆盖信息，并设置 paired 与 within-class shuffled control；
5. 不同时加入 EM、拓扑、额外 text alignment 或新的 Student，避免无法归因。

完整原始 JSON 保存在 results/uq_student_mass_20261009/，远程对应输出位于 /home/pc/research/TAG_condensation/results/ 下的 uq_student_mass_20261009_* 目录。
