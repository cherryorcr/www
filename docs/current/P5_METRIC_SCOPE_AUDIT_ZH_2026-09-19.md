# P5 指标口径与 C0/v4 版本审计

日期：2026-09-19  
范围：已存 artifact 和指标聚合复核；本轮没有重新训练、重放原始预测或读取 test 标签。

## 1. 核心结论

**累计 F1 主表没有因此失效，但此前 P5 的 old/new 与五层表不是同一种指标，不能直接比较。** 这也解释了为何大图报告中 old 和 new 都不低，累计 F1 却很低：三者不是同一支持集上的可加权分解。

当前共同控制仍为 `C0=response off`；历史 `v4=response on` 单列，不把其额外收益计入 C0。审计不修改冻结 evaluator，因为该函数还参与历史 response 选择，直接替换会改变算法和实验条件。

## 2. 两种指标的差别

| 项目 | P5 task-slice 口径 | 历史五层 pooled 口径 |
|---|---|---|
| 预测 | 全类别 prompt 中取 argmax | 分别保存 all/seen prompt 的预测指标 |
| old/new 的样本集 | 按真实标签筛出每个任务的样本，分别计算 | 全部已见 report-dev 样本一起计算 per-class F1 |
| 跨任务误报 | 其他任务样本被错判到本任务，不计入本任务 FP | 计入相应类别 FP |
| old 汇总 | 旧任务等权均值 | 旧类别等权均值 |
| cumulative | 全部已见样本一起算，使用目标中出现的类 | 全部已见样本一起算，使用协议指定的已见类 |

例如真实标签 `[0,0,1,1]`、预测 `[0,0,0,1]`，任务分别为类 0 和类 1。只取类 0 的真实样本时，old task-slice F1 是 100；全局计算时，类 1 被错判成类 0 产生一个 FP，old pooled F1 是 80。预测并没有改变，只是支持集不同。

若每个已见类别都有报告样本，两种 cumulative 定义一致；若某类没有报告样本，是否仍把该类计入宏平均也会产生差别。不能把小图已验证的一致性无条件推广到所有划分。阶段间 all/seen prompt 也不能混用；最终全部类别已见时才一致。

代码来源：

- P5：`run_unified_v3_continual.py` 调用 `code/frozen_v1_2026_09_09/run_gate1_continual.py::eval_stagewise`。
- 五层：`code/v2_llm_fusion/small_comparison_metrics.py::score_predictions`。

## 3. 当前 C0 与历史记录的对账

对 Cora/Citeseer 的 5% canonical 三阶段，已执行 1,558 项检查，错误数为 0：

- 原始数据 SHA256、类别阶段、累计预算、batch、student seeds 相符；C0 为 response off。
- 6 对阶段 memory 的 `x/text_emb/edge_index/mass/depth/stage` dtype 与 tensor 完全一致，文本字符串相同。
- 36 个逐 seed 阶段结果的 student steps、累计 F1、accuracy 与历史 `dispersion` 一致，浮点比较容差为 `1e-10`。
- C0 均值和总体标准差复算一致；历史 core/static/supervised 已保存记录的 old/new、cumulative、per-task 聚合均从 per-class F1 独立复算一致。

因此，可以把历史 dispersion 的全局指标作为**同构造、逐 seed 累计分数匹配的归档参照**，但不能称为“从本轮 P5 原始预测重新计算”。P5 没有保存 student checkpoint 或逐节点预测，只有 memory 和标量；同累计分数并不能在数学上保证预测逐点相同。

| 数据集 | C0 累计 F1 | P5 old/new：task-slice | 同构造历史 dispersion old/new：pooled | 历史 v4 old/new：pooled | raw 长训选点 old/new：pooled |
|---|---:|---:|---:|---:|---:|
| Cora | 49.85 | 75.05 / 40.91 | 65.44 / 29.06 | 65.44 / 29.06 | 60.85 / 38.59 |
| Citeseer | 59.17 | 76.67 / 55.48 | 66.46 / 44.61 | 68.00 / 46.29 | 63.90 / 39.01 |

只有后三列可以在其共同 pooled 口径内对比。Cora 历史同构造参照的新类缺口约 9.53 pp；不是拿 P5 的 40.91 与 raw 的 38.59 比较后宣称新类已超过完整数据。

## 4. 同 student 的性能保留率

| 项目 | Cora | Citeseer |
|---|---:|---:|
| C0 合成节点 / 完整合法 pool | 94 / 1,895 | 111 / 2,229 |
| 当前 C0 最终 F1 | 49.85 ± 3.00 | 59.17 ± 1.57 |
| 历史 v4 最终 F1 | 49.85 ± 3.00 | 60.77 ± 1.64 |
| 同 student 完整 pool 长训选点 F1 | 51.31 ± 2.89 | 55.60 ± 1.92 |
| C0 最终性能保留率 | **97.16%** | **106.42%** |
| C0 阶段平均性能保留率，all prompts | **88.71%** | **105.43%** |
| 历史 v4 最终性能保留率 | 97.16% | 109.28% |

保留率按未舍入的均值相除，不是逐 seed 比值均值；Citeseer 此前写成 106.43% 的舍入值更正为 106.42%。超过 100% 仅指超过当前经验完整数据参考，不是超越理论上限。约 5% 是节点比例，不是总训练时间、文本字节或边存储比例。canonical 单顺序的结果也不能取代多顺序证据。

## 5. 大图和成本结论的修正

大图 arXiv/BookChild/SportsFit 的 18.82/13.22/27.90 是原始累计 F1，保持不变；33.63/21.30、30.86/44.54、49.47/58.42 分别是 task-slice old/new，不是 pooled 分解。特别是 BookChild 的 new=44.54 **不能单凭这一项证明问题主要来自旧类，或新类塑性已经良好**。

阶段之间类别集合变化，因此累计曲线下降不能单独诊断遗忘。下一轮机制分析至少需要保存逐节点预测或 logits，统一报告 pooled old/new 与 task-slice，才可观察跨任务混淆和历史类轨迹。需要重评时，应固定 memory 并明确标注为新评估，不回填为原始 P5 预测。

历史运行缺失的 `text_embedding_seconds`、`stage_build_seconds` 现在保留为空值，不能用 0 秒代替。大图已记录的 student 时间大于阶段构造时间，但文本编码等未完整计时，所以不能据此声称端到端主要成本已全部定位，或总计算比完整图更省。

## 6. 验收范围与后续顺序

本轮完成指标语义、版本、已存 artifact 和聚合数值审计。它不替代历史 checkpoint 复评，不宣称重跑了原有五层验收，也不为大图补出不存在的 pooled 数值。P5 仍是第一轮执行闭环完成、稳健性能未过门禁。

工程验收：`test_p5_metric_scope.py`、`test_small_comparison.py` 与 `test_five_level_runners.py` 合计 **18 passed / 2 skipped**。跳过项依赖 CUDA，本机 CPU 环境未执行；测试覆盖跨任务 FP、任务/类别权重、缺失类别、聚合损坏检测、缺失成本及真实阶段标签。四组 P5 派生汇总已重生成，共 49 条阶段记录、13 个运行摘要，未改动原始 `results.json`。7 份报告/论文的 42 个本地链接及 Markdown 表格列数校验通过。

后续顺序：先让新增评估保存预测和完整成本，再做按阶段 source-size 匹配的第二顺序；保持 C0 不变。外部 TAGSAM/STAD 补强继续暂停。只有控制变量和独立确认支持候选，才考虑新版本；不能根据已有开发结果事后拼装 v5。

证据与复现：

- [审计 JSON](../../results/p5_metric_scope_audit_20260919/audit.json)、[逐口径对照 CSV](../../results/p5_metric_scope_audit_20260919/comparison.csv)。JSON 保存输入和 evaluator 文件 SHA256。
- [审计脚本](../../code/v2_llm_fusion/audit_p5_metric_scope.py)、[回归测试](../../code/v2_llm_fusion/test_p5_metric_scope.py)。
- 执行：`python code/v2_llm_fusion/audit_p5_metric_scope.py --output results/p5_metric_scope_audit_20260919`。
- [P5 总进展](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)、[当前 C0 基线表](P5_CURRENT_C0_BASELINE_AUDIT_ZH_2026-09-19.md)。
