# P5 独立类别顺序小图验收报告

日期：2026-09-19  
状态：**已完成一组新的 frequency 三阶段诊断顺序，不是预注册独立确认。后续大图首轮已执行，但没有因此补齐稳健性门禁，见 [P5 总报告](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)。**

## 1. 实验目的

验证 C0 的 5% 结果是否依赖 Cora/Citeseer 的 canonical 类别块。新运行不复用 canonical 固定分组，采用统一 runner 的确定性 `frequency` 分组，其他设置全部保持一致：

```text
5% fixed-rate + dispersion + Mean-X + joint_g5_c320 center-concat
+ frozen-old topology + cross-degree=1 + response off
+ student seeds 100..105, batch=16, protocol-scaled steps
```

## 2. 结果

old/current 为 task-slice F1，old 对任务等权且不计跨任务误报；不能与历史五层 pooled old/new 比较。详见 [指标审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。

| 数据集 | 顺序 | stage train counts | budgets | stage 0 F1 | stage 1 F1 | final cumulative F1 | old task-slice F1 | current task-slice F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| Cora | canonical | 501/858/536 | 25/67/94 | 58.11 | 51.29 | 49.85 | 75.05 | 40.91 |
| Cora | frequency | 754/441/700 | 37/59/94 | 32.64 | 55.11 | 51.61 | 66.85 | 60.16 |
| Citeseer | canonical | 778/953/498 | 38/86/111 | 64.52 | 69.16 | 59.17 | 76.67 | 55.48 |
| Citeseer | frequency | 874/710/645 | 43/79/111 | 76.08 | 64.91 | 52.79 | 75.41 | 37.92 |

最终累计 F1 相对 canonical 的变化为：

| 数据集 | frequency - canonical |
|---|---:|
| Cora | +1.76 pp |
| Citeseer | -6.38 pp |

## 3. 结果说明

1. Cora 的 frequency 顺序最终略高，但 stage 0 明显更低，说明最终平均值掩盖了早期阶段的困难；Citeseer 则出现最终明显下降。
2. `current F1` 不能跨顺序直接作为“新类能力”排名，因为最后阶段的类别集合变了。它只适合在同一顺序内诊断塑性。
3. 两个数据集的 stage source counts、预算进入时间和类别组合都发生了变化，因此该实验是顺序稳健性诊断，不是一个只改变 permutation、其它样本量完全相同的因果实验。
4. 当前证据足以说明 canonical 结果存在顺序依赖；还不足以宣称 C0 在所有类别到达顺序下稳定。

## 4. P5 决策

- C0 继续作为共同控制，不建立 v5。
- 不把 frequency 顺序的 Cora 高分或 Citeseer 低分单独包装成方法优势/失败。
- 当时建议在大图前再冻结类别块规则；后续虽已完成大图首轮，这项按阶段样本量匹配的独立确认仍未补齐，应作为下一轮工作而不是追认完成。
- 在论文中应报告 canonical 与独立顺序的范围、均值和失败条件，而不是只报告一个最好顺序。

## 5. 产物

- 原始结果：`results/p5_independent_order_small_20260919/`
- 更新后的汇总表：`results/p5_independent_order_small_20260919/summary/P5_LONGCHAIN_SUMMARY_ZH.md`
- canonical rate 对照：`results/p5_rate_curve_small_20260919/summary/`
