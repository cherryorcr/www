# P5 小图扩展完成报告（rate curve 与 5-stage 长链）

日期：2026-09-19  
状态：**本报告为小图阶段记录。后续 frequency 顺序诊断和大图 base+4 首轮也已完成，最新状态见 [P5 总报告](P5_TOTAL_PROGRESS_REPORT_ZH_2026-09-19.md)；不代表稳健性能门禁通过。**

## 1. 本轮目的

本轮只验证冻结 C0 在不同 memory rate 和更长持续链下是否仍然可解释、可复现。它不是候选模块晋升实验，也不改变 v4 主线。

冻结协议为：

```text
dispersion allocation
+ Mean-X
+ joint_g5_c320 center-concat
+ frozen-old topology
+ current cross-edge, degree=1
+ response off
+ student seeds 100..105
+ student batch=16, protocol-scaled steps
```

指标为 report-dev Macro-F1；test 未读取。1% rate 下首阶段 memory 小于 batch=16，因此同时记录了配置 batch 和实际有效 batch。

本文 old/new 均为 task-slice F1，不能与历史五层表的 pooled old/new 直接比较；old 按任务等权，跨任务误报被切片排除。累计 F1 仍在全部已见样本上计算，详见 [指标审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。

## 2. Rate curve

### 2.1 最终阶段

| 数据集 | rate | 最终 rows | cumulative F1 | old task-slice F1 | new task-slice F1 | student optimizer steps | student train s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 1% | 18 | 42.00 | 62.00 | 41.50 | 3,456 | 8.38 |
| Cora | 5% | 94 | 49.85 | 75.05 | 40.91 | 17,856 | 38.39 |
| Cora | 10% | 189 | 45.38 | 72.48 | 30.62 | 29,376 | 63.25 |
| Citeseer | 1% | 22 | 52.20 | 66.61 | 48.82 | 4,416 | 10.59 |
| Citeseer | 5% | 111 | 59.17 | 76.67 | 55.48 | 22,560 | 47.80 |
| Citeseer | 10% | 222 | 55.82 | 71.65 | 52.54 | 31,968 | 69.57 |

### 2.2 解释

- 5% 是两个小图当前最稳的工作点，但不能表述成全局最优 rate。
- 10% 没有带来收益：Cora 从 49.85 降到 45.38，Citeseer 从 59.17 降到 55.82；更多 memory 同时带来更多合成行和更多 student 更新，并没有自动改善新类塑性。
- Cora 1% 的旧类保持较弱且最终新类仍受限；Citeseer 1% 也明显低于 5%。因此 rate 曲线是非单调的，后续不应把 memory 扩容本身包装成方法改进。
- 1% 首阶段实际 batch 分别为 Cora 5、Citeseer 7，后续阶段才回到 16。该点不是与 5%/10% 完全匹配 batch 的比较，应作为有效 batch 差异单独标注。

## 3. 5-stage/base+4 长链

阶段划分采用 `base_incremental`，Cora 和 Citeseer 各为一个 base 加四个单类增量阶段，rate 固定为 5%。

| 数据集 | memory budgets | stage 0 F1 | stage 1 F1 | stage 2 F1 | stage 3 F1 | stage 4 F1 | old task-slice F1 | new task-slice F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 57/70/80/88/94 | 39.62 | 36.50 | 45.40 | 49.36 | 40.71 | 54.45 | 37.42 |
| Citeseer | 47/67/86/102/111 | 60.53 | 69.05 | 67.64 | 66.29 | 54.78 | 74.55 | 23.68 |

### 3.1 解释

- 长链最后阶段出现明显回落：Cora 从 49.36 降到 40.71，Citeseer 从 66.29 降到 54.78。
- 最后一类只获得少量新增 rows：Cora 6 行，Citeseer 9 行；Citeseer 最后 new task-slice F1 为 23.68，提示新类表现值得关注，但容量不足是否为主要原因仍需受控对照。
- Cora stage 1 的单类 `[6]` current F1 为 0。该类存在 report-dev 样本但本轮 student 没有命中，说明单类增量阶段的指标方差和类别顺序影响很强，不能与三阶段多类 current F1 直接等价比较。
- 这组结果不能单独证明 fixed-rate 失效，因为长链同时改变了类别块、阶段源规模和到达顺序；它证明的是：**在 base+4 长链下，C0 的最后阶段塑性不稳定，必须补独立顺序后再判断是否能上大图。**

## 4. 成本观察

- Cora 5-stage 总 student 更新为 37,344，训练时间约 79.69 秒。
- Citeseer 5-stage 总 student 更新为 39,648，训练时间约 85.82 秒。
- 长链比三阶段增加了重训次数和阶段轨迹长度；构造器本身仍然很快，主要成本来自每个阶段对 6 个 student seeds 的完整重训。
- 当前成本字段只统计 student 重训；文本编码、构造、磁盘和显存峰值仍需在大图主表中统一补齐。

## 5. 小图批次当时的结论与后续状态

以下第 3--4 项是本批次当时的推进门禁。随后已执行 frequency 顺序诊断并扩展大图首轮，但 frequency 不是预注册独立确认，不能把“已运行”追认为全部性能门禁通过。

1. C0 在 5% 下可复现，但 rate 与持续长度都存在明显数据集/顺序依赖。
2. 不把 10% 或 5-stage 结果用于晋升新的 v5 组件。
3. 大图前必须补至少一组未参与调参的独立类别顺序；否则无法区分“长链本身的困难”和“base+4 具体顺序造成的失败”。
4. 在扩展大图前，下一步优先级是：独立顺序小图确认、统一五层基线成本表，然后再决定 base+4 是否进入 arXiv/BookChild/SportsFit。

## 6. 产物

- Rate 原始结果：`results/p5_rate_curve_small_20260919/`
- Rate 阶段/最终汇总：`results/p5_rate_curve_small_20260919/summary/`
- 长链原始结果：`results/p5_longchain_small_20260919/`
- 长链阶段/最终汇总：`results/p5_longchain_small_20260919/summary/`
- 汇总脚本：`code/v2_llm_fusion/summarize_p5_rate_curve.py`
- 长链汇总脚本：`code/v2_llm_fusion/summarize_p5_longchain.py`
