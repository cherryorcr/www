# P5 大图 base+4 完成报告

日期：2026-09-19  
状态：**arXiv、BookChild、SportsFit 的冻结 C0 5-stage/base+4 首轮已完成；不构成 v5 晋升或 SOTA 声明。**

## 1. 实验范围

本轮把小图验收后的冻结 C0 直接外推到三个大图数据集。没有加入 C1、P2 attention、response、refresh 或新的 X 算子。

```text
5% fixed-rate
+ dispersion allocation
+ Mean-X
+ joint_g5_c320 center-concat
+ frozen-old topology + current cross-edge (degree=1)
+ response off
+ 6 student seeds: 100..105
+ protocol-scaled student steps, cap 2048
+ base + 4 incremental stages
```

指标为 report-dev Macro-F1；本轮 test 未读取。大图的 `base+4` 是类别增量协议，不等同于此前 canonical 三阶段表，因此只能做机制和长链对照，不能把两个数字直接当作同一 setting 的替代。

**口径更正：** 本文 old/new 为 task-slice F1，不计其他任务样本对该任务产生的误报，old 对旧任务等权。不能与历史 pooled old/new 拼表，也不能用来分解 cumulative；详见 [指标审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。累计分数不因这项说明而改变。

## 2. 最终结果

| 数据集 | memory budgets | stage 0 F1 | 最终 cumulative F1 | old task-slice F1 | new task-slice F1 | build s | student train s | student optimizer steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arXiv | 4962/5246/5502/5734/5926 | 29.62 | 18.82 | 33.63 | 21.30 | 61.16 | 146.80 | 61,440 |
| BookChild | 2344/2458/2551/2629/2690 | 16.85 | 13.22 | 30.86 | 44.54 | 31.21 | 159.44 | 61,440 |
| SportsFit | 4491/4965/5430/5802/6056 | 30.34 | 27.90 | 49.47 | 58.42 | 52.29 | 154.52 | 61,440 |

## 3. 阶段轨迹

| 数据集 | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|---:|
| arXiv cumulative F1 | 29.62 | 26.52 | 22.15 | 21.61 | 18.82 |
| BookChild cumulative F1 | 16.85 | 18.35 | 13.18 | 15.15 | 13.22 |
| SportsFit cumulative F1 | 30.34 | 31.45 | 29.52 | 28.71 | 27.90 |

对应的每阶段 new task-slice F1：

| 数据集 | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|---:|
| arXiv | 29.62 | 40.50 | 31.87 | 36.74 | 21.30 |
| BookChild | 16.85 | 49.31 | 14.89 | 41.68 | 44.54 |
| SportsFit | 30.34 | 66.83 | 56.95 | 42.86 | 58.42 |

## 4. 如何解释

### 4.1 arXiv

arXiv 在 base+4 下累计 F1 从 29.62 降到 18.82，最终 task-slice old/new 为 33.63/21.30。base 包含 16 类，后续每阶段新增 192--284 行；预算、长尾与阶段竞争都值得诊断，但尚不能证明预算不足是原因。类别集合变化和指标切片限制也使得这条曲线不能直接分解为遗忘与塑性损失。

### 4.2 BookChild

BookChild 最终为 13.22，低于 stage 1 的 18.35 和 stage 3 的 15.15。最后 new task-slice F1=44.54 不计旧任务样本误报到新类的 FP，不能据此断言主要问题在旧类或新类已学好；需 pooled 指标和跨任务混淆矩阵才能定位。它也不能与已有 canonical 三阶段分数直接横比，因为类别块和预算轨迹不同。

### 4.3 SportsFit

SportsFit 最终累计 F1 为 27.90，阶段曲线相对平滑；new task-slice F1=58.42 仍不能替代全局新类评分。累计指标从 31.45 回落，且只有一条类别顺序，因此不支持“fixed-rate 在长链上普遍无损”或已通过稳定性验收。

## 5. 成本结论

- 三个数据集每个都执行 `5 stages × 6 student seeds × 2048 steps = 61,440` 个 student optimizer steps。
- student 重训时间为 `146.80--159.44s`，显著高于后续阶段的构造时间；stage 0 的聚类/构造占主要 `stage_build_seconds`。注意全池 MiniLM 文本编码发生在阶段循环之前，本轮没有单独计入 `stage_build_seconds`，因此这里的成本仍是下界。
- 因此当前方法的优势更接近“构造本身便宜、历史访问受限、可审计”，而不是“一次完整流程总时间一定更快”。如果加入 response selector，student 训练成本还会进一步增加。

## 6. 与已有 canonical 三阶段证据的关系

此前统一主表已经有 arXiv、BookChild、SportsFit 的 canonical 三阶段 append 结果。该表可用于说明小阶段链上的基线水平，但不能把它与本轮 base+4 直接合并成一条时间曲线：

- canonical 三阶段与本轮 `base+4` 的类别块不同；
- 每阶段原始 source rows 不同；
- 最后阶段新增类别数不同；
- 因而预算进入时间和 old/new F1 的含义不同。

本轮新增证据回答的是：**当大图采用较长的 base+4 类别增量链时，冻结 C0 是否仍能运行、历史是否被实现层面保留、性能和成本如何变化。**答案是：可以运行且历史 synthetic 前缀/旧边保持，但最终 F1 存在数据集依赖和长链回落。

## 7. 当前评价与后续决定

1. 大图 base+4 首轮运行完成，记录了部分成本；缺失文本编码等耗时，不是完整成本验收通过。
2. 它没有通过“长链性能稳定”门禁，尤其是 arXiv 和 BookChild 的累计 F1 回落。
3. 不能因为 SportsFit 相对平滑，就把 C0 宣称为通用长链方案。
4. 当前最合理的论文定位仍是：可审计的 frozen-old continual append、预算增长和保持-塑性边界分析；不宣称 SOTA。
5. 后续如继续大图，应优先补同一数据集的第二个预注册类别顺序，或做按阶段 source-size 匹配的顺序对照；不应先把复杂候选模块组合进 base+4。

## 8. 产物

- 大图结果 manifest：`results/p5_large_base4_20260919/*/results.json`
- 大图汇总：`results/p5_large_base4_20260919/summary/P5_LONGCHAIN_SUMMARY_ZH.md`
- 小图 rate/长链报告：[P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md](P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md)
- 小图独立顺序报告：[P5_INDEPENDENT_ORDER_COMPLETION_ZH_2026-09-19.md](P5_INDEPENDENT_ORDER_COMPLETION_ZH_2026-09-19.md)
- 旧 synthetic 审计原始目录（远程）：`/home/pc/research/TAG_condensation/results/p5_large_base4_20260919/`
