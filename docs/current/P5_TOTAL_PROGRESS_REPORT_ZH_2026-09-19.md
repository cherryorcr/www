# P5 总进展报告：rate、长链、顺序与大图外推

日期：2026-09-19  
状态：**P5 第一轮执行闭环已完成；当前 C0 通过运行与审计门禁，但没有通过跨顺序/跨数据集稳定性能门禁。**

## 1. 当前冻结方法

```text
C0 = 5% fixed-rate growing memory
   + dispersion allocation
   + Mean-X
   + joint_g5_c320 center-concat text
   + frozen-old topology
   + current cross-edge, degree=1
   + response off
   + student seeds 100..105
```

所有本轮结果使用 report-dev Macro-F1，test 未读取。C1、attention、response、bounded refresh 没有被事后合入 C0。

**指标审计补充：** 下表 old/new 均为 **task-slice F1**，排除其他任务样本产生的误报，old 按旧任务等权。它们不能与五层表的 pooled old/new 直接比较，也不能加权还原 cumulative。C0 与历史 dispersion 的 6 对 memory、36 个逐 seed 累计分数一致；详见 [P5 指标口径审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)。该审计未新增训练。

## 2. P5 已完成项目

| 项目 | 数据/协议 | 状态 | 结论 |
|---|---|---|---|
| Rate curve | Cora/Citeseer，1%/5%/10%，三阶段 | 已完成 | 5% 是当前较稳工作点；rate 非单调 |
| 长链 | Cora/Citeseer，5-stage/base+4 | 已完成 | 最后阶段累计分数回落，原因需区分类别集合、顺序与预算影响 |
| 新顺序诊断 | Cora/Citeseer，frequency 三阶段 | 已完成一组，非预注册确认 | Cora +1.76 pp，Citeseer -6.38 pp，顺序依赖明显 |
| 大图 base+4 | arXiv/BookChild/SportsFit，5-stage | 已完成首轮 | 可以运行，但累计 F1 数据集依赖，不能宣称通用长链稳定 |
| 成本字段 | student steps、train time、stage build time | 已记录部分成本 | 大图已记录重训高于构造；旧运行未记文本编码，不能补作零秒 |

## 3. 小图 rate curve

| 数据集 | 1% final F1 | 5% final F1 | 10% final F1 | 5% final rows | 5% task-slice old/new F1 |
|---|---:|---:|---:|---:|---:|
| Cora | 42.00 | **49.85** | 45.38 | 94 | 75.05/40.91 |
| Citeseer | 52.20 | **59.17** | 55.82 | 111 | 76.67/55.48 |

含义：增加 memory 并不会自动改善 student；10% 比 5% 更差，说明压缩率应作为独立实验变量，而不是越大越好。1% 首阶段真实有效 batch 小于 16，不能把低 rate 点解释成完全匹配 batch 的对照。

## 4. 小图 5-stage

| 数据集 | memory trajectory | stage 0 F1 | stage 3 F1 | final F1 | final task-slice old/new F1 |
|---|---|---:|---:|---:|---:|
| Cora | 57/70/80/88/94 | 39.62 | 49.36 | 40.71 | 54.45/37.42 |
| Citeseer | 47/67/86/102/111 | 60.53 | 66.29 | 54.78 | 74.55/23.68 |

含义：长链最后阶段只增加少量 rows，且单类阶段的 task-slice new F1 波动很大；容量分配不足是一种待验证解释，不能仅凭这些指标归因于预算规则或遗忘。

## 5. 独立顺序

| 数据集 | canonical final F1 | frequency final F1 | 差值 |
|---|---:|---:|---:|
| Cora | 49.85 | 51.61 | +1.76 |
| Citeseer | 59.17 | 52.79 | -6.38 |

这组 frequency 分组是新的诊断条件，不应称作已经预注册的独立确认；它足以证明 canonical 分数存在顺序依赖，但不足以给出顺序平均性能。

## 6. 大图 base+4

| 数据集 | memory trajectory | initial F1 | final cumulative F1 | final task-slice old/new F1 | build s | student train s | artifact bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| arXiv | 4962/5246/5502/5734/5926 | 29.62 | 18.82 | 33.63/21.30 | 61.16 | 146.80 | 354,845,166 |
| BookChild | 2344/2458/2551/2629/2690 | 16.85 | 13.22 | 30.86/44.54 | 31.21 | 159.44 | 161,948,339 |
| SportsFit | 4491/4965/5430/5802/6056 | 30.34 | 27.90 | 49.47/58.42 | 52.29 | 154.52 | 357,924,852 |

三套大图均为 `5×6×2048=61,440` 个 student optimizer steps。这里的 `build s` 是阶段构造/聚类时间；旧版本实验的全池 MiniLM 编码未计入该字段，新 runner 已增加 `text_embedding_seconds` 字段供后续成本账使用。旧运行缺失时间显示为空值，不是零；已记录耗时只是总成本下界。

## 7. 综合判断

### 已经可以确定

1. frozen-old append 的历史访问边界是可实现、可审计的；小图 artifact 审计显示旧 synthetic `x/text/mass/depth/stage` 前缀和旧边集合保持。
2. C0 在 5% 下能够跨小图和大图运行，且不依赖 response selector 才能完成持续构造。
3. 5% 不是由单调 rate 曲线支持的“最优理论压缩率”，只是当前协议下最稳的工作点。
4. 长链和类别顺序会显著影响最终 F1；因此单一 canonical 顺序不能作为通用性能证据。
5. 大图首轮已记录的 student 重训时间高于阶段构造，但全池文本编码等成本缺失，不能声称端到端主要成本已完整定位；加入 response 选择还需计入额外训练。

### 目前不能宣称

- 不宣称 SOTA；
- 不宣称普遍超过 k-center、herding 或 kmeans-medoid；
- 不宣称 fixed-rate 在所有长链和数据集上稳定；
- 不宣称大图 base+4 的最终分数可与旧 canonical 三阶段分数直接横比；
- 不把 C1、文本 attention、response 或 refresh 组合成一个未经独立验证的新版本。

## 8. 后续建议

1. 下一轮先保存预测/logits 并补完整成本，再预注册一组**按阶段 source-size 匹配**的独立顺序，不继续无界扫描手工 X/文本模块。
2. 统一五层对照（持续 append、全历史重凝聚、联合重凝聚、强 coreset、完整 raw 参考）应单独标明是三阶段还是 base+4，不能跨协议混表。
3. 大图后续若只保留一轮扩展，优先补成本完整性和第二顺序；若成本/篇幅允许，再做大图 3-stage quick ablation。
4. 只有某个候选组件在独立顺序、成本和长链上同时通过，才建立 v5；否则论文主线保持 v4/C0 的机制分析与负结果定位。

## 9. 产物索引

- [P5 小图扩展完成报告](P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md)
- [P5 独立顺序报告](P5_INDEPENDENT_ORDER_COMPLETION_ZH_2026-09-19.md)
- [P5 大图 base+4 报告](P5_LARGE_BASE4_COMPLETION_ZH_2026-09-19.md)
- [当前 C0 与五层基线审计](P5_CURRENT_C0_BASELINE_AUDIT_ZH_2026-09-19.md)
- [P5 指标口径审计](P5_METRIC_SCOPE_AUDIT_ZH_2026-09-19.md)：1,558 checks / 0 errors，C0/v4 与 pooled/task-slice 分列
- 小图 rate 原始结果：`results/p5_rate_curve_small_20260919/`
- 小图长链原始结果：`results/p5_longchain_small_20260919/`
- 独立顺序原始结果：`results/p5_independent_order_small_20260919/`
- 大图结果 manifest：`results/p5_large_base4_20260919/*/results.json`
- 大图完整 artifact 远程目录：`/home/pc/research/TAG_condensation/results/p5_large_base4_20260919/`
