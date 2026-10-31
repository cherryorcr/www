# 强基线与 CaT 文本组件消融的分层比较

日期：2026-10-06

## 1. 结论先行

CaT 的不同 TAG 文本适配方式本质上共享同一个 CaT class-wise MMD 特征凝聚器，只改变文本 payload 的生成或选择。因此它们用于回答：

> 在 CaT 的 `X` 不变时，文本材料化策略本身带来什么差异？

它们是 **component ablation**，不是独立的外部强基线。

论文主表应比较以下强基线/控制：

| 层级 | 方法 | 作用 |
|---|---|---|
| 主方法 | Adaptive coherent | 当前候选方法：CaT `X` + X-conditioned routing + adaptive slots |
| 原方法控制 | C0 / X+P | 衡量相对已有主线的净增益 |
| 持续 Student 控制 | Direct append | 不做凝聚改进时的持续 append 水平 |
| 真实 coreset | Random real、K-center/FPS、Medoid | 强代表性/覆盖控制，不依赖 synthetic feature optimization |
| 持续凝聚适配 | CaT-MMD-TAG、GECC-hard-TAG | 两类公开持续/演化凝聚机制的 current-delta-only 适配 |
| 小图外部参照 | PUMA-inspired | 保留 pseudo-label、one-time propagation、edge-free memory 的适配；当前仅完成小图 |

## 2. CaT 文本组件消融

| 变体 | 只改变的部分 | 是否独立强基线 |
|---|---|---:|
| CaT + nearest | 每个合成行选择 graph-nearest 当前文本 | 否 |
| CaT + TAGSAM-style chunk | 采用 chunk/MMR 风格的当前 delta 文本组合 | 否 |
| CaT + graph-conditioned | 由 `X` 兼容区域形成 soft text target 再选文本 | 否 |
| CaT + graph-text aligned | 用 current-delta 图文 ridge map 选择文本 | 否 |
| CaT + CCTP top-3/top-5 | 每个合成行保留多个合法文本来源 | 否 |
| CaT + text-center | 每类共享一个文本中心，作为分析控制 | 否，且不能与 node-specific payload 直接排名 |

这些结果放在主方法后面的消融表中，不能写成“Adaptive coherent 超过了六个独立公开方法”。

## 3. 严格大图主表的协议

- 数据集：Arxiv、SportsFit、WikiCS、BookHis；
- 协议：`base+4`，5% cumulative memory；
- 构造：current-delta-only，历史 raw graph 读取次数为 0；
- 旧 memory：frozen-prefix，不能回读 raw 或全局重凝聚；
- Student：相同 Prompt zero-shot Student、edge-aware two-layer GCN、text projection、bidirectional InfoNCE；
- Student seeds：`100..105`；
- 指标：Final Macro-F1、Accuracy、Old F1、New F1、stage-average、worst-stage、构造时间。

CaT/GECC 是机制适配版，不是官方 static/full-graph 端到端复现。GraphCLIP、GCond、SFGC、GCondenser、官方 TAGSAM/STAD 数字若使用不同 Student、完整历史图或监督分类器，只能作为 reference/adapted 表，不进入 strict 主排名。

## 4. 小图外部基线

Cora/Citeseer 上已有三共同顺序的外部适配结果，包含：

- CaT-MMD-TAG；
- GECC-hard-TAG；
- PUMA-inspired；
- K-center/FPS；
- Medoid；
- XSP / XSP + ST-Sinkhorn。

PUMA 的实现保留其 pseudo-label、one-time propagation 和 edge-free memory 机制，但它与当前 Prompt zero-shot TAG Student 的任务接口仍有适配差异。当前应报告其小图结果和协议说明，不将其未完成的大图结果留空后当成全面大图 SOTA。

## 5. 结果使用规则

1. 主表优先看同协议的外部强基线，不把 CaT 文本组件消融当作独立方法数量。
2. 同一个数据集内优先比较 Final Macro-F1、Accuracy、Old/New F1；BookHis 不能只看 Accuracy。
3. Adaptive coherent 的收益必须同时和 C0/X+P、CaT、GECC、coreset 比较，不能只和某一种文本拼接方式比较。
4. 如果新补 CaT 文本实验使用独立 construction seed，应标为 independent-seed confirmation，不与另一 construction seed 的主结果无条件合并。
5. 官方静态方法只在协议审计通过后进入 strict 主表，否则放在 reference-only 表并解释协议差异。

## 6. 文件

- 适配矩阵：[PUBLIC_BASELINE_ADAPTATION_MATRIX_20261006.md](D:/Aprojects/continue_tag_condensation/docs/current/PUBLIC_BASELINE_ADAPTATION_MATRIX_20261006.md)
- 大图主表与图：[expanded_baseline_comparison_20261006](D:/Aprojects/continue_tag_condensation/results/expanded_baseline_comparison_20261006)
- 小图外部比较：[XSP_SMALL_EXTERNAL_THREE_ORDER_COMPARISON_ZH_2026-10-04.md](D:/Aprojects/continue_tag_condensation/docs/current/XSP_SMALL_EXTERNAL_THREE_ORDER_COMPARISON_ZH_2026-10-04.md)
- 统一大图 runner：[run_public_continual_baselines_large_20261001.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py)
