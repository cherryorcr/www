# 持续 TAG 凝聚基线适配矩阵

日期：2026-10-06

本文档记录外部方法如何适配到当前持续 TAG 凝聚协议。这里的“适配”不是声称复现原论文的完整系统，而是明确保留哪些机制、删除哪些不满足协议的访问路径，以及哪些结果可以进入严格主表。

## 1. 统一严格协议

所有进入严格主表的运行必须满足：

- `base+4` 类别到达协议；
- 每阶段 5% cumulative memory budget；
- 只访问当前 raw delta 和上一阶段 synthetic memory；
- `historical_raw_reads=0`；
- 旧 synthetic memory frozen-prefix，不能被重新读取 raw 或全局重凝聚；
- 不使用 report/test labels 构造 memory；
- 同一个 Prompt zero-shot Student：edge-aware two-layer GCN、graph/text projection 和 bidirectional InfoNCE；
- Student seeds `100..105`、batch 16、同一训练步数和同一 report/test 规则；
- 统一报告 Final Macro-F1、Accuracy、Old F1、New F1、stage-average F1、worst-stage F1 和构造成本。

## 2. 原方法到当前协议的适配

| 原方法/控制 | 原始核心机制 | 当前适配保留内容 | 为满足持续 TAG 协议做的改动 | 历史 raw | 修改旧 memory | 拓扑 | 严格主表 |
|---|---|---|---|---:|---:|---|---:|
| CaT | class-wise MMD 梯度优化 synthetic features | 使用 MMD 目标优化每个类别的 `X` | 每阶段只用 current delta；为每个 synthetic row 附加 TAG 文本；不使用官方静态全图流程 | 0 | 否 | edge-free | 是 |
| CaT + nearest payload | CaT 的 `X` + 图空间最近文本 | 保留 CaT `X` 和 nearest source-text assignment | 文本只能来自当前 delta；每个 synthetic row 单独匹配文本 | 0 | 否 | edge-free | 是 |
| CaT + TAGSAM-style chunk | 文本 chunk/payload 组合 | 借鉴 chunk 化文本 payload | 只在当前 delta 内生成合法 payload；Student 和预算不变 | 0 | 否 | edge-free | 是，机制适配 |
| CaT + graph-conditioned text | 由 `X` 与文本相似度决定 payload | 保留图条件文本选择 | 不改变 CaT `X`，只改变文本材料化规则 | 0 | 否 | edge-free | 是，机制适配 |
| CaT + graph-text aligned | 拟合图文对齐关系后分配文本 | 保留 current-delta 图文映射 | 映射只能由当前 delta 拟合；不能用 held-out/test 文本 | 0 | 否 | edge-free | 是，机制适配 |
| CaT + text-center | 类中心文本原型 | 作为 class-shared payload 控制 | 每类所有 synthetic row 共享一条 payload；单独标记，不与多 payload 方法混排名 | 0 | 否 | edge-free | 仅分析控制 |
| CaT + CCTP top-3/top-5 | 多候选文本 payload | 保留每行 top-k 真实文本候选 | 修复旧版将 top-k 错映射为 class-shared 的实现错误；实际按 synthetic row 分配 | 0 | 否 | edge-free | 是 |
| GECC | hard clustering / Lloyd-style prototype assignment | 保留 hard-hop/Lloyd 聚类机制 | 当前 delta-only；输出 synthetic features；附加合法 TAG 文本；不运行静态全图版本 | 0 | 否 | edge-free | 是，机制适配 |
| K-center / FPS | 特征空间覆盖/最远点采样 | 保留覆盖式代表点选择 | 在 current delta 上选真实节点；保留其真实文本；memory rows 与 Student 固定 | 0 | 否 | real-node edges only | 是 |
| Medoid coreset | 选择真实 medoid 作为代表 | 保留 medoid 目标 | 在 current delta 上选代表节点；不回看历史 raw；文本随真实节点保留 | 0 | 否 | real-node edges only | 是 |
| Random real coreset | 随机真实节点采样 | 保留随机采样 | current delta-only；同预算；文本随真实节点保留 | 0 | 否 | real-node edges only | 是 |
| XSP | CaT `X` + X-conditioned slot partition | 保留 slot partition 和 node-specific payload | 当前 delta-only；每行由 `X` 条件分配 slot；旧 memory append-only | 0 | 否 | edge-free | 是，内部候选/对照 |
| Adaptive coherent | CaT `X` + adaptive slot 数 + coherent routing | 保留 X-conditioned routing、hard materialization 和 payload diversity | slot 数由当前 delta 的离散度决定；不读取历史 raw；不修改旧 memory | 0 | 否 | edge-free | 候选主方法 |
| Adaptive coherent + anchor | Adaptive coherent + anchor penalty | 用于判断 anchor 是否改善跨数据集稳定性 | 只增加固定 anchor 控制，不引入新 Student | 0 | 否 | edge-free | 内部控制 |
| Direct Student continual | 直接在到达数据上持续训练 Student | 作为下游上限/控制 | 不生成可复用凝聚 memory；不能与 memory condensation 方法直接按构造成本排名 | 0 | N/A | raw graph | 控制，不进凝聚主表 |
| TAGSAM | 静态 TAG 凝聚流程 | 只借鉴文本 chunk/payload 思路 | 改成 current-delta-only、frozen-prefix 和同一 Prompt Student；若原流程依赖完整静态图，则不宣称官方复现 | 依原实现 | 依原实现 | 依原实现 | 适配成功后再进 |
| STAD | teacher/distillation + generated text | 只保留 teacher/distillation 机制 | 需改成 current-delta-only 并统一 Student；已有小图 reference/adapted 结果，大图尚未完成 | 待审计 | 待审计 | 待审计 | 暂不进 |
| PUMA | pseudo-label、confidence filtering、propagation | 作为 TAG 文本/伪标签适配候选 | 需证明不使用历史 raw，并把 Student/预算换成统一协议 | 需审计 | 需审计 | edge-free adapter | 小图候选 |
| GraphCLIP | static graph-text pretraining/condensation | 仅作为 static/reference-only 参考 | 不把完整 static/full-graph 数字放入 continual strict table | 通常会用 | 通常会改 | static graph | 不进 strict |
| GCond/SFGC/GCondenser | static graph condensation | 作为静态凝聚参考 | 缺少 current-delta/frozen-prefix 原生协议，不直接横向排序 | 通常会用 | 通常会改 | static graph | 不进 strict |

## 3. 数值表的分层规则

### Strict 主表

只放已经通过以下检查的方法：访问边界、预算、Student、阶段顺序、评估规则一致。CaT、GECC、coreset 和 Adaptive coherent 都属于“协议一致的机制适配”，不是原论文端到端复现。

### Internal control

包括 class-shared payload、anchor、XSP-ST 等。它们用于解释机制，不能因为某个数据集分数异常高就替代外部 baseline。

### Reference-only / adapted-only

包括官方静态 TAGSAM、GraphCLIP、GCond、SFGC、GCondenser，以及尚未完成 strict 访问审计的 STAD/PUMA 大图版本。它们可以放附录覆盖表，但不能与 strict 主表共用一个排名。

## 4. 当前实验解释重点

1. `CaT + CCTP top-5` 是最直接的 CaT-TAG 适配基线：它回答“仅用 CaT 的特征凝聚，加上合法的多文本 payload，能达到什么水平”。
2. `Adaptive coherent` 只增加 X-conditioned payload routing 和 adaptive slot allocation，因而可以直接检验文本多样性和图条件分配的增益。
3. `class-assigned prompt-margin` 不应作为普通强基线。它每类共享一个 payload，并显式利用类别 prompt，容易产生类别先验收益，且跨数据集不稳定。
4. BookHis 必须同时看 Macro-F1、New F1 和 Accuracy。高 Accuracy 不能掩盖新类预测塌缩。
5. 论文中应同时报告构造时间。Adaptive coherent 的计算成本明显高于 k-center/medoid，这是方法边界的一部分。

## 5. 可复现入口

- 统一 runner：`D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py`
- 公开基线冻结结果：`D:/Aprojects/continue_tag_condensation/results/public_baselines_frozen_20261004b_gpu0`、`D:/Aprojects/continue_tag_condensation/results/public_baselines_frozen_20261004b_gpu1`
- 多数据集汇总：`D:/Aprojects/continue_tag_condensation/results/multidataset_f1_accuracy_comparison_20261006`
- 本轮 CaT 文本模式补充结果：`D:/Aprojects/continue_tag_condensation/results/text_modes_strict_20261006`
