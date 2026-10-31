# 外部基线方法与适配说明

## 1. 本轮实际纳入严格比较的方法

本轮在 WikiCS 和 BookHis 上补齐了五类外部参照，使用同一个 `base+4` 类别增量协议、5% cumulative memory、6 个相同 Student seeds、current-delta-only construction 和 frozen-prefix 规则：

| 方法 | 原始思想 | 本项目中的适配 |
|---|---|---|
| Random real coreset | 从当前合法数据池随机保留真实节点 | 只从当前 delta 选真实节点，保留对应 TAG 文本，旧 memory 只作 prefix |
| K-center/FPS | 用 farthest-point sampling 覆盖特征空间 | 在当前 delta 的图/文本联合空间选真实节点，不读取历史 raw |
| Medoid | 选择联合空间中的代表性真实节点 | 当前 delta 内做 medoid 选择，文本随真实节点保留 |
| CaT-MMD-TAG | CaT 的 class-wise MMD 合成特征凝聚 | 用 CaT 风格的 class-wise MMD 生成新增 `X`，用当前 delta 合法文本候选构造 CCTP top-5 payload |
| GECC-hard-TAG | GECC 的 hard hop-weight / structure-free condensation 思路 | 在当前 delta 上运行 hard-hop 分支和 Lloyd 聚类，再附加合法 TAG 文本；不使用官方静态全图流程 |

因此表中的 CaT 和 GECC 都应称为 **mechanism-adapted strict continual baselines**，不是官方代码的原样复现。

## 2. 论文和代码来源

- **CaT**：*CaT: Balanced Continual Graph Learning with Graph Condensation*，arXiv: [2309.09455](https://arxiv.org/abs/2309.09455)，官方仓库 [CaT-CGL](https://github.com/superallen13/CaT-CGL)。
- **PUMA**：*PUMA: Efficient Continual Graph Learning for Node Classification with Graph Condensation*，arXiv: [2312.14439](https://arxiv.org/abs/2312.14439)。当前已有 Cora/Citeseer 的严格适配 runner，但没有把只覆盖小图的结果伪装成 WikiCS/BookHis 结果。
- **GECC**：*GECC: Scalable and Structure-free Graph Condensation with Evolving Capabilities*，官方仓库 [GECC](https://github.com/Emory-Melody/GECC)。其官方 evolving split 与本项目的 TAG 文本、5% fixed-rate、frozen-prefix 协议不同，因此采用 hard-hop 机制适配。
- **TAGSAM**：官方仓库 [TAGSAM](https://github.com/SundayVHan/TAGSAM)。它是静态 TAG condensation 参考，现有 runner 依赖 teacher/buffer 与静态图流程，不能直接作为同协议 strict continual 数值基线；只在 adapted/reference 表中保留。

## 3. 为什么没有把所有公开方法塞进同一主表

GCond、SFGC、GCondenser、官方 GraphCLIP、STAD、静态完整历史 TAGSAM 等可用实现至少有一项不满足当前严格协议：

- 需要访问完整历史 raw graph；
- 没有 current-delta append 接口；
- 修改或重建旧 memory；
- 使用监督分类器或不同 Student；
- 使用静态全图划分而不是类别到达流。

把这些数值直接和本轮主表比较会把“算法差异”和“协议差异”混在一起。因此它们应在论文中作为 reference/adapted table、方法背景或上界，而不是严格 continual SOTA 表。

## 4. 强基线与组件消融的分层

CaT 的不同 TAG 文本处理方式不应被误称为独立强基线。它们共享同一个 CaT class-wise MMD `X` 构造器，只替换文本 payload 的材料化规则，因此在论文中应放在 **CaT text component ablation** 表中：nearest payload、TAGSAM-style chunk proxy、graph-conditioned text、graph-text aligned、CCTP top-3/top-5，以及 class text-center 分析控制。

真正的强基线/外部参照应单列：

1. **Direct append / continual Student control**：只把当前阶段 memory append 到 prefix，使用同一个 Prompt zero-shot Student，回答“不做凝聚改进时”的持续学习水平。
2. **C0 / X+P prior method control**：已有主线的原始合成 memory，回答 Adaptive coherent 相对原方法的增益。
3. **Random real coreset、K-center/FPS、Medoid**：真实节点代表性方法，作为不依赖 synthetic feature optimization 的强 coreset 参照。
4. **CaT-MMD-TAG (CCTP top-5)**：CaT 的合成特征与合法 TAG payload 结合，是最直接的持续 TAG 凝聚外部机制适配。
5. **GECC-hard-TAG**：保留 GECC 的 hard clustering/hop 机制，是另一种持续/演化凝聚机制适配。
6. **PUMA-inspired**：若使用，必须单独报告 pseudo-label、one-time propagation、edge-free memory 等独有机制，不能用 CaT 结果代替。

推荐的主表顺序是：`Adaptive coherent`、`C0/X+P`、`CaT-MMD-TAG`、`GECC-hard-TAG`、`K-center`、`Medoid`、`Random real`、`Direct append`。CaT 文本适配结果放在紧邻的组件消融表，不参与“外部强基线排名”。

## 5. 本轮基线比较结论

完整数值见：

- [严格外部基线汇总表](EXTERNAL_BASELINE_COMPARISON_WIKICS_BOOKHIS_20261006.md)
- [逐数据集 CSV](../../results/external_baseline_comparison_wikics_bookhis_20261006/dataset_cells.csv)

当前结果显示：Adaptive coherent 在 WikiCS 上相对五个外部基线均有明显 Final Macro-F1 优势；在 BookHis 上 Final Macro-F1 基本持平，Accuracy 很高但 New F1 明显偏低，说明它在长尾新类上的优势尚未确认。论文中应把 WikiCS 的提升和 BookHis 的 trade-off 同时报告，不能只写 pooled 平均值。
