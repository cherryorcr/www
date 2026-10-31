# TGC P0 独立类别顺序与 construction seed 验证报告

日期：2026-09-26  
状态：P0 已完成。construction seed 确定性通过；独立类别顺序的跨数据集泛化未通过，semantic-recursive 的论文主张需要收窄。

## 1. P0 要回答的问题

本轮不修改 semantic-recursive，也不加入 selector、动态预算、旧 memory refresh、文本 attention、GraphGPS 或新的 Student。只回答两个基础问题：

1. canonical 大图结果是否依赖某个 construction seed，或者同一方法是否会因为构造随机性产生不同 memory；
2. 在未参与开发的独立类别到达顺序上，semantic-recursive 是否仍然相对冻结 C0 有方向一致的收益。

本轮的判断对象是当前候选方法的**泛化边界**，不是重新调参的机会。

## 2. 固定方法与协议

### 2.1 两个比较臂

- **C0**：冻结的 `dispersion + append_fixed` continual constructor；
- **semantic-recursive**：当前论文候选，流程为 `two-prototype structural proposal + class-local semantic conserved payload assignment + recursive append`。

两臂使用同一阶段预算、同一 C0 topology、同一 Student 和同一 Student seeds。semantic-recursive 不访问历史 raw graph，只把上一阶段 semantic memory 作为 parent，并追加当前 delta 形成的新 block。

### 2.2 Student 协议

| 项目 | 设置 |
|---|---|
| 数据集 | arXiv、BookChild、SportsFit |
| 类别协议 | `base+4`，共 5 stages |
| 独立顺序 seed | 49013 |
| memory rate | 5% fixed-rate，预算随累计数据增长 |
| Student | 原始两层 GCN + trainable text projection |
| 目标 | 双向 graph-text InfoNCE |
| batch | 16 |
| steps | 2048 |
| Student seeds | 100--105 |
| 评价 | report-dev cumulative / old-class / new-class Macro-F1 |
| report/test 标签 | 不参与构造或选择 |

固定 5% rate 下，不同类别顺序会导致每阶段到达样本数不同，因而 budget 也会自然不同。预算差异是协议的结果，不能直接解释成方法优势或劣势。

### 2.3 独立顺序的样本量和预算

| 数据集 | stage train counts | stage budgets |
|---|---|---|
| arXiv | 49977 / 26868 / 4375 / 7576 / 29743 | 2498 / 3842 / 4061 / 4439 / 5926 |
| BookChild | 21936 / 25275 / 1980 / 2513 / 2107 | 1096 / 2360 / 2459 / 2585 / 2690 |
| SportsFit | 21064 / 56626 / 15588 / 12214 / 15645 | 1053 / 3884 / 4663 / 5274 / 6056 |

## 3. 独立顺序的逐阶段结果

下表的 delta 均为 `semantic-recursive - C0`，单位为 Macro-F1 百分点。stage 0 两个方法完全相同，因此只把 stage 1--4 作为增量判断。

### 3.1 arXiv

| stage | cumulative delta | old-class delta | new-class delta |
|---:|---:|---:|---:|
| 1 | -0.176 | +0.869 | -3.220 |
| 2 | -0.914 | +2.483 | -4.415 |
| 3 | -0.751 | -0.426 | -2.125 |
| 4 | +0.349 | -0.152 | +0.220 |
| 增量平均 | **-0.373** | **+0.693** | **-2.385** |

只有 1/4 个增量阶段为正，最差阶段为 -0.914 pp。这里 semantic-recursive 保留旧类的能力并没有转化成新类塑性，导致累计结果下降。

### 3.2 BookChild

| stage | cumulative delta | old-class delta | new-class delta |
|---:|---:|---:|---:|
| 1 | +1.486 | +1.679 | +2.554 |
| 2 | +1.211 | +0.478 | +0.836 |
| 3 | -0.328 | -0.479 | -2.286 |
| 4 | -0.760 | -6.237 | +4.205 |
| 增量平均 | **+0.402** | **-1.139** | **+1.327** |

前两个增量阶段为正，但 stage 4 出现明显 old-class retention 损失。该结果支持“新类塑性与历史保持存在 trade-off”，不支持无条件提升。

### 3.3 SportsFit

| stage | cumulative delta | old-class delta | new-class delta |
|---:|---:|---:|---:|
| 1 | +1.338 | +0.972 | +4.309 |
| 2 | -2.434 | -2.343 | -3.970 |
| 3 | -0.723 | -3.766 | +7.212 |
| 4 | +1.941 | +2.062 | -1.790 |
| 增量平均 | **+0.030** | **-0.769** | **+1.440** |

SportsFit 的 new-class 平均是正的，但 stage 2 出现全轮最差 cell（-2.434 pp），说明局部新类收益不能被解释成稳定的全链收益。

### 3.4 汇总

| 数据集 | 增量 cumulative 平均 | old-class 平均 | new-class 平均 | 正收益 stages | 最差 stage |
|---|---:|---:|---:|---:|---:|
| arXiv | -0.373 | +0.693 | -2.385 | 1/4 | -0.914 |
| BookChild | +0.402 | -1.139 | +1.327 | 2/4 | -0.760 |
| SportsFit | +0.030 | -0.769 | +1.440 | 2/4 | -2.434 |
| 合计 | **+0.020** | -- | -- | **5/12** | **-2.434** |

合计均值只对 12 个增量 stage cells 计算。它接近 0，说明独立顺序下不能继续使用 canonical 顺序上的 `+0.858 pp` 作为跨顺序平均收益。

## 4. Construction seed 审计

每个数据集分别使用 construction seeds `9001, 9002, 9003`，不训练 Student，也不使用 report F1 选择候选。对每个 stage memory 计算完整 payload hash。

| 数据集 | bitwise seed invariant | stage memory hashes | mapping moves | Student trained | F1 used | historical raw reads |
|---|---|---|---|---:|---:|---:|
| arXiv | True | 3 seeds × 5 stages 完全一致 | 0 / 15 / 10 / 12 / 15 | 否 | 否 | 0 |
| BookChild | True | 3 seeds × 5 stages 完全一致 | 0 / 9 / 11 / 10 / 2 | 否 | 否 | 0 |
| SportsFit | True | 3 seeds × 5 stages 完全一致 | 0 / 5 / 3 / 5 / 8 | 否 | 否 | 0 |

这里的 `mapping moves` 是 semantic permutation 在各阶段实际移动的 payload 行数，不是随机性指标。它在同一数据集内对三个 construction seeds 完全相同。

因此可以确认：

- semantic-recursive 的 memory 构造是 bitwise deterministic；
- canonical 顺序的收益不是 construction seed 偶然造成的；
- 本轮没有 Student 选择泄漏，也没有历史 raw 回读；
- P0 的 seed audit 通过。

## 5. 结果解释

### 5.1 通过的部分：构造确定性

不同 construction seed 得到完全相同的 payload hash，说明当前方法的正负结果来自确定性的 source grouping、two-prototype split/merge 和 semantic assignment，而不是某次随机匹配或随机候选恰好更好。这为论文中的可复现性和审计性主张提供了证据。

### 5.2 未通过的部分：跨类别顺序泛化

独立顺序下 arXiv 的增量平均为 -0.373 pp，且 new-class 平均为 -2.385 pp；三数据集合计只有 5/12 个增量 cells 为正。因此当前方法不能表述为“对不同类别顺序稳定提升”。

canonical 顺序上的正向结果更准确的解释是：

> semantic-recursive 在部分数据集和类别到达顺序下，能够通过守恒的 payload 重排改善新类可重训性；但这种改善依赖当前 delta 的类组成、预算增量和 C0 topology，尚未成为顺序不变的普适规律。

### 5.3 为什么会有顺序依赖

当前方法的 semantic assignment 只在当前新增 block 内、同类别内部进行。它依赖三个顺序相关因素：

1. 当前阶段包含哪些类别，以及每个类别的 raw 样本量；
2. 固定 5% rate 产生的新增 rows 和类内 slot 数；
3. 冻结 C0 topology 对新 block 的邻接位置和传播角色。

当新类规模较小或类内多模态没有得到足够 rows 时，two-prototype proposal 可能改善局部 payload 几何，却无法改善 Student 在真实图上的新类传播。因而 old/new trade-off 会随顺序改变。

## 6. P0 验收结论

| 检查项 | 结论 | 说明 |
|---|---|---|
| construction seed invariance | **通过** | 三个大图的 3 个 seeds 均 bitwise 一致 |
| budget / topology / Student paired control | **通过** | semantic 与 C0 使用相同 budget、edge、batch、steps、seeds |
| historical raw access audit | **通过** | 构造阶段为 0 |
| independent-order positive generalization | **未通过** | arXiv 负，整体仅 5/12 正，均值约 +0.020 pp |
| universal stable improvement | **不支持** | 最差 cell -2.434 pp |

P0 不应被写成“新顺序也稳定超过 C0”。它的正确作用是把当前方法的证据边界从“可复现机制”进一步限定为“顺序条件下的方向性机制”。

## 7. 对主方法和论文的影响

当前建议保留以下版本关系：

- **semantic-recursive**：冻结为论文主方法候选和机制研究对象；
- **C0**：冻结为共同控制、稳定回退和失败对照；
- 不把独立顺序的负结果丢弃，也不从 independent order 反向调参；
- 不因为 P0 失败重新加入 response selector、dynamic budget、old-memory refresh、GraphGPS 或同时改变 X/文本/拓扑。

论文中可以主张：

1. 构造过程是确定性的、可审计的；
2. 在 canonical 大图 base+4 上有平均 +0.858 pp、9/12 正 cells 的方向性收益；
3. 独立顺序实验揭示其顺序依赖和 new-class / old-class trade-off；
4. continual condensation 的评价不能只看单一顺序的最终 F1，需要同时报告 seed、顺序、old/new 和最差阶段。

论文中不能主张：

- 跨顺序稳定优于 C0；
- 已经优于 k-center、静态 TAGSAM 或全历史重凝聚；
- universal SOTA；
- semantic matching 在理论上最优。

## 8. 后续执行建议

1. 先把本轮 independent-order 结果加入主表和失败分析，不修改 semantic-recursive。
2. 完成统一的强 coreset、full-history synthetic 和 raw reference 主表；这些结果用于定位性能差距，不与 continual 方法混为同一访问约束。
3. 若要继续算法改进，只做一个窄的、固定参数的 message-aware semantic matching，并保留失败回退到当前 semantic matching 的机制。
4. 新类塑性方向优先做 class-floor、最小多原型覆盖和同类 slot 约束的单因素诊断；不重新启用昂贵 response selector。
5. 在论文中把 P0 作为“可复现性通过、泛化边界暴露”的完整结果，而不是只报告正向 cell。

## 9. 证据文件

- 独立 C0 结果：`results/tgc_p0_independent_c0_20260926/order49013/`
- semantic 配对结果：`results/tgc_p0_semantic_recursive_20260926/order49013/`
- seed audit：`results/tgc_p0_semantic_recursive_20260926/order49013/seed_audit/`
- canonical 大图确认：[TGC_SEMANTIC_RECURSIVE_LARGE_CONFIRMATION_ZH_2026-09-26.md](TGC_SEMANTIC_RECURSIVE_LARGE_CONFIRMATION_ZH_2026-09-26.md)
- 五层基线对比：[TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md](TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md)
