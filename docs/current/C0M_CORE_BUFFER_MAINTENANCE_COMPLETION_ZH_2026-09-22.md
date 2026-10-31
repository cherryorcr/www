# C0-M Core-Buffer Memory Maintenance 结项报告

日期：2026-09-22  
状态：实验完成，结构审计通过，性能晋升失败；C0 继续作为主方法。

## 1. 研究问题

C0 在持续任务中只追加新 synthetic block，历史 rows 和 old-old edges 不改写。C0-M 检验一个有限放宽：保护一部分稳定 core，只允许对少量历史 buffer rows 做有界更新，观察是否能够缓解历史陈旧，同时不破坏旧类保持和新类塑性。

本轮没有改变 C0 的 incoming block、X 构造、文本构造、拓扑、student、batch、steps、seed 或评估 split。因此差异可以归因于历史 memory maintenance。

## 2. C0-M 算法

1. 第一个 memory 生成后，在 joint normalized X/text 空间用 farthest-point 选择约 80% stable core，剩余约 20% 作为 replaceable buffer。
2. core 行永久保护；buffer 行记录初始 `x/text` anchor。
3. 每个后续阶段计算 buffer 行的 redundancy，在当前 delta 中计算相对 core 的 novelty。
4. 选 redundancy 最高的旧 buffer 行和 novelty 最高的新 delta 行配对。
5. 对 `x/text/mass` 做 `blend_alpha=0.50` 的凸组合，并限制相对初始 anchor 的累计位移不超过 `anchor_radius=0.25`。
6. 随后按 C0 完全相同的方式追加新 block；总 rows、边数和 old-old edge block 保持不变。

`naive_refresh` 是已有的低 mass、10% old rows、alpha=0.25 refresh，作为负对照，不是 C0-M。

## 3. 冻结实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 顺序 | canonical、order19101、order19102 |
| 阶段 | 3 stages |
| memory | 5% fixed-rate growing memory |
| C0 block | `joint_g5_c320` + concat + dispersion |
| topology | old-old frozen，cross degree=1 |
| student | 两层 GCN/GraphCLIP student，双向图文 InfoNCE，从头训练 |
| seeds | 100--105，共 6 个 student seeds |
| 指标 | report-dev seen-class Macro-F1；累计、old、new 分解 |
| 总规模 | 6 cells × 3 arms × 3 stages × 6 seeds = 324 fits |

## 4. 最终阶段结果

数值是六个 student seeds 的均值，单位为 Macro-F1 百分点。

| 数据集 | 顺序 | C0 | naive refresh | C0-M | C0-M - C0 | old F1 变化 | new F1 变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | canonical | 49.85 | 49.79 | 47.73 | -2.13 | -1.06 | -3.55 |
| Cora | order19101 | 49.69 | 49.22 | 55.78 | +6.08 | +6.12 | +6.04 |
| Cora | order19102 | 54.48 | 51.25 | 51.21 | -3.27 | -2.95 | -3.70 |
| Citeseer | canonical | 59.17 | 56.53 | 57.17 | -2.00 | -1.17 | -3.66 |
| Citeseer | order19101 | 46.63 | 45.62 | 47.08 | +0.45 | -0.77 | +2.89 |
| Citeseer | order19102 | 56.70 | 53.99 | 55.55 | -1.14 | +0.09 | -3.60 |

跨 6 个 cell 的汇总：

| 指标 | C0-M 相对 C0 |
|---|---:|
| final cumulative F1 平均变化 | -0.34 pp |
| final cumulative F1 最差变化 | -3.27 pp |
| old F1 平均变化 | +0.04 pp |
| new F1 平均变化 | -0.93 pp |
| 不低于 C0-1 pp 的 cell | 2/6 |
| construction cost / C0 | 1.03x |
| final logical memory / C0 | 平均 1.63x，范围 1.60--1.70x |

## 5. 正确性审计

独立审计通过：

- 6 个 cell、324 个 fit 全部存在，无重复或缺失；
- 54 个 memory artifact 的 tensor hash 和 construction metadata 一致；
- 324 个 checkpoint 的 state hash、support hash 和逐 fit JSON 一致；
- C0-M 新增 slice 与 C0 新增 slice 完全一致；
- core membership、core 内容和 core text 未改变；
- old-old edges 未改变；
- C0 与 C0-M edge entry 数量一致；
- buffer anchor 最大相对位移不超过 0.25；
- rows 与 5% growing budget 一致；
- student steps 与统一协议一致。

因此，本轮负结果不是由预算变化、拓扑边数变化、checkpoint 错配或训练协议不一致造成的。

## 6. 晋升门禁

| 门禁 | 要求 | 结果 |
|---|---:|---:|
| cell 稳健性 | 至少 4/6 不低于 C0-1 pp | 2/6，失败 |
| 平均 final gain | 至少 +1 pp | -0.34 pp，失败 |
| 最差 final gain | 不低于 -2 pp | -3.27 pp，失败 |
| 最差 old gain | 不低于 -2 pp | 通过，但不足以挽救整体 |
| 平均 new gain | > 0 | -0.93 pp，失败 |
| construction cost | ≤ 2x | 1.03x，通过 |
| 结构与 checkpoint 审计 | 全部通过 | 通过 |

结论：**C0-M 不晋升为主方法。**

## 7. 结果解释

这次实验说明“历史 memory 可以被维护”这个想法在机制上并非完全无效：`Cora/order19101` 有 +6.08 pp，`Citeseer/order19101` 也有 +0.45 pp。但收益高度依赖类别顺序，另外四个 cell 下降，且新类 F1 的总体变化为负。

当前 core-buffer 规则使用的是静态 joint 几何 proxy：redundancy 判断哪些旧行可以替换，novelty 判断当前新行是否远离初始 core。它没有检查：

- 替换后旧 student 表示是否漂移；
- 当前 delta 是否真的覆盖了旧类缺口；
- 新 row 被写入历史 buffer 后是否会损害新类塑性；
- buffer anchor 的“几何稳定”是否等价于任务稳定。

因此，C0-M 的结果支持一个更谨慎的判断：**简单的几何 core-buffer maintenance 不能稳定解决 frozen-old 的问题；历史更新需要任务保持约束或可靠的 student-compatible evidence，而不是仅靠 redundancy/novelty。**

此外，C0-M 为每个 buffer row 保存 anchor，最终逻辑 memory 平均约为 C0 的 1.63 倍。即便构造时间只有 1.03 倍，memory overhead 也使它不适合在没有稳定收益时进入主线。

## 8. 后续决策

1. C0 继续作为当前主方法和所有后续比较的冻结底座。
2. C0-M 作为“低成本几何历史维护未通过”的负结果写入实验边界，不再追加 20% refresh、merge/split、topology refresh 或 response selector 组合。
3. 不把 C0-M 上大图；大图扩展应优先服务已通过小图门禁的 C0 或新的、理论约束更明确的 memory manager。
4. 若后续需要重新研究历史维护，应改为“有证据才更新”：同时使用旧 student response/表示保持、当前 delta 覆盖缺口和拓扑漂移约束，并先在独立顺序上做最小矩阵；不能把本轮 C0-M 直接包装成正向方法。

## 9. 产物

- 冻结协议：[C0M_CORE_BUFFER_MAINTENANCE_FROZEN_PROTOCOL_ZH_2026-09-22.md](C0M_CORE_BUFFER_MAINTENANCE_FROZEN_PROTOCOL_ZH_2026-09-22.md)
- 结果目录：`results/c0m_maintenance_20260922/`
- 方法实现：[c0m_memory_maintenance.py](../../code/v2_llm_fusion/c0m_memory_maintenance.py)
- 实验 runner：[run_c0m_maintenance.py](../../code/v2_llm_fusion/run_c0m_maintenance.py)
- 独立审计：[audit_c0m_maintenance.py](../../code/v2_llm_fusion/audit_c0m_maintenance.py)
- 汇总：[summarize_c0m_maintenance.py](../../code/v2_llm_fusion/summarize_c0m_maintenance.py)

