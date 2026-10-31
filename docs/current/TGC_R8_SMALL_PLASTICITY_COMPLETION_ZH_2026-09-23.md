# TGC-R8 小图塑性验收完成报告

## 1. 结论摘要

R8 完成了 Cora/Citeseer 的确定性稀疏 TGC-R 与 C0 新块特征锚定实验。实验覆盖 3 个独立类别顺序、3 个阶段、两组 teacher seeds 和 `lambda={0,.25,.5,1}`，所有候选完整评估，没有按顺序、seed 或 teacher group 选择最优结果。

结论不是“R8 已经成为稳定主方法”，而是：

1. C0 锚定可以在部分阶段改善新类塑性；例如 Cora/order19600/stage1 的 `det_g0_l000` 为 50.91，相对 C0 的 41.51 提升 9.40 pp，并超过同 cell 的 k-center 48.71。
2. 该收益不稳定。Cora 三顺序最终阶段的最佳确定性 arm 分别为 49.08、48.93、47.86，而 C0 分别为 45.06、45.60、50.77；最终平均没有形成稳定正收益。
3. Citeseer 的 `det_g0_l025` 最终平均为 56.69，相对 C0 的 54.56 提升 2.13 pp，但 order19600 仍低于 C0，最差最终 cell 为 -3.06 pp。
4. `lambda` 不呈单调规律；不同 teacher group 会改变排序，说明当前瓶颈仍是 teacher-response 方差与持续 parent 依赖，而不是简单的 C0/TGC 特征线性插值比例。
5. 确定性 segmented-SpMM 通过了 dense 前向/反向数值对齐和重复哈希审计，作用是可复现、可扩展的实现基础设施，不能单独作为算法增益宣称。

因此，R8 应作为“C0 锚定新类塑性机制的正向局部证据 + 稳定性边界”保留；当前论文主方法仍建议以 C0 为 fallback，不直接把某个 lambda 晋升为最终主方法，也不马上上大图。

## 2. 实验协议

- 数据集：Cora、Citeseer。
- 顺序：`order19600`、`order19609`、`order19610`，独立于 R7 的开发顺序。
- 三阶段，累计 5% synthetic memory。
- arms：C0、k-center、dense TGC-R，以及 teacher groups `(8800,8801)`、`(8810,8811)` 下的 `lambda=0/.25/.5/1`。
- 锚定只作用于当前新增 synthetic block 的 `X`：

  `X_new=(1-lambda)X_C0_new+lambda X_TGC_new`。

  旧行、旧边、新块文本、mass 和拓扑均保持对应 TGC 输出。
- student：统一 canonical 两层 GCN、冻结 MiniLM、双向图文 InfoNCE、batch 16、Adam `1e-3`、从头训练、报告 seeds `220..225`。
- 约束：历史 raw reads=0，report/test labels 不参与构造或选择。

## 3. 最终阶段结果

### Cora

| 方法 | order19600 | order19609 | order19610 | 最终均值 | 相对 C0 |
|---|---:|---:|---:|---:|---:|
| C0 | 45.06 | 45.60 | 50.77 | 47.14 | 0.00 |
| k-center | 51.41 | 47.15 | 46.63 | 48.40 | +1.25 |
| dense TGC-R | 46.94 | 42.47 | 48.01 | 45.81 | -1.33 |
| det g0, lambda=0 | 46.46 | 44.65 | 46.42 | 45.85 | -1.30 |
| det g0, lambda=.25 | 46.38 | 43.62 | 45.99 | 45.33 | -1.81 |
| det g0, lambda=.5 | 46.08 | 41.91 | 47.86 | 45.28 | -1.86 |
| det g0, lambda=1 | 46.31 | 40.42 | 46.65 | 44.46 | -2.68 |
| det g1, lambda=0 | 48.77 | 48.93 | 43.62 | 47.11 | -0.04 |
| det g1, lambda=.25 | 49.08 | 47.48 | 44.71 | 47.09 | -0.05 |
| det g1, lambda=.5 | 49.03 | 48.41 | 44.68 | 47.37 | +0.23 |
| det g1, lambda=1 | 48.69 | 47.43 | 45.07 | 47.06 | -0.08 |

最终旧类/新类 F1 均值：C0 `44.86/50.19`；k-center `44.00/54.25`；最好的确定性 arm `det_g1_l050` 为 `41.36/55.39`。这说明部分 R8 收益主要来自新类 plasticity，但旧类保持存在代价。

### Citeseer

| 方法 | order19600 | order19609 | order19610 | 最终均值 | 相对 C0 |
|---|---:|---:|---:|---:|---:|
| C0 | 57.26 | 53.09 | 53.32 | 54.56 | 0.00 |
| k-center | 57.95 | 54.18 | 48.53 | 53.56 | -1.00 |
| dense TGC-R | 54.21 | 54.72 | 58.78 | 55.90 | +1.35 |
| det g0, lambda=0 | 54.01 | 53.56 | 57.38 | 54.98 | +0.43 |
| det g0, lambda=.25 | 54.20 | 55.65 | 60.22 | 56.69 | +2.13 |
| det g0, lambda=.5 | 54.03 | 54.69 | 58.68 | 55.80 | +1.24 |
| det g0, lambda=1 | 52.95 | 55.51 | 57.40 | 55.29 | +0.73 |
| det g1, lambda=0 | 53.52 | 52.36 | 58.63 | 54.84 | +0.28 |
| det g1, lambda=.25 | 53.87 | 51.72 | 59.80 | 55.13 | +0.57 |
| det g1, lambda=.5 | 53.88 | 52.39 | 59.99 | 55.42 | +0.86 |
| det g1, lambda=1 | 53.38 | 52.41 | 59.61 | 55.14 | +0.58 |

最终旧类/新类 F1 均值：C0 `58.47/46.72`；k-center `58.87/42.93`；`det_g0_l025` 为 `61.88/46.30`。Citeseer 上 `det_g0_l025` 同时改善旧类保持和最终平均，但 order19600 仍出现负差，因此还不足以称为通用最优。

## 4. 成本与数值审计

- 两个数据集各 594 个 student fit slots，独立 checkpoint reload 的最大指标误差为 `2.85e-14`。
- 构造阶段的平均耗时约为：C0 `0.30--0.34 s`，k-center `0.005--0.006 s`，dense TGC-R `0.84--0.96 s`，deterministic TGC-R 约 `1.64--1.86 s`。
- 当前 deterministic 实现比 C0/k-center 慢，原因是每个 teacher group 都进行了 teacher 训练、assignment 优化和 materialization；它的价值是可审计与可扩展，不是当前成本优势。
- Cora/order19600/stage1 的 segmented-SpMM 审计：normalized adjacency 最大前向误差 `2.38e-7`、反向误差 `1.19e-6`；raw adjacency 最大前向误差 `1.91e-6`、反向误差 `5.72e-6`；重复执行哈希一致。
- 所有 audit 均报告 `historical_raw_reads=0`、`report_labels_used=false`。

## 5. 对后续的决定

### 保留

- C0 作为主控制和 fallback。
- C0 锚定作为候选 plasticity operator，保留 `lambda=0` 和有限的 `lambda=.25` 作为后续机制分析变量。
- deterministic segmented-SpMM 作为大图实现基础，但不把它单独写成创新点。

### 暂不晋升

- 不把 `det_g0_l025` 或 `det_g1_l050` 直接命名为最终主方法：它们的收益依赖数据集、顺序和 teacher group。
- 不使用本轮结果宣称稳定超过 k-center 或 SOTA。
- 不因单个 Cora/order19600/stage1 的大幅提升直接上大图。

### 下一步优先级

1. 先分析 teacher group 差异：比较同一 parent 下的 `H/P/X`、新类/旧类响应和 candidate regret，确认性能方差来自 teacher 训练、assignment 还是 materialization。
2. 将锚定从固定 lambda 改为有界、按新类响应触发的 gate；必须加入“增加 rows/teacher 成本”的匹配控制，避免把额外成本误当作方法收益。
3. 在 Cora/Citeseer 新的独立顺序上做一次小规模确认；只有平均收益、最差 cell 和旧类保持同时达标，才进行大图 base+4。
4. 若仍不能稳定改善，应将 R8 写成机制负结果：C0 anchor 能改善局部 plasticity，但固定线性混合无法消除 teacher-response variance；论文主线回到“持续 TAG 凝聚的可审计协议、保持-塑性权衡和失效边界”。

## 6. 产物

- 原始拉回目录：`results/tgc_r8_plasticity_20260923d_remote/`
- 逐 cell 指标：`results/tgc_r8_plasticity_20260923d_remote/r8_cell_metrics.csv`
- 汇总：`results/tgc_r8_plasticity_20260923d_remote/r8_summary_by_dataset_arm.csv`
- 成本：`results/tgc_r8_plasticity_20260923d_remote/r8_costs.csv`
- 独立审计：`results/tgc_r8_plasticity_20260923d_remote/r8_audits.csv`
- 机器生成报告：`results/tgc_r8_plasticity_20260923d_remote/TGC_R8_SMALL_PLASTICITY_REPORT_ZH_2026-09-23.md`
