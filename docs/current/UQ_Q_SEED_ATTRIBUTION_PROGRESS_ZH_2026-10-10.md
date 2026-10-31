# Unified-Q 共享 Q 归因实验进度

本轮实验用于冻结主方法后的机制归因，不用于根据成绩替换 sealed 主结果。比较三条分支：

- `uq_shared_q`：当前 edge-free In-loop CaT-Q；
- `uq_split_q`：特征 assignment 和文本 assignment 使用独立参数；
- `uq_no_residual`：当前方法关闭有界 X residual。

三条分支共用 fixed-schedule、类别顺序、memory rows、Student seeds `[150,151,152]`、2048 Student steps、30 个外层 Q steps 和 edge-free 拓扑。构造只访问当前 delta 和已有 synthetic prefix，结果评估在 report-dev，未使用 sealed test。

## 当前状态

计划为 3 个数据集 × 3 个顺序 × 3 个 construction seeds，共 27 个 cell；每个 cell 15 条阶段 readout。当前本地严格合格 26 个 cell；唯一缺失的是 `sportsfit_o9103_c271108`。正式汇总脚本只纳入注册目录、`status=complete` 且恰好包含 3 个 arm × 5 个 stage 的 cell。

远程中间文件使用 `/dev/shm/uq_q_seed_attribution_20261010`，本地 JSON 归档位于 `results/uq_q_seed_attribution_20261010`。不保存完整 memory tensor，只有阶段指标、审计、内容 hash 和日志。

## 已处理的问题

缺失 cell 的正式 2048 上限运行在某个阶段出现 Sinkhorn 行误差 `1.3417e-4`，略高于 `1e-4` 容差后退出；这不是 GPU OOM，也不是 Student 训练错误。随后用 `sinkhorn_max_steps=4096` 做了独立数值诊断，完整得到 15 条记录并保存在：

`results/uq_q_seed_attribution_20261010/diagnostic_solver4096_sportsfit_o9103_c271108/`

该诊断版本只用于确认数值可行性，不能和正式 2048 矩阵混合统计。它的最终 F1 为 Shared-Q `79.13`、Split-Q `78.03`、No-residual `78.20`；这些数值不作为正式归因主表。

## 解释边界

共享 Q 的收益只有在本矩阵完成后才能和 independent Qx/Qt 做 construction-seed 配对比较。当前 26 个正式 cell 已能给出阶段性配对结果，但仍缺 1 个正式 cell，不能称为完整 27-cell 矩阵。当前结果只能支持“共享 Q 的收益具有明显数据集依赖”，不能改写为“共享 Q 已被证明普遍有效”。`no-residual` 只用于判断 residual 是否稳定贡献，不改变 sealed 主方法。

## 26 个正式 cell 的阶段性结果

正式结果文件为 `results/uq_q_seed_attribution_20261010/summary.md`，当前是 report-dev，不是 sealed test。Final 指标按 construction cell 汇总，Student seeds `[150,151,152]` 先在 cell 内平均：

| 数据集 | 分支 | Final Macro-F1 | Accuracy | Old F1 | New F1 | cell 数 |
|---|---|---:|---:|---:|---:|---:|
| WikiCS | Shared-Q / full | 80.38 ± 0.85 | 83.09 ± 1.42 | 80.42 ± 1.15 | 80.01 ± 4.65 | 9 |
| WikiCS | Split-Q | 80.93 ± 0.85 | 83.47 ± 1.11 | 81.02 ± 1.05 | 80.10 ± 4.40 | 9 |
| WikiCS | No-residual | 80.96 ± 0.47 | 83.39 ± 0.74 | 81.14 ± 0.85 | 79.31 ± 4.79 | 9 |
| SportsFit | Shared-Q / full | 78.92 ± 0.86 | 84.53 ± 1.36 | 78.39 ± 2.28 | 81.81 ± 13.00 | 8 |
| SportsFit | Split-Q | 78.82 ± 1.18 | 84.56 ± 1.22 | 78.37 ± 1.94 | 81.32 ± 12.45 | 8 |
| SportsFit | No-residual | 78.52 ± 1.34 | 83.96 ± 1.97 | 78.11 ± 2.27 | 80.81 ± 13.79 | 8 |
| BookHis | Shared-Q / full | 41.97 ± 4.62 | 79.91 ± 1.89 | 41.29 ± 3.76 | 49.37 ± 20.52 | 9 |
| BookHis | Split-Q | 42.50 ± 4.48 | 80.19 ± 1.86 | 41.94 ± 3.78 | 48.60 ± 19.55 | 9 |
| BookHis | No-residual | 42.39 ± 3.66 | 80.44 ± 1.67 | 41.58 ± 2.91 | 51.24 ± 19.50 | 9 |

### 暂定归因结论

1. Shared-Q 相对 Split-Q 的 Final F1 差值为 WikiCS `-0.55`、SportsFit `+0.10`、BookHis `-0.53` 个百分点；不能声称共享 Q 在当前协议下稳定优于独立 Q。
2. Shared-Q 相对 No-residual 的 Final F1 差值为 WikiCS `-0.58`、SportsFit `+0.40`、BookHis `-0.42` 个百分点；bounded X residual 也没有跨数据集稳定收益。
3. SportsFit 的正向信号值得保留，但 WikiCS/BookHis 的负向结果说明目前更像数据集依赖的机制，而不是已经验证的普适组件。下一步应先分析文本软硬偏差、Q 熵和梯度贡献，再决定是否修改材料化或联合分布目标。
4. 这轮结果不能替换 sealed 主表，也不能支持“Shared-Q 已经解决文本-结构耦合”或“Unified-Q 已经优于所有外部基线”的结论。

## 机制诊断：当前更像 Q 信号过弱，而不是硬文本前向错误

对 26 个正式 cell 的最终阶段审计字段做了汇总：

| 数据集 | 分支 | Q entropy | Q gradient norm | residual gradient norm | 最大软硬文本误差 |
|---|---|---:|---:|---:|---:|
| WikiCS | Shared-Q | 1.674 | `8.49e-5` | `1.54e-2` | `1.90e-7` |
| WikiCS | No-residual | 0.946 | `5.51e-5` | `1.56e-2` | `1.96e-7` |
| SportsFit | Shared-Q | 1.170 | `2.32e-5` | `6.56e-3` | `2.01e-7` |
| SportsFit | No-residual | 0.980 | `3.40e-5` | `9.08e-3` | `1.71e-7` |
| BookHis | Shared-Q | 0.259 | `7.09e-5` | `3.61e-2` | `1.53e-7` |
| BookHis | No-residual | 0.226 | `8.79e-5` | `5.34e-2` | `1.53e-7` |

这里的 `Q gradient norm` 比 residual gradient norm 小约两个数量级；因此当前联合优化虽然“形式上”同时更新 Q 和 X residual，但实际参数变化更可能由 residual 分支主导。另一方面，straight-through 文本 materialization 的软硬前向误差约为 `1e-7`，说明当前主要问题不是软文本和最终保存文本不一致。

这给出下一步的单一修改方向：先做固定预算的 Q-gradient scale / loss contribution 诊断，确认 Q 是否被 `lambda_gradient`、distribution matching 和 anchor 项压制；不要同时加入 EM、复杂 attention、旧 memory refresh 或拓扑学习。只有证明 Q 梯度确实不足后，才考虑对 Q 分支进行受限重加权，并重新跑最小 paired confirmation。
