# XSP-Sinkhorn 独立 construction seed 确认报告

状态：独立 seed 已完成；Student 泛化验证已启动，尚未完成。

## 1. 实验目的

本轮只确认当前最有希望的低成本组件 **ST-Sinkhorn** 是否能在独立 construction seed 下复现相对 Current XSP 的收益。这里的 Current XSP 是 CaT 生成 `X` 后，用 X-conditioned slot partition 构造文本 payload，但不使用 straight-through assignment；ST-Sinkhorn 只增加 straight-through Sinkhorn，使 slot assignment 的离散前向结果在反向传播时仍能把 transport/alignment 梯度传回 `X`。

本轮不加入 coherence anchor、hard consistency、拓扑 revision、动态预算、selector 或新 Student；也不重新读取历史 raw graph。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Arxiv、BookChild、SportsFit |
| 阶段 | base+4，共 5 个阶段 |
| memory | cumulative fixed-rate 5% |
| construction seed | `314159` |
| Student | 原始 edge-aware 两层 GCN + text projection + 双向 InfoNCE |
| Student seeds | `100--105`，6 个 paired seeds |
| batch / steps | `16` / `max(50, min(2048, 16 * memory_rows))` |
| raw 访问 | 仅当前 delta；历史 raw reads = 0 |
| 监督信息 | 不使用 report/test labels 构造或选择 |
| 旧 memory | 不修改；旧 X、文本、拓扑和 budget 均保持不变 |

两种方法使用同一个 construction seed、相同 Student seeds 和相同阶段预算。结果来自冻结 memory 的完整逐阶段评估，而不是从旧汇总表复制。

## 3. 独立 seed 结果

所有分数为百分比，均为 6 个 Student seeds 的均值。

| 数据集 | 方法 | Final F1 | Accuracy | Old F1 | New F1 | Stage-average F1 |
|---|---|---:|---:|---:|---:|---:|
| Arxiv | Current XSP | 21.73 ± 2.07 | 34.52 ± 4.11 | 24.24 | 7.47 | 32.91 |
| Arxiv | ST-Sinkhorn | **22.05 ± 1.49** | **35.24 ± 4.31** | **24.46** | **8.42** | **33.66** |
| BookChild | Current XSP | **13.74 ± 2.03** | **14.90 ± 5.05** | **14.70** | 7.08 | 33.02 |
| BookChild | ST-Sinkhorn | 12.99 ± 0.96 | 12.57 ± 2.56 | 13.68 | **8.14** | **34.54** |
| SportsFit | Current XSP | 29.41 ± 3.59 | 33.44 ± 9.96 | 31.02 | 20.55 | 55.07 |
| SportsFit | ST-Sinkhorn | **31.72 ± 2.27** | **35.22 ± 5.42** | **32.62** | **26.73** | **56.70** |

### 3.1 ST 相对 Current XSP 的差值

| 数据集 | Final F1 | Accuracy | Old F1 | New F1 | Stage-average F1 |
|---|---:|---:|---:|---:|---:|
| Arxiv | +0.32 | +0.72 | +0.21 | +0.95 | +0.76 |
| BookChild | -0.76 | -2.33 | -1.02 | +1.06 | +1.51 |
| SportsFit | +2.30 | +1.79 | +1.60 | +6.18 | +1.63 |
| **三图等权平均** | **+0.62** | **+0.06** | **+0.26** | **+2.73** | **+1.30** |

## 4. 与首轮 construction seed 的关系

首轮 seed `270927` 的三图等权结果为：

| 方法 | Final F1 | Accuracy | Old F1 | New F1 |
|---|---:|---:|---:|---:|
| Current XSP | 20.69 | 25.66 | 21.90 | 13.52 |
| ST-Sinkhorn | 22.69 | 27.70 | 24.25 | 13.62 |
| ST - Current | +2.00 | +2.04 | +2.35 | +0.10 |

独立 seed 的平均增益下降到 `+0.62 pp`，说明 ST-Sinkhorn 的方向性信号可以复现，但增益大小受 construction seed 和数据集影响，不能把首轮 `+2.00 pp` 当作稳定提升幅度。

## 5. 结果说明

1. **ST-Sinkhorn 不是完全偶然的单 cell 提升。** Arxiv 和 SportsFit 在独立 seed 上仍为正，三图等权 Final F1 仍为正；因此“让 assignment 反向传播到 X”值得保留。
2. **收益主要不是统一的旧类保持。** SportsFit 同时提升 Old/New F1，Arxiv 轻微提升；BookChild Final F1、Accuracy 和 Old F1 下降，但 New F1 上升。这说明 assignment 可微化改善了部分 slot 的可学习性，却没有解决所有数据集上的分布漂移。
3. **BookChild 是当前主要风险 cell。** 它的 stage-average F1 和 New F1 上升，但最终累计指标下降，说明后期旧类累积或类别比例造成了 trade-off。不能用单一 Final F1 判断 ST 在该数据集完全失效，也不能把 stage-average 提升直接写成总体收益。
4. **当前核心问题仍是跨数据集稳定性，而不是是否存在梯度路径。** 梯度路径和独立 seed 正向信号均已确认；下一步要验证这种信号能否迁移到不同 Student，而不是继续加入更多 anchor 或 selector。

## 6. 当前方法定位

暂定保留：

```text
CaT X initialization
-> current-delta-only X-conditioned graph/text cost
-> class-restricted balanced entropic Sinkhorn
-> straight-through assignment for X gradients
-> slot-conditioned real text materialization
-> append-only edge-free memory
```

不纳入当前主线：

- coherence anchor：首轮无稳定增益；
- hard consistency：首轮收益更高但成本约 2.7 倍且 New F1 下降；
- 动态 budget、response selector、拓扑 revision、GraphGPS Student；
- 无界旧 memory refresh。

## 7. 下一步

1. 完成冻结 Current/ST memory 上的 Student 泛化：原始两层 GCN、单层 GCN；构造和文本完全不变。
2. 在同一冻结代码快照下重跑 Current XSP、ST-Sinkhorn、class-shared、k-center、medoid、GECC-TAG 和 CCTP top-5，避免历史代码快照差异污染主表。
3. 汇总 Macro-F1、Accuracy、Old/New F1、stage-average、worst-stage、construction time 和 Student time。
4. 若 ST 在至少两个 Student 和至少两个数据集上保持正向，正式将它冻结为主方法核心；否则保留为有效组件，并回到 class-shared 与 ST 的公平基线比较，不再继续堆叠新模块。

## 8. 文件与审计

- 独立确认结果：`results/x_slot_partition_confirm_20261004_gpu0/`、`results/x_slot_partition_confirm_20261004_gpu1/`
- 构造代码：`code/v2_llm_fusion/x_conditioned_payload_partition.py`
- 大图 runner：`code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py`
- Student 泛化脚本：`code/v2_llm_fusion/evaluate_xsp_student_generalization_20261004.py`

远程输出已确认三图全部完成；Student 泛化实验正在两张 4090 上运行。
