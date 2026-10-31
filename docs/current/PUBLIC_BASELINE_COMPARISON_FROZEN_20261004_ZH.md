# 冻结代码快照下的公开基线比较

日期：2026-10-04  
结果目录：`results/public_baselines_frozen_20261004b_gpu0`、`results/public_baselines_frozen_20261004b_gpu1`

## 1. 实验协议

本轮使用同一个冻结 runner，对 Arxiv、BookChild、SportsFit 执行 `base+4` 持续类别到达协议：

- memory：5% cumulative rate；
- construction seed：314159；
- Student：原始 edge-aware 两层 GCN、文本投影和双向 InfoNCE；
- Student seeds：100--105；
- Student batch：16；
- Student steps：`max(50, min(2048, 16 * memory_rows))`；
- 构造阶段只读取当前 delta，`historical_raw_reads=0`；
- 不使用 report/test labels 构造 memory；
- 所有方法使用同一阶段类别顺序、memory rows、Student 和评估规则。

需要强调：`cat_mmd_tag_*` 和 `gecc_hard_tag` 是在当前持续 TAG 协议下的机制适配版本，不是对应论文的官方端到端复现。

## 2. Final stage 结果

指标均为百分数。`XSP` 指 `cat_mmd_tag_x_slot_partition`，即 CaT X 初始化加 X-conditioned slot partition；`XSP-ST` 是加入 straight-through Sinkhorn 的变体。

| 数据集 | 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | 构造时间/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Arxiv | random real | 17.36 | 23.18 | 19.04 | 7.85 | 31.46 | 0.10 |
| Arxiv | k-center | 19.43 | 30.69 | 21.47 | 7.90 | 29.37 | 0.09 |
| Arxiv | medoid | 17.92 | 25.71 | 19.80 | 7.25 | 30.56 | 0.75 |
| Arxiv | CCTP top-5 | 19.36 | 29.18 | 21.63 | 6.53 | 31.74 | 7.84 |
| Arxiv | GECC hard TAG | 19.96 | 31.15 | 22.07 | 8.02 | 31.19 | 0.83 |
| Arxiv | XSP | **21.05** | **34.44** | **23.51** | 7.09 | **32.68** | 9.02 |
| Arxiv | XSP-ST | 20.81 | 32.20 | 23.10 | **7.85** | 32.85 | 10.00 |
| BookChild | random real | 12.69 | 12.47 | 13.38 | 7.85 | 34.23 | 0.02 |
| BookChild | k-center | 12.49 | 11.93 | 13.24 | 7.25 | 31.19 | 0.03 |
| BookChild | medoid | 14.45 | 14.20 | 15.39 | 7.88 | 36.72 | 0.18 |
| BookChild | CCTP top-5 | 13.66 | 13.14 | 14.59 | 7.17 | 34.72 | 5.02 |
| BookChild | GECC hard TAG | 14.88 | 14.67 | 15.75 | **8.78** | 38.37 | 0.26 |
| BookChild | XSP | **15.05** | **15.61** | **16.16** | 7.31 | 37.51 | 5.67 |
| BookChild | XSP-ST | 12.13 | 12.28 | 12.94 | 6.46 | 33.04 | 6.10 |
| SportsFit | random real | 26.14 | 29.76 | 27.33 | 19.55 | 49.89 | 0.12 |
| SportsFit | k-center | 29.42 | 32.01 | 31.42 | 18.46 | 53.95 | 0.14 |
| SportsFit | medoid | **30.17** | **35.12** | **31.79** | 21.22 | **57.32** | 1.82 |
| SportsFit | CCTP top-5 | 30.15 | 31.60 | 31.62 | **22.05** | 46.56 | 5.29 |
| SportsFit | GECC hard TAG | 25.09 | 25.54 | 26.81 | 15.62 | 45.22 | 4.44 |
| SportsFit | XSP | 28.68 | 31.32 | 30.26 | 20.02 | 56.35 | 6.11 |
| SportsFit | XSP-ST | 28.98 | 33.08 | 30.35 | 21.46 | 55.69 | 6.97 |

## 3. 三图平均

这里排除了 `class_assigned_prompt_margin`，因为它不是与 XSP 同一类的多 payload 方法，而是每个类别共享一个材料化 payload，并且显式使用类别 prompt 做选择；其异常高分需要单独审计，不能作为普通外部基线直接合并。

| 方法 | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 |
|---|---:|---:|---:|---:|---:|
| XSP | **21.60** | **27.12** | **23.31** | 11.47 | **42.18** |
| CCTP top-5 | 21.06 | 24.64 | 22.61 | 11.92 | 37.67 |
| medoid | 20.84 | 25.01 | 22.33 | **12.12** | 41.53 |
| XSP-ST | 20.64 | 25.86 | 22.13 | 11.92 | 40.52 |
| k-center | 20.45 | 24.88 | 22.04 | 11.21 | 38.17 |
| GECC hard TAG | 19.98 | 23.79 | 21.54 | 10.81 | 38.26 |
| random real | 18.73 | 21.80 | 19.92 | 11.75 | 38.53 |

相对非异常最强平均参照 medoid，XSP 的平均 Final F1 为 `+0.76 pp`，Accuracy 为 `+2.11 pp`，Old F1 为 `+0.98 pp`；但 New F1 为 `-0.65 pp`。因此当前收益主要来自整体/旧类保持，不是稳定的新类塑性提升。

## 4. class-assigned prompt-margin 的审计结论

该版本在 SportsFit 的 Final F1 为 `57.01`，明显高于其余方法，但构造审计显示：

- `class_shared=true`；
- `class_prototype_count=1`；
- 每个类别的所有合成行共享同一个 materialized payload；
- 候选选择显式使用 `prompt_emb` 和类别 margin；
- 不是 XSP 的多 slot payload 设定。

这会使每个类别的文本原型高度纯化，减少 Student 的文本噪声和同类 payload 差异；在 SportsFit 的类别名称与文本空间较匹配时，可能产生很强的类别先验效果。该版本在 Arxiv 为 `27.33`、BookChild 仅为 `3.45`，跨数据集不稳定，不能解释为普遍算法增益，也不应作为 XSP 主结果。

## 5. 当前判断

1. 公开基线重跑已经完成，且协议边界和数据访问约束满足当前持续 TAG 设定。
2. XSP 在 Arxiv 和 BookChild 的 Final F1、Accuracy 和 Old F1 均达到本轮非异常方法中的较好水平；SportsFit 上则略低于 medoid/CCTP。
3. XSP-ST 没有稳定优于 XSP，当前不应把 straight-through Sinkhorn 单独晋升为主方法结论。
4. 当前最稳妥的主方法表述是：**CaT 初始化的 X-conditioned slot partition，在固定 memory 和不可回读历史 raw graph 条件下，通过 X 条件的类别内 payload 分区保留多样性，并在两个大图上改善整体持续分类表现。**
5. 不能宣称普遍 SOTA；更准确的结论是平均小幅优于常规参照，同时暴露出新类塑性和数据集迁移仍是主要瓶颈。

另外，远程已有的 `x_slot_partition_confirm_20261004_gpu0` 与本轮冻结基线都记录了 construction seed `314159`，但 Arxiv/BookChild 的 XSP memory SHA 并不相同，Final F1 也分别相差约 `0.68 pp` 和 `-1.31 pp`。这说明它们不能直接当作独立 seed 结果合并；论文前需要固定代码快照、数据路径、CUDA/线程设置和 payload seed，并重新做一次真正可追溯的独立 construction seed 验证。

## 6. 后续只做必要验证

- 保留 XSP 作为论文主方法候选，暂不继续添加 GraphGPS、动态 selector、复杂 response gate 或无界 memory revision。
- 追加一个独立 construction seed，重点确认 Arxiv/BookChild 的 XSP 优势是否保持。
- 追加 XSP 的新类塑性诊断：per-stage new F1、class-wise recall 和 payload slot coverage。
- 对 class-assigned prompt-margin 单列为分析性消融或排除项，不把 SportsFit 高分放进主表的“公平强基线”结论。
- 在论文中同时报告构造成本：XSP 的完整序列构造时间显著高于 k-center/medoid，不能只报告 Student 时间。

## 7. 可复现文件

- [gpu0 summary](../../results/public_baselines_frozen_20261004b_gpu0/summary.json)
- [gpu1 summary](../../results/public_baselines_frozen_20261004b_gpu1/summary.json)
- [Arxiv results](../../results/public_baselines_frozen_20261004b_gpu0/arxiv_base4_results.json)
- [BookChild results](../../results/public_baselines_frozen_20261004b_gpu0/bookchild_base4_results.json)
- [SportsFit results](../../results/public_baselines_frozen_20261004b_gpu1/sportsfit_base4_results.json)
- [冻结 runner](../../code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py)
- [X-conditioned partition 实现](../../code/v2_llm_fusion/x_conditioned_payload_partition.py)
