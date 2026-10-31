# v3 Canonical Continual 主表

日期：2026-09-15

所有数值为 dev Macro-F1（%），student seeds=`100..105`，batch=`16`，steps=`max(50,min(2048,16*roots))`，rate=`0.05`。`test_accessed=false`，test 行为 `%10==2`，不参与构造、选择或调参。

## 1. 七数据集主表

| 数据集 | 方法 | s0 累计 | s1 累计 | s2 累计 | s2 old | s2 current | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| Cora | append_fixed | 65.77 | 60.69 | 51.82 | 77.49 | 36.07 | 合规 |
| Cora | recluster | 65.77 | 68.33 | 52.44 | 75.61 | 46.49 | 合规压力测试 |
| Cora | full_history | 65.77 | 54.30 | 53.47 | 73.56 | 54.17 | 违规上界参照 |
| Cora | SCA | 65.77 | 60.69 | 51.82 | 77.49 | 36.07 | 小图正式统一 run |
| Citeseer | append_fixed | 61.25 | 73.97 | 58.89 | 72.88 | 59.31 | 合规 |
| Citeseer | recluster | 61.25 | 63.94 | 56.75 | 77.18 | 42.40 | 合规压力测试 |
| Citeseer | full_history | 61.25 | 68.94 | 57.06 | 68.66 | 66.40 | 违规上界参照 |
| Citeseer | SCA | 61.25 | 73.97 | 58.89 | 72.88 | 59.31 | 小图正式统一 run |
| WikiCS | append_fixed | 65.36 | 63.77 | 53.08 | - | - | 核心 continual |
| BookHis | append_fixed | 18.71 | 17.49 | 17.41 | - | - | 核心 continual |
| arXiv | random_real | 25.91 | 21.94 | 17.54 | 28.79 | 21.83 | matched real-node baseline |
| arXiv | k_center | 31.66 | 26.09 | 19.68 | 31.82 | 21.89 | matched real-node baseline |
| arXiv | herding | 27.20 | 21.17 | 18.52 | 31.04 | 22.66 | matched real-node baseline |
| arXiv | kmeans_medoid | 26.92 | 23.25 | 18.66 | 29.33 | 23.07 | matched real-node baseline |
| arXiv | append_fixed | 27.09 | 23.59 | 20.43 | 33.24 | 25.18 | 冻结 classincr3 |
| arXiv | recluster | 27.09 | 24.15 | 19.39 | 32.50 | 19.33 | 同协议诊断 |
| arXiv | full_history | 31.29 | 23.31 | 18.99 | 32.34 | 20.90 | 违规上界参照 |
| arXiv | SCA | 30.06 | 24.11 | 20.03 | 32.48 | 22.67 | 独立 selection seeds |
| BookChild | random_real | 15.32 | 13.86 | 13.43 | 18.36 | 25.09 | matched real-node baseline |
| BookChild | k_center | 13.14 | 13.95 | 12.03 | 14.96 | 24.66 | matched real-node baseline |
| BookChild | herding | 16.76 | 14.97 | 14.58 | 19.75 | 27.32 | matched real-node baseline |
| BookChild | kmeans_medoid | 16.22 | 14.59 | 13.61 | 17.92 | 25.96 | matched real-node baseline |
| BookChild | append_fixed | 11.30 | 14.27 | 13.97 | 17.51 | 28.76 | 核心 continual |
| BookChild | recluster | 11.38 | 11.31 | 12.92 | 18.03 | 27.00 | 同协议诊断 |
| BookChild | full_history | 14.63 | 13.09 | 12.08 | 16.72 | 24.90 | 违规上界参照 |
| BookChild | SCA | 12.28 | 13.59 | 15.04 | 19.53 | 30.22 | 独立 selection seeds |
| SportsFit | append_fixed | 15.79 | 23.47 | 25.90 | 31.25 | 56.80 | 核心 continual |
| SportsFit | random_real | 24.81 | 27.04 | 29.36 | 35.58 | 58.21 | matched real-node baseline |
| SportsFit | k_center | 34.58 | 30.83 | 30.03 | 35.59 | 56.85 | matched real-node baseline |
| SportsFit | herding | 24.12 | 24.66 | 27.06 | 31.70 | 57.03 | matched real-node baseline |
| SportsFit | kmeans_medoid | 23.98 | 27.00 | 29.07 | 35.92 | 55.25 | matched real-node baseline |
| SportsFit | recluster | 16.44 | 31.06 | 28.13 | 34.72 | 49.09 | 同协议诊断 |
| SportsFit | full_history | 21.79 | 21.43 | 29.21 | 35.92 | 58.85 | 违规上界参照 |
| SportsFit | SCA | 25.64 | 28.51 | 26.22 | 32.95 | 54.10 | 独立 selection seeds |

注：Cora/Citeseer 的 `old/current` 取正式统一 runner 的 stage 2 统计；WikiCS/BookHis 的已有 canonical 结果只保存了核心累计指标，因此不补写不存在的分解。三套大图均已完成 append/SCA canonical 与 recluster/full-history 诊断。大图候选构造使用确定性的 GPU 分块后端，student、split、预算、batch 和 steps 仍遵循同一 v3 协议。

## 2. SCA 路径与稳定性

| 数据集 | SCA 选择路径 | s2 SCA - append |
|---|---|---:|
| arXiv | `joint_g3 -> xonly_g5 -> joint_g8` | -0.40 |
| BookChild | `xonly_g5 -> joint_g3 -> joint_g8` | +1.07 |
| SportsFit | `joint_g8 -> xonly_g5 -> joint_g5` | +0.31 |

SCA 的大图结论是“有阶段性收益，但不是普遍增益”：BookChild 和 SportsFit 的最终累计指标高于 append，arXiv 最终略低于 append；arXiv 的 stage 0/1 短期优势在 stage 2 没有完全保留。该结果支持 response mechanism 的分析价值，但不支持把 greedy selector 宣称为稳定 SOTA。

## 3. 主表解读

- append 是当前最可靠的合规基线：历史 memory 不回读，构造预算和评测协议清楚，跨数据集不依赖某一条选择路径。
- full-history 在早期阶段可能更强，但到后期在 arXiv、BookChild 等类别增量流中被 append 或 SCA 反超，说明历史重凝聚会发生类别主导和结构失配。
- recluster 没有显示普遍优势；它有时提高旧类保持，但常以当前类塑性或最终累计 F1 为代价。
- k-center 是强但不稳定的 matched baseline：它在 arXiv 和 SportsFit 的早期/最终阶段都很有竞争力，但在 BookChild 的最终累计 F1 低于 random、herding 和 kmeans-medoid；因此当前不能归因于“ours 实现配置错误”，更准确的结论是构造空间与数据集分布存在交互。
- SCA 的正确论文定位是 `student-response calibrated selection + topology-preserving append`，并同时报告独立 selection seed、path instability 和负结果。

## 4. 结果来源

- 小图：`results/formal_unified_cora/`、`results/formal_unified_citeseer/`
- WikiCS/BookHis：`results/formal_unified_wikics/`、`results/formal_unified_bookhis/`
- arXiv：`results/canonical_v3_2026_09_15_arxiv/`、`results/diagnostic_v3_2026_09_15_arxiv/`
- arXiv matched baselines：`results/baseline_v3_2026_09_15_arxiv/`
- BookChild：`results/canonical_v3_2026_09_15_bookchild/`、`results/diagnostic_v3_2026_09_15_bookchild/`
- BookChild matched baselines：`results/baseline_v3_2026_09_15_bookchild/`
- SportsFit：`results/canonical_v3_2026_09_15_sportsfit/`、`results/diagnostic_v3_2026_09_15_sportsfit/`
- SportsFit matched baselines：`results/baseline_v3_2026_09_15_sportsfit/`
- 外部方法资格审计：[MATCHED_SOTA_COMPARISON_V3_ZH_2026-09-15.md](MATCHED_SOTA_COMPARISON_V3_ZH_2026-09-15.md)

## 5. 已完成范围与限制

- 已完成 arXiv、BookChild、SportsFit 的六 seed canonical continual：`append_fixed` 与 SCA；并完成 `recluster`、`full_history` 诊断臂。
- 已完成七数据集的统一表格整理，但 WikiCS/BookHis 当前只保留核心 append 指标，尚未补齐所有记忆管理臂，因此不能把“七数据集完整”误读成“七数据集每个臂都完整”。
- 外部 continual SOTA 已完成严格资格审计；由于公开方法没有同时满足 TAG、label-free、zero-shot、class-incremental、frozen-old 和 v3 student 协议，外部数字不直接排序进入主表。
