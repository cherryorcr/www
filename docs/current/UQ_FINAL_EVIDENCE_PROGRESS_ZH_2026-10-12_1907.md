# Unified-Q-EF 最终证据进度（19:07 快照）

本文件是增量证据记录，不替代预注册、旧 sealed 表或最终论文主表。源快照：
`results/uq_final_evidence_20261012/snapshot_1907.json`；只读汇总：
`results/uq_final_evidence_report_20261012_1907/summary.md`。采集器检查 16 个结果项、58 条完整轨迹，无校验错误；两个运行中的轨迹也保留在快照中。

## 已完成：ArXiv 真实 5% 确认

合法构造池 118539 行，最终 synthetic memory 5927 行；3 个类别顺序 × 3 个构造种子，每个 cell 内平均 3 个 Student 种子。统一 Prompt GCN、MiniLM、2048 步、batch 24。以下是 sealed test，单位为百分数；各外部方法均为当前增量、冻结旧合成前缀的机制适配，而非官方原协议复现。

| 方法 | 配对 cell | Final Macro-F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unified-Q-EF | 9 | 52.91 | 71.15 | 52.25 | 54.27 | 62.23 |
| CaT-CCTP | 9 | 34.98 | 60.41 | 33.77 | 37.50 | 47.36 |
| GECC-hard | 9 | 52.68 | 70.94 | 51.45 | 55.24 | 62.34 |

相对 GECC-hard，Unified-Q-EF 的配对 F1 为 +0.22 pp，95% CI [-0.34, 0.79]，4/9 个 cell 为正，最差 -0.47 pp。不能据此声称稳定优于 GECC。相对 CaT-CCTP 为 +17.93 pp，95% CI [16.81, 19.05]，9/9 为正；它只是其中一个机制适配基线，不能据此推广为优于所有持续 TAG 方法。CI 单位为顺序 × 构造种子 cell，不把同一 memory 的三个 Student 视为独立样本。旧 ArXiv 6773/118539 为 5.71%，不进入此表。

## 进行中：共享 Q 归因

同一类别顺序的前两个构造种子已经形成配对；第三种子及其余无 residual 对照尚未齐。sealed Final F1（shared minus independent）：SportsFit -0.19 pp，BookHis -2.52 pp，ArXiv +0.13 pp。每图仅两个 cell，不能作为显著性结论。已有无 residual、等 active 参数预算、soft-stored 对照保留，但尚未完成所有配对；等预算参数化也改变了更新子空间，不是单纯参数数量隔离。

阶段诊断的前两个种子显示：共享分配的 assignment 参数数约为独立分配的一半；三图观察到的构造峰值分别为 ArXiv 12.11 vs 16.30 GiB、BookHis 2.78 vs 3.31 GiB、SportsFit 25.01 vs 32.92 GiB（shared vs independent）。图文 top-1 支持一致率 shared 为 1.00，这是构造定义保证的性质；independent 的观测均值约为 0.55、0.71、0.60。两者硬 slot 覆盖均为 1.00，故不能以覆盖率解释性能差异。上述阶段均值受不同阶段规模影响，不等于最终 memory 的汇总值。当前证据支持参数/显存和支持一致性的取舍，不支持共享 Q 的稳定精度增益。soft-only embedding 无法仅从保存的参考字符串重编码恢复，也应作为存储接口限制报告。

## 高保真强基线状态与偏差

GECC-soft 保留 GraphSlim 的 fuzzy c-means 与 balanced restart selection，文本使用共同 CCTP；TAGSAM 保留官方 GPT2-large PMI 选择，但蒸馏预算缩为 100 × 10；STAD 保留核心自训练、候选文本及合成拓扑，但本地 Qwen2.5-1.5B 替代官方 qwen3-max，且其内部 10%/10%/80% 标签划分与全监督主方法不同。三者均只可称机制适配，不能混称 official reproduction。小图接口检查已通过；BookHis 的 GECC-soft 仍在运行，暂无完整大图分数。

STAD 的文本生成偏离预注册中的“强制标签/摘要格式，失败停止”：核对官方原始文本候选筛选后，当前代码接收非空原始文本、记录格式遵循率，不强制 schema；小图检查中严格格式仅约 1/3。保留原预注册条目和失败日志，此处单独披露修订。生成质量和候选偏题可能显著影响强基线表现；若大图成绩较低，不能直接据此断言官方 STAD 弱于本方法。STAD 合成边必须留在自身 runner，不能由 edge-free 公共 runner 丢弃。

## 后续判定

运行中的 Q 归因与三个适配强基线按预设顺序继续；每批完成才生成新快照，不覆盖 sealed 原始结果。新结果进入论文前，逐 cell 校验类别顺序、合法池、report/test 掩码、阶段 ID、累计预算、MiniLM 和 Student 训练预算。强基线目前最多单顺序/单构造 seed，完整时只可做同 cell 探索性对照，不可伪造跨顺序置信区间。若共享 Q 的精度结论继续为负，论文贡献应落在统一图文支持及构造接口、成本和任务表现上，而非声称它单独导致提升。
