# Full-history retention compatibility audit

日期：2026-10-15

## 结论

现有 full-history 数值可以保留，但只能称为 **historical matched edge-free reference**，不能作为当前 sealed exact-5% 主表的严格 paired upper bound。原因不是 Student 不一致，而是 full-history 快照与当前主实验在类别顺序、阶段预算和历史访问规则上并未完全对齐。

## 字段核对

| 字段 | Full-history reference | 当前 Unified-Q-EF 主表 | 是否一致 |
|---|---|---|---|
| Student | GraphTextPromptStudent，两层对称归一化 GCN | 同上 | 是 |
| 文本编码器 | 冻结 MiniLM | 同上 | 是 |
| Student steps / batch | 2048 / 24 | 2048 / 24 | 是 |
| Student seeds | 150/151/152 | 150/151/152 | 是 |
| 读出 | 无分类头，类别 prompt cosine | 同上 | 是 |
| memory operator | edge-free/self-loop-only | edge-free/self-loop-only | 是 |
| 评估 split | report-dev，未使用 test 做构造 | report-dev 与 sealed 分开 | 部分一致 |
| 类别顺序 | canonical-v3 groups | 当前 order 9101 groups | 否 |
| 最终 memory rows | ArXiv 6773 | ArXiv 5927 | 否 |
| 名义压缩率 | 6773/118539 = 5.71% | 5927/118539 = 5.00% | 否 |
| 历史 raw 访问 | 允许累计 raw 重读并重写旧支持 | current-delta-only，旧 synthetic prefix 冻结 | 否 |
| synthetic topology | edge-free | edge-free | 是 |

## 可保留的 retention 数值

这些数值来自 `results/full_raw_prompt_reference_20261011/full_raw_vs_unified_q.csv`，分子和分母是同一历史快照中的 report-dev 配对结果：

| Dataset | Full-history F1/Acc | 5% snapshot F1/Acc | F1 retention | Acc retention |
|---|---:|---:|---:|---:|
| WikiCS | 80.09 / 83.05 | 80.85 / 84.07 | 100.95% | 101.23% |
| SportsFit | 78.26 / 88.96 | 79.44 / 85.72 | 101.50% | 96.36% |
| BookHis | 44.78 / 83.09 | 44.98 / 82.55 | 100.45% | 99.35% |
| ArXiv | 47.72 / 71.57 | 52.25 / 71.59 | 109.50% | 100.03% |

这里的“5% snapshot”在 ArXiv 实际为 5.71%，因此表格标题和论文脚注不能写成严格 exact-5% retention。超过 100% 只表示该历史快照下压缩 memory 的 Student 分数高于 full-history reference，不是理论上界，也不能据此证明压缩方法普遍优于完整 raw。

## 论文推荐表述

可以写：

> Under a matched edge-free Student reference from an earlier report-development snapshot, the approximately 5% memory retains around 100% of the Macro-F1 and Accuracy of full-history raw training on the evaluated datasets. This reference uses the same Student and text encoder but a different class-order/budget snapshot from the sealed exact-5% evaluation, and is therefore reported as a qualified retention reference rather than a strict upper bound.

不建议写：

- “5% memory strictly preserves full-history performance under the same protocol”；
- “full-history is an oracle upper bound”；
- “Unified-Q achieves 109.5% of full-history performance”而不披露 ArXiv 的 5.71% 实际率和 order mismatch。

## 来源

- `results/full_raw_prompt_reference_20261011/EXPERIMENT_SUPPLEMENT_20261011.md`
- `results/full_raw_prompt_reference_20261011/full_raw_vs_unified_q.csv`
- `results/full_raw_prompt_reference_20261011/arxiv_full_raw_prompt_student_edgefree.json`
- `results/uq_final_evidence_20261012/confirmation/arxiv_o9101_c271006/protocol.json`

