# 大图外部方法补充实验

日期：2026-10-11

本轮补齐了 STAD、DANCE、SR-GM 和 GraphCLIP 在大图上的实验。由于这些方法的原论文任务和数据访问接口并不完全等同于本文的历史受限持续 TAG 凝聚协议，结果按协议层级分开报告。

## 1. 实验层级

| 层级 | 方法 | 结果含义 |
|---|---|---|
| Strict matched | Unified-Q、GECC-hard-TAG、CaT-CCTP | 同一 Prompt Student、5% cumulative memory、current-delta-only、frozen synthetic prefix；可进入严格主表 |
| Mechanism-adapted large | STAD、DANCE、SR-GM | 保留原方法的主要机制并适配到本文大图流水线；不是官方端到端复现，不能与 strict 主表直接排名 |
| Official reference | GraphCLIP | 官方 GPS-12L checkpoint 在完整原图 ego context 上推理；不是凝聚 memory，也不是 continual constructor |

STAD、DANCE、SR-GM adapter 使用 3 个 Student seeds（150/151/152）、共同 Prompt GraphText Student、5% cumulative memory、三阶段 base+4 类别流。构造阶段只读取当前类别增量和冻结 synthetic prefix；阶段类别和 quota 分配使用构造协议中的类别标签，但不使用 report/test labels。旧 synthetic rows 不被重新凝聚。

## 2. 机制保留和适配

| 原方法 | 本项目名称 | 保留机制 | 为大图和持续协议所做的适配 |
|---|---|---|---|
| Semi-Supervised Text-Attributed Graph Distillation | STAD adapter | 图文协同 sketch、类内聚类、文本 payload 材料化 | 用 current delta 的图特征、文本和一跳邻域构造 joint sketch；用真实来源文本的多片段摘要替代 LLM 生成，避免大图逐节点调用 LLM；旧 prefix 冻结 |
| DANCE | DANCE adapter | neighbor-gated evidence、round-wise refresh、可审计来源 | 每阶段对 current delta 重新构造；每个 synthetic row 保存 root text 和邻居 evidence pack；不访问历史 raw，不做联邦多客户端聚合 |
| SR-GM | SR-GM adapter | gradient matching、梯度冲突正交投影、structural damping | 在每个新 block 上做小规模图文 gradient matching；使用梯度解耦和固定 kNN 结构阻尼；为保证大图可扩展只在最多 512 个 active rows 上做 refinement |
| GraphCLIP | official frozen reference | 官方 GPS 图编码器和 MiniLM text tower | 使用官方 checkpoint；完整原图 2-hop ego；大图 reference 对 ego 子图设置确定性的 512-node cap，防止高阶节点导致推理爆炸 |

## 3. 大图结果

数值为 `Final Macro-F1 / Accuracy`。adapter 的括号为 3 个 Student seeds 的标准差；GraphCLIP 没有 Student seed，因为使用单个 frozen checkpoint。

### ArXiv

| 方法 | Macro-F1 | Accuracy | 层级 |
|---|---:|---:|---|
| Unified-Q | 52.25 | 71.59 | strict matched |
| GECC-hard-TAG | 51.09 | 70.27 | strict matched |
| CaT-CCTP | 36.74 | 63.62 | strict matched |
| SR-GM adapter | 20.97 +- 0.58 | 32.13 +- 2.12 | mechanism-adapted |
| STAD adapter | 17.69 +- 0.78 | 24.70 +- 2.18 | mechanism-adapted |
| DANCE adapter | 17.61 +- 0.72 | 23.59 +- 2.27 | mechanism-adapted |
| GraphCLIP official frozen, 512-cap | 25.35 | 41.80 | official reference |

### SportsFit

| 方法 | Macro-F1 | Accuracy | 层级 |
|---|---:|---:|---|
| Unified-Q | 78.60 | 83.93 | strict matched |
| GECC-hard-TAG | 77.32 | 85.52 | strict matched |
| CaT-CCTP | 73.75 | 81.94 | strict matched |
| DANCE adapter | 33.07 +- 2.85 | 40.11 +- 4.75 | mechanism-adapted |
| SR-GM adapter | 30.89 +- 4.87 | 37.22 +- 11.39 | mechanism-adapted |
| STAD adapter | 28.71 +- 3.74 | 33.77 +- 4.98 | mechanism-adapted |
| GraphCLIP official frozen, 512-cap | 16.64 | 20.31 | official reference |

### BookHis

| 方法 | Macro-F1 | Accuracy | 层级 |
|---|---:|---:|---|
| Unified-Q | 41.97 | 80.34 | strict matched |
| GECC-hard-TAG | 41.23 | 79.97 | strict matched |
| CaT-CCTP | 24.83 | 75.45 | strict matched |
| STAD adapter | 15.91 +- 0.57 | 22.88 +- 2.31 | mechanism-adapted |
| SR-GM adapter | 15.74 +- 1.87 | 26.43 +- 4.79 | mechanism-adapted |
| DANCE adapter | 14.68 +- 0.56 | 20.51 +- 1.11 | mechanism-adapted |
| GraphCLIP official frozen, 512-cap | 26.60 | 45.12 | official reference |

### WikiCS

| 方法 | Macro-F1 | Accuracy | 层级 |
|---|---:|---:|---|
| CaT-CCTP | 80.35 | 83.44 | strict matched |
| Unified-Q | 79.86 | 82.85 | strict matched |
| GECC-hard-TAG | 79.08 | 82.22 | strict matched |
| STAD adapter | 52.51 +- 2.39 | 55.01 +- 1.42 | mechanism-adapted |
| DANCE adapter | 52.31 +- 3.53 | 57.95 +- 3.00 | mechanism-adapted |
| SR-GM adapter | 37.92 +- 5.85 | 41.25 +- 5.46 | mechanism-adapted |

## 4. 客观解释

1. 这些新增结果完成了“方法覆盖”，但没有形成新的严格 continual SOTA 证据。STAD、DANCE、SR-GM 的数值明显低于当前 strict Prompt Student 主表，最主要原因是它们的原生目标、文本材料化和拓扑构造并不是为本文的 frozen-prefix prompt Student 设计的。
2. SR-GM 在 ArXiv 上的 adapter F1 高于 STAD/DANCE，但在 WikiCS 上明显退化，说明其梯度解耦和结构阻尼对输入特征尺度、图文对齐方式和当前 Student 的敏感性较强。不能把它的原论文 multimodal benchmark 数字直接迁移到 TAG continual setting。
3. DANCE 在 SportsFit 的 adapter 相对另外两个 adapter 较好，和它依赖邻居 evidence pack 的机制一致；但它的原任务是 federated TAG condensation，当前集中式单流适配没有模拟客户端异质性，因此不能称为 DANCE 原生结果。
4. GraphCLIP reference 只说明官方 frozen graph-text 模型在部分大图上的原图推理位置。它使用完整原图上下文和 512-node capped ego，不使用 5% synthetic memory，所以只能放 reference 表。

## 5. 输出文件

- 机器可读统一表：[large_external_comparison.csv](../../results/large_external_adapters_20261011/large_external_comparison.csv)
- 完整附录表：[LARGE_EXTERNAL_BASELINE_APPENDIX_20261011.md](../../results/large_external_adapters_20261011/LARGE_EXTERNAL_BASELINE_APPENDIX_20261011.md)
- F1 图：[large_external_f1.png](../../results/large_external_adapters_20261011/figures/large_external_f1.png)
- Accuracy 图：[large_external_accuracy.png](../../results/large_external_adapters_20261011/figures/large_external_accuracy.png)
- STAD/DANCE/SR-GM runner：[run_external_large_adapters_20261011.py](../../code/v2_llm_fusion/run_external_large_adapters_20261011.py)
- GraphCLIP reference runner：[eval_official_graphclip_report.py](../../code/v2_llm_fusion/eval_official_graphclip_report.py)

结论：四类方法的大图实验已经完成，但论文中应把 STAD/DANCE/SR-GM 写成 mechanism-adapted external references，把 GraphCLIP 写成 official frozen-model reference；严格方法优劣仍以 Unified-Q、GECC-hard-TAG、CaT-CCTP 的同 Student 主表为准。
