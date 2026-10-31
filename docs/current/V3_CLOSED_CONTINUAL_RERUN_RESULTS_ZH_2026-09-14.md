# v3 封闭来源协议重跑结果

日期：2026-09-14  
远程环境：`pc@REMOTE_SERVER_HOST`，conda 环境 `tagsam`  
主协议：`tagsam-unified-v3-2026-09-13`

## 1. 为什么重跑

此前 Cora/Citeseer runner 只排除了 dev 节点，test 节点的特征、文本和边仍可能进入凝聚源。该版本不能作为最终协议结果。本轮将凝聚源严格限制为：

```text
pool = { original_id % 10 not in {1, 2} }
dev  = { original_id % 10 == 1 }
test = { original_id % 10 == 2 }
```

test 只保留为未读取分区，结果中记录 `test_accessed=false`。student 使用 6 个 seeds `100..105`、batch `16`，steps 为 `max(50, min(2048, 16 * roots))`。

## 2. 预算与协议审计

| 数据集 | pool stage counts | cumulative budgets | 最终 roots |
|---|---:|---:|---:|
| Cora | `577 / 985 / 604` | `28 / 78 / 108` | `108` |
| Citeseer | `898 / 1079 / 571` | `44 / 98 / 127` | `127` |

两套结果均满足：

- `protocol=tagsam-unified-v3-2026-09-13`；
- `student_batch=16`，seeds 为 `100..105`；
- `test_source_sealed=true`，`test_accessed=false`；
- student steps 随 roots 变化，而非固定 2048；
- append 的新类下限为 `15 * new_class_count`。

结果目录：

- `results/v3_unified_2026_09_14_remote/v3_unified_2026_09_14_closed_cora/`
- `results/v3_unified_2026_09_14_remote/v3_unified_2026_09_14_closed_citeseer/`

## 3. Cora

指标为 6 个 student seeds 的 dev Macro-F1 均值；`s0/s1/s2` 表示看到的类别阶段。

| arm | s0 | s1 | s2 cumulative | s2 current | s2 old mean |
|---|---:|---:|---:|---:|---:|
| append_fixed | 65.77 | 60.69 | **51.82** | 36.07 | 77.49 |
| recluster | 65.77 | 66.57 | 49.69 | 41.89 | 74.80 |
| full_history | 65.77 | 56.72 | **51.87** | 53.09 | 72.58 |

对应 student steps 为 `448 / 1248 / 1728`。最终阶段的 per-stage Macro-F1：

- append：`76.25 / 78.73 / 36.07`；
- recluster：`76.42 / 73.19 / 41.89`；
- full-history：`78.02 / 67.14 / 53.09`。

## 4. Citeseer

| arm | s0 | s1 | s2 cumulative | s2 current | s2 old mean |
|---|---:|---:|---:|---:|---:|
| append_fixed | 61.25 | 73.97 | **58.89** | 59.31 | 72.88 |
| recluster | 61.25 | 66.41 | 55.32 | 42.78 | 74.69 |
| full_history | 61.25 | 71.59 | 57.11 | 68.47 | 68.22 |
| raw upper bound | - | - | 55.45 | 58.84 | 69.40 |

对应 condensed student steps 为 `704 / 1568 / 2032`；raw upper 使用 `8192` steps。最终阶段的 per-stage Macro-F1：

- append：`66.22 / 79.54 / 59.31`；
- recluster：`71.01 / 78.38 / 42.78`；
- full-history：`59.97 / 76.48 / 68.47`；
- raw upper：`69.17 / 69.63 / 58.85`。

## 5. 当前科研判断

1. **只追加仍是最稳妥的持续操作。** `recluster` 在两个数据集最终阶段都低于 `append_fixed`：Cora `-2.13` pp，Citeseer `-3.57` pp。允许历史 synthetic 行重新聚类、合并和再融合，不能稳定改善可重训下游。
2. **full-history 不是可靠的性能上界。** Cora 与 append 基本持平（`51.87` vs `51.82`），Citeseer 反而低于 append（`57.11` vs `58.89`）。它依赖重新读取历史 raw 数据，仍只能作为违规诊断参照。
3. **当前结果不是 SOTA 证据。** 关闭 test-source 后预算变小，Cora/Citeseer 的最终 cumulative dev F1 也重新落在约 `50--59` 区间；不能继续引用旧 Gate 的 `58.07/56.99` 等数字来包装 v3。
4. **历史保持与新类塑性存在明显张力。** Cora append 的旧阶段均值较高但 current stage 只有 `36.07`；full-history current stage 提升到 `53.09`，代价是 stage-1 保持下降。Citeseer 也出现相同的阶段间权衡。
5. **论文主线应暂定为机制分析、负结果和可靠构造。** 当前最可信的工程结论是：冻结历史 synthetic memory、为新类保留显式预算、按统一协议评测；ASM、重聚类和复杂融合尚未给出跨数据集稳定增益。

## 6. 与旧版结果的边界

旧版 Cora budgets 为 `32/87/121`，Citeseer 为 `50/110/143`，因为 test-source 没有完全封闭。旧版结果仍可用于定位实现差异，但不能与本表纵向比较，也不能用于论文主表。

后续若继续做 k-center、ASM 或新构造，必须复用本文件的 pool、batch、steps、seeds 和 metadata 审计；任何偏离都单独命名为诊断实验。

## 7. 同协议 arXiv 补充结果

远程同批任务也已完成，结果均记录 `test_accessed=false`、6 个 student seeds 和 batch 16。

### Scale-gate

在固定扰动幅度 `0.9` 下，`n_pool=27,094` 时相对 frozen 的收益为 `+1.69` pp，bootstrap CI `[+0.27,+3.16]`；但 `n_pool=54,189` 时仅 `+0.34` pp（CI 跨 0），`n_pool=135,473` 时 `+0.04` pp（CI 跨 0）。更小池规模为负收益。当前 v3 证据更适合表述为“存在规模相关的局部扰动窗口”，不能宣称普遍的规模门槛或稳定收益。

### 类别增量 continual

arXiv budgets 为 `2259 / 4516 / 6773`，由于 roots 较大，三阶段 condensed student 均达到上限 `2048` steps。最终阶段 cumulative dev Macro-F1 为：

| arm | s2 cumulative |
|---|---:|
| append | 18.81 |
| full-history | 19.87 |
| raw | 20.42 |

这与 Cora/Citeseer 的趋势不同：在 arXiv 大图类别增量流中，append 没有追平违规 full-history 或 raw。该结果说明“只追加不重组”是小型 TAG 上的可靠工程规则，而不是跨规模、跨流式分布的普适性能定律；论文中应把它写成机制与边界条件，而非 SOTA 主张。
