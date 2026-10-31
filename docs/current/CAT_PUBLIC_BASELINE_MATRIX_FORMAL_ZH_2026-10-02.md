# CaT-TAG 公开基线统一比较报告

日期：2026-10-02  
状态：第一项公开/强基线统一协议比较完成；最新类共享真实 payload 已完成四顺序正式复跑。

## 1. 这轮回答的问题

本轮不是继续扫描文本参数，而是回答：

> 最新的 CaT-conditioned class-shared real payload，是否在相同持续 TAG 协议下优于已有的 CaT 文本控制、k-center、medoid 和 GECC 适配基线？

先前的类共享实验只覆盖了部分顺序，且曾使用不同的文本来源上限。本轮将正式矩阵统一为 5 个文本来源、320 字符预算，并补齐 Cora/Citeseer 的四个类别顺序和六个 Student seeds。

## 2. 严格比较协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19611`、`order19612`、`order19613`、`order19614` |
| 阶段 | 3 个类别增量阶段 |
| memory | 合法训练池的 5% cumulative rows |
| 访问边界 | 当前 raw delta + 已材料化 synthetic parent；不回读历史 raw |
| 旧记忆 | synthetic prefix append-only，不修改 |
| Student | 原有两层 GCN + text projection；冻结 MiniLM；双向图文 InfoNCE |
| Student seeds | 150--155，共 6 个 paired seeds |
| 文本预算 | 每个 payload 最多 5 个来源、320 字符 |
| 拓扑 | CaT/GECC 与文本实验使用 edge-free memory 语义 |
| 指标 | Final Macro-F1、Accuracy、Stage-average F1、Old F1、New F1、最差阶段 |
| 选择 | 不使用最终 report/test F1 选择构造结果 |

正式当前矩阵的构造、聚合和边界审计共通过 48 个 stage checks。所有 current-delta 文本来源都属于对应阶段 delta；历史 raw 读取为 0；旧 memory、拓扑和 `X` 在该文本实验中均未更新。

## 3. 方法定义

### 3.1 CaT-TAG nearest

CaT 类条件 MMD/CGM 生成合成特征 `X`，再按 `X` 与当前同类图传播特征的相似度选择文本来源。这是简单文本附着控制。

### 3.2 CaT-TAG CCTP

CaT 生成 `X`；对每个类别在当前 delta 内计算文本中心，用中心性和固定 MMR 选择最多 5 个真实文本来源，拼接到 320 字符后重新编码。它是当前文本方法的基本类中心版本。

### 3.3 CaT-TAG class-shared real payload

这是本轮候选方法：

1. CaT 按当前 delta 生成新增合成 `X`；
2. 每个类别建立有界 current-delta 支持池，候选由 CCTP 来源、`X` 的特征近邻和类别文本中心近邻组成；
3. 将候选真实文本切成有界 chunk；
4. 使用图分数、文本中心性、prompt/对比性和冗余惩罚进行 TAGSAM-inspired 贪心覆盖选择；
5. 对每个类别的 CaT 行选一个最接近类别 `X` 中心的代表行；
6. 该类别的所有合成行共享同一个真实材料化 payload；
7. 拼接结果严格限制为最多 5 个来源和 320 字符，再由冻结 MiniLM 编码；
8. 将新 block 追加到旧 synthetic prefix，Student 从当前 memory scratch 训练。

这里的 `TAGSAM-inspired` 只是机制适配，不是官方 TAGSAM 端到端复现。它的核心变化不是增加文本长度，而是控制同一类别内 payload 的方差，避免每个合成行绑定一组互相分裂的文本目标。

### 3.4 GECC-hard-TAG adapter

当前 delta 内做按类 hard clustering，多重重启后选择真实/合成中心，再使用统一 TAG Student 评价。它不是 GECC 官方监督下游结果，而是机制保持的 TAG 接口适配。

### 3.5 k-center / medoid native append

二者是 label-free 真实节点 coreset 控制：分别使用几何覆盖和 medoid 选择当前 delta 节点，按同样的 cumulative rows 追加。它们是强的低成本代表点参考，不是合成 TAG 方法。

## 4. 严格主表

数值是 8 个 dataset-order cell 的等权平均；分数单位为百分比。构造时间只在同一实现/同一 artifact 语义内解释，不能据此把旧封存基线当成零成本。

| 方法 | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 | 最差 stage | 构造秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CaT-TAG nearest | 54.46 | 60.05 | 50.35 | 50.91 | 60.13 | 33.20 | 13.35 |
| CaT-TAG CCTP | 66.44 | 74.64 | 71.63 | 64.75 | 68.97 | 46.17 | 19.56 |
| **CaT-TAG class-shared real payload** | **70.23** | **76.34** | **75.71** | **67.35** | **74.43** | **49.13** | 19.55 |
| GECC-hard-TAG adapter | 49.45 | 54.79 | 44.19 | 46.45 | 54.39 | 24.75 | 0.41 |
| k-center native append | 52.20 | 57.93 | 52.84 | 48.59 | 57.95 | 8.41 | 0.02 |
| medoid native append | 49.47 | 55.19 | 45.39 | 47.33 | 52.60 | 15.12 | 0.18 |

## 5. Paired 差值

| 对照 | 类共享 payload - 对照 Final F1 | Accuracy 差值 | 正向 F1 cell | 最差差值 |
|---|---:|---:|---:|---:|
| CaT-TAG nearest | +15.77 | +16.29 | 8/8 | +6.33 |
| CaT-TAG CCTP | +3.79 | +1.70 | 6/8 | -3.22 |
| GECC-hard-TAG adapter | +20.78 | +21.55 | 8/8 | +12.77 |
| k-center native append | +18.03 | +18.41 | 8/8 | +8.26 |
| medoid native append | +20.76 | +21.15 | 8/8 | +10.95 |

### 5.1 每个顺序的 Final F1

| 数据集 | 顺序 | CCTP | 类共享 payload | nearest | GECC | k-center | medoid |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | 19611 | 65.07 | **77.46** | 53.79 | 51.60 | 54.54 | 46.39 |
| Cora | 19612 | 64.61 | **77.15** | 51.42 | 47.13 | 50.60 | 46.58 |
| Cora | 19613 | 65.28 | **67.37** | 53.23 | 46.43 | 43.98 | 53.67 |
| Cora | 19614 | 64.16 | **69.78** | 54.37 | 45.60 | 53.32 | 51.06 |
| Citeseer | 19611 | 66.30 | **68.77** | 57.28 | 49.01 | 51.72 | 47.33 |
| Citeseer | 19612 | 70.10 | 68.13 | 51.04 | 54.37 | 59.87 | 48.96 |
| Citeseer | 19613 | 67.90 | 64.68 | 58.35 | 51.91 | 52.37 | 53.73 |
| Citeseer | 19614 | 68.09 | **68.51** | 56.20 | 49.56 | 51.19 | 48.08 |

## 6. 结果说明

1. **相对简单 coreset，当前候选有明显优势。** 在这组统一 Student 和 memory 协议下，类共享 payload 的 Final F1 高于 k-center 和 medoid 的全部 8 个 cell；但这不等于已经证明优于所有 TAG 凝聚论文方法，因为这里的 GECC/CaT 都是接口适配。
2. **主要增益来自类内文本一致性。** 相比 CCTP，平均 Final F1 提升 `+3.79 pp`，New F1 提升 `+5.46 pp`。这与之前逐行 payload 和双原型失败的证据一致：当前 Student 更容易利用稳定的类级文本锚点。
3. **不是每个顺序都超过 CCTP。** Citeseer 的 19612、19613 分别低于 CCTP 约 `-1.97` 和 `-3.22 pp`；因此当前版本应写成“平均提升、对强 coreset 稳定更优”，不能写成“跨顺序稳定超过 CCTP”。
4. **Cora/Citeseer 的绝对值不能直接跨数据集排序。** 类别分布、类别顺序和 report-dev 节点构成不同，应该只在同一 dataset/order 内做归因。
5. **构造成本仍需谨慎报告。** 当前候选和 CCTP 的构造量级接近，但 GECC/k-center 的旧结果来自已封存适配矩阵；论文中应报告完整构造成本，而不能只比较 Student 时间。

## 7. 已有但不混入严格主表的结果

以下实验之前做过，但协议不同，暂不伪装成四顺序严格基线：

- random real-node coreset；
- full-history synthetic re-condensation；
- raw full-data reference；
- direct Student continual learning / delta fine-tuning；
- static TAGSAM/PUMA/STAD 的 native 或监督下游结果。

它们可放在 appendix 的 protocol-stratified reference 表中。尤其 STAD 是监督 GCN 下游，不能与当前 InfoNCE Student 直接排主榜；direct Student continual 是 stateful learning control，也不是 memory condensation 方法。

## 8. 当前判断和下一步

第一项基线任务已经完成到“严格小图公开适配主表”层面。当前最合理的论文表述是：

> 在历史 raw 不可回读、固定 5% memory 和统一图文 InfoNCE Student 的持续 TAG 协议下，CaT-conditioned class-shared real payload 在最终阶段平均性能上优于 CaT 的简单文本附着和 label-free coreset 适配；其优势主要来自类级真实文本 payload 对类内语义方差的抑制，但在部分类别顺序上仍低于 CCTP，且尚不能宣称对所有公开 TAG 凝聚方法 SOTA。

正式结果和审计文件：

- `results/cat_public_baseline_matrix_formal_20261002/summary.md`
- `results/cat_public_baseline_matrix_formal_20261002/summary.json`
- `results/cat_public_baseline_matrix_formal_20261002/audit.json`
- `tools/summarize_cat_public_baseline_matrix_20261002.py`
- `tools/audit_cat_public_baseline_matrix_20261002.py`
- `code/v2_llm_fusion/run_cat_multimodal_minimal_20261002.py`

下一步不建议立即继续扫描文本权重。应先决定是否把该版本迁移到三张大图，并在大图上只做一次有限协议标定；同时保留 CCTP 作为最重要的 paired text baseline。
