# P 引导双原型内容生成独立确认报告

日期：2026-09-25  
实验目录：`results/tgc_p_content_two_proto_independent_20260925e/`

## 1. 目的

本轮验证开发集上表现较好的 **P-guided two-prototype** 是否能在未参与开发的类别顺序上稳定复现，并区分收益究竟来自：

1. P 引导的内容生成；
2. 同类 payload 的 semantic slot permutation；
3. 随机槽位重排；
4. 单纯替换 R11 内容。

本轮不改变 Student、C0 topology、文本预算和报告 seeds，也不使用最终 F1 选择候选。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 独立类别顺序 | `order19605`、`order19606`、`order19608` |
| 阶段 | stage 0、1、2；相对比较报告 stage 1/2 |
| Memory | 约 5% growing memory |
| Student | 原始两层 GCN + 可训练文本投影 + 双向 InfoNCE |
| Student seeds | `150..155` |
| 文本 | top-5 来源、320 字符上限、冻结 MiniLM 重编码 |
| 拓扑 | 完全复用 C0 `edge_index` |
| 访问约束 | 不回读历史完整 raw graph；旧 prefix 不修改 |
| 选择约束 | 不用 report/test F1 选择 arm |

## 3. 候选定义

### 3.1 P-guided two-prototype identity

在当前增量块中，使用 R11 的 `anchor025` assignment 作为 P 权重，对 C0 的局部 slot 做一次守恒 split/merge：

- split：按照 P 权重把原 C0 slot 的来源划分为两个子簇；
- merge：将两个原 slot 的 P 权重合并；
- X：对每个 proposal 做 P 加权平均；
- text：按 P 权重选择来源文本，再执行 top-5/320 字符重编码；
- rows、mass、来源覆盖和 C0 topology 保持不变；
- identity 版本不做额外槽位置换。

### 3.2 P-guided two-prototype + semantic recomputed

在同一 P 生成 payload 上，根据该 payload 重新计算图文 joint cost，再在同类 C0 slots 内做 Hungarian 完备双射。这样测试的是“P 内容 + semantic placement”的组合，而不是把旧 proposal 的 mapping 直接套到新内容上。

### 3.3 对照组

- `c0`：固定共同控制；
- `r11_content_c0_topology`：R11 内容、C0 topology；
- `semantic_perm`：已有 semantic placement、非 P 新内容；
- `random_perm`：随机 payload placement；
- `p_content_random_perm`：P 内容加随机 placement。

## 4. 独立确认结果

相对同一 dataset/order/stage 的 C0，指标为 cumulative Macro-F1 差值。`Pos` 表示 12 个 stage-cell 中为正的数量。

| 方法 | 平均差值 | 正 cell | 最差 cell | 新类 F1 平均差值 | 旧类 F1 平均差值 |
|---|---:|---:|---:|---:|---:|
| `semantic_perm` | **+1.59 pp** | 9/12 | -3.43 pp | -0.41 pp | +3.05 pp |
| `p_content_semantic_recomputed` | +0.71 pp | 8/12 | -5.24 pp | +0.96 pp | +1.28 pp |
| `p_content_two_proto_identity` | +0.35 pp | 7/12 | -6.13 pp | -1.01 pp | +1.80 pp |
| `r11_content_c0_topology` | -0.25 pp | 7/12 | -6.74 pp | -1.50 pp | +1.83 pp |
| `random_perm` | -0.62 pp | 6/12 | -6.28 pp | -3.51 pp | +2.84 pp |
| `p_content_random_perm` | -1.17 pp | 6/12 | -7.60 pp | -3.50 pp | +1.55 pp |

### 4.1 P-guided identity 的逐 cell 结果

| 数据集 | order | stage 1 | stage 2 |
|---|---|---:|---:|
| Cora | `order19605` | +4.62 | -0.74 |
| Cora | `order19606` | -6.13 | -0.90 |
| Cora | `order19608` | +1.99 | +4.94 |
| Citeseer | `order19605` | +0.86 | -4.37 |
| Citeseer | `order19606` | -1.95 | +1.61 |
| Citeseer | `order19608` | +3.98 | +0.36 |

数据集均值：Cora `+0.63 pp`，Citeseer `+0.08 pp`。阶段均值：stage 1 `+0.56 pp`，stage 2 `+0.15 pp`。

### 4.2 P 内容加 semantic placement 的逐 cell 结果

| 数据集 | order | stage 1 | stage 2 |
|---|---|---:|---:|
| Cora | `order19605` | -1.30 | -1.25 |
| Cora | `order19606` | +1.26 | +2.52 |
| Cora | `order19608` | +2.40 | +5.50 |
| Citeseer | `order19605` | -5.24 | -4.33 |
| Citeseer | `order19606` | +4.48 | +1.31 |
| Citeseer | `order19608` | +0.59 | +2.63 |

数据集均值：Cora `+1.52 pp`，Citeseer `-0.09 pp`。这说明 semantic placement 能部分抵消 P 内容风险，但不能消除顺序和数据集依赖。

## 5. 审计与成本

本轮 6 个独立 order-cell 全部完成，所有组合 arm 均写出 `results.json`。12 个 stage audit 均满足：

- `payload_multiset_preserved=true`；
- `source_coverage_preserved=true`；
- `same_c0_topology=true`；
- `same_old_prefix=true`；
- `mapping_bijective=true`；
- `historical_raw_reads=0`；
- `report_labels_used=false`、`selection_by_report=false`。

所有 arm 使用同一 Student steps 和 seeds。报告训练的平均 wall time 约为 17 秒/cell，差异主要来自不同 stage rows 对应的训练步数；P 内容构造没有通过额外 selector 放大 Student 训练成本。远端日志中没有失败 cell。

## 6. 结果解释

### 6.1 开发集正收益没有被 P 内容稳定复现

开发集上的 `P two-proto identity` 平均约 `+1.33 pp`，但独立顺序下降到 `+0.35 pp`，且最差 cell 为 `-6.13 pp`。因此开发集收益包含明显的顺序条件，不能作为稳定泛化证据。

### 6.2 P 内容生成不是当前主要正信号

在本轮共同 runner 中：

- P identity 只有小幅正平均，且新类平均下降；
- P + semantic recomputed 的平均略高，但 Citeseer 仍接近零；
- 非 P 的 `semantic_perm` 平均最高，为 `+1.59 pp`。

因此当前更可信的机制解释是：**合成 payload 被放入合适的传播槽位可能有价值，但 P 重新生成 X/text 本身还没有稳定改善 Student 的新类塑性。**

### 6.3 old/new trade-off 仍然存在

多数候选的旧类 F1 可以维持或上升，但新类 F1 下降。P identity 的 old-class 平均为 `+1.80 pp`，new-class 平均为 `-1.01 pp`，说明它倾向于提高历史保持而牺牲新类可分性，正好对应当前的主要瓶颈。

## 7. 晋升判定

预先设定的晋升条件包括：两个数据集正收益、总平均至少约 `+0.5 pp`、最差 cell 不低于约 `-2 pp`、新类塑性不系统下降、至少 8/12 cell 为正。

本轮判定：

| 候选 | 判定 | 原因 |
|---|---|---|
| P two-proto identity | **不晋升** | 平均仅 +0.35；7/12 正；最差 -6.13；new F1 -1.01 |
| P + semantic recomputed | **不晋升** | 平均 +0.71，但最差 -5.24，Citeseer -0.09 |
| semantic permutation | **保留为机制候选** | 平均 +1.59、9/12 正，但仍有 -3.43 最差 cell，new F1 略降 |

因此当前默认主方法仍为 C0。P two-prototype 可以作为已完成的负结果和机制诊断，不应写成稳定主算法。

## 8. 后续建议

1. 暂停继续扫描 P 权重、split 比例或 semantic cost；当前问题不是缺少一个局部超参数。
2. 将 `semantic_perm` 单独作为“槽位传播匹配”候选，先做任务无关的结构验证和成本分析，不把 P 内容生成强行并入主线。
3. 若继续追求新类塑性，应直接研究传播槽位与新类 message compatibility 的约束，而不是继续优化 teacher proxy 或短训 selector。
4. 论文中可以报告 P 内容生成的完整独立负结果：它提高了部分旧类保持，但没有稳定提高新类塑性，说明内容重构和 Student 传播角色之间存在目标错位。

