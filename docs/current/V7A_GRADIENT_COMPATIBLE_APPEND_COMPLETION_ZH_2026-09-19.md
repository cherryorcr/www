# v7-A Gradient-Compatible Append 小图结项报告

日期：2026-09-19  
状态：**实验与独立审计完成；未通过晋升门禁；保留 C0，不启动 v7-B、动态预算或大图扩展。**

## 1. 结论先行

本轮检验的假设是：在相同 student 初始化和 batch 前缀下，候选 append memory 与 C0 append memory 的 0/1/2-step 完整参数梯度距离，结合一个几何覆盖收益，能够以低成本预测最终 student utility。

结论为：

1. **梯度兼容性选择器没有通过。** 8 个 cell 的 selector 全部回退 C0，平均 selected gain 为 `0.00 pp`，平均 Spearman 仅 `0.125`，top-1 agreement 为 `2/8`。
2. **候选构造本身并非全部无效。** 16 个非 C0 候选中有 8 个在独立 report student 上取得正的 paired cumulative Macro-F1 gain；但当前分数无法稳定识别这些候选。
3. **多原型在 Cora stage 1 有强局部信号，但不是已成立的持续方法。** Cora canonical/confirmation 的 multi-prototype 分别比 C0 高 `+15.454 pp` 和 `+12.136 pp`；两个 stage 2 分别低 `-2.513 pp` 和 `-3.460 pp`。这四个结果是共享 C0 历史的单阶段反事实，不能拼接为一条递归持续轨迹。
4. **recluster 是较温和的候选。** 8 个 cell 中有 5 个 recluster report gain 为正，但平均幅度小且区间跨零；它在 Citeseer 两个 stage 2 分别为 `+1.595 pp` 和 `+2.410 pp`，仍不能称为稳定优势。
5. **成本门禁通过，但这不足以抵消效果和排序门禁失败。** 单次候选选择成本为 C0 对照的 `1.110--1.263x`，平均 `1.163x`，低于 `2x` 阈值。

因此当前正式主线仍为：

```text
C0 = 5% growing budget
   + dispersion class allocation
   + class-wise joint clustering
   + Mean-X per cluster
   + joint_g5_c320 concat/re-encode
   + frozen old synthetic rows / old-old edges
   + current text-nearest cross-edge
   + response off
```

本轮没有证明 gradient-compatible append，也没有证明 multi-prototype 或 recluster 可以进入 C0 主方法。

## 2. 本轮回答的问题

此前的 v3--v6 实验说明：完整 response curve 成本高，静态拓扑保持不能直接推出 scratch student 的保持性，局部 merge/split 也没有稳定收益。因此 v7-A 尝试把选择对象改成训练动力学：

> 候选 memory 是否会让共同 student 的早期梯度轨迹偏离 C0？这种偏离是否能够预测最终从头训练 student 的好坏？

本轮只检验这个选择问题，不把 report oracle 候选拼成新的持续方法。

## 3. 冻结协议

### 3.1 候选

每个阶段固定三个候选：

| 候选 | 做法 | 解释边界 |
|---|---|---|
| `c0` | 当前 C0 | 主控制 |
| `recluster` | donor/recipient 一行预算转移后，仅对受影响类别完整重聚类 | 同时改变 X、文本和新边，属于组合候选 |
| `multi_prototype` | 类内 joint subcluster，按 mass/dispersion 给子簇分配同类 rows，再用源簇均值生成合成节点 | 不是纯 X 算子消融；会连带改变文本和新边 |

旧 synthetic rows、旧旧边和 C0 历史保持一致。最终 memory 仍是合成节点；source ID 只用于离线审计。singleton 簇单独统计，不能声称每一行数值都不同于真实源节点。

### 3.2 选择分数

对每个候选，在相同初始化、相同 batch 前缀下计算完整 student 参数梯度：

\[
D_{dyn}(M)=\frac{1}{3}\sum_{k=0}^{2}
\left(1-\cos(g_k(M),g_k(M_{C0}))\right).
\]

几何覆盖项为：

\[
U_{cov}(M)=\frac{e(C0)-e(M)}{|e(C0)|+10^{-8}},
\]

最终选择分数：

\[
Q(M)=U_{cov}(M)-0.5D_{dyn}(M).
\]

两组 selection seeds 为 `(7300,7301)` 和 `(7400,7401)`。只有两组同时选择同一个非 C0 候选且平均 Q 增益大于 0，才接受候选，否则回退 C0。

### 3.3 student 与评估

- Cora、Citeseer；canonical 顺序加一个新的 confirmation 顺序。
- 每个 cell 使用 report seeds `9300..9305`。
- 两层 GCN，hidden/output `384`。
- 冻结 MiniLM 文本编码器。
- Adam，学习率 `1e-3`，batch size `16`。
- 从头训练，双向图文 InfoNCE。
- 训练步数 `max(50, min(2048, 16 * rows))`。
- 主指标：`report/all/cumulative_f1`，目标节点为已见类，预测竞争范围为全部类别。
- test 标签没有用于优化、选择或评价；source audit 中 test target 替换为 `-1`。

stage 1/2 是共享 C0 历史的单阶段反事实 cell，不是递归地把 stage 1 选择结果传入 stage 2。

## 4. 八个 cell 的完整结果

下表为六个独立 report seeds 的平均 cumulative Macro-F1。`Q(R/M)` 是 selection 阶段对 recluster/multi-prototype 的平均 Q，不是 report F1；`oracle gain` 是事后仅用于诊断的最好候选相对 C0 的 report gain，不能作为主方法结果。

| 数据集 | 顺序 | stage | rows | C0 | recluster | multi-prototype | report oracle | oracle gain | Spearman | Q(R/M) | 成本/C0 |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|
| Cora | canonical | 1 | 67 | 52.182 | 49.323 | **67.636** | multi | +15.454 | -0.5 | -0.039/-0.168 | 1.263x |
| Cora | canonical | 2 | 94 | 49.305 | **51.108** | 46.791 | recluster | +1.803 | 0.5 | -0.043/-0.064 | 1.118x |
| Cora | confirmation | 1 | 70 | 40.201 | 42.122 | **52.336** | multi | +12.136 | -1.0 | -0.031/-0.150 | 1.197x |
| Cora | confirmation | 2 | 94 | **54.529** | 53.437 | 51.069 | C0 | 0.000 | 1.0 | -0.014/-0.083 | 1.115x |
| Citeseer | canonical | 1 | 86 | **69.747** | 68.792 | 64.870 | C0 | 0.000 | 1.0 | -0.021/-0.147 | 1.205x |
| Citeseer | canonical | 2 | 111 | 57.970 | **59.565** | 56.442 | recluster | +1.595 | 0.5 | -0.025/-0.076 | 1.110x |
| Citeseer | confirmation | 1 | 83 | 59.868 | 60.205 | **61.058** | multi | +1.191 | -1.0 | -0.017/-0.159 | 1.185x |
| Citeseer | confirmation | 2 | 111 | 55.294 | **57.704** | 54.978 | recluster | +2.410 | 0.5 | -0.021/-0.076 | 1.115x |

### 4.1 old/new F1 的局部证据

最强的局部正信号来自 Cora stage 1：

| cell | 候选 | cumulative gain | old F1 | new F1 | 解释 |
|---|---|---:|---:|---:|---|
| Cora/canonical/s1 | multi-prototype | +15.454 | 67.755 | 67.517 | old/new 同时较高，说明不是单纯牺牲旧类换新类 |
| Cora/confirmation/s1 | multi-prototype | +12.136 | 40.015 | 64.657 | 新类塑性提升明显，但 seed 波动较大 |
| Cora/canonical/s2 | recluster | +1.803 | 60.799 | 38.186 | 小幅正收益，配对区间跨零 |
| Citeseer/confirmation/s2 | recluster | +2.410 | 62.090 | 48.930 | 小幅正收益，仍需独立顺序/seed 验证 |

这些 old/new 数字均为同一 cell、同一 candidate 的六个 report seed 均值；它们用于解释保持/塑性，不代表跨顺序的稳定结论。

## 5. 晋升门禁

| 门禁 | 阈值 | 实际结果 | 判定 |
|---|---:|---:|---|
| gradient rank 与 report rank | 平均 Spearman ≥ 0.30 | 0.125 | 失败 |
| selector 平均 gain | ≥ +0.50 pp | 0.000 pp | 失败 |
| 新顺序阶段平均 gain | Cora/Citeseer 均 ≥ 0 | 0.000 / 0.000 | 通过，但不足以晋升 |
| retention | cumulative ≥ -2 pp 且 old ≥ -3 pp | 通过 | 通过 |
| 单次成本 | ≤ 2x C0 | 1.110--1.263x | 通过 |
| 工件与隔离审计 | 全部通过 | 通过 | 通过 |

最终决定：

```json
{
  "continue_v7b": false,
  "accepted_cells": 0,
  "selected_mean_gain": 0.0,
  "mean_spearman": 0.125,
  "mean_regret": 4.3235283924,
  "geometry_mean_gain": -0.2084992655
}
```

`retention` 和 `new_order_stability` 通过主要是因为 selector 在所有 cell 回退到 C0，不能解释为新增机制已经证明保持性或顺序鲁棒性。

## 6. 为什么候选有收益，selector 却全部回退

### 6.1 当前 Q 存在明显 objective blind spot

非 C0 候选共有 16 个，其中 8 个 report gain 为正；这 8 个正收益候选中有 6 个 `coverage_gain <= 0`。由于 C0 的 `Q=0`、`D_dyn>=0`，只要 `U_cov<=0`，任何非负的动态惩罚系数都不可能使候选超过 C0。

这说明当前问题不只是 `lambda_dyn=0.5` 太大，而是 coverage proxy 本身没有覆盖最终 student utility。以 Cora canonical stage 1 为例，multi-prototype 的 report gain 为 `+15.454 pp`，但 geometry error 从 `1.1461` 增加到 `1.1560`，因此 coverage 项为负，最终 Q 为 `-0.1676`。

### 6.2 梯度距离只测“变化”，没有测“改善方向”

`D_dyn` 衡量候选梯度相对于 C0 的偏离程度，但不判断这个偏离是否把 student 推向更好的新类决策边界。对多原型候选而言，较大的梯度变化可能正是补充类内模式的原因；把它全部作为惩罚会误杀有用的塑性。

### 6.3 0/1/2-step 与完整训练 terminal utility 不一致

selector 只观察训练极早期的共同前缀；report student 使用完整步数。候选的优势可能在早期没有表现出来，或者需要经过多个阶段的参数适应后才出现。因此这轮结果不支持“两个 step 的梯度距离可以预测最终 F1”的强结论。

### 6.4 candidate construction 与单一模态归因不成立

recluster 和 multi-prototype 同时改变了合成 X、新文本和新边。它们的正负结果只能归因于“组合候选 memory”，不能据此声称多原型 X、文本或拓扑中的某一项单独有效。

## 7. 成本、审计与数值修订

### 7.1 训练和选择成本

- 正式 fits：`144 = 8 cells × 3 candidates × 6 report seeds`。
- 完整训练观察时点：`576`。
- selector/report split metric observations：`1152`。
- 正式 student optimizer steps：`206,208`。
- selection gradient runs：`96`。
- 每次 gradient run 两次 Adam 更新，合计 `192` 次更新和 `288` 个梯度观测。
- report student 训练时间总和：约 `475.433 s`；评估时间总和：约 `7.032 s`。
- 选择成本相对 C0：最小 `1.1095x`，平均 `1.1633x`，最大 `1.2625x`。

成本比值只包括当前 block、候选构造/追加、gradient/score 和一个最终 student；排除共享预处理、历史构造、实验枚举、审计和工件 I/O，不能写成全生命周期加速。

### 7.2 工件审计

- audit status：`passed`。
- 最终模型 reload：`144` 次。
- gradient reload：`96` 组。
- 最大最终 logits reload 误差：`3.814697265625e-6`。
- 最大 gradient reload 误差：`0.0`。
- 审计耗时：约 `12.083 s`。
- 15 个冻结依赖文件的本地/远程 freeze hash 已一致。

### 7.3 浮点数值修订

首次独立审计使用 float64 重算 Q，发现与冻结 selector 的 float32 cosine 计算存在最大 `6.231879980815418e-6` 的差异。随后只修正审计脚本，使其使用与冻结 selector 相同的 float32 公式并将阈值设为 `1e-7`：

- 原始训练、候选、选择、checkpoint 没有改变；
- float32 公式重算误差为 `0`；
- float64 与 float32 的 cell selection 变化为 `0/8`；
- 修正版 reload 审计通过。

这是审计数值口径修订，不是实验结果修订。原始失败日志仍保留，便于追溯。

## 8. 当前研究判断

本轮应保留的结论是：

> 便宜的训练动力学探针可以识别候选 memory 的变化，但当前“几何覆盖收益 - 梯度偏离惩罚”分数不能可靠识别最终 student utility。候选空间中存在局部正收益，然而正收益与当前几何 proxy 并不一致。

不应写成：

- gradient-compatible append 已经优于 C0；
- multi-prototype 已经是新的主方法；
- Cora stage 1 的 oracle 结果代表稳定持续收益；
- v7-A 证明了 selector 的泛化能力；
- 由于成本只有约 1.16x，就已经实现了训练加速。

## 9. 后续建议

### 9.1 立即决策

1. C0 继续作为冻结比较版本。
2. 不启动 v7-B 多原型自动选择。
3. 不启动 response-gain extra budget、动态 rate 或大图 v7 扩展。
4. 不根据本轮 report oracle 反调 `lambda`、coverage 权重或候选定义后重新宣布通过。

### 9.2 如果继续探索，应该怎样改

下一轮应先把问题拆开，而不是继续堆候选：

1. **固定选择器，单独验证候选构造。** 在预注册的新顺序和新 report seed 上比较 C0/recluster/multi-prototype，明确 multi-prototype 是否具有可重复的 stage-1 信号。
2. **重新设计 utility proxy。** 不能只用 joint geometry error；至少加入类内多模态覆盖、决策边界/文本属性覆盖和 computation-tree coverage，并把“有益的梯度变化”与“有害的漂移”区分开。
3. **增加一个较晚的 fidelity 点。** 不是恢复完整 response curve，而是在有限候选上比较早期 gradient、短 terminal loss/curve 和完整 report 的排序是否一致。
4. **严格分离 X、text、topology。** multi-prototype 需要一个只改 synthetic X、固定文本和拓扑的受控版本；否则无法形成方法贡献。
5. **若仍失败，转为机制/benchmark 论文路线。** 主贡献应是严格 continual TAG protocol、append-only synthetic memory、完整成本/顺序/保持性审计和系统负结果，而不是强行包装一个未通过的 selector。

## 10. 工件与复现

主要文件：

- 冻结协议：[V7A_GRADIENT_COMPATIBLE_APPEND_FROZEN_PROTOCOL_ZH_2026-09-19.md](V7A_GRADIENT_COMPATIBLE_APPEND_FROZEN_PROTOCOL_ZH_2026-09-19.md)
- 审计汇总：`results/v7a_gradient_append_20260919/audit_summary_reload.json`
- 精度诊断：`results/v7a_gradient_append_20260919/precision_diagnostic.json`
- cell 表：`results/v7a_gradient_append_20260919/cell_metrics.csv`
- candidate 表：`results/v7a_gradient_append_20260919/candidate_metrics.csv`
- 元数据交叉核对：`results/v7a_gradient_append_20260919/metadata_summary.json`
- 完整冻结代码快照：`results/v7a_gradient_append_20260919/code_snapshot.tar.gz`
- 完整 metadata 快照：`results/v7a_gradient_append_20260919/metadata_bundle.tar.gz`

大型 `.pt` checkpoint 没有下载到本地，完整 checkpoint 保留在远程结果目录；本地 metadata、JSON、日志、协议和代码快照足以复核表格、选择和审计摘要。

本地 metadata 交叉核对命令：

```powershell
D:\tools\python\python.exe code/v2_llm_fusion/summarize_v7_gradient_append.py `
  --root results/v7a_gradient_append_20260919
```
