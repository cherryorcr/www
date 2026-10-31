# 字段级 X 对照与固定 Outer 轨迹诊断

日期：2026-09-08

本报告记录两项新增的 development-only 诊断：

1. 在相同 representative IDs、文本 embedding、质量权重和 post-decoder topology 下，只替换新增 compact block 的 X，比较 centroid X 与真实代表 X；
2. 固定 stage-2 的 joint ASM outer snapshot `0/20/40/60/80/100`，每个 snapshot 用 student seeds `100/101` 独立重训，同时报告 validation-selected checkpoint 和固定 endpoint。

所有结果都没有读取 confirmation/test 分区，`confirmation_accessed=false`。它们用于归因，不能当作新的最终测试结果。

## 1. 字段级 X 对照

实验输入为 stage-1 的 Append memory 加 stage-2 delta，累计预算为 94 行，其中旧 memory 67 行、新 block 27 行。两种 artifact 的以下字段逐字节相同：

- representative `origin_ids`；
- readable text 和 BERT CLS `text_emb`；
- provenance weights；
- post-decoder `edge_index`。

只有新增 block 的 `x` 不同：

- `centroid_x_frozen`：聚类成员的特征 centroid；
- `real_x_frozen`：同一 representative ID 对应的真实 source feature。

### 结果

| condensation seed | centroid X | real representative X | real - centroid |
|---:|---:|---:|---:|
| 44 | 44.75 | 44.18 | -0.57 pp |
| 45 | 34.59 | 36.49 | +1.89 pp |

数值为两个独立 student 的累计 checkpoint Macro-F1 均值。每个 seed 内两种 artifact 使用相同 student seeds `100/101`。

### 解释

这个对照没有支持“新增 block 的 centroid X 是持续损失的稳定主因”。真实 representative X 在两个 condensation seed 上方向相反，平均差异也小于 seed 间波动。当前更谨慎的结论是：

1. X 聚合可能影响某些 seed，但不是已经被证明的单一瓶颈；
2. 代表文本/拓扑的组合、ASM 后期漂移和 student 选择噪声仍需保留为候选原因；
3. 后续方法不能只把 centroid 换成真实 representative X 就宣称解决持续凝聚。

## 2. 固定 Outer 轨迹

每个阶段 2 snapshot 都从同一 stage-1 memory 和同一 stage-2 初始化开始。outer selector 被禁用，因此结果没有被 validation student 选点压缩成一个数字。每一行给出：

- `selected`：每个 student 在其训练曲线中按 checkpoint partition 选择的最好点；
- `endpoint`：固定 student training endpoint；
- `old`：stage 0 和 stage 1 的平均 Macro-F1；
- `current`：stage 2 Macro-F1；
- `X mean/L2`：synthetic X 的漂移指标。

### condensation seed 44

| outer | selected | endpoint | old | current | X mean | X L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 44.27 | 42.18 | 76.47 | 33.53 | 1.626 | 16.064 |
| 20 | 44.31 | 42.35 | 75.48 | 28.46 | 1.915 | 18.799 |
| 40 | 42.51 | 39.21 | 72.59 | 30.54 | 2.116 | 20.768 |
| 60 | 43.17 | 38.50 | 73.00 | 32.41 | 2.261 | 22.205 |
| 80 | 42.33 | 42.14 | 74.45 | 25.14 | 2.373 | 23.337 |
| 100 | 41.62 | 39.27 | 70.94 | 31.62 | 2.458 | 24.218 |

逐阶段 selected Macro-F1（stage 0 / stage 1 / stage 2）为：

| outer | per-stage Macro-F1 |
|---:|---:|
| 0 | 82.96 / 69.99 / 33.53 |
| 20 | 82.43 / 68.53 / 28.46 |
| 40 | 82.41 / 62.78 / 30.54 |
| 60 | 86.47 / 59.54 / 32.41 |
| 80 | 87.65 / 61.25 / 25.14 |
| 100 | 86.86 / 55.03 / 31.62 |

seed 44 中，stage 1 的保持从 69.99 降到 55.03，和 X 漂移同步；stage 0 没有下降，说明“旧历史”不是一个整体同质变量。

### condensation seed 45

| outer | selected | endpoint | old | current | X mean | X L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34.59 | 26.28 | 59.50 | 35.86 | 1.667 | 16.396 |
| 20 | 39.80 | 32.58 | 67.19 | 31.28 | 1.959 | 19.182 |
| 40 | 40.02 | 31.35 | 65.44 | 32.53 | 2.175 | 21.307 |
| 60 | 37.49 | 29.78 | 60.81 | 35.51 | 2.345 | 22.970 |
| 80 | 39.34 | 32.56 | 63.46 | 36.77 | 2.471 | 24.221 |
| 100 | 39.99 | 33.46 | 64.34 | 36.59 | 2.565 | 25.169 |

逐阶段 selected Macro-F1（stage 0 / stage 1 / stage 2）为：

| outer | per-stage Macro-F1 |
|---:|---:|
| 0 | 76.05 / 42.96 / 35.86 |
| 20 | 71.00 / 63.38 / 31.28 |
| 40 | 69.14 / 61.74 / 32.53 |
| 60 | 62.09 / 59.53 / 35.51 |
| 80 | 65.49 / 61.42 / 36.77 |
| 100 | 68.80 / 59.88 / 36.59 |

seed 45 中，stage 0 会下降，但 stage 1 从极低的初始响应上升；因此 outer step 与保持/塑性的关系不是单调函数。仅凭 X norm 增长不能区分“旧 memory 破坏”和“新阶段学习不足”。

## 3. 后续完成的文本配对与 teacher alignment 归因

### 文本置换对照

固定 synthetic X、topology、weights 和 origin IDs，只在阶段内置换 synthetic text。实验覆盖 2 个 condensation seed、每个 seed 3 个置换种子，并用 student seeds `100/101` 重训。

六次置换的当前阶段 test Macro-F1 全部低于原配对，平均下降约 `11.20 pp`。因此文本对应不是可忽略的附属字段；即使下游使用 GCN，graph embedding 仍通过图文 InfoNCE 和七类文本 prompt 参与训练与分类。

### 无标签 teacher alignment

使用固定 joint teacher，在不读取标签的条件下评估每个 outer snapshot 的 graph-text symmetric InfoNCE。原 ASM 的全体 InfoNCE 为：

| condensation seed | outer 0 | outer 100 | 变化 |
|---:|---:|---:|---:|
| 44 | 1.54 | 2.27 | +0.73 |
| 45 | 1.06 | 2.07 | +1.01 |

随着 X 被更新，positive cosine、双向 Recall@1 和 margin 也整体恶化。固定字段哈希证明文本、weights、origin IDs 和 stage 没有变化；不同运行中原始 `edge_index` SHA 偶有不同，但边排序后的集合完全相同。因此差异来自 X 优化造成的图文 response 漂移，不是换了不同 topology。

## 4. Text-Anchored ASM 与权重 sweep

新增目标：

\[
L=L_{ASM}+\lambda L_{text-anchor}.
\]

`L_text-anchor` 使用冻结 joint teacher，对 synthetic graph response 与其固定文本做 symmetric InfoNCE。梯度只进入 X；文本、topology、weights 和 origin IDs 固定。实现支持 `scope=all/optimized`，并分别记录 ASM、anchor 和 total loss；`lambda=0` 保留旧数值路径。

开发 sweep 覆盖：

- condensation seeds `44/45`；
- `lambda = 0/0.001/0.003/0.005/0.01/0.05/0.1`；
- outer `0/20/40/60/80/100`；
- student seeds `100/101`。

唯一同时满足“两 seed 累计和当前 dev 都上升、历史跨 seed 均值不降、两 seed InfoNCE 都下降”的组合为 `lambda=0.01, scope=all, outer=60`。

### 同 outer=60 的确认

扩展到 student seeds `100–107` 后，共 16 个配对：

| 指标 | 平均变化 | 正向配对 |
|---|---:|---:|
| 累计 dev F1 | +2.89 pp | 9/16 |
| 累计 test F1 | +4.34 pp | 14/16 |
| 当前阶段 test F1 | +4.85 pp | 13/16 |
| 历史阶段 test F1 | +5.07 pp | 12/16 |

候选累计 test 为 `39.72%`，同 outer 的零权重基线为 `35.38%`。候选的全体无标签 InfoNCE 在 seeds 44/45 上分别降低 `2.023/1.831`，当前阶段 InfoNCE 分别降低 `1.474/1.401`。

### 各自 dev 最优的公平比较

零权重 ASM 自身 dev 最优点为 outer=20。候选 outer=60 相对该基线：

| 指标 | 候选 − 基线 |
|---|---:|
| 累计 train F1 | +0.01 pp |
| 累计 dev F1 | +0.44 pp |
| 累计 test F1 | +1.47 pp |
| 当前阶段 dev F1 | +3.03 pp |
| 当前阶段 test F1 | +3.47 pp |
| 历史阶段 dev F1 | −1.30 pp |
| 历史阶段 test F1 | +0.38 pp |

这说明锚定损失的机制有效，但公平选择后总体收益较小，主要改善当前阶段塑性。

## 5. 固定 label space 与严格递归复核

评测器已经显式固定 label space：单阶段使用该阶段类别，累计 seen 使用已经到达类别，future 使用未来类别。修复后所有历史矩阵数值逐项不变，因为当前 Cora 分区恰好包含各自完整类别；潜在口径错误已经消除。

随后执行真正的 stage 1→stage 2 递归：stage 1 只读取共同 stage-0 synthetic TAG 和当前 delta；用 student seeds `100/101` 从固定 outer 快照中选择 artifact；stage 2 只读取该 artifact 和 stage-2 delta；最后用独立 student seeds `102–107` 确认。

### stage 1

两个 condensation seed 下，零权重和 `lambda=0.01` 都选择 outer=0。两种方法的选中 artifact 除 metadata 和原始 COO 边顺序外完全相同，排序后 edge set 也相同。12 个独立 student 的共同结果为：累计 dev `53.99`、累计 test `55.27`、stage-0 test `78.15`、stage-1 test `44.63`、future stage-2 test `26.31`。

因此文本锚定在 stage 1 没有产生可选择的更新；它只能减缓部分 outer 的下降。

### stage 2

开发选择 student 上，候选在两个 condensation seed 的累计 dev 分别低 `0.30 pp` 和 `0.51 pp`，平均约 `−0.40 pp`。12 个独立确认 student 的均值为：

| 指标 | Text-Anchored | ASM λ=0 | 差值 | 正向配对 |
|---|---:|---:|---:|---:|
| 累计 dev | 40.78 | 38.86 | +1.91 pp | 7/12 |
| 累计 test | 39.10 | 37.49 | +1.62 pp | 7/12 |
| 历史阶段 test | 63.99 | 60.56 | +3.43 pp | 9/12 |
| 当前阶段 test | 32.25 | 31.13 | +1.12 pp | 8/12 |

但累计 test 差值在 condensation seed 44 为 `−0.84 pp`，seed 45 为 `+4.08 pp`。因此严格递归结果没有达到跨 seed 稳定通过条件。

## 6. 更新后的归因结论

1. **已确认：centroid X 不是跨 seed 稳定主因。**
2. **已确认：ASM 的 X 更新会破坏固定 graph-text correspondence；文本置换和无标签 teacher alignment 提供相互独立的证据。**
3. **已确认：问题同时涉及旧阶段保持和新阶段塑性，而且不同历史阶段方向可以相反。**
4. **尚未确认：跨阶段边缺失是主因。当前无需先引入跨阶段边才能解释已观察损失。**
5. **方法结论：Text-Anchored ASM 是有效的机制修复，但没有解决 stage-1 更新伤害，严格递归收益也不跨 condensation seed 稳定。**

下一步目标应改为 graph-text correspondence 与历史 teacher response 的联合保持。最小 gate 是让两个 condensation seed 的 stage-1 dev 都选择 `outer>0` 且不低于 outer=0；未通过前不扩展 GraphCLIP、transport、跨阶段边或新 prompt。

## 7. 生成文件与代码变更

- 轨迹 runner：[run_outer_trajectory.py](../../code/native_tag_v1/run_outer_trajectory.py)
- 字段诊断 runner：[run_fine_attribution.py](../../code/native_tag_v1/run_fine_attribution.py)
- c44 轨迹结果：`results_mirror/trajectory_joint_c44/trajectory.json`
- c45 轨迹结果：`results_mirror/trajectory_joint_c45/trajectory.json`
- c44 X 对照：`results_mirror/fine_x_c44/fine_attribution_summary.json`
- c45 X 对照：`results_mirror/fine_x_c45/fine_attribution_summary.json`
- 文本置换矩阵：`results_mirror/protocol_matrices_fixed_labels_2026_09_08/text_pairing.json`
- 文本锚定 sweep：`results_mirror/text_anchor_sweep_2026_09_08/aggregate.json`
- 同 outer 确认矩阵：`results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_confirmation_matrix.json`
- 各自 dev 最优矩阵：`results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_vs_devbest_matrix.json`
- 严格递归矩阵：`results_mirror/recursive_text_anchor_2026_09_08/recursive_protocol_matrix.json`
- 严格递归汇总：`results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json`
- dev-only 选择器：[select_trajectory_artifact.py](../../code/native_tag_v1/select_trajectory_artifact.py)
- 独立 artifact student runner：[train_saved_artifact_students.py](../../code/native_tag_v1/train_saved_artifact_students.py)
- 严格递归汇总器：[summarize_recursive_text_anchor.py](../../code/native_tag_v1/summarize_recursive_text_anchor.py)

最新全目录测试：`38/38 passed`，包括新增递归 artifact 选择与 provenance 恢复测试。
