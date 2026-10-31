# 持续 TAG 凝聚：文本对应归因、Text-Anchored ASM 与严格递归验证

日期：2026-09-08  
状态：开发阶段技术报告；Cora test/confirmation/future 已被访问，本文所有 test 数值只能作为开发证据

## 技术结论

当前实验已经确认一个属于文本属性图而非 plain graph 的独立困难：**ASM 只优化 synthetic graph feature `X` 时，会逐步破坏 synthetic graph response 与冻结文本之间的一一对应**。六次文本置换都降低当前阶段 test Macro-F1，平均约 `11.20 pp`；原始 ASM 从 outer 0 到 100 还使两次 condensation run 的全体图文 InfoNCE 分别由 `1.54` 升到 `2.27`、由 `1.06` 升到 `2.07`。因此，“文本只是初始化材料、凝聚时可以不再约束”不符合当前 TAG 证据。

基于该归因实现的 Text-Anchored ASM，在固定 stage-1 memory、只比较 stage-2 更新时有效：同一 outer=60 下，16 个独立 student 配对的累计 test 平均提高 `4.34 pp`；与零权重 ASM 各自的 dev 最优点比较，累计 test 仍提高 `1.47 pp`。但严格执行 stage 1→stage 2 递归以后，证据变弱：stage 1 的候选和基线都只能选择 outer=0；stage 2 的开发选择 student 上，候选平均低 `0.40 pp`；另 12 个确认 student 上候选累计 test 平均高 `1.62 pp`，但只在 `7/12` 个配对中为正，并且 condensation seed 44 为 `−0.84 pp`、seed 45 为 `+4.08 pp`。

所以当前可以保留的论文级结论是“文本对应漂移是持续 TAG 凝聚的可测瓶颈”，Text-Anchored ASM 是有机制依据的第一版修复。当前还不能把它定为最终算法，因为它没有跨 condensation seed 稳定地改善严格递归更新，也没有解决 stage 1 中“任何 ASM 更新都不如初始化”的问题。

## 研究问题与固定协议

研究对象是类别增量的持续 TAG 凝聚：

- stage 0 到达类别 `[0,1]`；
- stage 1 新增类别 `[2,3]`；
- stage 2 新增类别 `[4,5,6]`；
- 一阶段完成后，历史原图不可再次读取，只能保留满足累计预算的小型 synthetic TAG；
- 每一阶段的凝聚输入只能是上一阶段 synthetic TAG 与当前 delta；
- 类别名称和全局七类文本提示可见，训练和凝聚 artifact 不含节点标签；
- 下游固定为 TAGSAM 风格 student：冻结 BERT CLS、两层 GCN、可训练文本投影、图文 InfoNCE 和全局文本提示分类；
- GraphCLIP 暂不加入当前可行性验证。

主指标为固定 label space 的 Macro-F1。每个已到达阶段分别报告 train、dev/checkpoint、test；再报告累计 seen、旧阶段均值、当前阶段和未来阶段 test。训练图使用 stage-train induced graph；dev/test 使用该阶段 whole-stage transductive graph，并只在对应 mask 上读取标签。future 图只在模型和 artifact 冻结后加载。

本文使用两类比较，作用不同：

1. **同 outer 比较**控制优化步数，用于判断新增损失项本身是否改变结果；
2. **各自 dev 最优比较**允许每种方法独立选择 outer，用于判断完整方法在公平模型选择后的表现。

## 静态 TAG 凝聚仍存在上游误差

官方 TAGSAM 静态复核没有恢复 full teacher 的性能。七类静态结果为：

| 静态方法 | Macro-F1 |
|---|---:|
| minimal repair | 36.24 / 38.74 |
| add-adj | 40.23 |
| full teacher | 61.12 |

在当前 native stage-0 协议下，静态基线进一步显示，小图本身有用，但 ASM 并不稳定优于冻结初始化：

| 方法 | Train | Dev | Test | Future |
|---|---:|---:|---:|---:|
| Full-real | 80.73 | 84.90 | 83.24 | 21.89 |
| Random-small | 67.64 | 73.73 | 69.73 | 17.11 |
| Coreset | 64.72 | 68.66 | 62.22 | 12.31 |
| Frozen compact | 72.11 | 77.84 | 76.56 | 11.75 |
| Offline ASM | 68.22 | 76.50 | 73.05 | 15.80 |

这说明持续误差中一定混有静态凝聚误差。不过本文没有把修复全部静态误差设为前置条件，因为持续重凝聚还引入了两个静态问题没有覆盖的变量：旧 synthetic memory 代替旧原图，以及新旧 TAG 在同一图文对比空间中反复更新。后续方法必须用“同一静态起点的持续增量”对照，把这两类误差分开。

## 细粒度归因：问题不只是旧图、也不只是新图

### centroid X 不是稳定主因

固定 representative ID、文本、权重和 topology，只把新增 block 的 centroid X 换成真实 representative X 后，两个 condensation seed 的累计 dev 变化分别为 `−0.57 pp` 和 `+1.89 pp`。方向不一致，因此当前不能把“新增图特征聚合错误”作为主要解释。

### ASM 同时改变旧阶段保持与新阶段塑性

固定 outer 轨迹显示：seed 44 中，stage-1 Macro-F1 从 outer 0 的 `69.99` 降到 outer 100 的 `55.03`，而 stage 0 没有同步下降；seed 45 中，stage 1 反而从较差的 `42.96` 上升到 `59.88` 左右，但 stage 0 会下降。旧历史不是一个同质整体，单看累计 old 指标会掩盖阶段间相反变化。

当前阶段也不是单独的稳定受益方。不同 outer 下 current Macro-F1 会升会降，说明问题既包含旧 memory 被改坏，也包含新阶段没有形成可迁移 synthetic response。已有对照没有证明缺少两个小图之间的跨阶段边是主因；在加入跨阶段边以前，应先解释已经观察到的图文对应漂移和阶段级 response 冲突。

### 文本对应是独立、可重复的瓶颈

在 X 和 topology 固定的情况下，只在阶段内置换 synthetic text，共做 3 个置换种子 × 2 个 condensation seed。六次置换的当前阶段 test 均下降，平均约 `11.20 pp`。这个实验直接说明 GCN 下游仍然体现 TAG 特性：GCN 产生 graph embedding，训练和分类都要与对应文本 embedding 对齐；文本与图根的配对一旦错误，节点分类响应会系统下降。

无标签 teacher 对齐诊断也得到一致结果。原 ASM 的全体 symmetric InfoNCE 随 outer 增大而恶化：

| condensation seed | outer 0 | outer 100 | 变化 |
|---:|---:|---:|---:|
| 44 | 1.54 | 2.27 | +0.73 |
| 45 | 1.06 | 2.07 | +1.01 |

这些变化发生在文本、topology、weights 和 origin IDs 固定时，因此来源是 synthetic X 的更新把 graph response 推离了原来的文本锚点。

## 第一版方法：Text-Anchored ASM

方法在原 ASM 目标上加入冻结 teacher 的图文对应保持项：

\[
L = L_{ASM} + \lambda L_{text-anchor}.
\]

`L_text-anchor` 对 synthetic graph response 与其固定 BERT CLS 文本做 symmetric InfoNCE。teacher、文本、topology、weights 和 origin IDs 全部冻结，梯度只进入 synthetic X。当前候选使用：

- `text_anchor_weight = 0.01`；
- `scope = all`；
- stage-2-only 开发筛选点 `outer = 60`。

实现保留 `lambda=0` 的旧路径，并分别记录 ASM loss、anchor loss、total loss 和无标签 alignment。代码位于 [native_core.py](../../code/native_tag_v1/native_core.py) 与 [run_outer_trajectory.py](../../code/native_tag_v1/run_outer_trajectory.py)。

## stage-2-only 结果：机制有效，但收益依赖比较口径

### 相同 outer=60 的因果归因比较

候选和基线使用相同输入、outer、condensation seed 与 student seed。共 `2 × 8 = 16` 个 student 配对：

| 指标 | Text-Anchored | ASM λ=0 | 差值 | 正向配对 |
|---|---:|---:|---:|---:|
| 累计 dev | 41.22 | 38.33 | +2.89 pp | 9/16 |
| 累计 test | 39.72 | 35.38 | +4.34 pp | 14/16 |
| 历史阶段 test | 64.30 | 59.22 | +5.07 pp | 12/16 |
| 当前阶段 test | 33.15 | 28.30 | +4.85 pp | 13/16 |

同一 artifact 的无标签 InfoNCE 在两个 condensation seed 上分别降低 `2.023` 和 `1.831`，当前阶段 InfoNCE 分别降低 `1.474` 和 `1.401`。因此该比较支持“锚定项确实阻止了图文对应漂移，并能改变 student 响应”。

### 各自 dev 最优点的公平比较

零权重 ASM 自己的 dev 最优点是 outer=20。将候选 outer=60 与基线 outer=20 比较：

| 指标 | Text-Anchored | ASM dev-best | 差值 |
|---|---:|---:|---:|
| 累计 train | 36.62 | 36.61 | +0.01 pp |
| 累计 dev | 41.22 | 40.78 | +0.44 pp |
| 累计 test | 39.72 | 38.26 | +1.47 pp |
| 历史阶段 dev | 66.10 | 67.40 | −1.30 pp |
| 历史阶段 test | 64.30 | 63.91 | +0.38 pp |
| 当前阶段 dev | 35.80 | 32.77 | +3.03 pp |
| 当前阶段 test | 33.15 | 29.68 | +3.47 pp |

公平选择后主要收益集中在当前阶段塑性；累计收益明显缩小，历史 dev 还下降。这是把方法视为“有希望的修复”而非“已经成立的算法”的原因。

## 严格 stage 1→stage 2 递归验证

严格递归实验为每个 condensation seed 独立执行：

1. 从共同的 stage-0 synthetic TAG 出发；
2. 只读取该 TAG 与 stage-1 delta，生成 outer `0/20/40/60/80/100`；
3. 只用 student seeds `100/101` 的累计 dev Macro-F1 选择 stage-1 artifact；
4. stage 2 只读取被选中的 stage-1 artifact 与 stage-2 delta；
5. 再按同一 dev 规则选择 stage-2 artifact；
6. 用未参与 outer 选择的 student seeds `102–107` 重训，再读取 train/dev/test/future 矩阵。

### stage 1 暴露了优化伤害

四条轨迹，即两个 condensation seed × 两种方法，全部选择 outer=0。候选和基线的 stage-1 `X/text/topology set/weights/origin IDs/stage` 完全相同；原始 COO 边顺序不同，但排序后的边集合相同。独立确认结果因此逐项一致：

| 指标 | 两种方法共同结果 |
|---|---:|
| 累计 dev | 53.99 |
| 累计 test | 55.27 |
| stage-0 test | 78.15 |
| stage-1 test | 44.63 |
| future stage-2 test | 26.31 |

这一步说明 stage 1 并没有验证 anchor 的正收益。`lambda=0.01` 能减缓部分 outer 的下降，但无法让更新后的 artifact 超过初始化。

### stage 2 的均值上升，但没有跨 condensation seed 稳定

开发选择点为：

| condensation seed | ASM λ=0 | Text-Anchored |
|---:|---:|---:|
| 44 | outer 20，dev 45.05 | outer 0，dev 44.75 |
| 45 | outer 100，dev 40.10 | outer 100，dev 39.59 |

选择 student 上，候选累计 dev 平均低约 `0.40 pp`。在 12 个独立确认 student 上：

| 指标 | Text-Anchored | ASM λ=0 | 差值 | 正向配对 |
|---|---:|---:|---:|---:|
| 累计 train | 36.60 | 35.04 | +1.56 pp | 7/12 |
| 累计 dev | 40.78 | 38.86 | +1.91 pp | 7/12 |
| 累计 test | 39.10 | 37.49 | +1.62 pp | 7/12 |
| 历史阶段 test | 63.99 | 60.56 | +3.43 pp | 9/12 |
| 当前阶段 test | 32.25 | 31.13 | +1.12 pp | 8/12 |

均值看似正向，但 condensation seed 分解揭示明显不稳定：

| 指标差值（Text-Anchored − ASM） | seed 44 | seed 45 |
|---|---:|---:|
| 累计 dev | −1.45 pp | +5.27 pp |
| 累计 test | −0.84 pp | +4.08 pp |
| 当前阶段 test | −0.89 pp | +3.13 pp |

因此严格递归验证不支持“当前候选跨 seed 稳定胜出”。它支持的较弱结论是：anchor 能在某些递归轨迹上同时提高历史与当前阶段，但收益强烈依赖 condensation seed 和 outer 选择。

## 与现有持续基线的关系

原持续 stage-2 开发证据为：

| 方法 | 累计 test Macro-F1 |
|---|---:|
| Full-history | 49.37 |
| Append | 37.56 |
| Append-joint | 36.53 |
| Protected-joint | 37.44 |
| Union | 35.53 |
| 严格递归 Text-Anchored ASM | 39.10 |

Text-Anchored ASM 的平均值高于已有 compact baselines，但它低于 full-history 约 `10.27 pp`，且 seed 稳定性不足。这个差距仍是研究空间；不能用 compact baseline 的较低均值掩盖方法自身在 seed 44 上失败。

## 当前选题和创新点应如何收敛

选题可以继续保持为：**面向类别增量文本属性图的持续凝聚，研究递归重凝聚过程中 graph–text correspondence 与历史训练响应的协同保持。**

与 CaT 的主要差异应放在 TAG 的图文配对和文本条件分类响应，而不是只把 CaT 的图持续学习流程移植过来。与 DANCE 的差异应放在“持续更新一个可独立重训的 synthetic TAG memory”，而非下游 continual learner 的 replay 或正则化。与静态 TAGSAM 的差异是：方法必须处理旧 synthetic TAG 与新 delta 的递归合并，并证明阶段数增加时误差不会持续累积。

当前真正具备证据的创新切入点是：

1. 给出持续 TAG 重凝聚的阶段级误差分解，区分静态压缩误差、旧 memory 响应漂移、新阶段塑性不足和图文对应漂移；
2. 明确证明 TAG 文本对应在固定 GCN 下游中仍是决定性变量；
3. 设计同时受 **text correspondence** 和 **teacher response retention** 约束的递归更新，而不是只增加更强的文本相似度；
4. 使用 dev-only 逐阶段选择和独立 student 重训，防止在一个 stage 或一个 seed 上选择出偶然结果。

## 下一步方法设计与实验顺序

下一版不再单独提高 `lambda`。已有 sweep 表明更强锚定可以把 InfoNCE 降得很低，但不会自动变成稳定分类收益。下一步应加入一个能直接处理 stage 1 优化伤害的响应约束：

\[
L = L_{ASM} + \lambda_t L_{text-anchor}
  + \lambda_r L_{response-retention}.
\]

`L_response-retention` 应在无标签 reference roots 上保存上一阶段 teacher 的 graph-to-class prompt logits 或 pairwise response；旧 synthetic rows 保持旧 teacher 响应，新 rows匹配 current-delta teacher 响应。权重应由阶段级冲突诊断控制，而不是固定对所有阶段同强度施加。

实验按以下顺序推进：

1. 在 stage 1 上做最小 response-retention gate。成功标准是两个 condensation seed 都选择 `outer>0`，且累计 dev 不低于 outer=0；
2. 固定通过 gate 的权重，只做 stage 1→2 递归，不访问 test，检查两个 seed 的累计、历史、当前 dev 均不劣于各自 λ=0 dev-best；
3. 再用独立 student seeds 检查 test，并报告配对胜率与 condensation-seed 分解；
4. 通过后扩展到更多 condensation seeds 和第二个 TAG 数据集；
5. 最后才在新的、未被当前开发读取的数据集上做最终留出评估。

跨阶段边、transport、GraphCLIP 和额外 prompt 暂不加入。当前证据首先要求解决 response 与 text 两个目标之间的冲突；加入更多模块会让失败无法归因。

## 可复核材料

- 官方静态复核：[OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md](OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md)
- 原轨迹诊断：[DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md](DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md)
- 文本锚定 sweep：[aggregate.json](../../results/native/text_anchor_sweep_2026_09_08/aggregate.json)
- 同 outer 矩阵：[anchor_candidate_confirmation_matrix.json](../../results/native/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_confirmation_matrix.json)
- 各自 dev 最优矩阵：[anchor_candidate_vs_devbest_matrix.json](../../results/native/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_vs_devbest_matrix.json)
- 严格递归汇总：[recursive_summary.json](../../results/native/recursive_text_anchor_2026_09_08/recursive_summary.json)
- 严格递归完整矩阵：[recursive_protocol_matrix.json](../../results/native/recursive_text_anchor_2026_09_08/recursive_protocol_matrix.json)
- dev-only artifact 选择器：[select_trajectory_artifact.py](../../code/native_tag_v1/select_trajectory_artifact.py)
- 递归汇总器：[summarize_recursive_text_anchor.py](../../code/native_tag_v1/summarize_recursive_text_anchor.py)

