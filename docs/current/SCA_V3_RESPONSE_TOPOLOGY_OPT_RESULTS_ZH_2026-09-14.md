# SCA v3 响应曲线与拓扑冻结优化结果

日期：2026-09-14  
协议：`tagsam-unified-v3-2026-09-13`  
远程环境：`pc@REMOTE_SERVER_HOST`，conda `tagsam`

## 1. 本轮目标

本轮针对旧版 Student-Compatible Append 的两个明确缺陷进行优化：

1. stage2 的短训终点不能稳定预测 canonical student ranking；
2. append 后所有 synthetic edges 被全局重建，旧 memory 的拓扑没有真正冻结。

旧版使用单一 selection steps、selection seeds `100/101/102`，并且只在最终 canonical sweep 中检查候选排序。新版改为：

- selection seeds 与最终 student seeds 解耦：`200/201/202` vs `100..105`；
- 每阶段使用 stage-aware response curve，默认以 `max(128, final_steps/4)` 为基准，再取 `0.5x/1x/2x` 并截断到 canonical steps；因此实际默认点通常约为最终 student steps 的 `0.125x/0.25x/0.5x`；
- 在 `step × batch` 条件上计算候选相对排序的一致性；
- 用 response AUC、尾部均值、student 方差、旧类保持和排序一致性联合选择；
- 默认 `frozen_old` topology：旧行之间已有的边不变，只新增新块内部边和每个新行到最相似旧行的跨边；
- 保留 `global` topology 作为可复现消融。

## 2. 实现变化

主要代码：

- `code/frozen_v1_2026_09_09/run_student_compatible_append.py`
- `code/frozen_v1_2026_09_09/run_gate1_continual.py`
- `code/frozen_v1_2026_09_09/run_gate2_citeseer.py`

### 2.1 Student response curve

候选不再只看一个短训点。对每个候选，在多个 training horizons、多个 batch 和独立 selection seeds 上重新初始化 student，记录：

- 当前阶段 Macro-F1；
- 历史阶段平均 Macro-F1；
- cumulative Macro-F1；
- batch/seed 条件下的均值和方差。

曲线指标为：

```text
response_auc = current-F1 曲线的归一化梯形面积
tail_mean    = 最后两个 horizon 的 current-F1 均值
tail_std     = 最后两个 horizon 的平均 student std
```

选择 utility 为：

```text
0.5 * tail_mean
+ 0.5 * response_auc
- 0.5 * tail_std
+ 0.5 * old_delta
+ 5 * rank_consistency_weight * rank_consistency
```

同时保留 Pareto front，并要求旧类下降不超过 `2 pp`；默认候选排序一致性分数低于 `0.5` 时不进入首选集合，若全部候选都不满足则回退到 retention 约束。

### 2.2 Ranking consistency

对每个 `step × batch` 条件，先按候选 current-F1 得到一个 ranking，再与 pooled ranking 比较。每个候选记录：

- `mean_rank`；
- `rank_std`；
- `rank_stability`；
- `pairwise_agreement`；
- 综合 `score`。

它不是使用 canonical 结果作弊，而是只利用 selection 阶段内部条件的排序稳定性。canonical sweep 只在实验结束后用于独立验证。

### 2.3 Frozen-old topology

memory artifact 现在显式保存 `edge_index`。`shrink_by_mass` 会对保留行重映射边；append 时：

```text
old-old edges: 保留
new-new edges: 在新 block 内单独构建
old-new edges: 每个新行连接到最相似的 cross_degree 个旧行
```

`memory_to_trainable` 优先读取 artifact 中的 edge_index，因此 student 不会在训练时悄悄重新生成全局 topology。旧 artifact 没有 edge_index 时仍兼容旧逻辑，并按当前 text embedding 确定性重建。

## 3. 主结果：frozen-old SCA

最终评估使用 6 个 canonical student seeds、batch16 和协议步数。

### Cora

| 阶段 | 选择候选 | selection curve | current F1 | cumulative F1 |
|---|---|---|---:|---:|
| s0 | `joint_g8_c512` | `64/128/256` | 66.44 | 66.44 |
| s1 | `xonly_g5_c320` | `156/312/624` | 65.20 | **67.30** |
| s2 | `joint_g3_c192` | `216/432/864` | 56.31 | **56.95** |

### Citeseer

| 阶段 | 选择候选 | selection curve | current F1 | cumulative F1 |
|---|---|---|---:|---:|
| s0 | `joint_g5_c320` | `88/176/352` | 61.25 | 61.25 |
| s1 | `joint_g3_c192` | `196/392/784` | 85.53 | **76.67** |
| s2 | `joint_g8_c512` | `254/508/1016` | 62.58 | **64.13** |

结果目录：

- `results/sca_v3_opt_2026_09_14_cora/`
- `results/sca_v3_opt_2026_09_14_citeseer/`

两套结果均记录 `test_accessed=false`，凝聚源排除 `original_id % 10 in {1,2}`。

## 4. 归因：selection 与 topology 分开

### 4.1 同样 frozen-old topology 下，SCA 相对 fixed append

| 数据集 | s0 增益 | s1 增益 | s2 增益 |
|---|---:|---:|---:|
| Cora | +0.67 | **+7.72** | +1.57 |
| Citeseer | +0.00 | **+9.25** | +3.40 |

这里的 fixed append 使用同一 v3 协议、同一 frozen-old topology，只关闭 candidate selection。因此这些差异是 selection 在拓扑冻结之上的增益。

### 4.2 同一 response-based selector 下，frozen-old 相对 global topology

| 数据集 | s0 | s1 | s2 |
|---|---:|---:|---:|
| Cora | +0.00 | +1.57 | **+4.91** |
| Citeseer | +0.00 | +7.95 | +3.50 |

global 消融结果目录：

- `results/sca_v3_opt_global_2026_09_14_cora/`
- `results/sca_v3_opt_global_2026_09_14_citeseer/`

因此当前证据支持一个双机制解释：

```text
student-compatible selection 解决“新增 block 选谁”
frozen-old topology           解决“历史 block 的图结构是否被重写”
```

不能把两者合并称作单一 selection 增益。

## 5. Canonical ranking confirmation

对每阶段全部 4 个候选使用最终 canonical 协议复评。

### 新版 frozen-old SCA

| 数据集 | s0 | s1 | s2 |
|---|---:|---:|---:|
| Cora | 第1名 | 第1名 | 第1名 |
| Citeseer | 第1名 | 第1名 | 第1名 |

具体 canonical ranking：

- Cora s0：`joint_g8 > joint_g5 > joint_g3 > xonly`
- Cora s1：`xonly > joint_g8 > joint_g5 > joint_g3`
- Cora s2：`joint_g3 > joint_g5 > joint_g8 > xonly`
- Citeseer s0：`joint_g5 > joint_g3 > xonly > joint_g8`
- Citeseer s1：`joint_g3 > xonly > joint_g8 > joint_g5`
- Citeseer s2：`joint_g8 > xonly > joint_g5 > joint_g3`

canonical 结果目录：

- `results/sca_v3_opt_canonical_2026_09_14_cora/`
- `results/sca_v3_opt_canonical_2026_09_14_citeseer/`

旧版 selection 的 s2 选择排名为 Cora 第3、Citeseer 第4；新版两者均为第1。这是本轮最直接的机制证据。

## 6. 当前论文判断

本轮优化后，Student-Compatible Append 可以更有力地表述为：

> 在严格 v3 持续协议下，使用独立 student response curve 和跨条件 ranking consistency 选择新增 synthetic block，并冻结历史 synthetic topology，可以同时改善保持-塑性折衷；该收益在两个小型 TAG 上成立，但仍不应表述为普遍 SOTA。

可以主张的内容：

1. selection proxy 的 stage2 ranking 失配被 response curve + consistency 纠正；
2. frozen-old topology 与 global topology 的收益可以通过同协议消融分离；
3. Cora/Citeseer 的 s1/s2 cumulative F1 相对同拓扑 fixed append 有稳定正增益。

仍需保留的边界：

1. 当前证据只覆盖 Cora/Citeseer 小图，arXiv 大图尚未用 SCA 复核；
2. topology 的跨边规则仍是简单确定性规则，不应包装为一般动态图学习理论；
3. selection 使用 dev 评估，虽然 selection seeds 与 canonical seeds 分离，仍需要独立数据集或独立 dev split 做最终泛化确认；
4. 当前结果支持“机制分析 + 可靠构造”，暂不支持 SOTA 宣称。

## 7. 复现实验命令

```bash
python run_student_compatible_append.py \
  --dataset cora \
  --output /path/to/sca_v3_opt_cora \
  --topology-mode frozen_old

python run_student_compatible_canonical_sweep.py \
  --dataset cora \
  --artifact-dir /path/to/sca_v3_opt_cora \
  --output /path/to/sca_v3_opt_canonical_cora
```

使用 `--topology-mode global` 可复现实验中的拓扑消融。

## 8. Beam path 独立确认：修正“selector 已稳定”的判断

上一版文档把 3-seed greedy 在两个数据集上命中 canonical 第 1 名，表述得过于乐观。随后用 selection seeds `200..205` 做独立 6-seed 复核，得到：

| 数据集 | 3-seed greedy 路径 | 6-seed greedy 路径 | cumulative F1 |
|---|---|---|---:|
| Cora | `g8 -> xonly -> g3` | `g5 -> xonly -> g8` | 49.72 |
| Citeseer | `g5 -> g3 -> g8` | `g5 -> xonly -> g8` | 63.72 |

Cora 的差异说明：早期候选的 response 差距较小，selection seed 改变会导致父路径分叉；历史 memory 被 append 后，这个小差异会在后续阶段被放大。因此，单路径 greedy 的局部 ranking 不能被解释为全局路径稳定性。

### 8.1 Beam2 terminal 结果

对每一阶段保留两个完整父路径，并在最后阶段加入 full-step terminal confirmation。结果目录：

- `results/sca_v3_beam2_terminal_2026_09_14_cora/`
- `results/sca_v3_beam2_terminal_2026_09_14_citeseer/`

最终保留的路径为：

| 数据集 | beam 首路径 | 备选路径 |
|---|---|---|
| Cora | `g8 -> xonly -> g3` | `g5 -> xonly -> xonly` |
| Citeseer | `g5 -> g3 -> g8` | `g8 -> g3 -> g8` |

### 8.2 独立 canonical confirmation

使用与 selection 完全分离的 canonical student seeds `100..105`、batch `16` 和 v3 protocol-scaled steps，对两条 terminal path 重新训练：

| 数据集 | 路径 | cumulative F1 mean | std | current F1 mean |
|---|---|---:|---:|---:|
| Cora | `g8 -> xonly -> g3` | **56.95** | 1.88 | 56.31 |
| Cora | `g5 -> xonly -> xonly` | 48.98 | 3.23 | 40.50 |
| Citeseer | `g5 -> g3 -> g8` | **64.13** | 1.49 | 62.58 |
| Citeseer | `g8 -> g3 -> g8` | 62.33 | 2.48 | 53.20 |

独立确认与 beam 内部的 selection-seed terminal 分数方向一致，但不能据此宣称 beam 普遍提升性能：本实验只确认 beam 能保留一个不被 greedy 早期误删的高质量路径，尚未证明 beam 宽度、计算预算或路径搜索在更多数据集上稳定有效。

### 8.3 更新后的论文边界

当前更准确的写法是：

> response-based SCA 能够在严格 v3 协议下提供可解释的候选选择信号；frozen-old topology 分离了历史结构保持的作用；当局部 selection ranking 存在路径不稳定时，低宽度 beam 可以作为计算可控的风险缓解策略。

不应写成：

1. 单路径 selector 已经对 seed 稳健；
2. beam 是普遍优于 greedy 的新主算法；
3. 两个小型数据集上的 cumulative F1 足以支持 SOTA。

因此论文主线暂定为“student response 机制分析 + frozen-old 可靠 append + 负结果/路径不稳定性诊断”；beam 放在路径稳定性实验或附录，不作为主方法的必要组成部分。

独立确认产物：

- `results/sca_v3_beam2_terminal_confirm_2026-09-14_cora/`
- `results/sca_v3_beam2_terminal_confirm_2026-09-14_citeseer/`

## 9. Selection seeds `300..302` 的独立复核：batch 混合与终端 proxy 失配

为检查 beam 是否只是适配 `200..202`，使用新的 selection seeds `300..302`，保持 beam width `2`、selection batches `{8,16,32}`、canonical student seeds `100..105` 和 frozen-old topology 不变。

| 数据集 | beam 选择路径 | canonical cumulative F1 |
|---|---|---:|
| Cora | `g5 -> xonly -> g5` | **51.37** |
| Citeseer | `g5 -> g3 -> g8` | **64.13** |

对 Cora 的两条 terminal beam path 再用 canonical seeds 独立确认后，排名发生反转：

| 路径 | selection-terminal 分数对应的排序 | canonical cumulative F1 |
|---|---:|---:|
| `g8 -> xonly -> g8` | 1（selection-terminal 52.70） | **52.59** |
| `g5 -> xonly -> g5` | 2（selection-terminal 53.22） | 51.37 |

这说明当前 response proxy 的误差来源至少有两层：

1. selection 使用 `{8,16,32}` 的等权 pooled response，而 canonical 固定 batch `16`；Cora 的 batch-32 表现显著偏低，会改变候选的 pooled 排序；
2. 即使在 selection-terminal 使用同一批 selection seeds，短训终点仍不能完全预测 canonical full-step ranking。

所以 beam2 不能被表述为“解决路径不稳定”，它只能保留少量备选供事后诊断。后续应优先测试 batch-16 主导的 response proxy，并将 full-step terminal confirmation 视为独立评估而不是 selection 的可靠替代。

新增产物：

- `results/sca_v3_beam2_select300_2026-09-14_cora/`
- `results/sca_v3_beam2_select300_2026-09-14_citeseer/`
- 远程独立确认结果位于对应 `sca_v3_beam2_select300_confirm_2026-09-14_*` 目录。

## 10. Batch-16 主导 proxy 消融：未解决 Cora 失配

随后固定 selection batch 为 canonical batch `16`，其余设置保持不变：selection seeds `300..302`、beam width `2`、frozen-old topology、canonical student seeds `100..105`。该实验目录为 `results/sca_v3_beam2_batch16_2026-09-14_cora/`。

| 设置 | Cora beam 选择路径 | canonical cumulative F1 |
|---|---|---:|
| pooled batches `{8,16,32}` | `g5 -> xonly -> g5` | 51.37 |
| selection batch `{16}` | `g5 -> xonly -> g5` | 51.37 |

虽然中间 s1 的 beam 顺序发生变化，但最终 s2 路径和 canonical 分数没有改善；selection-terminal 对两个候选的排序仍与 canonical ranking 不一致。由此可以排除“batch 混合是唯一根因”的解释。当前更可能的瓶颈是：短训 response curve 与 full-step student dynamics 本身不一致，加上早期路径差异在后续 append 中被放大。

这项消融是负结果，当前不纳入主方法。后续若继续优化，应优先研究跨 horizon 的 trajectory/learning-curve 特征、独立 selection split，以及对路径不确定性的风险估计，而不是继续手工调整 batch 权重。

## 11. Terminal full-step lookahead：部分修正 premature pruning

为直接检验“terminal 候选在短训阶段被过早淘汰”的假设，新增 `--terminal-lookahead-all` 开关。开启后，前两个 stage 仍使用 response curve、rank consistency 和 beam2；最后一个 stage 则保留每个 beam parent 的全部候选，使用 selection seeds `300..302`、canonical batch `16` 和 protocol-scaled full steps 进行一次 terminal lookahead，再按 `mean - 0.25*std` 选出最终 beam。这个过程仍只使用 dev mask，canonical student seeds `100..105` 只用于独立确认。

| 数据集 | 普通 beam2，selection `300..302` | terminal-lookahead 选择 | canonical cumulative F1 |
|---|---|---|---:|
| Cora | `g5 -> xonly -> g5` | `g8 -> xonly -> g3` | **56.95** |
| Citeseer | `g5 -> g3 -> g8` | `g5 -> g3 -> g8` | **64.13** |

terminal-lookahead 结果目录：

- `results/sca_v3_lookahead_2026-09-14_cora/`
- `results/sca_v3_lookahead_2026-09-14_citeseer/`

该实验支持更具体的机制结论：Cora 的主要失配确实部分来自 terminal-stage 的 premature pruning；让 terminal 候选看到完整 selection budget 后，能从 `300..302` 的错误路径恢复到此前 canonical 更优的路径。Citeseer 原本已经稳定，因此 lookahead 不改变它的选择。

但这仍不是 SOTA 证据，也不是对所有数据集都成立的理论保证。当前论文可将它作为“风险感知的 terminal candidate confirmation”或稳定性增强组件；主贡献仍应聚焦于 student response 机制、历史拓扑冻结和对负结果的诚实分析。

## 12. Selection seeds `200..205` 的 lookahead 复核：仍存在目标分布失配

为检验 terminal lookahead 是否真正获得 seed 稳健性，进一步使用 selection seeds `200..205`，保持其他配置不变。Cora 结果为：

| terminal path | selection full-step cumulative F1 | selection score | canonical `100..105` |
|---|---:|---:|---:|
| `g8 -> xonly -> g8` | 53.63 | **52.52** | 52.59 |
| `g8 -> xonly -> g5` | 52.57 | 52.17 | 未单独确认 |
| `g8 -> xonly -> g3` | 52.76 | 51.68 | **56.95** |

因此，lookahead 消除了部分短训和 premature-pruning 问题，但在 `200..205` 上仍会把 canonical 更优的 `g3` 排在 `g8` 后面。该结果不能归因于 batch 混合，因为这里的 terminal lookahead 已固定 batch `16`；更准确的解释是 selection seed 集合对 full-step student response 的估计仍有较大抽样误差。

### 12.1 新 student seeds `106..111` 的独立路径比较

为了避免把 `100..105` 当作唯一目标，直接用未参与选择的 seeds `106..111` 评估同一父路径下的三个 terminal candidates：

| 路径末块 | cumulative F1 mean | std |
|---|---:|---:|
| `g3` | **55.25** | 3.28 |
| `g8` | 54.51 | 3.91 |
| `g5` | 53.39 | 1.78 |

`106..111` 与 `100..105` 的方向一致：`g3` 是更可靠的 terminal 构造，尽管 selection seeds `200..205` 的 proxy 仍偏向 `g8`。这说明当前的主要瓶颈不是某一组 canonical seeds 的偶然性，而是 selection 与最终评估之间的 seed-distribution shift。

### 12.2 当前方法决策

因此暂不把固定的 terminal candidate 选择器宣称为稳健算法。更合适的论文处理是：

1. 将 response curve、topology freeze 和 terminal lookahead 作为机制组件；
2. 报告 selection seed sensitivity 以及 `200..205` 与 `300..302` 的路径分歧；
3. 用独立 seed 集合 `106..111` 说明路径质量排序并非只由 `100..105` 决定；
4. 将 seed-set ensemble、bootstrap risk estimate 或独立 selection split 留作后续增强方向，而不在当前结果上过度包装。

新增产物：

- `results/sca_v3_lookahead_6seed_2026-09-14_cora/`
