# TGC-R4 关系响应凝聚：正式小图结果报告

日期：2026-09-23。运行目录：`results/tgc_r4_relational_20260923c`。

本报告只使用修正版正式运行 `20260923c`。首次运行 `20260923` 因 CUDA Adam 路径数值不一致而中止；`20260923b` 为放宽容差的试运行，均不作为正式证据。

## 1. 执行结论

本轮完整小图矩阵已经完成，但主候选 `relation_new` 未通过预注册晋升门槛，因此不能进入独立 confirmation、递归 rollout、大图或 LP。结论是：

> lifted support 确实解决了 teacher 参考图中的旧传播算子偏差；关系响应外层优化也能降低短程 validation relation loss；但是“只优化新合成节点、保持旧关系”的短程目标没有稳定转化为正式 student 的最终 Macro-F1 提升。

主候选最终四个 cell 的平均相对 C0 为 `-0.040 pp`，相对 matched-random 为 `-0.341 pp`。因此 R4 的关系机制目前只能作为已验证的机制诊断和负结果，不能替换 C0 成为主方法。

值得继续分析的是 `relation_old`：它在最终四 cell 的描述性平均提升为 `+1.224 pp`，但 stage 1 在四个开发 cell 上平均 `-1.356 pp`，三阶段等权平均 `-0.044 pp`。它的最终均值、至少 3/4 cell 在 C0-1 pp 以内、最差最终差值、旧类和新类差值均满足原筛选数值条件；最差最终 cell `-1.210 pp` 仍在允许的 `-2 pp` 以内。不能把它写成“数值门槛失败”。原晋升规则只为主候选 `relation_new` 注册，现有 random 也只匹配新行；因此 `relation_old` 尚无有效的完整晋升判定。它是值得新一轮确认的探索信号，不能直接事后替换主候选。

复核说明：本报告后续修正了 old-only 门槛解释、区间口径、非递归结果的解释、support 对照范围及成本倍数。原始封存候选、CSV、summary 和 `relation_new` 未晋升的决定均未改动。下一轮分析见 [R4 结果复核与推进判断](TGC_R4_INTERPRETATION_AND_NEXT_DECISION_ZH_2026-09-23.md)。

## 2. 固定协议与完成度

| 项目 | 正式配置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 开发顺序 | `order19341`、`order19342` |
| 阶段 | 3；s0 共用 C0，s1/s2 从相同未修订 C0 父记忆出发 |
| 累计预算 | 合法 pool 的 5% |
| C0 | dispersion allocation、`joint_g5_c320`、mean-X、冻结 MiniLM 重编码、frozen-old topology、cross degree 1 |
| teacher | seed 8000/8001，128 steps，batch 64，Adam 0.001 |
| inner student | scratch two-layer GCN、冻结 MiniLM、双向 InfoNCE、batch 16、32 steps、Adam 0.001 |
| outer optimization | 16 steps，Adam 0.01，相对 X 半径 0.1，梯度裁剪 5 |
| 旧关系约束 | 平均双向 KL ≤ 0.001，非负 dual，末端可行缩放 |
| report student | seeds 100--105；从头训练；`max(50,min(2048,16×rows))` steps |
| 指标 | all-prompt pooled cumulative Macro-F1，同时报告 old/new F1 |
| 候选 | C0、relation_new、matched-random、stop、InfoNCE new/old、relation_old、ASM new、free/joint/shuffled controls |

流水线 `prepare -> seal -> evaluate -> audit` 全部返回 0。共生成 792 个结果槽位，实际执行 504 次 student 训练；其余是 s0/no-op 或共享结果复用。四个 dataset/order audit 全部通过，独立重载指标最大误差约 `2.84e-14 pp`，没有使用 test labels 进行构造或选择。

## 3. 最终阶段主表

下面是 stage 2 的四个开发 cell 等权统计。`Delta` 是同一 cell、同一 report seed 下相对 C0 的配对差值，单位为百分点。**本表区间按四个 cell 的六-seed 均值差计算，n=4，仅为描述性 t 区间**；四个 cell 共用两个数据集，不能当作四个独立数据集推断。逐 cell 的六-seed paired t 区间另见 `cells.csv`，两种区间不可混称。

| 方法 | 最终 Macro-F1 | 相对 C0 | 描述性 95% 配对区间 | 最差 cell | 最差 old F1 差值 | 新类平均差值 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 54.427 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.000 |
| relation_new | 54.387 | -0.040 | [-2.813, 2.734] | -1.218 | -1.918 | -0.442 |
| matched-random new | 54.728 | +0.301 | [-1.807, 2.409] | -0.917 | -1.242 | +0.533 |
| relation_old | 55.651 | +1.224 | [-2.096, 4.544] | -1.210 | -0.229 | +1.749 |
| InfoNCE new | 54.313 | -0.114 | [-2.484, 2.257] | -1.990 | -1.997 | +0.006 |
| InfoNCE old | 54.341 | -0.085 | [-3.651, 3.481] | -2.383 | -1.553 | -0.175 |
| ASM new | 54.435 | +0.008 | [-3.436, 3.452] | -3.061 | -1.794 | -0.417 |
| relation_new_free | 54.513 | +0.087 | [-1.980, 2.153] | -1.065 | -2.209 | +0.443 |
| relation_joint | 54.602 | +0.176 | [-2.144, 2.495] | -1.119 | -1.791 | +0.310 |
| relation_shuffled | 54.350 | -0.077 | [-2.518, 2.365] | -1.634 | -1.355 | +0.098 |

`stop` 与 C0 完全相同，说明 no-op 路径正确。`relation_new` 没有达到 `+0.5 pp` 平均提升、至少 3/4 cell 不低于 `C0-1 pp`、新类平均不下降或不劣于 random 的要求；因此晋升结果为 **不通过**。

### 3.1 主候选逐 cell 结果

| 数据集 / 顺序 | stage 1 C0 | stage 1 relation_new | stage 1 差值 | stage 2 C0 | stage 2 relation_new | stage 2 差值 |
|---|---:|---:|---:|---:|---:|---:|
| Cora / order19341 | 52.315 | 50.654 | -1.661 | 48.982 | 51.482 | +2.500 |
| Cora / order19342 | 69.959 | 72.102 | +2.143 | 55.496 | 54.356 | -1.139 |
| Citeseer / order19341 | 66.066 | 68.509 | +2.443 | 56.108 | 54.890 | -1.218 |
| Citeseer / order19342 | 58.968 | 59.473 | +0.505 | 57.121 | 56.820 | -0.301 |

模式显示出阶段与顺序依赖：relation_new 在三个 stage 1 cell 为正，但四个最终 stage 2 cell 的平均几乎为零。stage 1 的旧类平均差值为 `+1.896 pp`，新类为 `-0.181 pp`，因此不能解释为已改善新类塑性。s1/s2 各自从未修订 C0 父记忆出发，尚未检验递归累积收益，也不能由该表推断长期遗忘或收益被下一阶段抵消。

## 4. 支持图机制诊断

### 4.1 lifted support 保持旧传播块

将 synthetic 新簇的跨边按簇内 raw 节点数分摊到 teacher support 后，相对于本阶段 compact C0 图，旧节点的度数和旧旧归一化 adjacency block 得到保持。新侧特征、归一化跨块及多跳传播仍可不同，不能称整个旧传播输出不变。8 个增量 cell 中：

- lifted uniform / compact 的旧归一化块最大误差约 `5.96e-7`；
- legacy raw-append support 的误差约 `0.667--0.968`；
- 误差来自旧节点度数被 raw 新节点数量放大，而不是 sampling 比例造成。

这验证了一个具体的算子诊断：**support context 应按最终 synthetic cluster 的规模映射，不能直接把 raw delta 节点当成 compact new rows。** 这是结构正确性证据，不是下游 F1 定理。

### 4.2 sampling 与 support 的分离

teacher 旧行采样比例依条件变化：uniform 通常约 `0.025--0.135`，compact-ratio 通常约 `0.328--0.750`。两者都能降低 direct distillation validation KL。**本轮仅对事前固定的 `lift_compact` 构造正式候选；另外三个支持条件只有 teacher/direct-distillation 诊断，没有各自的正式下游 F1 对照。** 因此尚不能判断 compact-ratio 或 lifted support 的独立下游收益，也不能宣称它们已在性能上优于其它支持条件。

## 5. teacher 与 inner response 证据

在所有 support 条件中，direct distillation validation KL 都由约 `3.0--3.2` 降至约 `0.16--0.47`；这说明同架构 student 能在固定 probe 上学习 teacher relation。该实验回答的是接口和可拟合性问题：teacher 目标不是数值空壳。

但它不等于压缩图能改善完整训练。正式 `relation_new` 的 outer validation improvement 平均约 `0.00923`，旧 validation KL 平均约 `0.000238`；约束每个 outer run 平均有 15 个 active steps，最终可行 scale 平均约 `0.281`，平均修改约 33 行，最大相对位移约 `0.0281`。也就是说，优化确实做了受限的 new-X 修改，但短程 relation loss 与长期 report F1 的相关性不足。

## 6. 候选对照给出的机制解释

1. `relation_new` 低于 matched-random `0.341 pp`，说明在当前候选空间中，外层 relation gradient 的方向没有稳定超过同位移随机方向。
2. `relation_shuffled` 最终为 `-0.077 pp`，接近 relation_new，说明真实 teacher 对应关系的有效信号较弱，或者优化后的局部关系变化被后续训练冲掉。
3. `relation_new_free` 为 `+0.087 pp`，但 old F1 最差下降 `-2.209 pp`。去掉旧约束没有得到足以支持晋升的改善，且出现较大旧类退化；不能据此声称放松任何约束都只会增加风险。
4. `relation_joint` 为 `+0.176 pp`，仍低于 random，且最差 cell 为 `-1.119 pp`，说明同时编辑旧、新行没有解决方向错位。
5. `relation_old` 的 `+1.224 pp` 是最强描述性信号，最终数值满足原筛选条件中的效果阈值，但 stage 1 平均 `-1.356 pp`，三阶段均值约持平，且缺少 old-only matched-random/shuffled 控制。本轮 old-only 改 2--8 行、平均最终缩放 0.9375，new-only 改 28--42 行、平均缩放 0.28125；编辑维数和有效幅度不同，尚不能将两臂差异全部归因于“旧行更值得优化”。
6. `asm_new` 与 C0 几乎相同，说明在相同 inner schedule 下，换成 ASM 风格目标没有自动产生提升；不能把本轮收益归因于普通 teacher/student 优化。

## 7. 成本与部署含义

| 项目 | 平均 | 范围 |
|---|---:|---:|
| C0 构造 | 0.173 s | 0.124--0.221 s |
| support 物化 | 0.049 s | 0.033--0.063 s |
| 全部 support 诊断 | 4.118 s | 3.886--4.403 s |
| 选定 teacher | 0.619 s | 0.590--0.670 s |
| relation outer optimization | 9.078 s | 8.633--10.534 s |
| primary build + diagnostics | 13.419 s | 12.682--15.207 s |
| selected components lower bound | 9.919 s | 9.388--11.470 s |
| peak outer GPU allocation | 1229 MB | 1209--1248 MB |

按上述八个阶段的平均计时，selected components lower bound 为 C0 构造的约 `57.2x`，primary build + diagnostics 约 `77.4x`，不是“两个数量级以上”。这里是已计时、已有缓存条件下的局部构造费用，不是完整冷启动成本或构造加正式训练的总倍数。主候选没有可靠收益时，额外计算不值得默认启用；绝对约 10--15 秒的构造费用也不应单凭该倍数否定后续学习型算法。

## 8. 晋升门槛逐项判断

| 门槛 | 要求 | relation_new | 判断 |
|---|---:|---:|---|
| 四 cell 平均最终提升 | ≥ +0.5 pp | -0.040 pp | 失败 |
| 至少 3/4 cell ≥ C0-1 pp | 3 | 2 | 失败 |
| 最差最终 cell | ≥ -2 pp | -1.218 pp | 通过 |
| 最差 old F1 | ≥ -2 pp | -1.918 pp | 通过 |
| 新类平均差值 | ≥ 0 pp | -0.442 pp | 失败 |
| 不劣于 matched-random | ≥ 0 pp | -0.341 pp | 失败 |
| 单阶段 primary build 成本 | ≤ 600 s | 15.207 s | 通过 |

最终：`primary_eligible_for_confirmation = false`。

## 9. 对当前方法的评价与后续建议

R4 没有证明“关系响应凝聚”整体无效，而是排除了一个具体配方：`lift_compact + relation_new + old KL constraint + 32-step scratch response + new-only X revision` 不能作为稳定主线。它留下了两个可复用结果：

- 支持图跨边按 synthetic cluster 分摊满足明确的旧归一化子块保持性质，可作为后续参考图控制；性能必要性尚未被隔离验证；
- 旧行方向在最终阶段出现较大但不稳定的描述性增益，值得用独立顺序和事前固定的 old-adaptation 假设验证。

但按当前停止规则，本轮不做独立 confirmation、递归、大图和 LP。下一步若继续 TGC，应先冻结一个极小的新实验，而不是叠加所有模块：

1. 以 `relation_old` 作为待验证假设，单独设计独立顺序、固定 old editable mask、同位移 random control 和 C0；不要使用本轮四个 cell 选择配置。
2. 增加 response horizon 对照（32/128/完整 canonical prefix），检查短程关系是否能迁移到最终训练。
3. 记录 old/new relation、old/new F1 与 row-level drift 的对应关系，判断收益来自历史传播修复还是新类拟合。
4. 只有 old-adaptation 在独立小图确认通过后，才考虑递归或大图；否则保留 C0 作为主方法，将 R4 写入“结构机制验证与负结果”。

本文不应把 relation_old 的事后均值写成已成立的新主方法，也不应把 lifted support 的代数保持写成无遗忘保证。

## 10. 证据文件

- 冻结协议：[TGC_R4_RELATIONAL_FROZEN_PROTOCOL_ZH_2026-09-23.md](TGC_R4_RELATIONAL_FROZEN_PROTOCOL_ZH_2026-09-23.md)
- 设计审查：[TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md](TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md)
- 机器汇总：[summary.json](../../results/tgc_r4_relational_20260923c/summary.json)
- 逐 cell：[cells.csv](../../results/tgc_r4_relational_20260923c/cells.csv)
- 逐 seed：[per_seed.csv](../../results/tgc_r4_relational_20260923c/per_seed.csv)
- 支持图诊断：[supports.csv](../../results/tgc_r4_relational_20260923c/supports.csv)
- 优化与成本：[optimizations.csv](../../results/tgc_r4_relational_20260923c/optimizations.csv)、[costs.csv](../../results/tgc_r4_relational_20260923c/costs.csv)
