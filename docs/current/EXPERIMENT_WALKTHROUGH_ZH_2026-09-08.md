# 经典持续 TAG 图凝聚实验逐项讲解

日期：2026-09-08  
用途：用通俗语言解释每组实验在哪里、怎样做、结果是什么以及能够推出什么结论  
说明：本文不包含新实验；Cora test/confirmation/future 已被开发访问，相关结果不能作为论文最终未见测试

## 一、先理解整个实验到底在做什么

Cora 中每个节点是一篇论文。每个节点同时有：

1. 图特征 `X`；
2. 论文文本经过 BERT 得到的文本 embedding；
3. 与其他论文的引用边；
4. 一个类别标签，但训练和凝聚代码不直接使用训练节点标签。

TAG 凝聚的目标，是把一张大文本属性图压成一张很小的 synthetic TAG。随后丢掉大训练图，只把小图交给一个全新初始化的 student。若 student 只用小图训练，也能在真实图节点上分类，说明小图保存了训练效用。

当前类别流为：

- stage 0：类别 0、1；
- stage 1：新增类别 2、3；
- stage 2：新增类别 4、5、6。

持续限制是：stage 0 结束后不能再读 stage-0 原始训练图；stage 1 只能使用 stage-0 synthetic TAG 和 stage-1 新图。stage 2 同理，只能使用 stage-1 synthetic TAG 和 stage-2 新图。

下游 student 是两层 GCN 加文本投影头。训练时，每个节点的 GCN graph embedding 应接近自己的文本 embedding，远离同一批次其他节点的文本。分类时，节点 graph embedding 与七个类别文本 prompt 比较，最相似的 prompt 是预测类别。

因此，虽然图编码器叫 GCN，文本仍然决定训练正对、负对和最终分类。图根与文本配错会直接改变分类。

## 二、几个常见指标分别是什么意思

### 1. Accuracy 和 Macro-F1

- Accuracy 是预测正确的节点比例；
- Macro-F1 先对每个类别计算 F1，再对类别平均，能减少大类别支配总体结果的问题。

官方 TAGSAM 复现实验使用 Accuracy，native 持续实验主要使用 Macro-F1。这两类数字不能直接横向比较。

### 2. Train、dev、test 和 future

- `train`：原始训练节点上的诊断；
- `dev/checkpoint`：选择 student checkpoint 和 synthetic outer snapshot；
- `test/confirmation`：候选确定后，用独立 student 读取；
- `future`：模型和 artifact 冻结后，观察尚未到达类别。

### 3. 当前、历史和累计

- `current`：只在本阶段新增类别上计算；
- `old/history`：分别计算旧阶段，再取平均；
- `cumulative seen`：合并所有已经到达阶段的预测和标签后计算；
- `per-stage`：把每个阶段单独列出来。

所以“stage-2 累计 test=39.10”不是只测新增节点，也不是把 Cora 所有节点混成一个数字。它合并 stage 0、1、2 各自 test mask；同时还会单列历史阶段和当前 stage 2。

## 三、实验 1：官方 TAGSAM 静态复核

### 在哪里

- 总报告：`D:/Adocs/简历/research/graph-condensation/TAGSAM/OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md`
- 本地结果：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/official_static/results/`
- official master 远程运行：`/home/pc/research/TAGSAM_official_cora_master_20260907`
- official add-adj 远程运行：`/home/pc/research/TAGSAM_official_cora_addadj_20260907`

### 想回答什么

原始 TAGSAM 能不能直接作为持续研究的可靠静态凝聚器？此前复现不好，究竟是方法不工作，还是 evaluator、cache、拓扑等实现问题？

### 怎么做

实际运行了三条静态链路：

1. official master：原始 self-loop synthetic topology；
2. official add-adj：artifact 中保存 learned synthetic edges；
3. minimal repair：只修复尾批、设备/训练状态和七类评测，不加入新方法模块。

核对发现 official Cora evaluator 写死五类，而且旧的五类 subgraph cache 可能在修改成七类后继续被静默复用。因此先把五类和七类结果分开，再重新生成独立七类 cache。

### 结果

minimal repair 的七类 Test Accuracy：

| condensation seed | step 0 | step 500 | 变化 |
|---:|---:|---:|---:|
| 44 | 33.40 | 36.24 | +2.84 pp |
| 45 | 33.81 | 38.74 | +4.93 pp |

add-adj 七类五次 student 均值：

| artifact | Test Accuracy |
|---|---:|
| step 0 | 38.58 |
| step 500 | 40.23 |

七类 full teacher 为 61.12%。

### 结果是什么意思

官方 TAGSAM 的外层优化不是完全没有作用：minimal repair 两个 seed 都比初始化高，add-adj 也提高约 1.65 pp。但 synthetic student 与 full teacher 仍有约 20 pp 以上差距，student 波动也较大。

这说明静态 TAG 凝聚存在可用信号，但尚不是稳定、接近 full-data 的组件。持续实验需要保留静态误差这一背景。

### 不能推出什么

- 不能把 official 五类数字和七类数字比较；
- 不能把 add-adj 的增益只归因于修复，因为它同时改变了 topology；
- 不能因为一个运行时 student 数字较高，就认为独立重训一定复现；
- 不能把 full teacher 称为严格数学上界。

## 四、实验 2：可审计的 native 静态起点

### 在哪里

- 核心实现：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/native_core.py`
- runner：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/run_native_experiments.py`
- 结果镜像：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/results_mirror/static_r05_c44_check32/` 等目录

### 想回答什么

在统一七类、固定数据划分、独立 student 和可追溯 artifact 的协议下，小 TAG 是否有训练价值？ASM 优化是否真的优于不优化的初始化？

### native 后端与官方 TAGSAM 的关系

native 后端保留了 TAGSAM 的核心接口：冻结 BERT CLS、两层 GCN、文本投影、双向 InfoNCE 和 ASM。它额外实现分阶段数据权限、train/dev/test/future、独立 student 重训、outer 选择、artifact 哈希和持续 provenance。

它是 TAGSAM-derived 开发后端，不是官方论文代码的逐行等价复现。这样做是为了让持续归因可审计，而不是用 native 结果替代官方复现结果。

### 怎么做

在 stage 0 和 5% 压缩率下比较：

- `Full-real`：全部真实训练图；
- `Random-small`：随机选择同样数量的真实节点；
- `Coreset`：代表点小图，保留真实 X；
- `Frozen compact`：联合图文初始化形成的小图，不做 ASM；
- `Offline ASM`：从同一类 compact 初始化继续优化 synthetic X。

每个导出 artifact 都交给全新 student 训练，再在相同划分上计算结果。

### 结果

| 方法 | Dev Macro-F1 | Test Macro-F1 | Future Macro-F1 |
|---|---:|---:|---:|
| Full-real | 84.90 | 83.24 | 21.89 |
| Random-small | 73.73 | 69.73 | 17.11 |
| Coreset | 68.66 | 62.22 | 12.31 |
| Frozen compact | 77.84 | 76.56 | 11.75 |
| Offline ASM | 76.50 | 73.05 | 15.80 |

### 结果是什么意思

`Frozen compact=76.56` 高于 random 和普通 coreset，说明图文联合初始化保存了有效训练信号。但 ASM 从 76.56 降到 73.05，表示当前 native ASM 优化没有稳定改善初始化，甚至可能破坏已有信号。

`Full-real=83.24` 与 `Frozen compact=76.56` 的差距属于静态压缩误差。Future 很低，是因为 stage 0 只训练类别 0、1，future 测的是尚未到达类别，并非当前主优化目标。

## 五、ASM 实际在优化什么，以及为什么会出问题

### 基本过程

每个 outer step 都做四件事：

1. 从当前 synthetic TAG 初始化一个新 student；
2. student 在 synthetic TAG 上做若干 inner-step 图文 InfoNCE 训练；
3. 把训练后的 student 放到真实 source 的无标签 reference roots 上；
4. 比较 student 与 teacher 的 graph-to-text response matrix，然后把误差反向传播到 synthetic `X`。

当前 topology、文本、weights 和代表来源 ID 都固定，只更新 `X`。

### 关键矛盾

ASM 只要求“用小图训练后的 student 在 reference 上像 teacher”，却没有直接要求“更新后的 synthetic graph root 仍然匹配原来那条文本”。因此 `X` 可能为了降低短期 ASM loss 而移动，固定文本却不动，graph-text pair 逐渐错位。

这就是后续文本归因和 Text-Anchored ASM 的出发点。

## 六、实验 3：最初的持续方法比较

### 在哪里

- 持续 runner：`run_native_experiments.py`、`run_native_attribution.py`
- 三阶段结果镜像：`results_mirror/preserve_r05_c44_3stage_sharedteacher/`、`results_mirror/preserve_r05_c45_3stage_sharedteacher/`
- 汇总说明：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md`

### 各方法是什么意思

- `Append`：旧 synthetic TAG 保持不动，再拼接当前 delta 的 compact block；
- `Union`：把旧 synthetic 与当前真实 delta 当作 source，重新凝聚整个累计 memory；
- `Append-joint`：旧 block 和新 compact block 拼接后，用 joint teacher 和 joint ASM 更新全部 X；
- `Protected-joint`：与 Append-joint 相同，但旧 X 冻结，只更新新 rows；
- `Full-history`：允许重读全部历史原图的诊断参考，不满足部署限制。

### stage-2 结果

| 方法 | 累计 test Macro-F1 |
|---|---:|
| Full-history | 49.37 |
| Append | 37.56 |
| Append-joint | 36.53 |
| Protected-joint | 37.44 |
| Union | 35.53 |

### 结果是什么意思

`Full-history` 比所有 compact 方法高，说明历史原图中仍有 synthetic memory 没保存下来的训练信息。

简单 Append 反而高于 Union 和 Append-joint，说明“再次优化整个小图”并不会自动变好。Protected-joint 接近 Append，提示冻结旧 X 能避免一部分改坏，但不能解决新增块质量和整体 response 问题。

这些结果提出了问题，却还不能区分究竟是旧 memory 差、新 block 差，还是旧新交互差。

## 七、实验 4：stage-2 输入组成归因

### 在哪里

- 报告：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/STAGE2_SOURCE_COMPOSITION_ATTRIBUTION_2026-09-07.md`
- runner：`run_native_stage2_attribution.py`
- Append-old：`results_mirror/stage2_source_composition_r05_c44_appendold/`
- Union-old：`results_mirror/stage2_source_composition_r05_c44_unionold/`

### 想回答什么

固定 stage-1 old memory 后，stage-2 掉点主要来自旧 memory，还是新增的 27 行 compact block？

### 怎么做

stage 2 累计真实训练根为 1891，5% 总预算为 94 行。旧 memory 已占 67 行，因此新 block 只有 27 行。比较：

- 全部 1891 行 full-history raw；
- 67 行 old memory only；
- 532 行 new raw only；
- old + 全部 532 行 new raw，共 599 行，属于超预算诊断；
- old + 随机 27 行 new raw，同预算；
- old + 27 行 new compact，同预算。

### Append old 的结果

| 输入 | 行数 | 累计 F1 | stage 1 F1 | stage 2 F1 |
|---|---:|---:|---:|---:|
| Full-history raw | 1891 | 51.16 | 73.35 | 39.52 |
| Old only | 67 | 38.70 | 68.09 | 14.95 |
| Old + new raw，超预算 | 599 | 47.39 | 70.75 | 33.39 |
| Old + new raw，同预算 | 94 | 40.08 | 65.59 | 16.97 |
| Old + new compact，同预算 | 94 | 42.98 | 68.79 | 27.47 |

### 结果是什么意思

在同样只给新增部分 27 行时，compact block 的 stage-2 F1 为 27.47，随机 raw 子集只有 16.97。说明 compact 初始化确实比随便留下 27 个真实节点有效。

但 old + new compact 的累计 42.98 仍低于超预算 old + 全量 new raw 的 47.39，也低于 full-history 51.16。说明有限预算的新 block 没有保存全部新增训练效用。

将 old memory 换成较差的 Union old 后，相同新 block 的累计 F1 从 42.98 降到 33.18，stage-1 F1 从 68.79 降到 39.25。这证明旧 memory 质量会把误差传递到下一阶段。

因此旧 memory 和新增 block 都有问题，不能只归因于其中一个。

## 八、实验 5：旧 X 是否应该冻结、新块是否应该继续 ASM

### 在哪里

- 报告：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/FINE_GRAINED_CONTINUAL_TAG_ATTRIBUTION_2026-09-07.md`
- old-X 运行：`results_mirror/fine_old_update_r05_c44_s2_v3/`、`fine_old_update_r05_c45_s2_v3/`
- new-quality 运行：`results_mirror/fine_new_quality_r05_c44_s2/`、`fine_new_quality_r05_c45_s2_same_raw/`

### 旧 X 更新对照

同一个旧 memory、同一个新初始化、同一个 teacher 和 reference 下比较：

- `Initial`：不做 ASM；
- `All-X`：旧、新 X 都更新；
- `New-X-only`：旧 X 精确冻结，只更新新 X。

| seed | Initial | All-X | New-X-only |
|---:|---:|---:|---:|
| 44 | 44.75 | 45.05 | 43.59 |
| 45 | 34.59 | 34.59 | 36.00 |

seed 44 支持更新旧 X，seed 45 支持冻结旧 X。两 seed 均值几乎相同。最后迭代中旧 X 的相对漂移达到 1.21–1.30，说明 ASM 确实大幅修改旧 memory，但修改是否有益取决于 seed 和选点。

### 新 block 质量对照

| seed | Raw-matched | Compact-frozen | New-only ASM | Joint ASM | Protected-joint |
|---:|---:|---:|---:|---:|---:|
| 44 | 40.08 | 44.66 | 42.87 | 44.69 | 43.45 |
| 45 | 32.33 | 34.59 | 35.47 | 34.59 | 36.00 |
| 均值 | 36.21 | 39.63 | 39.17 | 39.64 | 39.72 |

Compact-frozen 平均比 27 行 raw subset 高 3.42 pp，说明初始化有用。继续做 new-only ASM 平均反而低 0.46 pp；Joint 和 Protected 的均值也只比 frozen 高 0.01–0.10 pp。

结论是：问题不是“把旧 X 冻结”这样一个开关就能解决。当前 ASM 的更新方向和 outer 选择本身不稳定。

## 九、实验 6：centroid X 是不是主要问题

### 在哪里

- 报告：`D:/Adocs/简历/research/graph-condensation/TAGSAM/classical_work/native_tag_v1/DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md`
- 结果：`results_mirror/fine_x_c44/`、`results_mirror/fine_x_c45/`

### 怎么做

固定同一批 representative IDs、文本、权重和 post-decoder topology。只把新 block 的 `X` 从 cluster centroid 换成该 representative 的真实 X。

### 结果

| seed | centroid X | real representative X | real − centroid |
|---:|---:|---:|---:|
| 44 | 44.75 | 44.18 | −0.57 pp |
| 45 | 34.59 | 36.49 | +1.89 pp |

### 结果是什么意思

真实 representative X 在两个 seed 上一负一正。centroid X 可能影响单个 run，但不是跨 seed 稳定主因。仅仅把 centroid 换成真实节点特征不能成为下一版方法。

## 十、实验 7：固定 outer 轨迹，观察哪一阶段先坏

### 在哪里

- runner：`run_outer_trajectory.py`
- seed 44：`results_mirror/trajectory_joint_c44/trajectory.json`
- seed 45：`results_mirror/trajectory_joint_c45/trajectory.json`
- 报告：`DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md`

### 为什么要做

如果只看 dev 选中的一个 outer snapshot，会看不到优化过程。这里禁用 artifact selector，固定保存 outer 0、20、40、60、80、100，每个 snapshot 都用独立 student 训练。

### 关键结果

seed 44 中：

- stage 0：outer 0 的 82.96，到 outer 100 仍为 86.86；
- stage 1：69.99 降到 55.03；
- stage 2：33.53 到 31.62，中间也会波动。

seed 45 中：

- stage 0：76.05 下降到 68.80；
- stage 1：42.96 上升到 59.88；
- stage 2：35.86 到 36.59，中间波动。

### 结果是什么意思

不同历史阶段并不一起变坏。seed 44 是 stage 1 被改坏，seed 45 是 stage 1 被修复但 stage 0 受损。累计 old mean 会掩盖这种冲突。

这说明持续方法需要保持阶段条件 response，而不能只对全部旧节点施加一个同强度正则。

## 十一、实验 8：文本配对是否真的重要

### 在哪里

- 结果矩阵：`results_mirror/protocol_matrices_fixed_labels_2026_09_08/text_pairing.json`
- 对齐分析器：`analyze_text_alignment.py`
- 报告：`DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md`

### 文本置换实验

固定 synthetic `X`、edge、weights 和 origin IDs，只在同阶段内打乱 text，使文本分布不变但一一对应关系被破坏。覆盖两个 condensation seed，每个做三个置换，共六次。

六次置换的当前阶段 test Macro-F1 全部下降，平均下降约 11.20 pp。

这说明有用的不是“这张小图里存在一些文本”，而是“每个 graph root 对应哪一条文本”。这是 TAG 相对 plain graph 的核心差异。

### 无标签 teacher alignment

不读取标签，直接用冻结 teacher 测每个 outer snapshot 的 graph-text InfoNCE：

| seed | outer 0 | outer 100 |
|---:|---:|---:|
| 44 | 1.54 | 2.27 |
| 45 | 1.06 | 2.07 |

InfoNCE 越低表示 graph root 越容易找回自己的文本。outer 增加时两 seed 都明显恶化，证明 ASM 更新 X 的同时破坏了固定 graph-text correspondence。

## 十二、实验 9：Text-Anchored ASM

### 在哪里

- 方法实现：`native_core.py`
- 轨迹入口：`run_outer_trajectory.py`
- sweep：`results_mirror/text_anchor_sweep_2026_09_08/aggregate.json`
- 固定候选确认：`results_mirror/text_anchor_candidate_confirmation_2026_09_08/`
- 技术报告：`TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md`

### 方法

在 ASM 目标上加入：

\[
L=L_{ASM}+\lambda_tL_{text-anchor}.
\]

`L_text-anchor` 要求冻结 teacher 在 synthetic graph 上仍能把每个 graph root 与自己的固定文本匹配。梯度只更新 X。开发 sweep 从多个权重中选出 `lambda=0.01, scope=all`。

### 同 outer=60 的比较

这是机制对照：两种方法都走 60 个 outer step。

| 指标 | Text-Anchored − ASM | 正向 student 配对 |
|---|---:|---:|
| 累计 dev | +2.89 pp | 9/16 |
| 累计 test | +4.34 pp | 14/16 |
| 历史 test | +5.07 pp | 12/16 |
| 当前 test | +4.85 pp | 13/16 |

结果说明新增 loss 确实阻止部分文本漂移，并能改善 student。

### 各自 dev-best 的比较

实际完整方法应该允许 ASM 自己选择更好的 outer。零权重 ASM 的 dev-best 是 outer 20，Text-Anchored 候选是 outer 60。此时累计 test 增益从 +4.34 pp 缩小为 +1.47 pp，历史 dev 还下降 1.30 pp。

所以 +4.34 是“相同步数下这个 loss 有效”的机制证据，+1.47 才更接近公平模型选择后的方法收益。

## 十三、实验 10：严格 stage 1→2 递归

### 在哪里

- 完整目录：`results_mirror/recursive_text_anchor_2026_09_08/`
- 汇总：`results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json`
- 完整协议矩阵：`results_mirror/recursive_text_anchor_2026_09_08/recursive_protocol_matrix.json`
- artifact 选择：`select_trajectory_artifact.py`
- 独立 student：`train_saved_artifact_students.py`
- 汇总器：`summarize_recursive_text_anchor.py`

### 为什么这比固定 stage-2 对照更严格

固定 stage-2 实验给两种方法同一个 stage-1 memory，只测 stage-2 更新。严格递归则让两种方法从 stage 1 开始独立产生自己的历史，并且 stage 2 只能读取各自在 stage 1 用 dev 选出的 artifact。

### stage 1

两个 condensation seed、ASM 和 Text-Anchored 四条轨迹，全部选择 outer 0。选中的两种 artifact 实际内容相同。

12 个独立 student 的共同结果：累计 dev 53.99、累计 test 55.27、stage-0 test 78.15、stage-1 test 44.63、future stage-2 test 26.31。

最重要的含义是：在 stage 1，任何现有 ASM 更新都没有超过初始化。Text anchor 只能让部分下降变慢，不能产生一个值得选择的更新结果。

### stage 2

| seed | ASM dev-best | Text-Anchored dev-best |
|---:|---:|---:|
| 44 | outer 20，45.05 | outer 0，44.75 |
| 45 | outer 100，40.10 | outer 100，39.59 |

12 个确认 student：

| 指标 | Text-Anchored | ASM | 差值 |
|---|---:|---:|---:|
| 累计 test | 39.10 | 37.49 | +1.62 pp |
| 历史 test | 63.99 | 60.56 | +3.43 pp |
| 当前 test | 32.25 | 31.13 | +1.12 pp |

但累计 test 只在 7/12 student 配对中为正。按 condensation seed 分开后，seed 44 为 −0.84 pp，seed 45 为 +4.08 pp。

### 最终解释

平均值为正，说明 text anchor 有希望；跨 condensation seed 一负一正，说明它还不稳定。尤其 stage 1 完全没有接受任何优化，所以当前不能把 Text-Anchored ASM 称为最终持续算法。

## 十四、把全部证据合在一起

目前最符合证据的因果图是：

1. 静态凝聚先损失一部分 full-data 训练效用；
2. 旧 synthetic memory 的质量会沿递归路径传到下一阶段；
3. 新 delta 被压成很少的行后也会丢失信息；
4. ASM 同时更新新旧 X 时，不同阶段的 response 会发生冲突；
5. 因为文本固定，X 的移动还会破坏 graph-text pair；
6. dev outer 选择和 condensation seed 会决定最终接受哪一种冲突状态；
7. 单独修复 text alignment 有效，却不足以保持各阶段 teacher response。

现有证据不支持把全部问题归结为 centroid X，也不支持直接断言缺少跨阶段边是主因。

## 十五、现在进行到哪一步

已经完成：

1. 官方 TAGSAM 静态代码与七类协议复核；
2. 可审计的 native 静态基线；
3. Append、Union、Joint、Protected 和 full-history 持续对照；
4. 旧 memory、新 block、old-X、centroid-X 和 outer 轨迹归因；
5. 文本置换与无标签 teacher alignment；
6. Text-Anchored ASM 实现、权重 sweep、同 outer 确认和严格递归；
7. 独立 student、dev-only artifact 选择和 provenance 审计。

已经确认的核心 challenge 是 graph-text correspondence 漂移和阶段 response 冲突。尚未完成的是一个能在 stage 1 真正优于 outer 0，并在不同 condensation seed 上稳定的算法。

下一版原计划是在 text anchor 外加入无标签 teacher response-retention，但当前按要求暂停，没有实现或运行。正式实现前还要决定：保存 pairwise response 还是 class-prompt logits、历史 teacher 目标如何获得，以及额外状态是否计入 memory 预算。

