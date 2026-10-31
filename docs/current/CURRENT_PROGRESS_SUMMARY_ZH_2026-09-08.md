# 经典持续 TAG 图凝聚：截至 2026-09-08 的进展总结

日期：2026-09-08  
状态：阶段性总结；已暂停新方法实现与新实验  
证据边界：Cora 的 test、confirmation 和 future 分区已经在开发过程中访问，本文涉及的这些结果只能作为开发证据，不能作为论文最终未见测试结果

## 1. 当前结论

当前选题已经从早期原型 memory、GraphCLIP 适配和多种下游分支，收敛到一个明确问题：

> **在类别增量文本属性图中，当历史原图不可重读、只能保存累计预算内的 synthetic TAG 时，如何递归更新这张小图，同时保持旧阶段训练响应、吸收新类别，并避免图结构表征与对应文本逐阶段错位。**

截至目前，问题可行性已经成立，第一版机制也已经实现并完成严格递归验证，但最终算法尚未成立：

1. 官方 TAGSAM 静态链路已经重新核对和运行，但七类 Cora 上仍明显低于 full-real teacher，静态凝聚本身存在误差；
2. 持续凝聚相对 full-history 和简单 Append 还存在额外损失，这些损失不能只用静态误差解释；
3. 已经确认一个 TAG 特有瓶颈：ASM 更新 synthetic `X` 时会破坏图根与其固定文本之间的对应；
4. 已实现 `Text-Anchored ASM`，在固定 stage-2 条件下有效，但严格 stage 1→2 递归后没有跨 condensation seed 稳定胜出；
5. 当前最关键的未解问题是 stage 1 中所有 ASM 轨迹都选回 `outer=0`，说明“开始优化反而不如初始化”；
6. 下一版方法应联合保持 graph-text correspondence 和 teacher response。该方法尚未实现，也没有启动新实验。

因此，当前研究成熟度可以概括为：**协议已固定，静态起点已审计，持续损失已细分，TAG 文本漂移已被因果对照确认，方法 v1 已验证但未通过稳定性门槛，方法 v2 尚待设计。**

## 2. 现行研究协议

### 2.1 数据到达方式

当前采用类别增量流：

- stage 0：类别 `[0,1]`；
- stage 1：新增类别 `[2,3]`；
- stage 2：新增类别 `[4,5,6]`。

这是一版用于建立可解释因果链的开发 setting，并不意味着未来论文只能使用类别增量。选择它的原因是：类别边界明确，旧类保持和新类塑性可以分开报告，也与 CaT 等持续图学习协议更容易对照。时间增量可以在方法稳定后作为第二类 stream 扩展。

### 2.2 持续访问约束

阶段完成后：

- 历史原始训练图不可再次读取；
- 只能保留上一阶段导出的 synthetic TAG；
- memory 根节点数服从累计训练节点数乘固定压缩率的预算；
- 下一阶段的合法输入只有旧 synthetic TAG、当前真实 delta、类别名称和全局文本提示；
- 凝聚 artifact 不含节点标签，teacher、student 和凝聚损失均不直接读取训练节点标签；
- checkpoint/dev 标签用于选择 teacher、student checkpoint 和 synthetic outer snapshot。

当前是“固定压缩率、随累计数据增长的 memory”协议，还不是固定总字节容量协议。真实 artifact 字节数和文本字节数会另外记录，防止只用节点数掩盖存储差异。

### 2.3 下游模型

当前下游固定为 TAGSAM 风格接口：

- 冻结 `bert-base-uncased` 的 CLS 文本 embedding；
- 两层 GCN 产生 graph embedding；
- 可训练文本投影头产生 text embedding；
- 图与文本经过归一化，用双向 graph-text InfoNCE 训练；
- 推理时把节点 graph embedding 与七类 class-prompt embedding 比较，执行文本提示分类。

所以当前下游任务是**节点分类**，更准确地说是使用文本类别提示完成的零样本文本条件节点分类。虽然 graph encoder 是 GCN，TAG 的文本并没有被忽略：每个 synthetic graph root 都必须与自己的文本形成正对，其余文本形成对比负例；分类又直接依赖 class prompt。文本置换实验已经证明这一接口对配对关系高度敏感。

GraphCLIP 目前暂停。原因不是它没有研究价值，而是当前问题首先要在 TAGSAM 原生接口上建立可靠的静态起点和持续归因。如果同时改变凝聚目标与下游模型，失败时无法判断来自持续凝聚还是 GraphCLIP 适配。

## 3. 评测到底在哪些节点上进行

当前并非只报告一个“所有节点准确率”，也不是只测试新增节点。每个阶段训练完成后同时报告以下口径：

| 口径 | 节点范围 | 回答的问题 |
|---|---|---|
| 当前阶段 | 本阶段新到达类别对应的 held-out mask | 新知识是否学好 |
| 历史阶段 | 每个旧阶段各自的 held-out mask，再取阶段均值 | 旧知识是否保持 |
| 累计 seen | 将所有已经到达阶段的 held-out 预测与标签合并 | 当前总体效用 |
| 分阶段 | 分别列出 stage 0、1、2 | 避免累计均值掩盖相反方向 |
| Future | 尚未到达阶段的 held-out mask | 对未来类别的迁移能力，仅作冻结后诊断 |

每个阶段还区分：

- train：原始训练图上的诊断；
- dev/checkpoint：用于训练 checkpoint 和 outer artifact 选择；
- test/confirmation：候选冻结后由独立 student 读取；
- future：模型和 artifact 冻结后读取未来阶段。

评估图采用 whole-stage transductive message passing，再只在对应 mask 上计算指标。这意味着评测标签没有进入训练，但图结构和节点输入在该阶段内可参与消息传播；当前并非严格的节点归纳评测。

主指标是固定 label space 下的 Macro-F1，同时记录 Accuracy。累计、历史、当前和 future 的 label space 已显式修复，避免不同阶段因缺类而使用不同隐式类别集合。

## 4. 为什么先核对静态 TAGSAM，以及静态结果如何

持续更新的每个阶段都要调用静态凝聚算子。如果静态算子本身不能产生可独立重训的小 TAG，那么持续下降既可能来自静态压缩，也可能来自递归更新，二者无法分开。因此静态起点不是把研究改成静态任务，而是持续归因所需的受控组件。

这并不等于放弃 CaT 或 DANCE。当前做法是：用 CaT/PUMA 提供“历史不可重读、按阶段更新 memory”的持续框架，用 TAGSAM 提供“图与文本共同训练的 synthetic TAG”接口，然后研究二者交叉处的新损失。直接套用 CaT 会缺少 TAG 文本对应机制；直接套用 DANCE 更接近持续 learner 的 replay/正则，而当前目标是输出一个可以让任意独立 student 从头训练的 synthetic TAG artifact。

### 4.1 官方 TAGSAM 复核

官方代码已实际运行，而不是被跳过：

- official master 的 Cora evaluator 写死为五类，并可能复用陈旧五类 cache；
- add-adj 分支确实保存并在评估时读取 learned synthetic topology；
- 清理 cache、改为七类后，add-adj 的五次 student Test Accuracy 从 step 0 的 `38.58%` 提高到 step 500 的 `40.23%`；
- minimal repair 只修尾批、设备/训练状态和七类协议，在 condensation seeds 44/45 上分别从 `33.40→36.24%`、`33.81→38.74%`；
- 七类 full teacher 的独立结果为 `61.12%`。

这些数值是 Accuracy，不能与下方 native 协议的 Macro-F1 直接横比。结论是：官方静态优化在两个 minimal-repair seed 上方向为正，add-adj 也有小幅效用，但离 full teacher 仍远，且 student 方差较大。

### 4.2 为什么后续没有直接把官方 master 当持续核心

官方 master 不能直接承担现行持续实验，主要因为：

1. 原始 Cora evaluator 是五类而当前研究是七类类别流；
2. master 默认 synthetic self-loop，add-adj 又属于不同 topology 分支；
3. 旧 cache 会使七类修改静默回退成五类任务；
4. 官方运行没有现行的 train/dev/test/future 分阶段矩阵、历史访问审计、累计预算和 recursive provenance；
5. 持续实验需要导出 artifact 后由独立 student 从头重训，并显式选择 outer snapshot。

因此实现了一个经过审计的 TAGSAM-derived native 后端。它保留核心 GCN、冻结 BERT CLS、文本投影、双向 InfoNCE 和 ASM 思路，同时加入持续协议、独立 student、provenance、预算与分阶段评估。它不是论文代码的逐行等价复现，报告中一直明确区分二者。

### 4.3 native 静态基线

stage-0、5% 压缩率下的 Macro-F1 为：

| 方法 | Train | Dev | Test | Future |
|---|---:|---:|---:|---:|
| Full-real | 80.73 | 84.90 | 83.24 | 21.89 |
| Random-small | 67.64 | 73.73 | 69.73 | 17.11 |
| Coreset | 64.72 | 68.66 | 62.22 | 12.31 |
| Frozen compact | 72.11 | 77.84 | 76.56 | 11.75 |
| Offline ASM | 68.22 | 76.50 | 73.05 | 15.80 |

小 TAG 明显具有训练效用，但现有 ASM 不稳定优于冻结初始化。静态凝聚误差是真实存在的背景误差，不能把持续模型与 full-real 的全部差距都归因于持续更新。

## 5. 持续凝聚基线与损失位置

已经实现和比较的主要持续构造包括：

- `Append`：旧 synthetic TAG 与当前新 compact block 直接拼接；
- `Union`：旧 memory 与当前 delta 组成合法 source，再凝聚到累计预算；
- `Append-joint`：从旧 block 与新 compact block 初始化，使用 joint teacher 和 joint ASM，所有 `X` 可更新；
- `Protected-joint`：与 Append-joint 相同，但冻结旧 `X`，只更新新增 rows；
- `Full-history`：允许读取完整历史原图的非部署上界；
- `Text-Anchored ASM`：在 joint ASM 上增加图文对应约束。

stage-2 的已有开发证据为：

| 方法 | 累计 test Macro-F1 |
|---|---:|
| Full-history | 49.37 |
| Append | 37.56 |
| Append-joint | 36.53 |
| Protected-joint | 37.44 |
| Union | 35.53 |
| 严格递归 Text-Anchored ASM | 39.10 |

这些数字表明存在两层差距：

1. **静态压缩差距**：full-real 或 full-history teacher 到静态 synthetic TAG 的下降；
2. **持续递归差距**：同样使用 compact memory 时，旧 synthetic 与新 delta 再次合并、更新和选择造成的额外下降。

当前研究重点是第二层。静态算子不完美会限制最终上限，但无需先完全解决静态 TAGSAM 才能研究持续特有误差；关键是使用同一静态起点做配对对照。

## 6. 已完成的细粒度归因

### 6.1 新 block 的 centroid `X` 不是稳定主因

固定 representative ID、文本、权重和 topology，只把新增 block 的 centroid `X` 换成真实 representative `X`。两个 condensation seed 的累计 dev 变化为 `−0.57 pp` 和 `+1.89 pp`，方向相反。

因此，新增特征聚合可能影响个别 seed，但不能解释稳定的持续下降。

### 6.2 不能把问题简单归为“旧图忘记”或“新图没学好”

固定 outer 轨迹显示不同阶段会向相反方向变化：

- seed 44 中 stage-1 Macro-F1 从 outer 0 的 `69.99` 降到 outer 100 的 `55.03`，而 stage 0 没有同步下降；
- seed 45 中 stage 1 从 `42.96` 上升到约 `59.88`，但 stage 0 会下降；
- current stage 也会随 outer 上升或下降，没有稳定单调收益。

所以“历史保持”不是一个同质变量。必须逐阶段报告，区分旧 memory response 漂移、新阶段塑性不足以及二者之间的冲突。

### 6.3 TAG 文本对应漂移已经被确认

两类独立证据指向同一结论。

第一，只在阶段内置换 synthetic text，保持 `X`、topology、weights 和 origin IDs 不变。`2` 个 condensation seed × `3` 个置换 seed 共六次，当前阶段 test Macro-F1 全部下降，平均约 `11.20 pp`。

第二，使用冻结 joint teacher 做无标签 alignment 诊断。原 ASM 从 outer 0 到 100 时，全体 graph-text symmetric InfoNCE：

| condensation seed | outer 0 | outer 100 | 恶化 |
|---:|---:|---:|---:|
| 44 | 1.54 | 2.27 | +0.73 |
| 45 | 1.06 | 2.07 | +1.01 |

文本、权重、origin IDs、stage 和排序后的边集合均未改变，因此漂移来源是 synthetic `X` 更新改变了 graph response，使它逐渐偏离固定文本。

### 6.4 尚未确认跨阶段边缺失是主因

现有证据已经可以用 response 冲突和图文错位解释明显损失。跨阶段边仍可能有帮助，但目前没有因果对照证明它是主要瓶颈。现在加入 learned cross-stage edges 会同时改变消息传播、负例关系和存储预算，反而降低可归因性。

## 7. 第一版方法：Text-Anchored ASM

方法目标为：

\[
L = L_{ASM} + \lambda_t L_{text-anchor}.
\]

`L_text-anchor` 使用冻结 joint teacher，让 synthetic graph response 与每个 root 自己的冻结文本继续满足 symmetric InfoNCE。实现中 teacher、文本、topology、weights 和 origin IDs 固定，梯度只进入 synthetic `X`。当前筛选出的开发配置为 `lambda_t=0.01`、`scope=all`。

### 7.1 固定 stage-2、相同 outer 的机制结果

在 outer=60、16 个独立 student 配对下，相对 `lambda=0`：

| 指标 | 变化 | 正向配对 |
|---|---:|---:|
| 累计 dev | +2.89 pp | 9/16 |
| 累计 test | +4.34 pp | 14/16 |
| 历史 test | +5.07 pp | 12/16 |
| 当前 test | +4.85 pp | 13/16 |

这证明新增损失确实降低了图文漂移，并且能改变独立 student 的分类响应。

### 7.2 与 ASM 各自 dev 最优点的公平比较

零权重 ASM 自己的 dev 最优点是 outer=20。候选 outer=60 相对该基线：累计 dev `+0.44 pp`、累计 test `+1.47 pp`、当前 test `+3.47 pp`，但历史 dev `−1.30 pp`。

因此，同步数比较中的大幅收益不能直接当作完整方法收益。允许每个方法独立选点以后，收益缩小且主要集中在当前阶段。

## 8. 严格 stage 1→2 递归结果

严格递归不是拿同一个 stage-1 memory 比较 stage 2，而是每种方法独立经历完整历史：

1. 从同一 stage-0 synthetic TAG 出发；
2. stage 1 只读取旧 artifact 与 stage-1 delta；
3. student seeds `100/101` 只按累计 dev 选择 outer `0/20/40/60/80/100`；
4. 只把选中的 stage-1 artifact 传入 stage 2；
5. stage 2 再按同一规则选择；
6. student seeds `102–107` 独立重训后读取 train/dev/test/future。

### 8.1 stage 1

两个 condensation seed 下，ASM 和 Text-Anchored ASM 全部选择 `outer=0`。选中图在 `X`、文本、权重、来源 ID、stage 和排序后边集合上完全一致。12 个确认 student 的共同结果为：

| 指标 | Macro-F1 |
|---|---:|
| 累计 dev | 53.99 |
| 累计 test | 55.27 |
| stage-0 test | 78.15 |
| stage-1 test | 44.63 |
| future stage-2 test | 26.31 |

该结果直接指出当前最大障碍：文本锚定可以减缓部分 outer 的下降，却不能使任何优化后的 stage-1 artifact 超过初始化。

### 8.2 stage 2

开发选择点为：

| condensation seed | ASM | Text-Anchored ASM |
|---:|---:|---:|
| 44 | outer 20，dev 45.05 | outer 0，dev 44.75 |
| 45 | outer 100，dev 40.10 | outer 100，dev 39.59 |

在 12 个独立确认 student 上：

| 指标 | Text-Anchored | ASM | 差值 | 正向配对 |
|---|---:|---:|---:|---:|
| 累计 dev | 40.78 | 38.86 | +1.91 pp | 7/12 |
| 累计 test | 39.10 | 37.49 | +1.62 pp | 7/12 |
| 历史 test | 63.99 | 60.56 | +3.43 pp | 9/12 |
| 当前 test | 32.25 | 31.13 | +1.12 pp | 8/12 |

但累计 test 的 condensation-seed 分解为：seed 44 `−0.84 pp`，seed 45 `+4.08 pp`。因此均值为正仍不足以称为稳定算法。

## 9. 当前已成立、未成立和仍待回答的内容

| 判断 | 状态 | 证据 |
|---|---|---|
| 类别增量持续 TAG 凝聚存在研究空间 | 已成立 | compact 方法与 full-history 仍有约 10 pp 差距 |
| 静态 TAGSAM 自身存在误差 | 已成立 | 官方七类 synthetic 显著低于 full teacher；native ASM 不稳定优于 frozen |
| 持续损失全部来自静态凝聚 | 未成立 | 递归更新引入阶段 response 冲突和图文漂移 |
| centroid `X` 是主要原因 | 当前不支持 | 两个 condensation seed 方向相反 |
| 旧图整体被遗忘 | 过度简化 | 不同旧阶段随 outer 呈相反变化 |
| 新图一直没有凝聚好 | 过度简化 | current 指标随 outer 和 seed 可升可降 |
| 图文对应漂移是独立瓶颈 | 已成立 | 六次文本置换和无标签 teacher alignment 一致 |
| 跨阶段边缺失是主因 | 尚未确认 | 缺少单因素因果对照 |
| Text-Anchored ASM 有机制效用 | 已成立 | 同 outer 配对和 alignment 指标明显改善 |
| Text-Anchored ASM 是最终方法 | 未成立 | stage 1 选择 outer=0，严格递归不跨 seed 稳定 |
| GraphCLIP 适配已经必要 | 尚未成立 | 当前原生接口仍有未解决的核心凝聚问题 |

## 10. 选题与创新点应如何表述

当前最稳妥的题目方向是：

> **面向类别增量文本属性图的持续图凝聚：递归更新中的图文对应与训练响应协同保持。**

潜在贡献由三部分组成：

1. **问题与协议贡献**：定义历史原图不可重读、仅传递 synthetic TAG 的类别增量凝聚协议，并建立当前、历史、累计、future 的分阶段评测；
2. **诊断贡献**：把总误差分成静态压缩、旧 memory response、新阶段塑性和 graph-text correspondence，并证明文本错位是 plain graph 方法没有显式处理的 TAG 瓶颈；
3. **算法贡献**：联合约束图文对应与历史/当前 teacher response，使递归更新后的 synthetic TAG 能由独立 student 从头重训，并随着 stage 增加保持稳定。

与已有工作的差异应集中在：

- **相对 TAGSAM**：从一次性静态 TAG 凝聚扩展到历史不可重读的递归更新，并处理阶段误差累积；
- **相对 CaT**：关注 graph-text pair、文本条件 response 和独立可重训 TAG，而不是只延续 plain graph 的持续凝聚流程；
- **相对 DANCE**：目标产物是一个可导出的 synthetic TAG memory，student 可以从头训练，不依赖持续 learner 自身保存旧参数或 replay 正则状态。

目前前两项贡献已有较强证据，第三项只有第一版机制证据，尚未达到论文主算法门槛。

## 11. 当前代码与可复核产物

主要实现：

- `native_core.py`：GCN/BERT-text 接口、InfoNCE、ASM、Text-Anchored loss；
- `run_outer_trajectory.py`：固定 outer 快照和多 student 轨迹；
- `select_trajectory_artifact.py`：只用 dev 选择 artifact，并恢复递归 provenance；
- `train_saved_artifact_students.py`：从冻结 artifact 独立重训；
- `summarize_recursive_text_anchor.py`：严格递归汇总与 condensation-seed 分解；
- `evaluate_protocol_matrix.py`：train/dev/test/future 与固定 label space 矩阵。

主要证据：

- 官方 TAGSAM 静态复核：`../../OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md`；
- 细粒度归因：`DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md`；
- Text-Anchored ASM 技术报告：`TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md`；
- 严格递归汇总：`results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json`；
- 严格递归完整矩阵：`results_mirror/recursive_text_anchor_2026_09_08/recursive_protocol_matrix.json`；
- 自包含可交互报告：`reports/text_anchored_asm_2026_09_08/report.html`。

现有全目录测试最近一次为 `38/38 passed`。本总结只整理既有证据，没有修改算法，也没有运行新实验。

## 12. 下一步计划（当前暂停，尚未执行）

下一版最小目标是：

\[
L=L_{ASM}+\lambda_tL_{text-anchor}+\lambda_rL_{response-retention}.
\]

`response-retention` 应在无标签条件下直接保持 teacher 的 pairwise response 或 graph-to-class-prompt logits：旧 rows 约束历史 response，新 rows 对齐当前 delta response。正式实现前还需要冻结“保存哪一种 response、目标 teacher 从何而来、是否计入 memory 预算”三个协议细节，防止隐式保存上一阶段完整模型而破坏只保留 synthetic TAG 的主约束。

后续实验顺序保持为：

1. 只做 stage-1 gate；
2. 两个 condensation seed 都必须选择 `outer>0`；
3. 两个 seed 的累计 dev 都不得低于各自 outer=0；
4. 通过后固定权重，执行完整 stage 1→2 dev-only 递归；
5. 再用独立 student 读取 test，并报告配对胜率和 seed 分解；
6. 通过后扩展更多 condensation seed、第二个 TAG 数据集和新的未访问最终留出。

在 stage-1 gate 通过前，不加入 GraphCLIP、transport、跨阶段边、额外 prompt 或动态预算。这些方向保留为后续扩展，不用于掩盖当前核心更新失败。

