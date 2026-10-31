# Bounded Student-Compatible X-P-A 后续实验报告

日期：2026-09-30

## 1. 本轮目标

本轮围绕暂定主方法 **Bounded Student-Compatible X-P-A Revision** 完成四类检查：

1. 验证 X、P、A 是否真的收到 Student outer gradient；
2. 验证稀疏旧图 anchor 是否能改善旧类保持；
3. 验证放大新类 loss 是否能改善新类塑性；
4. 用统一的三数据集、base+4、5% memory 和 6 个 Student seeds 做 X、X+P、X+P+A 组件归因。

所有新实验均使用原始 Student：两层 GCN、可训练文本投影和双向图文 InfoNCE。实验构造只访问当前阶段 raw delta 与上一阶段 synthetic parent，不回读历史完整 raw graph。

## 2. 实验入口修复

本轮首先发现 runner 与大图 artifact 的协议包装不一致：

- `results.json` 的 frozen protocol 位于 `results.json["protocol"]`；
- source memory 文件名是 `xpa_s{stage}_memory.pt`，不是 `append_fixed_s{stage}_memory.pt`；
- base+4 评估需要累计已见类别，而不是只传当前新增类别；
- legacy evaluator 的全局类别表必须和 unified runner 同步。

这些问题已在 [run_bounded_xpa_large_20260929.py](/D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_bounded_xpa_large_20260929.py) 中修复。修复后的 stage-0 smoke 能完整生成 `results.json`，本地 [test_direct_student_outer.py](/D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/test_direct_student_outer.py) 为 **34 passed**。

## 3. 当前方法流程

### 3.1 X：class-aware feature revision

对当前阶段新增 synthetic rows 建立有界相对变量：

\[
X'_{i}=X_i + \Delta_i,
\qquad
\|\Delta_i\|/\|X_i\|\le r_{new}.
\]

本轮配置为 `x_new_radius=0.10`、`x_old_radius=0.05`。class-aware X 只更新新阶段 synthetic rows，并按 slot class 路由梯度。

### 3.2 P：payload transport

使用 `p_logits` 形成 Sinkhorn assignment，在固定行列质量约束下将当前支持数据的语义 payload 运输到新增 synthetic rows。默认配置为 `p_mix=0.25`、temperature `0.20`、24 次 Sinkhorn iteration。

### 3.3 A：bounded topology revision

A 使用候选跨块边池和 straight-through top-k 选择。默认每个新 synthetic row 最多修改一条边：

\[
\mathrm{revision\_count}(i)\le 1.
\]

边数、memory rows 和历史旧旧拓扑均保持不变。最终 materialization 后检查 degree、revision cap 和 edge budget。

### 3.4 Student-compatible outer objective

Student 先进行固定 inner unroll，再将 support 上的 old/new InfoNCE、margin、class representation、P balance 和 anchor 项反向传播到 X、P、A 参数。

正式完整模式的 outer gradient 变量审计为：

```text
['x_class_relative', 'p_logits', 'a_logits']
```

arXiv、BookChild、SportsFit stage 1 的首个 outer step 梯度均非零。例如：

| 数据集 | grad X | grad P | grad A |
|---|---:|---:|---:|
| arXiv | 0.1652 | 0.00269 | 0.00713 |
| BookChild | 0.2090 | 0.00263 | 0.02017 |
| SportsFit | 0.5575 | 0.00110 | 0.01086 |

因此当前实现确实在优化 X、P、A，而不是只做 append-only 启发式构造。

## 4. 稀疏旧图 anchor 配对

比较完全修复后的同一 runner、同一 source、同一 construction seed 和 6 个 Student seeds：

- control：`old_graph_anchor_weight=0`；
- sparse anchor：`weight=0.25, old_graph_anchor_fraction=0.25`；
- 只在候选图中选表示漂移最大的 25% 旧 rows 计算 anchor loss。

### 4.1 Stage 1 paired 结果

表中差值为 sparse anchor 减 control，单位为百分点。

| 数据集 | Macro-F1 | Accuracy | New-F1 | Old-F1 | 正向 Student 数 |
|---|---:|---:|---:|---:|---:|
| arXiv | +0.980 | +3.451 | +0.750 | +1.074 | 4/6 |
| BookChild | -1.106 | +2.103 | -1.320 | -0.644 | 1/6 |
| SportsFit | +0.205 | +1.096 | -0.179 | +0.385 | 4/6 |
| 三数据集平均 | +0.026 | +2.217 | -0.250 | +0.272 | - |

结论：稀疏 anchor 对 Accuracy 和部分旧类响应有帮助，但没有稳定提升 Macro-F1 或 New-F1，不能晋升为主方法。它应作为受限旧图保持机制的负结果和后续候选，而不是默认模块。

## 5. 新类 loss 权重实验

只修改：

\[
L_{new}\leftarrow \gamma_{new}L_{new}.
\]

测试 `gamma_new=1.5` 和 `2.0`，均使用 anchor=0、相同 source 和 seeds。

### 5.1 Stage 1 三数据集平均差值

| `gamma_new` | Macro-F1 | Accuracy | New-F1 | Old-F1 |
|---:|---:|---:|---:|---:|
| 1.5 | -0.046 | +0.957 | -0.077 | +0.528 |
| 2.0 | -0.855 | +0.103 | -0.852 | -0.261 |

结论：新类性能下降不是简单由 new loss 权重不足造成。继续增大该权重会损伤总体 F1 或 New-F1，因此不加入主方法。

## 6. X/P/A 组件消融

三种模式固定 memory、Student、seed、outer steps 和 source：

- **X**：`x-class`，只优化 class-aware X；
- **X+P**：`x-class+p`，增加 payload transport；
- **X+P+A**：`x-class+p+a`，完整模式。

### 6.1 Stage 1 汇总

| 数据集 | X Macro-F1 | X+P Macro-F1 | X+P+A Macro-F1 | X+P+A Accuracy |
|---|---:|---:|---:|---:|
| arXiv | 24.979 | 25.603 | 25.152 | 28.873 |
| BookChild | 16.762 | 16.481 | 17.139 | 11.118 |
| SportsFit | 30.132 | 28.688 | 30.941 | 29.075 |

X+P 相对于完整模式的差值为：

- arXiv：`+0.451 pp`；
- BookChild：`-0.657 pp`；
- SportsFit：`-2.253 pp`。

这表明 A 在 BookChild 和 SportsFit 的当前阶段有正向作用，但在 arXiv 上可能扰动了原有传播结构。X+P 模式没有启用完整模式的 candidate prompt margin，因此该表用于结构归因，不应被解读为纯 A 因果估计。

### 6.2 严格 topology-off 对照

为保持完整模式的所有 loss，只将 `a_max_revisions_per_row` 从 1 改为 0：

| 数据集 | 完整 X+P+A | cap=0 | 完整减 cap=0 | New-F1 差值 | Old-F1 差值 |
|---|---:|---:|---:|---:|---:|
| arXiv | 25.152 | 26.562 | -1.410 | -1.349 | -1.771 |
| BookChild | 17.139 | 14.314 | +2.825 | +3.176 | +2.477 |
| SportsFit | 30.941 | 33.395 | -2.454 | -2.779 | -5.797 |
| 三数据集平均 | 24.410 | 24.757 | -0.347 | -0.317 | -1.697 |

完整模式实际 revision 数量为：arXiv 25、BookChild 0、SportsFit 18。结果说明拓扑 revision 的价值具有数据集依赖性：它不是普适增益，且在当前 heuristic candidate pool 下可能损伤消息传播。

## 7. 三大图五阶段基线汇总

以下是已有完整 base+4、5% fixed-rate 结果。Student、report-dev 规则和 6 seeds 一致；raw-long 是允许访问全部历史 raw 的上界参考，不属于合法持续方法。

### 7.1 Final Macro-F1

| 方法 | arXiv | BookChild | SportsFit |
|---|---:|---:|---:|
| 当前 X-P-A | 18.525 | 11.853 | 30.257 |
| append-fixed | 18.817 | 13.216 | 27.898 |
| k-center | 16.667 | 11.712 | 27.840 |
| semantic-recursive | 19.227 | 13.313 | 28.076 |
| full-history re-condensation | 20.544 | 12.407 | 26.467 |
| raw-long reference | 20.155 | 17.217 | 29.502 |

当前 X-P-A 相对于 k-center 为 `+1.858/+0.141/+2.417 pp`，相对于 append-fixed 为 `-0.292/-1.363/+2.359 pp`。它在三图上优于强 coreset，但没有稳定超过 semantic-recursive 或 full-history。

### 7.2 Final Accuracy（已有基线 artifact 可重新聚合）

| 方法 | arXiv | BookChild | SportsFit |
|---|---:|---:|---:|
| 当前 X-P-A | 28.791 | 11.340 | 33.500 |
| append-fixed | 25.488 | 13.608 | 37.497 |
| k-center | 22.546 | 11.262 | 35.160 |
| semantic-recursive | 26.950 | 13.536 | 33.448 |
| full-history re-condensation | 31.739 | 14.373 | 32.913 |
| raw-long reference | 29.436 | 16.545 | 28.457 |

Accuracy 与 Macro-F1 不总是同方向，说明不能只用单一指标判断 X-P-A 是否有效。

## 8. 当前结论

### 8.1 可以确认的内容

1. X-P-A 是真正的 Student-compatible outer optimization，而不是只在结果命名上使用梯度；X、P、A 梯度均可审计。
2. 在固定 5% memory 下，当前方法相对 k-center 在三大图的最终 Macro-F1 均不差，并在 arXiv、SportsFit 有明显提升。
3. P 和 A 的作用不是普适的：P 的收益依赖数据集，A 的 revision 对传播结构可能有正也可能有负影响。
4. 稀疏 old-memory anchor 和简单 new-loss 放大都没有形成稳定主方法增益。
5. 当前最大瓶颈不是“有没有梯度”，而是可微候选空间和最终离散拓扑之间存在 construction-to-student mismatch。

### 8.2 不能宣称的内容

- 不能宣称稳定 SOTA；
- 不能宣称 X-P-A 普遍优于 append-only 或 semantic-recursive；
- 不能把 raw-long 作为合法持续基线；
- 不能把某一个数据集或某一个 seed 的正收益写成普遍规律。

## 9. 当前主方法决定

当前保留 **X-P-A 作为暂定主方法结构**，但不把 sparse anchor、new-loss scale 或无条件 topology revision 写成已验证增益。

论文中更稳妥的算法表述是：

> 在历史 raw 不可回读的持续 TAG 凝聚协议下，提出一种受限 Student-compatible X-P-A revision operator，通过短 Student unroll 将图文任务响应反馈到合成特征、语义 payload 和有界拓扑候选；同时通过 memory、访问边界、degree 和 revision cap 保证持续过程可审计。

实验贡献应强调：

- 统一的持续 TAG 凝聚协议；
- 对 X/P/A 梯度和访问约束的显式审计；
- 在强 coreset 和公开静态/全历史参考下的跨数据集比较；
- 对 anchor、new-loss、拓扑 revision 的系统失效边界分析。

## 10. 后续只保留一个优化方向

下一步不再继续扫描 anchor、loss scale 或 A temperature。最有价值的是实现一个**验证集约束的拓扑 revision fallback**：

1. 同时生成 cap=0 和 cap=1 两个候选；
2. 只用 sealed validation support loss 选择二者；
3. 不读取 report/test 标签；
4. 若 cap=1 没有改善 validation message/new-margin，回退 cap=0；
5. 保持 X、P、文本、Student 和 memory budget 不变。

这个实验直接针对本轮发现的 arXiv/SportsFit 拓扑退化问题，比继续调全局 `a_temperature` 更有解释性。若 fallback 在三个大图上仍不能稳定改善，则主方法应冻结为“X-P-A operator with bounded topology fallback”，并进入论文写作，而不是继续叠加模块。

## 11. 主要 artifact

- 稀疏 anchor：`results/bounded_xpa_anchor_pair_sparse6_20260930b_*`
- corrected control：`results/bounded_xpa_anchor_pair_control6_20260930_*` 和 `control6b_20260930_sportsfit`
- new loss 1.5：`results/bounded_xpa_newscale15_20260930_*`
- new loss 2.0：`results/bounded_xpa_newscale20_20260930_*`
- X-only：`results/bounded_xpa_ablation_x_20260930_*`
- X+P：`results/bounded_xpa_ablation_xp2_20260930_*`
- topology cap=0：`results/bounded_xpa_ablation_noa_20260930_*`
- 完整 X-P-A：`results/bounded_xpa_large_full_20260930_*`

## 12. Validation-gated topology fallback（2026-09-30）

### 12.1 实验目的与固定协议

本轮只检验一个问题：在同一份已经完成 X/P outer 优化的候选上，是否应该保留最多一次的 A 拓扑 revision，还是退回 `cap=0` 的原始候选拓扑。两种候选共享：

- 相同 parent、当前 raw delta、X、P、文本和 memory rows；
- 相同 Student、inner/outer steps、batch、construction seed 和 6 个 report Student seeds；
- 相同 5% fixed-rate、base+4 大图协议和 report-dev 掩码；
- 不访问历史完整 raw graph，不使用 report/test 标签。

门控只读取 construction 阶段独立的 sealed validation support response。它记录 weighted support loss、old response 和 new-class margin。固定规则为：old response 不得变差；并且 support response 或 new margin 至少有一项超过预注册的最小改善，同时另一项不超过固定容差。最终 F1 不参与选择。

实现入口为 [run_bounded_xpa_large_20260929.py](/D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_bounded_xpa_large_20260929.py)，正式运行使用 `--topology-gate --student-graph-mode edge-aware`。每个 stage 保存 `candidate_cap1_revision.pt`、`candidate_cap0_control.pt`、最终 `memory.pt` 和 `topology_gate.json`。

### 12.2 Sealed validation gate 的选择结果

| 数据集 | stage | support loss 差值（cap1-cap0） | old response 差值 | new margin 差值 | 选择 |
|---|---:|---:|---:|---:|---|
| arXiv | 1 | -0.000845 | +0.000267 | -0.023367 | cap=0 |
| BookChild | 1 | -0.000281 | +0.000399 | -0.015107 | cap=0 |
| SportsFit | 1 | -0.001529 | -0.000865 | -0.002994 | cap=1 |

因此本轮 gate 只在 3 个 stage-1 cell 中接受 1 个 revision。arXiv 和 BookChild 的 revision 确实改善了 new margin，但同时使 old validation response 微弱变差，严格 old-protection 规则选择回退；SportsFit 同时改善 support、old response 和 new margin，接受 revision。

### 12.3 Edge-aware Student 的正式结果

这张表来自正式 edge-aware runner 的最终选择 memory。它使用原始两层 GCN、双向 InfoNCE Student，并直接使用 memory 中的 `edge_index`；不是换 Student。

| 数据集 | stage | 选择拓扑 | Macro-F1 | Accuracy | Old F1 | New F1 |
|---|---:|---|---:|---:|---:|---:|
| arXiv | 0 | C0 | 29.6182 | 26.8681 | - | 29.6182 |
| arXiv | 1 | cap=0 | 23.8774 | 25.3199 | 27.8382 | 25.1755 |
| BookChild | 0 | C0 | 18.3566 | 12.7322 | - | 18.3566 |
| BookChild | 1 | cap=0 | 15.4391 | 11.3393 | 17.2384 | 16.7860 |
| SportsFit | 0 | C0 | 27.3262 | 22.8543 | - | 27.3262 |
| SportsFit | 1 | cap=1 | 30.4205 | 29.1351 | 30.5737 | 32.3689 |

stage 0 只作为同一运行的基准，不用于评价 gate。stage 1 的最终选择结果显示：当前 gate 的行为是数据集依赖的，SportsFit 保留 revision 后 new F1 较高；另外两个数据集为了保护 old response 回退到冻结拓扑。

### 12.4 直接 cap=1/cap=0 的 paired downstream probe

为判断 proxy 是否与下游一致，另对保存的两个候选使用相同 6 个 Student seeds 做 edge-aware report-dev 评估。单次 probe 的差值如下：

| 数据集 | cap=1 Macro-F1 | cap=0 Macro-F1 | cap=1-cap=0 | cap=1 Accuracy | cap=0 Accuracy | Accuracy 差值 |
|---|---:|---:|---:|---:|---:|---:|
| arXiv | 26.1249 | 25.4243 | +0.7006 | 31.6302 | 28.4301 | +3.2001 |
| BookChild | 16.8342 | 16.0249 | +0.8093 | 11.7401 | 12.4873 | -0.7473 |
| SportsFit | 28.4078 | 27.3077 | +1.1001 | 23.8233 | 23.5036 | +0.3197 |

这张表只能作为候选方向性诊断，不能替代多次独立重复：GPU sparse message passing 在同一 memory、同一 seed 的重复评估中仍出现可见数值漂移。因此，当前证据支持“拓扑 revision 可能改善某些数据集的下游响应”，但不支持把无门控 cap=1 宣称为稳定增益。

### 12.5 本轮发现的协议问题

旧版 `U.run_students` 通过 `memory_to_trainable` 按 text embedding 重新构图，会忽略保存的 `memory["edge_index"]`。如果用该 legacy adapter 评估 A，cap=1/cap=0 的拓扑差异会被抹掉，不能用于拓扑结论。本轮因此增加显式 `student-graph-mode=edge-aware`；此前 legacy X-P-A 大图 aggregate 仍保留为历史兼容结果，但不能和本节的 topology-aware 结果拼成同一主表。

这也解释了为什么此前 A 的下游收益看起来不稳定：一部分不稳定来自候选拓扑本身，另一部分来自 construction Student 与 downstream Student 使用了不同的图算子。后续所有涉及 A 的 baseline 必须统一使用 edge-aware Student，否则只能比较 X/P，不应比较拓扑。

### 12.6 结论与下一步

1. validation gate 已完成并可审计，但接受率为 `1/3`，不能作为已经验证的主方法增益。
2. A revision 有方向性正信号，尤其在 SportsFit；但 old/new trade-off 和 GPU 数值稳定性仍需独立重复。
3. 当前推荐主方法仍是 **bounded X-P-A with optional validation-gated topology fallback**，其中 gate 是保守安全机制，不把 cap=1 强制写入所有数据集。
4. 若要把拓扑 revision 写进论文主贡献，下一轮必须把 random coreset、k-center、append-fixed、静态 TAG baseline 和当前方法全部用同一个 edge-aware Student 重跑至少 3 个顺序/seed；否则应把 A 降级为诊断组件，主表只报告 X/P。
5. 还需补一轮确定性控制：固定 `CUBLAS_WORKSPACE_CONFIG`、记录 GPU/torch 版本，并报告同一候选的重复方差或置信区间。不能用单次 F1 差值作为最终晋升依据。

### 12.7 新增 artifact

- legacy gate smoke：`tmp_remote_results/xpa_topology_gate_large_20260930/`
- edge-aware 正式结果：`results/xpa_topology_gate_edgeaware_large_20260930_{arxiv,bookchild,sportsfit}/`（远程服务器）
- edge-aware candidate evaluator：[evaluate_bounded_xpa_gate_candidates_20260930.py](/D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/evaluate_bounded_xpa_gate_candidates_20260930.py)

