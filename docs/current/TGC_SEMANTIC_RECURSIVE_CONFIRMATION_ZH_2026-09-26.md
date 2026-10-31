# Semantic Conserved Payload Assignment 递归链路确认报告

日期：2026-09-26  
状态：已完成 Cora/Citeseer 两个新顺序、两阶段的真正递归 parent 确认；该版本满足当前小图晋升门槛，可进入大图 base+4 验证。

## 1. 为什么要补这轮实验

此前 fresh role-aware 评估的 stage2 都从 `c0_s1_memory` 作为 parent，能够验证“某个阶段内 semantic assignment 是否有效”，但不能回答完整持续任务中的关键问题：

> stage1 已经使用 semantic permutation 后，stage2 是否能够继续在这个 synthetic memory 上追加新数据并保持收益？

本轮只增加一个协议修正：stage2 的 semantic candidate 读取 stage1 的 `semantic_perm_memory` 作为历史 parent。没有加入新的 selector、role/message/plasticity cost、dynamic budget、GraphGPS、文本 attention 或 Student 改动。

## 2. 当前确认的方法

当前确认的方法可以写成：

`C0 topology + two-prototype structural proposal + semantic conserved payload assignment + recursive append + original Student`

每个阶段的完整操作如下：

1. **保留上一阶段 synthetic parent**：stage1 的 parent 是 C0 stage0；stage2 的 parent 是已生成的 semantic stage1 memory。历史 raw graph 不回读。
2. **读取当前合法 delta**：只读取当前阶段新到达的原始节点、属性和文本 embedding。
3. **使用 C0 的当前阶段结构 proposal**：按冻结的 two-prototype split/merge 构造新 block，生成新的合成节点属性、文本表示、质量/来源记录和当前阶段 topology。
4. **固定 topology**：候选使用当前 C0 的 `edge_index`；semantic assignment 不改变边、度数或旧边。
5. **类内 semantic assignment**：在每个新类内部，将 structural proposal 中的 payload rows 与当前 class slots 的 graph-text joint representation 做一对一最小代价匹配。它只移动完整 payload，不重新平均、不复制、不删除来源。
6. **递归 append**：将上一步得到的新 block 追加到上一阶段 semantic parent。旧 synthetic rows 的 `x`、text embedding、文本、mass、depth 和 stage 保持不变；只新增当前 block。
7. **Student 重训**：使用原始两层 GCN、可训练 text projection 和双向图文 InfoNCE，从头训练 Student；batch=16，seeds=150--155，步数使用 `protocol_v3.student_steps(rows)`。
8. **评估**：使用真实 report-dev 图和类别 prompt 做 all-prompt cumulative/old/new Macro-F1。report/test 标签不参与构造或选择。

这里的“semantic conserved payload”不是重新学习或优化 feature，而是对已经构造好的完整 `(x, text, mass, source record)` 单元进行类内守恒重排。它的作用是让新 block 中不同语义角色的内容落到更匹配的 C0 structural slot，同时不改变 topology 和 memory budget。

## 3. 递归确认协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 新类别顺序 | `order19611`、`order19612` |
| 阶段 | stage 1、stage 2，共 8 个 cells |
| Student | 原始两层 GCN + trainable text projection |
| 目标 | 双向图文 InfoNCE，不使用节点分类监督 |
| batch | 16 |
| Student seeds | 150--155 |
| steps | 每个 cell 的 `protocol_v3.student_steps(rows)` |
| memory | C0 与 semantic recursive 两个 arm；同 cell、同预算 |
| topology | 当前阶段 C0 topology，semantic arm 与 C0 完全相同 |
| 评价 | all-prompt report-dev cumulative/old/new Macro-F1 |
| 历史 raw graph | 0 次 |
| report/test 标签 | 不用于构造、匹配或选择 |

## 4. 结果

所有差值相对同一 cell 的 C0，单位为百分点；每个 cell 内先对 6 个 Student seed 求均值，再在 cells 上等权平均。

| Cell | cumulative F1 | old F1 | new F1 | C0 cumulative | semantic cumulative |
|---|---:|---:|---:|---:|---:|
| Cora `order19611_s1` | +0.518 | -1.137 | +2.173 | 51.559 | 52.077 |
| Cora `order19611_s2` | +0.702 | -0.819 | +2.729 | 49.512 | 50.214 |
| Cora `order19612_s1` | +5.629 | +6.301 | +4.957 | 30.232 | 35.861 |
| Cora `order19612_s2` | +2.440 | +0.287 | +5.311 | 47.697 | 50.137 |
| Citeseer `order19611_s1` | +3.324 | +1.443 | +5.205 | 59.491 | 62.815 |
| Citeseer `order19611_s2` | +1.085 | +2.240 | -1.225 | 56.619 | 57.704 |
| Citeseer `order19612_s1` | -0.047 | -3.465 | +3.372 | 59.114 | 59.067 |
| Citeseer `order19612_s2` | -0.528 | -0.220 | -1.144 | 54.535 | 54.007 |

### 4.1 汇总

| 范围 | cumulative | old | new | 正 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|
| 全部 8 cells | **+1.640** | +0.579 | **+2.672** | **6/8** | **-0.528** |
| Cora 4 cells | +2.322 | +1.158 | +3.792 | 4/4 | +0.518 |
| Citeseer 4 cells | +0.958 | -0.001 | +1.552 | 2/4 | -0.528 |
| stage 1 | +2.356 | +0.785 | +3.927 | 3/4 | -0.047 |
| stage 2 | +0.925 | +0.372 | +1.418 | 3/4 | -0.528 |

## 5. 与 stage-local fresh 结果的区别

前一轮 stage-local fresh 结果为：平均 cumulative `+1.898 pp`、new `+2.988 pp`、正 cell `6/8`、最差 `-2.625 pp`。递归链路的平均值略低，但尾部明显改善：

| 版本 | cumulative 平均 | new 平均 | 正 cell | 最差 cell | 解释 |
|---|---:|---:|---:|---:|---|
| stage-local semantic | +1.898 | +2.988 | 6/8 | -2.625 | 每个 stage 从 C0 parent 构造 |
| recursive semantic | +1.640 | +2.672 | 6/8 | -0.528 | stage2 继承 semantic stage1 |

这不是“递归一定提高平均 F1”的证据。它说明递归 parent 改变了后续合成 memory 的状态，减少了一个严重尾部失败；同时 Citeseer `order19612_s2` 出现 `-0.528 pp`，表明跨阶段收益仍有数据集/顺序依赖。因而应该选择递归版本作为正式协议基础，但继续保留 C0 和 stage-local 结果作为机制对照。

最明显的局部变化是 Cora `order19611_s2`：stage-local semantic 为 `-2.625 pp`，真正递归 semantic 为 `+0.702 pp`。这说明旧 synthetic payload 的排列不是一个可以在每阶段独立重置的无关细节；它会改变下一阶段新 block 与历史 memory 的相互作用。

## 6. 守恒与实现审计

8 个递归 cell 均满足：

- `same_c0_topology=true`；
- `same_parent_prefix=true`；
- `payload_multiset_preserved=true`；
- stage2 parent arm 为 `semantic_perm`；
- 训练时 rows、batch、Student seeds 与 C0 一致；
- 没有用 report/test 标签做构造或选择。

因此本轮观察到的差异来自递归 parent 的 payload 状态，而不是 memory 行数、边数、Student 训练预算或额外监督标签。

## 7. 晋升判断

预先设定的晋升门槛为：平均 cumulative 至少 `+0.5 pp`、至少 `6/8` 正 cell、最差不低于约 `-2.5 pp`、new-class 平均不下降超过 `1 pp`。递归 semantic 满足：

- `+1.640 pp` cumulative；
- `+2.672 pp` new-class；
- `6/8` positive cells；
- worst `-0.528 pp`；
- 不增加 Student steps，不改变 topology，不使用 report/test 标签。

因此它可以从“小图候选”晋升为**小图已确认主方法基础**，进入大图 base+4 实验。但这仍不等于已经证明 SOTA，也不等于已经证明所有任务都有效。

## 8. 下一步

1. 在 arXiv、BookChild、SportsFit 上采用相同递归 parent 规则，做 base+4 主实验。
2. 大图主表至少包含 C0、recursive semantic、k-center/medoid 和完整数据参考；所有方法统一 Student、memory rate、batch、steps、seed 和类别顺序。
3. 保留 Cora/Citeseer 的 stage-local 结果作为“递归 parent 的必要性”消融，不把它与 recursive 主方法混合。
4. 继续记录 old/new F1、最差顺序、memory rows、construction time 和 Student training time；暂不加入 role/message/plasticity/adaptive 组件。
5. 如果大图收益消失，论文仍可写成“递归 parent 对持续 TAG condensation 的必要性与 semantic payload assignment 的机制分析”，不能只报告平均小图提升。

## 9. 证据文件

- recursive construction：`results/tgc_fresh_semantic_recursive_20260926/`
- recursive Cora evaluation：`results/tgc_fresh_semantic_recursive_eval_20260926_cora/results.json`
- recursive Citeseer evaluation：`results/tgc_fresh_semantic_recursive_eval_20260926_citeseer/results.json`
- recursive construction runner：`code/v2_llm_fusion/run_tgc_semantic_recursive_confirmation_20260926.py`
- previous stage-local report：`docs/current/TGC_ROLE_AWARE_FRESH_CONFIRMATION_ZH_2026-09-26.md`
