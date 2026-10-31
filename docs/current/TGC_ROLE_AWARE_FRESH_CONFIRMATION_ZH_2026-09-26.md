# TGC Role-aware 新顺序独立确认报告

日期：2026-09-26  
状态：已完成 `order19611/19612` 的 fresh 构造与统一 Student 重载；`semantic_perm` 获得独立正信号，但尚未满足全部稳定性晋升条件。

## 1. 实验目的

上一轮 12-cell 评估中，`semantic_perm` 是当前最好的简单候选。本轮不再开发新模块，而是使用新的类别顺序和 fresh R10/C0 source，验证该收益是否能在未参与 role-aware 开发的条件下复现。

本轮仍然冻结：

- 原始两层 GCN Student；
- 双向图文 InfoNCE；
- batch=16；
- Student seeds=150--155；
- C0 topology、旧 prefix、文本策略和 memory rows；
- 不使用 report/test 标签进行构造或选择；
- 不加入 selector、dynamic budget、GraphGPS、文本 attention 或旧 memory refresh。

## 2. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 新类别顺序 | `order19611`、`order19612` |
| 阶段 | stage 1、stage 2，共 8 个 fresh cells |
| Student | 原始两层 GCN + trainable text projection |
| 目标 | 双向图文 InfoNCE，无节点分类监督 |
| 训练 | 每个 arm 使用同 cell 的 `protocol_v3.student_steps(rows)`、batch=16、seed 150--155 |
| 评价 | report-dev、all-prompt、已见类 cumulative/old/new Macro-F1 |
| 选择 | 构造后冻结重载；不按 report-F1 选择 |

每个新 cell 同时评估 `c0`、`identity`、`semantic_perm`、`role_perm`、`anchor_role_perm`、`message_role_perm`、`plasticity_role_perm`、`adaptive_role_perm` 和 `random_perm`。

## 3. Fresh 汇总结果

下表所有差值都相对同一 cell 的 C0，单位为百分点。`positive` 是累计 F1 严格高于 C0 的 cell 数。

| 方法 | cumulative F1 | old F1 | new F1 | positive | 最差 cell |
|---|---:|---:|---:|---:|---:|
| C0 | 0.000 | 0.000 | 0.000 | 0/8 | 0.000 |
| identity | +0.153 | -0.830 | +1.283 | 5/8 | -5.226 |
| semantic_perm | **+1.898** | **+0.930** | **+2.988** | **6/8** | **-2.625** |
| role_perm | +0.207 | -0.884 | +1.537 | 5/8 | -3.853 |
| anchor_role_perm | +1.001 | +0.335 | +1.797 | 6/8 | -3.853 |
| message_role_perm | -0.038 | -1.448 | +1.602 | 4/8 | -2.679 |
| plasticity_role_perm | +1.048 | +0.190 | +2.052 | 6/8 | -1.791 |
| adaptive_role_perm | +1.048 | +0.190 | +2.052 | 6/8 | -1.791 |
| random_perm | +0.944 | +1.893 | +0.254 | 5/8 | -1.864 |

### 3.1 按数据集

| 数据集 | 方法 | cumulative | old | new | positive |
|---|---|---:|---:|---:|---:|
| Cora | semantic_perm | +2.017 | +1.593 | +2.499 | 3/4 |
| Cora | role_perm | +0.138 | -0.230 | +0.840 | 3/4 |
| Citeseer | semantic_perm | +1.779 | +0.267 | +3.477 | 3/4 |
| Citeseer | role_perm | +0.276 | -1.539 | +2.234 | 2/4 |

`semantic_perm` 在两个数据集上都保持正的平均累计提升，且新类 F1 平均提升，而不是仅靠旧类保持得到累计分数。

## 4. 逐 cell 的关键结果

| Cell | semantic cumulative | semantic old | semantic new | role cumulative | role old | role new |
|---|---:|---:|---:|---:|---:|---:|
| Cora `order19611_s1` | +0.518 | -1.137 | +2.173 | +0.518 | -1.137 | +2.173 |
| Cora `order19611_s2` | -2.625 | -2.306 | -3.050 | +0.159 | -0.895 | +1.564 |
| Cora `order19612_s1` | +5.629 | +6.301 | +4.957 | +2.494 | +6.700 | -1.712 |
| Cora `order19612_s2` | +4.545 | +3.516 | +5.916 | -2.620 | -5.586 | +1.334 |
| Citeseer `order19611_s1` | +3.324 | +1.443 | +5.205 | +3.324 | +1.443 | +5.205 |
| Citeseer `order19611_s2` | +1.532 | +1.155 | +2.288 | -3.853 | -4.052 | -3.453 |
| Citeseer `order19612_s1` | -0.047 | -3.465 | +3.372 | -0.670 | -5.481 | +4.141 |
| Citeseer `order19612_s2` | +2.305 | +1.935 | +3.045 | +2.305 | +1.935 | +3.045 |

这里最值得注意的是：`semantic_perm` 的唯一明显尾部退化是 Cora `order19611_s2` 的 `-2.625 pp`，而同一 cell 的 `role_perm` 接近持平。说明 semantic assignment 的平均效果是真实的，但仍存在与阶段/顺序有关的局部失配；不能只看 8-cell 平均值。

配对 Student seed 审计显示，该 cell 中 `semantic_perm` 只有 6 个 seed 中的 1 个高于 C0，seed-level 差值为 `-4.828、-1.781、+1.038、-2.592、-6.464、-1.122 pp`，因此退化不是单个异常 seed 造成的。主要负贡献来自类别 `6`、`5`、`2` 和 `4`，平均 per-class 差值约为 `-6.10、-4.72、-4.10、-3.85 pp`。

## 5. 结果解释

### 5.1 semantic permutation 有独立正信号

相对于上一轮 12-cell 的 `+1.690 pp`，fresh 8-cell 仍有 `+1.898 pp`；新类 F1 从上一轮的接近持平变为 `+2.988 pp`。这说明该方法不是只改善旧类保持，也不是只依赖某一个已开发顺序。

### 5.2 还不能称为稳定晋升

预注册的晋升门槛是：平均累计 F1 至少 `+0.5 pp`、至少 `6/8` 正 cell、最差 cell 不低于约 `-2.5 pp`、new-class 平均下降不超过 `1 pp`。本轮前三项中，前两项和 new-class 约束满足，但最差 cell 为 `-2.625 pp`，比尾部门槛低 `0.125 pp`。

因此当前结论应写成：

> `semantic_perm` 在独立新顺序上得到稳定的平均正收益和新类收益，但尾部 cell 仍未完全受控；它是“已确认有潜力的候选主线”，还不是无条件替换 C0 的最终方法。

### 5.3 role-aware 复杂项没有证明比 semantic 更好

`role_perm` fresh 平均只有 `+0.207 pp`，old F1 下降 `-0.884 pp`，最差 `-3.853 pp`。它确实在部分 cell 提升新类，但跨顺序不稳定，不能因为单个 cell 的提升而进入主方法。

`plasticity_role_perm` 与 `adaptive_role_perm` 在这 8 个 fresh cell 的下游汇总完全相同。代码中的代价函数并不相同，但这轮没有产生可区分的 Student 效果，所以不能把两者作为两个已验证贡献分别宣称，后续需要先做 assignment/memory 差异审计。

`random_perm` 在 Cora 新顺序上出现较高正值，但 Citeseer 平均为负；它不能作为有效方法，只能说明 permutation 的收益仍有顺序依赖，不能用随机重排替代语义分配。

## 6. 当前晋升判断

### 保留为候选主线

当前最合理的候选仍是：

`C0 topology + two-prototype proposal + semantic conserved payload assignment + original Student`

理由：

- 两轮合计 20 个小图 cell 中，上一轮和 fresh 轮都保持正的总体平均提升；
- fresh 轮两个数据集都正；
- fresh 轮新类 F1 平均提升 `+2.988 pp`；
- 不需要额外 Student selector，也没有增加 Student 训练步数；
- 构造只改变类内 payload assignment，C0 topology 和历史 prefix 不变，容易审计。

### 暂不把它写成最终主方法

原因只有一个，但很关键：`order19611_s2/Cora` 的尾部退化仍超过预设容忍度。当前不能用平均值掩盖这一点，也不能把 `role_perm` 的局部正结果与 `semantic_perm` 混成一个更强的组合方法。

## 7. 下一步只做的事情

1. **最差 cell 机制审计**：检查 Cora `order19611_s2` 的每类 assignment、semantic cost gain、payload source、C0/semantic 的 per-class F1 和 Student seed 方差；不使用 report 标签进行重新选择。当前审计已经确认：预算为 94 行、旧 prefix 为 60 行、新 block 为 34 行；payload multiset、C0 topology、旧 prefix 均守恒；semantic 三个受影响类别的 assignment gain 为 `-0.122/-0.064/-0.308`，没有发现空结果或内容丢失。
2. **检查 arm 可区分性**：确认 `plasticity_role_perm` 与 `adaptive_role_perm` 的 assignment 是否真的不同；若 memory 不同但 Student 完全相同，则记录为负结果，不继续堆叠。
3. **只做一个有界尾部控制候选**：在 semantic assignment 上增加类内 identity anchor/回退，必须与增加 rows、改变 topology 等因素隔离，并在新的顺序上重新验证。
4. **若尾部控制不能同时保持平均收益与最差 cell**：冻结 C0 为安全基线，semantic_perm 作为论文中的机制候选/正向结果，不再继续叠加 role、message、plasticity、adaptive 模块。
5. **补完整 continual 链路**：本轮 `stage2` 的 arm 是从同一 source 的 `c0_s1_memory` 构造，而不是从 `semantic_perm_s1_memory` 继续追加。因此 fresh 结果不能单独证明 semantic permutation 沿多阶段递归保持收益；在上大图前必须做一条真正的 `C0 -> semantic_perm_s1 -> semantic_perm_s2` 链，并与 `C0 -> C0_s1 -> C0_s2` 对照。
6. **暂不上大图最终实验**：先完成上述审计和真正递归链路的最小确认，再决定是否将 semantic_perm 上推到 arXiv/BookChild/SportsFit 的 base+4。

## 8. 证据文件

- Fresh Cora detailed evaluation：`results/tgc_fresh_role_aware_detailed_eval_20260926_cora/results.json`
- Fresh Citeseer detailed evaluation：`results/tgc_fresh_role_aware_detailed_eval_20260926_citeseer/results.json`
- Fresh construction source：远程 `results/tgc_fresh_r10_source_20260926`
- Fresh role-aware memory：远程 `results/tgc_fresh_role_aware_20260926`
- 上一轮 12-cell 报告：`docs/current/TGC_ROLE_AWARE_FULL_ARM_EVALUATION_ZH_2026-09-26.md`
- 统计与 artifact 审计：`results/tgc_role_aware_fresh_confirmation_audit_20260926.json`
- 审计脚本：`tools/audit_tgc_fresh_confirmation.py`

## 9. 审计与协议边界

本轮统计由 `tools/audit_tgc_fresh_confirmation.py` 从两个详细评估 JSON 独立重算，得到与报告一致的 `semantic_perm` 汇总：累计 `+1.8976 pp`、old `+0.9302 pp`、new `+2.9880 pp`、正 cell `6/8`、最差 `-2.6249 pp`。审计还检查了每个 Student seed、训练步数、评价类别集合和 per-class F1 的算术一致性。

需要避免一个容易误读的地方：fresh 构造脚本的 stage2 读取 `c0_s1_memory.pt` 作为 parent，stage2 semantic memory 并不读取 stage1 semantic memory。因而本轮结果适合回答“同一个阶段内，semantic payload assignment 是否有局部效用”，不适合直接声称“完整持续链路已经被验证”。真正的主方法确认必须追加递归 parent 实验；否则论文中应把本轮标为 stage-local mechanism confirmation。
