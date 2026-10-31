# TGC Prompt-Paired Selector 结项报告

日期：2026-09-24  
范围：Cora/Citeseer，小图，`order19601`，stage 1/2  
前一版本：V3 label-free InfoNCE/margin response  
本轮定位：优化 selector 目标，不改变候选构造和 student

## 1. 本轮修改

本轮只修改候选选择器，保持以下内容完全不变：

- `fallback_content`、`p_initial`、`anchor025`、`joint` 四个候选；
- 共享 assignment 生成 X、文本和二值拓扑的 materialization；
- 原始两层 GCN student；
- 冻结 MiniLM 和可训练 text projection；
- 双向 graph-text InfoNCE；
- batch=16、selector seeds `300,301,302`、report seeds `150..155`；
- C0 historical prefix、stage-wise 运行方式和 report-dev 审计。

修改后的 selector 包含两个信号：

1. **Class-prompt response**：使用已知 zero-shot 类别 prompt，计算当前增量相对旧类别 prompt 的 gap，以及当前类别内部的 top-1/top-2 prompt margin；
2. **Paired response**：候选和 `fallback_content` 使用相同 seed、相同 horizon 和相同 batch 随机序列，计算 candidate-minus-fallback 的响应差值。

选择阶段仍不读取节点标签或 report-dev F1。

## 2. 结果

封存后的 report-dev 累计 Macro-F1（%）与 V3 相同，因为四个候选的构造和正式 report 过程没有改变：

| 数据集 | stage | fallback | p_initial | anchor025 | joint | report 最优 |
|---|---:|---:|---:|---:|---:|---|
| Cora | 1 | 56.015 | 56.388 | **58.971** | 54.662 | anchor025 |
| Cora | 2 | 50.768 | 49.397 | **51.736** | 51.644 | anchor025 |
| Citeseer | 1 | 53.237 | 55.129 | **56.214** | 53.820 | anchor025 |
| Citeseer | 2 | 54.405 | **57.203** | 54.484 | 55.517 | p_initial |

prompt-paired selector 的实际选择：

| 数据集 | stage | 选择结果 | report 最优 | regret | 相对 fallback |
|---|---:|---|---|---:|---:|
| Cora | 1 | fallback_content | anchor025 | 2.956 | 0.000 |
| Cora | 2 | anchor025 | anchor025 | 0.000 | +0.968 |
| Citeseer | 1 | fallback_content | anchor025 | 2.976 | 0.000 |
| Citeseer | 2 | fallback_content | p_initial | 2.798 | 0.000 |

## 3. 与 V3 对照

| 指标 | V3：InfoNCE/margin | Prompt-paired | 变化 |
|---|---:|---:|---:|
| 平均 selection regret | **1.540 pp** | 2.182 pp | 变差 0.642 pp |
| 平均实际相对 fallback 收益 | **+0.885 pp** | +0.242 pp | 下降 0.643 pp |
| 选中 report 最优的 cell | 1/4 | 1/4 | 无变化 |
| selector/report 排序相关 | -0.25 | -0.10 | 略改善 |
| 有正实际收益的 cell | 2/4 | 1/4 | 下降 |

## 4. 结果解释

### 4.1 prompt 目标更接近最终任务，但还不是可靠的最终目标

相比 instance-level InfoNCE，class-prompt gap 在形式上更接近最终 zero-shot 分类：最终预测确实是图节点表示与类别 prompt 的比较。因此排序相关从 `-0.25` 改善到 `-0.10`，说明方向没有完全错误。

但它仍然不能稳定预测 report Macro-F1，原因包括：

- prompt gap 只反映当前 student 表示的局部相对方向，不反映真实图上的邻域传播；
- selector 在 raw delta 和 synthetic prefix 上测量，report 在真实完整图上测量；
- 当前增量内部的 prompt gap 不等价于各类别的 Macro-F1，尤其不能反映少数类召回；
- report 的类别竞争是在所有 seen prompts 上进行，而 selector 将新类和旧类分组汇总，丢失了类别级细节。

### 4.2 paired response 降低了部分噪声，但没有消除目标错位

相同 seed 和 batch 序列使候选与 fallback 的差值更容易解释，尤其适合消除初始化噪声。然而 candidate 仍然改变了图结构、X 和文本，因此训练梯度轨迹本身会改变。配对只能消除共享随机数源，不能保证 proxy 与最终 Macro-F1 同向。

### 4.3 主要失败来自过严的固定门限

本轮使用的 prompt 分数不是 F1 百分点，而是归一化表示空间中的余弦量级。`0.02` 的 retention/gain tolerance 在这个量纲下偏严格，导致很多候选在 proxy 上略为负时直接被拒绝：

- Cora stage 1 的 `anchor025` report 最优，但被 selector 认为 utility 为负；
- Citeseer stage 2 的 `p_initial` report 最优，但也被拒绝；
- selector 最终表现为“宁可回退”，降低了误接受，却增加了误拒绝。

因此，不能把本轮结果解释为 class-prompt 或 paired response 已经失败；更准确的结论是：当前固定权重和硬门限组合没有通过验证。

## 5. 成本与协议审计

每个 dataset-stage：

- selector：4 个候选的响应评估，fallback 响应复用，共约 36 次 student fit；
- report：4 个候选 × 6 个 report seeds，共 24 次 student fit；
- 总计约 60 次 student fit/cell，四个 cell 约 240 次。

选择阶段：

- 未使用节点标签；
- 未使用 report-dev F1；
- 未使用 test；
- 使用合法的类别 prompt；
- 输出 protocol、selection、build audit、progress 和 results JSON。

## 6. 是否进入主方法

不进入。理由不是候选空间没有收益，而是 selector 没有满足晋升要求：

- 平均 regret 没有下降；
- 只有 `1/4` cell 选中 report 最优；
- 平均实际收益只有 `+0.242 pp`；
- 尚未做独立类别顺序确认；
- 尚未做递归 selected-parent rollout。

当前准确定位仍是：

```text
C0/R11                    稳定主线
shared assignment          有条件收益的候选构造
V3 response selector       未通过的无标签排序器
prompt-paired selector     更接近任务目标但过于保守的诊断版本
```

## 7. 下一步建议

下一轮不再继续叠加新的构造模块，而是做一个小而严格的 selector 校准实验：

1. 保留 class-prompt 的逐类别 margin，不再只汇总成 old/new 两个 group；
2. 使用 paired response 的置信区间和 rank/Pareto 规则，取消未经标定的 `0.02` 硬阈值；
3. 把候选分为“明确改善、统计不确定、明确恶化”三类；
4. 对统计不确定候选默认回退，但记录 false rejection；
5. 只在新的独立顺序上验证一次，不能用本轮 report 结果调参后回写；
6. 若校准后平均 regret 仍大于约 1 pp，则停止自动 selector 主线，把 shared assignment 和 response gate 写成机制分析/负结果。

不能根据本轮四个 cell 事后调整权重并重新宣称收益；任何新的阈值和权重都必须先冻结，再在独立顺序上运行。

原始结果：

`results/tgc_shared_assignment_gcn_prompt_paired_20260924/`

代码：

`code/v2_llm_fusion/run_tgc_joint_student_prompt_paired.py`
