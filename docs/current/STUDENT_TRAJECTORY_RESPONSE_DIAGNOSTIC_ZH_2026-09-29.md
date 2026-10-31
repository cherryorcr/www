# Student Trajectory Response 诊断报告

日期：2026-09-29  
状态：完成 `x-provenance r=0.10b` 与共同 control 的只读轨迹诊断；未修改任何 memory。

## 1. 目的

上一轮已经确认：直接对新 synthetic `X` 做 Student outer 更新存在真实梯度，但 construction-time support InfoNCE 与最终 report-dev 分类 F1 的对应关系不稳定。因此本轮不再添加新的 `X/P/A` 损失，而是先回答：

> 在完全相同的 Student 初始化、优化器和 batch 顺序下，candidate 相对 C0 的短期训练响应，能否预测最终下游 F1 的增减？

只有这个响应估计器可靠，才有理由让它反向指导后续 `X` revision。

## 2. 固定协议

比较对象为四个 cell 中的：

- `control_*`：本轮 direct Student outer 实验共同的 C0 输入；
- `provenance_x_r10b_*`：只修改新 synthetic rows 的 `X`，半径 `r=0.10` 的 provenance-aware candidate；
- support：对应 cell 的 `previous synthetic rows + current raw delta` 封存构造支持集。

Student 和训练轨迹固定为：

| 项目 | 设置 |
|---|---|
| 数据集/顺序 | Cora/Citeseer × `canonical`/`order19102` |
| Student | canonical 两层 GCN + 双向图文 InfoNCE |
| 优化器 | Adam，`lr=1e-3` |
| batch | 16 |
| Student seeds | 150--155，candidate/control 成对使用 |
| checkpoint | `0/2/4/8/16/32/64` steps |
| support response | 封存 support validation split 的 old/new/total InfoNCE |
| memory 修改 | 无 |
| 历史 raw 访问 | 0 |
| report F1 使用 | 轨迹写出后，仅用于离线相关性分析 |

每个 seed 的 candidate 与 control 使用同一个初始化、同一个 Adam 初始状态和同一组随机 batch。`text_emb`、文本、拓扑、mass、depth 和 stage 均经过逐字段一致性审计。

## 3. 轨迹如何计算

对 memory `M`，Student 按 canonical 下游训练方式迭代：

\[
\theta_{t+1}
=
\operatorname{Adam}
\left(
\nabla_{\theta_t}
\mathcal L_{\mathrm{InfoNCE}}(M;\theta_t)
\right).
\]

在每个 checkpoint，使用同一个封存 construction support 计算：

\[
\Delta L_t^{old}
=
L_t^{old}(M_{cand})-L_t^{old}(M_{C0}),
\]

\[
\Delta L_t^{new}
=
L_t^{new}(M_{cand})-L_t^{new}(M_{C0}),
\]

\[
\Delta L_t^{total}
=
0.25\Delta L_t^{old}+\Delta L_t^{new}.
\]

若 response 真能作为优化信号，则更低的 candidate response 应该与更高的最终 F1 相对应。因此报告对 `-Delta L` 与最终 F1 delta 计算 Pearson、Spearman 和符号一致率。

## 4. 最终 F1 对照

| cell | candidate 相对 control Final F1 | New F1 | Old F1 |
|---|---:|---:|---:|
| Cora / canonical | +0.668 | +1.076 | +0.426 |
| Cora / order19102 | +0.635 | +0.877 | -0.003 |
| Citeseer / canonical | -1.926 | -1.684 | -1.223 |
| Citeseer / order19102 | +0.511 | +2.614 | -2.901 |
| **四 cell 平均** | **-0.028** | **+0.721** | **-0.925** |

这四个结果本身已经显示出 old/new trade-off：`r=0.10` 可以在部分 cell 提升新类，但不能稳定保住旧类；Citeseer/canonical 是一个重要的反例。

## 5. pooled 轨迹结果

下面使用四个 cell × 六个 paired Student seeds，共 24 个观察。`Spearman(-response, F1)` 越接近正值，越支持“response 越低，最终 F1 越高”。

| checkpoint | total vs Final F1 Spearman | total vs Final F1 Pearson | total vs New F1 Spearman | total vs Old F1 Spearman | 符号一致率 |
|---:|---:|---:|---:|---:|---:|
| 2 | -0.362 | -0.360 | -0.136 | -0.503 | 37.5% |
| 4 | -0.343 | -0.351 | -0.105 | -0.473 | 37.5% |
| 8 | -0.165 | -0.214 | -0.003 | -0.310 | 33.3% |
| 16 | +0.086 | +0.142 | +0.339 | -0.184 | 41.7% |
| 32 | +0.149 | +0.117 | +0.214 | +0.068 | 50.0% |
| 64 | **+0.237** | +0.087 | +0.225 | +0.183 | **54.2%** |

结论不是“64 步已经有效”，而是：随着 horizon 增长，方向从明显负相关逐渐靠近 0，并出现很弱的正相关；但在 24 个配对观察上，最大 Spearman 仍只有 `0.237`，方向判断仍接近随机，不能用于 candidate 选择或梯度更新。

## 6. 为什么不能据此继续更新 X

### 6.1 construction support 和最终任务不是同一个算子

support response 使用的是 `previous synthetic + current raw delta` 的封存 support 图；最终评估使用 candidate synthetic memory 在真实 full graph 上训练 Student，再用 prompt 做零样本分类。candidate 改变的是 synthetic `X`，但它最终影响的是 full-graph normalization、全局消息传播和分类边界。support loss 只看到一个局部构造图，无法完整反映这一变化。

### 6.2 InfoNCE 目标和 report-dev 分类目标不一致

轨迹测量的是图文对齐损失，而最终指标是按类别 prompt 的节点分类 Macro-F1/Accuracy。即使 candidate 让图文 embedding 更接近，也可能把节点推向错误的类别 prompt，或者改善大类而损害少数新类。

### 6.3 horizon 仍不足以覆盖下游训练动力学

`2/4/8` 步的 response 与最终 F1 明显反向或接近无关；延长到 `64` 步只能削弱这种错位，不能形成可靠预测。正式下游训练通常远长于这个诊断 horizon，直接扩大 horizon 的成本会快速接近重新训练所有候选 Student。

### 6.4 正收益主要是 cell/顺序条件性的

例如 Citeseer/canonical 在 64 步的平均 total response 下降约 `-0.001055`，看起来比 C0 更好，但最终 Final F1 反而下降 `-1.926 pp`。这说明 response 下降并不等价于下游任务收益。

## 7. 当前决策

本轮之后：

1. `x-provenance r=0.10b` 保留为有解释力的 X-only 候选和负结果，不晋升主方法；
2. 不把 `response-gain`、大规模 selector 或 trajectory response 放入主方法；
3. 不再继续扫描更多 radius、prompt CE、old barrier 或冻结文本组合；
4. P/A 联合优化暂缓，因为当前连 X-only response estimator 都没有通过可靠性验证；
5. 这轮实验支持一个清晰的机制结论：**梯度可以传到 X，但短期 label-free support response 不能可靠代理最终 Student 分类效用。**

## 8. 后续优化边界

下一轮如果继续研究梯度路线，必须先改变“用什么监督信号评价 X”，而不是继续增加损失项。优先级如下：

### P1：延迟目标的无标签响应校准

只允许使用当前 delta 和 synthetic parent，固定一个独立 construction validation split，比较：

- 图表示在新类 support 上的 margin；
- 新旧类之间的 prompt-relative separation；
- Student 参数更新前后的 gradient/representation direction consistency；
- response 与 full-graph representation drift 的关系。

这些量先只做诊断，不直接更新 X。若仍不能跨 cell 预测最终 F1，则终止 Student-guided X revision。

### P2：只做受限的几何 X 更新

如果必须继续改 `X`，只允许一个变量、一个固定目标：

\[
L_X = L_{parent-anchor}
       + \lambda_{sep}L_{new/old\ separation}
       + \lambda_{smooth}L_{slot\ smoothness}.
\]

其中 parent anchor 保证旧语义不漂移，separation 只提高新类与已见 prompt 的可分性，slot smoothness 防止同类 synthetic rows 发生无约束发散。不得同时改变 `P`、文本、拓扑和预算，也不得使用最终 F1 选系数。

### P3：独立顺序确认

任何新的 X 目标都必须在未用于调参的新顺序上验证，并同时报告 Final F1、New F1、Old F1、Accuracy、最差 cell 和构造成本。只有至少两个数据集平均正收益、最差 cell 可解释且不超过预设退化阈值，才考虑重新进入主方法。

## 9. 产物与复现

- 诊断实现：[run_student_trajectory_response.py](../../code/v2_llm_fusion/run_student_trajectory_response.py)
- 单元测试：[test_student_trajectory_response.py](../../code/v2_llm_fusion/test_student_trajectory_response.py)
- v2 汇总：`results/direct_student_outer_20260928/trajectory_response_r10b_v2_summary.json`
- v3 长 horizon 汇总：`results/direct_student_outer_20260928/trajectory_response_r10b_v3_summary.json`
- 四个 cell 的逐 seed 轨迹 JSON 位于同一目录，文件名前缀分别为 `trajectory_response_r10b_v2_` 和 `trajectory_response_r10b_v3_`。

