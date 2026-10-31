# TGC-R2B2：student 训练阶段对齐实验结项报告

日期：2026-09-22  
状态：正式矩阵完成，独立审计通过；未达到晋升主方法的门槛。

## 1. 一句话结论

让历史合成 `X` 的可微修订发生在 student 训练中期，确实产生了比随机位移更有希望的信号；其中 `warm-up=64, suffix=1` 的未加 gate 候选在第二增量阶段相对 C0 平均提升 **+0.85 pp**，但冻结的无标签 gate 将它全部拒绝，最终可部署的 guarded 策略没有超过 C0。因此，本轮支持“训练阶段对齐值得研究”，不支持把它直接升为 TGC-R 主方法。

## 2. 实验问题与冻结协议

本轮只改变 student 响应发生的训练阶段，保持 C0 的其它因素不变：

- 数据：Cora、Citeseer；类别顺序：`canonical`、`order19101`。
- 结构：每个 cell 的 s1、s2 都从同一个 C0 父 memory 开始，不把候选逐阶段递归拼接。
- memory：5% growing memory，C0 新增 block、文本、text embedding、全部拓扑、row 数和 row order 固定。
- 修订：只允许修改预先限定的旧 synthetic X 行；protected/editable rows、位移半径 `0.1`、outer steps `16`、outer lr `0.01` 固定。
- student：两层 GCN，hidden/output 384；冻结 MiniLM；双向图文 InfoNCE；最终 student 仍从头训练，不继承临时 student。
- 训练阶段矩阵：warm-up `0/64/256` 步，随后可微 Adam suffix `1/4` 步。
- 对照：C0、teacher、student、等幅 random、stop-gradient、guarded。
- report seeds：100--105；元优化 train seeds 7500/7501，validation seeds 7510/7511。
- 约束：修订算子不访问历史完整 raw graph，只使用旧 synthetic memory 与当前 raw delta；selection/gate 不读取 report/test 标签。

这是一个共享 C0 父 memory 的单阶段机制诊断，不是递归 continual rollout。

## 3. 执行与审计结果

| 项目 | 结果 |
|---|---:|
| 正式记录槽位 | 1,248 |
| 新执行 student fit | 360 |
| 合法复用记录 | 888 |
| 独立重载的 student 状态 | 648 |
| 预测、memory、checkpoint 审计 | 通过，0 errors |
| 最大独立指标重算误差 | `<= 1e-7 pp` |
| 远程 GPU 前置测试 | 17 passed |
| 本地 reporting 测试 | 6 passed |
| warm-up=64 前缀时间 | 4.80 s |
| warm-up=256 前缀时间 | 19.08 s |
| 新 student 元优化时间 | 30.69 s |
| stop 对照优化时间 | 17.92 s |

所有正式 cell 均完成：Cora/canonical、Cora/order19101、Citeseer/canonical、Citeseer/order19101；未出现异常退出或数据缺失。

## 4. 主要性能结果

以下为四个数据集-顺序条件等权平均；差值相对同阶段 C0，单位为百分点。

### 4.1 s1

| 方法 | Macro-F1 | 相对 C0 | old F1 差值 | new F1 差值 |
|---|---:|---:|---:|---:|
| C0 | 55.33 | 0.00 | 0.00 | 0.00 |
| student, w0-k1 | 55.42 | +0.09 | +0.41 | +0.68 |
| student, w64-k1 | 56.68 | **+1.35** | +0.21 | +2.40 |
| student, w256-k1 | 55.51 | +0.18 | +0.10 | -0.42 |
| guarded, w0-k1 | 55.52 | +0.20 | +0.65 | +0.97 |
| guarded, w64-k1 | 55.33 | +0.01 | +0.02 | +0.25 |
| guarded, w256-k1 | 55.33 | 0.00 | 0.00 | 0.00 |

`w64-k1` 在 s1 的未加 gate 信号明显，但 gate 几乎全部回退，因此没有转化为部署策略收益。

### 4.2 s2

| 方法 | Macro-F1 | 相对 C0 | old F1 差值 | new F1 差值 | 相对匹配 random |
|---|---:|---:|---:|---:|---:|
| C0 | 51.34 | 0.00 | 0.00 | 0.00 | - |
| student, w0-k1 | 51.80 | +0.46 | +0.24 | +0.95 | +0.84 |
| student, w64-k1 | **52.19** | **+0.85** | +0.13 | **+2.14** | **+1.52** |
| student, w256-k1 | 51.02 | -0.32 | -1.10 | +0.88 | -0.07 |
| guarded, w0-k1 | 51.40 | +0.06 | +0.13 | -0.03 | +0.45 |
| guarded, w64-k1 | 51.34 | 0.00 | 0.00 | 0.00 | +0.67 |
| guarded, w256-k1 | 51.31 | -0.03 | -0.18 | +0.16 | +0.22 |

其它 suffix 的 s2 汇总如下：

| 方法 | s2 相对 C0 | s2 相对匹配 random |
|---|---:|---:|
| student, w0-k4 | -0.58 | +0.51 |
| student, w64-k4 | -0.11 | +0.23 |
| student, w256-k4 | -0.45 | -0.37 |
| guarded, w0-k4 | -0.85 | +0.24 |
| guarded, w64-k4 | -0.22 | +0.12 |
| guarded, w256-k4 | -0.40 | -0.32 |

因此当前最清楚的模式是：**中期 warm-up + 单步 suffix 有方向性；增加 suffix 长度或 warm-up 到 256 步并没有稳定收益。**

## 5. gate 为什么没有把收益留下来

冻结 gate 要求两个 validation seeds 都满足：new loss 至少改善 `1e-4`，old loss 不恶化超过 `0.01`。它是无标签的合法选择规则，但仍不能预测最终长程 student F1。

关键现象：

- `w64-k1` 在 s2 的四个 cell 都被拒绝，但其最终 report F1 的平均差值为 **+0.85 pp**。
- `w0-k1` 在 s2 只有部分 cell 被接受，guarded 平均只剩 **+0.06 pp**。
- `w256-k1` 在 s2 的 student 平均为 **-0.32 pp**，guarded 平均为 **-0.03 pp**，没有恢复出稳定收益。
- gate 依据的是 1/4 步 suffix 的 probe loss，而正式 student 需要约 1,000--1,800 步；短段 loss 与长程 zero-shot Macro-F1 仍然错位。
- old-loss hinge 在这一轮没有成为决定因素；主要拒绝原因仍是 new-gain 的逐 seed 阈值。也就是说，问题不是简单把 old tolerance 调小或调大就能解决。

这说明当前机制有两个分离问题：

1. **候选生成问题：** `w64-k1` 的梯度方向相对 random 有效，说明中期 student 状态包含比随机初始化更有用的局部信号。
2. **候选选择问题：** validation probe 的短期 new loss 不能可靠地预测最终共同 prompt 下的长程分类结果。直接放宽 gate 会重新引入坏候选，不能作为修复。

## 6. 重要解释与边界

### 6.1 不能把未加 gate 的最高分当成已验证方法

`student_w64_k1` 是研究诊断臂，不是最终算法，因为它没有通过预注册的 guarded 晋升条件：

- s2 平均至少 `+0.5 pp`：w64-k1 的 student 达到，但 guarded 没达到；
- 至少 3/4 cells 不低于 `-1 pp`：部分 guarded 条件满足；
- s1 不下降：满足；
- 新类不下降：w64-k1 student 满足，guarded 只在平均意义上接近；
- 优于 random、且不低于 teacher：不少条件满足；
- **完整 guarded 策略必须同时满足以上条件：未满足。**

按预先固定规则，不能事后选择 `student_w64_k1` 的 report 结果，再把 gate 删除后称为主方法。

### 6.2 没有统计显著性结论

四个开发 cell、每个六个 report seeds 的配对区间仍然较宽；本轮没有重复构造 seed，也没有多重比较校正。因此 `+0.85 pp` 是值得继续验证的开发信号，不是稳定泛化提升。

### 6.3 没有违反持续访问约束

修订阶段只读旧 synthetic memory 和当前 raw delta；历史完整 raw graph 没有提供给修订算子。完整真实图仍只用于独立的下游评估，这两个访问范围必须区分。

## 7. 对 TGC-R 的判断

本轮比上一轮更具体地说明了：

- “student 反向优化历史 X”不是没有信息；`w64-k1` 比同 suffix 的 w0 平均高 **+0.39 pp**，比匹配 random 高 **+1.52 pp**（s2）。
- 但目前它仍是一个**候选生成机制**，不是完整可靠的 continual 算法。完整算法还需要一个能与长程 student 目标对齐的无标签选择器。
- `warm-up=256` 没有形成更好主线，不能简单理解为训练越充分越好。
- 继续增大 suffix、放宽 gate 或扩大位移半径都没有现有证据支持；盲目调参会把开发集信号变成选择偏差。

因此当前版本状态为：

> C0 仍是冻结主方法和所有新版本的共同参照；TGC-R2B2 保留为“训练阶段对齐的可微历史 X 修订”研究分支，暂不进入主表，不启动递归 rollout、文本联合修订、拓扑联合修订或大图扩展。

## 8. 后续最小路线

下一步不应直接把 `w64-k1` 写进最终算法，而应针对“候选生成有效、选择器失效”做最小验证：

1. 冻结 `w64-k1`，在未暴露的新类别顺序和新 construction seed 上复验未加 gate 候选；保持 C0、random、stop 和 teacher 对照。
2. 只替换 selection signal：比较短期 new loss、old/new 混合 probe、多个 suffix checkpoint 的 response curve；不改变 X、文本、拓扑和预算。
3. 记录选择 regret、误接受、误拒绝和额外成本；若不能稳定识别 w64-k1，再停止 TGC-R 分支。
4. 只有候选和 selector 都通过独立确认，才考虑递归 continual rollout；在此之前不组合文本或拓扑模块。

## 9. 可复现证据

- 冻结协议：[TGC_R2B2_RESPONSE_ALIGNMENT_PROTOCOL_ZH_2026-09-22.md](TGC_R2B2_RESPONSE_ALIGNMENT_PROTOCOL_ZH_2026-09-22.md)
- 机器汇总：[summary.json](../../results/tgc_r2b2_response_alignment_20260922/summary.json)
- 逐 cell 指标：[cells.csv](../../results/tgc_r2b2_response_alignment_20260922/cells.csv)
- 阶段汇总：[aggregate.csv](../../results/tgc_r2b2_response_alignment_20260922/aggregate.csv)
- 机制与 gate：[mechanism.csv](../../results/tgc_r2b2_response_alignment_20260922/mechanism.csv)
- 成本：[costs.csv](../../results/tgc_r2b2_response_alignment_20260922/costs.csv)
- 独立审计：[audit.json](../../results/tgc_r2b2_response_alignment_20260922/audit.json)
- 实现：[run_tgc_r2b_response_alignment.py](../../code/v2_llm_fusion/run_tgc_r2b_response_alignment.py)
- 审计：[audit_tgc_r2b_response_alignment.py](../../code/v2_llm_fusion/audit_tgc_r2b_response_alignment.py)
- 汇总：[summarize_tgc_r2b_response_alignment.py](../../code/v2_llm_fusion/summarize_tgc_r2b_response_alignment.py)
