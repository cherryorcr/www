# 下一步：Response-Retention 最小验证 Gate

状态：**已整理方案，尚未实现，尚未运行实验。** 本页把现有报告中已经确定的下一步合并为一个可执行入口；若与日期更晚的实验记录冲突，以日期更晚者为准。

## 要解决的具体失败

Text-Anchored ASM 已证明 graph-text correspondence 是有效干预点，但还没有解决严格递归失败：

- stage 1 的 ASM 和 Text-Anchored 轨迹均被 dev 选择回 `outer=0`，说明当前优化没有稳定优于初始化；
- stage 2 的收益没有跨 condensation seed 稳定；
- 当前 loss 约束参数轨迹和图文对应，却没有直接保持旧阶段 teacher 的预测响应，也没有明确协调历史响应与当前增量响应。

因此下一版只增加一个变量：

\[
L = L_{ASM} + \lambda_t L_{text-anchor} + \lambda_r L_{response-retention}.
\]

## 最小候选实现

先实现一种响应目标并在代码和结果中记录选择：

1. **class-prompt logits**：保存上一阶段 teacher 对旧 synthetic rows 的 graph-to-class-prompt logits；新 rows 对齐当前 delta teacher 的 logits。
2. **pairwise response matrix**：保存 graph-to-text 或 graph-to-class-prompt 的成对响应关系，用距离损失保持相对结构。

首轮优先采用 class-prompt logits，因为输出维度和语义清楚，也更容易审计。开始实现前必须明确三项口径：历史与当前 teacher 如何生成 target、保存的 response 是否计入持续 memory 预算、`lambda_r` 的 dev 候选集合。任何额外 teacher 状态都要在 artifact manifest 中列出字节数。

## 实验顺序

1. 只跑 **stage 1 gate**，比较 `outer=0`、现有 `lambda_r=0` dev-best 与 response-retention 候选。
2. stage 1 通过后固定响应形式和权重，再跑完整 **stage 1→2 dev-only 严格递归**。
3. 所有方法与超参数冻结后，才用独立 student seeds 读取 test；同时报告 condensation-seed 分解和 paired win rate。

本 gate 暂不同时加入 GraphCLIP、transport、跨阶段边、额外 prompt 或动态预算，以便把结果归因到 response retention。

## 通过条件

stage 1 的最低通过条件：

- seed 44 和 seed 45 的 dev 都选择 `outer>0`；
- 两个 seed 的累计 dev 都不低于各自 `outer=0` 初始化。

进入完整递归后，stage 1 和 stage 2 的累计、历史、当前 dev 都应在两个 condensation seeds 上不劣于各自 `lambda_r=0` dev-best。只有达到该条件，才进入冻结后的 test 确认。

## 对照和解释边界

报告必须保留 Append、Append-joint、Protected-joint、Union 和 Full-history 的已有参照，并同时给出：

- 相对 `outer=0` 的变化，判断凝聚优化本身是否有效；
- 相对 `lambda_r=0` 的变化，隔离 response-retention 的贡献；
- stage 1 与 stage 2 分开结果，避免 stage 2 偶然补偿 stage 1 失败。

Cora 的 test、confirmation 和 future 已在开发中访问。即使本 gate 通过，Cora 数值也只能作为开发证据；最终论文结论需要新的未访问数据集。

## 依据

- [当前进展总结](../current/CURRENT_PROGRESS_SUMMARY_ZH_2026-09-08.md)
- [Text-Anchored ASM 技术报告](../current/TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md)
- [逐项实验讲解](../current/EXPERIMENT_WALKTHROUGH_ZH_2026-09-08.md)
- [执行协议](EXECUTION_PROTOCOL_2026-09-07.md)
