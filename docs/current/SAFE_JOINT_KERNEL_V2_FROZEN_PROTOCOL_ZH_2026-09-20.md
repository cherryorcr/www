# Block-Preserving Joint-Kernel V2 冻结协议

日期：2026-09-20  
状态：实验前冻结；本文件只定义本轮实验，不包含结果。

## 1. 目的

本轮只在 Cora 和 Citeseer 小图上检查四类问题：

1. 联合核选边是否真正实现了预期的度、覆盖和旧块传播约束；
2. X、文本 T、源图签名 G 及 X+T 分量分别是否有可辨别作用；
3. graph signature 的 FP16/64 维 sketch 是否能降低持久记忆字节，而不改变节点语义字段；
4. 在不使用 report-dev 标签做构造选择的前提下，候选是否值得进入新的确认矩阵。

本轮不加入 response selector、dynamic budget、old-row refresh、attention 或大图扩展。C0 保持冻结，上一轮 Phase-12 结果不被改写。

## 2. 共同协议

| 项目 | 冻结值 |
|---|---|
| 数据集 | Cora、Citeseer |
| 顺序 | canonical、order19102；均为开发条件，不是独立泛化确认 |
| 阶段 | 3、5 |
| 压缩率 | 1%、5% |
| 预算 | `floor(rate * 截至当前阶段 pool 节点数)`，严格增长 |
| C0 行、X、文本、mass | 所有等节点候选逐字段相同 |
| C0 文本 | joint_g5_c320，拼接后冻结 MiniLM 重编码 |
| X 算子 | Mean-X |
| 旧记忆 | 行、文本、X、mass、签名和旧旧边不可改写 |
| 新边预算 | 与同条件 C0 相同；operator-stable 使用每新行两个单向旧邻居，保持交叉边 entry 数相等 |
| Student | 两层 GCN/GraphCLIP student，hidden/out=384，冻结 MiniLM，Adam lr=.001，双向 InfoNCE，从头训练 |
| Batch | `min(16, rows)` |
| Steps | `protocol_v3.student_steps(rows)`，候选间相同；byte 对照也沿用 C0 steps |
| Seed | 100--105 |
| 指标 | report-dev 上所有已见 prompt 的 pooled Macro-F1，同时报告 old/new F1、std、阶段曲线 |
| 标签使用 | 构造和选边不读取 report-dev/test 标签；标签只用于最后统一评估 |
| 成本 | 共同 C0 构造成本计入所有候选；记录 exclusive construction、student wall time、持久 payload bytes |

## 3. 候选与控制

基础 topology arms：

- `c0`：当前冻结持续追加控制；
- `k_center`：同协议强 coreset 参照；
- `joint_kernel`：X/T/G 权重 `.4/.3/.3` 的原始联合核；
- `degree_balanced`：新块内部硬 cap=2，并对跨块旧端点使用 `ceil(new/old)` 容量；
- `coverage`：在 cap=2 可行约束下，对边中点联合核相似度加入确定性冗余惩罚；
- `operator_stable`：内部核选边保持不变，只将交叉边改为每新行接收两个旧行的单向边；
- `drift_gated`：以同一旧记忆为起点比较 joint candidate 与 C0 append 的旧块归一化漂移和旧节点有效消息漂移；若候选任一项更差则回退 C0。该 gate 不看 student 指标。

canonical/5% 额外分量和存储 arms：

- `kernel_x`、`kernel_t`、`kernel_g`、`kernel_xt`（权重 4/7、3/7、0）；
- `node_fp16`、`node_sketch64`：同节点数，仅压缩 persistent graph signature；
- `byte_fp32`、`byte_fp16`、`byte_sketch64`：以 C0 每阶段 logical payload bytes 为上限，降低新增 rows 直到不超过上限；沿用 C0 student steps，明确报告行数变化。

所有候选仍使用合法当前 delta 生成图签名，不读取历史 raw nodes。signature sketch 使用固定 seed=20260920 的 Rademacher projection；其投影维度、dtype、metadata bytes 均记入审计。

## 4. 事前判定

本轮只是机制筛查。候选只有同时满足下列条件，才允许进入新的独立顺序/seed confirmation：

1. 至少 3/4 的数据集×顺序组合中平均 final Macro-F1 不低于 C0，且至少 2 个组合有不低于 1 pp 的提升；
2. 平均 final 提升不低于 +1 pp，最差组合不低于 -2 pp；
3. pooled old F1 最差不低于 C0 -2 pp，且新类 F1 或阶段 plasticity 有可重复改善；
4. 训练成本不超过 C0 的 2 倍；
5. 若声称存储改进，必须在 equal-node 和 equal-byte 两种表中同时报告，不能以相同节点数替代相同字节预算；
6. 所有硬约束审计、checkpoint reload 和 deterministic reconstruction 通过。

若不满足，保留 C0 为当前主线，并将该候选记录为机制诊断或负结果；不扩展 36 条件，不宣称 SOTA。

## 5. 完整性要求

实验目录必须先写入 `freeze.json`，并保存源文件 hash、数据 hash、文本 cache hash、PyTorch 版本、GPU、student seed、阶段、rate、arm 和 expected fit count。恢复运行时这些字段必须逐项一致。每个 memory、construction audit、fit checkpoint、prediction reload 和汇总表必须可由 hash 追溯。

## 6. 解释边界

本轮若出现收益，只能说明对应控制在当前小图开发条件下有信号；若失败，不能单独推出大图无效。联合核的持久签名约束、拓扑改变和 student 任务效用必须分开解释，不能把更多 bytes、不同 rows 或更多训练更新造成的差异归因给算子本身。
