# TGC-R13 Bounded Student-Response Gate 完成报告

## 1. 结论

R13 针对 R11 的唯一候选 `anchor025` 增加一个无标签 student-response gate：

\[
\text{deploy}(P)=
\begin{cases}
P_{anchor025},&\text{response rule accepts}\newline
P_{C0},&\text{otherwise.}
\end{cases}
\]

它只在 `C0` 和 `anchor025` 之间选择，不修改文本、X 算子、预算或拓扑算子，也不递归使用上一阶段被选择的结果。

最终结论：

- `selected_curve` 和更保守的 `selected_conservative` 在 12 个增量 stage 全部回退 C0；
- `selected_new64` 接受 3/12 个 stage，最终平均比 C0 高 `+0.226 pp`；
- 但最终 new F1 仍低 `-0.273 pp`，阶段平均为 `-0.054 pp`；
- 3 个接受 cell 中，Citeseer/order19601/stage1 的正式 F1 下降约 `-2.90 pp`，存在误接受；
- R13 证明短期 new-only response 不能可靠决定是否启用 R11；
- R11 仍不能进入默认主方法，C0 继续冻结。

## 2. 方法流程

每个增量阶段：

1. 读取 sealed R12 中的 C0 memory 和 R11 `anchor025` memory；
2. 构造当前旧 synthetic memory + 当前 raw delta 的合法 probe support；
3. 对 C0 和 anchor025 分别从相同随机初始化开始训练 scratch student；
4. 在 32、64、128 步读取 old/new/mixed InfoNCE probe loss；
5. 计算 C0 loss 减 candidate loss 的 response gain；
6. 用预先固定的规则选择：
   - `selected_curve`：三时点 mixed gain 平均为正，且 old loss 全程安全；
   - `selected_new64`：64 步 new gain 为正，且 old loss 安全；
   - `selected_conservative`：三个 selector seed 均有正 new gain，且所有时点 old loss 安全；
7. 选择失败则严格回退到 C0；随后才进行正式 report student 训练。

selector 使用 `13000,13001,13002`，report 使用 `150..155`，两者不重叠；selector 不读取 report F1、report labels 或历史完整 raw graph。

## 3. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 顺序 | `order19601/19602/19603` |
| 阶段 | base + 2 incremental |
| memory | 5% growing memory |
| 候选 | C0、R11 `anchor025` |
| selector checkpoints | 32、64、128 |
| selector seeds | 13000、13001、13002 |
| report seeds | 150--155 |
| student | 两层 384 维 GCN/GraphCLIP，冻结 MiniLM，双向 InfoNCE |
| 训练 | batch 16，Adam `1e-3`，`max(50,min(2048,16*rows))` |
| 历史访问 | 旧 synthetic memory + 当前 raw delta；历史 raw reads = 0 |
| 选择 | 先封存 selector 决策，再训练 report student |

实验共有 `2 × 3 × 3 × 5 × 6 = 540` 个 report fit slots，实际执行 180 次，其余为相同 memory 的合法复用。R12 source、memory、checkpoint audit 全部通过；R13 selector replay 额外复算 216 个 response curve points。

## 4. Gate 决策统计

| 策略 | 接受增量 stage | 启用 anchor stage | selector 平均 gain |
|---|---:|---:|---:|
| `selected_curve` | 0/12 | 0/12 | -0.0102 |
| `selected_new64` | 3/12 | 3/12 | -0.0123 |
| `selected_conservative` | 0/12 | 0/12 | -0.0123 |

`selected_new64` 接受的三个 stage 是：

- Cora/order19601/stage1；
- Cora/order19601/stage2；
- Citeseer/order19601/stage1。

## 5. 最终阶段结果

下表为 stage 2 all-prompt pooled Macro-F1，括号为相对同一顺序 C0 的差值。

| 数据集 | 顺序 | C0 | anchor025 | selected_curve | selected_new64 | selected_conservative |
|---|---|---:|---:|---:|---:|---:|
| Cora | order19601 | 50.382 | 51.736 (+1.354) | 50.382 (0) | 51.736 (+1.354) | 50.382 (0) |
| Cora | order19602 | 53.048 | 52.037 (-1.011) | 53.048 (0) | 53.048 (0) | 53.048 (0) |
| Cora | order19603 | 46.152 | 46.645 (+0.493) | 46.152 (0) | 46.152 (0) | 46.152 (0) |
| Citeseer | order19601 | 54.535 | 54.484 (-0.051) | 54.535 (0) | 54.535 (0) | 54.535 (0) |
| Citeseer | order19602 | 52.035 | 54.205 (+2.169) | 52.035 (0) | 52.035 (0) | 52.035 (0) |
| Citeseer | order19603 | 52.933 | 50.829 (-2.104) | 52.933 (0) | 52.933 (0) | 52.933 (0) |

## 6. 汇总结果

| 方法 | 最终 F1 | 最终相对 C0 | old F1 差 | new F1 差 | 阶段平均相对 C0 | 最差最终 cell |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anchor025 | 51.656 | +0.142 | +1.629 | -2.059 | -0.856 | -2.104 |
| selected_curve | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| selected_new64 | **51.740** | **+0.226** | +0.599 | -0.273 | -0.054 | 0.000 |
| selected_conservative | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 7. 结果说明

### 7.1 R11 的稳定化能力是真实的，但范围有限

`selected_new64` 在 Cora/order19601/stage2 启用 anchor 后获得 `+1.354 pp`，复现了 R11 该 cell 的正收益，并且没有启用 Cora/order19602 和 Citeseer/order19603 的明显坏 cell。这说明 gate 能过滤一部分显然危险的 anchor 使用。

### 7.2 new-only response 仍会误接受

Citeseer/order19601/stage1 的 64 步 new response 为正且 old guard 通过，因此启用 anchor；但正式 stage1 F1 为 `56.214`，C0 为 `59.114`，下降约 `2.900 pp`。因此短期 new InfoNCE 改善不等价于长期 new-class Macro-F1 改善。

### 7.3 mixed curve 和 conservative rule 过于保守

两者全部回退 C0，因此最终没有负收益，但也没有保留任何 anchor 的正收益。它们可作为安全回退策略，不能作为性能提升方法。

### 7.4 当前瓶颈不是简单的 gate 阈值

如果只放宽阈值，会重新启用 Citeseer/order19601/stage1 这样的坏候选；如果加强 old guard，则会把 Cora/order19601 的正收益一起拒绝。问题在于 selector 的短期 InfoNCE 目标与最终长程 zero-shot 分类目标不一致，而不是单一阈值没有调好。

## 8. 晋升判定

| 条件 | 要求 | `selected_new64` | 判定 |
|---|---:|---:|---|
| 最终平均相对 C0 | `>= +0.50 pp` | `+0.226` | 未通过 |
| 阶段平均 | `>= 0` | `-0.054` | 未通过 |
| new F1 平均 | `>= 0` | `-0.273` | 未通过 |
| 最差最终 cell | `>= -2 pp` | `0.000` | 通过 |
| 无标签选择 | 必须满足 | 通过 | 通过 |
| selector replay audit | 必须通过 | 216 points 通过 | 通过 |

因此 R13 不能成为新主方法，也不能把 `selected_new64` 的 `+0.226 pp` 写成稳定性能提升。

## 9. 后续决定

1. C0 继续作为默认主方法；R11 作为机制诊断和可选稳定化候选。
2. 暂停继续调 response gate 的阈值；当前 evidence 已说明短期 InfoNCE 不能可靠预测最终 F1。
3. 下一步若继续研究新类塑性，应做真正的 modality attribution：在相同 assignment 下分别固定/替换 X、文本和拓扑，并记录 new-class response、梯度可达性与最终 F1。
4. 只有找到一个能在至少两个数据集、多个顺序上改善 new F1 的构造因素，才重新设计候选生成器；否则保留 C0 + R10/R11/R12/R13 的机制分析和负结果主线。

## 10. 复现文件

- [R13 runner](../../code/v2_llm_fusion/run_tgc_r13_plasticity_gate.py)
- [R13 replay audit](../../code/v2_llm_fusion/audit_tgc_r13_plasticity_gate.py)
- [R13 summarizer](../../tools/summarize_tgc_r13.py)
- [R13 sealed results](../../results/tgc_r13_plasticity_gate_20260924a/)
- [R12 preceding report](TGC_R12_CLASSFLOOR_COMPLETION_ZH_2026-09-24.md)
