# C3 跨阶段传播算子诊断报告

日期：2026-09-18  
状态：单阶段传播诊断完成；没有候选晋升为默认主方法。  
共同协议：C0 的 5% fixed-rate growing memory、dispersion allocation、Mean-X、`joint_g5_c320`、三阶段、六个 student seeds（100--105）。只改变新旧 block 的 cross message operator。

## 1. 为什么做 C3

frozen-old 只冻结旧旧边，但新增跨边仍会改变归一化传播。C3 将跨边拆成方向和强度问题：

| 算子 | 传播方向 | 目的 |
|---|---|---|
| `current_bidir` | 旧到新、新到旧 | C0 当前控制 |
| `new_receive_old` | 旧到新 | 让新 synthetic rows 借用历史语义，避免新行扰动旧行 |
| `old_receive_new` | 新到旧 | 直接测试旧类吸收新类消息的代价 |
| `bounded_bidir` | 双向但 cross weight=0.25，保留历史加权支持 | 降低跨边扰动，同时保留双向交互 |

图卷积约定为 `adj[row]` 接收 `adj[row, column] * x[column]`，所以 `new_receive_old` 的旧到新边只改变新行的消息，不改变旧行的 cross message。

## 2. 实现审计

### 2.1 共同控制

- block、X、文本、rows、旧旧边、类别顺序和 student 完全固定；
- `current_bidir` stage 0/1/2 复现 C0：Cora final 49.8518，Citeseer final 59.1747；
- 72 个 student fits 完成，未读取 test labels；
- 每个算子使用相同 18 fits/数据集和相同 student optimizer steps。

### 2.2 bounded-bidir 修复

首版实现会在 stage 2 丢弃上一阶段 `adj_values`，并可能重复加入 self-loop，不能用于结论。修复后：

- 上一阶段加权 support 原样保留；
- 只为当前新 block 构造标准局部归一化 support；
- 只为本轮 cross edges 添加衰减权重；
- `edge_index` 与 `adj_values` 一一对应，避免重复 self-loop。

因此下面 bounded-bidir 结果只采用修复后的重跑产物。

## 3. 最终阶段结果

指标为 final seen-class dev cumulative Macro-F1 六个 student seed 的均值。

| 数据集 | current bidir | new receive old | old receive new | bounded bidir |
|---|---:|---:|---:|---:|
| Cora | **49.85** | **53.37** | 42.94 | 48.40 |
| Citeseer | **59.17** | 55.25 | 56.50 | 54.23 |

相对当前双向控制：

| 数据集 | new receive old | old receive new | bounded bidir |
|---|---:|---:|---:|
| Cora | **+3.52** | -6.91 | -1.45 |
| Citeseer | -3.92 | -2.68 | -4.95 |

最终阶段 old/new F1：

| 数据集 | 算子 | old F1 | new F1 |
|---|---|---:|---:|
| Cora | current bidir | 65.44 | 29.06 |
| Cora | new receive old | 64.38 | 38.70 |
| Cora | old receive new | 58.20 | 22.60 |
| Cora | bounded bidir | 58.27 | 35.24 |
| Citeseer | current bidir | 66.46 | 44.61 |
| Citeseer | new receive old | 60.97 | 43.81 |
| Citeseer | old receive new | 63.77 | 41.95 |
| Citeseer | bounded bidir | 63.39 | 35.90 |

## 4. 传播漂移解释

在 current bidir 下，Cora stage 1 的旧传播相对漂移为 0.384，新到旧 cross message 相对量为 0.447；Citeseer 对应为 0.352 和 0.476。`new_receive_old` 将旧行接收到的新行消息置零，因此 old propagation drift 为 0；但它并不保证新类一定更好，因为新行只得到历史消息，无法跨数据集稳定表达新类自身结构。

`old_receive_new` 让旧行接收新信息，Cora/Citeseer 的 old retention 和 final F1 都变差，支持“新到旧传播容易破坏历史表征”的机制判断。

修正后的 bounded-bidir 将 cross message 相对量降到 Cora 0.061、Citeseer 0.050，但同时 final F1 下降，说明简单衰减并没有保留当前双向传播的有效信息；它不是当前算子的可靠替代。

## 5. 结论

1. 跨边方向确实是重要算子，而不是无关实现细节。
2. `old_receive_new` 不适合作为默认组件，两个数据集均退化。
3. `new_receive_old` 在 Cora 有明显塑性收益，但在 Citeseer 退化，不能晋升为跨数据集主方法。
4. 简单 bounded cross weight 也未通过门禁；降低传播强度会同时削弱有益的交互。
5. C0 当前双向 cross-edge 规则继续保留为默认控制；C3 作为机制分析和负结果进入论文。

## 6. 后续含义

不继续扫描大量固定 cross weight。若 P5 长链实验显示旧类遗忘随阶段增加，才研究有保持约束的方向选择，例如按 old-retention 风险关闭 new-to-old 消息、按新类覆盖缺口保留 old-to-new 消息，并要求在独立顺序上确认。下一步仍按路线进入 C1 多原型 X，不能把 C3 的 Cora 局部收益与其它组件组合后重新包装成 v5。

## 7. 产物

- C3 runner：`code/v2_llm_fusion/run_c3_propagation_diagnostic.py`
- 初始四臂结果：`results/c3_propagation_diagnostic_20260918/`
- 修正 bounded-bidir 结果：`results/c3_propagation_diagnostic_20260918_fixed/`
- P4 报告：`docs/current/P4_MEMORY_MANAGEMENT_COMPLETION_ZH_2026-09-18.md`

