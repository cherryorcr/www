# Role-Match Synthetic-Margin Gate 验收报告

> **修订提示（2026-09-24）：本报告依赖的旧 `role_match` candidate 存在 payload/source coverage 不守恒，已被 [TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md](TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md) supersede。** gate 数字不应作为有效 role 机制证据。

日期：2026-09-24  
状态：完成六顺序；能回退部分候选，但不能稳定预测下游 Macro-F1

## 1. 目的

固定权重 `role_match` 在六个 stage-1 cell 上平均有收益，但存在明显负尾部。本轮测试一个保守回退机制：

1. 生成 `c0_320`、semantic `matched` 和 `role_match`；
2. 使用独立 gate Student seeds `300,301` 训练每个候选；
3. 只读取合成 memory 上的 synthetic class margin，不读取 report/test 标签和 report F1；
4. 选择 synthetic margin 最大的候选；
5. 用独立 report seeds `150..155` 评估选中的 memory。

## 2. 协议与审计

| 项目 | 设置 |
|---|---|
| 数据集/顺序 | Cora/Citeseer 的 `order19601/19602/19603` |
| 阶段 | stage 1 |
| gate seeds | `300,301` |
| report seeds | `150..155` |
| 选择信号 | synthetic margin only |
| report labels used for selection | false |
| 文本 | top-5，显式 320 字符，MiniLM re-encode |
| 拓扑 | C0 topology bitwise copy |
| rows | 与 C0 相同 |
| 历史 raw graph | 不访问历史完整 raw graph |

### 关于 mass/source partition

role matching 会把 proposal 内容写入新的同类 slots，因此 `mass` 元数据总和不一定等于 `c0_320` 的 mass 总和；这不是本轮 Student F1 的隐变量，因为 `memory_to_trainable` 返回的 `weights` 没有进入当前 `train_student` 的 InfoNCE 计算，训练只使用 `x`、`text_emb` 和 `edge_index`。本轮真正固定的是：

- 总 synthetic rows；
- proposal 数量与 proposal source piece multiset；
- C0 topology；
- Student 协议。

因此结果应解释为“同一组 prototype 内容替换不同同类 slots 的归因实验”，不能写成严格的 raw source partition 保持实验。

## 3. 选择结果

| 数据集/顺序 | gate 选择 | synthetic margin 最大者 | 独立 report 相对 c0-320 |
|---|---|---:|
| Cora/order19601 | `c0_320` | c0 | `0.000 pp` |
| Cora/order19602 | `role_match` | role | `+2.560 pp` |
| Cora/order19603 | `role_match` | role | `+5.055 pp` |
| Citeseer/order19601 | `c0_320` | c0 | `0.000 pp` |
| Citeseer/order19602 | `matched` | matched | `+2.839 pp` |
| Citeseer/order19603 | `matched` | matched | `-1.702 pp` |

汇总：

| 指标 | 结果 |
|---|---:|
| 六 cell 平均 | `+1.459 pp` |
| 36 个 report paired seed 平均 | `+1.459 pp` |
| report 正 seed | 17/36 |
| 最差 report seed | `-5.493 pp` |
| 正 cell 数 | 3/6 |

## 4. 结论

synthetic-margin gate 有两个作用：

- 它成功回退了 Cora/order19601 和 Citeseer/order19601 的 role candidate；
- 它保留了 Cora/order19602、Cora/order19603 的明显正收益。

但它仍然错误地保留了 Citeseer/order19603 的 semantic matched，而该 cell 相对 c0 为 `-1.702 pp`；同时它回退掉了 Cora/order19601 中真实 report 上为正的候选。因此 synthetic margin 与最终 zero-shot Macro-F1 之间存在明显 proxy mismatch。

这与此前 teacher message loss 的结论一致：局部 synthetic/teacher proxy 可以解释一部分机制，但不能直接作为 downstream 选择准则。

## 5. 晋升判定

本轮不把 gate 版本升为主方法：

- 只有 3/6 cell 正；
- 最差 seed `-5.493 pp`；
- gate 不能稳定预测 report F1；
- 选择信号仍然依赖短 Student 训练轨迹。

当前最可靠的论文主线仍是 C0；role-aware matching 保留为有平均收益但未稳定的候选和机制分析。

## 6. 后续建议

不再继续扫描 synthetic-margin 阈值或 gate seed。若还要推进，必须先解决 selector proxy mismatch：

1. 用独立 gate seeds 和多长度 response curve，而不是单一 synthetic margin；
2. 同时约束 old/new message drift 和 Student 训练稳定性；
3. 预注册回退规则，禁止按最终 F1 选择；
4. 在 stage 2 和更长链上先做小规模确认；
5. 若仍不能把最差 cell 控制在 `-2 pp` 内，则停止该候选，不上大图。

## 7. 产物

- Runner：[run_tgc_role_match_gate.py](../../code/v2_llm_fusion/run_tgc_role_match_gate.py)
- Remote wrapper：[tgc_role_match_gate_remote.py](../../tools/tgc_role_match_gate_remote.py)
- 结果目录：[tgc_role_match_gate_20260924](../../results/tgc_role_match_gate_20260924/)
- Role-match 六顺序报告：[TGC_ROLE_AWARE_MATCHING_CONFIRMATION_ZH_2026-09-24.md](TGC_ROLE_AWARE_MATCHING_CONFIRMATION_ZH_2026-09-24.md)
