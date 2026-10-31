# PUMA 低分诊断（2026-10-03）

## 结论

PUMA-inspired 的低分是需要在汇报中主动说明的风险，但当前证据不能把它归因于单一因素。三个 PUMA arm（`full_puma`、`no_pseudo`、`shuffled_pseudo`）都使用相同的 edge-free memory 和同一个 GCN + MiniLM + bidirectional InfoNCE Student，因此现有消融没有 edgeful control，不能单独估计 edge-free memory 的影响。

伪标签准确率来自只读脚本 `code/v2_llm_fusion/diagnose_puma_pseudolabels.py`，对应服务器诊断产物 `results/puma_adapter_formal_20261002/pseudo_label_diagnostic_order19612_{cora,citeseer}.json`；候选节点属于当前 legal pool，未使用 test labels。

当前更直接的结论是：PUMA 的伪标签/传播构造与本实验 Student 的交互对顺序很敏感。Cora `order19612` 中，伪标签在当前 legal candidate 上的审计准确率为 97.6%--99.4%，但 `full_puma` 最终 Macro-F1 只有 9.54；`no_pseudo` 为 13.89，`shuffled_pseudo` 为 17.86。也就是说，低分不能简单解释为伪标签明显错误，且打乱伪标签反而提高分数，提示 synthetic block 的特征-标签对应、类别分配或 edge-free 表示与 InfoNCE Student 存在适配失配。

## 正式结果（stage 2）

| Dataset/order | full_puma | no_pseudo | shuffled_pseudo | 7/6 类均匀随机 Macro-F1 参考 | 观察 |
|---|---:|---:|---:|---:|---|
| Cora/order19611 | 17.41 | 10.06 | 9.10 | 14.29 | full_puma 高于两个消融 |
| Cora/order19612 | 9.54 | 13.89 | 17.86 | 14.29 | full_puma 低于两个消融和随机参考 |
| Citeseer/order19611 | 16.90 | 12.49 | 21.68 | 16.67 | 顺序敏感，shuffled_pseudo 最高 |
| Citeseer/order19612 | 12.53 | 21.63 | 17.08 | 16.67 | full_puma 低于两个消融 |

随机值只是均匀类别的直观参考，不是严格的 Macro-F1 零假设，不能代替正式统计检验。

## 当前可说与不可说

- 可以说：PUMA 独有机制已经真正实现；它在当前 TAG/InfoNCE 适配下表现不稳定，尤其是 Cora `order19612`。
- 可以说：高 confidence 不等于正确的 synthetic graph-text 对齐；legal candidate 的伪标签准确率不能证明 memory 对 Student 有效。
- 不能说：已经证明 edge-free memory 是低分的唯一原因。
- 不能说：PUMA 实现一定错误；目前没有与 PUMA 相同构造、仅改变 topology 的 edgeful 对照。

## 汇报时建议

把 PUMA 作为“机制合规但适配后性能不稳定”的结果报告，并同时展示三个 arm，而不是只报 `full_puma` 一个数字。若组长要求定位 edge-free 的单独影响，再补一个固定 construction/labels、只恢复 topology 的 edgeful control；这应是后续诊断实验，不应改写当前正式结果。
