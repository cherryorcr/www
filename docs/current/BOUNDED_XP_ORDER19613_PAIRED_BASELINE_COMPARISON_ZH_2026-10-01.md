# Bounded X+P：order19613 独立顺序与基线比较

> 这是额外类别顺序验证，不替换两顺序主表。所有方法使用相同 Cora/Citeseer、`order19613`、三阶段、5% cumulative memory、原始 Student 和 6 个 report seeds。

## 1. Final 与阶段平均结果

| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 | 阶段平均 Accuracy | memory rows |
|---|---|---:|---:|---:|---:|---:|
| cora | Bounded X+P | 47.42 ± 2.11 | 55.78 ± 2.00 | 45.20 | 54.00 | 94 |
| cora | C0 append-edge anchor | 49.18 ± 4.31 | 57.13 ± 3.18 | 44.97 | 53.51 | 94 |
| cora | CaT-MMD-TAG adapted | 54.03 ± 1.24 | 61.69 ± 1.19 | 47.24 | 50.79 | 94 |
| cora | GECC-hard-TAG adapted | 49.15 ± 0.82 | 53.38 ± 0.79 | 50.69 | 47.96 | 94 |
| cora | k-center native append | 43.98 ± 2.66 | 47.17 ± 4.20 | 38.94 | 41.68 | 94 |
| cora | medoid native append | 53.67 ± 1.90 | 60.70 ± 1.26 | 48.45 | 52.12 | 94 |
| citeseer | Bounded X+P | 53.16 ± 2.47 | 58.46 ± 1.97 | 47.80 | 45.48 | 111 |
| citeseer | C0 append-edge anchor | 52.20 ± 1.20 | 57.58 ± 1.70 | 47.66 | 45.07 | 111 |
| citeseer | CaT-MMD-TAG adapted | 60.09 ± 1.32 | 64.11 ± 1.52 | 54.07 | 49.21 | 111 |
| citeseer | GECC-hard-TAG adapted | 48.84 ± 3.90 | 53.24 ± 3.66 | 41.51 | 39.01 | 111 |
| citeseer | k-center native append | 52.37 ± 1.47 | 59.09 ± 1.55 | 49.52 | 53.50 | 111 |
| citeseer | medoid native append | 53.73 ± 2.49 | 57.31 ± 1.44 | 43.58 | 43.09 | 111 |

## 2. X+P 相对同顺序方法的差值

正值表示 X+P 更高；该表是每个数据集内的配对描述，不是跨数据集显著性检验。

| 数据集 | 对比方法 | X+P - 对比 Final F1 | X+P - 对比 Accuracy | X+P - 对比阶段平均 F1 |
|---|---|---:|---:|---:|
| cora | C0 append-edge anchor | -1.76 | -1.35 | +0.24 |
| cora | CaT-MMD-TAG adapted | -6.61 | -5.90 | -2.03 |
| cora | GECC-hard-TAG adapted | -1.73 | +2.40 | -5.48 |
| cora | k-center native append | +3.44 | +8.61 | +6.26 |
| cora | medoid native append | -6.24 | -4.92 | -3.25 |
| citeseer | C0 append-edge anchor | +0.96 | +0.89 | +0.15 |
| citeseer | CaT-MMD-TAG adapted | -6.93 | -5.64 | -6.26 |
| citeseer | GECC-hard-TAG adapted | +4.32 | +5.22 | +6.29 |
| citeseer | k-center native append | +0.79 | -0.63 | -1.72 |
| citeseer | medoid native append | -0.57 | +1.15 | +4.23 |

## 3. 解释

- Cora/order19613 上，X+P 的 Final F1 为 47.42；它高于 GECC-hard-TAG 与 k-center，但低于 CaT-MMD-TAG 和 medoid。Accuracy 为 55.78，高于 CaT、GECC 和 medoid，低于 k-center。
- Citeseer/order19613 上，X+P 的 Final F1 为 53.16；它低于 CaT、k-center 和 medoid，高于 GECC。Accuracy 为 58.46；它低于 CaT 和 k-center，高于 GECC 与 medoid。
- 因而该顺序支持“X+P 有方向性收益、但不是普遍 SOTA”，也说明 F1 与 Accuracy 对图文 memory 的偏好并不完全一致。
- C0 只作为内部 append-edge anchor，用来衡量 outer optimization 的增量；公开比较仍应以适配后的 CaT、GECC、k-center 和 medoid 为主，并在论文中明确适配边界。

## 4. 证据限制

order19613 曾在此前拓扑实验中出现，因此这是独立于当前 X+P 参数选择的额外顺序验证，但不能称为完全未使用过的盲测顺序。CaT/GECC 仍是本项目 Student/continual 接口下的机制适配，不是官方端到端复现。

JSON：`D:/Aprojects/continue_tag_condensation/results/bounded_xp_final_evidence_20261001/order19613_xp_baseline_comparison.json`
