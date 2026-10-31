# 小图严格 X+P：construction plan seed 稳定性

固定类别顺序、memory budget、X/P 配置、Student seeds 和 report 规则，只将 `plan_seed` 从 9320 改为 9321。该实验衡量 support batch plan 的敏感性，不是新的方法或参数调优。

| 数据集 | 顺序 | seed 9320 Final F1 | seed 9321 Final F1 | 差值 | seed 9320 Accuracy | seed 9321 Accuracy | 新类 F1 差值 |
|---|---|---:|---:|---:|---:|---:|---:|
| cora | order19611 | 49.308 | 51.286 | +1.978 | 56.765 | 57.011 | +5.491 |
| cora | order19612 | 45.750 | 48.449 | +2.699 | 54.736 | 57.934 | +1.913 |
| citeseer | order19611 | 55.937 | 56.135 | +0.198 | 60.502 | 61.129 | -0.903 |
| citeseer | order19612 | 54.711 | 53.077 | -1.634 | 61.546 | 59.561 | -2.296 |

四个 cell 的平均 Final F1 变化为 `+0.810 pp`，正向 cell 为 `3/4`。如果该变化量接近 Student 数值方差，后续应报告 construction seed 方差，不能只展示单个 plan seed。

JSON 产物：`results/bounded_xp_final_evidence_20261001/small_xp_plan_seed_stability.json`。
