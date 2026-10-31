# 大图 base+4 五层比较：持续追加、coreset、全历史重凝聚与完整 raw 参考

日期：2026-09-26

## 1. 结论摘要

本报告在三个大图的统一 `base+4`、固定 5% memory rate、report-dev mod=3、Student batch=16、Student seeds=100--105 协议下，比较持续合成追加、真实节点 coreset、全历史合成重凝聚和完整 raw graph 参考。

`semantic-recursive` 在 12 个增量 stage cells 上相对同进程 C0 平均提升 `+0.858 pp`，9/12 个 cell 为正，新类平均提升 `+1.764 pp`；最差 cell 为 `-2.381 pp`。它是大图上的方向性有效机制，不是已经稳定超过所有基线的主方法。

完整 raw graph 和 full-history synthetic 允许读取累计历史 raw 并重写历史记忆，因此是离线性能参考，不是合法 continual constructor。静态 TAGSAM 只有小图同协议适配，本报告不把小图分数挪入大图主表。

## 2. 五层含义与权限边界

| 层次 | 方法 | 历史 raw | 可改写旧 memory | 用途 |
|---|---|---|---|---|
| continual | C0 / semantic-recursive | 否 | C0 旧前缀保持 | 回答持续凝聚能否追加并支持重训 |
| continual reference | random / k-center / real-node FPS | 否 | 不适用真实节点回放 | 强 coreset 参考，不是合成凝聚 |
| offline synthetic | full-history synthetic | 是 | 是 | 相同构造族的全历史参考 |
| offline raw | raw equal / raw long | 是 | 是 | 完整真实图的同步数与长训参考 |

raw_equal 使用与 synthetic student 相同的 2048 steps；raw_long 固定 8192 optimizer steps。8192 不是理论上界，也不代表充分收敛。

## 3. 共享协议审计

审计文件的 all_pass = `True`。所有方法均完成五个 stage，阶段类别、预算、report-dev、Student batch 和 Student seed 与 C0 一致；full-history/raw 的 offline 权限被单独标记。

### arxiv: stage scores and C0-relative deltas

| method | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 | incremental mean delta | old delta | new delta | positive | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 / continual append | 29.618 | 26.519 | 22.153 | 21.610 | 18.817 | 0.000 | 0.000 | 0.000 | 0/4 | 0.000 |
| semantic-recursive append | 29.618 | 26.098 | 23.616 | 22.706 | 19.227 | 0.637 | -0.157 | 2.971 | 3/4 | -0.421 |
| random real-node coreset | 26.495 | 23.424 | 20.998 | 20.953 | 18.169 | -1.389 | -2.443 | -1.284 | 0/4 | -3.095 |
| k-center coreset | 27.807 | 24.297 | 21.471 | 18.847 | 16.667 | -1.955 | -2.385 | -1.482 | 0/4 | -2.764 |
| real-node FPS coreset | 26.220 | 23.215 | 20.968 | 19.538 | 18.058 | -1.830 | -1.770 | -2.374 | 0/4 | -3.304 |
| full-history synthetic re-condensation | 31.827 | 27.603 | 21.300 | 21.256 | 20.544 | 0.401 | -0.712 | -4.372 | 2/4 | -0.853 |
| full-history synthetic (GPU implementation) | 30.398 | 27.192 | 20.707 | 20.622 | 19.692 | -0.222 | -1.892 | -0.735 | 2/4 | -1.446 |
| complete raw graph, equal steps | 30.774 | 27.063 | 24.774 | 21.917 | 18.842 | 0.874 | 1.279 | -1.563 | 4/4 | 0.025 |
| complete raw graph, 8192 steps | 30.922 | 27.532 | 24.894 | 23.380 | 20.155 | 1.715 | 2.256 | 2.409 | 4/4 | 1.013 |

### arxiv: recorded cost

| method | construction seconds | student seconds | raw wall seconds | reported total |
|---|---:|---:|---:|---:|
| C0 / continual append | 0.0 | 145.1 | 0.0 | 145.1 |
| semantic-recursive append | 0.0 | 144.3 | 0.0 | 144.3 |
| random real-node coreset | 0.4 | 152.6 | 0.0 | 153.0 |
| k-center coreset | 0.3 | 146.8 | 0.0 | 147.1 |
| real-node FPS coreset | 0.3 | 146.9 | 0.0 | 147.2 |
| full-history synthetic re-condensation | 612.6 | 148.7 | 0.0 | 761.4 |
| full-history synthetic (GPU implementation) | 102.8 | 156.1 | 0.0 | 258.9 |
| complete raw graph, equal steps | 0.0 | 1669.3 | 1669.3 | 1669.3 |
| complete raw graph, 8192 steps | 0.0 | 6662.9 | 6662.9 | 6662.9 |

### bookchild: stage scores and C0-relative deltas

| method | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 | incremental mean delta | old delta | new delta | positive | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 / continual append | 17.093 | 16.960 | 14.746 | 14.389 | 12.607 | 0.000 | 0.000 | 0.000 | 0/4 | 0.000 |
| semantic-recursive append | 17.093 | 17.728 | 13.195 | 15.281 | 13.313 | 0.203 | -0.361 | 2.241 | 3/4 | -1.552 |
| random real-node coreset | 16.603 | 18.231 | 12.210 | 12.799 | 13.136 | -0.582 | 0.550 | 8.899 | 2/4 | -2.536 |
| k-center coreset | 15.820 | 16.370 | 13.229 | 14.181 | 11.712 | -0.803 | -0.827 | 0.789 | 0/4 | -1.518 |
| real-node FPS coreset | 14.422 | 16.805 | 14.184 | 12.729 | 13.480 | -0.376 | 0.456 | 5.036 | 1/4 | -1.660 |
| full-history synthetic re-condensation | 16.045 | 17.403 | 13.204 | 13.355 | 12.407 | -0.583 | -2.774 | -1.042 | 1/4 | -1.543 |
| full-history synthetic (GPU implementation) | 18.940 | 18.672 | 14.023 | 14.449 | 14.425 | 0.717 | -0.131 | 1.301 | 3/4 | -0.723 |
| complete raw graph, equal steps | 21.016 | 23.811 | 18.375 | 18.445 | 18.018 | 4.987 | 11.112 | 15.368 | 4/4 | 3.629 |
| complete raw graph, 8192 steps | 22.314 | 21.616 | 18.312 | 18.187 | 17.217 | 4.158 | 9.855 | 16.001 | 4/4 | 3.566 |

### bookchild: recorded cost

| method | construction seconds | student seconds | raw wall seconds | reported total |
|---|---:|---:|---:|---:|
| C0 / continual append | 0.0 | 136.7 | 0.0 | 136.7 |
| semantic-recursive append | 0.0 | 136.8 | 0.0 | 136.8 |
| random real-node coreset | 0.3 | 143.9 | 0.0 | 144.1 |
| k-center coreset | 0.1 | 139.6 | 0.0 | 139.7 |
| real-node FPS coreset | 0.1 | 137.4 | 0.0 | 137.5 |
| full-history synthetic re-condensation | 141.4 | 139.6 | 0.0 | 281.0 |
| full-history synthetic (GPU implementation) | 28.2 | 150.4 | 0.0 | 178.6 |
| complete raw graph, equal steps | 0.0 | 769.4 | 769.4 | 769.4 |
| complete raw graph, 8192 steps | 0.0 | 3074.8 | 3074.8 | 3074.8 |

### sportsfit: stage scores and C0-relative deltas

| method | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 | incremental mean delta | old delta | new delta | positive | worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 / continual append | 26.482 | 26.758 | 31.144 | 27.349 | 25.013 | 0.000 | 0.000 | 0.000 | 0/4 | 0.000 |
| semantic-recursive append | 26.482 | 32.981 | 28.763 | 27.384 | 28.076 | 1.735 | 2.158 | 0.080 | 3/4 | -2.381 |
| random real-node coreset | 27.759 | 28.539 | 30.815 | 29.253 | 26.756 | 1.275 | 1.105 | 1.392 | 3/4 | -0.328 |
| k-center coreset | 27.896 | 34.269 | 33.349 | 28.282 | 27.840 | 3.369 | 3.916 | -1.246 | 4/4 | 0.934 |
| real-node FPS coreset | 33.736 | 28.205 | 29.620 | 30.212 | 26.147 | 0.980 | 2.335 | 0.977 | 3/4 | -1.524 |
| full-history synthetic re-condensation | 27.930 | 28.639 | 25.648 | 25.848 | 26.467 | -0.915 | -2.176 | -6.439 | 2/4 | -5.496 |
| full-history synthetic (GPU implementation) | 23.108 | 29.019 | 30.834 | 25.899 | 28.357 | 0.962 | -0.186 | -6.586 | 2/4 | -1.449 |
| complete raw graph, equal steps | 27.122 | 32.733 | 31.849 | 34.345 | 26.769 | 3.858 | 0.574 | 0.032 | 4/4 | 0.706 |
| complete raw graph, 8192 steps | 28.102 | 25.794 | 36.069 | 35.318 | 29.502 | 4.105 | -0.314 | 0.465 | 3/4 | -0.964 |

### sportsfit: recorded cost

| method | construction seconds | student seconds | raw wall seconds | reported total |
|---|---:|---:|---:|---:|
| C0 / continual append | 0.0 | 159.0 | 0.0 | 159.0 |
| semantic-recursive append | 0.0 | 159.1 | 0.0 | 159.1 |
| random real-node coreset | 0.6 | 184.7 | 0.0 | 185.3 |
| k-center coreset | 0.5 | 157.8 | 0.0 | 158.2 |
| real-node FPS coreset | 0.4 | 157.4 | 0.0 | 157.8 |
| full-history synthetic re-condensation | 586.9 | 149.0 | 0.0 | 735.9 |
| full-history synthetic (GPU implementation) | 99.7 | 156.2 | 0.0 | 255.9 |
| complete raw graph, equal steps | 0.0 | 1549.1 | 1549.1 | 1549.1 |
| complete raw graph, 8192 steps | 0.0 | 6185.7 | 6185.8 | 6185.8 |

## 4. 结果解释

1. C0 是当前共同控制。它提供最稳定的低成本 continual 参照，不因某个单独阶段低于其它方法而被替换。
2. semantic-recursive 的收益不是来自更多 rows、更多 Student steps 或改变拓扑：它与 paired C0 使用相同预算、边、batch、steps 和 seeds；区别在当前新增 block 的 structural proposal 与 semantic conserved payload assignment，并递归使用上一阶段 semantic memory。
3. k-center 等真实节点 coreset 在 SportsFit 的部分阶段较强，但在 arXiv/BookChild 并不稳定；这说明“覆盖几何”与“可重训合成 memory”不是同一目标，不能以单一数据集的最好结果替换主线。
4. full-history synthetic 和 complete raw 是性能参照，不是 continual 方法。它们回答的是“如果可以回读历史原图并重写记忆/直接训练，性能位置在哪里”。
5. raw_equal 与 raw_long 的差异应解释为训练预算敏感性，而不是凝聚算法收益。

## 5. 当前可写入论文的结论

> 在统一大图 `base+4` 协议下，semantic-recursive append 相对冻结 C0 在 12 个增量 stage cells 上平均提升 0.858 个百分点，并将 new-class Macro-F1 平均提高 1.764 个百分点。该收益在 9/12 个 cell 中出现，但最差 cell 下降 2.381 个百分点，说明机制具有跨大图的方向性效用而非无条件优势。完整 raw 与 full-history 结果仅作为离线参考，不能用于宣称 continual SOTA。

## 6. 缺失与后续

- 静态 TAGSAM/STAD 尚未完成同协议大图适配；小图结果不能替代该主表。
- 仍需在独立类别顺序和 construction seed 上确认 semantic-recursive，避免把当前三条固定流解释成普遍泛化。
- 应继续报告 memory rows、构造成本、Student 成本、old/new F1 和最差阶段，而不是只报告最终累计 F1。
- 若要升级为论文主方法，应先完成更强静态 TAG 基线与独立顺序确认；在此之前 C0 是共同冻结基线，semantic-recursive 是有证据的候选机制。

## 7. 证据文件

- 汇总 JSON：`results/tgc_large_baseline_comparison_20260926.json`
- 汇总 Markdown：`results\tgc_large_baseline_comparison_20260926.md`
- 协议审计 JSON：`results/tgc_large_baseline_protocol_audit_20260926.json`
- 协议审计 Markdown：`results\tgc_large_baseline_protocol_audit_20260926.md`
