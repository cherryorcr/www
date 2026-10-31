# Class-shared 构造随机性复核

日期：2026-10-03

## 1. 目的

本轮只改变构造随机种子，固定其余协议，用来判断上一轮 class-shared real payload 的收益是否依赖单一 CaT 初始化。

固定内容：

- arXiv、SportsFit；
- `base+4` 类别增量顺序；
- 5% cumulative memory；
- edge-aware 2-layer GCN Student、冻结 MiniLM、双向 InfoNCE；
- Student seeds `100--105`；
- CCTP top-5 与 class-shared real payload 成对比较；
- current-delta-only、历史 raw 不回读、旧 synthetic memory 不修改、拓扑不更新。

原配置使用 CaT seed `270927`，payload seed `271003`；复核使用 CaT seed `314159`，payload seed `314235`。

## 2. 结果

| 数据集 | 指标 | 原 seed 差值 | 独立 seed 差值 | 独立 seed 正向 Student seeds |
|---|---:|---:|---:|---:|
| arXiv | Final Macro-F1 | +0.08 | +0.69 | 3/6 |
| arXiv | Accuracy | -4.18 | -2.94 | 1/6 |
| arXiv | Stage-average F1 | +2.79 | +0.95 | 4/6 |
| arXiv | Old F1 | -0.33 | +0.97 | 4/6 |
| arXiv | New F1 | +2.41 | -0.90 | 2/6 |
| SportsFit | Final Macro-F1 | +27.08 | +29.92 | 6/6 |
| SportsFit | Accuracy | +25.53 | +34.45 | 6/6 |
| SportsFit | Stage-average F1 | +33.11 | +36.42 | 6/6 |
| SportsFit | Old F1 | +30.74 | +32.98 | 6/6 |
| SportsFit | New F1 | +6.97 | +13.08 | 6/6 |

独立 seed 的配对 Student 95% 区间：

- arXiv Final F1：`+0.69`，`[-0.97, +2.36]`；
- SportsFit Final F1：`+29.92`，`[+26.76, +33.07]`；
- SportsFit New F1：`+13.08`，`[+3.05, +23.11]`。

逐阶段 Final F1 差值：

| 数据集 | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|---:|
| arXiv | +5.97 | +3.67 | +0.25 | +2.71 | +0.69 |
| SportsFit | +26.82 | +27.68 | +37.89 | +37.31 | +29.92 |

## 3. 结论

1. class-shared 的收益不是完全由原始构造 seed 造成：两个复核数据集的独立 seed Final F1 都为正，SportsFit 的 6 个 Student seed 全部提升。
2. 收益具有明显数据集依赖性。SportsFit 的提升大且稳定；arXiv 的 Final F1 仅小幅提升，配对区间跨 0，Accuracy 下降，New F1 也没有稳定提升。
3. 因此 class-shared real payload 可以作为当前有效候选和机制研究对象，但不能据此宣称普遍稳定优于 CCTP，也不能直接冻结为最终论文主方法。
4. 本轮只验证 construction-seed 稳定性，没有验证独立类别顺序泛化；类别顺序和 BookChild 仍需补充。

## 4. 结果与代码

- 汇总报告：[summary.md](../../results/class_shared_construction_seed_comparison_20261003_paired/summary.md)
- 配对 Student 数据：[paired_student.csv](../../results/class_shared_construction_seed_comparison_20261003_paired/paired_student.csv)
- 新 seed 原始结果：[arxiv results](../../results/class_shared_large_seed314159_20261003/arxiv_base4/results.json)、[SportsFit results](../../results/class_shared_large_seed314159_20261003/sportsfit_base4/results.json)
- 复核脚本：[compare_class_shared_construction_seeds_20261003.py](../../tools/compare_class_shared_construction_seeds_20261003.py)
- seed 参数化 runner：[run_public_continual_baselines_large_20261001.py](../../code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py:341)

下一步应优先做一个独立类别顺序或 BookChild 的第二 construction seed，再把通过复核的结果与 k-center、medoid、CaT/GECC adapter 放回统一基线表；暂不继续扫描文本参数。
