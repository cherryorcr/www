# X-conditioned Slot Partition 大图结果对比

日期：2026-10-03

## 1. 实验对象

本轮验证的是 `X-conditioned Slot Partition`，对应代码：

- [x_conditioned_payload_partition.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/x_conditioned_payload_partition.py)
- [run_public_continual_baselines_large_20261001.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py)

它在 CaT 生成的每个新合成特征槽位上单独分配文本 payload，不再把一个类别的同一条文本广播给全部新合成节点。分配使用类别内的 balanced entropic Sinkhorn，代价同时使用当前 X 和当前 delta 的文本表示；随后将硬分配落成真实文本并重新运行冻结 MiniLM。

## 2. 统一协议

- 数据集：Arxiv、BookChild、SportsFit。
- 类别协议：base+4，每个数据集使用固定的一条类别顺序。
- memory：累积 5% fixed-rate。
- Student：原 edge-aware 2-layer GCN、冻结 MiniLM、文本投影和双向 InfoNCE。
- Student seeds：100--105。
- CaT steps：3000；construction seed：270927。
- 构造只读取当前 delta；历史 raw 读取次数为 0。
- 构造不使用 report/test 标签；旧 synthetic memory 不修改；本轮不修改拓扑。
- 表中的 GECC、k-center、medoid 和 CaT-TAG 均是同一 Student 下的机制适配对照，不应写成对应论文的官方端到端复现。

## 3. 每个数据集的结果

数值均为百分比。`Build s` 是构造时间，`Stage F1` 是五个阶段 F1 的平均值。

| 数据集 | 方法 | Final F1 | Accuracy | Stage F1 | Old F1 | New F1 | Worst F1 | Build s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Arxiv | CaT-TAG CCTP top-5 | 24.27 | 48.39 | 24.44 | 27.54 | 5.76 | 24.27 | 152.2 |
| Arxiv | GECC-TAG hard | 20.23 | 31.43 | 30.51 | 22.44 | 7.66 | 20.23 | 187.0 |
| Arxiv | k-center | 19.43 | 30.69 | 29.37 | 21.47 | 7.90 | 19.36 | 28.0 |
| Arxiv | medoid | 18.06 | 24.71 | 29.68 | 19.91 | 7.53 | 18.06 | 40.2 |
| Arxiv | class-shared real payload | 24.79 | 42.56 | 27.54 | 27.48 | 9.56 | 24.79 | 148.5 |
| Arxiv | **X-conditioned Slot Partition** | **21.38** | **33.73** | **33.11** | **23.81** | **7.61** | **21.38** | **319.7** |
| BookChild | CaT-TAG CCTP top-5 | 4.28 | 31.97 | 4.25 | 4.84 | 0.35 | 3.40 | 77.2 |
| BookChild | GECC-TAG hard | 14.28 | 14.06 | 37.46 | 15.15 | 8.20 | 14.28 | 93.1 |
| BookChild | k-center | 12.49 | 11.93 | 31.19 | 13.24 | 7.25 | 12.49 | 6.7 |
| BookChild | medoid | 12.98 | 13.70 | 35.09 | 13.51 | 9.28 | 12.98 | 9.0 |
| BookChild | class-shared real payload | 5.46 | 21.07 | 7.82 | 5.80 | 3.10 | 2.99 | 77.5 |
| BookChild | **X-conditioned Slot Partition** | **14.89** | **14.55** | **38.43** | **15.75** | **8.82** | **14.87** | **106.3** |
| SportsFit | CaT-TAG CCTP top-5 | 23.99 | 43.02 | 28.88 | 24.30 | 22.27 | 23.99 | 399.7 |
| SportsFit | GECC-TAG hard | 26.44 | 26.95 | 46.43 | 27.72 | 19.44 | 26.44 | 679.4 |
| SportsFit | k-center | 29.42 | 32.01 | 53.95 | 31.42 | 18.46 | 29.42 | 24.7 |
| SportsFit | medoid | 27.00 | 27.78 | 54.33 | 28.27 | 20.01 | 27.00 | 42.8 |
| SportsFit | class-shared real payload | 46.83 | 63.44 | 59.75 | 49.78 | 30.56 | 44.41 | 408.3 |
| SportsFit | **X-conditioned Slot Partition** | **29.88** | **35.05** | **55.26** | **31.06** | **23.37** | **26.33** | **407.6** |

## 4. 三图等权汇总

| 方法 | Final F1 | Accuracy | Stage F1 | Old F1 | New F1 | Worst F1 | Build s |
|---|---:|---:|---:|---:|---:|---:|---:|
| class-shared real payload | 25.69 | 42.36 | 31.70 | 27.69 | 14.41 | 24.07 | 211.4 |
| **X-conditioned Slot Partition** | **22.05** | **27.78** | **42.26** | **23.54** | **13.27** | **20.86** | **277.9** |
| k-center | 20.45 | 24.87 | 38.17 | 22.04 | 11.21 | 20.43 | 19.8 |
| GECC-TAG hard | 20.31 | 24.14 | 38.13 | 21.77 | 11.76 | 20.31 | 319.8 |
| medoid | 19.35 | 22.06 | 39.70 | 20.57 | 12.27 | 19.35 | 30.7 |
| CaT-TAG CCTP top-5 | 17.51 | 41.13 | 19.19 | 18.89 | 9.46 | 17.22 | 209.7 |

相对主要对照的等权差值：

| 对照 | Delta Final F1 | Delta Accuracy | Delta Stage F1 | Delta Old F1 | Delta New F1 |
|---|---:|---:|---:|---:|---:|
| CaT-TAG CCTP top-5 | +4.54 | -13.35 | +23.07 | +4.65 | +3.81 |
| GECC-TAG hard | +1.74 | +3.64 | +4.13 | +1.77 | +1.51 |
| k-center | +1.60 | +2.91 | +4.09 | +1.50 | +2.06 |
| medoid | +2.70 | +5.72 | +2.56 | +2.97 | +1.00 |
| class-shared real payload | -3.64 | -14.58 | +10.56 | -4.15 | -1.14 |

## 5. 实现审计

最终阶段的审计显示：

- Arxiv：192 个新槽位，192 条唯一文本 payload，unique ratio 1.0。
- BookChild：61 个新槽位，61 条唯一文本 payload，unique ratio 1.0。
- SportsFit：253 个新槽位，252 条唯一文本 payload，unique ratio 0.9961。
- 三个数据集 historical raw reads 均为 0。
- `X` 确实更新；旧 memory、旧拓扑和总预算均未被修改。
- 分配为类别受限的 balanced entropic Sinkhorn；每个 X 槽位有独立的 materialized payload。

因此本轮没有复现之前“每个类别只剩一条文本，InfoNCE 同类重复文本塌缩”的实现错误。BookChild 从此前约 4--5 F1 的错误/类共享结果恢复到 14.89 F1，说明该修复有效。

## 6. 结果解释

这版方法的主要正向信号不是最终 F1 在所有数据集都最高，而是：

1. 它解决了 class-shared 文本广播造成的结构性问题，能够让不同 X 槽位使用不同真实 payload。
2. 它在三个数据集上都超过 k-center 和 medoid 的最终 F1；相对 GECC 也都更高。
3. 它的 Stage F1 在三图上均高于公开适配基线，说明中间阶段的增量适应能力较强。
4. BookChild 上同时改善了最终 F1、Old F1 和 New F1，表明 slot-level 文本分配确实能缓解长尾图上的文本塌缩。

但它还没有成为数值上的最终主方法：

- 三图等权 Final F1 仍低于 class-shared real payload 3.64 个百分点。
- Arxiv 和 SportsFit 的最终阶段退化明显，Final F1 低于对应的 class-shared 方法。
- Accuracy 明显低于 class-shared 和 CCTP；这说明更高的文本多样性没有直接转化为整体分类准确率。
- 构造成本约为 k-center 的 14 倍，且高于 class-shared 和 CCTP。
- 当前只有每个数据集一条 base+4 顺序和一个 construction seed，不能据此宣称稳定泛化。

最合理的解释是：class-shared 的强结果来自同类文本的一致性和低噪声，而 slot partition 虽然避免了文本塌缩，却可能引入过大的类内 payload 方差。当前 Sinkhorn 只保证质量平衡和 X/text 局部匹配，还没有显式保证同类不同槽位之间的语义一致性，因此出现了“Stage F1 变好、最终保持变差”的现象。

## 7. 下一步

本轮不应继续盲目扫描大量温度或 margin。建议按以下顺序：

1. 先用独立 construction seed 复现三图，确认 `X-slot` 的正向结果不是一次性构造轨迹。
2. 保留 slot-level payload，但增加类内语义一致性约束：以类别中心作为弱 anchor，同时保留槽位差异；不要退回单类广播。
3. 对最后阶段退化单独做诊断，报告每个阶段的 Old F1/New F1 和文本槽位方差。
4. 若独立 seed 后仍保持超过 k-center、medoid、GECC，则可以把它作为“有结构创新但仍需稳定化”的候选主方法；若要作为最终论文主方法，还需要解决 Arxiv/SportsFit 相对 class-shared 的最终保持差距。

原始新方法结果：

- [GPU0 summary](D:/Aprojects/continue_tag_condensation/results/x_slot_partition_large_20261003_gpu0/summary.md)
- [GPU1 summary](D:/Aprojects/continue_tag_condensation/results/x_slot_partition_large_20261003_gpu1/summary.md)
- [Arxiv results](D:/Aprojects/continue_tag_condensation/results/x_slot_partition_large_20261003_gpu0/arxiv_base4/results.json)
- [BookChild results](D:/Aprojects/continue_tag_condensation/results/x_slot_partition_large_20261003_gpu0/bookchild_base4/results.json)
- [SportsFit results](D:/Aprojects/continue_tag_condensation/results/x_slot_partition_large_20261003_gpu1/sportsfit_base4/results.json)
