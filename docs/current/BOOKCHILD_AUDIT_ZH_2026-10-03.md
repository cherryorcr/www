# BookChild 审计报告（2026-10-03）

## 结论

BookChild 当前出现的 2.8%--6.1% Macro-F1 不能直接解释为“这个数据集接近随机”或“方法完全失效”。审计发现，`cat_mmd_tag_cctp_top5` 的实现存在一个具体的策略映射错误：CCTP top-k 被映射成了 `text_center`，导致每个类别的所有合成行共享同一条文本，而不是为每个合成行选择其 graph-nearest 的 top-k 文本来源。

这个错误会破坏当前 Student 的双向 InfoNCE。一个 batch 内同类别合成行拥有完全相同或几乎相同的文本 embedding，但 InfoNCE 仍把非对角项当作负样本，训练会发生类别塌缩。修正后，BookChild 暂时不应删除；应先重跑修正后的 CCTP/nearest 版本，再决定它是否作为主表数据集或压力测试数据集。

## 数据与协议核对

远端原始数据：`/home/pc/research/TAGSAM/data/tagbench/bookchild.pt`

- 节点数：76,875
- 类别数：24
- report-dev：7,688
- pool：53,811
- test：7,688
- 划分规则：`protocol_v3.split_masks_v4(original_ids, report_dev_mod=3)`
- 持续协议：base+4
- memory budget：`[2344, 2458, 2551, 2629, 2690]`
- Student：原始 edge-aware 2-layer GCN + text projection + bidirectional InfoNCE
- Student seeds：`100..105`
- Student steps：最终各阶段均为 2,048

报告集类别分布高度不均衡。五个到达阶段在 report-dev 中分别包含：

| 阶段 | 类别数 | 节点数 | 占比 |
|---|---:|---:|---:|
| base | 10 | 6,748 | 87.77% |
| increment 1 | 4 | 310 | 4.03% |
| increment 2 | 4 | 280 | 3.64% |
| increment 3 | 3 | 212 | 2.76% |
| increment 4 | 3 | 138 | 1.80% |

因此 BookChild 是一个严重长尾、增量阶段极小的压力协议。它可以用于检验长尾与文本歧义下的稳定性，但不适合只看一个 canonical order 就下普遍结论。

## 随机与多数类参照

最终报告集的 24 类真实先验下：

| 参照 | Macro-F1（期望） | Accuracy（期望） | 含义 |
|---|---:|---:|---|
| 多数类预测 | 2.00% | 31.48% | 只预测最大类 |
| 均匀随机 | 2.82% | 4.17% | 24 类均匀采样 |
| 按真实先验随机 | 4.17% | 15.12% | 按 report-dev 类别比例采样 |

当前错误实现的最终结果：

- CCTP top-5：4.16% F1，31.80% Accuracy；预测分布中约 7,200--7,400 个节点被预测成同一个最大类，属于多数类塌缩，Macro-F1 接近按先验随机。
- class-assigned：2.80% F1，4.85% Accuracy，接近均匀随机。
- class-shared：6.06% F1，19.02% Accuracy，略高于简单随机，但仍明显低于合理 real-node baseline。

所以“接近随机”的判断对当前错误 CCTP/class-assigned 结果成立，但原因首先是表示/文本材料化塌缩，而不是已证明 BookChild 不可用。

## 实现审计结果

### 已确认没有问题的部分

1. `run_unified_v3_continual.py::load_unified` 将 `label_list` 按 `all_labels` 映射为连续整数，BookChild 的 24 个标签完整覆盖 `0..23`。
2. prompts 使用同一类别顺序生成：`a document about {labels_desc[c]}`。
3. `stage_ids` 只从 `pool_mask` 取构造源；report/test 标签没有进入构造。
4. report-dev、pool、test 没有重叠。
5. pool 文本 cache 形状为 `(76875, 384)`，held-out 行为零；与当前 MiniLM 对原文重编码的样本余弦相似度约为 1.0。
6. memory 中保存的文本 embedding 与保存字符串重新编码后的余弦相似度约为 1.0。
7. `eval_stagewise_exact` 只在 report-dev 上评估，使用正确的 24 类标签集合。

### 已确认的具体问题

文件：`code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py`

原先的映射是：

```python
"cat_mmd_tag_cctp_top5": "text_center"
```

`external_continual_mechanisms.py::_select_text_sources` 中，`text_center` 只按当前类别文本中心排序，完全不使用当前合成行的 `x_row`。因此同一类别的每一行都会选出相同的 top-5 文本。

在 BookChild 最终 memory 上的实际统计：

| 版本 | memory 行数 | 唯一文本数 | Student 最终 F1 | Accuracy |
|---|---:|---:|---:|---:|
| old/nearest top-5 | 2,690 | 2,594 | 约 13.30% | 约 13.73% |
| 错误的 text-center top-5 | 2,690 | 24 | 约 4.04% | 约 31.82% |

两者使用相同 BookChild、相同 base+4、相同 memory budget、相同 construction seed `270927` 和相同 Student seeds；CaT X 的更新范数几乎相同。因此差异来自文本材料化，而不是 X 凝聚本身。

已将 CCTP top-3/top-5 映射修正为：

```python
"cat_mmd_tag_cctp_top3": "nearest"
"cat_mmd_tag_cctp_top5": "nearest"
```

这与 CCTP top-k 的含义一致：每个合成行根据 graph compatibility 选择自己的 top-k 文本来源。

## 修正版三图确认

修正版已在远端重新上传并通过 `py_compile`。在不改变 base+4、memory budget、Student、construction seed 和 Student seeds `100..105` 的条件下，重新运行了三张大图的 `cat_mmd_tag_cctp_top5`：

| 数据集 | Final Macro-F1 | Std | Accuracy | Old F1 | New F1 | 最终 memory 唯一文本 |
|---|---:|---:|---:|---:|---:|---:|
| arXiv | 19.15% | 1.55 | 30.27% | 21.37% | 6.57% | 5,771 / 5,926 |
| BookChild | 13.33% | 1.51 | 12.96% | 14.23% | 7.01% | 2,599 / 2,690 |
| SportsFit | 27.57% | 2.76 | 29.03% | 27.98% | 25.29% | 5,253 / 6,056 |

修正版 BookChild 的单 seed 复现为 Final Macro-F1 `14.75%`；六 seed 汇总为 `13.33% ± 1.51%`，与原先约 `13.30%` 的 nearest top-5 结果一致。三张图均不再出现 4% 左右的多数类塌缩，说明原异常主要来自实现映射错误，而不是 BookChild 数据集或 CCTP top-k 机制本身。

远端修正版结果目录：

- `/home/pc/research/TAG_condensation/results/public_baselines_cctp_top5_fixed_full_20261003_gpu0`：arXiv、BookChild；
- `/home/pc/research/TAG_condensation/results/public_baselines_cctp_top5_fixed_full_20261003_gpu1`：SportsFit。

最终 memory 的文本多样性也已核对：每个节点仍有自己的 graph-nearest 文本 payload，而不是每类共享一条 `text_center` 文本。

## 与其它方法的对照

已有同协议 BookChild 结果显示，合理方法并非都接近随机：

| 方法 | Final Macro-F1 |
|---|---:|
| random real | 12.37 |
| k-center | 12.49 |
| medoid | 12.98 |
| append-edge | 12.60 |
| X+P | 13.30 |
| GECC-hard-TAG 适配 | 14.28 |
| 错误 CCTP top-5 | 4.28 |
| 错误 class-shared | 5.46 |

这进一步说明，BookChild 的低分不是所有 Student/评估实现共同造成的；当前失败主要集中在这组 CaT 文本策略及其重复 payload。

## 当前决定

1. **不删除 BookChild 数据集。** 先删除的是错误实验结论，不是数据集。
2. **修正后的 nearest top-k 已完成复现。** BookChild 回到约 13% F1 的量级，三图六 seed 结果可作为修正版 CCTP 基线；原先的 4% 结果应标记为无效实现结果。
3. 若修正后 CCTP 与 real/coreset/GECC 基线处于同一量级，BookChild 可以保留为长尾压力测试；若主方法仍明显落后，则将 BookChild 从主结果表降为诊断/附录数据集，并明确报告失败原因。
4. 远端根分区已清理至约 96% 使用率，可继续保存新的 memory/results；本次修正版三图结果已经完整落盘并完成审计。

## 已完成的最小实验

已完成以下修复确认：

1. 固定 split、groups、budgets、Student seeds 和 construction seed；
2. 跑完修正后的 `cat_mmd_tag_cctp_top5`；
3. 核对唯一文本数、Final Macro-F1、Accuracy、Old/New F1；
4. 确认三图结果回到合理基线量级；
5. 不再把 `text_center` 版本当作 CCTP 结果。

