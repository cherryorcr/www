# Unified-Q 拓扑分支实验报告

## 1. 实验目标

本轮实现并验证了 Unified-Q 的 topology-enabled 分支，比较：

- `unified_q_edge_free`：当前主方法，Q 用于 X 和文本，但 memory 不保存 synthetic edges；
- `unified_q`：用同一个 Q 额外诱导新节点之间的拓扑；
- `unified_q_boundary`：在 `unified_q` 基础上加入当前增量到历史 synthetic prefix 的 boundary sketch。

拓扑使用：

\[
A_{new,new}=Q^\top A_{\Delta}Q
\]

并通过受控的 edge budget 做硬材料化。boundary 分支使用当前增量与历史 synthetic support-ID 的合法边界摘要，不读取历史 raw 特征或历史 raw 文本。

## 2. 节点分类

Cora/Citeseer 使用同一类别顺序、相同 5% cumulative budget、相同 prompt-only GraphText Student 和相同 Student seeds。

| Dataset | Arm | Final Macro-F1 | Accuracy | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|
| Cora | edge-free | 80.81 +/- 0.35 | 83.39 | 80.92 | 80.65 |
| Cora | topology | 80.63 +/- 1.96 | 83.15 | 81.91 | 78.91 |
| Cora | boundary | 81.61 +/- 0.40 | 84.01 | 82.77 | 80.07 |
| Citeseer | edge-free | 68.68 +/- 0.57 | 73.25 | 75.10 | 55.82 |
| Citeseer | topology | 69.64 +/- 0.19 | 74.19 | 75.56 | 57.82 |
| Citeseer | boundary | 66.73 +/- 0.53 | 71.37 | 74.58 | 51.04 |

结论：拓扑没有稳定改善节点分类。它在 Citeseer 有小幅正向信号，在 Cora 基本持平；boundary 分支跨数据集不稳定。因此 topology 分支不替换当前 edge-free 主方法。

## 3. 链路预测结构探针

链路预测使用 3 个独立 edge split：`29001/29002/29003`。构造阶段只使用 `train_pos`，验证和测试边没有传入 constructor。

当前结果是一个固定的 slot-level link probe：使用 X/text fused cosine，并在 topology 分支中加入保存的 synthetic-edge indicator。它是结构保持与可用性的下游探针，不冒充完整 GAE 官方复现。

| Dataset | Arm | AUC | AP | Held-out edge recall | Degree Pearson | Synthetic edges |
|---|---|---:|---:|---:|---:|---:|
| Cora | edge-free | 0.8325 +/- 0.0117 | 0.8242 +/- 0.0090 | 0.000 | 0.000 | 0 |
| Cora | topology | 0.8591 +/- 0.0089 | 0.8546 +/- 0.0054 | 0.207 | 0.718 | 78 |
| Cora | boundary | 0.8595 +/- 0.0159 | 0.8575 +/- 0.0133 | 0.220 | 0.734 | 86 |
| Citeseer | edge-free | 0.8479 +/- 0.0083 | 0.8344 +/- 0.0196 | 0.000 | 0.000 | 0 |
| Citeseer | topology | 0.8388 +/- 0.0178 | 0.8267 +/- 0.0234 | 0.153 | 0.794 | 60 |
| Citeseer | boundary | 0.8497 +/- 0.0136 | 0.8437 +/- 0.0176 | 0.170 | 0.780 | 69 |

结论：拓扑分支确实让 memory 具有结构任务能力，但收益具有数据集依赖性。Cora 上提升明显，Citeseer 上接近持平且方差更大。boundary 不能视为普适增强。

## 4. 大图 BookHis

BookHis 使用五阶段 base+4、5% cumulative memory、同一 prompt Student。为控制远程磁盘占用，本轮只保存 JSON 指标，不保存大图 checkpoint。

| Arm | Final Macro-F1 | Accuracy | Old F1 | New F1 | Synthetic edges |
|---|---:|---:|---:|---:|---:|
| edge-free | 39.58 | 76.46 | 38.38 | 52.76 | 0 |
| topology | 40.20 | 78.18 | 38.81 | 55.56 | 5,165 |

大图上拓扑分支有正向信号，但增益不大，暂时只能说明可扩展和结构信息没有完全失效，不能据此声称拓扑分支已经普遍优于主方法。

## 5. 最终定位

当前建议保持：

\[
\boxed{\text{Unified-Q-EF 主方法} + \text{Unified-Q-T 结构扩展}}
\]

主论文主线继续使用 edge-free Unified-Q，贡献集中在历史受限条件下用共享可微 Q 同时完成特征、文本和分配优化。拓扑版作为第二个实验分支，展示同一 Q 可以进一步诱导可存储的 synthetic topology，并支持链路预测/结构保持探针。

boundary 版保留为扩展消融，不作为主方法核心。

## 6. 后续必要实验

1. 在 topology branch 上补一个真正训练的 GAE/GNN link predictor，和当前固定 probe 分开报告；
2. 在至少两个大图上补 topology 的独立 construction seeds；
3. 报告边存储成本、构造显存和链路预测收益的 trade-off；
4. 如果 topology 在更多数据集上仍不稳定，则论文明确限定主任务为图文节点分类，并把 topology 定义为结构扩展而非主贡献。

## 7. 代码与结果

- 拓扑节点分类 runner：`code/v2_llm_fusion/run_unified_q_topology_compare_20261011.py`
- 链路预测结构探针：`code/v2_llm_fusion/evaluate_unified_q_topology_downstream_20261011.py`
- 大图对照 runner：`code/v2_llm_fusion/run_unified_q_large_topology_compare_20261011.py`
- 汇总脚本：`code/v2_llm_fusion/summarize_unified_q_topology_20261011.py`
- 汇总表和可视化：`results/unified_q_topology_report_20261011/`
