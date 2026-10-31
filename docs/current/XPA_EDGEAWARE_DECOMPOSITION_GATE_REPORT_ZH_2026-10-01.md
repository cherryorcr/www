# X-P-A Edge-Aware 组件归因与拓扑门控报告

日期：2026-10-01

## 1. 本轮目的

本轮不再扫描 X-P-A 的大量超参数，只回答两个问题：

1. 在同一个 Student 图算子下，性能问题主要来自 `X/P` 还是拓扑 `A`；
2. 是否可以用 construction-time sealed validation 在 `cap=0` 和 `cap=1` 之间安全选择。

SR / Semantic-Recursive 不属于本轮比较对象。

## 2. 统一协议

- 数据集：arXiv、BookChild、SportsFit；
- 协议：base+4、5% fixed-rate memory；
- Student：原始两层 GCN、可训练 text projection、双向图文 InfoNCE；
- Student seeds：100--105，共 6 个；
- 图算子：`edge-aware`，直接使用 memory 中保存的 `edge_index`；
- X-P-A：`x_new_radius=0.10`、`p_mix=0.25`、24 次 Sinkhorn、每个新 row 最多一次 A revision；
- 历史 raw graph 访问次数：0；
- report/test 标签：不参与 construction 或 gate 选择。

两种候选的区别只有拓扑：

- `X+P`：优化 X/P，冻结 C0 跨块拓扑；
- `X+P+A`：在同一 X/P 结果上允许有界 A revision；
- `gated X+P+A`：同时生成 `cap=0` 和 `cap=1`，用 sealed validation support response、old response 和 new margin 选择。

## 3. X+P 与 X+P+A

最终阶段结果如下，单位为 Macro-F1；old/new 是最后阶段的 old-class/new-class F1。

| 数据集 | 方法 | Final F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 |
|---|---|---:|---:|---:|---:|---:|
| arXiv | X+P | 18.213 | 29.802 | 26.526 | 20.009 | 22.808 |
| arXiv | X+P+A | 17.681 | 29.506 | 25.575 | 19.387 | 22.067 |
| BookChild | X+P | 11.124 | 13.421 | 15.384 | 12.859 | 13.878 |
| BookChild | X+P+A | 11.255 | 12.260 | 15.793 | 12.947 | 13.922 |
| SportsFit | X+P | 25.320 | 31.051 | 32.468 | 29.283 | 29.762 |
| SportsFit | X+P+A | 28.649 | 40.129 | 36.861 | 32.805 | 30.769 |

相对 X+P，A 的变化为：

| 数据集 | Final F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 |
|---|---:|---:|---:|---:|---:|
| arXiv | -0.532 | -0.296 | -0.951 | -0.622 | -0.741 |
| BookChild | +0.131 | -1.161 | +0.409 | +0.088 | +0.044 |
| SportsFit | +3.329 | +9.078 | +4.393 | +3.522 | +1.007 |

结论：A 的收益不是普适的。它在 SportsFit 有明显正向作用，在 arXiv 上破坏传播，在 BookChild 上只带来很小的 F1 变化但 Accuracy 下降。当前最可靠的核心应是 X+P，A 只能作为条件启用组件。

## 4. 完整拓扑门控

门控在 12 个增量阶段中接受了 3 次 revision：

| 数据集 | 接受的 stage | 其它 stage |
|---|---|---|
| arXiv | stage 4 | stage 1--3 回退 cap=0 |
| BookChild | stage 3 | stage 1、2、4 回退 cap=0 |
| SportsFit | stage 1 | stage 2--4 回退 cap=0 |

门控规则从未读取最终 report/test F1。它要求旧类 validation response 不恶化，并且 support response 或 new margin 达到固定改善条件。

门控最终结果：

| 数据集 | Final F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 |
|---|---:|---:|---:|---:|---:|
| arXiv | 17.216 | 28.869 | 25.215 | 18.844 | 23.050 |
| BookChild | 11.726 | 13.342 | 16.403 | 13.409 | 14.171 |
| SportsFit | 29.303 | 39.809 | 37.686 | 33.383 | 30.195 |

这些绝对值来自独立 gate 运行，不能与另一张 GPU 上的 X+P/X+P+A 逐点相减作为严格 paired effect。它们支持的结论是：gate 确实能把 revision 限制在少数阶段，但还没有证明能够稳定提高最终 F1。

## 5. 确定性检查

同一个 BookChild stage-0 C0、相同 seeds 和 edge-aware Student，在 deterministic PyTorch/CUBLAS 配置下得到：

| 运行 | Final F1 |
|---|---:|
| GPU0 | 17.470 |
| GPU1 | 19.062 |
| GPU0 repeat | 18.962 |

因此当前 sparse GCN 路径仍存在明显 CUDA 数值波动。`torch.use_deterministic_algorithms(True)` 没有消除该差异。小于约 1 pp 的改进不能在当前协议下作为稳定算法收益；后续正式结果必须报告重复方差或使用同一候选的 paired evaluation。

## 6. 严格 edge-aware append-fixed 参考

随后补跑了 append-fixed 的严格 edge-aware 参考。它直接读取每个阶段保存的 `append_fixed_s{stage}_memory.pt` 和其中的 `edge_index`，使用与 X-P/X-P-A 相同的 Student、seeds 和 report 规则，不再按文本重新构图。

| 数据集 | 方法 | Final F1 | Accuracy | Old F1 | New F1 | 阶段平均 F1 |
|---|---|---:|---:|---:|---:|---:|
| arXiv | edge-aware append-fixed | 17.058 | 25.338 | 24.597 | 18.817 | 22.427 |
| arXiv | X+P | 18.213 | 29.802 | 26.526 | 20.009 | 22.808 |
| arXiv | X+P+A | 17.681 | 29.506 | 25.575 | 19.387 | 22.067 |
| BookChild | edge-aware append-fixed | 10.820 | 10.847 | 15.037 | 12.457 | 14.585 |
| BookChild | X+P | 11.124 | 13.421 | 15.384 | 12.859 | 13.878 |
| BookChild | X+P+A | 11.255 | 12.260 | 15.793 | 12.947 | 13.922 |
| SportsFit | edge-aware append-fixed | 20.835 | 25.140 | 26.339 | 24.758 | 26.915 |
| SportsFit | X+P | 25.320 | 31.051 | 32.468 | 29.283 | 29.762 |
| SportsFit | X+P+A | 28.649 | 40.129 | 36.861 | 32.805 | 30.769 |

最终 F1 相对 edge-aware append-fixed 的变化为：

| 数据集 | X+P | X+P+A |
|---|---:|---:|
| arXiv | +1.155 | +0.624 |
| BookChild | +0.304 | +0.435 |
| SportsFit | +4.485 | +7.814 |

这说明 X-P-A 并不是在所有协议下都低于 C0。此前的 legacy 大图表中，Student 会按文本重新构图并忽略保存的 `edge_index`，所以不能用于判断 A 的拓扑收益。严格 edge-aware 对照给出了更积极的信号，但仍需同 GPU paired repeat 处理数值波动后才能正式晋升。

## 7. 当前方法决定

本轮之后暂定主线收敛为：

> Bounded Student-Compatible X-P Revision with Optional Validation-Gated Topology

具体含义：

1. X/P 的 Student outer optimization 保留；
2. A 不再无条件写入最终 memory；
3. cap=0 是安全 fallback；
4. cap=1 只有在 sealed validation 满足旧类保护和新类响应条件时才启用；
5. 不加入新的 selector、attention、dynamic budget 或 GraphGPS Student。

这不是“已经稳定超过 C0”的结论。当前证据支持：

- 相比 k-center，X-P-A 在三个大图的最终 Macro-F1 均有正差值；
- 相比 append-only C0，提升只在部分数据集和阶段出现；
- A 的作用具有数据集和阶段依赖；
- X/P/A 梯度确实存在，但梯度存在不等于最终下游收益稳定。

## 8. 下一步

本轮确认性实验已经完成：

1. BookChild 已补做同进程 paired repeat；
2. arXiv、SportsFit 已按相同脚本、相同 edge-aware Student 和同一组 seeds 补齐统一 JSON；
3. 报告了逐 Student paired difference、标准差和近似 95% CI；
4. 没有根据本轮最终 F1 反向修改 X/P/A 参数。

基于本轮结果，后续不再把“X+P 也不能超过 append-only”作为默认预期，而是：

5. 将 X+P 作为当前核心方法候选，继续用独立顺序和公开基线验证；
6. 将 A 保留为可选拓扑扩展，不把无条件 X+P+A 写成所有数据集都适用的主算法；
7. 暂不继续堆叠 selector、attention、dynamic budget 或 GraphGPS Student。

## 10. 同进程 paired repeat

### 10.1 目的与协议

为控制 sparse CUDA GCN 的运行间数值波动，三种 memory 在同一个 Python 进程中按固定顺序评估：

- `append-edge`：从 `p5_large_base4_20260919` 读取 append-only memory 及其保存的 `edge_index`；
- `X+P`：读取同一轮 edge-aware X/P 构造结果；
- `X+P+A`：读取同一轮 edge-aware X/P+A 构造结果；
- Student seeds：`100--105`；
- stage：最终 `base+4` stage 4；
- 图算子：直接使用各 memory 中保存的 `edge_index`，不按文本重新构图；
- 每个数据集的三种 memory 由同一进程、同一设备和同一批 Student seeds 依次评估。

这里的 paired difference 是“同一 Student seed 下候选 memory 的结果之差”，不是跨数据集的统计检验。95% CI 使用 6 个 paired Student differences 的正态近似，仅用于显示方差量级，不能替代更多 seeds 或多顺序实验。

### 10.2 最终阶段结果

| 数据集 | 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|
| arXiv | append-edge | 17.058 | 25.338 | 24.597 | 18.817 |
| arXiv | X+P | 18.213 | 29.802 | 26.526 | 20.009 |
| arXiv | X+P+A | 17.681 | 29.506 | 25.575 | 19.387 |
| BookChild | append-edge | 10.868 | 11.810 | 14.933 | 12.572 |
| BookChild | X+P | 11.424 | 12.577 | 16.025 | 13.121 |
| BookChild | X+P+A | 11.470 | 12.203 | 16.032 | 13.187 |
| SportsFit | append-edge | 22.750 | 30.373 | 29.050 | 26.325 |
| SportsFit | X+P | 27.012 | 33.717 | 33.799 | 31.011 |
| SportsFit | X+P+A | 27.651 | 36.402 | 34.294 | 31.781 |

### 10.3 Paired 差值

相对严格 `append-edge` reference：

| 数据集 | X+P: Final F1 | X+P: Accuracy | X+P+A: Final F1 | X+P+A: Accuracy |
|---|---:|---:|---:|---:|
| arXiv | +1.155 | +4.464 | +0.623 | +4.169 |
| BookChild | +0.556 | +0.766 | +0.603 | +0.393 |
| SportsFit | +4.263 | +3.343 | +4.902 | +6.028 |

6 个 paired Student differences 的 Final F1 统计为：

| 数据集 | 比较 | mean difference | std | 近似 95% CI |
|---|---|---:|---:|---:|
| arXiv | X+P - append | +1.155 | 1.705 | [-0.209, +2.519] |
| arXiv | X+P+A - append | +0.623 | 0.984 | [-0.165, +1.411] |
| arXiv | X+P+A - X+P | -0.532 | 1.540 | [-1.765, +0.700] |
| BookChild | X+P - append | +0.556 | 1.277 | [-0.466, +1.578] |
| BookChild | X+P+A - append | +0.603 | 0.616 | [+0.110, +1.096] |
| BookChild | X+P+A - X+P | +0.047 | 0.742 | [-0.547, +0.641] |
| SportsFit | X+P - append | +4.263 | 6.515 | [-0.950, +9.476] |
| SportsFit | X+P+A - append | +4.902 | 4.727 | [+1.120, +8.684] |
| SportsFit | X+P+A - X+P | +0.639 | 6.615 | [-4.654, +5.933] |

### 10.4 解释

这次 paired repeat 支持三个较窄的结论：

1. **X+P 的方向是正的。** 在三张大图的最终阶段，X+P 相对严格 append-edge 均为正差值；这比把 X+P 与不同进程、不同 GPU 的绝对结果直接比较更可信。
2. **A 不是普适增益。** X+P+A 相对 X+P 在 arXiv 为 `-0.532`，BookChild 为 `+0.047`，SportsFit 为 `+0.639`。因此 A 可以解释为数据集依赖的拓扑扩展，不能无条件并入主方法。
3. **差异仍受 Student 数值方差影响。** arXiv 的 X+P 和 SportsFit 的 X+P paired CI 包含 0；这表示当前 6 个 seeds 足以发现方向性信号，但不足以宣称每张图上的显著性或普遍 SOTA。后续主表应增加 construction seed、类别顺序和更大 Student seed 集合，并报告均值、标准差和正向 cell 比例。

因此，当前方法的工作性冻结版本为：

> **Bounded Student-Compatible X+P**：保留 X/P 的 Student outer optimization，使用保存的 edge-aware C0 topology；A 作为受限、可回退的拓扑扩展单独报告。

这一步不是把 X+P 宣称为已经完成的最终算法，而是确认它具备进入公开基线、多顺序和成本账阶段的证据基础。

## 9. 证据目录

- `tmp_remote_results/xpa_edgeaware_xp_ablation_20261001/`
- `tmp_remote_results/xpa_edgeaware_xpa_ablation_20261001/`
- `tmp_remote_results/xpa_edgeaware_gate_full_20261001/`
- `tmp_remote_results/edgeaware_append_fixed_reference_20261001/`
- 远程完整结果：`/home/pc/research/TAG_condensation/results/xpa_edgeaware_*_20261001/`
- 远程 edge-aware append reference：`/home/pc/research/TAG_condensation/results/edgeaware_append_fixed_reference_20261001/`
- paired JSON：`tmp_remote_results/paired_xpa_20261001/`
- 远程 paired JSON：`/home/pc/research/TAG_condensation/results/paired_xpa_20261001/`
