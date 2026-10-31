# X-conditioned Differentiable Multi-Payload Routing 统一实验报告

## 1. 方法与目标

本轮固定 CaT 合成特征 X，只研究一个类别内不同合成节点是否应使用不同的真实文本 payload。每个类别从当前 raw delta 构造一个有界真实候选池；每个合成行根据自身 X 通过可微 assignment logits 进行路由，最后用 argmax 材料化为真实文本。该模块不修改旧 memory、不读取历史 raw、不修改拓扑，也不根据最终测试 F1 选文本。

## 2. 固定协议

- 数据集：Cora、Citeseer；三个独立类别顺序；每个 cell 六个 Student seeds。
- memory rate、source delta、report-dev/test 划分、CaT 训练预算和 Prompt Student 完全一致。
- Student 为无 learned classifier head 的 prompt zero-shot 图文模型；所有方法使用同一个 Student 与同一训练协议。
- 候选池默认最多 3 个 slot，routing steps=30，temperature=0.15，load weight=0.03，anchor weight=0.10。

## 3. 结果汇总

| 方法 | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat_nearest | 76.61 | 7.66 | 79.47 | 76.30 | 77.09 | 68.82 | 6 |
| formal_prompt_margin | 77.09 | 7.58 | 79.78 | 76.92 | 77.35 | 69.50 | 6 |
| x_conditioned_routing | 76.75 | 7.77 | 79.59 | 76.50 | 77.12 | 69.16 | 6 |
| gecc_hard_tag | 74.54 | 8.30 | 78.09 | 73.00 | 77.01 | 65.90 | 6 |
| random_real | 69.24 | 7.16 | 74.76 | 67.81 | 71.32 | 60.26 | 6 |
| k_center | 60.09 | 7.44 | 65.36 | 56.92 | 64.30 | 49.39 | 6 |
| kmeans_medoid | 71.45 | 8.87 | 75.15 | 71.41 | 71.29 | 62.91 | 6 |
| fixed_x_xsp | 76.16 | 7.89 | 79.14 | 75.78 | 76.73 | 68.52 | 6 |
| xsp_x_cattext | 76.85 | 7.55 | 79.71 | 76.56 | 77.28 | 69.00 | 6 |
| xsp_st | 76.53 | 8.06 | 79.49 | 76.23 | 77.01 | 68.61 | 6 |
| full_xsp | 76.59 | 7.58 | 79.48 | 76.17 | 77.21 | 69.06 | 6 |

## 4. 解释口径

CaT nearest pooled Final F1 为 76.61。
x_conditioned_routing 只有在多个独立顺序、两个数据集上都表现稳定时，才可作为文本模块的正向证据；如果仅改善少数 cell，应将其定位为有解释性的组件，而不是主方法整体提升。

路由审计显示六个最终阶段中每个类别均使用了 2--3 个不同的真实 payload，三个 slot 均有非零使用；因此本轮不是实现退化为 class-shared text，而是多 payload 路由确实生效但收益有限。

相对 `cat_nearest`，`x_conditioned_routing` 的 pooled Final F1 为 76.75（+0.15 pp），Accuracy 为 79.59（+0.12 pp），Old F1 为 76.50（+0.20 pp），New F1 为 77.12（+0.03 pp）；六个 paired cell 中 4 个为正向。它超过了 GECC hard-tag、random real、k-center 和 k-means medoid，但没有超过 `formal_prompt_margin`（77.09，+0.48 pp，6/6 cell 正向），也没有形成稳定超过 CaT 的大幅优势。

因此本轮的结论是：X-conditioned multi-payload routing 已经证明“同类不同合成节点使用不同文本”可以在不修改 CaT X 的情况下被实现并审计，但当前候选池仍主要来自有限的 MMR/CCTP/anchor 组合，路由目标也主要是图文兼容和负载均衡，尚未直接优化类别内的语义覆盖或下游决策边界。这是收益有限的主要原因，而不是多 payload 没有真正生效。

## 5. 当前决策与后续

本版本暂不升级为最终主方法，也不立即扫描 temperature、slot 数和 routing steps。下一步应先固定当前结果，优先检查候选池生成：是否能根据 CaT 的 X 在同一类别内形成互补的语义子簇，而不是只从已有 CCTP 文本中挑选。之后只做一个受控对照：相同 routing 目标下比较当前候选池、X-conditioned class partition 候选池和 class-shared payload。若改进后的候选池能在两个数据集和独立顺序上保持正向，再考虑把它作为 CaT 的文本组件；否则保留 `formal_prompt_margin` 作为当前更稳的文本 assignment 版本。

本报告只汇总同一 Prompt Student 下的本轮结果，不与旧 Student 或不同协议的历史结果直接混合。
