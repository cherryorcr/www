# X-conditioned Differentiable Multi-Payload Routing 统一实验报告

## 1. 方法与目标

本轮固定 CaT 合成特征 X，只研究一个类别内不同合成节点是否应使用不同的真实文本 payload。每个类别从当前 raw delta 构造一个有界真实候选池；每个合成行根据自身 X 通过可微 assignment logits 进行路由，最后用 argmax 材料化为真实文本。该模块不修改旧 memory、不读取历史 raw、不修改拓扑，也不根据最终测试 F1 选文本。

## 2. 固定协议

- 数据集：Cora、Citeseer；本轮使用 `order19611`；每个 cell 三个 Student seeds（150--152）。这是候选池改动的确认实验，不替代之前三顺序、六 seeds 的主结果。
- memory rate、source delta、report-dev/test 划分、CaT 训练预算和 Prompt Student 完全一致。
- Student 为无 learned classifier head 的 prompt zero-shot 图文模型；所有方法使用同一个 Student 与同一训练协议。
- 候选池默认最多 3 个 slot，routing steps=30，temperature=0.15，load weight=0.03，anchor weight=0.10。

## 3. 结果汇总

| 方法 | Final F1 | Order std | Accuracy | Old F1 | New F1 | Worst F1 | Cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| cat_nearest | 76.65 | 9.66 | 79.65 | 81.09 | 68.43 | 69.82 | 2 |
| formal_prompt_margin | 77.36 | 9.34 | 80.48 | 82.13 | 68.64 | 70.75 | 2 |
| x_conditioned_routing | 76.96 | 8.99 | 80.12 | 81.82 | 68.11 | 70.60 | 2 |
| x_conditioned_routing_x_coverage | 76.85 | 9.43 | 80.20 | 81.75 | 67.79 | 70.18 | 2 |
| gecc_hard_tag | 74.44 | 9.64 | 77.95 | 77.54 | 68.76 | 67.62 | 2 |
| random_real | 68.98 | 4.35 | 74.74 | 72.88 | 61.16 | 65.90 | 2 |
| k_center | 63.75 | 2.39 | 69.35 | 64.92 | 60.41 | 62.06 | 2 |
| kmeans_medoid | 68.82 | 9.14 | 73.26 | 74.20 | 58.84 | 62.36 | 2 |
| fixed_x_xsp | 76.10 | 10.17 | 79.45 | 80.57 | 67.72 | 68.91 | 2 |
| xsp_x_cattext | 77.14 | 9.01 | 80.09 | 81.19 | 69.63 | 70.77 | 2 |
| xsp_st | 76.69 | 9.92 | 80.07 | 81.12 | 68.40 | 69.68 | 2 |
| full_xsp | 77.10 | 10.01 | 80.24 | 81.59 | 68.79 | 70.02 | 2 |

## 4. 解释口径

CaT nearest pooled Final F1 为 76.65。
x_conditioned_routing 只有在多个独立顺序、两个数据集上都表现稳定时，才可作为文本模块的正向证据；如果仅改善少数 cell，应将其定位为有解释性的组件，而不是主方法整体提升。

路由审计显示两个最终 cell 中每个类别均使用了 2--3 个不同的真实 payload，三个 slot 均有非零使用；因此本轮不是实现退化为 class-shared text，而是多 payload 路由确实生效但收益有限。

## 5. 候选池改动的结论

`x_conditioned_routing_x_coverage` 只改变候选池的贪心选择：在文本/图特征多样性之外，增加候选对不同 CaT X 行的 graph-response profile 覆盖，不改变 routing loss、slot 数、温度、Student 或 memory budget。

相对 `cat_nearest`，它在 Cora 为 `+0.04 pp`、Citeseer 为 `+0.36 pp`；相对原 `x_conditioned_routing`，Cora 约 `+0.20 pp`，但 Citeseer 约 `-0.43 pp`。因此该改动没有跨数据集稳定收益，不能进入主方法。它说明候选池的简单响应分散会牺牲部分原有图文兼容性，当前瓶颈不是候选 profile 不够分散，而是候选 payload 是否真正覆盖有用的类别子语义。

当前决策：保留该版本作为负结果和候选池消融；不继续扫描其权重。主线仍使用原 `x_conditioned_routing` 作为多 payload 组件，`formal_prompt_margin` 作为更稳的文本 assignment 对照。

本报告只汇总同一 Prompt Student 下的本轮结果，不与旧 Student 或不同协议的历史结果直接混合。
