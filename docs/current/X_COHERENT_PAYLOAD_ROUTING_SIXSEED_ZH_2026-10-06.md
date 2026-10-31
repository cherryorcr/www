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
| cat_nearest | 76.69 | 7.75 | 79.56 | 76.39 | 77.15 | 68.82 | 6 |
| formal_prompt_margin | 77.00 | 7.47 | 79.71 | 76.74 | 77.37 | 69.50 | 6 |
| x_conditioned_routing_coherent | 76.97 | 7.95 | 79.81 | 76.85 | 77.15 | 69.52 | 6 |

## 4. 六 seeds 的最终判断

在三顺序、六 Student seeds 的完整确认中，`x_conditioned_routing_coherent` 的 pooled Final F1 为 `76.97`，CaT 为 `76.69`，formal prompt-margin 为 `77.00`。相对 CaT 为 `+0.29 pp`；Accuracy 为 `79.81`，Old F1 为 `76.85`，New F1 为 `77.15`。它没有稳定超过 formal prompt-margin，6 个 paired cell 中 3 个为正向。

路由审计显示六个最终阶段中每个类别均使用了 2--3 个不同的真实 payload，三个 slot 均有非零使用；因此本轮不是实现退化为 class-shared text，而是多 payload 路由确实生效。coherent raw-medoid 的改动是将候选生成从多来源 chunk 拼接改为当前 delta 中完整文本的兼容性 medoid，再交给相同的 differentiable routing。

该版本的主要收益体现在旧类保持和 order 稳定性：Cora 的 order std 为 `0.14`，Citeseer 为 `0.17`，均低于 CaT；但新类 F1 的平均提升很小，且 cell-level 正向比例只有 `50%`。因此它是当前最好的多 payload 文本组件候选，不应表述为稳定优于所有基线的完整主方法。

## 5. 决策

保留三层结果：`formal_prompt_margin` 作为最稳的文本 assignment baseline，`x_conditioned_routing` 作为原始多 payload 对照，`x_conditioned_routing_coherent` 作为当前改进候选。后续不再继续添加候选池 selector 或响应覆盖权重；若进入论文主表，应报告 coherent 版本的平均收益、order std、Old/New F1 和成本，并明确它主要改善 memory 的旧类保持与文本连贯性。

本报告只汇总同一 Prompt Student 下的本轮结果，不与旧 Student 或不同协议的历史结果直接混合。
