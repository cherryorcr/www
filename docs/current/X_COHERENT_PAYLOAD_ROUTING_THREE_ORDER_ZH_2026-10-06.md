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
| cat_nearest | 76.54 | 7.50 | 79.54 | 76.13 | 77.15 | 69.59 | 6 |
| formal_prompt_margin | 77.31 | 7.44 | 80.12 | 77.13 | 77.61 | 70.12 | 6 |
| x_conditioned_routing | 76.78 | 7.67 | 79.68 | 76.45 | 77.30 | 69.09 | 6 |
| x_conditioned_routing_coherent | 77.18 | 7.51 | 80.06 | 76.94 | 77.58 | 69.97 | 6 |
| fixed_x_xsp | 75.93 | 7.95 | 79.14 | 75.61 | 76.36 | 68.45 | 6 |

## 4. 解释口径

CaT nearest pooled Final F1 为 76.54。
x_conditioned_routing 只有在多个独立顺序、两个数据集上都表现稳定时，才可作为文本模块的正向证据；如果仅改善少数 cell，应将其定位为有解释性的组件，而不是主方法整体提升。

路由审计显示六个最终阶段中每个类别均使用了 2--3 个不同的真实 payload，三个 slot 均有非零使用；因此本轮不是实现退化为 class-shared text，而是多 payload 路由确实生效但收益有限。

本报告只汇总同一 Prompt Student 下的本轮结果，不与旧 Student 或不同协议的历史结果直接混合。
