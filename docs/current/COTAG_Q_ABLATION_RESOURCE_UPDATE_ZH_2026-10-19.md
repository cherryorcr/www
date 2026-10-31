# CoTAG-Q 消融与资源更新

## 本轮决定

- 数据集协议表只保留最终 ArXiv base+4 行；SportsFit 和 BookHis 的五阶段类别安排在正文说明，不重复占用表格行。
- 消融图合并为一个 Macro-F1 区间图。Fixed-Q、No-DM、No-GM 不是重复实验，而是三个不同控制；Accuracy 只保留在消融表中。
- 额外 mask 的重复实验补齐 Accuracy 的均值和 cell SD，但没有为尚未完成的官方原生复现填入估计值。
- CoTAG-Q 主方法仍保留 GM。No-GM 是资源/机制控制，不是新的主方法。

## 核心消融

| Full minus control, Macro-F1 points | ArXiv | SportsFit | BookHis |
|---|---:|---:|---:|
| Fixed-Q | +12.76 [11.78, 13.73] | +21.67 [20.62, 22.71] | +8.91 [7.19, 10.64] |
| No-DM | +1.91 [0.82, 3.00] | +14.74 [13.26, 16.21] | +1.30 [0.57, 2.04] |
| No-GM | -0.24 [-1.17, 0.69] | -0.45 [-2.21, 1.31] | +0.06 [-1.38, 1.50] |

区间单位为 F1 points，括号为配对 95% interval。Fixed-Q 和 No-DM 在九个
construction cells 中均支持完整方法；No-GM 的区间均跨过 0，不能声称它带来
稳定精度提升。

## No-GM 的方法含义

No-GM 仅设置 lambda_gradient=0。Q 仍通过 distribution matching、
propagation matching、Q anchor 和 residual anchor 更新；硬文本仍由 Q
材料化并保存。

但当前主配置的 joint text objective 为 0，因此去掉 GM 后，straight-through
文本分支仍会被计算和保存，却不再通过 proxy gradient 向 Q 提供文本特异的
二阶反馈。GM 仍是完整 graph-text 构造接口的一部分。No-GM 偶尔略高更可能是
二阶代理梯度噪声或特定 construction seed 的正则化效应，不应解释成 GM 有害。

## 资源审计

| Dataset | Dense edge-free | Sparse edge-free | Peak reduction |
|---|---:|---:|---:|
| ArXiv | 11.56 GiB / 82.0 s | 11.50 GiB / 82.2 s | 0.06 GiB |
| SportsFit | 23.44 GiB / 123.1 s | 23.33 GiB / 123.1 s | 0.11 GiB |
| BookHis | 1.17 GiB / 20.0 s | 1.17 GiB / 19.6 s | 0.001 GiB |

稀疏 edge-free 算子通过 forward、first-gradient 和 second-gradient 等价性检查，
但端到端节省很小。因此显存瓶颈不是 self-loop identity，而是当前增量特征/文本、
dense Q 与 Sinkhorn 状态、proxy activation 以及 GM 的二阶中间量。历史原始节点
没有跨阶段驻留；后一阶段只加载当前增量和冻结的 synthetic prefix。分阶段访问
减少了历史数据累积，但不能消除“大当前增量 + 二阶构造图”的峰值。

## 论文落点

- 主文只将 learned Q 和 DM 写成有稳定归因证据的机制。
- Shared-Q 的优势写成紧凑参数化、共同 provenance 和较低 assignment cost，
  不写成普遍精度提升。
- GM 保留在主方法定义中；No-GM 作为成本/反馈路径对照。
- 资源章节同时报告 Student/inference 的低开销和 construction 的高峰值，并明确
  稀疏算子是已验证的小优化，而 blockwise Sinkhorn、激活 checkpoint/offload
  是后续工程方向。
