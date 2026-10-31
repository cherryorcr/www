# Unified-Q 多模态持续 TAG 凝聚：实现与实验记录

## 1. 目标

本分支验证一个减法后的构造器：不再依次运行“CaT 优化 X、X 选择文本、再用 Student 梯度修正 X/文本、最后单独生成拓扑”，而是使用一个类约束的软分配矩阵 `Q` 同时决定新合成节点的特征、文本 payload 和新增拓扑。

实现文件：

- `code/v2_llm_fusion/unified_q_multimodal.py`
- `code/v2_llm_fusion/run_unified_q_multimodal_20261006.py`
- `tools/summarize_unified_q_multimodal_20261006.py`

## 2. 方法流程

### 2.1 Q 的结构化初始化

当前增量原始节点为 `Delta_t`，合成节点数为 `m`。首先按类别分配 slot，禁止跨类别分配。对每个原始节点和 slot 计算三种兼容性：

\[
C_{ij}=\lambda_x C^x_{ij}+\lambda_m C^m_{ij}+\lambda_t C^t_{ij}.
\]

其中 `C_x` 是特征相似度，`C_m` 是局部消息相似度，`C_t` 是文本相似度。再对每个类别独立运行容量约束 Sinkhorn，得到：

\[
Q_{ij}\geq 0,\quad \sum_jQ_{ij}=1,
\quad \sum_iQ_{ij}\approx |\Delta_c|/m_c.
\]

因此 Q 不是仅按 X 或文本相似度初始化，而是显式包含结构消息。

### 2.2 Q 同时生成 X、文本和拓扑

合成特征为：

\[
X_Q=D_Q^{-1}Q^T X_\Delta+\Delta X.
\]

文本 payload 为：

\[
T_Q=\operatorname{norm}(D_Q^{-1}Q^T E_\Delta).
\]

优化前向使用 straight-through hard payload：每个 slot 的前向文本取该
slot 当前 Q 权重最高的一个真实增量文本，反向梯度仍通过 soft Q 聚合传递。
最终保存同一个截断后的真实文本并重新编码。因此训练对象和 Student 实际
看到的文本对象一致；这也是硬一致版本相对初版更稳定的主要修正。

新节点内部拓扑由：

\[
A_{nn}=D_Q^{-1/2}Q^TA_\Delta QD_Q^{-1/2}
\]

诱导。优化期间使用软加权拓扑，最终保存时根据软边权选择固定数量的二值边。

### 2.3 图文梯度匹配

真实侧使用当前 raw delta、父 synthetic memory 和合法的边界摘要构造 reference graph；合成侧使用 `Q` 生成的 X、文本和拓扑。二者进入同一个 prompt-only GraphText Student proxy：

\[
g_r=\nabla_\theta L_{proxy}(G_r),
\qquad
g_s=\nabla_\theta L_{proxy}(G_s).
\]

优化目标为：

\[
L=L_{GM}+\lambda_pL_{prop}+\lambda_qL_Q+\lambda_xL_X
 +\lambda_{cat}L_{cat-anchor}.
\]

这里 `L_prop` 约束合成新节点消息接近 Q 聚合的真实消息；`L_Q` 防止分配矩阵过度漂移；`L_X` 限制残差更新；`L_cat-anchor` 只在控制版本中启用，不是主方法必要组件。

### 2.4 新旧边

每个阶段保存每个 synthetic slot 的原始支持节点 ID。下一阶段只读取当前新增节点到历史节点的边 ID，并通过父 memory 的支持 ID 映射到旧 slot，构造 boundary sketch `B`。新旧软边为：

\[
A_{no}=D_Q^{-1/2}Q^TB.
\]

该过程不读取历史 raw 的特征和文本。没有可用的当前到历史边时，方法退化为 boundary-free 版本，不能声称恢复了真实历史边。

## 3. 对比版本

| 版本 | 含义 |
|---|---|
| `cat_x` | CaT 优化 X，使用已有 current-delta 文本附着，edge-free |
| `cat_gm_decoupled` | 原 CaT warm-start 的 decoupled multimodal gradient matching |
| `unified_q_edge_free` | Q 生成 X/文本，但 Student 使用 edge-free 新 memory，作为拓扑消融 |
| `unified_q` | Q 同时生成 X、文本和新节点拓扑，不使用边界摘要 |
| `unified_q_cat_anchor` | Unified-Q 加 CaT X anchor，作为稳定性控制 |
| `unified_q_boundary` | Unified-Q 加当前到历史的 boundary sketch |

所有版本使用相同的 current-delta-only 访问、memory budget、prompt-only Student、类别顺序和 report seeds。`unified_q_boundary` 是扩展协议，因为它额外保存原始支持 ID 和边界摘要；它不应和严格 edge-free memory 直接宣称完全等价。

## 4. 实验计划与当前状态

- 数据集：Cora、Citeseer。
- 类别顺序：`order19611`、`order19612`、`order19613`。
- 三阶段、5% cumulative memory。
- Student report seeds：150、151、152。
- construction：CaT 3000 steps，GM 30 steps，Unified-Q 30 steps；Unified-Q 使用
  straight-through 文本和硬拓扑前向，最终 materialize 为单个真实文本来源。
- GPU：远程两张 RTX 4090，Cora/Citeseer 并行。

结果由 `tools/summarize_unified_q_multimodal_20261006.py` 生成，最终报告同时包含 Macro-F1、Accuracy、Old F1、New F1、阶段均值和构造时间。

## 5. 硬一致版本结果

数值为三个类别顺序 `order19611/19612/19613` 的最终阶段均值；括号为顺序间标准差。

### Cora

| 方法 | Final Macro-F1 | Final Accuracy | Stage-average F1 | Stage-average Accuracy |
|---|---:|---:|---:|---:|
| CaT X (`cat_x`) | 82.963 ± 1.252 | 84.912 ± 0.669 | 88.843 | 90.632 |
| CaT + decoupled GM | 82.548 ± 1.370 | 84.666 ± 0.780 | 88.667 | 90.509 |
| Unified-Q edge-free | 80.119 ± 0.601 | 82.329 ± 0.877 | 86.018 | 88.558 |
| Unified-Q | 79.967 ± 1.275 | 82.329 ± 0.406 | 86.449 | 88.664 |
| Unified-Q + CaT anchor | 80.012 ± 1.156 | 82.411 ± 0.460 | 86.531 | 88.620 |
| Unified-Q + boundary | 80.062 ± 1.267 | 82.534 ± 1.479 | 86.066 | 88.519 |

### Citeseer

| 方法 | Final Macro-F1 | Final Accuracy | Stage-average F1 | Stage-average Accuracy |
|---|---:|---:|---:|---:|
| CaT X (`cat_x`) | 68.799 ± 0.064 | 73.946 ± 0.493 | 79.343 | 82.521 |
| CaT + decoupled GM | 68.839 ± 0.685 | 73.668 ± 0.666 | 78.903 | 82.041 |
| Unified-Q edge-free | 67.625 ± 1.236 | 71.752 ± 1.011 | 78.036 | 80.721 |
| Unified-Q | 67.900 ± 0.135 | 71.508 ± 0.996 | 77.678 | 80.255 |
| Unified-Q + CaT anchor | 66.658 ± 1.777 | 70.428 ± 1.341 | 77.396 | 79.926 |
| Unified-Q + boundary | 63.970 ± 2.551 | 66.388 ± 2.951 | 75.498 | 77.829 |

## 6. 结果解释

1. 硬一致修正有效。初版训练时用 soft text/soft topology、评估时用 hard
   materialization，Unified-Q 在 Citeseer 明显下降；硬一致版本把差距缩小到
   约 0.90 个最终 F1 点，说明之前部分退化来自训练-保存对象不一致。
2. Q 本身是可行的，但尚未超过 CaT。Cora 仍低约 3.00 个最终 F1 点，Citeseer
   低约 0.90 个点。当前结论应是“结构简化后可运行且竞争力尚可”，不能写成
   已经优于 CaT。
3. CaT anchor 没有稳定收益。它在 Cora 与无 anchor 接近，在 Citeseer 反而下降，
   因而 CaT 更适合作为 baseline 或可选传播正则，而不是主方法必需模块。
4. boundary sketch 仍是瓶颈。它在 Cora 接近 Unified-Q，但 Citeseer 低约 3.93
   个最终 F1 点，说明当前边界摘要只保留了连接质量，没有保留足够的跨阶段
   message compatibility；暂不应把该版本作为最终主方法。

当前最稳妥的候选是 `Unified-Q`，拓扑作为同一 Q 的诱导结果保留；boundary
版本暂列扩展和负结果。若论文需要与 CaT 公平比较，应报告“接近但未超过”，
并把贡献放在 current-delta-only、图文拓扑统一变量和硬一致 materialization，
而不是 SOTA 宣称。

## 7. 解释边界

该分支的目的不是预先保证 SOTA，而是验证“一个 Q 是否可以替代多段重复优化”。如果 `unified_q` 在多数顺序只小幅落后 CaT/GM，说明减法方向可行；如果 `unified_q_boundary` 明显更差，则说明边界摘要的消息兼容性仍不足，应作为失败扩展，而不是强行放入主方法。

