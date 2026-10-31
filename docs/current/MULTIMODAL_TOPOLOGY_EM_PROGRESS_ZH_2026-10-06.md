# 多模态梯度匹配的拓扑与 EM 阶段进展

日期：2026-10-06

## 1. 本轮目标

在 CaT 初始化的多模态梯度匹配方法中加入合成拓扑，并验证 EM-like 交替优化是否能够改善 raw graph 到 synthetic memory 的消息传播一致性。

本轮仍遵守：每个阶段只访问当前 raw delta 和已经材料化的 parent memory，不读取历史完整 raw graph；Student、文本编码器、memory budget 和评估协议不变。

## 2. 当前实现

### 2.1 固定 Q 拓扑

CaT 先得到初始合成特征：

\[
X^{(0)}=\operatorname{CaT}(\Delta_t).
\]

对当前 delta 的 raw 节点和合成 slot 建立类别内 Sinkhorn assignment：

\[
Q_{ij}\propto\exp\left(
\frac{\lambda_x\cos(\hat A_\Delta x_i,X_j)
 +\lambda_t\cos(e_i,T_j)}{\tau}
\right).
\]

Q 的行和为 1，列质量由当前类别的 raw 节点数和 synthetic slot 数确定。

合成软邻接矩阵为：

\[
W=Q^\top A_\Delta Q.
\]

最终按照 raw delta 的平均度缩放边数，并保存为 hard `edge_index`。旧 memory 的边保持不变，新边只来自当前阶段的新 block。

### 2.2 拓扑一致的 CaT refinement

原 CaT 直接让 synthetic X 匹配已经传播过的 raw feature。加入合成边后，如果继续直接使用该 X，会产生重复传播。因此新增：

\[
L_{\text{topo-CaT}}
=
\operatorname{MMD}\left(
\hat A_\Delta X_\Delta,
\hat A_s X_s
\right).
\]

该 refinement 从 CaT 的 X 出发，并带有 CaT anchor，避免拓扑分支完全破坏 CaT 解。

### 2.3 EM-like 分支

E-step 使用当前 synthetic message 重新计算 Q：

\[
C_{ij}=C_{ij}^{x}+C_{ij}^{text}
 +\lambda_m\left(1-\cos(m_i^r,m_j^s)\right).
\]

M-step 固定 Q 和合成拓扑，使用当前 decoupled multimodal gradient matching 更新 X 与文本 routing。当前实现每隔固定 M-step 进行有限次 Q 重估，属于 Generalized EM / EM-like alternating optimization。

## 3. 代码文件

- `D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/multimodal_gradient_matching.py`
- `D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_multimodal_gradient_matching_20261006.py`
- `D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/test_multimodal_gradient_matching.py`

新增的主要接口：

- `build_fixed_topology`
- `refine_topology_consistent_x`
- `topology_mode="fixed_q"`
- `topology_mode="em_q"`
- `cat_gm_decoupled_topology`
- `cat_gm_decoupled_em`

本地和服务器单元测试均通过：`5 passed`。

## 4. 结果

正式修正版使用 Cora/Citeseer、三个类别顺序、相同 Student seeds 和相同预算。

### Cora 最终阶段

| 方法 | Macro-F1 平均 | Accuracy 平均 |
|---|---:|---:|
| CaT X + decoupled GM | 82.659 | 84.994 |
| 固定 Q 拓扑 + decoupled GM | 80.746 | 83.067 |

固定拓扑三个顺序的 F1 为：80.960、80.970、80.307；对应 decoupled GM 为：80.952、84.778、82.246。

### Citeseer 最终阶段

| 方法 | Macro-F1 平均 | Accuracy 平均 |
|---|---:|---:|
| CaT X + decoupled GM | 68.029 | 73.180 |
| 固定 Q 拓扑 + decoupled GM | 68.497 | 72.588 |

固定拓扑三个顺序的 F1 为：70.101、68.718、66.672。该结果表现为部分顺序提升，但最差顺序明显下降。

EM-like smoke 结果在 Cora order19611 上低于固定拓扑和无拓扑版本，说明当前 message-aware Q 更新会放大错误拓扑，尚不能作为主方法。

## 5. 结果说明

### 已经确认

1. 合成拓扑不能直接附加到原 CaT X 上，否则会出现 raw propagation 和 Student propagation 的重复计算。
2. 拓扑一致的 CaT refinement 可以消除这种实现层面的重复传播问题。
3. 固定 Q 拓扑分支可以合法写入 memory，并在构造与 Student 评估阶段使用同一传播算子。
4. EM-like Q 更新能够实际改变 assignment 和拓扑，但当前消息代价还不足以保证拓扑质量。

### 尚未确认

1. 当前 Q 是否能学习到真正有用的 synthetic edge structure。
2. 当前 delta-only 拓扑是否足以恢复新旧阶段之间的边界关系。
3. 拓扑是否能够在不同顺序和数据集上稳定改善最终 Student。

## 6. 当前问题的主要来源

当前拓扑只从当前 delta 的 induced edges 构造。它没有访问历史 raw graph，因此无法恢复：

- 新节点与旧节点之间的历史边；
- 被采样到 delta 之外的邻居消息；
- 合成 slot 之间真实的局部连接模式。

此外，Q 的初始代价主要来自特征和文本相似度，message compatibility 目前只是后续 EM 的附加项，因此容易得到“语义相似但传播不兼容”的边。

## 7. 下一步建议

暂时不把固定拓扑或 EM 写成最终主方法。下一步优先顺序应为：

1. 加入可审计的 boundary-message sketch：当前新节点只携带其可见的旧 memory slot 消息统计，不读取历史完整 raw graph。
2. 让 Q 的 E-step 同时匹配 raw boundary message 和 synthetic boundary message。
3. 先做 `message-aware assignment`，固定 X 和文本，确认 Q/拓扑本身是否有正向信号。
4. 再做 `message-aware assignment + multimodal GM` 联合版本。
5. 只有在至少两个数据集的平均结果不退化后，才继续增加 EM 轮数或允许旧拓扑 revision。

当前最稳妥的主线仍是：

> CaT-initialized hard-consistent decoupled multimodal gradient matching with X-conditioned text routing。

拓扑和 EM 目前应作为待验证扩展，而不是替代当前无拓扑的最佳梯度版本。
