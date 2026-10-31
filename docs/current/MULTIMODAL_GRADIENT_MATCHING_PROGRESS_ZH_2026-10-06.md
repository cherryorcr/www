# 多模态梯度匹配分支阶段性总结

日期：2026-10-06

## 1. 本轮目标

本轮围绕“CaT 初始化 + 一步多模态 Gradient Matching + 可微文本路由”推进。目标不是继续堆叠 selector、EM 或拓扑模块，而是回答三个问题：

1. CaT 生成的特征 `X` 能否接受 Student-compatible 的梯度修正；
2. 文本 payload 能否通过可微 routing 参与同一个图文目标；
3. 训练阶段的软文本对象与最终保存的真实文本是否一致。

当前分支仍遵守：每个阶段只访问当前 raw delta 和上一阶段已经材料化的 synthetic memory，不读取历史完整 raw graph。

## 2. 当前实现流程

### 2.1 CaT 初始化

对当前阶段 delta 运行原 CaT，得到新合成特征：

\[
X^{(0)}_t = \operatorname{CaT}(\Delta_t,S_{t-1}).
\]

CaT 仍负责主要的图特征凝聚。当前实验没有修改 CaT 的核心目标，也没有把 CaT 的结果替换成随机特征。

### 2.2 文本候选池

当前 delta 中每个节点的文本先按最终材料化规则进行空白归一化和长度限制，然后重新编码：

\[
E_t=\{e_1,\ldots,e_K\},
\qquad e_k=\operatorname{MiniLM}(\operatorname{truncate}(text_k)).
\]

这一点是本轮后期修正的关键。旧实现优化的是未截断文本 embedding，最后却保存截断文本并重新编码，造成优化对象与实际 Student 输入不一致。

### 2.3 X-conditioned routing

对每个 CaT 合成行 `j`，只在同类别的当前 delta 文本候选中建立 routing logits。初始 logits 来自 `X_j^(0)` 与候选原始图特征的相似度：

\[
z_{jk}^{(0)} = \cos(X_j^{(0)},x_k).
\]

文本的软表示为：

\[
\tilde T_j=\sum_k \pi_{jk}e_k,
\qquad
\pi_j=\operatorname{softmax}(z_j/\tau).
\]

当前默认最终材料化为每个合成节点的 top-1 真实文本，因此不同合成节点可以使用不同文本，不是每个类别共享一条文本。

### 2.4 Prompt-only GraphText Student proxy

代理 Student 是两层对称归一化 GCN 加冻结 MiniLM 文本 embedding、可训练文本投影和 prompt cosine 分类，没有分类器 head。代理损失包括：

\[
L_{proxy}=0.5L_{InfoNCE}+L_{prompt}+0.1L_{margin}.
\]

真实侧和 synthetic 侧分别对代理 Student 参数求梯度：

\[
g_r=\nabla_\theta L_{proxy}(\Delta_t,S_{t-1}),
\qquad
g_s=\nabla_\theta L_{proxy}(S_t).
\]

梯度匹配采用余弦方向差和梯度范数差。`blockwise` 版本分别处理 GCN 参数块和文本投影参数块；`decoupled` 版本进一步规定：

- `X` 主要承担图参数块匹配；
- routing 主要承担文本投影参数块匹配；
- 两者仍来自同一个图文 proxy loss。

### 2.5 约束与材料化

新 `X` 受 CaT anchor 和相对 trust region 约束；旧 memory 默认不更新。routing 增加初始化保持项：

\[
L_{route-anchor}=\frac{1}{|R|}\sum_j
\left\|\frac{z_j-z_j^{(0)}}{\max(\|z_j^{(0)}\|,1)}\right\|_2^2.
\]

最后使用真实文本字符串材料化并重新编码。最后一个实验还使用 straight-through routing：前向使用真正的 top-1 文本，反向使用 soft routing 梯度，从而使训练时的前向对象与最终保存对象一致。

当前合成拓扑仍是 empty/self-loop-only。这是刻意保留的隔离设置，因此本分支尚未证明完整 TAG 拓扑凝聚效果。

## 3. 已完成实验

所有实验使用同一个 prompt-only Student、相同 memory budget、相同 report seeds `[150,151,152]`，并在三个类别顺序 `order19611/19612/19613` 上进行最终阶段评估。

### 3.1 版本定义

| 版本 | X | routing | 训练/保存文本一致 | 说明 |
|---|---|---|---|---|
| CaT | CaT 固定 | nearest text | 是 | 内部参照 |
| plain decoupled | X 可更新 | soft routing | 否 | 原始联合梯度版本 |
| multi-proxy | X 可更新 | 多 proxy seed 平均 | 否 | 检查梯度目标方差 |
| route-anchor | X 可更新/文本可路由 | soft routing + route anchor | 部分 | 检查 routing 漂移 |
| hard-consistent | 同上 | 候选先截断重编码 | 是 | 修正 soft-hard 对象不一致 |
| straight-through | 同上 | hard 前向、soft 反向 | 是 | 当前最完整联合版本 |

### 3.2 当前最完整版本：straight-through

最终阶段 Macro-F1 和 Accuracy 的三顺序平均如下。数值是三个顺序的 cell 平均，不是根据最好顺序挑选。

| 数据集 | CaT F1 | Straight-through text-only F1 | Straight-through decoupled F1 | CaT Acc | Text-only Acc | Decoupled Acc |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 83.279 | 82.581 | 83.585 | 85.240 | 85.076 | 85.486 |
| Citeseer | 68.799 | 68.414 | 68.810 | 73.946 | 73.424 | 73.563 |

相对 CaT：

| 数据集 | Text-only ΔF1 | Decoupled ΔF1 | Text-only ΔAcc | Decoupled ΔAcc |
|---|---:|---:|---:|---:|
| Cora | -0.698 | +0.305 | -0.164 | +0.246 |
| Citeseer | -0.386 | +0.010 | -0.522 | -0.383 |

因此，当前最完整的 `straight-through decoupled` 是一个有正向信号、但还没有稳定 SOTA 证据的候选。它在 Cora 平均提升，在 Citeseer 基本持平 F1，但 Accuracy 没有同步提升。

### 3.3 关键负结果

1. **旧 memory X revision**：在初始顺序 Citeseer 最终 F1 低于 CaT 约 `0.53`，说明无拓扑条件下更新旧 X 会放大历史误差。
2. **多 proxy seed 平均**：在 order19612 上，Cora text-only 相对 CaT 下降约 `1.61` F1，Citeseer 下降约 `0.42`，不能作为稳定化方案。
3. **仅 route anchor**：能改善部分顺序，但三顺序平均不能同时稳定提高 Cora 和 Citeseer。
4. **直接 soft routing**：第一顺序有收益，第二和第三顺序存在退化，说明 proxy 梯度方向与最终 top-1 文本材料化仍有偏差。
5. **X + 文本联合更新**：比只更新文本更有创新性，但对数据集和顺序敏感，不能仅凭一个正向 cell 宣称有效。

## 4. 目前已经证明和没有证明的内容

### 已经证明

- 可以在 CaT 的 `X` 初始化上构造可微文本 routing；
- 可以通过 proxy Student 的梯度匹配同时产生对 `X` 和文本 routing 的更新；
- blockwise/decoupled 能降低 flat gradient matching 中文本块对图块的支配；
- straight-through 能消除训练前向与最终 top-1 文本之间的显式对象不一致；
- 当前流程可以在两个小图和三个类别顺序上稳定运行，且没有历史 raw graph 读取。

### 尚未证明

- 还没有证明它稳定优于 CaT；
- 还没有证明它优于公开 TAG 凝聚基线；
- 还没有证明合成拓扑有效，因为当前分支暂时没有真实合成边；
- 还没有证明代理 Student 的梯度方向代表多个下游 Student；
- 还没有证明旧 memory revision 在当前图文目标下有收益。

## 5. 当前建议的主方法定位

暂定主方法可以写成：

> CaT-initialized, hard-consistent decoupled multimodal gradient matching with X-conditioned text routing.

中文可称为“CaT 初始化的硬一致解耦多模态梯度匹配凝聚”。

但论文结论暂时不能写成“稳定超过所有基线”或“已经是 SOTA”。更准确的定位是：

1. CaT 提供稳定的图特征凝聚起点；
2. 新方法通过同一个图文 proxy loss 对 `X` 和文本 payload 进行受限梯度匹配；
3. hard-consistent/straight-through 解决了软文本与最终文本材料化不一致的问题；
4. 结果在 Cora 上有平均增益，在 Citeseer 上基本保持，但收益受顺序和数据集影响；
5. 完整 TAG 方法仍需要补齐合成拓扑和公开基线对比。

## 6. 下一步优先级

### P0：冻结当前小图梯度版本

不再继续扫描 routing anchor、proxy seed、trust radius 或 GM steps。保留当前 straight-through decoupled 作为机制候选，同时把 plain、multi-proxy、route-anchor 和 hard-consistent 作为消融/负结果。

### P1：统一传播算子并补合成拓扑

当前真实侧使用真实图传播，而 synthetic 侧是 edge-free，这会让梯度匹配目标先天不对齐。下一步应固定 CaT 的合成拓扑或一个不增加新 Student 的简单拓扑材料化规则，让真实侧和 synthetic 侧使用同一种归一化传播算子，再重跑当前三顺序。不要同时修改 `X`、文本、拓扑和 Student。

### P2：公开基线与成本

在同一 current-delta-only、frozen-prefix、prompt-only Student 协议下比较 CaT、静态 TAG 适配、coreset、raw reference 和公开持续凝聚适配方法，同时报告 F1、Accuracy、Old/New F1、构造时间和 Student 时间。

### P3：大图与多下游 Student

只有当 P1 在两个小图上不出现系统性退化后，才迁移到大图。随后使用至少两个不同 Student 初始化和一个轻量下游任务检查，以判断方法是否是凝聚算法而不是单一 proxy Student 的拟合。

## 7. 客观结论

本轮已经把“没有梯度”“文本和 X 完全割裂”的问题改成了一个实际可运行的梯度匹配流程，并发现了两个重要实现约束：候选文本必须先按最终规则编码，软 routing 必须考虑最终 hard materialization。

但是，当前证据仍然支持“有研究价值的候选方法”，还不支持“已经足够作为 WWW 主方法并宣称 SOTA”。真正决定后续论文贡献的关键不再是继续增加文本模块，而是：

- 统一真实图和合成图的传播算子；
- 补齐合成拓扑；
- 在公开基线和多 Student 协议下证明泛化；
- 把当前顺序敏感的负结果如实写入机制分析。
