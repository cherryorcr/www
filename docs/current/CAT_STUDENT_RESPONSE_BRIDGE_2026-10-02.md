# CaT Student-Compatible Response Bridge 实验报告

日期：2026-10-02  
状态：最小 Student-response 可行性验证完成；不晋升为最终主方法

## 1. 实验目的

本轮不是重新设计完整的持续 TAG 凝聚算法，而是针对 CaT 的一个明确缺口做最小验证：

> CaT 先构造当前新增合成块，随后用与下游任务一致的 Student 训练响应，反向修正当前新增合成节点的特征 `X`，能否比不使用 Student 响应的 `cat_x` 更好。

本轮保持原协议不变：

- 原始两层 GCN Student；
- 冻结 MiniLM 文本编码器；
- 双向图文 InfoNCE；
- 5% cumulative memory；
- edge-free topology；
- 旧 synthetic prefix 不修改；
- 不读取历史完整 raw graph；
- 不引入 P/A、拓扑 revision、动态预算或 GraphGPS；
- Cora/Citeseer，4 个类别顺序，6 个 Student report seeds。

因此，结果只反映“当前新增 `X` 的短程 Student-response 修正”，不代表完整的 X/P/A 联合优化。

## 2. 基线与候选方法

### 2.1 `cat_x`

`cat_x` 是本轮的冻结 CaT 对照：

1. 从当前合法 raw delta 生成当前新增的 `X`；
2. 用相同的 CaT 文本构造生成当前文本 embedding；
3. 将新块 append 到已保存的 synthetic prefix；
4. 使用 edge-free Student 图算子训练下游 Student。

旧 prefix 在当前阶段不变，memory 行数、文本来源数量、Student 训练步数和报告规则均固定。

### 2.2 `support_x_only`

它在 `cat_x` 的当前新增 `X` 上增加一层短程可微 Student response：

1. 把当前新增 `X` 作为可优化变量 `X = X_0 + ΔX`；
2. 固定文本、旧 prefix 和拓扑；
3. 用当前合法 raw delta 的传播特征和文本作为 support；
4. 对 canonical Student 做短程 inner unroll；
5. 对新合成块的图文响应和当前 raw support 响应求外层损失；
6. 反向传播到 `ΔX`；
7. 施加每行半径约束，再 append 当前块。

该版本已经消除了 text response 带来的大幅退化，但总体只是基本不劣：Cora 平均 `-0.132 pp`，Citeseer 平均 `+0.043 pp`。

### 2.3 `mmd_support_x_only`

这是本轮新增的最小稳定化版本：

\[
L = L_{response} + \lambda_{mmd} L_{MMD} +
\lambda_x\|\Delta X\|^2.
\]

其中：

\[
L_{response} =
\lambda_{new}L_{NCE}^{new}+
\lambda_{support}L_{NCE}^{support}+
\lambda_{prompt}L_{CE}^{new}+
\lambda_{support-prompt}L_{CE}^{support}.
\]

`L_MMD` 使用当前合法 raw delta 与新增 synthetic `X` 的类别条件分布匹配，并通过固定随机 probe 计算，避免引入额外可训练判别器。

在本轮的 `x_only` 模式中：

- `X` 可更新；
- 文本 embedding 固定；
- 旧 prefix detached；
- 拓扑固定为空边；
- 只允许当前新增 block 发生改变；
- 当前阶段之后不回写旧 synthetic 行。

## 3. 梯度路径与访问边界

对当前新增块，外层更新路径为：

\[
X_0 \rightarrow X_0+\Delta X
\rightarrow \text{Student inner unroll}
\rightarrow L_{response}
\rightarrow \nabla_{\Delta X}L.
\]

Student 参数在 inner loop 中更新，但最终保存的是修正后的当前 `X`，不是 Student 权重。旧 synthetic prefix 仅作为 detached context，历史 raw graph 没有被重新读入。

实现中的关键限制是：

- `old_prefix_updated = False`；
- `historical_raw_reads = 0`；
- `topology_updated = False`；
- `budget_updated = False`；
- 当前新增行的相对更新受到 `x_radius=0.05` 的投影约束。

## 4. 正式矩阵结果

### 4.1 与 `support_x_only` 的对照

| 数据集 | 方法 | 平均 F1 差值 | 正收益 cell | 最差 cell | 平均 Accuracy 差值 | 构造成本倍率 |
|---|---|---:|---:|---:|---:|---:|
| Cora | `support_x_only` vs `cat_x` | -0.132 pp | 7/12 | -2.460 pp | -0.084 pp | 1.32x |
| Citeseer | `support_x_only` vs `cat_x` | +0.043 pp | 6/12 | -1.528 pp | +0.142 pp | 1.34x |

### 4.2 `mmd_support_x_only` vs `cat_x`

| 数据集 | 平均 cell F1 差值 | 阶段平均差值 | 正收益 cell | 正收益顺序 | 最差 cell | 最差顺序阶段平均 | Accuracy 差值 | 构造成本倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | **+0.301 pp** | **+0.301 pp** | 6/12 | 3/4 | -0.804 pp | -0.033 pp | **+0.252 pp** | 1.32x |
| Citeseer | -0.139 pp | -0.139 pp | 6/12 | 0/4 | -1.656 pp | -0.300 pp | -0.027 pp | 1.33x |

补充统计：

- Cora 的 cell 差值标准差为 `0.960 pp`；Citeseer 为 `0.848 pp`；
- Cora 最好 cell 为 `+3.183 pp`；Citeseer 最好 cell 为 `+1.199 pp`；
- 两个数据集的 cell 正收益比例都只有 `50%`；
- 候选构造的平均耗时约为 CaT 的 `1.32--1.33x`；
- 所有矩阵均使用同一组 6 个 Student seeds，候选与 CaT 是 paired comparison。

### 4.3 `new_margin_x_only` vs `cat_x`

该版本只把当前新增 synthetic block 的 prompt 分类 margin 作为 outer 目标，文本、拓扑、旧 prefix、support loss 和 MMD 均固定或关闭。

| 数据集 | 平均 cell F1 差值 | 阶段平均差值 | 正收益 cell | 正收益顺序 | 最差 cell | 最差顺序阶段平均 | Accuracy 差值 | 构造成本倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | -0.207 pp | -0.207 pp | 5/12 | 2/4 | -2.228 pp | -0.893 pp | -0.139 pp | 1.30x |
| Citeseer | -0.127 pp | -0.127 pp | 3/12 | 1/4 | -0.995 pp | -0.334 pp | -0.073 pp | 1.32x |

该矩阵使用与前两轮相同的 4 个顺序和 6 个 paired Student seeds。它说明新类 margin 梯度可以正确计算并产生可审计的 `X` 更新，但“直接让新类 prompt 间隔变大”不是一个足够的单独凝聚目标。

## 5. 结果解释

### 5.1 MMD 确实有局部稳定作用，但不是普适任务目标

Cora 上，MMD 约束将 `support_x_only` 的平均结果从 `-0.132 pp` 推到 `+0.301 pp`，并将最差 cell 从 `-2.460 pp` 收窄到 `-0.804 pp`。这说明原始 Student response 在修改 `X` 时存在分布漂移，MMD 能阻止一部分过度偏移。

但 Citeseer 上平均结果仍为负，且四个顺序的阶段平均全部低于 `cat_x`。因此 MMD 解决的是“不要偏离当前 raw 分布太远”，不是“让合成节点更适合最终 Student 分类”。当 raw feature 分布与下游图文判别边界不完全一致时，MMD 甚至可能把 `X` 拉回一个并不利于 prompt 分类的方向。

### 5.2 当前主要问题不是没有梯度，而是 response 目标与最终评价仍不完全一致

当前 outer objective 能够对 `X` 产生真实梯度，平均每个候选阶段的 `X` 相对行位移约为 `2.3%`，说明梯度路径没有失效。但存在三个限制：

1. inner unroll 很短，不能完整代表最终 Student 训练轨迹；
2. support loss 使用当前合法 raw delta 的局部传播响应，最终评估使用的是 synthetic memory 上的长程训练响应；
3. `X` 仍在 CaT 初始解附近做小半径修正，不能重新发现节点分配、文本来源或结构角色。

因此，`loss` 下降或 response 变小，并不自动等价于最终 Macro-F1 提升。

### 5.3 不能据此继续扩大 response selector

本轮只优化当前新增 `X`，已经付出约 `1.32x` 构造成本，却没有在两个数据集上稳定提升。继续增加 response seeds、候选 selector、动态预算或长 unroll，会进一步放大成本和选择噪声，暂时缺乏依据。

### 5.4 新类 margin 的失败原因

`new_margin_x_only` 的失败不是实现错误，而是目标错配的证据：

1. margin 只观察当前阶段新类的 prompt 判别，不约束旧类响应；
2. inner unroll 的短期 margin 改善可能在最终较长 Student 训练后消失；
3. synthetic rows 共享一个 edge-free memory，单行 margin 改善可能改变整个 batch 的 InfoNCE 几何；
4. 新类 margin 没有保留 CaT 对当前 raw delta 的类内分布约束。

因此，单纯把 `L_response` 换成 `L_new-margin` 会出现“新类目标更明确、总体结果反而下降”。这支持继续研究 Student-compatible 梯度，但不支持把 margin 单独当作主凝聚目标。

## 6. 当前结论

本轮不把 `mmd_support_x_only` 或 `new_margin_x_only` 升级为论文主方法，原因不是它们完全无效，而是它们没有满足稳定晋升条件：

- 两个数据集平均均为正；
- 多数独立顺序阶段平均为正；
- 最差 cell 退化受控；
- 增加的成本值得其收益。

它目前更适合作为以下证据：

> CaT 的新增 `X` 可以接受 Student-response 梯度修正；MMD 能在部分数据集上抑制 response-induced distribution drift，但简单的 support response + MMD 仍不能保证跨数据集的下游收益。

目前能够确认的最小事实是：

- Student-response 对 `X` 的梯度路径真实存在；
- support response 单独使用基本不劣但没有稳定平均收益；
- MMD 在 Cora 上能收窄最差退化，在 Citeseer 上不能泛化；
- new-class margin 单独使用会加剧顺序敏感性；
- 下一步应做受限目标融合，而不是继续替换或增加更多模块。

## 7. 下一步优化方向

不再继续盲目调 `mmd_weight`、`outer_lr`、`x_radius` 或 margin。下一轮只验证一个受限混合目标：

### 7.1 受限的新类边界响应

把当前新增类的 prompt 分类 margin 作为主目标，同时保留小权重的 CaT/MMD anchor：

\[
L_{new-margin} =
\frac{1}{|S_{new}|}\sum_{i\in S_{new}}
\max(0,m-s(g_i,t_{y_i})+
\max_{c\ne y_i}s(g_i,t_c)).
\]

margin 不能再单独承担全部 outer objective，而应作为小权重项加入 support/CaT 保持目标。固定 `m`、固定权重和固定半径，不根据最终测试 F1 扫描选择。首个版本只允许当前新增 `X` 更新，旧 prefix 仍然冻结。

### 7.2 受限混合目标

下一版候选形式为：

\[
L_{bounded} = L_{support}
 + \lambda_m L_{new-margin}
 + \lambda_{old}L_{old}
 + \lambda_a\|\Delta X\|^2.
\]

其中 `λ_m` 固定为小权重，`λ_old` 只在存在旧 prefix 时生效；不加入新模块、不修改文本和拓扑、不允许回写旧行。实验先做一个 smoke 和一个独立顺序确认，再决定是否跑完整矩阵。

### 7.3 保留 CaT anchor，限制更新对象

如果新类目标有效，再逐步加入：

- 新增 `X` 与 CaT 初始 `X_0` 的小半径约束；
- 旧 prefix 的响应保持项，但不更新旧 prefix；
- 当前 raw delta 的 class-balanced support；
- 最后才考虑有限的旧新 payload transport。

### 7.4 暂不加入的模块

下一轮不同时加入：

- 连续文本 response；
- P/A 联合优化；
- 拓扑 revision；
- 动态预算；
- GraphGPS Student；
- 大规模 response selector；
- 无界旧 memory refresh。

这些模块会掩盖“新类目标本身是否有效”这一核心问题。

## 8. 代码与证据

- `D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/student_response_bridge.py`
- `D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_cat_response_bridge_20261002.py`
- `D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/test_student_response_bridge.py`
- `D:/Aprojects/continue_tag_condensation/tools/summarize_cat_response_bridge_20261002.py`
- `D:/Aprojects/continue_tag_condensation/tmp_remote_results/cat_response_xonly_full_20261002/summary.json`
- `D:/Aprojects/continue_tag_condensation/tmp_remote_results/cat_response_xonly_mmd_full_20261002_v2/summary.json`
- `D:/Aprojects/continue_tag_condensation/tmp_remote_results/cat_response_new_margin_full_20261002/summary.json`

本地单元测试：`4 passed`。

