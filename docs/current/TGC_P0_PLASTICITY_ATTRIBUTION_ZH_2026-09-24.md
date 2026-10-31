# TGC-P0：新类塑性归因与下一轮方法决策

日期：2026-09-24  
状态：完成，分析产物和远程 checkpoint 已同步。  
结果目录：[tgc_p0_plasticity_attribution_20260924c](../../results/tgc_p0_plasticity_attribution_20260924c/)

## 1. 目的

R15/R17 已经表明：R11 的内容构造存在局部正向信号，但 R11 拓扑和 teacher-message gate 都没有稳定解决新类塑性。本轮不再构造新 memory，而是直接读取封存的 R15/R17 memory 和 student checkpoint，回答三个问题：

1. synthetic 新行是否已经具有正确类别的 prompt margin；
2. synthetic 新行的 margin 与真实图中新类节点的 margin 是否存在迁移差距；
3. 跨块传播和旧块归一化漂移是否比内容/Student transfer 更可能解释 new-F1 下降。

本轮不使用 report F1 选择任何构造参数，也不访问历史完整 raw graph。最终 F1 只用于把归因代理量与结果做相关分析。

## 2. 协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19601`、`order19602`、`order19603` |
| 评估阶段 | stage 1、stage 2 |
| student seeds | 150--155 |
| memory | 5% growing memory，沿用 R15/R17 封存结果 |
| student | C0 两层 GCN + trainable text projection + 双向 InfoNCE |
| 归因臂 | C0、R11 `anchor025`、R11 content + C0 topology、R17 `message_gate_2` |
| margin | 正确类别 prompt logit 减去最大错误类别 prompt logit |
| message mass | normalized adjacency 中跨旧/新块的接收比例 |
| propagation drift | 追加后旧块内部 normalized adjacency 相对 parent 的 Frobenius 漂移 |
| InfoNCE proxy | synthetic/raw 均使用 batch 16、固定 8 个 deterministic batches，仅作诊断 |

共生成 288 条逐 seed 观测记录。原始数据位于 `observations.csv`，相关性位于 `correlations.json`。

## 3. 汇总结果

下面统计 stage 1/2、两个数据集、三个顺序和六个 student seeds 的均值。

| 方法 | synthetic margin | real new margin | real - synthetic gap | new-F1 | final F1 |
|---|---:|---:|---:|---:|---:|
| C0 | -1.285 | 0.615 | 1.899 | 56.777 | 52.599 |
| R11 `anchor025` | -0.103 | 0.254 | 0.357 | 54.467 | 51.315 |
| R11 content + C0 topology | -0.610 | 0.187 | 0.797 | 55.474 | 52.128 |
| R17 `message_gate_2` | -0.590 | 0.323 | 0.913 | 54.857 | 51.576 |

### 3.1 传播结构诊断

| 方法 | new receive old message ratio | old receive new message ratio | old normalized adjacency drift |
|---|---:|---:|---:|
| C0 | 0.259 | 0.184 | 0.283 |
| R11 `anchor025` | 0.245 | 0.180 | 0.413 |
| R11 content + C0 topology | 0.259 | 0.184 | 0.283 |
| R17 `message_gate_2` | 0.259 | 0.184 | 0.302 |

R11 自己的 topology 确实带来更大的旧块归一化漂移；回退到 C0 topology 后该风险消失，但 new-F1 仍没有稳定超过 C0。因此拓扑是 R11 的放大器/风险源，不是当前全部问题的根因。R17 的局部 edge swap 只造成很小的结构变化，不能修复内容到真实图的迁移差距。

### 3.2 分数据集和阶段的 new-F1

| 数据集 | 阶段 | C0 | R11 `anchor025` | R11 content + C0 topology | R17 `message_gate_2` |
|---|---:|---:|---:|---:|---:|
| Cora | 1 | 51.212 | 51.226 | 48.377 | 50.248 |
| Cora | 2 | 54.720 | 51.904 | 52.316 | 51.079 |
| Citeseer | 1 | 65.763 | 60.627 | 66.012 | 63.768 |
| Citeseer | 2 | 55.414 | 54.111 | 55.191 | 54.331 |

R11 content + C0 topology 在 Citeseer stage 1 有局部正向，但 Cora stage 1 和 Citeseer stage 2 没有稳定收益；这与 R15 的整体晋升失败一致。

## 4. 相关性结果

在 72 个 stage/order/seed 观测上，real new margin 与 new-F1 的关系最稳定：

| 方法 | real margin Pearson | real margin Spearman | synthetic margin Pearson | margin gap Pearson |
|---|---:|---:|---:|---:|
| C0 | 0.799 | 0.588 | 0.653 | 0.355 |
| R11 `anchor025` | 0.851 | 0.674 | 0.728 | 0.269 |
| R11 content + C0 topology | 0.878 | 0.691 | 0.817 | 0.117 |
| R17 `message_gate_2` | 0.860 | 0.659 | 0.804 | 0.112 |

解释：

- 真实图中新类节点的 prompt margin 是当前最有解释力的诊断量；它比“synthetic margin 与 real margin 的差值”更能解释最终 new-F1。
- 仅把 synthetic margin 做高并不充分。R11 content 确实把 synthetic margin 从 C0 的负区间推高，但 real margin 没有同步提高，说明它可能改善了 synthetic support 上的可分性，却没有改善真实图上的迁移。
- synthetic InfoNCE loss 与 new-F1 的相关性接近零，说明“在合成图上训练得更好”不能作为主选择目标。
- 统一 batch 后，raw-delta InfoNCE 与 new-F1 呈中等负相关，但它仍是无标签的 post-training transfer proxy，需要在独立 current-delta split 上验证，不能直接当作最终指标。

## 5. 当前问题的定位

当前主要瓶颈按优先级排序为：

1. **synthetic-to-real transfer gap**：合成行在自己的图上容易被 student 拟合，但传播到真实完整图后，新类 prompt margin 没有同步提升。
2. **内容、拓扑和 student objective 的耦合**：R11 内容和 R11 topology 分开看都不能稳定改善；局部 topology gate 也只能降低 teacher message loss，不能保证真实新类 margin。
3. **scratch student 的支持集过拟合**：synthetic InfoNCE 很低，但这并不意味着在真实 current delta 上具有相同的图文对齐能力。
4. **拓扑归一化是次要但真实的风险**：R11 topology 的 old-block drift 明显更大，因此在没有 transfer 证据时不应继续使用它替换 C0 topology。

这轮结果不支持继续扫描简单 `alpha`、更多固定 edge swap 或单纯优化 synthetic InfoNCE。

## 6. 决策

### 保留

- C0 作为协议和拓扑锚点。
- R11 content + C0 topology 作为“内容方向有局部信号、但未晋升”的诊断臂。
- R17 message gate 作为“局部 message loss 可下降但 downstream plasticity 不稳定”的负结果。
- real new margin 和 current-delta transfer loss 作为后续无标签诊断量。

### 暂不晋升

- R11 `anchor025`。
- R11 content + C0 topology。
- R17 `message_gate_1/2`。
- 任何只根据 synthetic InfoNCE、teacher message loss 或固定 alpha 选择的版本。

## 7. 下一轮实验路线

下一轮不再先优化 selector，而是直接验证“student 可迁移性”是否能成为构造目标。

### P1：transfer-aware synthetic X 小实验

固定 C0 的 rows、class allocation、文本和 C0 topology，只对当前新块的 synthetic X 做有界优化：

1. student 在 synthetic memory 上执行固定短 inner horizon；
2. 用不含标签的 current-delta graph/text InfoNCE 作为 transfer objective；
3. 只允许修改新行，旧行和旧边完全冻结；
4. 加入 C0 X anchor 和 displacement bound，失败立即回退 C0；
5. 使用独立的 current-delta probe split，避免把同一批 raw rows 同时用于优化和选择；
6. 只在 Cora/Citeseer、三个已注册顺序上先做 smoke，再决定是否扩大。

该实验要回答：直接让 synthetic X 适应真实 current delta，是否比继续调 alpha 或 teacher proxy 更能提升 real new margin 和 new-F1。

### P2：共享 assignment 的有限多原型

只有 P1 有正向 transfer 信号后才做：固定总 rows，在类内高离散区域分配两个共享 X/text prototype；X 与文本使用同一 assignment，保留 C0 topology。比较 single-mean、two-prototype 和 C0，不与 topology 或 budget 同时改变。

### P3：student 训练协议诊断

若 P1 的 raw transfer objective 仍不能提升 real margin，则比较 scratch、warm-start 和 class-balanced InfoNCE；这一步只改变 student protocol，不修改 memory 构造。

### 晋升条件

新版本至少满足：两个数据集、三个独立顺序上平均 new-F1 不下降；stage mean 不下降；最差 final cell 不低于约 -2 pp；real new margin 有同步提升；构造成本不超过 C0 的 2 倍。否则回退到 C0，并将 P1/P2 写入机制分析和负结果。

## 8. 可复现文件

- 分析器：[run_tgc_plasticity_attribution.py](../../code/v2_llm_fusion/run_tgc_plasticity_attribution.py)
- 远程入口：[tgc_plasticity_attribution_remote.py](../../tools/tgc_plasticity_attribution_remote.py)
- 原始观测：[observations.csv](../../results/tgc_p0_plasticity_attribution_20260924c/observations.csv)
- 相关性：[correlations.json](../../results/tgc_p0_plasticity_attribution_20260924c/correlations.json)
- 远程运行状态：288/288 条观测完成，未重新构造 memory，未使用 report F1 选择参数。
