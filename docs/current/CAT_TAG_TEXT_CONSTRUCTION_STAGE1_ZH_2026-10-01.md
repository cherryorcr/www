# CaT-TAG 文本构造第一阶段实验报告

日期：2026-10-01  
状态：第一阶段受控实验、四顺序公开基线矩阵、top-k/soft prototype 消融和三张大图 base+4 验证已完成；CCTP 保留为当前主方法候选，但大图结果显示明显数据集依赖，尚未宣称跨数据集稳定优于基线

## 1. 实验目的

这一轮实验固定 CaT 的类条件 MMD 合成节点构造，只改变合成节点的文本附着方式。目标是回答：

> 在相同 `X`、相同 memory budget、相同 Student 和相同训练协议下，TAG 特有的图文文本选择是否能稳定改善持续图凝聚结果？

这一步刻意不加入 Student outer unroll、拓扑 revision、动态预算、GraphGPS 或旧 memory refresh。这样可以把结果归因到文本构造，而不是多个模块的混合。

## 2. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19611`、`order19612` |
| memory | 累积合法 pool 的 5% |
| 合成骨架 | CaT class-balanced MMD / CGM，`cat_steps=3000` |
| 合成节点分配 | 当前 delta 内按类别比例分配，并保证每个当前类别至少一行 |
| Student | 原有两层 GCN + text projection，双向图文 InfoNCE，从 synthetic memory scratch training |
| Student steps | `max(50, min(2048, 16 * memory_rows))` |
| Student batch / lr | 16 / 0.001 |
| report seeds | 150--155，共 6 个 |
| 文本来源 | 只能来自当前 delta，不能回读历史 raw |
| 文本来源数 | 每个合成节点最多 5 个 |
| 文本字符预算 | 每个合成节点最多 320 字符，包含分隔空格 |
| 图拓扑 | CaT memory 保持 edge-free；不使用合成图边 |
| 评价 | seen-node report-dev，transductive full real graph；不读取 test label |

当前实验是公开 CaT 方法的 TAG 适配，不是官方 CaT 的端到端复现。CaT 原始监督 TiM/GNN 接口替换为项目统一的图文 InfoNCE Student；因此结果用于比较 TAG 文本机制在统一持续协议下的作用。

## 3. 四种文本模式

### 3.1 `nearest`

这是冻结的 CaT-TAG 基线。对每个合成节点，使用其合成 `X` 与当前 delta 中同类节点的图传播特征计算余弦相似度，选取最多 5 个最近文本，截断并拼接，再经过冻结 MiniLM 重新编码。

它只回答“CaT 的 X 加一个简单可审计文本附着”能达到什么结果。

### 3.2 `tagsam_style`

对当前同类候选同时计算：

\[
s_i=0.55s_i^{graph}+0.35s_i^{text}+0.10s_i^{prompt}.
\]

其中图分数衡量候选图传播特征与合成 `X` 的相似度，文本分数衡量候选文本与当前类别文本中心的相似度，prompt 分数衡量候选文本与类别 prompt 的相似度。之后使用固定多样性系数 `0.15` 的贪心 MMR，避免 5 个来源完全重复。

该模式是受 TAGSAM 机制启发的适配，不应写成官方 TAGSAM 复现。

### 3.3 `graph_conditioned`

先由图相似度构造软图区域：

\[
q_i=\operatorname{softmax}(s_i^{graph}/0.25),
\qquad
z_{gt}=\operatorname{norm}\left(\sum_iq_i t_i\right).
\]

再根据候选文本与图条件文本 prototype 的相似度打分：

\[
s_i=0.55s_i^{graph}+0.30s_i^{cross}
      +0.10s_i^{text}+0.05s_i^{prompt}.
\]

最后同样使用固定预算的 MMR 选择来源。

### 3.4 `graph_text_aligned`

只使用当前 delta 拟合一个带 bias 的 ridge graph-to-text 线性映射：

\[
W=\arg\min_W\|[G,1]W-T\|_2^2+\lambda\|W_{graph}\|_2^2,
\qquad \lambda=0.01.
\]

再将图条件 prototype 映射到冻结文本空间，与候选文本计算对齐分数：

\[
s_i=0.55s_i^{graph}+0.40s_i^{aligned}+0.05s_i^{prompt}.
\]

该映射只使用当前 delta，不能访问历史 raw，也不是下游 Student 参数。每个合成节点最终仍是固定最多 5 个文本源的截断拼接和重新编码。

## 4. 最终阶段 Macro-F1

### 4.1 Cora

| 方法 | order19611 | order19612 | 两顺序均值 |
|---|---:|---:|---:|
| CaT `nearest` | 53.903 | 51.148 | 52.526 |
| CaT `tagsam_style` | **57.118** | **54.358** | **55.738** |
| CaT `graph_conditioned` | 57.084 | 53.184 | 55.134 |
| CaT `graph_text_aligned` | **58.196** | 51.288 | 54.742 |

相对 `nearest` 的两顺序均值：

| 方法 | Δ Final Macro-F1 |
|---|---:|
| `tagsam_style` | +3.212 |
| `graph_conditioned` | +2.608 |
| `graph_text_aligned` | +2.216 |

### 4.2 Citeseer

| 方法 | order19611 | order19612 | 两顺序均值 |
|---|---:|---:|---:|
| CaT `nearest` | **57.278** | 51.036 | **54.157** |
| CaT `tagsam_style` | 49.938 | 50.884 | 50.411 |
| CaT `graph_conditioned` | 49.883 | **52.899** | 51.391 |
| CaT `graph_text_aligned` | 51.192 | 52.345 | 51.768 |

相对 `nearest` 的两顺序均值：

| 方法 | Δ Final Macro-F1 |
|---|---:|
| `tagsam_style` | -3.746 |
| `graph_conditioned` | -2.766 |
| `graph_text_aligned` | -2.389 |

## 5. 最终阶段 Accuracy

Accuracy 与 Macro-F1 同时报告，避免只观察类别均衡指标。

| 数据集 | 方法 | order19611 | order19612 | 两顺序均值 |
|---|---|---:|---:|---:|
| Cora | `nearest` | 58.733 | 61.439 | 60.086 |
| Cora | `tagsam_style` | **64.022** | **62.423** | **63.223** |
| Cora | `graph_conditioned` | 62.546 | 61.870 | 62.208 |
| Cora | `graph_text_aligned` | 63.161 | 59.410 | 61.285 |
| Citeseer | `nearest` | **60.658** | **55.068** | **57.863** |
| Citeseer | `tagsam_style` | 51.672 | 53.814 | 52.743 |
| Citeseer | `graph_conditioned` | 51.933 | **55.747** | 53.840 |
| Citeseer | `graph_text_aligned` | 52.612 | 55.643 | 54.127 |

## 6. 旧类、新类和阶段平均

以下为两个顺序的平均。阶段平均是三个阶段的 aggregate Macro-F1 的等权平均，用于观察整个增量过程；它不是按节点数加权的最终 F1。

| 数据集 | 方法 | Stage-average F1 | Final F1 | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|
| Cora | `nearest` | 47.215 | 52.526 | 47.538 | 59.175 |
| Cora | `tagsam_style` | **53.076** | **55.738** | **52.406** | 60.180 |
| Cora | `graph_conditioned` | 52.691 | 55.134 | 50.488 | **61.329** |
| Cora | `graph_text_aligned` | 52.027 | 54.742 | 50.638 | 60.215 |
| Citeseer | `nearest` | **52.954** | **54.157** | **55.546** | **51.378** |
| Citeseer | `tagsam_style` | 49.242 | 50.411 | 52.448 | 46.337 |
| Citeseer | `graph_conditioned` | 51.467 | 51.391 | 53.312 | 47.549 |
| Citeseer | `graph_text_aligned` | 49.616 | 51.768 | 53.099 | 49.107 |

Cora 上文本选择主要改善了旧类保持和新类表现；Citeseer 上四种模式均没有同时保持旧类和新类，说明问题不是简单的“文本越复杂越好”。

## 7. 选择和成本诊断

### 7.1 文本来源复用

每个合成节点最多选择 5 个来源。以所有阶段的构造审计为准，`nearest` 的来源复用较低，而 MMR 类方法为了增加多样性会让来源池更分散，但不保证下游有效：

| 现象 | 观察 |
|---|---|
| `nearest` | 每个阶段来源复用约 1.02--1.15 次，主要集中在图相似候选 |
| `tagsam_style` | 约 1.12--1.41 次，来源更加分散；Cora 受益，Citeseer 反而下降 |
| `graph_conditioned` | 约 1.08--1.34 次，介于两者之间 |
| `graph_text_aligned` | 约 1.09--1.32 次；没有显示出比其它模式更稳定的来源结构 |

### 7.2 对齐拟合并不能代表有效性

`graph_text_aligned` 的当前-delta graph-to-text 拟合余弦约为：

- Cora：约 `0.878--0.928`；
- Citeseer：约 `0.917--0.938`。

Citeseer 的拟合余弦更高，但最终 F1 更低。这说明当前 ridge 映射只是“当前 delta 中图表示可以线性拟合文本表示”的构造诊断，不等价于文本对 Student 有用，也不能作为 selector 权重的替代目标。

### 7.3 选择分数与 new-class F1 的关系

在 fixed-X 的 48 个 dataset-order-arm-stage 记录上做了探索性相关分析。结果为：

| 构造诊断量 | 与 aggregate new-class F1 的 Pearson 相关 |
|---|---:|
| 综合 selection score | 0.220 |
| 图相似度分数 | 0.278 |
| 文本中心性分数 | 0.636 |
| 来源复用率 | 0.122 |

这个结果只能说明文本中心性比单纯图相似度更值得进一步诊断，不能说明提高文本中心性就会提高 F1。样本数仍然很小，而且同一 dataset/order 的多个阶段并非完全独立。它支持下一步做 source coverage 和文本表示控制，不支持继续围绕当前加权公式调参。

### 7.4 成本

最终阶段每个 cell 的文本构造时间约为 4.5--5.4 秒，四种模式之间差异很小；Student wall time 主要由 memory rows 和 Student steps 决定，文本模式没有造成主要训练成本变化。所有模式的文本最大长度均为 319 字符，满足 320 字符上限。

## 8. 审计结果

本轮 8 个 dataset-order 实验 cell 全部通过审计：

- memory 行数与 5% budget 一致；
- 历史 synthetic prefix 没有被改写；
- 文本来源全部属于当前 delta；
- edge-free 约束成立；
- checkpoint reload 后指标误差为 `0.0 pp`；
- 已完成文本适配的 audit fit count：每个 cell 18，最大指标误差 `0.0 pp`。

因此本轮下降不是统计缓存错误，也不是 checkpoint 读取误差。

## 9. 结果解释

### 9.1 Cora 的正向信号

Cora 上三个 TAG 文本选择器相对 `nearest` 都提高最终 F1 和 Accuracy，说明 CaT 的 MMD `X` 与当前图文表示之间存在可利用的文本选择空间。特别是 `tagsam_style` 的两顺序平均最终 F1 为 `55.738`，比 `nearest` 高 `3.212 pp`。

### 9.2 Citeseer 的负向信号

Citeseer 上 `nearest` 反而最好。文本中心性、图条件 prototype 和 ridge 对齐会改变文本来源的组成，但它们没有改善 Student 的类间竞争，甚至破坏了原有的最近邻语义锚点。两个顺序的差异也说明当前 selector 对类别到达顺序敏感。

### 9.3 当前不能得出的结论

不能据此宣称：

1. `graph_text_aligned` 是稳定主方法；
2. TAGSAM-style 文本处理已经优于 CaT 基线；
3. graph-to-text 拟合本身能够优化下游凝聚；
4. 5 个文本来源或 320 字符预算是理论最优。

本轮能支持的结论是：

> 在固定 CaT `X`、固定 Student 和固定 5% memory 下，TAG-specific 文本选择确实能够在 Cora 产生明显正向信号，但该收益具有数据集和顺序依赖，当前文本 selector 还不能作为跨数据集稳定贡献。

## 10. 下一步实验顺序

### P1：文本来源诊断，不改方法

对已生成 memory 统计：

1. 每个类别的 source coverage；
2. 每个合成节点的来源重复率和文本 cosine dispersion；
3. 来源是否集中在少数高度频繁节点；
4. 每个类别的文本长度、token 数和空文本比例；
5. selector 分数与最终 Student new-class F1 的相关性。

该步骤只回答“选择器到底选了什么”，不根据最终 F1 调整参数。

### P2：固定 CaT X 的文本-only 对照

保留同一份 CaT 合成 `X`，分别替换文本为：

- 真实类中心文本；
- `nearest` 文本；
- TAGSAM-style 文本；
- graph-conditioned 文本；
- graph-text aligned 文本；
- 当前 delta 内随机文本；
- 打乱类别的文本。

如果文本-only 变化在 Citeseer 仍然不稳定，就说明瓶颈不在 selector 权重，而在冻结 Student 对文本 memory 的利用方式或图文表示空间不匹配。

### P3：有限的 soft text assignment

不再只把 5 个来源硬拼接。保持 320 字符预算，先计算候选来源权重：

\[
w_i=\operatorname{softmax}(s_i/\tau),
\qquad
\tilde t=\operatorname{norm}\left(\sum_iw_it_i\right).
\]

先只作为文本 embedding 的诊断，不改变 Student 和 `X`。如果 soft prototype 比硬拼接稳定，再考虑按来源权重分配字符预算。

### P4：暂缓的方向

以下内容暂不与文本 selector 同时加入：Student outer unroll、X/P 联合梯度、拓扑 revision、动态预算、GraphGPS、无界 old-memory refresh。它们会同时改变多个误差来源，无法解释 Citeseer 的下降原因。

## 11. 当前阶段结论

当前最稳妥的骨架仍然是：

```text
CaT class-balanced MMD X condensation
+ current-delta-only text attachment
+ fixed 5-source / 320-character budget
+ append-only synthetic prefix
+ original InfoNCE Student
```

`tagsam_style` 和 `graph_text_aligned` 可以作为后续候选机制，但不能在这一轮结果上晋升为论文主方法。下一轮应优先完成文本来源诊断和固定 X 的 text-only 对照；只有确认文本处理对 Student 的作用机制后，才决定是否引入 soft assignment 或有界 old-text update。

## 12. 代码和结果位置

- 文本机制实现：[external_continual_mechanisms.py](../../code/v2_llm_fusion/external_continual_mechanisms.py)
- 统一实验入口：[run_external_continual_adapters_20260927.py](../../code/v2_llm_fusion/run_external_continual_adapters_20260927.py)
- Cora/Citeseer 三模式结果：`tmp_remote_results/cat_tag_text_small_20261001/`
- aligned 结果：`tmp_remote_results/cat_tag_text_aligned_small_20261001/`
- 构造审计副本：`tmp_remote_results/cat_tag_text_constructions_20261001/`

## 13. 严格 fixed-X 复核

上一轮每个 text arm 都独立调用一次 CaT 构造。虽然理论上 text mode 不进入 `X` 优化目标，但 GPU 稀疏传播的重复计算可能产生极小数值差异。为排除这一点，追加了严格 fixed-X 实验：每个 dataset-order-stage 只运行一次 CaT `X` 优化，四个文本 arm 共享同一个 `X`，之后只重新生成文本并从头训练 Student。

### 13.1 fixed-X 最终 Macro-F1

| 数据集 | 方法 | order19611 | order19612 | 两顺序均值 |
|---|---|---:|---:|---:|
| Cora | `nearest` | 53.616 | 51.907 | 52.761 |
| Cora | `tagsam_style` | 55.603 | 54.698 | 55.150 |
| Cora | `graph_conditioned` | 56.491 | 53.592 | 55.041 |
| Cora | `graph_text_aligned` | **58.078** | 52.771 | **55.425** |
| Citeseer | `nearest` | **57.278** | 51.036 | **54.157** |
| Citeseer | `tagsam_style` | 49.938 | 50.884 | 50.411 |
| Citeseer | `graph_conditioned` | 49.883 | **52.899** | 51.391 |
| Citeseer | `graph_text_aligned` | 51.192 | 52.345 | 51.768 |

fixed-X 最终 Accuracy：

| 数据集 | 方法 | order19611 | order19612 | 两顺序均值 |
|---|---|---:|---:|---:|
| Cora | `nearest` | 58.364 | 61.870 | 60.117 |
| Cora | `tagsam_style` | 61.931 | **62.731** | **62.331** |
| Cora | `graph_conditioned` | 62.669 | 62.116 | 62.392 |
| Cora | `graph_text_aligned` | **63.284** | 60.947 | 62.116 |
| Citeseer | `nearest` | **60.658** | **55.068** | **57.863** |
| Citeseer | `tagsam_style` | 51.672 | 53.814 | 52.743 |
| Citeseer | `graph_conditioned` | 51.933 | **55.747** | 53.840 |
| Citeseer | `graph_text_aligned` | 52.612 | 55.643 | 54.127 |

fixed-X 相对同一 fixed-X `nearest` 的两顺序平均：

| 数据集 | `tagsam_style` | `graph_conditioned` | `graph_text_aligned` |
|---|---:|---:|---:|
| Cora | +2.389 | +2.280 | **+2.664** |
| Citeseer | -3.746 | -2.766 | -2.389 |

### 13.2 fixed-X 审计

四个 dataset-order cell 全部通过 checkpoint reload audit：

- 每个 cell 72 个 Student checkpoint fit；
- `maximum_metric_error_pp=0.0`；
- 每个阶段四个 arm 的 `shared_x_hash` 完全一致；
- 文本来源仍全部属于当前 delta；
- prefix、memory budget 和 edge-free 约束均成立。

因此上一轮的结论可以保留，而且归因更干净：

> Cora 的文本构造正向信号不是由重复 CaT `X` 优化造成的；Citeseer 的下降也不是由 X 数值漂移造成的。

### 13.3 对下一步的影响

fixed-X 复核后不再继续扫描四个 selector 的权重。下一步应转向两类机制诊断：

1. 统计每个类别的文本来源覆盖、来源集中度、文本之间的语义冗余以及 source score 与 new-class F1 的关系；
2. 增加随机文本、类中心文本和打乱类别文本控制，判断 Citeseer 的问题究竟来自 selector，还是来自冻结 Student 对文本 memory 的利用方式。

如果这些控制仍不能在 Citeseer 形成稳定差异，就不应把“更复杂文本选择”作为主贡献，而应考虑在 CaT 的图文 Student 接口中加入一个单独、可解释且固定预算的文本表示适配模块。

## 14. 文本原型控制与独立确认

fixed-X 复核后继续加入三个控制：

- `text_center`：在当前类别内部按文本 embedding 到类别文本中心的相似度选择文本，并用 MMR 保留最多 5 个中心来源；
- `random`：在当前类别内部确定性随机选择文本；
- `wrong_class`：从其它类别选择文本，作为类别语义被破坏的负控制。

### 14.1 控制结果

| 数据集 | nearest | text-center | random-text | wrong-class-text |
|---|---:|---:|---:|---:|
| Cora，两顺序平均 Final F1 | 52.924 | **64.583** | 52.871 | 5.697 |
| Citeseer，两顺序平均 Final F1 | 54.157 | **68.265** | 47.203 | 7.701 |

这组结果说明：

1. random text 不能复现 text-center 的收益，因此提升不是“换一批文本”造成的；
2. wrong-class text 几乎使 Student 失去分类能力，说明文本类别语义确实参与了当前图文 Student 的决策；
3. text-center 主要构造的是每个类别的稳定文本原型，而不是让每个合成节点绑定不同的局部文本。

### 14.2 独立 construction seed 确认

将 construction seed 从 `270927` 改为 `271928`，其它协议不变，只比较 nearest 与 text-center：

| 数据集 | nearest | text-center | 提升 |
|---|---:|---:|---:|
| Cora，两顺序平均 Final F1 | 51.590 | 61.573 | **+9.983** |
| Citeseer，两顺序平均 Final F1 | 57.002 | 69.334 | **+12.332** |

四个确认 cell 全部通过 audit，每个 cell 36 个 checkpoint fit，最大 reload 指标误差为 `0.0 pp`。

### 14.3 第三、第四顺序扩展

在原两顺序之外，使用 `order19613` 和 `order19614`，保持 seed `270927`、6 个 Student seeds 和其它设置不变：

| 数据集 | nearest 四顺序平均 | text-center 四顺序平均 | 提升 |
|---|---:|---:|---:|
| Cora | 53.223 | **64.469** | **+11.246** |
| Citeseer | 55.716 | **68.184** | **+12.468** |

四个扩展 cell 也全部通过 audit。当前 text-center 在四个顺序上没有出现负 cell，说明它比前面的 TAGSAM-style、graph-conditioned 和 graph-text-aligned 更值得作为 CaT-TAG 的主方法候选。

## 15. 暂定主方法：CaT-TAG Class-Central Text Prototype

暂定名称：**CaT-TAG Class-Central Text Prototype**，简称 **CaT-TAG-CCTP**。

完整流程为：

1. 对当前合法 delta 按类别分配新增 memory rows；
2. 使用 CaT class-balanced MMD 优化新增合成节点 `X`；
3. 对每个合成行 `j`，读取其类别 `c_j`；
4. 只在当前 delta 的类别 `c_j` 内计算文本 embedding 到类别文本中心的相似度；
5. 使用固定的 MMR 选择最多 5 个中心文本来源；
6. 在 320 字符预算内拼接来源文本并重新编码；
7. 将新增 block 追加到旧 synthetic prefix，旧 prefix 不改写；
8. 使用原有两层 GCN + text projection + bidirectional InfoNCE Student 重新训练和评估。

文本中心为：

\[
\bar t_c=\operatorname{norm}\left(\frac{1}{|D_c|}
\sum_{i\in D_c}\operatorname{norm}(t_i)\right),
\]

候选文本得分为：

\[
s_i^{center}=\operatorname{cos}(t_i,\bar t_c).
\]

它不需要遍历所有可能的合成簇，也不需要访问历史 raw；每个新合成节点只扫描当前类别的文本 embedding，并用固定 top-5/MRR 选择。因此构造复杂度是当前 delta 文本候选数的线性扫描加上最多 5 次 MMR 选择。

### 15.1 与 nearest 的本质差异

`nearest` 让每个合成 `X` 绑定到图传播空间中最接近的原始节点文本，容易产生：

- 同一类别内不同 synthetic rows 绑定到相互矛盾的文本；
- 合成 `X` 的图位置与文本局部邻居不一致；
- 小类别或长阶段中某些文本来源过度重复或过度偏置。

`CCTP` 放弃逐行的图最近邻绑定，改为同类文本中心原型。这样牺牲了一部分局部图文对应关系，但提高了每个类别文本条件的一致性，正好匹配当前 Student 使用类别 prompt 进行全局图文对比的训练目标。

### 15.2 当前仍需谨慎的点

text-center 的强提升已经通过四个顺序和一个独立 construction seed 确认，但仍需补：

1. text-center 的 `top-1`、`top-3`、`top-5` 和 soft prototype 消融；
2. 大图 base+4 的迁移；
3. 旧类别再次到达时的类别中心更新策略。

当前不建议重新加入 Student outer unroll、拓扑 revision 或动态预算。先把 CCTP 作为唯一新增模块完成公开基线和大图验证，避免再次把贡献分散成多个 selector。

## 16. 四顺序公开方法适配基线矩阵

### 16.1 比较范围和可比性

本轮已经完成 Cora/Citeseer 的四个类别顺序（`order19611`--`order19614`）和六个 paired Student seeds（150--155）。所有方法使用相同的：

- 5% cumulative legal-pool memory；
- current-delta-only 构造访问边界；
- 旧 synthetic prefix 不修改；
- 原有两层 GCN、text projection 和双向 InfoNCE Student；
- `max(50, min(2048, 16 * memory_rows))` 的 Student steps；
- seen-node report-dev 评估，不使用 test label 选择。

这里的 CaT、GECC 不是官方论文数字，而是将公开方法的凝聚机制适配到当前 TAG 持续协议后的参照。`k-center` 和 `medoid` 是相同 Student 下的强 coreset 参照。CCTP 与 `CaT-TAG nearest` 共享 CaT 的 `X` 构造和预算，主要差异是文本附着规则。

### 16.2 四顺序均值

| 数据集 | 方法 | Final Macro-F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 | 跨顺序标准差 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | CaT-TAG nearest | 53.20 | 60.55 | 48.89 | 48.91 | 58.93 | 1.28 |
| Cora | **CaT-TAG CCTP** | **64.31** | **75.58** | **71.84** | **61.56** | **67.97** | **0.26** |
| Cora | GECC-hard-TAG adapter | 47.69 | 53.47 | 44.85 | 44.49 | 51.96 | 2.68 |
| Cora | k-center native append | 50.61 | 55.60 | 49.67 | 46.24 | 56.45 | 4.72 |
| Cora | medoid native append | 49.43 | 56.20 | 42.44 | 45.93 | 54.09 | 3.56 |
| Citeseer | CaT-TAG nearest | 55.72 | 59.56 | 51.81 | 52.91 | 61.33 | 3.24 |
| Citeseer | **CaT-TAG CCTP** | **68.18** | **73.39** | **71.15** | **67.34** | **69.87** | **1.61** |
| Citeseer | GECC-hard-TAG adapter | 51.21 | 56.11 | 43.53 | 48.41 | 56.82 | 2.45 |
| Citeseer | k-center native append | 53.79 | 60.27 | 56.01 | 50.95 | 59.46 | 4.09 |
| Citeseer | medoid native append | 49.52 | 54.18 | 48.34 | 48.72 | 51.12 | 2.88 |

### 16.3 CCTP 相对基线的 Final F1 差值

| 数据集 | 对照方法 | 平均提升 | 正向顺序数 |
|---|---|---:|---:|
| Cora | CaT-TAG nearest | **+11.11** | 4/4 |
| Cora | GECC-hard-TAG adapter | **+16.62** | 4/4 |
| Cora | k-center native append | **+13.70** | 4/4 |
| Cora | medoid native append | **+14.88** | 4/4 |
| Citeseer | CaT-TAG nearest | **+12.47** | 4/4 |
| Citeseer | GECC-hard-TAG adapter | **+16.97** | 4/4 |
| Citeseer | k-center native append | **+14.40** | 4/4 |
| Citeseer | medoid native append | **+18.66** | 4/4 |

### 16.4 结果解释

这组结果改变了前面“只在 Cora 有文本收益”的阶段性判断：在四顺序矩阵中，CCTP 相对 `CaT-TAG nearest` 的 Final F1 在 Cora 和 Citeseer 的四个顺序全部为正，平均提升分别为 `+11.11` 和 `+12.47` 个百分点；相对 k-center 的提升也分别为 `+13.70` 和 `+14.40` 个百分点。CCTP 的跨顺序标准差同时低于 nearest 和 k-center，说明当前提升不是由单个类别顺序驱动。

这仍然不能直接写成“超过官方 CaT/GECC”或普遍 SOTA，原因是公开方法被适配到了统一的 TAG 文本 Student 和持续访问协议，尚未完成各自官方端到端训练目标的复现。可以支持的更准确结论是：

> 在相同持续 TAG 协议、相同 CaT MMD `X` 骨架和相同图文 Student 下，类别中心文本原型显著优于逐行图最近邻文本，并在四个类别顺序上同时优于当前 GECC、k-center 和 medoid 适配参照。

### 16.5 审计和成本

八个 dataset-order cell 的审计均通过，checkpoint reload 最大指标误差为 `0.0 pp`。CCTP 与 nearest 的构造时间几乎相同（四顺序均值约 13 秒），Student 每 seed 的训练时间也基本一致；新增成本主要是当前 delta 内的文本中心计算和固定 top-5 MMR，不是外层 Student 反向传播。因此这轮结果可以作为后续大图验证的稳定起点。

完整逐 cell、Accuracy、成本和差值区间见：

- `results/cctp_baseline_matrix_20261001/summary.md`
- `results/cctp_baseline_matrix_20261001/summary.json`
- `results/cctp_baseline_matrix_20261001/cells.csv`

## 17. 下一步

当前不再继续修改 CCTP 的核心定义。下一轮按以下顺序推进：

1. 固定当前 CCTP，补 `top-1/top-3/top-5` 与 soft prototype 对照，确认收益来自类别中心而不是固定 5 个来源的偶然设置；
2. 在一个小图和一个大图上做 base+4 迁移，使用预注册参数，不根据最终测试 F1 反复调参；
3. 补文本来源覆盖、token、构造时间和 Student 成本账；
4. 若大图仍保持正向，再加入旧类别重新到达和新旧混合到达协议；
5. 最后再冻结论文主表，不把当前适配结果表述为官方公开方法的原始结果。

## 18. CCTP top-k 与 soft prototype 消融

### 18.1 冻结条件

本轮每个 dataset/order/stage 只优化一次 CaT `X`，四个文本臂共享同一 `shared_x_hash`；只改变文本附着，所有其它条件保持不变：四个类别顺序、六个 Student seeds、5% memory、当前 delta-only 来源和相同 checkpoint reload 审计。

- `top-1`、`top-3`、`top-5`：类别中心排序后保留对应数量的文本来源，拼接后重新编码；
- `soft prototype`：保留 top-5 来源用于审计和文本字段，但 Student 使用当前 delta 类内文本 embedding 的温度 `0.07` 加权连续原型；
- 没有根据最终 F1 选择 top-k，也没有修改 CaT `X` 或 Student。

### 18.2 结果

| 数据集 | 方法 | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 | 跨顺序标准差 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | CCTP top-1 | 59.79 | 74.25 | 64.01 | 53.39 | 68.32 | 1.04 |
| Cora | **CCTP top-3** | **66.43** | **77.58** | **72.89** | **64.57** | 68.92 | **0.33** |
| Cora | CCTP top-5 | 64.69 | 75.91 | 71.93 | 62.07 | 68.18 | 0.37 |
| Cora | CCTP soft prototype | 64.23 | 75.43 | 69.80 | 59.99 | **69.88** | 0.24 |
| Citeseer | CCTP top-1 | 64.01 | 71.15 | 67.81 | 62.61 | 66.82 | 1.01 |
| Citeseer | CCTP top-3 | 67.11 | 72.05 | 73.41 | 66.07 | 69.18 | **0.90** |
| Citeseer | **CCTP top-5** | **68.18** | **73.39** | 71.15 | **67.34** | **69.87** | 1.61 |
| Citeseer | CCTP soft prototype | 67.16 | 72.48 | **76.99** | 66.02 | 69.43 | 1.23 |

| 方法 | 全部八个 cell Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|
| CCTP top-1 | 61.90 | 72.70 | 65.91 | 58.00 | 67.57 |
| **CCTP top-3** | **66.77** | **74.82** | 73.15 | **65.32** | 69.05 |
| CCTP top-5 | 66.44 | 74.65 | 71.54 | 64.70 | 69.03 |
| CCTP soft prototype | 65.69 | 73.95 | **73.40** | 63.01 | **69.65** |

### 18.3 解释

1. `top-1` 明显低于 top-3/top-5，说明单个文本来源不足以覆盖类别语义，CCTP 的收益不是简单依赖一个“最中心”节点。
2. `top-3` 在 Cora 上优于 top-5，在 Citeseer 上略低于 top-5；全八个 cell 的平均差异只有 `+0.33 pp`，描述性区间跨过 0。因此暂时不能把 top-3 说成普遍最优，只能把它作为大图的候选版本。
3. soft prototype 的 Old/New trade-off 较好、阶段平均较高，但 Final F1 略低于 top-3/top-5，说明直接平均文本 embedding 并不能完全替代冻结编码器对拼接文本的非线性重编码。
4. 下一轮大图同时保留 top-3 和 top-5，避免在小图上事后挑选一个版本作为最终方法。

完整结果和审计副本：

- `results/cctp_text_controls_20261001/summary.md`
- `results/cctp_text_controls_20261001/summary.json`
- `results/cctp_text_controls_20261001/cells.csv`
- `results/cctp_text_controls_20261001/cctp_text_controls_bars.png`

## 19. 三张大图 base+4 验证

### 19.1 协议

本轮在 arXiv、BookChild 和 SportsFit 上使用相同的 `base+4` 类别增量协议。所有方法共享：

- 5% cumulative memory；
- current-delta-only 构造，历史 raw 读取次数为 0；
- 相同 edge-aware 两层 GCN、text projection 和双向 InfoNCE Student；
- Student seeds `100--105`；
- `cat_steps=3000`；
- `max(50,min(2048,16*memory_rows))` Student steps；
- 不使用 report/test labels 选择构造或参数。

`append-edge` 和 `X+P` 是不带 CCTP 文本中心的封闭控制；CaT/CCTP 仍是统一 TAG 协议下的机制适配，不是官方论文端到端结果。

### 19.2 各大图结果

| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 | 构造秒 | Student 秒/seed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arXiv | append-edge | 17.06 | 25.34 | 22.43 | 24.60 | 18.82 | 0.2 | 24.4 |
| arXiv | X+P | 17.68 | 29.51 | 22.07 | 25.58 | 19.39 | 0.2 | 24.3 |
| arXiv | **CCTP top-3** | **23.54** | 47.99 | 31.71 | 33.22 | **24.65** | 136.1 | 24.6 |
| arXiv | CCTP top-5 | 23.35 | **50.59** | **33.57** | **33.72** | 23.97 | 115.6 | 24.9 |
| BookChild | append-edge | 10.75 | 11.10 | 14.12 | 14.97 | 12.47 | 0.1 | 22.7 |
| BookChild | X+P | **12.02** | 13.09 | **14.47** | 16.74 | **13.88** | 0.1 | 22.7 |
| BookChild | CCTP top-3 | 3.30 | 13.80 | 4.37 | 4.24 | 3.55 | 72.7 | 23.1 |
| BookChild | CCTP top-5 | 4.30 | **33.54** | 4.15 | 5.89 | 4.28 | 78.7 | 22.8 |
| SportsFit | append-edge | 21.43 | 27.55 | 26.06 | 27.82 | 25.17 | 0.2 | 24.7 |
| SportsFit | X+P | 28.76 | 38.97 | 29.18 | 37.24 | 32.99 | 0.2 | 24.6 |
| SportsFit | **CCTP top-3** | **46.54** | **47.71** | **50.76** | **54.41** | **48.58** | 364.1 | 24.6 |
| SportsFit | CCTP top-5 | 20.50 | 39.55 | 29.62 | 30.31 | 22.42 | 386.0 | 24.7 |

### 19.3 三图 pooled 结果

| 方法 | Final F1 | Accuracy | 阶段平均 F1 | Old F1 | New F1 | 平均构造秒 |
|---|---:|---:|---:|---:|---:|---:|
| append-edge | 16.41 | 21.33 | 20.87 | 22.46 | 18.82 | 0.19 |
| X+P | 19.48 | 27.19 | 21.90 | 26.52 | 22.09 | 0.19 |
| **CCTP top-3** | **24.46** | 36.50 | **28.95** | **30.62** | **25.60** | 190.98 |
| CCTP top-5 | 16.05 | **41.23** | 22.45 | 23.30 | 16.89 | 193.44 |

相对控制的 Final F1 差值为：

| 方法 | 对照 | 平均差值 | 正向大图数 |
|---|---|---:|---:|
| CCTP top-3 | append-edge | **+8.04** | 2/3 |
| CCTP top-3 | X+P | **+4.97** | 2/3 |
| CCTP top-5 | append-edge | -0.36 | 1/3 |
| CCTP top-5 | X+P | -3.43 | 1/3 |

三图差值的描述性 95% 区间很宽（例如 top-3 相对 append-edge 为 `[-32.54,+48.63]`），因此不能将三个大图视为足以建立普遍结论的独立样本。结果应被解释为跨规模的正向信号，而不是稳定 SOTA 证据。

### 19.4 结果含义

1. **CCTP top-3 有真实但有条件的迁移信号。** arXiv 提升 `+6.48 pp`，SportsFit 提升 `+25.11 pp`，相对 append-edge 的三图平均提升 `+8.04 pp`；相对 X+P 的平均提升 `+4.97 pp`。
2. **BookChild 暴露了当前方法的失效边界。** top-3 相对 append-edge 下降 `-7.46 pp`，相对 X+P 下降 `-8.72 pp`；这说明类别文本中心并不总能代表大图上的局部传播语义。
3. **top-5 不能作为默认版本。** arXiv 上 top-5 的 Accuracy 和阶段平均 F1 较高，但 BookChild 和 SportsFit 的 Final F1 均下降；仅增加文本来源数没有稳定收益。
4. **CCTP 的主要成本是构造而不是 Student。** 三图中 append-edge/X+P 的构造约 `0.2 s`，CCTP 约 `73--386 s`，而 Student 仍约 `23--25 s/seed`。后续论文必须把这部分成本单独报告，不能只报告下游 F1。

因此，当前最准确的结论是：

> CCTP 在小图四顺序和部分大图 base+4 协议上能够显著改善统一图文 Student 的持续凝聚效果，但收益依赖数据集和 top-k，尚不能声称对所有 TAG 数据集稳定优于基线。

### 19.5 下一步

大图结果回来后暂不继续扫描大量 `k` 或权重。下一步按以下顺序推进：

1. 对 BookChild/SportsFit/arXiv 统计每阶段每类的合法文本数、类别中心离散度、来源覆盖、空/短文本比例和 top-3/top-5 重复率；
2. 检查失败大图是否是类别中心过度压缩了局部文本，还是文本候选数量不足；
3. 只设计一个预注册的“大图条件化中心”诊断版本，例如在类别中心中保留有限的局部簇原型，不同时改 `X`、拓扑和 Student；
4. 重新在三图上做一次确认，并继续保留 append-edge 与 X+P 控制；
5. 在该诊断完成前，不把 CCTP top-3 最终冻结为普适主方法，也不把 top-5 加入主线。

大图汇总文件：

- `results/cctp_large_baselines_20261001/summary.md`
- `results/cctp_large_baselines_20261001/summary.json`
- `results/cctp_large_baselines_20261001/rows.csv`
- `results/cctp_large_baselines_20261001/text_diagnostics.md`
- `results/cctp_large_baselines_20261001/text_diagnostics.json`
- `tmp_remote_results/cctp_large_baselines_20261001/sportsfit_results.json`

