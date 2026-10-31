# Text-aware CaT 持续 TAG 凝聚实验报告

日期：2026-10-02  
状态：实现完成，四顺序完整矩阵完成；当前版本不晋升为主方法。

## 1. 实验目的

本轮验证一个与 Student-response 不同的方向：

> 不在 CaT 完成后用下游 Student loss 修改 `X`，而是在 CaT 优化 `X` 的过程中，让当前阶段的文本 payload 通过可微软分配参与凝聚目标。

本轮不引入 Student unroll、拓扑 revision、旧 memory refresh、动态预算或 GraphGPS。Student 只用于最终评价。

## 2. 方法定义

### 2.1 CaT 对照 `cat_x`

当前阶段读取合法 raw delta，按类别比例分配新增 rows，用 CaT 风格的 class-wise random-feature MMD/CGM 优化合成特征 `X`。优化完成后使用统一的 TAG 文本适配：

1. 在当前 delta 的同类别文本中计算文本中心；
2. 选择 top-3 文本来源；
3. 在固定约 320 字符预算内截断并拼接；
4. 通过冻结 MiniLM 重新编码；
5. 将新 block append 到旧 synthetic prefix。

本轮 `cat_x` 是同一协议下的主要 anchor，不是原始论文的未经适配的端到端 CaT 复现。

### 2.2 `cat_textaware_x`

该方法只优化当前新增 `X`，但在每个 CaT 迭代中根据当前 `X` 计算软文本 payload。

设当前 delta 的图传播特征、文本 embedding 和标签为：

\[
G=\{g_i\},\quad E=\{e_i\},\quad y_i.
\]

首先只使用当前 delta 拟合 ridge 图文映射：

\[
R=\arg\min_R\|GR-E\|_2^2+\eta\|R\|_2^2.
\]

对于合成节点 `j`，由当前合成特征 `x_j` 得到文本查询：

\[
q_j=\operatorname{Normalize}([\operatorname{Normalize}(x_j),1]R).
\]

只在同类别 raw 文本候选中计算软分配：

\[
\pi_{ji}=\operatorname{softmax}_i
\left(\frac{\operatorname{sim}(q_j,e_i)}{\tau}\right),
\qquad y_i=\hat y_j.
\]

软文本 payload 为：

\[
\hat e_j=\sum_i\pi_{ji}e_i.
\]

CaT 图目标保留为：

\[
L_X=\operatorname{MMD}_{class}(G,X).
\]

新增文本分布和图文一致性目标：

\[
L_T=\operatorname{MMD}_{class}(E,\hat E),
\]

\[
L_{cross}=\frac{1}{m}\sum_j
\left(1-\cos(q_j,\hat e_j)\right).
\]

最终优化目标为：

\[
L=
\frac{L_X}{s_X}
+\lambda_T\frac{L_T}{s_T}
+\lambda_C\frac{L_{cross}}{s_C}
+\lambda_X L_{anchor}^{X}
+\lambda_T^aL_{anchor}^{T}.
\]

其中 `s_X`、`s_T`、`s_C` 是由初始 CaT 状态计算的固定尺度，用于避免 cosine loss 的数值量级压过 CaT MMD。文本 anchor 约束的是初始软 payload，而不是直接把不兼容的 soft embedding 拉向 CCTP 拼接 embedding。

这一版本的梯度路径是：

\[
X\rightarrow q\rightarrow \pi\rightarrow \hat E
\rightarrow L_T,L_{cross}\rightarrow \nabla_XL.
\]

因此没有使用 Student loss。

### 2.3 文本材料化控制

实现了三种材料化方式：

| 版本 | CaT 中是否使用软文本目标 | 最终 Student 文本 |
|---|---:|---|
| `cat_textaware_x` | 是 | 保留原 CCTP 文本 embedding |
| `cat_textaware_soft` | 是 | 直接使用连续软文本 embedding |
| `cat_textaware_hard` | 是 | 按最终软分配选 top-k 文本并重新编码 |

完整矩阵只运行 `cat_textaware_x`，因为 smoke 已显示直接把 soft embedding 或 hard top-k 文本替换进 Student 会产生明显材料化断层；这两个版本保留为诊断控制。

## 3. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19611`--`order19614` |
| 阶段数 | 3 |
| memory | 5% cumulative fixed-rate |
| 访问范围 | 当前 delta 加已有 synthetic parent |
| 历史 raw graph | 不读取 |
| 旧 synthetic prefix | 不修改 |
| 拓扑 | edge-free |
| Student | 原有两层 GCN、冻结 MiniLM、双向图文 InfoNCE |
| Student seeds | 150--155，共 6 个 |
| CaT steps | 3000 |
| Text-aware steps | 300 |
| 文本候选 | 当前 delta 同类别文本 |
| 文本来源数 | 3 |
| 报告指标 | Macro-F1、Accuracy |
| 选择规则 | 不依据最终测试 F1 选择参数或版本 |

## 4. 完整结果

所有差值均为同一 order、stage 和 Student seed 配对后的结果。

### 4.1 Cora

| 方法 | 平均 Macro-F1 | 平均 Accuracy | 相对 `cat_x` 的 F1 | 正收益 cell | 最差 cell | 构造成本 |
|---|---:|---:|---:|---:|---:|---:|
| `cat_x` | 73.037 | 81.543 | 0.000 | - | - | 1.00x |
| `cat_textaware_x` | 72.866 | 81.483 | **-0.170 pp** | 5/12 | -0.943 pp | 2.27x |

四个顺序的 F1 差值分别为：

| 顺序 | F1 差值 |
|---|---:|
| `order19611` | -0.125 pp |
| `order19612` | +0.011 pp |
| `order19613` | -0.243 pp |
| `order19614` | -0.324 pp |

### 4.2 Citeseer

| 方法 | 平均 Macro-F1 | 平均 Accuracy | 相对 `cat_x` 的 F1 | 正收益 cell | 最差 cell | 构造成本 |
|---|---:|---:|---:|---:|---:|---:|
| `cat_x` | 73.405 | 77.920 | 0.000 | - | - | 1.00x |
| `cat_textaware_x` | 73.379 | 77.811 | **-0.026 pp** | 6/12 | -2.302 pp | 2.63x |

四个顺序的 F1 差值分别为：

| 顺序 | F1 差值 |
|---|---:|
| `order19611` | -0.194 pp |
| `order19612` | +0.541 pp |
| `order19613` | +0.827 pp |
| `order19614` | -1.278 pp |

## 5. 结果解释

### 5.1 文本确实参与了 CaT 的 `X` 梯度

审计结果显示：

- `X` 发生了非零更新；
- text-MMD 和 graph-text consistency 都参与 loss；
- 软分配概率、熵和 top-1 mass 均被保存；
- `historical_raw_reads=0`；
- `old_memory_updated=False`；
- `topology_updated=False`。

所以本轮不是“文本模块没有运行”，而是文本目标虽然成功进入了优化过程，却没有转化为稳定的下游收益。

### 5.2 主要问题是图文映射不是任务充分统计量

当前文本项依赖当前 delta 上拟合的 ridge 映射 `G -> E`。这个映射只能说明局部图特征和文本 embedding 的相关性，不能保证：

1. 映射后的文本查询等价于最终 Student 的文本投影；
2. 文本 MMD 下降对应更好的图文对比边界；
3. 合成节点之间的对比关系得到保持；
4. 当前阶段的局部图文关系可以泛化到旧 synthetic memory。

因此它比单纯的文本后处理更合理，但仍然是一个代理目标。

### 5.3 硬文本材料化存在明显 gap

soft payload 是原始文本 embedding 的凸组合，而 hard 版本是 top-k 字符串拼接后重新送入 MiniLM。这两个对象并不等价：

\[
\operatorname{MiniLM}(\operatorname{concat}(t_1,t_2,t_3))
\neq
\sum_i\pi_i\operatorname{MiniLM}(t_i).
\]

smoke 中 soft/hard payload 与 base CCTP embedding 的相似度明显偏低，因此直接使用 soft 或 hard 结果会造成较大性能下降。这个现象说明连续文本目标和最终可审计文本之间还缺少一致性约束。

### 5.4 成本不够划算

完整矩阵中，文本感知版本需要在每个 CaT block 后额外运行 300 次文本感知迭代，构造成本约为：

- Cora：`2.27x`；
- Citeseer：`2.63x`。

在平均 Macro-F1 没有提升的情况下，当前成本不支持把该版本作为主方法。

## 6. 本轮结论

### 已证明

1. 文本处理可以直接接入 CaT 的 `X` 优化循环；
2. 不依赖 Student response 也可以构造合法的图文联合凝聚目标；
3. 文本梯度可以通过软分配传回 `X`；
4. 访问约束、append-only memory 和旧 prefix 保持规则没有被破坏。

### 尚未证明

1. 当前 ridge graph-to-text map 是足够好的图文桥接；
2. text-MMD 或 cross-modal consistency 能稳定改善下游任务；
3. 连续软 embedding 可以可靠地材料化为真实文本；
4. 该方法可以在两个数据集和多个顺序上超过 CaT。

因此，`cat_textaware_x` 当前应被定位为：

> **CaT 内部图文联合目标的可行性和负结果诊断版本。**

它不能晋升为论文主方法，也不能宣称已经优于 CaT。

## 7. 后续建议

不继续扫描温度、MMD 权重或大量 selector。下一步应针对已暴露的两个具体问题做最小修正：

### 7.1 先解决文本材料化一致性

候选方式按优先级为：

1. 预先构造与 CCTP 相同的候选文本组 embedding，而不是直接平均单条文本 embedding；
2. 在 CaT 优化中对“候选组 embedding”做 soft assignment；
3. 以最终 MiniLM 重编码结果作为文本目标，而不是单条文本 embedding 的凸组合；
4. 先做 soft-vs-hard gap 实验，再决定是否继续联合优化 `X`。

### 7.2 再改图文桥接目标

不直接使用自由 ridge map 作为主要语义约束，而是比较：

- 类别内图文本相似度矩阵；
- 类别间相对距离结构；
- CCTP baseline 的文本中心保持项；
- 同一 synthetic row 的 graph/text assignment consistency。

第一版只允许新增一个跨模态结构项，仍然不加入 Student、拓扑和旧 memory revision。

### 7.3 晋升条件

后续版本只有在以下条件满足后才进入主方法候选：

- 两个数据集平均不低于 `cat_x`；
- 至少 3/4 个类别顺序阶段平均为正；
- 最差 cell 不超过约 `-1 pp`；
- 构造成本不超过 `1.5x`；
- hard text materialization 与 soft objective 的差距受控。

## 8. 代码和结果

- 实现：[cat_multimodal_minimal.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/cat_multimodal_minimal.py)
- 实验脚本：[run_cat_multimodal_minimal_20261002.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/run_cat_multimodal_minimal_20261002.py)
- 单元测试：[test_cat_multimodal_minimal.py](D:/Aprojects/continue_tag_condensation/code/v2_llm_fusion/test_cat_multimodal_minimal.py)
- Cora 远程结果：`/home/pc/research/TAG_condensation/results/cat_textaware_allorders_cora_20261002`
- Citeseer 远程结果：`/home/pc/research/TAG_condensation/results/cat_textaware_allorders_citeseer_20261002`

