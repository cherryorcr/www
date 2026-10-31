# CaT 多模态最小可行性实验报告

日期：2026-10-02

## 1. 实验目的

本轮只验证一个问题：能否把连续文本 embedding 纳入当前 CaT 风格的类别凝聚目标，并在不改变 Student、拓扑、memory budget 和访问约束的前提下改善持续 TAG 凝聚效果。

本轮没有加入 Teacher、Student unroll、拓扑 revision、旧 memory refresh 或动态预算。这样可以把结果归因到文本/X 凝聚目标本身。

## 2. 四个实验版本

### 2.1 `cat_x`

- 使用 CaT 风格的 class-wise random-feature MMD 生成当前阶段的合成特征 `X`；
- 使用现有 CCTP top-3 文本选择和冻结 MiniLM 重编码得到文本 embedding；
- `X` 和文本 embedding 在构造后均不再更新；
- 这是本轮主要 anchor。

### 2.2 `cat_t_only`

- `X` 固定为 `cat_x` 的同一份初始化；
- 仅把当前阶段的连续文本 embedding 设为可优化参数；
- 用原始当前阶段文本 embedding 的 class-wise MMD 作为目标；
- 加入文本 anchor，限制优化后的 embedding 偏离 CCTP 初始 embedding；
- 文本字符串只保留为 provenance，Student 实际读取优化后的连续 embedding。

### 2.3 `cat_xt`

- `X` 和文本 embedding 同时优化；
- `X` 使用 class-wise feature MMD，文本使用 class-wise text MMD；
- 两者分别带 anchor 约束；
- 当前版本没有显式的 X-text cross-modal loss，也没有 Student response loss，因此它是“联合优化两个 payload”的最小版本，而不是完整的 Student-compatible 联合凝聚。

### 2.4 `random_text`

- 保持 `cat_x` 的 `X` 和 memory 结构不变；
- 将文本替换为随机文本控制；
- 用于验证文本 payload 是否具有实际作用。

## 3. 目标函数

设当前原始阶段的特征、文本和类别标签为 `(x, t, y)`，合成节点为 `(X, T, \hat y)`。对每一步重新采样 random-feature projection，class-wise MMD 记为：

\[
L_x = \operatorname{MMD}_{class}(x, X),
\qquad
L_t = \operatorname{MMD}_{class}(t, T).
\]

文本 only 版本优化：

\[
L_{text-only}=L_t+\lambda_t L_{anchor}^{t},
\]

联合版本优化：

\[
L_{joint}=L_x+\lambda_t L_t
          +\lambda_t^{a}L_{anchor}^{t}
          +\lambda_x^{a}L_{anchor}^{x}.
\]

其中：

- `L_anchor^t` 是优化文本与 CCTP 初始文本的 cosine anchor；
- `L_anchor^x` 是优化 `X` 与 CaT 初始 `X` 的平方距离；
- 默认优化步数为 300；
- `lr_text=5e-3`，`lr_x=1e-3`；
- `lambda_text=1.0`，两个 anchor 系数均为 `0.05`。

这个目标只保证合成 payload 接近当前 raw pool 的类别分布，不保证经过 Student 图传播和 InfoNCE 训练后一定得到更好的分类边界。

## 4. 统一协议

| 项目 | 设置 |
| --- | --- |
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19611`、`order19612`、`order19613`、`order19614` |
| 阶段数 | 3 |
| memory | 5% cumulative fixed-rate，历史 synthetic prefix append-only |
| 原始数据访问 | 每个阶段只使用当前 delta；不读取历史 raw graph |
| 旧 memory | 不修改 |
| 拓扑 | edge-free synthetic materialization，四个 arm 完全一致 |
| Student | 原有两层 GCN、冻结 MiniLM、双向图文 InfoNCE |
| Student seeds | 150--155，共 6 个 |
| 评价 | all-prompt report-dev Macro-F1、Accuracy、Old F1、New F1 |
| 构造选择 | 不使用最终 F1 选择，不做 post-hoc 调参 |

多顺序运行时曾发现一个评估状态污染问题：`stage_classes` 是共享 data object 的可变状态，原始脚本在多 order 评估前没有恢复当前 order。该问题已修正为在 `evaluate_cell` 开始时重新调用 `BASE.set_stream(data, order)`。第一轮受污染的目录保留作诊断，不纳入本报告；下表来自修正后的 `v2` 重跑。

## 5. 完整结果

下表是四个类别顺序的最终阶段结果，数值是 6 个 Student seed 的均值；括号为四个 order 之间的标准差。

### 5.1 Cora

| 方法 | Final Macro-F1 | Final Accuracy | Stage-average Macro-F1 | Old F1 | New F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cat_x` | 66.345 ± 0.396 | 77.552 ± 0.301 | 72.985 | 64.559 | 68.725 |
| `cat_t_only` | 65.833 ± 1.016 | 77.183 ± 0.613 | 73.260 | 63.333 | 69.166 |
| `cat_xt` | 65.898 ± 0.813 | 77.153 ± 0.488 | 73.096 | 63.768 | 68.737 |
| `random_text` | 54.038 ± 4.998 | 61.900 ± 4.257 | 51.473 | 51.296 | 57.694 |

相对 `cat_x`：

| 方法 | Final 差值 | Stage-average 差值 | Final 正收益 order 数 |
| --- | ---: | ---: | ---: |
| `cat_t_only` | -0.512 | +0.276 | 0/4 |
| `cat_xt` | -0.447 | +0.112 | 1/4 |
| `random_text` | -12.306 | -21.512 | 0/4 |

### 5.2 Citeseer

| 方法 | Final Macro-F1 | Final Accuracy | Stage-average Macro-F1 | Old F1 | New F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cat_x` | 67.105 ± 0.900 | 72.048 ± 0.713 | 73.405 | 66.066 | 69.184 |
| `cat_t_only` | 66.862 ± 1.005 | 71.904 ± 0.599 | 73.465 | 65.898 | 68.792 |
| `cat_xt` | 66.927 ± 0.293 | 71.957 ± 0.464 | 73.341 | 65.889 | 69.003 |
| `random_text` | 49.091 ± 1.039 | 53.096 ± 0.837 | 49.292 | 46.540 | 54.194 |

相对 `cat_x`：

| 方法 | Final 差值 | Stage-average 差值 | Final 正收益 order 数 |
| --- | ---: | ---: | ---: |
| `cat_t_only` | -0.243 | +0.060 | 2/4 |
| `cat_xt` | -0.178 | -0.064 | 2/4 |
| `random_text` | -18.014 | -24.113 | 0/4 |

## 6. 凝聚目标与下游结果的关系

### 6.1 连续文本确实被优化

四个 order、两个数据集的 `cat_t_only` 和 `cat_xt` 都有非零文本更新。典型文本更新范数约为 `0.36--0.73`；联合版本的 `X` 更新范数约为 `0.42--0.63`。因此结果不是因为 optimizer 没有更新参数。

### 6.2 text-MMD 在构造过程中下降

典型区间如下：

| 数据集 | 初始 text-MMD | 最终 text-MMD | 结论 |
| --- | ---: | ---: | --- |
| Cora | 约 0.0029--0.0048 | 约 0.0024--0.0040 | 稳定下降 |
| Citeseer | 约 0.0033--0.0038 | 约 0.0025--0.0035 | 稳定下降 |

### 6.3 但 MMD 下降没有转化成稳定 F1 提升

这说明当前目标优化的是“合成文本 embedding 的类别分布”，而最终任务优化的是“经过 Student 图传播、文本投影和双向 InfoNCE 后的响应”。两者之间缺少可学习的桥接项：

1. MMD 不知道 Student 的文本投影矩阵如何变化；
2. MMD 不知道合成节点与其他 synthetic 节点之间的 batch/对比关系；
3. 当前 edge-free materialization 不提供真实邻域消息；
4. `cat_xt` 的 X-MMD 和 text-MMD 是分开的，并不保证 X 与 T 的跨模态关系一致；
5. 小 memory 下，少量文本改变可能只影响旧/新类竞争，而不改善最终 macro-F1。

## 7. 成本与访问约束

- `cat_x` 每阶段构造约 4--5.5 秒；
- `cat_t_only` 每阶段约 4.7--6.4 秒；
- `cat_xt` 每阶段约 5.2--6.9 秒；
- 文本优化增加了约 0.6--1.4 秒/阶段；
- 不读取历史 raw graph；
- 不修改旧 synthetic prefix；
- memory 行数、append-only 顺序和 edge-free 结构保持一致。

## 8. 本轮结论

### 已证明

1. 连续文本 embedding 可以稳定地接入 CaT-style class-wise MMD 目标；
2. 文本参数确实发生了更新，且 text-MMD 可下降；
3. 随机文本控制会在两个数据集上显著降低 F1 和 Accuracy，说明文本 payload 对当前 Student 确实重要；
4. 文本优化的构造成本仍然可接受。

### 尚未证明

1. 文本 MMD 优化能稳定改善最终 Student；
2. 同时优化 X 和文本优于只优化 X 或只优化文本；
3. 该最小版本能改善新类塑性；
4. 该目标在更大图或更长阶段链上具有泛化收益。

### 晋升判断

当前 `cat_t_only` 和 `cat_xt` 不应作为新的主方法。它们应冻结为：

> **CaT multimodal minimal feasibility / negative control**

也就是说，当前结果支持“文本是重要 payload，简单把文本 MMD 加入 CaT 有技术可行性”，但不支持“简单文本 MMD 就是有效的持续 TAG 凝聚算法”。

## 9. 下一步建议

不继续堆叠 attention、Teacher、动态预算和拓扑模块。下一步只引入一个能够连接凝聚目标和 Student 响应的最小桥接项，候选优先级为：

1. 在 `cat_xt` 上加入固定 Student-compatible cross-modal loss，使 `X` 和 `T` 的更新围绕同一个 Student 图文相似度，而不是两个独立 MMD；
2. 保持 X、文本和 topology 不变，只比较 `MMD-only` 与 `MMD + cross-modal`；
3. 若 cross-modal 目标仍不能在四 order 上带来稳定收益，则停止继续扩展文本模块，转而检查当前 edge-free Student/任务协议是否是主要瓶颈；
4. 不使用最终 F1 做动态选择，不恢复大规模 response selector。

相关实现：

- `code/v2_llm_fusion/cat_multimodal_minimal.py`
- `code/v2_llm_fusion/run_cat_multimodal_minimal_20261002.py`
- `code/v2_llm_fusion/test_cat_multimodal_minimal.py`
