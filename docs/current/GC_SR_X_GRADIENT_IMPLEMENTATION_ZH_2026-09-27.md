# GC-SR 第一轮实现与验收记录

日期：2026-09-27

## 1. 这轮实现回答什么问题

Semantic-Recursive 的 memory 现在可以作为稳定初始化，但此前 `X` 基本没有接受 canonical Student 的直接训练反馈。GC-SR（Gradient-Consistent Semantic-Recursive）第一轮只验证一个最小问题：

> 在不改变文本、预算、拓扑和下游 Student 的条件下，让合成 `X` 直接对齐合法混合支持上的 Student 梯度，是否能够形成可审计的构造更新。

这轮不是最终方法，也没有声称已经提升 F1。

## 2. 输入和访问约束

- `base`：上一版 C0/Semantic-Recursive 的当前阶段合成 memory。
- `support`：上一阶段 synthetic parent 与当前 raw delta 的合法组合。
- 允许修改：选定的 `x` 行。
- 固定不变：`text_emb`、原始文本、`mass`、`depth`、`stage`、`edge_index` 和总 rows。
- 不使用：report-dev/test 标签、最终 F1 选择、历史完整 raw graph。

`run_gc_sr_x_only.py` 接受已经准备好的 `base` 和 `support` 文件，因此 raw 数据切片和访问审计在上游完成，优化器不会偷偷回读历史数据。

## 3. 目标函数

在同一个随机初始化的 canonical Student 上分别计算：

\[
g_{ref}=\nabla_\theta L_{InfoNCE}(support),\qquad
g_{syn}=\nabla_\theta L_{InfoNCE}(candidate).
\]

梯度项为：

\[
L_{grad}=1-\cos(g_{syn},g_{ref}).
\]

可选的 output 项比较整张合成图的 graph embedding 均值：

\[
L_{output}=\|\operatorname{mean}(z_{syn})-
\operatorname{mean}(z_{ref})\|_2^2.
\]

旧行 anchor 使用相对位移：

\[
L_{anchor}=\operatorname{mean}_{i\in old}
\left\|\frac{\Delta x_i}{\|x_i\|_2}\right\|_2^2.
\]

旧行和新行使用不同 trust-region 半径；默认旧行半径为 `0.05`，新行半径为 `0.20`。

## 4. 四个第一轮 arm

| arm | gradient | output | old anchor | 用途 |
|---|---:|---:|---:|---|
| `output` | 0 | 1 | 0 | 判断全局 Student 输出匹配是否有信号 |
| `gradient` | 1 | 0 | 0 | 判断直接梯度方向是否有信号 |
| `joint` | 1 | 1 | 0 | 判断输出与梯度联合是否互补 |
| `gradient_anchor` | 1 | 0 | 0.05 | 判断旧行保护是否改善梯度更新 |

每个 arm 独立保存 `memory.pt`、`audit.json` 和 `protocol.json`，不覆盖已有实验。

## 5. 当前实现验收

`test_gc_sr_gradient.py` 已通过：

- 诊断可以计算有限的 gradient cosine 和 output loss；
- `X` 更新后 memory hash 改变；
- 只有指定 `X` 行发生变化；
- 旧/新 trust-region 投影生效；
- 文本、拓扑和其它 payload 字段保持不变；
- output-only arm 也能建立 Student 反向路径。

远程回归结果为 `16 passed`。这是数值与不变量验收，不是数据集性能结果。

## 6.1 Cora 真实维度 stage-1 smoke

使用已有 Cora `canonical` C0 stage-0/stage-1 memory 和当前 stage raw delta，
不做完整 Student F1 矩阵，只运行 1 个 train seed、1 个 validation seed、1 个
outer step，编辑 2 个新行和 1 个旧行。该 smoke 的目的只是检查真实输入维度、真实
GCN 拓扑和二阶梯度路径。

| arm | validation gradient cosine | validation output loss | candidate InfoNCE loss | construction seconds | max relative drift |
|---|---:|---:|---:|---:|---:|
| `gradient` | 0.017540 | 0.00003914 | 1.462685 | 0.267 | 0.144722 |
| `output` | 0.017494 | 0.00003873 | 1.463186 | 0.139 | 0.131124 |
| `joint` | 0.017494 | 0.00003892 | 1.463186 | 0.250 | 0.144722 |

共同的 base/support 初始诊断为：gradient cosine `0.017540`，base loss
`1.462685`，legal mixed-support loss `1.342274`。因此当前主要观察是：

1. C0 的 Student 梯度与当前 delta mixed support 的梯度方向几乎不一致；
2. X-only 目标可以产生非零更新，并且不会改动文本、预算和拓扑；
3. 一个 outer step 尚未把 candidate loss 拉到 support loss，不能据此宣称性能提升；
4. `output`、`gradient`、`joint` 的差异需要在更长 horizon 和正式 Student seeds 上比较，不能根据这个 smoke 选主方法。

产物目录：

`results/gc_sr_stage_smoke_20260927/cora_canonical_s1/`

本记录中的 Cora 数字是 construction/gradient smoke，不是 report-dev Macro-F1。

## 7. 下一步实验顺序

1. 在一个已有 Cora/Citeseer stage 上准备合法 `support`，先跑单步或两步 smoke。
2. 比较四个 arm 的 gradient cosine、output loss、位移和构造成本。
3. 只有 smoke 没有数值异常后，才运行小图正式 Student seeds。
4. 如果 X-only 对新类塑性无效，再加入 P2 的 soft routing；不要同时改变文本和拓扑。
5. 最后单独加入稀疏 `new-old` 拓扑，不把 topology 收益和 X 收益混在第一轮。

## 8. 不能提前宣称的内容

- 能反传不等于能提升最终 Macro-F1；
- gradient cosine 不等于 zero-shot 分类收益；
- output matching 仍是全局摘要，不是严格的原始节点对应；
- 目前尚未完成 Cora/Citeseer 正式 GC-SR 矩阵；
- 当前主方法仍是 Semantic-Recursive，GC-SR 只有通过独立顺序、Student seed、old/new F1 和成本门槛后才可能晋升。

## 9. 全量 X 更新的正式小图验证（2026-09-27）

为排除“只更新少数敏感行”造成的选择偏差，第二轮固定使用
`joint_new` 目标，并允许当前 memory 中全部 `X` 行参与更新：旧行使用相对
trust-region `0.05`，新行使用 `0.20`；`outer_steps=4`；构造使用一个训练
seed 和一个验证 seed，随后用 Student seeds `150--155` 做正式评估。文本、预算、
拓扑、Student 和 report-dev 规则均保持不变。

| 数据集 | 顺序 | Semantic-Recursive F1 | GC-SR 全量 X F1 | 差值 | Accuracy 差值 | 新类 F1 差值 | 旧类 F1 差值 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | order19611 | 52.077 | 50.966 | -1.111 | -1.734 | -0.816 | -1.223 |
| Cora | order19612 | 35.861 | 33.965 | -1.897 | -2.295 | -3.057 | -0.650 |
| Citeseer | order19611 | 62.815 | 62.022 | -0.792 | -0.741 | +0.314 | -1.367 |
| Citeseer | order19612 | 59.067 | 56.135 | -2.932 | -3.236 | -3.915 | -2.398 |

四个 cell 的构造审计均满足：`text_changed=false`、`budget_changed=false`、
`topology_changed=false`、`historical_raw_reads=0`。因此这轮下降不能归因于
访问约束或 payload 被意外修改。

### 9.1 结果解释

1. 全量 `X` 更新没有解决行选择问题，反而把局部的 gradient/output mismatch
   扩散到全部 memory，导致整体表征漂移。
2. `joint_new` 在 Citeseer/order19611 上确实带来新类 F1 `+0.314 pp`，但
   旧类 F1 `-1.367 pp`，最终 F1 仍下降 `0.792 pp`。这说明“新类塑性信号存在”
   不等于“整体 Student 任务收益成立”。
3. 全局参数梯度 cosine 是跨所有行、所有类别和随机初始化的代理目标；它没有
   明确区分当前新类的塑性梯度与历史类保持梯度，容易出现方向冲突。
4. 旧行 anchor 没有解决问题：之前的局部实验已经显示 anchor 会压制新类，
   本轮取消 anchor 后，全量更新仍然不稳定，说明根因不只是旧行移动过大。

### 9.2 决策

GC-SR 第一轮不晋升为主方法，也暂不继续叠加 soft routing、跨块拓扑或文本更新。
下一轮若继续研究 Student-compatible construction，应改为更局部、可解释的目标：

- 只对新 block 做 class-conditional feature/InfoNCE matching；
- 将 old/new 梯度分开计算，并显式加入有界的 retention constraint；
- 用少量支持统计或短 Student trajectory 估计增量响应，避免用全局参数梯度
  作为唯一 proxy；
- 先在一个顺序上完成目标函数消融，再用独立顺序确认；
- 若局部目标仍不能在两个数据集上稳定改善，则保留 Semantic-Recursive 为主方法，
  将 GC-SR 作为“Student 反馈优化的失败边界”写入负结果章节。

正式结果文件：

`results/gc_sr_fullx_eval_20260927/`

构造审计目录（远程同名目录）：

`results/gc_sr_fullx_smoke_20260927/`

## 10. 新类专向梯度目标验证（2026-09-27）

为区分“全局 old/new 梯度冲突”和“梯度代理本身无效”，第三轮只允许新 block
的 `X` 更新，旧行完全冻结；目标只保留新 block 的 gradient alignment：

\[
L=L_{grad}^{new},\qquad
\Delta X_{old}=0,
\]

同时保持相同的文本、预算、拓扑、Student 和 trust-region 设置。四个 cell 均使用
Student seeds `150--155` 评估。

| 数据集 | 顺序 | Semantic-Recursive F1 | 新类专向 F1 | 差值 | Accuracy 差值 | 新类 F1 差值 | 旧类 F1 差值 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | order19611 | 52.077 | 51.096 | -0.981 | -0.867 | -0.499 | -1.063 |
| Cora | order19612 | 35.861 | 34.845 | -1.016 | -1.297 | +1.302 | -1.904 |
| Citeseer | order19611 | 62.815 | 60.496 | -2.318 | -2.074 | -1.361 | -2.834 |
| Citeseer | order19612 | 59.067 | 57.891 | -1.177 | -0.566 | -3.800 | +0.237 |

这组结果说明：

1. 只更新新 block 并不能保证新类塑性；新类 F1 只有一个 cell 上升，且该 cell
   的旧类损失超过新类收益。
2. 因此问题不只是旧行被更新过多，也不只是全局 old/new 梯度互相冲突。当前
   `L_grad^{new}` 与 report-dev 分类 F1 之间缺少稳定的任务对应关系。
3. 固定文本与拓扑时，`X` 的移动会改变 message passing 后的图表示，但梯度目标
   没有约束其与新类语义原型、类间间隔和历史决策边界的关系。

### 10.1 GC-SR 阶段性结论

GC-SR 已完成三层验证：

- 少量行更新：存在局部正信号，但依赖行选择；
- 全量 `X` + `joint_new`：四格最终 F1 全部下降；
- 仅新 block 梯度：四格最终 F1 仍全部下降。

因此当前不再继续加入 soft routing、跨块拓扑、文本 attention 或更复杂 selector。
若未来重启 Student-compatible construction，必须先重定义监督代理，至少引入
class-conditional feature target 或短 Student trajectory 的增量收益估计，而不是
继续调节同一个全局 gradient cosine。当前论文主方法保持 Semantic-Recursive，
GC-SR 作为“直接 Student 梯度反馈的可行性与失效边界”记录。

新类专向结果文件：

`results/gc_sr_newonly_eval_20260927/`

构造审计目录：

`results/gc_sr_newonly_smoke_20260927/`
