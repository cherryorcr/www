# TGC Shared-P + Bounded-X 真冻结拓扑验收报告

日期：2026-09-26  
状态：协议修复后重跑完成；shared-PX 不晋升为主方法

## 1. 重要协议更正

第一轮输出目录：

`results/tgc_shared_p_bounded_x_frozen_topology_independent_20260926/`

不能作为“冻结拓扑优化”的有效证据。审计发现，旧脚本的 `--freeze-topology` 只在最终写出 memory 时把 `edge_index` 替换成 C0，Student-compatible 内层 rollout 仍使用 `P^T A_delta P` 的投影拓扑。该批结果保留为协议审计失败的预运行，不用于论文结论。

已修复脚本，使冻结 C0 拓扑同时作用于：

1. baseline old-loss rollout；
2. 可微 Student rollout；
3. probe rollout；
4. 最终 materialization。

修正版真结果位于：

- 构造：`results/tgc_shared_p_bounded_x_frozen_topology_true_20260926/`
- Student 评估：`results/tgc_shared_p_bounded_x_frozen_topology_true_eval_20260926/`

## 2. 方法定义

本轮不是三个完全独立的优化器，而是同一个 P+X 构造过程的三个输出臂。

### C0

当前可靠基线。新阶段使用现有 C0 的内容、文本和拓扑构造规则，旧 memory 前缀不变。

### shared-P

使用一个类别合法的软分配矩阵 P 对当前 delta 进行材料化：

- P 有类别 mask 和 0.02 uniform floor；
- P 用于新文本的聚合和重编码；
- P 的内容输出是按 P 对当前 delta 的 X 加权平均；
- 真冻结拓扑实验中，最终边直接使用 C0 的 `edge_index`。

它不是一个单独运行的“只优化 P”实验。P 是在同一 P+X 外层优化中得到的最佳 P，之后使用该 P 的材料化结果作为 shared-P 输出。

### shared-PX

在相同 P 和相同文本的基础上，替换新块 X：

- X 的初值是 C0 当前阶段的新 X；
- P 和 X 在同一外层目标中联合优化；
- Student inner seeds 为 `8200, 8201`；
- outer steps 为 8，inner steps 为 8；
- 使用新 delta transfer loss、旧 memory 保持项、P 锚定项、X 重构项；
- 每个新合成节点的 X 位移被限制在初始范数的 5% 内；
- 真冻结拓扑中，所有 rollout 使用 C0 的邻接矩阵；
- 最终输出与 shared-P 使用同一文本和 C0 拓扑，只把新块 X 替换为优化后的 X。

因此，`PX-P` 主要测量在同一 P、同一文本、同一 C0 拓扑下，bounded X 更新能否恢复 P 材料化损失。

## 3. 统一评估协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19605`、`order19606`、`order19608` |
| 阶段 | stage 1、stage 2 |
| cell 数 | 12 |
| Student | 两层 GCN + trainable text projection + frozen MiniLM |
| 训练目标 | 双向图文 InfoNCE，不使用分类标签训练 |
| batch | 16 |
| Student report seeds | `150..155` |
| Student steps | v3 规则；每个 cell 使用 784 steps |
| 评价指标 | report-dev 上 all-prompt cumulative Macro-F1、old F1、new F1 |
| 候选选择 | 不使用 report/test F1 选择 |
| 构造约束 | 不读取历史完整 raw graph；不使用 report/test 标签 |

## 4. 真冻结拓扑逐 cell 结果

数值是六个 Student seeds 的均值，单位为 F1 百分点。

| 数据集 | 顺序/阶段 | C0 | shared-P | shared-PX | P-C0 | PX-C0 | PX-P |
|---|---:|---:|---:|---:|---:|---:|---:|
| Citeseer | 19605/s1 | 53.379 | 55.135 | 55.246 | +1.756 | +1.867 | +0.111 |
| Citeseer | 19605/s2 | 55.330 | 55.790 | 54.765 | +0.460 | -0.565 | -1.026 |
| Citeseer | 19606/s1 | 65.691 | 67.424 | 66.571 | +1.733 | +0.880 | -0.853 |
| Citeseer | 19606/s2 | 51.452 | 49.174 | 47.752 | -2.278 | -3.700 | -1.422 |
| Citeseer | 19608/s1 | 54.961 | 57.471 | 57.937 | +2.511 | +2.976 | +0.466 |
| Citeseer | 19608/s2 | 46.950 | 47.625 | 45.781 | +0.675 | -1.170 | -1.844 |
| Cora | 19605/s1 | 50.374 | 47.879 | 53.106 | -2.495 | +2.733 | +5.227 |
| Cora | 19605/s2 | 50.563 | 49.942 | 47.995 | -0.621 | -2.568 | -1.947 |
| Cora | 19606/s1 | 57.252 | 53.718 | 54.879 | -3.534 | -2.373 | +1.161 |
| Cora | 19606/s2 | 44.111 | 47.232 | 47.625 | +3.121 | +3.514 | +0.393 |
| Cora | 19608/s1 | 52.973 | 47.890 | 50.068 | -5.083 | -2.905 | +2.178 |
| Cora | 19608/s2 | 52.655 | 51.832 | 52.123 | -0.823 | -0.532 | +0.291 |

## 5. 汇总结果

### 全部 12 个 cell

| 方法 | 平均累计 F1 | 相对 C0 | old F1 相对 C0 | new F1 相对 C0 | 正 cell | 最差 cell |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 52.974 | 0.000 | 0.000 | 0.000 | - | - |
| shared-P | 52.593 | -0.382 | -0.026 | -1.073 | 6/12 | -5.083 |
| shared-PX | 52.821 | -0.154 | +1.754 | -2.295 | 5/12 | -3.700 |

shared-PX 相对 shared-P 平均恢复 `+0.228 pp`。这说明 X 更新有局部作用，但它没有转化为相对 C0 的稳定收益。

### 按数据集

| 数据集 | C0 | shared-P | shared-PX | PX-C0 | PX old-C0 | PX new-C0 | PX 正 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 51.321 | 49.749 | 50.966 | -0.355 | +3.478 | -4.594 | 2/6 |
| Citeseer | 54.627 | 55.437 | 54.675 | +0.048 | +0.031 | +0.005 | 3/6 |

Cora 上 X 相对 P 平均有 `+1.217 pp` 的恢复，但 new-class F1 仍平均下降 `4.594 pp`；Citeseer 上最终几乎回到 C0，平均只高 `0.048 pp`。

### 构造后 Student 评估成本

shared-PX/C0 的 report Student wall-time 平均比约为 `0.997x`，没有出现评估阶段成本爆炸。这个数字不包含构造阶段的 2 个 inner seeds、8 个 outer steps 和 8 个 inner steps，论文中仍需单独报告构造成本。

## 6. Memory 级审计

对 12 个 cell 的 `c0_memory.pt`、`shared_p_memory.pt`、`shared_px_memory.pt` 逐项重载检查：

- 24 个对照 arm 的完整 `edge_index` 与 C0 bitwise 一致；
- 24 个 arm 的旧 X 前缀一致；
- 24 个 arm 的旧文本一致；
- 24 个 arm 的旧文本 embedding 一致；
- 构造 audit 的 `historical_raw_reads=0`；
- 构造阶段 `report_labels_used=false`；
- 12/12 cell 的 `optimization_topology_mode=frozen_c0`。

旧输出中的 `same_p_text_topology=true` 字段曾经把“P 用于文本”和“最终拓扑冻结为 C0”混写。该字段已在脚本中修正为 `same_p_text_assignment`、`same_p_text_topology` 和 `topology_matches_c0` 三个明确字段；真结果的 memory 级检查不依赖这个旧字段。

## 7. 结果解释

### 7.1 X 更新存在局部正信号

shared-PX 相对 shared-P 在 Cora 上平均提高 `+1.217 pp`，在全部 cell 上平均提高 `+0.228 pp`。这说明在固定 C0 拓扑时，bounded X 的 Student-facing 更新不是完全无效的。

### 7.2 但主要问题不是“只要改 X 就能解决”

shared-PX 相对 C0 的累计 F1 仍为 `-0.154 pp`，正 cell 只有 `5/12`，最差为 `-3.700 pp`。更关键的是 new-class F1 平均为 `-2.295 pp`，而 old F1 反而为 `+1.754 pp`。

这表明当前目标更容易保护旧类，而不能稳定改善新类的类间可分性。Cora/order19605/s1 是最典型的例子：PX 相对 C0 提高 `+2.733 pp`，但 old F1 提高约 `+11.013 pp`，new F1 下降约 `5.547 pp`。累计 F1 的提升来自旧类，不代表新类塑性改善。

### 7.3 P 和 X 的联合机制仍然有顺序依赖

shared-PX 在 Citeseer 的 19605/s1、19606/s1、19608/s1 等 cell 有正收益，但在 Citeseer/19606/s2 下降 `-3.700 pp`；Cora 也呈现相反方向的顺序差异。这说明当前 transfer loss、P 的类内分配和最终 zero-shot Macro-F1 之间仍然存在目标错位，不能把少数正 cell 包装成稳定方法增益。

## 8. 晋升判定

预注册门槛：

- 平均提升至少 `+1 pp`；
- 至少 `8/12` 正 cell；
- 最差 cell 不低于约 `-2 pp`；
- new-class F1 平均不下降；
- 构造成本受控。

shared-PX 实际结果：

- 平均累计 F1：`-0.154 pp`，失败；
- 正 cell：`5/12`，失败；
- 最差 cell：`-3.700 pp`，失败；
- new-class F1：`-2.295 pp`，失败；
- report 阶段成本约 `0.997x`，仅成本项通过。

结论：**shared-PX 不进入当前主方法，C0 继续作为可靠默认主线。**

## 9. 对后续设计的影响

本轮可以保留的研究信号是：

> 在固定 C0 拓扑时，P+bounded-X 的联合更新相对 P 材料化存在局部恢复；但当前 Student response 目标主要改善旧类保持，不能稳定提升新类塑性。

因此后续不再继续扫描 selector、alpha、X 半径或 outer steps。若继续推进，应只引入一个显式的新类塑性约束，例如新类与旧类的 prompt margin / class-conditional contrastive gain，并与 C0 anchor 和严格回退机制绑定。该目标必须在独立 selection seeds 下验证，不能只用 teacher transfer loss 选择。

在新的塑性约束没有通过至少两个数据集、独立顺序和最差 cell 门槛之前，不启动更复杂的文本 attention、动态预算、拓扑重连或旧记忆 revision。

## 10. 可复现实验产物

- 真构造：[tgc_shared_p_bounded_x_frozen_topology_true_20260926](../../results/tgc_shared_p_bounded_x_frozen_topology_true_20260926/)
- 真评估：[tgc_shared_p_bounded_x_frozen_topology_true_eval_20260926](../../results/tgc_shared_p_bounded_x_frozen_topology_true_eval_20260926/)
- 修正版脚本：[run_tgc_shared_p_bounded_x_20260926.py](../../code/v2_llm_fusion/run_tgc_shared_p_bounded_x_20260926.py)
- 评估脚本：[run_tgc_role_aware_detailed_eval_20260926.py](../../code/v2_llm_fusion/run_tgc_role_aware_detailed_eval_20260926.py)
- 构造文本证据包：`tgc_shared_p_bounded_x_frozen_topology_true_20260926_text_evidence.tar.gz`
- 评估文本证据包：`tgc_shared_p_bounded_x_frozen_topology_true_eval_20260926_text_evidence.tar.gz`
