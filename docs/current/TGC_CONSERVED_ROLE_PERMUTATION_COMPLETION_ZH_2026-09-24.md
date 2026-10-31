# TGC 守恒 payload 置换实验报告

日期：2026-09-25  
状态：完成开发集 stage-1/stage-2 与独立顺序确认；修正旧 role-match 实现后，完成完整 payload 守恒审计

## 1. 为什么要重做

上一轮 `role_match` 将 changed proposal 直接写入新的同类 C0 slots，但没有把被占用 slot 的完整内容迁移到剩余 slot。这样会改变 source coverage、重复或丢失内容，并造成 mass 元数据不守恒。即使当前 Student 不把 mass 作为 InfoNCE 权重，这仍然改变了 X/text 的覆盖，不能归因于“槽位角色”。

因此本轮不沿用旧 `role_match` 数字，改成两步：

1. 只构造一次局部 shared two-prototype split/merge block；
2. 在每个类别内部对完整 synthetic row payload 做双射置换。

每个 payload 同时包含 `X`、`text_emb`、文本、mass、depth 和 stage；边、旧 prefix、总 rows 和 Student 协议保持不变。

## 2. 候选方法

设结构构造后同一类别的 changed payload 为 `r`，候选 C0 slots 为 `s`。

### Identity

双原型 proposal 保持在原始结构构造产生的三个 slots 中。它测量双原型内容构造本身的效果。

### Semantic permutation

使用 joint graph-text center 的代价，将 changed payload 分配到同类所有 slots：

\[
C_{sem}(r,s)=\|\operatorname{norm}(\mu_r)-\operatorname{norm}(c_s)\|_2^2.
\]

分配后，所有被挤出的 payload 通过确定性剩余匹配迁移，形成完整双射；不会重复或删除 source piece。

### Role permutation

在 semantic cost 上加入 degree、direct old-neighbor degree、new-new degree、old-neighbor ratio、old-endpoint load 和 mass 项：

\[
C(r,s)=C_{sem}(r,s)+0.50C_{role}(r,s)+0.10C_{mass}(r,s).
\]

权重在实验前固定，不使用 report/test F1。

### Random permutation

在相同的同类可行槽位集合内使用固定随机种子置换，作为“只移动 payload 也可能改变传播结果”的控制。

## 3. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集/顺序 | Cora `order19601/19602/19603`；Citeseer `order19601/19602/19603` |
| 阶段 | stage 1 |
| memory | 5% growing memory；总 rows 与 C0 相同 |
| Student | 原始两层 GCN + trainable text projection + 双向 InfoNCE |
| Student seeds | `150..155`；每个 arm 成对使用 |
| 文本 | top-5，显式 320 字符 cap，冻结 MiniLM 重编码 |
| 拓扑 | sealed C0 `edge_index` bitwise copy |
| 历史 raw graph | 不访问完整历史 raw graph |
| 选择 | 无 report/test F1 selector |

每个候选自动检查：行数、source coverage、每类 mass、完整 X/text payload multiset、旧 prefix、旧文本 prefix 和 C0 edge_index。

## 4. 结果

差值均为相对于同 cell 的重构 `c0_320` 的 paired cumulative Macro-F1，单位为百分点；`min` 是六个 Student seed 中最差值。

| 数据集/顺序 | Identity | Semantic permutation | Role permutation | Random permutation |
|---|---:|---:|---:|---:|
| Cora/order19601 | `+1.675` (min `-1.725`) | `+1.646` (min `-1.561`) | `+1.675` (min `-1.725`) | **`+4.716`** (min `+2.224`) |
| Cora/order19602 | `-0.885` (min `-3.269`) | `-0.771` (min `-8.668`) | `-0.885` (min `-3.269`) | `+1.708` (min `-2.357`) |
| Cora/order19603 | `-2.467` (min `-5.312`) | **`+7.184`** (min `+1.055`) | `-2.467` (min `-5.312`) | `-1.063` (min `-10.014`) |
| Citeseer/order19601 | `-1.352` (min `-4.121`) | **`+1.502`** (min `-1.066`) | `-1.352` (min `-4.121`) | `-0.586` (min `-5.228`) |
| Citeseer/order19602 | `-0.606` (min `-4.152`) | `-1.040` (min `-6.681`) | **`+4.177`** (min `-0.997`) | `+2.493` (min `-0.300`) |
| Citeseer/order19603 | `-1.205` (min `-4.074`) | `-1.524` (min `-6.380`) | `-1.205` (min `-4.074`) | `-1.913` (min `-5.181`) |
| 六 cell 平均 | `-0.806` | **`+1.166`** | `-0.009` | `+0.892` |
| 正 cell 数 | 1/6 | 3/6 | 2/6 | 3/6 |

相对于 identity 双原型，semantic permutation 的六 cell 平均提升为 `+1.973 pp`，其中 Cora/order19603 为 `+9.651 pp`，Citeseer/order19601 为 `+2.854 pp`；但 Citeseer/order19602、Citeseer/order19603 仍为负，不能称为稳定泛化。

### 4.1 Stage-2 结果

Stage-2 使用完全相同的构造、Student、seed 和审计协议，只把增量阶段从 stage-1 切换到 stage-2。差值仍为相对于同 cell `c0_320` 的 paired cumulative Macro-F1，单位为百分点。

| 数据集/顺序 | Identity | Semantic permutation | Role permutation | Random permutation |
|---|---:|---:|---:|---:|
| Cora/order19601 | `+3.182` | `-1.155` | `+3.182` | `-1.511` |
| Cora/order19602 | `-0.304` | `+3.548` | `-0.304` | `-4.267` |
| Cora/order19603 | `-2.052` | `+4.035` | `-2.052` | `+5.630` |
| Citeseer/order19601 | `+2.532` | `+2.929` | `+2.532` | `+0.010` |
| Citeseer/order19602 | `+0.544` | `+0.534` | `+0.544` | `+1.920` |
| Citeseer/order19603 | `+0.778` | `+0.069` | `+0.778` | `-1.862` |
| 六 cell 平均 | `+0.780` | **`+1.663`** | `+0.780` | `-0.014` |
| 正 cell 数 | 4/6 | **5/6** | 4/6 | 3/6 |

Stage-2 的 semantic permutation 在 Cora/order19601 为负，但其余五个 cell 均为正；因此它比 stage-1 更稳定，但仍不是无条件稳定提升。

### 4.2 Stage-1 + Stage-2 合并结果

合并 12 个 cell 后：

| 方法 | 平均相对 C0 | 正 cell | 最差 cell | 最好 cell |
|---|---:|---:|---:|---:|
| Identity 双原型 | `-0.013 pp` | 5/12 | `-2.467 pp` | `+3.182 pp` |
| **Semantic permutation** | **`+1.414 pp`** | **8/12** | `-1.524 pp` | `+7.184 pp` |
| Role permutation | `+0.385 pp` | 6/12 | `-2.467 pp` | `+4.177 pp` |
| Random permutation | `+0.439 pp` | 6/12 | `-4.267 pp` | `+5.630 pp` |

按数据集汇总，semantic permutation 在 Cora 六个 cell 平均 `+2.416 pp`，4/6 cell 为正；在 Citeseer 六个 cell 平均 `+0.412 pp`，4/6 cell 为正。按阶段汇总，stage-1 为 `+1.166 pp`，stage-2 为 `+1.663 pp`。

### 4.3 Old/New F1 分解

相对于同 cell `c0_320`，semantic permutation 的 12-cell 平均：

| 指标 | 平均差值 | 正 cell 数 | 最差 cell |
|---|---:|---:|---:|
| Cumulative Macro-F1 | `+1.414 pp` | 8/12 | `-1.524 pp` |
| Old-class F1 | `+0.674 pp` | 5/12 | `-4.513 pp` |
| New-class F1 | **`+2.621 pp`** | 6/12 | `-7.204 pp` |

这说明整体收益主要来自部分顺序中的新类塑性改善，但不是每个 cell 都改善新类；Cora 的 new-class 平均增益为 `+4.491 pp`，Citeseer 仅为 `+0.751 pp`。因此 semantic assignment 可能改善新 proposal 与传播槽位的匹配，却还没有解决跨数据集的稳定性问题。

### 4.4 Stage-2 守恒审计

六个 stage-2 cell 中的 18 个置换 arm 全部通过：

- `payload_multiset_preserved = true`；
- `source_coverage_preserved = true`；
- `same_c0_topology = true`；
- `same_old_prefix = true`。

本轮没有使用 report/test F1 选择候选，也没有访问历史完整 raw graph。因而 stage-2 的正向差值可以归因于守恒 payload 的槽位重分配，而不是行数、拓扑、历史覆盖量或事后选择改变。

### 4.5 独立类别顺序确认

为避免只在 `order19601/02/03` 上解释局部增益，使用预先存在但未参与本轮开发的 `order19605/06/08`，在 Cora 和 Citeseer 上重新执行 stage-1/2。配置、Student seeds、文本 cap、C0 拓扑和 semantic cost 均不改变。

| 数据集/独立顺序 | stage-1 | stage-2 | 两阶段说明 |
|---|---:|---:|---|
| Cora/order19605 | `-0.210 pp` | `+0.592 pp` | 接近持平 |
| Cora/order19606 | `+5.411 pp` | `-1.264 pp` | 阶段依赖明显 |
| Cora/order19608 | `+6.334 pp` | `+5.684 pp` | 稳定正向 |
| Citeseer/order19605 | `+1.720 pp` | `-3.631 pp` | stage-2 明显退化 |
| Citeseer/order19606 | `-3.105 pp` | `+2.081 pp` | 顺序依赖明显 |
| Citeseer/order19608 | `+4.782 pp` | `+4.144 pp` | 稳定正向 |
| 12 cell 平均 |  |  | **`+1.878 pp`** |

独立集汇总：

| 指标 | Semantic permutation | Random permutation |
|---|---:|---:|
| 平均相对 C0 | **`+1.878 pp`** | `-0.339 pp` |
| 正 cell 数 | **8/12** | 7/12 |
| 最差 cell | `-3.630 pp` | `-6.481 pp` |
| Old-class F1 平均差值 | `+2.625 pp` | 不作为主判据 |
| New-class F1 平均差值 | `+0.550 pp` | 不作为主判据 |

按数据集，Cora 平均为 `+2.757 pp`，Citeseer 平均为 `+0.998 pp`；但 Citeseer 的 new-class F1 平均为 `-0.698 pp`。全部 36 个独立置换 arm 均通过 payload、source coverage、C0 topology 和 old-prefix 审计。

## 5. 结果解释

### 5.1 已确认

1. 旧 role-match 的 source/mass 不守恒是实质性实现缺陷；旧报告中的 role 正收益不能继续作为证据。
2. 局部双原型内容构造本身不是稳定增益：identity 六格平均为 `-0.806 pp`。
3. 将完整 proposal payload 重新放置到同类 C0 slots 可能产生明显收益；semantic permutation 在 stage-1 六格平均 `+1.166 pp`、stage-2 六格平均 `+1.663 pp`，12 格合并为 `+1.414 pp`，说明“内容与合成槽位的匹配”比单独修改 X/text 更有潜力。
4. 当前 role 代价没有显示稳定价值：六格平均约为 `-0.009 pp`。在 Cora 三格和 Citeseer/order19601、order19603 中，它直接退化为 identity；只有 Citeseer/order19602 发生了有益置换。
5. random permutation 在部分 cell 也能提升，但其 12 格平均只有 `+0.439 pp`，且最差为 `-4.267 pp`；semantic permutation 的平均收益和最差 cell 均更好，但随机重排仍说明收益不能全部归因于当前 semantic cost 已经精确刻画了 student response。
6. semantic permutation 的收益与 new-class F1 有关，但 old/new trade-off 仍依赖数据集：Cora 更明显，Citeseer 较弱。

### 5.2 不能下的结论

本轮可以把 semantic permutation 从“只在 stage-1 的局部候选”提升为“通过 stage-2、但未通过独立顺序晋升门槛的研究候选”，仍不宜替换 C0 为唯一主方法，原因是：

- 独立顺序虽然平均为正，但最差 cell 为 `-3.630 pp`，未达到预设的约 `-2 pp` 门槛；
- Citeseer 独立集的 new-class F1 平均为 `-0.698 pp`，新类塑性仍不稳定；
- random control 在独立集也有 7/12 个正 cell，说明不能把所有局部收益都归于 semantic cost；
- 尚未完成 link prediction 和新增构造成本的完整分解。

## 6. 下一步

下一轮不再优化 role 权重，也不再运行昂贵 selector。优先级为：

1. 冻结本轮 semantic cost 和代码，不再按已有 cell 调权重；当前候选不晋升为默认主方法。
2. 将 semantic permutation 作为机制分析和可选增强报告，明确列出独立集的负 cell 与 new-class 退化。
3. 补做 link prediction、构造时间和 memory 变化量；确认槽位重排是否对结构任务更敏感，避免只用节点分类判断其价值。
4. 若后续要继续算法化，优先研究不依赖最终 F1 的守恒 assignment 约束或 task-aware objective；不恢复昂贵 selector，也不把单一新类顺序的增益直接写成主方法。

## 7. 代码与产物

- Runner：[run_tgc_two_prototype_role_permutation.py](../../code/v2_llm_fusion/run_tgc_two_prototype_role_permutation.py)
- 单元测试：[test_tgc_two_prototype_role_permutation.py](../../code/v2_llm_fusion/test_tgc_two_prototype_role_permutation.py)
- 远程 wrapper：[tgc_two_prototype_role_permutation_remote.py](../../tools/tgc_two_prototype_role_permutation_remote.py)
- stage-1 结果目录：[tgc_two_prototype_role_permutation_20260924](../../results/tgc_two_prototype_role_permutation_20260924/)
- stage-2 结果目录：[tgc_two_prototype_role_permutation_20260925](../../results/tgc_two_prototype_role_permutation_20260925/)
- 独立顺序结果目录：[tgc_two_prototype_role_permutation_independent_20260925](../../results/tgc_two_prototype_role_permutation_independent_20260925/)

旧的 `TGC_ROLE_AWARE_MATCHING_CONFIRMATION_ZH_2026-09-24.md`、`TGC_TWO_PROTOTYPE_ROLE_DIAGNOSTIC_ZH_2026-09-24.md` 和 `TGC_ROLE_MATCH_SYNTHETIC_MARGIN_GATE_ZH_2026-09-24.md` 中关于 role-match 正收益的数字仅作历史记录；本报告中的守恒实验优先。
