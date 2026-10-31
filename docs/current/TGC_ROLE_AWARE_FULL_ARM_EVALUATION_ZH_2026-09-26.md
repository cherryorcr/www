# TGC Role-aware 全 arm 统一 Student 评估报告

日期：2026-09-26  
状态：上一轮 12-cell 评估已完成；`semantic_perm` 的 8-cell fresh 独立确认也已完成，平均收益复现但尾部 cell 尚未完全通过晋升门槛。详见 `TGC_ROLE_AWARE_FRESH_CONFIRMATION_ZH_2026-09-26.md`。

## 1. 实验目的

本轮的目标是把此前分散的 `C0`、identity、semantic permutation、role-aware 及其控制变体放到同一套 Student 协议中，回答三个问题：

1. 之前观察到的提升是否来自固定的 payload assignment，而不是 Student seed、训练步数或指标口径；
2. 哪一种 assignment 在 Cora/Citeseer、多顺序、多阶段上最稳定；
3. 是否可以在“整体累计 F1 提升、new-class 只允许小幅波动”的设定下晋升一个候选方法。

本轮不加入 selector、dynamic budget、shared-PX、文本 attention、GraphGPS 或旧 memory refresh，避免把多个未验证模块混在一起。

## 2. 冻结协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19605`、`order19606`、`order19608` |
| 评估阶段 | stage 1、stage 2，共 12 个 dataset/order/stage cells |
| Student | 原始两层 GCN + trainable text projection |
| 训练目标 | 双向图文 InfoNCE，不使用节点分类监督训练 |
| batch | 16 |
| steps | `protocol_v3.student_steps(rows)`，每个 arm 与同 cell 的 C0 相同 |
| Student seeds | 150--155，共 6 个 |
| 指标 | report-dev、all-prompt、按已见类别计算 cumulative Macro-F1、old-class F1、new-class F1 |
| 构造时标签 | 不使用 report/test 标签；不按最终 F1 选择 |
| 历史 raw graph | 读取次数为 0 |

每个 cell 的候选 memory 在冻结后重新加载，所有 arm 使用完全相同的 Student 初始化 seed、batch 和步数。评估阶段记录的是 Student 训练后的 all-prompt F1，不是 seen-prompt 的选择分数。

## 3. 方法臂定义

所有候选都从同一个 two-prototype structural proposal 出发，保持 C0 的 binary topology 与旧 prefix；区别只在于新 payload 如何分配到候选合成行。

| arm | 机制 |
|---|---|
| `c0` | 冻结的共同基线 |
| `identity` | 不做语义/角色 permutation 的结构 proposal |
| `semantic_perm` | 在同类候选内，用图文联合代价进行 payload assignment |
| `role_perm` | `semantic_perm` 加 degree、cross-degree、new-new degree、old-neighbor ratio、old-endpoint load 等结构角色代价 |
| `anchor_role_perm` | role assignment；当归一化增益不足时回退到类内 identity anchor |
| `message_role_perm` | 在 role assignment 中加入 C0 一跳 message compatibility proxy |
| `plasticity_role_perm` | 在 role assignment 中加入当前类 prompt margin proxy |
| `adaptive_role_perm` | 在限定范围 `[0.25, 0.75]` 内自适应 graph/text agreement 权重 |
| `random_perm` | 类内随机 permutation 控制，用来检验提升是否只是重排效应 |

其中 `semantic_perm` 和 `role_perm` 是最重要的两个候选；其余 arm 主要用于区分“语义分配本身”与“加入更多启发式结构项”是否继续有益。

## 4. 12-cell 汇总结果

表中差值均相对同一 cell 的 C0，单位为百分点；`positive` 表示累计 F1 严格高于 C0 的 cell 数。

| 方法 | 累计 F1 平均提升 | old F1 平均变化 | new F1 平均变化 | positive cells | 最差 cell |
|---|---:|---:|---:|---:|---:|
| C0 | 0.000 | 0.000 | 0.000 | 0/12 | 0.000 |
| identity | +0.630 | +1.475 | -0.254 | 6/12 | -5.507 |
| semantic_perm | **+1.690** | **+2.983** | -0.027 | **9/12** | **-1.904** |
| role_perm | +1.468 | +2.212 | **+0.558** | 7/12 | -2.042 |
| anchor_role_perm | +1.027 | +2.286 | -0.530 | 7/12 | -1.904 |
| message_role_perm | +0.710 | +1.747 | -0.417 | 8/12 | -3.479 |
| plasticity_role_perm | +1.029 | +2.323 | -0.568 | 8/12 | -3.825 |
| adaptive_role_perm | +1.029 | +2.323 | -0.568 | 8/12 | -3.825 |
| random_perm | -0.319 | +1.452 | -2.340 | 5/12 | -7.088 |

### 4.1 按数据集汇总

| 数据集 | 方法 | 累计 F1 平均提升 | old F1 变化 | new F1 变化 | positive |
|---|---|---:|---:|---:|---:|
| Cora | semantic_perm | +2.570 | +4.291 | +0.328 | 6/6 |
| Cora | role_perm | +2.474 | +3.451 | +1.141 | 4/6 |
| Cora | anchor_role_perm | +1.961 | +3.846 | -0.154 | 4/6 |
| Citeseer | semantic_perm | +0.810 | +1.675 | -0.381 | 3/6 |
| Citeseer | role_perm | +0.461 | +0.973 | -0.025 | 3/6 |
| Citeseer | anchor_role_perm | +0.094 | +0.725 | -0.907 | 3/6 |

结果显示：semantic permutation 的收益不是只来自 Cora，但 Citeseer 上明显较弱；role-aware 结构项并没有在跨数据集上稳定超过 semantic assignment，反而增加了顺序依赖和尾部风险。

## 5. 成本与约束审计

### 5.1 Student 重载成本

所有 arm 在同一 cell 使用相同的 rows、steps、batch 和 seed 集合。12-cell 汇总中，各 arm 的 Student 重载 wall time 平均约 17.4 秒，差异来自测量噪声和 cell rows，不是方法额外训练造成的；本轮没有 selector 的多候选重复训练。

### 5.2 构造成本

Role-aware assignment 的独立构造审计覆盖 Cora/Citeseer 的 12 个 cell：

- construction time 平均 `0.0588 s`；
- 最小 `0.0359 s`，最大 `0.0873 s`；
- role incremental 部分平均 `0.0253 s`；
- 所有 cell 保持 `same_c0_topology=true`、`same_old_prefix=true`。

这说明 assignment 计算本身不是主要成本瓶颈；真正的成本仍然来自后续 Student 训练。这里不能据此声称总训练更快，只能说明新增构造开销很小。

### 5.3 历史与泄漏约束

- 不读取历史完整 raw graph；
- 不使用 report/test 标签进行构造或选择；
- 旧合成 prefix 与 C0 一致；
- 旧历史边保持 C0 结构；
- `random_perm` 显著变差，说明提升不能简单解释为任意 permutation 或行重排。

## 6. 晋升判断

预先设定的候选晋升门槛为：平均累计 F1 至少 `+0.5 pp`，至少 `8/12` 正 cell，最差 cell 不低于约 `-2.5 pp`，new-class 平均下降不超过 `1 pp`，并且不依赖 report-F1 选择。

### 6.1 通过的小图候选

`semantic_perm` 满足当前门槛：

- 平均累计 F1：`+1.690 pp`；
- 正 cell：`9/12`；
- 最差 cell：`-1.904 pp`；
- new-class 平均变化：`-0.027 pp`；
- Cora 和 Citeseer 均为正的平均提升；
- random permutation 不具备同样收益。

因此，`semantic_perm` 是当前最值得进行新顺序和大图确认的候选。

### 6.2 尚不晋升的候选

`role_perm` 的平均提升和 new-class 结果都很好，但正 cell 只有 `7/12`，没有达到稳定性门槛。它更适合作为解释性 ablation，而不是直接替代 semantic_perm。

`anchor_role_perm`、`message_role_perm`、`plasticity_role_perm`、`adaptive_role_perm` 没有在整体稳定性上超过 semantic_perm；其中部分 arm 虽然提高 old F1，却产生更大的顺序尾部退化，不应继续堆叠到主方法。

## 7. R11 的独立结果边界

R11 的独立重载结果目前只完成 `order19605`，不能与本轮 12-cell full-arm 表混成主结果。stage 2 的结果如下：

| 数据集 | C0 | joint | anchor025 | anchor050 | anchor075 |
|---|---:|---:|---:|---:|---:|
| Cora | 50.563 | 52.628 | 49.791 | 50.008 | 51.448 |
| Citeseer | 55.330 | 56.132 | 54.943 | 55.978 | 55.183 |

R11 joint 在这个单一顺序上相对 C0 有正收益，但证据覆盖不足，且它与本轮 semantic permutation 不是同一个构造协议。因此 R11 作为历史探索结果保留，不作为当前主候选的直接替代。

## 8. 结论

本轮最清晰的结论不是“所有 role-aware 变体都有效”，而是：

> 在保持 C0 topology、旧 prefix、Student 和训练协议不变的前提下，类内 graph-text semantic permutation 能够以很小的构造开销提高持续合成 memory 的可重训效用；加入更多手工 role/message/plasticity 项没有稳定带来额外收益。

这给出了一个可审计、相对保守的候选主线：

`C0 topology + two-prototype proposal + semantic conserved payload assignment + original Student`

但它目前仍应称为“小图独立确认候选”，而不是已经完成的大图主方法。下一步只做确认性实验：

1. 在预先固定的新类别顺序和新 Student seed group 上复现 `semantic_perm`；
2. 若仍满足同一门槛，再做 arXiv/BookChild/SportsFit 的 base+4；
3. 大图只比较 C0、semantic_perm、k-center/medoid 和已有静态 TAG 凝聚基线；
4. 暂不把 role/message/plasticity/adaptive selector 叠加进主方法。

## 9. 证据文件

- Role-aware assignment 输入 memory：`results/tgc_role_aware_assignment_independent_20260926/`
- Cora 全 arm Student 重载：`results/tgc_role_aware_detailed_eval_20260926_all2/`
- Citeseer 全 arm Student 重载：`results/tgc_role_aware_detailed_eval_20260926_gpu1/`
- 构造成本：`results/tgc_role_assignment_cost_20260926_cora_payload.json`、`results/tgc_role_assignment_cost_20260926_citeseer_payload.json`
- R11 部分独立结果：`results/tgc_r11_c0_anchor_independent_20260925/`

## 10. Fresh 独立确认更新

对未参与上一轮 role-aware 开发的新顺序 `order19611`、`order19612` 进行 8-cell 重载后，`semantic_perm` 相对同 cell C0 的结果为：

- cumulative F1：`+1.898 pp`；
- old F1：`+0.930 pp`；
- new F1：`+2.988 pp`；
- 正 cell：`6/8`；
- 最差 cell：`-2.625 pp`。

这确认了 semantic assignment 的平均正信号和新类收益，但最差 cell 略低于预设的 `-2.5 pp` 容忍线。因此当前状态应从“可以直接上大图的候选”改为“已独立复现、需要做一个尾部诊断和有界回退实验的候选”。`role_perm` 在 fresh 轮平均只有 `+0.207 pp` 且最差 `-3.853 pp`，不应与 `semantic_perm` 合并成一个主方法。

完整 fresh 结果见：

- `docs/current/TGC_ROLE_AWARE_FRESH_CONFIRMATION_ZH_2026-09-26.md`；
- `results/tgc_role_aware_fresh_confirm_summary_20260926.json`；
- `results/tgc_fresh_role_aware_detailed_eval_20260926_cora/results.json`；
- `results/tgc_fresh_role_aware_detailed_eval_20260926_citeseer/results.json`。
