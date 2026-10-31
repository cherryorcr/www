# TGC-R12 Class-Preserving Coverage Floor 完成报告

## 1. 结论先行

R12 只针对一个问题：**新类别进入后，assignment 是否因过度集中而造成新类塑性不足**。它没有加入新的文本、预算、拓扑或 student 优化模块。

R12 在 R11 的 `anchor025` 上加入类别内 coverage floor：

\[
P_{\mathrm{floor}}=(1-\gamma)P_{\mathrm{anchor025}}+\gamma U_{\mathrm{class}},
\]

其中 `U_class` 只允许当前增量样本与同类合成行发生 assignment。候选为：

| arm | 定义 |
|---|---|
| `c0` | 冻结 C0，共同控制 |
| `joint` | R10 sealed joint assignment |
| `anchor025` | `0.75 P_initial + 0.25 P_joint` |
| `classfloor010` | `0.90 P_anchor + 0.10 U_class` |
| `classfloor025` | `0.75 P_anchor + 0.25 U_class` |
| `classfloor050` | `0.50 P_anchor + 0.50 U_class` |
| `randomfloor025` | 同类约束的固定随机分布，`gamma=0.25`，负控制 |

主要结果：

- `anchor025` 仍是本轮最好的候选，但最终平均只比 C0 高 `+0.142 pp`；
- `classfloor010/025/050` 的最终平均分别为 `-0.309/-1.086/-2.555 pp`；
- `classfloor010` 的新类 F1 平均为 `-3.140 pp`，没有改善塑性；
- floor 越强，old/new trade-off 越差，`classfloor050` 同时伤害旧类和新类；
- `randomfloor025` 为 `-1.437 pp`，说明同类约束本身并不是稳定收益来源；
- R12 不满足晋升门禁，C0 继续作为冻结主线；
- 本轮的正面价值是排除了一个看似合理的“类别覆盖下限”机制，并进一步说明问题不在简单的类内覆盖不足。

## 2. 实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19601`、`order19602`、`order19603` |
| 阶段 | base + 2 incremental，共 3 stages |
| memory | 累计合法 pool 的 5% growing rows |
| report seeds | `150..155`，6 个 |
| student | 两层 GCN/GraphCLIP student，从头训练；batch 16；Adam `1e-3` |
| student steps | `max(50, min(2048, 16*rows))` |
| 文本 | 由同一 assignment 选最多 5 个来源，320 字符 concat 后冻结 MiniLM 重编码 |
| 拓扑 | 同一 assignment 投影后的二值拓扑 |
| 历史访问 | 只使用旧 synthetic memory 与当前 raw delta；历史 raw reads = 0 |
| 选择 | 所有 arm 全部封存评估，不根据 report F1 选参数 |
| test/report labels | 构造和选择均未使用 |

每个 dataset/order 有 `3 stages × 7 arms × 6 seeds = 126` 个 fit slots；两数据集合计 `756` 个 slots，实际执行 `540` 次 student fit，其余为合法的相同 memory 复用。

## 3. 具体算法流程

每个增量 stage 的 R12 流程如下：

1. 从 sealed R10/R11 artifact 读取 C0 parent、当前 raw delta、`P_initial` 和 `P_joint`；
2. 先计算 `P_anchor = 0.75 P_initial + 0.25 P_joint`；
3. 根据当前 delta 的类别与合成行类别构造 `U_class`，每个当前样本只向同类合成行分配均匀质量；
4. 对 `gamma ∈ {0.10, 0.25, 0.50}` 生成 class-floor assignment；
5. 用同一个 assignment 同步生成：
   - `X_new = P^T X_delta / column_mass`；
   - top-5/320-char concat-reencode text；
   - assignment 投影后的二值 topology；
6. 旧行的 `X/text/mass/depth/stage` 和历史旧边保持不变；
7. 使用统一 student 协议从头训练，并报告 all-prompt pooled Macro-F1、old F1 和 new F1；
8. 独立 audit 重建 assignment、X、文本 embedding、拓扑和 checkpoint 指标。

本轮修复过一个纯实现问题：sealed R10/R11 assignment 在 CPU，而 `U_class` 初始在 CUDA；现已统一 assignment 的存储设备，材料化时再搬到目标 GPU。修复后重新清空 R12 未完成目录并完整重跑，避免混入旧 protocol hash。

## 4. 最终阶段结果

下表是 stage 2 的 all-prompt pooled Macro-F1，括号为相对同一 dataset/order 的 C0 差值。

| 数据集 | 顺序 | C0 | joint | anchor025 | classfloor010 | classfloor025 | classfloor050 | randomfloor025 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | order19601 | 50.382 | 51.644 (+1.262) | 51.736 (+1.354) | 50.698 (+0.315) | 51.099 (+0.717) | 49.243 (-1.139) | 50.071 (-0.312) |
| Cora | order19602 | 53.048 | 48.005 (-5.043) | 52.037 (-1.011) | 49.369 (-3.679) | 50.073 (-2.974) | 49.250 (-3.797) | 52.946 (-0.101) |
| Cora | order19603 | 46.152 | 48.365 (+2.213) | 46.645 (+0.493) | 48.185 (+2.033) | 46.900 (+0.748) | 43.164 (-2.988) | 45.750 (-0.402) |
| Citeseer | order19601 | 54.535 | 55.517 (+0.982) | 54.484 (-0.051) | 54.499 (-0.036) | 53.116 (-1.420) | 53.547 (-0.988) | 52.912 (-1.624) |
| Citeseer | order19602 | 52.035 | 54.405 (+2.369) | 54.205 (+2.169) | 54.384 (+2.348) | 53.190 (+1.155) | 53.348 (+1.312) | 50.473 (-1.563) |
| Citeseer | order19603 | 52.933 | 50.843 (-2.090) | 50.829 (-2.104) | 50.095 (-2.838) | 48.193 (-4.740) | 45.202 (-7.731) | 48.314 (-4.619) |

## 5. 汇总结果

| 方法 | 最终 F1 | 最终相对 C0 | old F1 差 | new F1 差 | 阶段平均相对 C0 | 最差最终 cell | 正收益 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/6 |
| joint | 51.463 | -0.051 | +1.342 | -1.917 | -0.920 | -5.043 | 4/6 |
| anchor025 | **51.656** | **+0.142** | +1.629 | -2.059 | -0.856 | -2.104 | 3/6 |
| classfloor010 | 51.205 | -0.309 | +1.559 | -3.140 | -0.996 | -3.679 | 3/6 |
| classfloor025 | 50.429 | -1.086 | +0.368 | -3.455 | -1.526 | -4.740 | 3/6 |
| classfloor050 | 48.959 | -2.555 | -1.056 | -5.186 | -2.595 | -7.731 | 1/6 |
| randomfloor025 | 50.078 | -1.437 | -0.475 | -3.083 | -1.537 | -4.619 | 0/6 |

## 6. 结果解释

### 6.1 类别覆盖下限没有解决新类塑性

如果新类塑性不足的主要原因是新类样本被 assignment 过度集中到少数旧类别行，那么加入 `U_class` 应该提高 new F1。但结果相反：`classfloor010` 的 new F1 平均下降 `3.14 pp`，强 floor 下降更多。因此“同类覆盖不足”不是当前主要瓶颈，或者 assignment 的类别约束破坏了更重要的图文联合结构。

### 6.2 主要损失来自共同 assignment 改变后的可重训性

R12 同时改变 X、文本来源和拓扑。即使类别覆盖更均匀，也会改变 student 看到的三种信号之间的共同几何关系。结果中 `classfloor050` 的 old/new 都恶化，说明强行 coverage floor 不是只在旧类保持与新类塑性之间做简单交换，而是破坏了合成样本的整体结构。

### 6.3 顺序依赖仍然是主问题

在 Cora/order19603，`classfloor010` 最终高于 C0 `+2.033 pp`；但在 Cora/order19602 低 `-3.679 pp`，Citeseer/order19603 低 `-2.838 pp`。这说明局部 cell 的正收益不能外推为通用机制。`randomfloor025` 在所有最终 cell 均不为正，进一步排除了“只要加入一点同类随机覆盖就会变好”的解释。

### 6.4 R11 的正信号仍然是稳定化而非塑性

`anchor025` 把 joint 的最差最终 cell 从 `-5.043 pp` 改善到 `-2.104 pp`，但 new F1 仍下降 `-2.059 pp`。因此当前最可靠的联合机制结论仍是：anchor 能限制 joint 的坏偏移，却没有把新类信息变成稳定的 student plasticity。

## 7. 晋升门禁

| 条件 | 要求 | 最好候选 anchor025 | 判定 |
|---|---:|---:|---|
| 最终平均相对 C0 | `>= +0.50 pp` | `+0.142 pp` | 未通过 |
| 阶段平均 | `>= 0` | `-0.856 pp` | 未通过 |
| 最差最终 cell | `>= -2 pp` | `-2.104 pp` | 未通过 |
| new F1 平均 | `>= 0` | `-2.059 pp` | 未通过 |
| student 成本 | 与 C0 同量级 | 约相同 | 通过 |
| memory/历史访问审计 | 全部通过 | 通过 | 通过 |

## 8. 对后续路线的影响

R12 之后不应继续堆叠更多 assignment floor、文本压缩或 topology 小模块。当前证据支持的路线是：

1. C0 继续作为冻结主方法和共同控制；
2. R10 joint、R11 anchor、R12 class-floor 作为机制诊断与负结果；
3. 如果仍要攻新类塑性，下一次应直接测量 student 的增量学习响应或新类梯度可达性，而不是再对 assignment 做静态类别修正；
4. 新实验必须把 X、text、topology 的共同变化拆开记录，并加入 matched-random displacement 对照；
5. 只有在至少两个数据集、三个顺序上平均正收益，且 new F1 不下降、最差 cell 不超过 `-2 pp` 后，才考虑晋升为新主方法。

本轮结论不是“持续 TAG 凝聚不可行”，而是更具体地说明：**简单的类别覆盖约束不足以弥补 student-compatible construction 的响应缺口**。后续应从 student 可学习性和增量信号传递机制入手，而不是继续增加静态 assignment 规则。

## 9. 结果与代码

- [R12 runner](../../code/v2_llm_fusion/run_tgc_r12_classfloor.py)
- [R12 summarizer](../../tools/summarize_tgc_r12.py)
- [R12 sealed results](../../results/tgc_r12_classfloor_20260923a/)
- [R12 report artifacts](../../reports/tgc_r12_classfloor_20260923a/)
- [R11 preceding report](TGC_R11_C0_ANCHOR_COMPLETION_ZH_2026-09-23.md)

