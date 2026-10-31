# 双原型 Topology-Role 诊断报告

> **修订提示（2026-09-24）：本报告的 role matching 结果已被 [TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md](TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md) supersede。** 原实现直接覆盖目标 slots，没有完成完整 payload 双射；本报告只保留为缺陷发现和历史记录。

日期：2026-09-24  
状态：完成三格六 seed 诊断；确认槽位角色是重要因素，但未得到可直接泛化的单一 degree 规则

## 1. 实验问题

双原型文本上限复现实验显示：双原型内容在 Cora/order19601 有正向信号，但在 Citeseer 和另一条 Cora 顺序不稳定。下一步需要区分：

- 双原型内容本身是否无效；
- 还是同一组内容被写入不同 C0 synthetic row slots 后，继承了不同的传播角色，从而改变 Student 的结果。

本轮固定同一组双原型 source pieces、X、文本、总 rows 和 C0 topology，只改变“哪些同类 C0 row slots 接收这些改变”。这是 role attribution，不是 report-F1 selector，也不是新的主方法。

## 2. 统一协议

| 项目 | 设置 |
|---|---|
| 数据集/顺序 | Cora `order19601`、Cora `order19602`、Citeseer `order19601` |
| 阶段 | stage 1 |
| Student seeds | `150..155` |
| memory | 5% growing memory，总 rows 与 sealed C0 相同 |
| Student | 原始两层 GCN、trainable text projection、双向 InfoNCE |
| 文本 | top-5 source snippets，显式 320 字符上限，冻结 MiniLM 重编码 |
| 拓扑 | 每个 arm 使用同一 sealed C0 `edge_index`，bitwise 不变 |
| 旧 prefix | bitwise 不变 |
| 历史 raw graph | 不访问历史完整 raw graph |
| baseline | `c0_320`：用同一 current source assignment 重构的 C0 内容 |

注意：不使用 sealed C0 直接作为唯一 baseline。此前已发现 sealed C0 与重构 `c0_320` 可能有非零差异，所以本轮所有 role effect 都相对于同一 `c0_320` 配对。

## 3. 方法臂

`matched` 是原双原型的 Hungarian semantic matching。其余 arm 保留同一组 changed source pieces，只在每个类别内部选择目标 slots：

| 方法 | 改变写入的 slots |
|---|---|
| `matched` | 原双原型的语义 Hungarian matching |
| `degree_low` | 同类 C0 slots 中 degree 最低的若干 slots |
| `degree_high` | 同类 C0 slots 中 degree 最高的若干 slots |
| `cross_degree_low` | 同类 C0 slots 中 direct old-neighbor degree 最低的若干 slots |
| `cross_degree_high` | 同类 C0 slots 中 direct old-neighbor degree 最高的若干 slots |

其中 degree、cross-degree 只作为诊断标签，不参与最终 report-F1 选择。所有 arm 的 changed proposal multiset 相同，避免把差异误归因于 source split/merge 本身。

## 4. 累计 Macro-F1 结果

差值相对于同一 cell 的 `c0_320`，单位为百分点；`pos` 是六个 Student seeds 中正差值的数量。

| 数据集/顺序 | matched | degree-low | degree-high | cross-low | cross-high |
|---|---:|---:|---:|---:|---:|
| Cora/order19601 | `+1.520` (4/6) | `-3.376` (0/6) | **`+2.909` (5/6)** | `-1.211` (2/6) | `-3.650` (1/6) |
| Citeseer/order19601 | `-1.970` (1/6) | **`+11.045` (6/6)** | `-0.161` (3/6) | `+1.771` (5/6) | `+0.647` (5/6) |
| Cora/order19602 | `-0.072` (3/6) | `-1.922` (1/6) | `-8.749` (0/6) | `-3.205` (3/6) | `-5.968` (0/6) |

相对于 `matched` 的差值同样显示出明显的角色依赖：

| 数据集/顺序 | degree-low | degree-high | cross-low | cross-high |
|---|---:|---:|---:|---:|
| Cora/order19601 | `-4.896` | `+1.389` | `-2.732` | `-5.171` |
| Citeseer/order19601 | `+13.015` | `+1.809` | `+3.741` | `+2.617` |
| Cora/order19602 | `-1.851` | `-8.678` | `-3.133` | `-5.896` |

## 5. Old/New 与 margin 分解

| 数据集/顺序/arm | old-F1 差 | new-F1 差 | real-new margin 差 | synthetic margin 差 |
|---|---:|---:|---:|---:|
| Cora/19601 matched | `+3.751` | `+0.099` | `+0.125` | `-0.160` |
| Cora/19601 degree-low | `-3.612` | `-4.144` | `+0.217` | `+0.529` |
| Cora/19601 degree-high | `+7.599` | `+3.810` | `+0.709` | `+0.352` |
| Citeseer/19601 matched | `-2.507` | `+0.408` | `+0.853` | `+0.292` |
| Citeseer/19601 degree-low | `+15.449` | `+6.958` | `+1.511` | `+0.565` |
| Citeseer/19601 degree-high | `+1.533` | `+0.027` | `+0.450` | `-0.348` |
| Cora/19602 matched | `+3.988` | `-4.256` | `-1.432` | `-1.284` |
| Cora/19602 degree-low | `+1.184` | `-5.387` | `-1.694` | `-0.663` |
| Cora/19602 degree-high | `-2.779` | `-12.014` | `-3.077` | `-1.823` |

这些结果说明：role 选择不仅影响 old retention，也影响 new plasticity；但同一个 role 方向并不保证同一个数据集上的 new-class gain。尤其 Cora/order19602 中，所有 role arm 的 new F1 都下降，说明该 cell 可能存在更深的 source assignment 或 Student response 问题。

## 6. 结果解释

### 6.1 已确认

1. 双原型收益不是只由内容 source pieces 决定；目标 C0 row slot 的传播角色会显著改变结果。
2. 低/高 degree 不是普适的好坏规则。Cora/order19601 偏向高 degree，Citeseer/order19601 偏向低 degree，Cora/order19602 两者都不能恢复。
3. cross-degree 也不能单独作为通用门控变量。
4. Student 的 real-new margin 只能部分解释结果。比如 Citeseer degree-low 同时有较高 margin 和较大 F1 增益，但 Cora/order19601 degree-low margin 并不差，F1 仍显著下降，说明 old/new message competition 和 slot role 的联合影响不能由单一 margin 代理概括。

### 6.2 尚未确认

本轮没有证明“role-aware 方法已经有效”。因为 low/high 是事后诊断臂，不是合法的在线选择器；如果把所有 role 结果挑最好，会产生严重的 report-F1 选择偏差。当前只能说：

> 内容 prototype 应与 topology role 联合匹配，固定 degree 规则不足以泛化。

## 7. Role-aware matching 实验更新

下一步不再扫描固定 low/high 规则，而实现最小的 role-aware slot matching：

\[
C(r,s)=\lambda_{sem}\lVert\mu_r-c_s\rVert_2^2+
\lambda_{role}\lVert\phi_r-\psi_s\rVert_2^2+
\lambda_{mass}\left|\log\frac{m_r+\epsilon}{m_s+\epsilon}\right|
\]

其中：

- `r` 是 split/merge proposal；
- `s` 是同类候选 C0 slot；
- `mu_r`、`c_s` 是 joint content center；
- `phi_r` 是 proposal 的预期传播角色摘要；
- `psi_s` 是 C0 slot 的 degree、cross-degree、new-new degree、old-neighbor ratio 和局部负荷；
- `m_r`、`m_s` 是 proposal/source mass。

本轮随后已实现并完成固定权重的最小版本：

- `lambda_sem=1.0`；
- `lambda_role=0.50`；
- `lambda_mass=0.10`；
- proposal 在每个类别的所有 C0 slots 中匹配，而不是只在原三个 slots 中匹配；
- 权重在实验前固定，没有使用 report/test F1。

六个 dataset/order cell 的 `role_match` 相对同一 `c0_320` 结果为：

| 数据集/顺序 | role_match 相对 c0_320 | 正 seed |
|---|---:|---:|
| Cora/order19601 | `+1.187` | 5/6 |
| Cora/order19602 | `+2.560` | 5/6 |
| Cora/order19603 | `+5.055` | 5/6 |
| Citeseer/order19601 | `-0.956` | 3/6 |
| Citeseer/order19602 | `+1.751` | 3/6 |
| Citeseer/order19603 | `-0.111` | 2/6 |

六格平均为 `+1.581 pp`，相对原始 semantic matched 平均高 `+1.417 pp`，但最差 seed 为 `-7.713 pp`。因此它是目前最有潜力的候选，不是已经稳定的主方法。

下一步不再扫描固定 low/high 规则，而是为 `role_match` 增加固定的保守回退门：

1. `alpha_content`、X、文本、rows、C0 topology 不变；
2. 不使用 report/test F1；
3. 只在每个类的三个受影响 slots 内做 matching；
4. 只比较 `lambda_role=0` 与一个预注册的小值，不扫描大量超参数；
5. 先在一个开发顺序上估计 role 统计，再在独立顺序和 Citeseer 上确认，避免用同一 cell 选择再报告；
6. 若回退门不能把最差 cell 控制到 `-2 pp` 内，停止双原型路线，不再叠加 topology gate。

这里的关键不是“把新内容总是放到高 degree 或低 degree”，而是让 proposal 的内容/质量/质量负荷与目标 slot 的传播角色相容。

## 8. 当前决策

- C0 继续作为可靠主方法和主比较锚点。
- 双原型在严格文本 cap 下未被确认是稳定跨数据集方法。
- role attribution 提供了新的机制证据；固定权重 role-aware matching 有平均收益，但仍有严重负尾部。
- role-aware matching 暂不升为主方法，下一步只做构造时保守回退验证。
- 暂不进入大图，不将本轮 role 诊断写成性能主结果，不宣称 SOTA。

## 9. 产物

- Runner：[run_tgc_two_prototype_role_diagnostic.py](../../code/v2_llm_fusion/run_tgc_two_prototype_role_diagnostic.py)
- Remote wrapper：[tgc_two_prototype_role_diagnostic_remote.py](../../tools/tgc_two_prototype_role_diagnostic_remote.py)
- 结果目录：[tgc_two_prototype_role_diagnostic_20260924](../../results/tgc_two_prototype_role_diagnostic_20260924/)
- 文本上限确认报告：[TGC_TWO_PROTOTYPE_TEXTCAP_CONFIRMATION_ZH_2026-09-24.md](TGC_TWO_PROTOTYPE_TEXTCAP_CONFIRMATION_ZH_2026-09-24.md)
- Role-aware matching 六顺序确认：[TGC_ROLE_AWARE_MATCHING_CONFIRMATION_ZH_2026-09-24.md](TGC_ROLE_AWARE_MATCHING_CONFIRMATION_ZH_2026-09-24.md)
