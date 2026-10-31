# Role-Aware Matching 六顺序确认报告

> **修订提示（2026-09-24）：本报告已被 [TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md](TGC_CONSERVED_ROLE_PERMUTATION_COMPLETION_ZH_2026-09-24.md) supersede。** 本报告中的 `role_match` 实现没有迁移被占用 slot 的完整 payload，导致 source coverage/mass 不守恒；其中的正收益数字不再作为 role 机制证据，仅保留作历史审计。

日期：2026-09-24  
状态：完成 Cora/Citeseer 三顺序 stage-1 确认；平均有收益，但未通过稳定晋升门槛

## 1. 方法

本轮在局部双原型 split/merge 基础上实现固定代价的 `role_match`。它不改变：

- source split/merge proposal；
- 合成 X 和文本来源；
- 总 rows 与 mass 总预算；
- 旧 synthetic prefix；
- C0 `edge_index`；
- Student、batch、steps 和 report seeds。

它只改变三个 changed proposals 写入同类 C0 row slots 的匹配。对每个类别，在所有同类 C0 slots 中做一次 Hungarian matching：

\[
C(r,s)=\lVert\mu_r-c_s\rVert_2^2
 +0.50\lVert\phi_r-\psi_s\rVert_2^2
 +0.10\left|\log\frac{m_r+1}{m_s+1}\right|
\]

其中 `mu/c` 是 joint content center，`phi/psi` 是由 degree、direct old-neighbor degree、new-new degree、old-neighbor ratio 和 old-endpoint load 构成的传播角色向量，`m` 是 source mass。权重在实验前固定为 `1.0/0.50/0.10`，没有按 report/test F1 选择。

## 2. 协议

| 项目 | 设置 |
|---|---|
| 数据集/顺序 | Cora `order19601/19602/19603`；Citeseer `order19601/19602/19603` |
| 阶段 | stage 1 |
| Student seeds | `150..155` |
| memory | 5% growing memory，总 rows 与 C0 相同 |
| 文本 | top-5，显式 320 字符 cap，冻结 MiniLM 重编码 |
| 拓扑 | C0 topology bitwise copy |
| baseline | 同一 source assignment 重构的 `c0_320` |
| 历史 raw graph | 不访问历史完整 raw graph |
| 选择 | 无 report-F1 selector |

## 3. 六个 cell 的结果

差值为相对于 `c0_320` 的 paired cumulative Macro-F1，单位为百分点。

| 数据集 | 顺序 | semantic matched | role_match | role_match 正 seed | role_match 最差 seed | role_match - matched |
|---|---|---:|---:|---:|---:|---:|
| Cora | order19601 | `+1.520` | `+1.187` | 5/6 | `-3.498` | `-0.333` |
| Cora | order19602 | `-0.072` | `+2.560` | 5/6 | `-4.455` | `+2.631` |
| Cora | order19603 | `+0.371` | `+5.055` | 5/6 | `-3.989` | `+4.685` |
| Citeseer | order19601 | `-1.970` | `-0.956` | 3/6 | `-7.713` | `+1.014` |
| Citeseer | order19602 | `+2.839` | `+1.751` | 3/6 | `-1.640` | `-1.088` |
| Citeseer | order19603 | `-1.702` | `-0.111` | 2/6 | `-1.482` | `+1.591` |

汇总：

| 汇总 | semantic matched | role_match |
|---|---:|---:|
| 6 cell 平均 | `+0.164` | **`+1.581`** |
| 36 个 paired seed 平均 | `+0.164` | **`+1.581`** |
| 正 seed 数 | 17/36 | 23/36 |
| 正 cell 数 | 3/6 | 4/6 |
| 最差 seed | `-8.063` | `-7.713` |
| role_match 相对 matched | - | **`+1.417`** |

## 4. 结果解释

### 4.1 role-aware matching 确实比原始 semantic matching 更有希望

在 Cora/order19602、Cora/order19603 上，加入 role cost 后相对 semantic matched 分别提高 `+2.631` 和 `+4.685 pp`。这支持了一个具体机制判断：只按 joint semantic center 将 proposal 对齐到 row slots，不足以描述 GCN 中的传播角色；将 slot role 纳入匹配能够减少一部分错误放置。

### 4.2 仍未达到稳定主方法标准

role_match 在 Citeseer/order19601 仍为 `-0.956 pp`，最差 paired seed 为 `-7.713 pp`；Citeseer/order19602 也低于 semantic matched。当前收益主要由 Cora 三顺序驱动，不能称为跨数据集稳定提升。

### 4.3 失败不应再用“继续调 degree 方向”解决

此前 low/high degree 诊断已证明不同数据集的最优方向相反。固定 role cost 的平均收益更高，但仍存在极端负 seed，说明缺失的不是一个更大的 role 权重，而是：

1. proposal role summary 仍然过于粗糙；
2. role cost 没有估计 old/new message risk；
3. 一个 proposal 是否安全取决于类别顺序和 Student 初始化；
4. role-aware matching 没有保守回退机制。

## 5. 当前晋升判定

role_match 暂不进入论文主方法，原因是：

- 只完成 stage 1，尚未确认多阶段长链；
- 6 个 cell 中只有 4 个正；
- 最差 seed 低于 `-2 pp`；
- Citeseer 仍没有稳定提升；
- 角色成本是启发式代理，尚未证明能预测 Student response。

但它不应被归为纯负结果。它是目前最有潜力的下一版候选，因为它在相同内容和拓扑下，仅改变 proposal-to-slot assignment，就能相对 semantic matching 获得六格平均 `+1.417 pp` 的改善。

## 6. 下一步：保守 role-aware matching

下一轮不再扫描更多 `lambda_role`，而是加入一个固定、可审计的回退门：

1. 先生成 semantic matched 和 role_match 两个候选；
2. 只使用不依赖 report/test label 的 construction-time 信号估计风险，包括 role cost、proposal-to-slot semantic cost、old/new message drift 和 C0 anchor distance；
3. 若 role_match 的局部风险超过预注册阈值，则回退 semantic matched；
4. 若 semantic matched 也出现高风险，则回退 `c0_320`；
5. 不使用 Student 最终 Macro-F1 做选择；
6. 先在一条开发顺序上封存阈值，再在其余五条顺序确认；
7. 同时报告平均收益、最差 cell、old/new F1、负 seed 比例和构造成本。

这个回退机制的目标不是让每个 cell 都提升，而是限制 role_match 的灾难性负尾部。如果它不能把最差 cell 控制到 `-2 pp` 内，role_match 只能作为机制分析，C0 继续作为主方法。

## 7. 产物

- Runner：[run_tgc_two_prototype_role_diagnostic.py](../../code/v2_llm_fusion/run_tgc_two_prototype_role_diagnostic.py)
- Remote wrapper：[tgc_two_prototype_role_diagnostic_remote.py](../../tools/tgc_two_prototype_role_diagnostic_remote.py)
- 结果目录：[tgc_two_prototype_role_diagnostic_20260924](../../results/tgc_two_prototype_role_diagnostic_20260924/)。其中本报告只采用包含 `c0_320`、`role_match` 的 `*_v2`/`*_v3` 或 canonical 六顺序目录；早期未加入重构 C0 控制的 smoke 目录不参与结论。
- 前置文本确认：[TGC_TWO_PROTOTYPE_TEXTCAP_CONFIRMATION_ZH_2026-09-24.md](TGC_TWO_PROTOTYPE_TEXTCAP_CONFIRMATION_ZH_2026-09-24.md)
