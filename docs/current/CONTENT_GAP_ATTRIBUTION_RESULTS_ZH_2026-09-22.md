# C0 与 k-center 内容差距归因报告

日期：2026-09-22。范围：Cora/Citeseer，canonical 三阶段，5% growing memory。

## 1. 结论摘要

本轮只改变新追加节点的内容，固定 rows、row order、C0 topology、student、batch、steps 和 6 个 report seeds。
因此可以回答 k-center 的优势是否主要来自拓扑。结果显示：在固定 C0 拓扑后，k-center 的内容优势仍然存在于部分阶段，但不能解释为最终阶段普遍更强；Cora 最终反而低于 C0，Citeseer 也低于 C0。

anchor 引导的 synthetic X/text 没有形成稳定收益。只替换文本在 Cora 最终有小幅改善，但 Citeseer 退化；只替换 X 在 Citeseer 前两阶段有改善，最终回到 C0；X 与文本同时替换反而在两图最终都下降。说明当前问题不是单一的 Mean-X 或文本选择器，而是 X、文本和持续阶段之间的配对一致性。

**决策：anchor-guided synthetic 暂不晋升为主方法，不上大图；C0 继续冻结。** 下一步应优先做局部多原型/配对一致性诊断，而不是直接叠加更多 anchor 或拓扑规则。

## 2. 受控方法

| 臂 | 新行内容 | 拓扑 | 是否 synthetic |
|---|---|---|---|
| C0 | Mean-X + joint_g5_c320 concat/re-encode | 完整 C0 | 是 |
| Anchor-X | 每簇用 0.5 Mean-X + 0.5 局部 anchor X | 完整 C0 | 是；多源簇不复制原行 |
| Anchor-text | C0 X + 以局部 anchor 为中心选文本 | 完整 C0 | 是 |
| Anchor-X+text | 同时替换 X 与文本 | 完整 C0 | 是 |
| k-center fixed topology | 真实 k-center X/text | 完整 C0 | 否，仅参照 |

anchor 只用于构造指导；真实节点没有作为 synthetic row 写入。singleton 簇属于不可压缩情况，单独豁免 exact-copy 审计。

## 3. 最终结果与阶段平均

最终 F1 和阶段平均 F1 均为 report-dev pooled all-prompt Macro-F1，单位为百分数。

| 数据集 | 方法 | 最终 F1 | 阶段平均 F1 | 最终 old F1 | 最终 new F1 | 最终相对 C0 | 阶段均值相对 C0 |
|---|---|---:|---:|---:|---:|---:|---:|
| cora | C0: Mean-X + C0 text | 49.85 +/- 3.00 | 53.08 | 65.44 | 29.06 | +0.00 | +0.00 |
| cora | Anchor-X + C0 text | 46.13 +/- 1.86 | 50.34 | 64.48 | 21.66 | -3.72 | -2.74 |
| cora | C0 X + anchor-local text | 51.86 +/- 2.52 | 54.37 | 65.19 | 34.08 | +2.01 | +1.29 |
| cora | Anchor-X + anchor-local text | 48.70 +/- 2.34 | 52.38 | 63.31 | 29.21 | -1.15 | -0.70 |
| cora | k-center content + C0 topology | 45.95 +/- 4.04 | 55.66 | 67.97 | 16.59 | -3.90 | +2.58 |
| citeseer | C0: Mean-X + C0 text | 59.17 +/- 1.57 | 64.29 | 66.46 | 44.61 | +0.00 | +0.00 |
| citeseer | Anchor-X + C0 text | 59.17 +/- 2.42 | 67.46 | 66.23 | 45.07 | -0.00 | +3.17 |
| citeseer | C0 X + anchor-local text | 57.12 +/- 2.37 | 65.06 | 65.86 | 39.65 | -2.05 | +0.78 |
| citeseer | Anchor-X + anchor-local text | 56.41 +/- 2.10 | 63.71 | 64.35 | 40.55 | -2.76 | -0.57 |
| citeseer | k-center content + C0 topology | 54.34 +/- 2.44 | 65.96 | 62.29 | 38.45 | -4.83 | +1.68 |

## 4. 分阶段结果

| 数据集 | stage | C0 | Anchor-X | Anchor-text | Anchor-X+text | k-center/C0 topology |
|---|---:|---:|---:|---:|---:|---:|
| cora | 0 | 58.11 | 54.40 | 55.13 | 48.53 | 49.33 |
| cora | 1 | 51.29 | 50.49 | 56.13 | 59.92 | 71.70 |
| cora | 2 | 49.85 | 46.13 | 51.86 | 48.70 | 45.95 |
| citeseer | 0 | 64.52 | 71.37 | 68.43 | 67.26 | 77.56 |
| citeseer | 1 | 69.16 | 71.83 | 69.62 | 67.46 | 65.97 |
| citeseer | 2 | 59.17 | 59.17 | 57.12 | 56.41 | 54.34 |

## 5. 机制解释

### 5.1 k-center 的优势不是单纯拓扑造成
固定 C0 边后，k-center 仍保留真实 X-text 配对、非线性边界和尾部样本。Cora stage 1 的 k-center 为 71.70，相对 C0 高 20.41 pp；Citeseer stage 0 为 77.56，相对 C0 高 13.04 pp。这说明内容表示/配对确实是重要差异。
但最终阶段 k-center 为 Cora 45.95、Citeseer 54.34，分别低于 C0 3.90 和 4.83 pp。它不能被表述为“k-center 始终更强”，而应视为强内容参照和顺序/阶段诊断。

### 5.2 Mean-X 不是唯一问题
Anchor-X 在 Cora 最终下降 3.72 pp，在 Citeseer 最终基本持平；说明用局部尾部方向替代均值可能在早期带来覆盖，但会损害后期旧/新平衡。它不是一个普适的 X 修复。
Anchor-text 在 Cora 最终提高 2.01 pp，但 Citeseer 下降 2.05 pp；文本局部选择存在数据集依赖。
同时替换 X 与文本后 Cora/Citeseer 最终分别为 -1.15/-2.76 pp。两个局部替换的收益不能相加，说明当前 anchor X 与 anchor text 没有共享同一 student-compatible prototype。

### 5.3 old/new 分解
Cora 中 Anchor-text 的最终 new F1 比 C0 高 5.02 pp，但 old F1 低 0.25 pp，说明它主要改善新类而非整体保持；Anchor-X 则 new F1 低 7.40 pp。Citeseer 中 Anchor-X 最终 new F1 仅高 0.46 pp，而联合臂 new F1 低 4.07 pp，不能支持联合构造。

## 6. 审计与限制

- Cora：90/90 checkpoint 独立重载通过；metrics、support hash、rows 和 C0 topology 全部一致。
- Citeseer：90/90 checkpoint 独立重载通过；metrics、support hash、rows 和 C0 topology 全部一致。
- 总计 180 次独立 checkpoint 重算，0 failure。
- 只测 canonical 三阶段和 5% rate，不能据此宣称跨顺序泛化。
- k-center 使用真实节点，文本/特征的实际字节数高于 synthetic arms；本轮是同 rows、同 topology 的内容归因，不是等字节 memory 对比。
- 本轮 anchor 规则是最小诊断实现，不是最终候选；不能把单阶段正收益写成方法优势。

## 7. 后续建议

1. 暂不把 anchor-guided synthetic 升为 v6 主方法，也不上大图。
2. 下一轮只做一个更窄的机制：同一局部子簇内生成两个或多个 synthetic prototypes，并让 X 与 text 共享同一 prototype assignment；固定总 rows，比较 single-mean、two-prototype、k-center real reference。
3. 重点报告 prototype-level X-text cosine、簇内方差保留率、student response 和 old/new F1，检验联合臂下降是否来自跨模态失配。
4. 如果多原型仍不能在两个小图和独立顺序上稳定改善，就停止继续堆叠 synthetic 算子，论文主线采用 C0 协议、强 coreset 参照、内容/拓扑归因和负结果。
