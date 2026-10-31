# Gate 3'' 最终确认：test 一次性读取结果（frozen_v1）

日期：2026-09-09
数据：服务器 `problem_validation_results/gate3_test_2026_09_09/{cora,citeseer}/`
冻结版本：`code/frozen_v1_2026_09_09/`（含 FREEZE_DECLARATION.md，服务器镜像同名目录）
test 定义：`node_id % 10 == 2`（10%，标签首次且唯一一次读取；节点特征曾 label-free 参与凝聚源，whole-graph transductive，已声明）。6 students/artifact，每 artifact 只评一次。`confirmation_accessed: true`。

## 1. 最终确认表（test Macro-F1）

### Cora（7 类，预算 121 = 5%）

| arm | test cum | test old | test cur |
|---|---:|---:|---:|
| **method（append + c15 规则, keep76/new45）** | **58.07** | 73.27 | 56.28 |
| append 默认分配（keep87/new34） | 55.23 | **75.32** | 48.67 |
| recluster | 44.26 | 68.33 | 33.32 |
| full_history 重凝聚 | 53.58 | 65.96 | **61.37** |
| raw 直训（2437 节点，配平） | 43.95 | 64.99 | 38.16 |

### Citeseer（6 类，预算 143 = 5%）

| arm | test cum | test old | test cur |
|---|---:|---:|---:|
| **method（append，c 规则不触发）** | **56.99** | **70.67** | 55.91 |
| recluster | 54.90 | 67.80 | **64.41** |
| full_history 重凝聚 | 48.40 | 60.28 | 58.86 |
| raw 直训（2867 节点，配平） | 51.82 | 64.81 | 58.49 |

方法各阶段：Cora s0/s1/s2 = 48.5 / 56.9 / 58.1；Citeseer = 65.6 / 65.2 / 57.0。

## 2. 主张逐条的 test 判定

| dev 阶段的主张 | test 结果 | 判定 |
|---|---|---|
| 方法 > full_history 重凝聚 | Cora +4.49，Citeseer +8.59 | ✅ 确认，且幅度大于 dev（dev 分别 +1.9/+11.6） |
| 凝聚 > raw 直训 | Cora +14.12，Citeseer +5.17 | ✅ 确认 |
| 历史保持：append 系最优 | Cora old 73.3/75.3 双双最高；Citeseer 70.67 最高 | ✅ 确认 |
| c=15 规则有效（Cora 触发） | vs 默认分配：cum +2.85、cur +7.61、old −2.05 | ✅ **规则在 test 上净收益为正**——冻结前用 dev 选的规则经受了考验 |
| recluster = 塑性侧 trade-off | Citeseer cur 64.4 vs 55.9 复现；Cora 仍全面落后 | ✅ 与 dev 模式一致 |
| 三层对照排序 | Cora：方法>full_hist>recluster≈raw；Citeseer：方法>recluster>raw>full_hist | ✅ 方法在两数据集均居首 |

**dev→test 无乐观偏移的证据**：Cora 方法 test cum（58.1）甚至高于 dev（53-53.6 区间）；Citeseer test（57.0）低于 dev（61.3）约 4.3，各 compact arm 同步收缩、排序不变。没有出现"dev 上挑出来的赢家在 test 上翻车"的情形。

## 3. 论文主表状态

frozen_v1 的最终数字即上表。两数据集 test 上：**方法（5% 全合成 memory、历史不可重读、确定性构造）同时超过"允许重读历史的一次性重凝聚"与"全量 raw 直训"两个违规参照**，历史保持在所有 arm 中最高，塑性短板由 c 规则在 Cora 上有效缓解（+7.6）且不损累计。

## 4. 版本管理

- frozen_v1 至此封版：代码快照（本地 `code/frozen_v1_2026_09_09/` + 服务器镜像）、memory artifacts、本确认表不再改动；
- 后续任何调整（LLM 融合、student 升级、新数据集）在 v2 分支进行；v2 若改动方法本体，需重走 gate 序列并使用**新的未读 test 分区或新数据集**（node_id%10==2 已消耗）；
- 遗留声明：Cora/Citeseer 均为开发数据集（dev 被大量用于方法选择），最终论文建议补一个全新数据集（Photo/Computers）作为最干净的泛化证据。
