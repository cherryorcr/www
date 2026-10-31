# P4 历史合成记忆管理验收报告

日期：2026-09-18  
协议：冻结 C0，5% fixed-rate growing memory，dispersion allocation，Mean-X，`joint_g5_c320`，frozen-old topology，student seeds 100--105。  
状态：P4 最低闭环完成；bounded-refresh 未晋升。

## 1. 目的

P4 检查历史 synthetic memory 是否需要随新阶段到来而局部更新。实验严格禁止回读历史 raw 数据，只允许读取旧 synthetic memory、当前 delta 和保存的统计字段。

比较三条路径：

1. `frozen_old`：旧 `x/text/edges` 均不改写，只追加当前 delta。
2. `rescore_only`：当前 runner 中作为不改写 synthetic 内容和拓扑的空操作控制；统计重评分接口尚未实现，因此不能把它称为已验证的质量管理器。
3. `bounded_refresh_10%`：每个阶段最多刷新旧 rows 的 10%，以当前 block 的图文相似度选择来源，`alpha=0.25` 做小幅插值。

## 2. 正确性审计

### 2.1 rescore-only 控制

两套数据集 stage 2 的 frozen-old 与 rescore-only memory 文件 SHA256 完全一致：

| 数据集 | frozen-old SHA256 | rescore-only SHA256 | 结论 |
|---|---|---|---|
| Cora | `93B5611D...BCDB31A8E9` | 相同 | 内容和拓扑未改变 |
| Citeseer | `186E1401...487207C39` | 相同 | 内容和拓扑未改变 |

因此 rescore-only 是有效的零变化控制，而不是一个隐含的重凝聚方法，也不构成“统计重评分有效”的证据。

### 2.2 bounded-refresh 中断修复

首轮运行在 stage 1 复制 `refresh_metadata` 字典时将字典误当作 Tensor 调用 `.clone()`，导致两个数据集均中断。修复为递归复制 Tensor、list、dict 和 tuple 后，仅重跑受影响的 bounded-refresh 分支；已完成的 frozen-old/rescore-only 产物没有重算或覆盖。

## 3. 结果

### 3.1 阶段结果

| 数据集 | 管理方式 | stage 0 | stage 1 | stage 2 / final |
|---|---:|---:|---:|---:|
| Cora | frozen-old | 58.11 | 51.29 | **49.85** |
| Cora | rescore-only | 58.11 | 51.29 | **49.85** |
| Cora | bounded-refresh 10% | 58.11 | 53.91 | **49.79** |
| Citeseer | frozen-old | 64.52 | 69.16 | **59.17** |
| Citeseer | rescore-only | 64.52 | 69.16 | **59.17** |
| Citeseer | bounded-refresh 10% | 64.52 | 67.55 | **56.53** |

表中指标为累计 seen-class dev Macro-F1 的六个 student seed 均值；stage 1/2 的 current F1 另外审计如下：

| 数据集 | 管理方式 | final current F1 | final old mean F1 |
|---|---:|---:|---:|
| Cora | frozen-old | 40.91 | 75.05 |
| Cora | bounded-refresh 10% | 42.03 | 74.79 |
| Citeseer | frozen-old | 55.48 | 76.67 |
| Citeseer | bounded-refresh 10% | 53.53 | 73.95 |

相对 frozen-old 的 final 变化为：

| 数据集 | cumulative F1 | current F1 | old mean F1 |
|---|---:|---:|---:|
| Cora | -0.06 | +1.13 | -0.26 |
| Citeseer | **-2.64** | -1.95 | -2.72 |

### 3.2 实际刷新规模

| 数据集 | stage 1 刷新 | stage 2 刷新 | 10% 上限是否遵守 |
|---|---:|---:|---|
| Cora | 2 / 25 | 7 / 67 | 是 |
| Citeseer | 4 / 38 | 9 / 86 | 是 |

刷新对象为旧 rows 的低 mass 行，旧 rows 数量、总 memory budget 和追加 rows 均未改变；因此收益或退化不能归因于更多 memory。

## 4. 结论与门禁判断

P4-B 没有通过主方法晋升门禁：

- 两个数据集没有同时改善 cumulative F1 或 old retention；
- Citeseer 的 final cumulative、current 和 old F1 均明显下降；
- Cora 的 final cumulative 基本不变，只有 current F1 小幅上升，同时 old retention 略降；
- 当前小图证据不足以证明刷新能解决长链历史过时问题。

因此：

1. `rescore_only` 保留为审计控制，不作为算法组件。
2. `bounded_refresh_10%` 保留为 P4 负结果，不追加 20% refresh、merge/split 或 importance eviction。
3. C0 的 frozen-old memory 管理继续作为默认方案。
4. 下一步转入 C3 单阶段受控传播实验；不把 P4 与拓扑、文本或 response 组合。

## 5. 当前方法含义

这个结果不证明历史记忆永远不应更新，只说明在当前小图、三阶段、5% budget、Mean-X 和当前 student 接口下，简单的低 mass 局部刷新没有稳定价值。刷新同时改变了旧行语义，哪怕只插值 25%，也可能破坏旧类的可重训表示；而新阶段信息不足以可靠决定哪些历史 rows 应被改写。

若未来长链实验显示 frozen-old 随阶段明显过时，应优先设计带保持约束的局部 refresh，例如以旧 student response、拓扑漂移和覆盖缺口共同决定刷新，而不是直接扩大刷新比例。

## 6. 产物

- 代码修复：`code/v2_llm_fusion/run_student_compatible_append.py`
- frozen/rescore 结果：`results/p4_memory_management_20260918/`
- bounded-refresh 重跑结果：`results/p4_memory_management_20260918_retry/`
- 远程协议：5% fixed-rate、Cora/Citeseer canonical 三阶段、六个 student seeds。
