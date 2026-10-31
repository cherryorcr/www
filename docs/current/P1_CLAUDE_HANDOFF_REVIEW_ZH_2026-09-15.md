# P1 Claude 交接审查与受控重跑记录

日期：2026-09-15  
审查对象：

- `P1_HANDOFF_TO_CHATGPT_ZH_2026-09-15.md`
- `P1_ROUND1_BUDGET_ALLOCATION_ZH_2026-09-15.md`

## 1. 指令与已完成工作的区分

Claude 交接文档同时包含结果报告和后续执行清单，不能把清单中的内容视为已完成。

### 已完成且可复核

- Cora/Citeseer 第一轮 30 个 fixed-rate 结果；
- `dispersion_floor` 负对照 2 个结果；
- BookChild/SportsFit/arXiv 第二轮 21 个结果；
- 合计 53 个 `results.json`，所有已检查结果 `test_accessed=false`、`student_batch=16`、student seeds 为 100--105。

### 仍未完成

- fixed-total、hybrid、adaptive-rate 的性能实验；
- 在 matched keep/new quota 下的 allocation 复核；
- 按子簇复杂度分配 rows；
- P1 的统一成本曲线、forgetting 和覆盖度表；
- P2--P5。

因此，原交接文档中的“P1 完成”只能解释为“第一轮 fixed-rate 结果完成”，不能解释为 P1 总体验收完成。

## 2. 关键审查发现：allocation 与追加配额混淆

原始 53 个结果并没有严格满足“只改变类内 allocation、保持历史保留和新增 rows 相同”。以 Cora 5% 为例：

| allocation | s2 keep old | s2 new | total |
|---|---:|---:|---:|
| `legacy_15` | 49 | 45 | 94 |
| `equal` | 67 | 27 | 94 |
| `dispersion` | 67 | 27 | 94 |

这是 runner 中 `legacy_15` 的历史 floor 配额逻辑导致的：它会优先保证每个新类 15 行，再从旧 memory 中回收 rows。其它 allocation 使用普通 `k_t - old_rows` 逻辑。

所以原报告中“Cora 上 legacy_15 胜出、Citeseer 上 dispersion 胜出”的方向，混合了两个因素：

1. 类内 rows 如何分配；
2. 新旧 memory 如何分配总预算。

该结果仍可作为“历史 legacy append policy”的结果，但不能作为纯 allocation 因果结论。

## 3. 版本审查

53 个历史结果包含两个 runner hash：

- `2175e546...`：第一轮主要结果及早期大图任务；
- `832facea...`：round 3 和第二轮后续结果。

差异对应 `dispersion_floor` 补丁；普通 allocation arm 的核心流程基本一致，但不能在 manifest 层面称为单一代码版本。后续结果必须保存 runner、protocol、SCA 和 gate runner 的完整 hash，并按 hash 分组聚合。

## 4. 已实施的修正

本地 `run_unified_v3_continual.py` 增加了显式开关：

```text
--legacy-floor-quota
```

默认关闭时，所有 allocation 使用同一套 `keep_old_rows / new_rows` 配额，`legacy_15` 只表示类内分配时的 floor；打开该开关才复现历史配额行为。

同时新增：

```text
code/v2_llm_fusion/launch_p1_controlled_quota_2026_09_15.sh
```

远程受控结果目录：

```text
/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/p1_controlled_quota_2026_09_15/
```

第一批先运行 Cora/Citeseer、5%、五种 allocation、`append_fixed + k_center + random_real`，用于判断原始 allocation 结论是否仍成立。

截至当前，Cora 5% 五种 allocation 已完成，且全部使用相同的阶段配额 `[25, 67, 94]`，s2 都是 `keep=67, new=27`：

| allocation | append_fixed s2 cumulative F1 | k-center s2 cumulative F1 |
|---|---:|---:|
| `legacy_15` | 45.50 | 46.35 |
| `equal` | 45.50 | 46.35 |
| `linear` | 44.52 | 46.35 |
| `sqrt` | 47.28 | 46.35 |
| `dispersion` | 49.85 | 46.35 |

这已经证明：原始 Cora 表中 `legacy_15=57.87` 的优势主要来自它把 s2 新增 rows 提高到 45，而不是来自 legacy 的类内分配本身。受控结果暂时支持 `dispersion`，但还不能在 Citeseer 和大图上外推。

## 5. 当前结论的有效范围

- v3 fixed-rate 协议下，append 方法在 Cora/Citeseer 和 BookChild 的历史配额版结果中通常优于 real-node 对照；
- SportsFit 上主方法仍然落后或接近真实节点基线，说明短文本图上的优势尚未建立；
- rate 曲线非单调，不能宣称“memory 越大越好”；
- allocation 有明显数据集依赖，但其纯 allocation 因果关系必须等待受控重跑；
- 不能依据这 53 个结果宣称 SOTA，也不能把 `legacy_15` 解释成已经验证的通用动态分配方法。

## 6. 下一步验收顺序

1. 完成受控 Cora/Citeseer 5% allocation 重跑；
2. 在同一 matched-quota runner 上运行 fixed-total、hybrid、adaptive-rate；
3. 把每阶段 `keep/new/allocation`、student cost、old/current/cumulative F1 汇总到统一表；
4. 只有在受控结果成立后，才决定是否把 `sqrt+dispersion` 或其它策略提升为 v4 主方法；
5. 再进入 P2 文本压缩，不把预算、allocation 和文本机制同时改动。
