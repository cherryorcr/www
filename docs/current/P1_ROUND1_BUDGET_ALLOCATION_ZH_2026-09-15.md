# P1 第一轮结果：预算 sweep 与 rows allocation（Cora/Citeseer）

日期：2026-09-15
数据：服务器 `graphclip_unified_v3_2026_09_13/p1_results/`（30 个实验，全部 exit 0）
协议：unified v3（P0 基础设施：batch=16、6 student seeds、pool-only 文本缓存、manifest 完整）
Arms：`append_fixed`（主方法）+ `k_center`/`random_real`（真实节点对照）

## 1. 表 1 · 压缩率 sweep（allocation=legacy_15，s2 cumulative dev Macro-F1）

| dataset | rate | append_fixed | k_center | random_real |
|---|---|---|---|---|
| citeseer | 0.5% | **54.60±1.66** (r11) | 48.18 | 28.75 |
| citeseer | 1% | **58.94±0.84** (r22) | 45.57 | 45.37 |
| citeseer | 5% | **53.79±1.28** (r111) | 51.47 | 47.84 |
| cora | 0.5% | **45.22±1.73** (r9) | 20.64 | 40.24 |
| cora | 1% | **42.29±1.68** (r18) | 24.75 | 34.27 |
| cora | 5% | **57.87±1.95** (r94) | 50.99 | 45.56 |

**主结论 1：在 P0 修正后的协议（batch=16）下，`append_fixed` 在全部 6 个 (dataset, rate) 格点上都优于两个真实节点基线。** 这与 `BATCH_CONTROLLED_RERUN` 的结论一致，且现在是在**持续三阶段协议**下的确认（此前只有静态单阶段）。k-center 之前的"优势"确认为 batch 混淆的伪影。

**主结论 2：rate 曲线非单调且跨数据集形状不同。** citeseer 峰值在 1%（58.94，**高于 5% 的 53.79**）；cora 在 1% 处有低谷（42.29）、5% 最高（57.87）。**"更多预算更好"在 citeseer 上不成立**——1% 的 22 行比 5% 的 111 行好 5 分。这对论文是一个可报告的观察：压缩率与效用的关系是数据集形状相关的。

## 2. 表 2 · rows allocation 对比（append_fixed，matched memory）

| dataset | rate | legacy_15 | equal | linear | sqrt | dispersion |
|---|---|---|---|---|---|---|
| citeseer | 0.5% | 54.60* | 54.60* | 52.96 | 52.96 | 54.25 |
| citeseer | 1% | 58.94* | 58.94* | 58.94* | 58.94* | 52.20 |
| citeseer | 5% | 53.79 | 56.70 | 58.52 | 57.75 | **59.17*** |
| cora | 0.5% | 45.22*（全同） | — | — | — | — |
| cora | 1% | 42.29 | 42.29 | **46.20*** | 39.64 | 42.00 |
| cora | 5% | **57.87*** | 45.50 | 44.52 | 47.28 | 49.85 |

（* 为该行最优；rows 完全一致，满足 matched-memory 验收条件）

**主结论 3：低 rate 下 allocation 无信号**（预算太小，各策略分配结果重合——floor 效应）。

**主结论 4（最重要）：5% 下 allocation 效应巨大但方向翻转。**
- **citeseer：dispersion 最优（59.17），比 legacy_15 高 +5.38**；
- **cora：legacy_15 最优（57.87），所有替代策略都差很多（equal 崩到 45.50，−12.37）**。

## 3. 表 3 · 保持/塑性分解（rate=5%）

| dataset | alloc | cum | old | cur |
|---|---|---:|---:|---:|
| citeseer | legacy_15 | 53.79 | 72.05 | 47.55 |
| citeseer | **dispersion** | **59.17** | **76.67** | **55.48** |
| cora | **legacy_15** | **57.87** | **76.34** | **57.68** |
| cora | equal | 45.50 | 74.96 | **29.29** |

**主结论 5：citeseer 上 dispersion 是 Pareto 改进**（old 和 cur 同时更好，+4.6/+7.9）——这满足 P1 验收条件"matched memory 下提升"。**但 cora 上方向相反**：legacy_15 的每类 15 行下限保护了 3 个新类的塑性（cur 57.68），而 equal 按类均分让新类 cur 崩到 29.29。

**机制解读（初步）**：cora 的 s2 有 **3 个新类**、citeseer 只有 2 个且类规模更均衡。legacy_15 的下限本质是"新类塑性保护"，在新类多、类不均衡时至关重要（cora）；在类均衡时反而浪费容量（citeseer），此时按离散度分配（dispersion）更优。**这不是"哪个策略更好"，而是"策略应答的是不同的失衡形态"。**

## 4. 对 P1 验收的判定

- ✅ **matched memory 下有提升的策略存在**（citeseer dispersion +5.38，Pareto）；
- ❌ **但没有单一策略跨数据集获胜**——与本项目此前四次判据证伪的模式完全一致；
- **诚实的表述**：allocation 的正确选择取决于新类数量与类别均衡度。一个可检验的组合规则是"**dispersion 分配 + 每新类最低配额下限**"（把 legacy_15 的保护和 dispersion 的效率合并），值得作为 round 3 验证。

## 5. 待办

1. 第二轮：bookchild / sportsfit / arxiv（已启动）；
2. Round 3 候选：`dispersion + per-class floor` 组合规则；
3. hybrid / adaptive-rate 预算模式（P1 剩余部分，需 U_t 指标）。
