# batch 受控重跑：在各自最优配置下，本方法重新领先

日期：2026-09-11
数据：`problem_validation_results/BCB_{cora,citeseer}`（本地镜像 `docs/current/final_2026_09_11/bcb_*.json`）
代码：`run_batch_controlled_baselines.py`
前置：`BATCH_SIZE_CONFOUND_ZH_2026-09-11.md`（batch 是 20+ 分的混淆因素）

---

## 1. 逐 batch 的完整结果（Cora，k=108，seeds 700–702）

| batch | **ours** | kcenter | random_real |
|---:|---:|---:|---:|
| 8 | 51.82 | 47.56 | **52.13** |
| **16** | **55.14** | 50.78 | 45.59 |
| 24 | 47.10 | 45.77 | **47.67** |
| 32 | 42.33 | **48.53** | 47.89 |
| 48 | 36.99 | **47.87** | 40.59 |
| 64 | 32.10 | **48.64** | 37.43 |

## 2. 逐 batch 的完整结果（Citeseer，k=127）

| batch | **ours** | kcenter | random_real |
|---:|---:|---:|---:|
| 8 | **56.77** | 54.68 | 55.37 |
| **16** | **58.09** | 53.85 | 50.64 |
| 24 | 53.48 | **53.90** | 45.89 |
| 32 | 51.96 | **52.23** | 37.18 |
| 48 | 46.85 | **47.84** | 28.96 |
| 64 | **48.15** | 47.53 | 27.69 |

## 3. 核心结论：**在各自最优 batch 下，本方法重新领先**

| 数据集 | arm | 最优 batch | **最优 F1** | 最优 Acc |
|---|---|---:|---:|---:|
| Cora | **ours** | 16 | **55.14** | **58.30** |
| Cora | random_real | 8 | 52.13 | 57.07 |
| Cora | kcenter | 16 | 50.78 | 56.70 |
| Citeseer | **ours** | 16 | **58.09** | **62.28** |
| Citeseer | random_real | 8 | 55.37 | 60.08 |
| Citeseer | kcenter | 8 | 54.68 | 62.17 |

**在公平调参下（每个 arm 各自取最优 batch）：**
- Cora：ours **55.14** > random_real 52.13 > kcenter 50.78（**+3.01 / +4.36**）
- Citeseer：ours **58.09** > random_real 55.37 > kcenter 54.68（**+2.72 / +3.41**）

**"k-center 优于本方法"的结论被推翻。** 之前的差距（Cora 33.30 vs 50.37）是 **batch 不对称**造成的伪影：`run_multiscale_asm` 用 64（对 k=108 灾难性），而基线脚本用 24。

## 4. 但必须同时报告的第二个事实：**排序对 batch 极度敏感**

| batch | Cora 排名 | Citeseer 排名 |
|---:|---|---|
| 8 | random > **ours** > kcenter | **ours** > random > kcenter |
| 16 | **ours** > kcenter > random | **ours** > kcenter > random |
| 24 | random > **ours** > kcenter | kcenter > **ours** > random |
| 32 | kcenter > random > **ours** | kcenter > **ours** > random |
| 48 | kcenter > **ours** > random | kcenter > **ours** > random |
| 64 | kcenter > **ours** > random | **ours** > kcenter > random |

**排名在 6 个 batch 设定下翻转了 5 次。** 而且**本方法对 batch 最敏感**（Cora：55.14 → 32.10，摆动 23 分），两个基线对 batch 的敏感度明显更低（kcenter 47.6–50.8，摆动 3.2 分；random_real 37.4–52.1，摆动 14.7 分）。

**这意味着两件事：**

**（a）本方法的"最优性能更高"是真的，但"对超参更脆弱"也是真的。** kcenter 在全 batch 范围内稳定在 47–51，而 ours 在 32–55 之间摆动。**一个需要精确调参才能领先的方法，在实用性和可信度上都弱于一个处处都还行的基线。**

**（b）此前所有"arm 间比较"都不可信——包括正面的。** 我们没有理由认为 batch=16 对其它数据集也是最优（sportsfit 的最优 ≤138，与 Cora 的 16 相差 8 倍），所以**"ours 最优 55.14"这个数字本身也不能代表方法在合理配置下的表现**。

## 5. 对论文的最终判断

**（a）可以恢复的主张（需在最优 batch 下重新验证）**：本方法在 Cora/Citeseer 上的最优表现优于 k-center 与 random 基线（+2.7 ~ +4.4）。

**（b）必须同时报告的限制**：本方法对 batch 的敏感度**显著高于**基线（23 分 vs 3 分），这是方法的真实弱点，不能隐瞒。**在论文中应当诚实写成"本方法在调优后领先，但对训练 batch 更敏感，这是一个需要解决的问题"。**

**（c）最紧急的修正**：`run_multiscale_asm.py` 的 `batch=64` 默认值必须修掉，并把**所有大 k 数据集上的实验**用合理 batch 重跑（大 k 上 64 接近最优，所以影响小；但**不能假定**，必须验证）。

**（d）需要重新审视的实验清单**（因为 batch 会改变结论）：
1. ASM 全部实验（在 batch=64 下做的，而 Cora/Citeseer 上这个值灾难性）→ **ASM 在小图上的"有害"结论很可能是 batch 伪影**；
2. 扰动模块的全部格点（batch=24）→ 需在 16 复核；
3. scale-gate 的规模门槛（batch=64）→ Cora 级别的 k 受严重影响；
4. **`FUSE_CHARS`、`合成 X`、`覆盖性` 三个证伪实验**（batch=24）→ 需复核，因为它们都是在非最优配置下做的。

**（e）一个重要的结构性发现**：**本方法的脆弱性有机制解释**。合成行是"簇均值"——分布被平滑、行间差异小，因此**需要小 batch 才能形成有效负例**（小 batch 时同 batch 内的行差异相对更大）。真实节点行的分布天然分散，对 batch 不敏感。**"簇均值 X 是弱点"这个假设最初被 `XSR` 实验证伪，但在 batch 敏感的框架下重新成立**——它不是"均值 vs medoid"的问题，而是"平滑分布需要不同的训练配置"的问题。

## 6. 立即待办

1. **修 `run_multiscale_asm.py` 的默认 batch**，改为显式参数并默认 16；
2. **重跑 ASM（Cora/Citeseer，batch=16）**——检验"ASM 有害"是否为伪影；
3. **重跑扰动格点（batch=16）**——检验不稳定性是否变化；
4. 在 bookhis/bookchild/sportsfit/wikics 上各扫一次 batch，确认 64 是否确实接近最优；
5. 全部完成后，重建基线主表与扰动表。
