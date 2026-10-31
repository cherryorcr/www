# P1 执行交接报告（给 ChatGPT 继续推进 P1-P5）

日期：2026-09-15
执行环境：远程服务器 `pc@REMOTE_SERVER_HOST`（SSH 免密已配置），conda 环境 `/home/pc/miniconda3/envs/tagsam/bin/python`，2× RTX 4090
工作目录：`/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/`
本报告接续 `FUTURE_METHOD_OPTIMIZATION_ROADMAP_ZH_2026-09-15.md`（P0 已完成），汇报 P1 的执行状态、全部数据与判定，并列出剩余工作。

---

## 一、执行状态总览

| 项 | 状态 |
|---|---|
| P1 第一轮（cora/citeseer，rate sweep × allocation sweep，30 实验） | ✅ 完成，全部 exit 0 |
| P1 round 3 候选（`dispersion_floor` 组合规则，2 实验） | ✅ 完成，**判定为失败**（见 §四） |
| P1 第二轮（bookchild/sportsfit/arxiv，21 实验） | ⏳ **进行中**（截稿时 6/21 完成，两条 nohup 队列会自行跑完，无需干预） |
| P1 剩余（hybrid / adaptive-rate 预算模式、fixed-total、多子簇、成本曲线） | ❌ 未开始 |
| P2–P5 | ❌ 未开始 |

**第二轮队列自查方式**（跑完后所有 `results.json` 会在 `p1_results/` 下）：

```bash
ssh pc@REMOTE_SERVER_HOST "cd /home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13 && grep -c done logs/queue_r2_a.log logs/queue_r2_b.log; find p1_results -name results.json | wc -l"
# 全部完成时总数应为 30(round1) + 2(round3) + 21(round2) = 53
```

## 二、实验设置（所有实验统一）

- Runner：`run_unified_v3_continual.py`（P0 产物；manifest 完整、batch=16、6 student seeds 100–105、pool-only 文本缓存、test 封存未读）
- 协议：3 阶段类别增量，fixed-rate 预算，dev 分区报告（**所有数字都是 dev，test 从未读取**）
- Arms：`append_fixed`（主方法，synthetic-only）+ `k_center`、`random_real`（真实节点对照）
- 主指标：s2 cumulative dev Macro-F1（附 Accuracy、old/cur 分解）

## 三、全部已完成数据

### 3.1 第一轮 · rate sweep（allocation=legacy_15）

| dataset | rate | append_fixed（F1/Acc/old/cur） | k_center | random_real |
|---|---|---|---|---|
| cora | 0.5% | **45.22** / 58.98 / 73.7 / 29.0 | 20.64 | 40.24 |
| cora | 1% | **42.29** / 53.44 / 70.1 / 32.6 | 24.75 | 34.27 |
| cora | 5% | **57.87** / 62.98 / 76.3 / 57.7 | 50.99 | 45.56 |
| citeseer | 0.5% | **54.60** / 55.33 / 63.6 / 69.3 | 48.18 | 28.75 |
| citeseer | 1% | **58.94** / 63.17 / 74.3 / 61.0 | 45.57 | 45.37 |
| citeseer | 5% | 53.79 / 57.58 / 72.1 / 47.5 | 51.47 | 47.84 |

**判定 1**：修正协议（batch=16）下，`append_fixed` 在 cora/citeseer 全部 6 个格点上优于两个真实节点基线——此前"k-center 更强"确认为 batch 混淆伪影（见 `BATCH_SIZE_CONFOUND_ZH_2026-09-11.md`）。

**判定 2**：rate 曲线非单调且跨数据集形状不同：citeseer 峰值在 **1%**（58.94 > 5% 的 53.79）；cora 在 1% 有低谷、5% 最高。"更多预算更好"不成立。

### 3.2 第一轮 · allocation 对比（rate=5%，matched memory，rows 完全相同）

| dataset | legacy_15 | equal | linear | sqrt | dispersion |
|---|---|---|---|---|---|
| cora | **57.87** | 45.50 | 44.52 | 47.28 | 49.85 |
| citeseer | 53.79 | 56.70 | 58.52 | 57.75 | **59.17** |

old/cur 分解（5%）：

| dataset | alloc | cum | old | cur |
|---|---|---:|---:|---:|
| citeseer | dispersion | **59.17** | **76.7** | **55.5**（对 legacy_15 是 **Pareto 改进**：+4.6/+7.9） |
| cora | legacy_15 | **57.87** | 76.3 | **57.7**（equal 的 cur 崩到 29.3） |

**判定 3**：低 rate（0.5%/1%）下各 allocation 大多重合（预算太小，floor 效应），无信号。

**判定 4（P1 最重要结论）**：5% 下 allocation 效应巨大（cora 上极差 12.4 分）但**方向跨数据集翻转**——citeseer 上 dispersion 是 Pareto 最优，cora 上 legacy_15 全面最优。机制解读：cora s2 有 3 个新类且类不均衡，legacy_15 的每类 15 行下限保护了新类塑性；citeseer 类均衡（2 个新类），下限浪费容量、按离散度分配更优。**allocation 的正确选择取决于新类数量与类别均衡度，不存在单一最优策略**——与本项目此前"四次判据证伪"的模式一致。

### 3.3 Round 3 · `dispersion_floor` 组合规则（dispersion 权重 + 每类 15 行下限）

| dataset | dispersion_floor | vs legacy_15 | vs dispersion |
|---|---:|---|---|
| cora（5%） | 55.52 | **−2.35** | +5.67 |
| citeseer（5%） | 53.79 | ±0.00（数值完全相同） | **−5.38** |

**判定 5：组合规则失败。** 原因是结构性的：当下限接近吃满新块预算时（cora s2：15×3=45=全部预算；citeseer s2：30/33），dispersion 权重只能分配残余几行，行为退化为 legacy_15。**"下限保护 + 离散度分配"不能通过简单叠加获得两者优点。** 该规则已实现为 runner 的 `--allocation-mode dispersion_floor`（代码已保留，备份 `run_unified_v3_continual.py.bak_p1r3`），可作 negative control 引用。

### 3.4 第二轮（大图）已完成部分 · rate sweep（legacy_15）

| dataset | rate | append_fixed（F1/Acc/old/cur） | k_center | random_real |
|---|---|---|---|---|
| bookchild | 0.5% | **11.94** / 13.55 / 16.1 / 27.2 | 8.83 | 10.51 |
| bookchild | 1% | **13.82** / 14.43 / 18.2 / 26.9 | 11.73 | 12.01 |
| bookchild | 5% | **15.59** / 16.37 / 20.4 / 28.4 | 14.24 | 13.24 |
| sportsfit | 0.5% | 28.74 / 37.56 / 34.2 / 57.2 | **30.44** | 28.89 |
| sportsfit | 1% | 25.84 / 35.67 / 32.3 / 46.1 | 31.25 | **32.31** |
| sportsfit | 5% | 29.52 / 34.68 / 36.5 / 60.6 | **30.00** | 27.04 |

**判定 6（初步，需等第二轮跑完再定稿）**：大图上结论分裂——bookchild 上 `append_fixed` 全 rate 领先且随 rate 单调上升；**sportsfit（短文本商品图）上主方法在 3 个 rate 中 3 次输给真实节点基线**（0.5% 差 −1.7、1% 差 −6.5、5% 差 −0.5）。**主方法的优势在长/中文本数据集上成立，在短文本图上不成立**——这是必须写进论文边界的发现。arxiv 与大图 allocation 对比仍在队列中。

## 四、给 ChatGPT 的待办清单（按优先级）

### P1 收尾
1. **等第二轮 21 个实验跑完**（自动），跑 `/tmp/p1_final_agg.py`（已在服务器）聚合全表；重点看：(a) arxiv 上主方法 vs 基线；(b) 大图上 allocation 方向是否也翻转；(c) sportsfit 劣势是否在 allocation 变体下缓解；
2. **P1 剩余协议**：`--budget-mode fixed-total / hybrid / adaptive-rate`（runner 已支持，CLI 开关就绪，只需按第一轮同样的队列模式跑）；adaptive-rate 需检查 `stage_utilities`（第一轮日志中为 null，说明 U_t 计算只在 adaptive 模式下激活，需先冒烟验证）；
3. **成本曲线**：manifest 已记录 `student_train_seconds`、`total_student_optimizer_steps`，直接聚合即可，无需重跑；
4. **多子簇 synthetic node**（roadmap §5.2）未实现，需要改 `build_stage_block`。

### P2–P5（未开始）
- P2 文本压缩：runner 已有 `--text-operator {concat,mean_pool,weighted_pool,set_pool}` 开关，但 diversity-aware extractive 与 dual prototype 未实现；注意历史结论：`no_reencode`（平均 embedding）在 5/6 数据集上显著劣于重编码（最多 −12.5），mean_pool 预计复现该结果；
- P3 SCA：`--arms sca` 与 selection-seeds/batches 开关已存在，未在 P1 中运行；
- P4 memory 管理：`--refresh-mode bounded_refresh` 与 `--refresh-ratio/alpha` 开关已存在，未运行；split/merge/evict 未实现；
- P5 多阶段：`--n-stages` 与 `--split-mode base_incremental` 开关已存在，未运行 base+4 / 5/8-stage。

## 五、必须知道的历史坑（避免重蹈）

1. **batch 是 20+ 分的混淆因素**（小图最优 16，`run_multiscale_asm.py` 旧默认 64 曾污染大量结论）——P0 已把协议 batch 定为 16，但**大图的最优 batch 未系统扫过**（sportsfit 旧扫描显示 ≤138 区间平坦），若 sportsfit 劣势要归因，先排除 batch；
2. **单 seed / 单 draw 的方向性结论在本项目已三次被推翻**——任何新结论先过 6 seeds + 配对 bootstrap 再写入文档；
3. **test 分区从未读取**（`test_accessed: false` 全局成立），读 test 是一次性动作，必须等方法完全冻结；Cora/Citeseer 的旧 test（`%10==2`）已在 frozen_v1 时代消耗过，unified v3 用的是新的 report-dev/test 分离（见 protocol_v3.py）；
4. **`FUSE_CHARS` 直觉陷阱**：曾诊断"320 字符截断是缺陷"，实验证明加长文本反而更差（citeseer 47.6→31.4）——文本长度的方向与直觉相反，改文本方案前先跑小对照；
5. 服务器 `DATA_CFG`（v2_codex 的旧 runner）只有 cora/citeseer/arxiv；**大图数据集只在 unified v3 runner 中注册**（bookchild/bookhis/sportsfit/wikics，数据在 `/home/pc/research/TAGSAM/data/tagbench/`）。

## 六、资源索引

| 资源 | 位置 |
|---|---|
| 主 runner（P0 产物，已含 dispersion_floor 补丁） | `graphclip_unified_v3_2026_09_13/run_unified_v3_continual.py`（补丁前备份 `.bak_p1r3`） |
| 协议常量 | 同目录 `protocol_v3.py` |
| P1 全部结果 | 同目录 `p1_results/p1_<ds>_r<rate>_<alloc>/results.json`（manifest 内含预算、rows、steps、seeds、hash） |
| 队列与实验日志 | 同目录 `logs/`（`queue_cora/citeseer/r2_a/r2_b.log` + 每实验一个 log） |
| 启动脚本（可复用改造） | 同目录 `launch_p1_experiments.sh`、`launch_p1_round2.sh` |
| 聚合脚本 | 服务器 `/tmp/p1_final_agg.py`（建议移入代码目录固化） |
| 本地文档索引 | `D:\Aprojects\continue_tag_condensation\docs\INDEX.md`（本报告与第一轮文档已登记） |
| 历史结论文档 | `docs/current/`：`BATCH_SIZE_CONFOUND`、`BATCH_CONTROLLED_RERUN`、`GATE_THIRD_FALSIFICATION`、`P1_ROUND1_BUDGET_ALLOCATION` 等 |

## 七、一句话总结

P1 第一轮 + round 3 完成、第二轮过半：**主方法在修正协议下于 cora/citeseer/bookchild 全面胜过真实节点基线**（此前的反向结论是 batch 伪影）；**allocation 策略效应大但方向数据集相关**（citeseer: dispersion Pareto 最优 / cora: legacy_15 最优），**简单组合规则（dispersion+floor）已被证伪**；**sportsfit（短文本）上主方法暂居劣势**，是当前最需要归因的开放问题。剩余：第二轮收尾、adaptive-rate/hybrid 预算、以及 P2–P5 全部。
