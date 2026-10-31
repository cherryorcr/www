# Student-Compatible Append 结果

日期：2026-09-14  
协议：`tagsam-unified-v3-2026-09-13`  
远程环境：`pc@REMOTE_SERVER_HOST`，conda `tagsam`

## 1. 方法实现

本方法保持历史 synthetic memory 的内容不可重写。每个阶段只读取当前 delta，并生成多个确定性 compact block 候选：

| 候选 | 聚类空间 | 文本候选数 | 字符预算 |
|---|---|---:|---:|
| `joint_g3_c192` | X+text joint | 3 | 192 |
| `joint_g5_c320` | X+text joint | 5 | 320 |
| `joint_g8_c512` | X+text joint | 8 | 512 |
| `xonly_g5_c320` | X-only | 5 | 320 |

候选筛选使用 `3 seeds × 3 batches (8/16/32)` 的短训网格，指标为：

- 当前阶段真实 dev 节点上的 Macro-F1，代表新类塑性；
- 历史阶段 dev Macro-F1 相对 reference 的变化，代表保持能力；
- seed/batch 间 current F1 方差，代表 student 可重训稳定性。

先计算三目标 Pareto front（新类 F1 越高、旧类保持越高、方差越低越好），再在旧类下降不超过 `2 pp` 的候选中用固定 robust utility 选择。最终 artifact 再用 6 个 canonical student seeds、batch 16 和协议步数复评。

## 2. Cora

预算为 `28/78/108`，旧行保留为 `0/28/63`，新增行分别为 `28/50/45`。

| 阶段 | 选择候选 | current F1 | cumulative F1 | student steps |
|---|---|---:|---:|---:|
| s0 | `joint_g8_c512` | 66.44 | 66.44 | 448 |
| s1 | `xonly_g5_c320` | 68.35 | **65.74** | 1248 |
| s2 | `xonly_g5_c320` | 46.37 | **52.03** | 1728 |

与同协议 fixed append 对照 `65.77/60.69/51.82` 相比，SCA 的 cumulative F1 差值为：

```text
s0 +0.67 pp, s1 +5.05 pp, s2 +0.21 pp
```

stage1 的提升在 selection grid 和 6-seed canonical 复评中都保留，说明“按 student response 选新增 block”确实有潜力改善新类塑性。

## 3. Citeseer

预算为 `44/98/127`，旧行保留为 `0/44/97`，新增行分别为 `44/54/30`。

| 阶段 | 选择候选 | current F1 | cumulative F1 | student steps |
|---|---|---:|---:|---:|
| s0 | `joint_g5_c320` | 61.25 | 61.25 | 704 |
| s1 | `joint_g5_c320` | 83.94 | **73.97** | 1568 |
| s2 | `joint_g8_c512` | 60.25 | **57.11** | 2032 |

与同协议 fixed append 对照 `61.25/73.97/58.89` 相比，SCA 的 cumulative F1 差值为：

```text
s0 +0.00 pp, s1 +0.00 pp, s2 -1.78 pp
```

因此 SCA 当前不能宣称跨数据集稳定增益。它改善了 Cora 的中期新类塑性，但在 Citeseer 最终阶段没有超过简单 append。

## 4. 当前判断

1. **创新点成立为可检验的方法机制。** Student response 可以作为新增 block 的选择信号，且候选确实会随数据集和阶段改变；它不是固定手工规则。
2. **性能证据还不够强。** Cora stage1 的 `+5.05 pp` 值得继续追踪，但 Citeseer s2 的 `-1.78 pp` 说明当前 selection utility、短训长度或候选空间仍不稳定。
3. **下一步应优先做归因，而不是扩充候选数量。** 重点检查：短训 ranking 与 canonical ranking 的一致性、batch-robustness 是否真的预测最终 F1、以及 stage2 新类预算过小导致的候选空间受限。
4. **论文表述可暂定为：** “student-compatible candidate selection improves the retention-plasticity trade-off in selected stages, with explicit cross-dataset limits”，而不是 SOTA。

## 5. 结果与审计文件

- Cora：`results/sca_v3_corrected_results_2026_09_14/cora/results.json`
- Citeseer：`results/sca_v3_corrected_results_2026_09_14/citeseer/results.json`
- runner：`code/frozen_v1_2026_09_09/run_student_compatible_append.py`
- 两套结果均记录 `test_accessed=false`、`test_source_sealed=true`、6 个 canonical seeds 和 batch 16。

首轮正式运行曾因 `len(selected_memory)` 误计字典字段数而只保留 6 个旧行，已标记为 invalid；本文件只引用 corrected rerun。

## 6. Canonical 全候选确认

为检查短训 selection 是否过拟合，已对每阶段全部 4 个 candidate memory 用 canonical 6-seed、batch 16 和完整 steps 重新评估。

| 数据集 | 阶段 | canonical cumulative 排名 | selector 选择 | 选择排名 |
|---|---:|---|---|---:|
| Cora | s0 | `joint_g8 > joint_g5 > joint_g3 > xonly` | `joint_g8` | 1 |
| Cora | s1 | `xonly > joint_g8 > joint_g5 > joint_g3` | `xonly` | 1 |
| Cora | s2 | `joint_g8 > joint_g5 > xonly > joint_g3` | `xonly` | 3 |
| Citeseer | s0 | `joint_g5 > joint_g3 > xonly > joint_g8` | `joint_g5` | 1 |
| Citeseer | s1 | `joint_g5 > xonly > joint_g8 > joint_g3` | `joint_g5` | 1 |
| Citeseer | s2 | `joint_g3 > xonly > joint_g5 > joint_g8` | `joint_g8` | 4 |

因此当前机制结论需要收紧：student-compatible selection 在 s0/s1 的 ranking 可迁移，但在 s2 两个数据集都失配。后续应优先解决后期小 delta 的 selection proxy 失配，例如增加后期 warm-up、采用 stage-specific selection budget，或把 canonical ranking consistency 纳入选择约束；不应通过继续扩大候选数量来掩盖该问题。

确认结果：

- `results/sca_v3_canonical_confirm_2026_09_14/cora/results.json`
- `results/sca_v3_canonical_confirm_2026_09_14/citeseer/results.json`
- runner：`code/frozen_v1_2026_09_09/run_student_compatible_canonical_sweep.py`
