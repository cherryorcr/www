# TGC-R14：基于 R11 的分歧加权新类塑性实验完成报告

日期：2026-09-24  
状态：**实验完成；发现了比 R11 更好的局部版本，但尚未通过主方法晋升门禁**

## 1. 结论先行

R14 没有重新设计 TGC。它保留 R11 的核心机制：

\[
P_{anchor}=0.75P_{initial}+0.25P_{joint}
\]

只将固定的全局 `alpha=0.25` 改为按当前新样本的 assignment 分歧调节：

\[
d_i=\frac{1}{2}\|P_{joint,i}-P_{initial,i}\|_1,
\qquad
\alpha_i=\operatorname{clip}(0.25+\beta d_i,0,0.75)
\]

\[
P_i=(1-\alpha_i)P_{initial,i}+\alpha_iP_{joint,i}
\]

然后仍用同一个 `P` 同步生成新节点 X、文本和拓扑。

结果说明：

- `plastic050` 最终阶段平均相对 C0 为 **+0.571 pp**，高于 R11 `anchor025` 的 `+0.142 pp`；
- `plastic050` 的最终 Cora 平均提升约 `+1.00 pp`，Citeseer 平均提升约 `+0.14 pp`；
- 但 `plastic050` 阶段平均仍为 `-0.644 pp`，最终新类 F1 仍为 `-2.185 pp`；
- 最差最终 cell 为 `-2.626 pp`，超过当前 `-2 pp` 门禁；
- `plastic025` 最终平均为 `-0.064 pp`，没有形成有效增益；
- 因此 R14 证明了“R11 上进行分歧加权”有局部收益，但**尚不能作为稳定新主方法**。

最准确的研究结论是：

> R11 的 assignment anchor 确实可以稳定一部分 joint 的旧类传播；对高分歧样本恢复 joint 信息可以提高部分最终 cell，但新类塑性瓶颈并不等价于 assignment 分歧本身。当前还存在内容/拓扑材料化与 student 长程响应之间的错位。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19601`、`order19602`、`order19603` |
| 阶段 | base + 2 incremental |
| memory | 5% growing synthetic memory |
| report seeds | `150..155` |
| student | 两层 384 维 GCN/GraphCLIP，冻结 MiniLM，双向 InfoNCE |
| batch/steps | batch=16；`max(50,min(2048,16*rows))` |
| 文本 | 最多 5 个来源、约 320 字符，拼接后重新编码 |
| 拓扑 | R11 相同的共享 assignment 投影二值拓扑 |
| 访问约束 | 旧 synthetic C0 memory + 当前 raw delta；历史 raw reads=0 |
| 选择 | 不使用 report F1 选择；所有 arm 完整报告 |
| 新增组件 | 只增加 row-wise disagreement-weighted alpha |

对照为：C0、R11 `anchor025`、封存 `joint`、`plastic025`、`plastic050`。

## 3. 汇总结果

数值为六个最终数据集/顺序 cell 的均值；差值相对同 cell 的 C0。

| 方法 | 最终 F1 | 最终相对 C0 | 最终 old F1 | 最终 new F1 | 阶段平均相对 C0 | 最差最终 cell | 正收益 final cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.514 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0/6 |
| R11 `anchor025` | 51.656 | +0.142 | +1.629 | -2.059 | -0.856 | -2.104 | 3/6 |
| joint | 51.463 | -0.051 | +1.342 | -1.917 | -0.920 | -5.043 | 4/6 |
| R14 `plastic025` | 51.450 | -0.064 | +1.593 | -2.525 | -0.697 | -3.294 | 2/6 |
| R14 `plastic050` | **52.086** | **+0.571** | **+2.381** | -2.185 | -0.644 | -2.626 | 3/6 |

R14 `plastic050` 的最终平均超过了预设的 `+0.5 pp` 条件，但不能只看这一列晋升，因为它同时没有通过阶段平均、新类 F1 和最差 cell 条件。

## 4. 最终阶段逐 cell 结果

括号为相对同一 cell C0 的差值，单位为百分点。

| 数据集 | 顺序 | C0 | R11 anchor025 | joint | plastic025 | plastic050 |
|---|---|---:|---:|---:|---:|---:|
| Cora | order19601 | 50.382 | 51.736 (+1.354) | 51.644 (+1.262) | 50.000 (-0.382) | 51.227 (+0.844) |
| Cora | order19602 | 53.048 | 52.037 (-1.011) | 48.005 (-5.043) | 49.754 (-3.294) | 50.422 (-2.626) |
| Cora | order19603 | 46.152 | 46.645 (+0.493) | 48.365 (+2.213) | 49.973 (+3.821) | **50.944 (+4.792)** |
| Citeseer | order19601 | 54.535 | 54.484 (-0.051) | 55.517 (+0.982) | 54.313 (-0.222) | 54.396 (-0.140) |
| Citeseer | order19602 | 52.035 | 54.205 (+2.169) | 54.405 (+2.369) | 53.583 (+1.548) | 54.114 (+2.079) |
| Citeseer | order19603 | 52.933 | 50.829 (-2.104) | 50.843 (-2.090) | 51.079 (-1.854) | 51.411 (-1.522) |

### 4.1 R14 的确改善了哪些 cell

- Cora/order19603：`plastic050` 相对 C0 `+4.792 pp`，相对 R11 再提升约 `+4.299 pp`；
- Cora/order19601：`plastic050` 保留了 R11 的大部分正收益，仍为 `+0.844 pp`；
- Citeseer/order19602：`plastic050` 保持 `+2.079 pp` 的正收益；
- Citeseer/order19603：R11 的 `-2.104 pp` 改善为 `-1.522 pp`，说明最差 cell 有所缓解，但仍未达到门禁。

### 4.2 R14 没有消除的坏 cell

- Cora/order19602 仍然低于 C0 `-2.626 pp`；
- Citeseer/order19601 仍低于 C0 `-0.140 pp`，且前一阶段下降更明显；
- 分歧加权没有把所有高风险顺序统一推向 C0 或 joint 的较优方向。

## 5. 对新类塑性的直接判断

R14 的核心目标是恢复新类信息，但新类结果没有改善到可晋升水平：

| 方法 | Cora 最终 new F1 差 | Citeseer 最终 new F1 差 | 总体 new F1 差 |
|---|---:|---:|---:|
| R11 `anchor025` | -2.817 | -1.302 | -2.059 |
| joint | -4.205 | +0.371 | -1.917 |
| plastic025 | -3.358 | -1.692 | -2.525 |
| plastic050 | -2.460 | -1.910 | -2.185 |

`plastic050` 相对 R11 将 old F1 差从 `+1.629` 提高到 `+2.381`，但 new F1 差从 `-2.059` 仍保持在 `-2.185`。这说明其最终 F1 增益主要来自旧类保持和顺序波动，而不是新类塑性恢复。

## 6. 为什么这个方向只部分有效

### 6.1 assignment 分歧不是新类有效性的充分统计量

R14 假设：`P_joint` 与 `P_initial` 分歧大的当前样本更值得保留 joint 信息。但结果显示，高分歧样本不一定对应最终 student 能正确学习的新类结构。teacher joint 的变化可能包含：

- 旧块消息传播的补偿；
- 共同表示空间中的局部重排；
- 对 teacher proxy loss 有利、但对从头训练 student 不利的 assignment 改变。

因此分歧只能表示“teacher 与 C0 看法不同”，不能直接表示“该样本对新类 student 有用”。

### 6.2 仍然是 X、文本和拓扑的联合材料化问题

R14 保持同一个 assignment 生成三种模态，避免了显式的 X/text/topology 不一致；但它仍然继承了 R11 的三个限制：

1. synthetic X 仍是 assignment 加权均值；
2. 文本仍是 top-5/320-character concat-reencode；
3. 拓扑仍是 coarse support 投影后选边。

如果新类信息在这三个材料化步骤中被共同压缩，单独调整 `alpha_i` 无法恢复丢失的学习信号。

### 6.3 新类与旧类的传播目标相冲突

R14 更强地保留 joint assignment 后，old F1 继续提高，但新 F1 没有同步提高。这说明当前 joint assignment 可能优先修正旧块传播或全局结构，而不是为当前新类提供可分离的局部信号。

## 7. 晋升判定

| 条件 | 要求 | plastic050 | 判定 |
|---|---:|---:|---|
| 最终平均相对 C0 | `>= +0.50 pp` | `+0.571` | 通过 |
| 阶段平均 | `>= 0` | `-0.644` | 未通过 |
| new F1 平均 | `>= 0` | `-2.185` | 未通过 |
| 最差最终 cell | `>= -2 pp` | `-2.626` | 未通过 |
| 多顺序正收益 | 至少 4/6 | 3/6 | 未通过 |
| 历史访问与审计 | 必须通过 | 6/6 audit 通过 | 通过 |

结论：`plastic050` 是**比 R11 更好的局部候选**，但不是已经成立的新主方法。不能因为最终平均刚好超过 `+0.5 pp` 就忽略阶段和 new F1 退化。

## 8. 下一步：仍基于 R11，但改变实验对象

R14 说明继续扫描更多 `beta` 的价值有限。下一步不再把 assignment 分歧直接当作 new-class utility，而是做一个受控的 R15 归因：

### R15-A：R11 内容与 C0 拓扑交叉对照

固定 R11 `P_anchor` 生成 X/text，但把新块拓扑替换为 C0 拓扑；以及固定 R11 X/text，只替换为 C0 内容。目的不是直接发明新算法，而是回答：

- R11 的 old retention 是否主要来自 topology；
- R11 的 new plasticity 损失是否来自 assignment 投影拓扑；
- 是否需要把 R11 的“稳定 assignment”与 C0 的“可靠 topology”组合起来。

### R15-B：保留 R11 anchor 的新类局部残差

在 R11 `P_anchor` 不变的情况下，只对当前新块的 synthetic X 做 class-conditional residual expansion：保持每类均值不变，适度恢复类内方差；文本和拓扑仍与 `P_anchor` 配对。该实验应使用两个固定强度，并与 R11/C0 配对比较。

### R15 的停止条件

如果 R15 仍然只能改善旧类而不能改善 new F1，说明瓶颈已不在 R11 的 anchor，而在当前 downstream student 的图文训练目标和 synthetic topology 表达能力。此时应停止继续调 R11 系数，保留：

- C0 作为可靠主方法；
- R11/R14 作为“稳定化有效但塑性不足”的机制证据；
- 新类塑性负结果作为论文贡献的一部分。

## 9. 产物与复现

- 协议：[TGC_R14_R11_PLASTICITY_PROTOCOL_ZH_2026-09-24.md](TGC_R14_R11_PLASTICITY_PROTOCOL_ZH_2026-09-24.md)
- runner：[run_tgc_r14_r11_plasticity.py](../../code/v2_llm_fusion/run_tgc_r14_r11_plasticity.py)
- summarizer：[summarize_tgc_r14.py](../../tools/summarize_tgc_r14.py)
- 结果：[tgc_r14_r11_plasticity_20260924b](../../results/tgc_r14_r11_plasticity_20260924b/)
- 远程快照：`work_runs/tgc_r14_r11_plasticity_20260924_snapshot`
- 远程 pipeline：`prepare -> seal -> evaluate -> audit` 全部返回 0
