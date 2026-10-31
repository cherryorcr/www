# TGC-R3：独立顺序与从头训练响应选择器冻结协议

日期：2026-09-22。状态：运行前冻结；本文件不是结果报告。

## 1. 本轮问题

R2B2 的 `warm-up=64 + suffix=1` 在开发条件上有正向均值，但旧 gate 未能保留收益。本轮固定该候选生成器，只检验新顺序、新元优化种子、新 report seeds 下的候选有效性与无标签选择可靠性。不同时修改预算、合成文本、拓扑或新 block。

主策略预先指定为 `selected_curve`。其余 selector 是诊断对照，不得在读完 report 后改选最优策略或按 cell 拼表。

## 2. 数据与独立性范围

| 数据集 | 顺序 | 三阶段类别组 |
|---|---|---|
| Cora | order19221 | [0,5] / [3,4] / [1,2,6] |
| Cora | order19222 | [5,6] / [3,4] / [0,1,2] |
| Citeseer | order19221 | [0,3] / [2,4] / [1,5] |
| Citeseer | order19222 | [3,4] / [2,5] / [0,1] |

- 元优化 train seeds：7800、7801；validation seeds：7810、7811。
- 从头训练 selector seeds：7820、7821、7822；probe 批次随机种子：7830 + stage。
- 正式 report seeds：9600--9605，共六个。
- 数据集、合法 pool 和 report-dev 划分沿用现有项目；这是新顺序/种子确认，不是新数据集或全新数据划分。
- C0 是确定性构造；没有声称新增独立 clustering seed。teacher seed 9000 与既有算法保持一致。

## 3. 不变项

每阶段 memory 大小为 `floor(0.05 * 累计合法 pool 节点数)`，全部旧行保留，新预算为相邻阶段之差。每个新顺序从零重建 C0，禁止借用 order19102 的旧 memory。

C0 使用 dispersion 类预算、联合图文聚类、mean-X、g5/c320 concat 后冻结 MiniLM 重编码、frozen-old 追加拓扑。临时与正式 student 都是现有两层 384 维 GCN + text projection，双向图文 InfoNCE，batch=16、Adam lr=0.001。正式训练步数为 `max(50,min(2048,16*memory_rows))`。

候选固定 R2B2：C0 上先训练 64 步；固定并 detach 该前缀；只穿过 1 步 Adam suffix 反传到允许修订的历史 X。outer steps=16、lr=0.01、相对半径=0.1。80% 旧行受保护，最多 10% 旧行可编辑；行选择仍由既有 teacher 规则确定，若 teacher 无接受行则取预定 eligible 顺序的 quota。新 block、全部文本、拓扑、row order、rows、mass、depth、birth stage 固定。

每个 s1/s2 都从对应 C0 父 memory 起步，不继承上一阶段修订。这是单阶段机制确认，不是递归 continual rollout。s0 各臂相同。

## 4. 选择器

每个 selector seed 分别在 C0/candidate memory 上从随机初始化训练到 128 步；只在同一条轨迹内部复用 32/64/128 步前缀。两臂有相同初始参数与 minibatch 行序列。它不复用候选生成器的 64 步模型。

probe 只含旧 synthetic memory 与当前 raw delta，图文对齐无分类标签。每类 probe 取四批；old/new 各最多 16 行，mixed 每批 8 old + 8 new，mixed InfoNCE 包含新旧交叉负例。

| 策略 | 信号 | 接受条件 |
|---|---|---|
| selected_gate | 原 R2B2 validation suffix 的 new/old loss | 两个 validation seed 都满足 new 改善至少 1e-4、old 恶化不超过 0.01 |
| selected_new64 | 从头训练 64 步后的 new probe gain | 平均改善至少 1e-4；至少 2/3 seeds 改善至少 1e-4；每个 seed 的 old loss 恶化不超过 0.01 |
| selected_mixed64 | 从头训练 64 步后的 mixed probe gain | 同上 |
| selected_curve（主策略） | 每 seed 的 32/64/128 mixed gain 等权均值 | 均值与 2/3 seed 条件同上；每个 seed、每个 checkpoint 均通过 old loss 保护 |

所有 selector 在 C0 与同一个 candidate 之间二选一。probe 数据与候选生成共享合法数据来源，不将其称为独立 held-out 数据。旧 synthetic probe 上的保护不等于历史真实分布风险保证。

## 5. 对照、时序与审计

九臂：C0、teacher、candidate、等幅随机位移 random、stop-gradient、四种 selected 策略。共 4 cells × 3 stages × 9 arms × 6 seeds = 648 记录槽位。最多新训 216 个正式 student；相同 memory 的策略/stop/teacher 直接复用同 seed 源 checkpoint，不额外重训。

先运行所有 cell 的 `prepare`，保存 memory、prefix optimizer/RNG、selector 三个 checkpoint 状态、batch 序列、probe loss 和选择决策。全矩阵 `seal` 后才允许任何 report fit。报告阶段不得改变 selector 或阈值。最终指标仍是 report-dev 的节点分类 Macro-F1，同时分开报告 all-prompt 与 seen-prompt 的 old/new 指标，test 不评估。

审计重建合法 delta/support；验证所有不变字段、修订半径、stop 无操作、random 等幅；重放 scratch selector 轨迹、核对无标签选择；检查 seal 时间与哈希；重载正式 student，以独立 TP/FP/FN 实现重算 report 指标。逐 cell-阶段-方法只能汇总一次，每组严格六个配对 seeds。

## 6. 决策门槛与成本

继续递归 rollout 的必要门槛，全部对主策略检查：s2 四 cell 等权平均至少 +0.5 pp；至少 3/4 cells 不低于 -1 pp；最差 cell 不低于 -2 pp；s1 平均不下降；s2 新类平均不下降、最差旧类下降不超过 2 pp；s2 平均优于 random 且不低于 teacher。

达到门槛也只是进入下一轮，不自动宣布主方法/SOTA。其余 selector 的最佳结果不得代替主策略通过验收。

报告候选收益、各策略接受率、按 cell 均值符号定义的误接受/误拒绝、配对六种子 t 区间、二选一 hindsight regret。后者是事后诊断，不是可部署 oracle。

成本分开记录 C0 构造、teacher、64 步前缀、元优化、selector、stop 控制、正式 student 与审计。selector 三策略共享轨迹，整个矩阵只计一次实际成本；单独部署 64 步策略只需两臂 × 三 seeds × 64 步，curve 需要 768 步。明确排除模型加载/数据编码等计时范围，不能把已有缓存当成零成本算法。

若主策略或候选不能通过独立确认，本轮停止扩展递归、大图或联合文本/拓扑修订，保留 C0 并记录失败边界。

## 7. 入口

- `code/v2_llm_fusion/run_tgc_selector_confirmation.py`
- `code/v2_llm_fusion/tgc_selector_confirmation.py`
- `code/v2_llm_fusion/audit_tgc_selector_confirmation.py`
- `code/v2_llm_fusion/summarize_tgc_selector_confirmation.py`
- `code/v2_llm_fusion/launch_tgc_r3_selector_20260922.sh`

远程使用独立代码快照与 conda `tagsam`；结果目录：`results/tgc_r3_selector_confirmation_20260922`。凭证不落盘。
