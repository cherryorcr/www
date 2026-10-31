# TGC-R2 容量扫描、response 排序与 gate 因果诊断

日期：2026-09-22。

## 结论与完成边界

本轮实际完成容量扫描、独立 seed 的 response 离线诊断、共享 teacher 的 gate 回放及 checkpoint 预测重载，不再停留在参数化准备。

- 新增 **336 次 canonical student 训练**，复用 672 个相同训练记录槽位，共核验 1008 条记录。
- **144 条 selection-student 轨迹**，另有 8 条无中间探测的前缀验证轨迹；selection seeds 700/701 与 report seeds 100--105 分离。
- **20 组共享 teacher/相同父 memory 回放**：原配置 12 组，大 quota 配置 8 组。
- 14 个容量运行单元、28 个新配置/数据集/顺序 cell 的结构与 checkpoint 哈希审计通过。
- 独立重载 **408 份不同 checkpoint**，复算覆盖 1008 条记录的报告指标，最大差异 **0.0 pp**，0 errors；未使用 test 标签。

**没有一个容量配置通过开发门槛。文本一致性和拓扑联合修改按原计划的条件分支未启动，不能标成“已经验证失败”或“全部实验完成”。** response 是离线机制诊断，尚未运行端到端逐阶段 selector。

主要新发现：

1. 原始 preservation gate 在 12 个增量阶段完全不触发拒绝；固定同一 teacher 回放，gate/no-preserve 的 X 逐位相同。此前二者 F1 的差异不能作为 gate 有效性的证据。
2. 增大 revision 容量未稳定改善 final F1；减小半径只能减少部分损失。结果不支持仅用“改动太小”解释失败。
3. 64 步 response 有值得保留的信号，但排序依赖阶段/条件，尚不足以推广成主方法。

## 1. 实验口径

Cora、Citeseer，各 canonical、order19101，共 4 个开发 cells；3 stages，5% fixed-rate growing memory。这些条件已在历史实验中被查看，**不是独立泛化集**。

Teacher：128 steps，batch 64，seed 9000，只读取上一阶段 synthetic memory 和当前合法 raw delta。
Student：原两层 GCN、冻结 MiniLM、双向 InfoNCE；每阶段从头训练，batch 16，lr .001，steps = max(50, min(2048, 16 × rows))，seeds 100--105。

固定 C0 新增 block、总 rows、文本、text embedding、旧旧边和全部 topology，只修改允许范围内的历史 synthetic X。类标签用于阶段与 C0 的按类构造，但不用于 student InfoNCE。

final 为报告集 all-prompt cumulative Macro-F1；old/new 为同阶段 seen-prompt 指标。最终阶段两种 prompt 集合重合，中间阶段不能混用。
差值单位为百分点（pp）：先平均每 cell 六个 report seeds，再等权平均四个 cells。

原 Phase A **-0.88 pp** 是六 cell 均值；本轮原 TGC-R **-0.47 pp** 是四个开发 cell 的子集均值，不是修改后变好了。

## 2. Round 1：容量扫描

实际 quota = min(非保护行数, floor(quota × old_rows))。保护 80% 时不可能实际修改 50% 的旧行。
增加 protect060/protect040 控制，以区分“扩大候选集合”和“同一集合内修改更多行”。

| 配置 | 保护比例 | quota | 半径 | final Δ | old Δ | new Δ | 正收益 cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 TGC-R | 80% | 10% | .10 | -0.47 | -0.62 | -0.12 | 0/4 |
| radius002 | 80% | 10% | .02 | -0.13 | +0.09 | -0.49 | 1/4 |
| radius005 | 80% | 10% | .05 | -0.58 | -0.40 | -0.78 | 2/4 |
| radius020 | 80% | 10% | .20 | -0.32 | -0.88 | +0.81 | 2/4 |
| protect060 | 60% | 10% | .10 | -1.23 | -1.24 | -1.29 | 0/4 |
| quota025 | 60% | 25% | .10 | -0.45 | -0.27 | -0.75 | 2/4 |
| protect040 | 40% | 10% | .10 | -0.34 | -0.62 | -0.06 | 1/4 |
| quota050 | 40% | 50% | .10 | -0.31 | -0.71 | +0.18 | 2/4 |

C0 final 在下表四列依次为 49.85、49.69、59.17、46.63。不能逐列挑最优后拼成一个“方法”。

| 配置 | Cora canonical | Cora order19101 | Citeseer canonical | Citeseer order19101 |
|---|---:|---:|---:|---:|
| 原 TGC-R | -0.67 | -0.13 | -0.85 | -0.24 |
| radius002 | -0.46 | -0.06 | -0.92 | +0.92 |
| radius005 | -2.29 | +0.69 | -0.87 | +0.15 |
| radius020 | -1.34 | +1.17 | -1.38 | +0.26 |
| protect060 | -1.38 | -0.48 | -1.61 | -1.46 |
| quota025 | -1.11 | +0.78 | -1.94 | +0.48 |
| protect040 | -0.01 | -0.86 | -0.71 | +0.21 |
| quota050 | -2.36 | -0.49 | +0.32 | +1.30 |

### 实际修改与成本

这是四 cells、两个增量阶段合计的**更新次数**，不是跨阶段去重的节点数。

| 配置 | 尝试次数 | 接受次数 | 平均构造成本/C0 | 最大构造成本/C0 |
|---|---:|---:|---:|---:|
| 原 TGC-R | 39 | 39 | 2.46x | 2.72x |
| radius002 | 39 | 39 | 2.50x | 2.61x |
| radius005 | 39 | 39 | 2.31x | 2.50x |
| radius020 | 39 | 38 | 2.31x | 2.50x |
| protect060 | 39 | 39 | 2.31x | 2.49x |
| quota025 | 104 | 104 | 2.35x | 2.53x |
| protect040 | 39 | 39 | 2.30x | 2.49x |
| quota050 | 215 | 198 | 2.41x | 2.61x |

成本是模型/缓存已加载后的 block + operator 时间，不含冷启动、原始文本预编码和审计，不是总计算成本。C0 相同 block/拟合复用原始实测记录；两轮硬件负载不完全相同，不解释很小的时间差。

### 门槛与不确定性

冻结开发门槛：平均 final >= +0.5 pp；至少 3/4 cells >= -1 pp；平均 new >=0；每 cell 构造成本 <=3x。**含原配置共 8 个配置均未通过。** 最好均值的 radius002 仍为 -0.13 pp，new 为负。

这不是证明这些操作在所有图上无效。本轮每 cell 六 seed 的配对 t 区间均跨零；teacher/construction 只有一个 seed，数据集与顺序不独立，未做多重比较校正。小幅正负不能包装成显著改善或等效性。配对区间见 CSV。

- quota025 比同保护比例的 protect060 好，但仍未超过 C0；两种结论不能混同。
- quota050 与 protect040 均值接近，前者在 Citeseer 两个条件改善，在 Cora canonical 下降 2.36 pp。
- radius020 的 new 平均 +0.81 pp、old -0.88 pp。Citeseer order19101 中 new +5.79 pp、old -2.51 pp，是保持/塑性取舍的诊断信号，不是更优算法的证明。
- 小半径更接近 C0，也可能只是扰动减小，不能据此宣称 teacher 学得更好。

## 3. Round 2：无标签 response 离线诊断

Round 1 未通过，因而没有部署复杂 selector；执行协议允许的解释性诊断：9 个已有候选（C0 + 原配置 + 7 个变化），4 cells × 2 个增量阶段 × 2 个 selection seeds，共 144 条 128-step 轨迹。

读取 32/64/128 步相同训练前缀并计算多长度均值。CPU 测试和远程 8 条对照轨迹均确认：有/无中间探测时检查点状态哈希一致。训练沿用 canonical student 的运算顺序、Adam 和 minibatch 序列。

共享合法 probe 为 `C0 old synthetic memory + current raw delta`：固定 8 个旧 memory batch、8 个新 delta batch，old/new InfoNCE 等权平均；不读 report/test 标签。另保留自身 memory 的 training loss 对照。

top-k=2/4 集合是 C0 fallback 加 teacher relative loss gain 排名前 k-1 的候选，不按 F1 筛选。决策先落盘，再读取 canonical F1 算相关性和 regret。

**这不是端到端持续策略**：阶段 2 候选来自各自不同历史，公共 C0 probe 是诊断辅助 memory，不能把逐阶段选择分数拼成已运行的持续算法。

### 排序一致性

下表是四 cells 中“负 balanced loss”与 canonical F1 的 Spearman 相关均值。s1/s2 按代码 0-based stage，分别是第一次、第二次增量。

| 信号 | s1 平均相关 | s2 平均相关 | s2 最差相关 |
|---|---:|---:|---:|
| 32 steps | -0.033 | -0.200 | -0.700 |
| 64 steps | +0.096 | +0.292 | -0.433 |
| 128 steps | +0.012 | +0.037 | -0.300 |
| 三长度平均 | +0.100 | +0.062 | -0.150 |

64 步有研究价值，但精确均值 0.291667 未达到建议的 .3，且最差条件负相关。不因四舍五入接近门槛而宣布通过，也不只凭任意阈值断言无价值；主要问题是跨阶段/条件不稳定。

### 离线选择结果

最终阶段、4 候选、balanced response 的结果如下。regret 相对于同一 k=4 集合内的事后最优 F1，oracle 不进入选择。

| 选择信号 | 相对 C0 平均 Δ | 平均 regret | 回退 C0 的 cells |
|---|---:|---:|---:|
| 32 steps | -1.01 | 1.71 | 1/4 |
| 64 steps | +0.41 | 0.28 | 2/4 |
| 128 steps | +0.37 | 0.33 | 2/4 |
| 多长度平均 | -0.30 | 0.99 | 2/4 |

64 步下 teacher-only 和固定随机选择均为 -0.31 pp，shuffled 为 -0.25 pp；自身 memory training-loss 选择为 +0.49 pp。balanced response 尚未显示优于所有简单评分的独特优势。old-loss tolerance=.02 的 guard 在这些选择中与不加 guard 相同，不主张它产生增益。

144 条诊断轨迹及评分约 49.30 秒；8 条前缀验证另约 2.65 秒。这是全部 9 候选的诊断时间，不是已部署 top-4 selector 的端到端成本；未新增 canonical fits。

## 4. 修正：原始 preservation gate 的收益没有被证明

原 Phase A 六 cells、两个增量阶段合计 12 组，60 次提议全部接受，preservation 拒绝为零。gate/no-preserve 原本分别拟合 teacher，独立轨迹的旧 X 存在数值差异并在后续阶段累积，F1 差异不能归因于 preservation。

本轮加载已保存 teacher，在**相同父 memory、相同 delta、确定性 CPU 运算**下分别运行两臂：

| 范围 | 回放组数 | X 逐位相同 | 真正 preservation 拒绝 |
|---|---:|---:|---:|
| 原 TGC-R 六 cells | 12 | 12/12 | 0 |
| quota050 四 cells | 8 | 5/8 | 17/215 次提议 |

17 次拒绝分别在 Cora order19101 s2（2 次）、Citeseer canonical s2（13 次）、Citeseer order19101 s1（2 次）。其他阶段没有 gate/no-preserve 干预差异。

**gate 在大 quota 下能约束改动，但约束生效不等于提高 student F1。** CPU 回放没有新增 canonical 训练，不估计 gate 的下游因果增益。共享 teacher 的成对消融应成为后续默认设置；原 Phase A 结果保留，不事后改分。

## 5. 机制解释与下一步

目前优化固定 teacher 在“旧 synthetic + 当前真实 delta”上的输入损失：

`min_deltaX L_teacher(old_X + deltaX, raw_delta)`。

实际需要的是，在最终 compact memory 上重新训练 student 后改善合法 probe：

`min_deltaX L_probe(theta_K(M_compact(deltaX))) + lambda * preservation`。

后者梯度含 student 更新对 X 的依赖。让既有 teacher 更容易解释某些输入，不自动让这些输入产生更好的重新训练梯度。teacher 的新端是大量真实节点，student 的新端是少量 compact rows；传播、节点权重和 InfoNCE 负样本也改变。

这是代码可见的目标差异，仍是机制解释而非已完成根因归因。teacher gain 与 final 增益在四条件的相关分别为 -0.333、-0.071、+0.048、+0.405，不支持以 teacher loss 下降替代最终效果。

### 任务准确状态

| 任务 | 状态 | 边界 |
|---|---|---|
| runner 参数化/保护-quota 耦合修复 | 完成 | subset arm、计数、manifest、reference、契约测试 |
| Round 1 targeted 容量扫描 | 完成 | 7 个新配置，336 次新训练；不是 36 组合全因子网格 |
| Round 2 独立 seeds/多长度/排序/随机与 shuffled 诊断 | 完成 | 144 条轨迹，完整性及前缀审计 |
| 真正逐阶段 selector rollout | 未启动 | Round 1 无通过候选；离线增益不能替代 rollout |
| Round 3 X-text/joint penalty | 按条件未启动 | 无先行稳定收益；continuous-text 还需独立协议 |
| Round 4 X-topology 联合修正 | 按条件未启动 | 不在无可靠单组件证据时叠加 |
| 独立新顺序/seed、大图 | 未启动 | 本轮为暴露开发条件上的诊断 |

不要求最终论文永远采用 C0。TGC-R 仍可研究，但下一步应改**学习信号**，不是继续扩大 radius 网格：

1. 固定 teacher、父 memory 和候选做真正成对干预，避免独立拟合变化冒充 gate 效应。
2. 单增量步骤测试 batch 匹配的梯度或短段 student 更新目标，和原 teacher-input-gradient、stop-gradient、随机方向比较；合法 probe 仍仅来自旧 synthetic 与当前 delta。
3. 同一候选同时检查 teacher gain、short response gain、canonical F1；没有转化时，不立即解冻文本/拓扑。
4. 出现稳定候选后做真正逐阶段 selector、辅助 memory/选择成本核算，再用未看过的顺序/seed 确认，最后触发文本/拓扑实验。

以上是新结果导出的下一版建议，**没有在本轮实施或验证**。

## 6. 可复现证据

入口位于 `code/v2_llm_fusion/`：`run_tgc_r2_capacity.py`、`run_tgc_r2_response_diagnostic.py`、`diagnose_tgc_r2_gate_controls.py`。

最终工程验收：三个相关测试文件合计 **12 passed**；14 个新增/修改 Python 脚本通过 `py_compile`；下方 8 个本地证据链接均存在。测试命令：

```powershell
python -m pytest code/v2_llm_fusion/test_tgc_revision_gate.py code/v2_llm_fusion/test_tgc_r2_capacity.py code/v2_llm_fusion/test_tgc_r2_response.py -q
```

所有本轮实验和审计作业已结束，验收时远程 GPU 无计算进程。工程测试与结果审计通过不代表性能晋升通过。

- [冻结协议](TGC_R2_FROZEN_OPTIMIZATION_PROTOCOL_ZH_2026-09-22.md)
- [容量汇总](../../results/tgc_r2_capacity_20260922/local_report/summary.json)
- [逐 cell/配对区间 CSV](../../results/tgc_r2_capacity_20260922/local_report/rows.csv)
- [配置汇总 CSV](../../results/tgc_r2_capacity_20260922/local_report/aggregate.csv)
- [预测重载审计](../../results/tgc_r2_capacity_20260922/prediction_audit.json)
- [response 汇总](../../results/tgc_r2_response_20260922/aggregate.json)
- [response 审计](../../results/tgc_r2_response_20260922/audit.json)
- [共享 teacher 回放](../../results/tgc_r2_response_20260922/gate_controls.json)

远程为 `tagsam`，容量/response 各用独立快照，不覆盖 Phase A。完整 canonical checkpoints 留在服务器；本地有 JSON/CSV/日志、memory/certificate 和证据压缩包，容量证据包不含重复的 student checkpoint 大文件。
