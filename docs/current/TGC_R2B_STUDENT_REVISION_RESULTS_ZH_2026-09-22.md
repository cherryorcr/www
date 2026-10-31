# TGC-R2B：student 反向修订历史 X 的小图验收

日期：2026-09-22。状态：本批正式矩阵、checkpoint 审计、数值测试和成本补测完成；未通过递归 rollout 的预设触发条件。

## 1. 结论

这次不是只对已有候选做 response 排序，而是真正把梯度穿过临时 student 的训练更新，反向优化历史合成节点 X。

**机制已实现且数值验收通过；k1 出现值得继续研究的正向信号，但完整的 guarded 策略仍不可靠。**

- 未加接受/回退的 `student_k1`：第二增量阶段相对 C0 平均 +0.459 pp，旧类 +0.238 pp，新类 +0.946 pp；4 个条件中 3 个正收益。
- 同一条件下，k1 比 teacher 平均高 1.466 pp，比等幅随机方向高 0.844 pp；四个条件都高于本轮 teacher。
- 加上冻结的无标签 gate 后，`guarded_k1` 只剩 +0.060 pp，新类 -0.034 pp。
- k4、k8 没有显示“反传步数越多，最终效果越好”。三个 guarded horizon 均未通过预设条件。
- 不能把失败仅归结为 k1 距 +0.5 pp 差约 0.04：真正部署的 guarded 策略收益更小，接受/拒绝方向仍与最终 F1 脱节。

**没有自动启动 R2-C 文本联合修订、R2-D 拓扑联合修订、递归持续 rollout 或大图。** 这按冻结计划的条件分支执行，不表示上述方向已经实验失败。

## 2. 实验范围与计数

| 项目 | 实际设置 |
|---|---|
| 数据 | Cora、Citeseer |
| 顺序 | canonical、order19101，均为已暴露的开发条件 |
| memory | 5% 随数据增长，C0 的新增 block 固定 |
| 阶段 | s1、s2，即第 1、2 次增量；每次均从相同 C0 父 memory 出发 |
| 修改范围 | 保护约 80% 旧行，最多修改约 10% 旧行；相对位移半径 .1 |
| 固定内容 | 文本、text embedding、全部边、rows、row order、mass、depth、stage |
| student | 两层 GCN，hidden/out 384；冻结 MiniLM；双向 InfoNCE；Adam .001、batch 16 |
| 正式训练 | 从头训练，seeds 100--105；steps = max(50,min(2048,16×rows)) |
| 元优化/验证 seeds | 7500/7501、7510/7511；与正式 seeds 分离 |
| 优化 | inner 1/4/8 steps；outer 16 steps、lr .01 |
| 规模 | 8 个单阶段条件 × 14 臂 × 6 seeds = 672 个记录槽位 |
| 实际正式 fit | 新训练 624 次，复用 C0 48 次 |

两阶段的具体 memory 与正式步数：

| 条件 | s1 rows / steps | s2 rows / steps |
|---|---:|---:|
| Cora canonical | 67 / 1072 | 94 / 1504 |
| Cora order19101 | 66 / 1056 | 94 / 1504 |
| Citeseer canonical | 86 / 1376 | 111 / 1776 |
| Citeseer order19101 | 71 / 1136 | 111 / 1776 |

**本轮不是端到端持续方法成绩。** s2 使用 C0 的 s1 memory，不使用某个候选自身在 s1 产生的 memory。这样可以隔离单次修订的作用，但不能把两次干预的分数拼成已完成的递归持续实验。

报告指标为 report-dev 的 all-prompt cumulative Macro-F1；old/new 表使用同阶段 seen-prompt 指标。s2 所有类别已见，两种 prompt 集合相同；s1 不同。标签用于既有类增量划分和 C0 构造，不进入元目标、gate 或 checkpoint 选择；未评估 test 标签。

## 3. 实际算法

1. 读取固定旧 synthetic memory、当前合法 raw delta，以及固定的 C0 compact memory。历史完整 raw graph 不作为修订算子的输入；完整真实图仍用于独立下游评估。
2. 用同一个 128-step teacher 确定各臂共享的可修改旧行，保留其 X-input-gradient 版本作为 teacher 对照。
3. 对可修改行引入相对位移 u：`x'_i = x_i + ||x_i|| u_i`。逐行投影保证 `||u_i|| <= .1`。
4. 从头初始化临时 student，在修改后的 compact memory 上执行 K=1/4/8 步可微 Adam。
5. 在固定合法 probe，即修改前的旧 synthetic memory 加当前 raw delta，上计算外层损失：

   `L_new + max(0, L_old - L_old(C0-paired student) - .01) + .01 mean(||u_i||²)`。

6. 将此损失穿过 student 的参数更新反向传到 u，执行 16 次外层优化。临时 student 每次重新初始化，不传递到最终模型。
7. 在 outer step 4/8/12/16，按独立 validation seeds 的无标签目标选择候选。guard 要求两个 validation seeds 的 new loss 均至少下降 1e-4，old loss 均不恶化超过 .01，否则回退 C0。
8. 候选、接受决策和 memory hash 先落盘，再进行正式从头训练与报告。

旧/新 probe 各自拆分 optimization/validation，采样各 4 个 batch。这只是合法支持集内的拆分：当前 delta 已参与构造，图上下文也共享，不能称作未见原始数据泛化集。

对照包括 C0、teacher、各 horizon 的 student、stop-gradient、等幅随机方向和 guarded。stop 阻断训练更新到 X 的梯度；random 逐行保持与 student 相同的位移范数。

## 4. 主要结果

### 4.1 s2：六个正式 seeds 的平均 Macro-F1

| 方法 | Cora canonical | Cora order19101 | Citeseer canonical | Citeseer order19101 | 平均相对 C0 |
|---|---:|---:|---:|---:|---:|
| C0 | 49.85 | 49.69 | 59.17 | 46.63 | 0.00 |
| teacher | 46.94 | 49.38 | 59.42 | 45.59 | -1.01 |
| student k1 | 48.45 | 51.33 | 59.83 | 47.56 | +0.46 |
| student k4 | 48.38 | 50.06 | 56.87 | 47.70 | -0.58 |
| student k8 | 49.70 | 50.31 | 58.27 | 47.40 | +0.08 |
| guarded k1 | 48.45 | 51.33 | 59.17 | 46.63 | +0.06 |
| guarded k4 | 48.38 | 50.06 | 56.87 | 46.63 | -0.85 |
| guarded k8 | 49.70 | 50.31 | 58.27 | 46.63 | -0.11 |

差值单位为 pp，先在每个条件内平均 6 seeds，再等权平均 4 个条件。teacher 是本轮共享 C0 父 memory 的干预，不是上轮递归 TGC-R 的最终分数；不能与上轮 -0.47 pp 直接解释为同方法回退。

### 4.2 阶段、保持/塑性与随机对照

| 方法 | s1 cumulative Δ | s2 cumulative Δ | s2 old Δ | s2 new Δ | s2 相对匹配 random |
|---|---:|---:|---:|---:|---:|
| teacher | +0.43 | -1.01 | -1.34 | -0.30 | 不适用 |
| student k1 | +0.09 | +0.46 | +0.24 | +0.95 | +0.84 |
| student k4 | +0.30 | -0.58 | -0.45 | -0.64 | +0.51 |
| student k8 | -1.50 | +0.08 | -0.06 | +0.31 | +0.06 |
| guarded k1 | +0.20 | +0.06 | +0.13 | -0.03 | +0.45 |
| guarded k4 | +0.37 | -0.85 | -0.14 | -2.06 | +0.24 |
| guarded k8 | -1.02 | -0.11 | -0.21 | +0.04 | -0.13 |

random k1/k4/k8 的 s2 相对 C0 均值分别为 -0.385/-1.094/+0.023 pp。student 在若干配置优于普通扰动，不等于所有 horizon 都能稳定优于 C0。

所有 12 个 s2 student 候选条件的六-seed 配对 95% t 区间均跨零。这里只能称为开发信号，不能称为显著提升。构造 seeds 未重复，数据集和类别顺序也不是独立样本；没有多重比较校正。

## 5. 哪里出了问题

### 5.1 gate 仍不能预测最终训练结果

以 k1 的 s2 为例：

| 条件 | 未加 gate 的 F1 Δ | gate 决策 | 验证 new loss 平均改善 |
|---|---:|---|---:|
| Cora canonical | -1.402 | 接受 | +0.000172 |
| Cora order19101 | +1.642 | 接受 | +0.000300 |
| Citeseer canonical | +0.660 | 拒绝 | +0.000263 |
| Citeseer order19101 | +0.935 | 拒绝 | -0.000009 |

Citeseer canonical 平均 new loss 改善仍被拒绝，是因为规则要求两个 seeds 都达到阈值，不能用均值代替逐 seed 条件。问题不只是 gate 太严格：它也接受了 Cora canonical 的负结果；k4 还接受了 Citeseer canonical 的 -2.306 pp 候选。

在八个单阶段条件上，k1/k4/k8 分别接受 4/5/5 次；拒绝全部由 new-gain 条件触发，没有一次由 old-loss tolerance 触发。

### 5.2 保护惩罚实际上没有激活

24 个 student 优化、每个 16 次外层迭代、每次 2 个训练 seeds，共 768 次旧端 loss 检查。相对各自 C0 的旧 loss 增幅最大 0.003414，均低于 .01，因此训练端 old hinge 激活数为 0。

所有 validation 候选也都满足旧端 .01 tolerance。不能据此宣称这个 preservation 项贡献了保持收益；旧类 F1 仍可能下降，例如 Cora canonical 的 k1 old F1 为 -1.924 pp。

### 5.3 优化对象与最终使用方式还不够接近

代码可见的差异包括：

- 元优化只穿过随机初始化后的 1/4/8 步，而正式训练为 1056--1776 步。
- probe 的 old/new InfoNCE 分开采样，未显式测量旧新类别之间的竞争；最终分类需要在共同类别 prompt 中区分。
- old probe 只有历史合成节点，不是真实旧分布；这是访问约束下必要的近似，不是旧任务保持的充分保证。
- 24 个候选的最大相对位移均达到 .1 边界。增大 inner K 不等于修订幅度更合适，也不自动改善泛化。

这些是有依据的后续假设，不是本轮已通过消融证明的全部根因。尤其不能把 k8 变差直接说成已证实的过拟合。

## 6. 工程与数值验收

- 覆盖全部 672 条记录、112 个方法/阶段槽位；共享输入、protected rows、位移半径、固定字段和 row order 检查通过。
- 重载 378 个不同 student 状态，独立用各类 TP/FP/FN 复算报告指标；最大差异约 2.84e-14 pp，容差 1e-7 pp，0 errors。
- C0 复用核对数据、阶段节点、pool/report mask、共同 student 源码、memory 和 checkpoint 哈希。修订算子不接收历史真实 X/text，probe 按当前 delta 重建。
- GPU 上 1/4/8-step 可微 Adam 与普通 Adam 对照通过；两个真实图 fixture 中最大参数差约 5.10e-7。
- 双精度方向有限差分在 1e-7、1e-8 两个尺度收敛，容差 rtol=.001、atol=1e-6。较大扰动的有限差分不一致没有被用来放宽最终验收；优化算法本身未因此修改。
- 本地新旧相关测试合计 23 passed、1 skipped，跳过项为本地无 CUDA。远程本轮测试 12 passed，包含 GPU Adam 对照。
- 7 个本轮 Python 文件通过 py_compile。

执行偏差：runner 没有实现预期的 stop/guarded fit 复用，额外重训了 288 次。保留全部原记录，另按冻结决策生成 paired-policy 表。本轮这些同 memory/seed 的重复训练末状态和 F1 全部一致，因此 raw-executed 与 paired-policy 的指标和晋升结论一致。

完整有限差分和扩展 GPU 验收是在正式运行过程中/之后补齐，并非协议要求的全部前置完成；这一执行次序偏差保留记录。补验未修改元优化算法、候选、正式训练或晋升阈值。

## 7. 成本

| 项目 | 计量 |
|---|---:|
| 624 次新正式 student 训练累计时间 | 2080.88 秒 |
| 24 次 student 元优化实测累计时间 | 38.14 秒 |
| 24 次 stop 对照优化实测累计时间 | 21.10 秒 |
| k1 单候选元优化平均 | 0.672 秒 |
| k4 单候选元优化平均 | 1.495 秒 |
| k8 单候选元优化平均 | 2.600 秒 |

累计训练时间是各 fit 训练循环用时之和，不含最终预测评估，不是两个 GPU 同时运行的墙钟时间，也不是部署一个方法需要重训 624 次。正式运行使用两张 RTX 4090，成本复测使用其中的 GPU 0。

原 runner 没有记录共享 teacher 的完整耗时，不能事后冒充原始实测。另在相同八组合法输入上每组重测 3 次 teacher，取中位数，再与历史 C0 构造时间和本轮元优化时间组成**分项估计**：

| horizon | 每阶段 C0构造 + teacher + 元优化估计 | 相对历史 C0 的平均分项比值 |
|---|---:|---:|
| k1 | 1.212 秒 | 8.28x |
| k4 | 2.035 秒 | 13.97x |
| k8 | 3.140 秒 | 21.57x |

这个估计混合不同时点的计时，不含冷启动、原文预编码、文件 IO 和审计，不是端到端成本比，也不是训练加速比。绝对时间在小图上不大，但相对非常便宜的 C0 并不免费。不能由此宣称总计算高效。

持久 memory 的字段、rows、文本字节数与边数保持不变；临时 teacher、student、优化器和计算图不传给最终模型。

## 8. 计划完成边界与下一步

| 任务 | 当前状态 |
|---|---|
| R2-A 容量扫描 | 前一批已完成，无稳定晋升配置 |
| R2-B 离线多长度 response 排序 | 前一批已完成，仅离线诊断 |
| 本轮 student-to-old-X 可微修订 | 672 槽位、完整小图矩阵和审计完成 |
| 原定 guarded 触发判断 | 已完成；k1/k4/k8 全部未通过 |
| 真正递归 continual rollout | 本轮条件未满足，未启动 |
| R2-C X-text consistency | 条件后续，未启动，不标成失败 |
| R2-D X-topology 联合修订 | 条件后续，未启动，不标成失败 |
| 新顺序/新 seed group 与大图 | 仍需未来冻结候选后进行 |

下一轮值得保留的是“直接优化 student 的训练响应”这个方向，而不是继续扩 teacher 容量或降低当前门槛。建议重新预注册以下小规模机制实验，本报告尚未执行：

1. **训练阶段对齐**：在合法 compact memory 上先做共同 warm-up，再从同一 student/Adam 状态和 minibatch 前缀展开短段更新。比较随机初始化短段与中期短段；评估仍从头训练的正式 student。必须说明这是 truncated surrogate，不冒充完整训练全过程的精确梯度。
2. **旧新竞争对齐**：保持 rows/X/text/topology 和预算不变，只比较 old/new 分开 probe 与混合 probe。混合负样本是否更能预测共同 prompt 的分类表现，需对照验证。
3. **选择与候选拆开验收**：保留未加 gate、冻结 gate、stop、匹配随机控制。既检查候选收益，也检查 gate 是否拒绝好候选、接受坏候选，不直接按 report F1 改阈值。
4. **保护项是否真生效**：先在 selection-only 观测中量化尺度与跨 seed 方差，再冻结约束。不要只因为代码中存在 hinge，就把历史保持写成独立贡献。
5. 只有得到可靠候选与选择规则，才做递归 rollout、未见顺序/seed 确认，最后触发文本/拓扑联合修订。

这不意味着最终只能用 C0 写论文，也不构成 TGC-R2B 已有效的正式结论。当前最准确的定位是：**k1 的方向有开发信号，短程 gate 仍失败；C0 保持冻结共同参照。**

## 9. 可复现证据

- [冻结协议与执行补记](TGC_R2B_STUDENT_REVISION_PROTOCOL_ZH_2026-09-22.md)
- [全部汇总](../../results/tgc_r2b_student_revision_20260922/summary.json)
- [逐条件指标与配对区间](../../results/tgc_r2b_student_revision_20260922/cell_metrics.csv)
- [汇总指标](../../results/tgc_r2b_student_revision_20260922/aggregate.csv)
- [元优化、接受和保护项统计](../../results/tgc_r2b_student_revision_20260922/mechanism.csv)
- [完整 memory/checkpoint/预测审计](../../results/tgc_r2b_student_revision_20260922/audit.json)
- [原始计时账](../../results/tgc_r2b_student_revision_20260922/costs.csv)
- [事后 teacher 成本复测](../../results/tgc_r2b_student_revision_20260922/cost_remeasurement.json)
- [远程数值测试日志](../../results/tgc_r2b_student_revision_20260922/numerical_tests.log)
- [原始结果与 56 份 memory 证据包](../../results/tgc_r2b_evidence_20260922.tar.gz)

实现入口：`code/v2_llm_fusion/run_tgc_r2b_student_revision.py`；元优化：`tgc_student_revision.py`；验收：`audit_tgc_r2b_student_revision.py`；汇总：`summarize_tgc_r2b_student_revision.py`；成本复测：`measure_tgc_r2b_costs.py`。

远程环境为 `pc@100.126.175.112` 的 conda `tagsam`；快照 `work_runs/tgc_r2_20260922_snapshot`，结果 `results/tgc_r2b_student_revision_20260922`。正式 student checkpoints 留在远程；本地保存指标、决策、日志和 memory 证据，未替换历史 Phase A/R2 结果。

证据包 SHA256 为 `212e5f1c4a753fe20d081162bd0fc90244171413f308fccb9edbcf8d4f8e1883`，本地和远程一致。收尾检查两张 GPU 均无本轮计算作业。证据中的远程新旧联合测试尝试因缺少旧 response 测试文件而未执行，不能计作通过；旧用例的完整回归已在本地执行，本轮独立数值测试在远程通过。
