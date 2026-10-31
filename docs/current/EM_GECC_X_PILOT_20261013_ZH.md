# EM-GECC-X Pilot：审计后的结果与决策

## 结论

**这次有限的 EM-GECC-X 尝试未通过预先设定的继续门槛，停止扩展，返回 Unified-Q-EF 论文主线。** 原始负结果完整保留，不再依据这轮 report-dev 分数调整方法或替换 sealed 主表。

这不等于证明 EM 或老师提出的方向普遍无效，也不构成 Unified-Q-EF 正式主版本优越性的证据。本轮 Unified-Q 对照实际使用旧构造器且关闭 DM，必须与论文冻结主版本区分。

## 1. 范围与协议

实验覆盖 Cora、Citeseer，各 3 个阶段。只有一个类别顺序 `order19611`、一个 construction seed `271006`，每份 memory 使用 Student seeds `150/151/152` 评估。

- 构造只使用当前合法 delta 和历史 synthetic prefix；历史特征、文本及来源保持冻结，不测试旧 memory 更新。
- 预算为当前合法池累计节点数的 `floor(0.05 * N_seen)`，不是固定节点数或固定字节预算。
- Cora 各阶段 memory 行数为 `43/60/94`，Student 步数为 `688/960/1504`；Citeseer 为 `42/78/111`、`672/1248/1776`。各方法共用调度。
- 共同下游 Student 是两层 GCN 的 GraphTextPromptStudent，使用同一冻结 MiniLM、类别 prompt 和训练预算，预测使用 prompt cosine 而非额外分类头。
- 保存的 memory 均无 synthetic edges。构造仍使用当前图的传播特征。
- 只评估 `report_dev`，不是 sealed test，也不是多顺序、多 construction-seed 的确认实验。

## 2. 比较对象与实际实现

| 报告名称 | 实际做法 | 复现性质 |
|---|---|---|
| GECC-hard + nearest-5 TAG | 在当前 delta 的传播特征上做同类硬聚类，sklearn Lloyd 使用 20 次重启，按 SSE 与容量平衡评分选择；为中心附加同类 nearest-5 文本，320 字符内拼接并重新编码 | GECC 机制适配；不是 official reproduction，也不是原生 soft 分支；文本附着不是 GECC 官方模块 |
| Unified-Q pilot（DM off / legacy code） | 原始运行键为 `unified_q_ef`，实际模块 SHA256 为 `f621d4575a9bb660fad8031e4d7135e2570cc72c02d16145c08aa989bee99525`，`lambda_distribution` 与 `lambda_joint_distribution` 均为 0 | 旧代码筛选对照，不能标为冻结的 Unified-Q-EF 论文主方法 |
| EM-GECC-X | 以同一 GECC-hard 的新阶段 X 为初始化，交替计算类内 Q/硬文本来源，并直接梯度更新自由的新 X | 新试验分支，EM-like，而非有似然模型的严格 EM |
| EM-no-GM | 相同初始化与交替流程，仅保留 Q 条件的几何目标 | 组件消融，不是外部基线 |
| EM-no-geometry | 相同初始化与交替流程，仅保留 Student proxy 梯度匹配 | 组件消融，不是外部基线 |

EM 的 E-step 使用传播特征相似度、文本相似度和固定 proxy 表示相似度，权重分别为 `0.45/0.25/0.30`。类掩码 Sinkhorn 使用温度 `0.20`、40 次迭代，再通过同类 Hungarian 匹配为每个 slot 选一个互不重复的真实文本来源。

M-step 将新 X 直接作为参数，以 Adam、`lr=0.005` 更新。完整分支优化 blockwise gradient matching 加 `0.25` 倍几何损失；几何目标为 Q 对当前传播特征的池化。总共 4 轮，每轮 5 个 M-step。Q 在 E-step 计算并 detach，不是一个与 X 同时反向优化的参数；旧 X 始终不变。

**M-step 前向使用已选中的硬文本 embedding，软文本只用于诊断。** 最后再执行一次 E-step 保存来源。因此不能把本轮失败解释为“训练使用软文本、保存变成硬文本”。

## 3. 最终阶段结果

Macro-F1/Accuracy 为百分数，`+/-` 表示 3 个 Student seeds 的样本标准差，不包含顺序或 construction-seed 方差。Old/New F1 使用累计类别集合进行预测，再分别统计旧/新类别样本。

| 数据集 | 方法 | Macro-F1 | Accuracy | Old F1 | New F1 | 构造时间(s) |
|---|---|---:|---:|---:|---:|---:|
| Cora | GECC-hard + nearest-5 TAG | 82.11 +/- 0.81 | 85.36 +/- 0.56 | 82.89 | 81.07 | 0.48 |
| Cora | Unified-Q pilot（DM off / legacy code） | 80.31 +/- 0.26 | 83.03 +/- 0.37 | 80.39 | 80.19 | 4.46 |
| Cora | EM-GECC-X | 70.14 +/- 0.65 | 74.29 +/- 0.56 | 75.88 | 62.48 | 1.67 |
| Cora | EM-no-GM | 79.65 +/- 0.82 | 82.41 +/- 0.77 | 81.12 | 77.68 | 1.08 |
| Cora | EM-no-geometry | 68.97 +/- 0.72 | 73.68 +/- 0.43 | 74.77 | 61.24 | 1.59 |
| Citeseer | GECC-hard + nearest-5 TAG | 65.81 +/- 1.80 | 69.80 +/- 1.27 | 71.75 | 53.93 | 0.39 |
| Citeseer | Unified-Q pilot（DM off / legacy code） | 68.68 +/- 0.70 | 73.25 +/- 0.48 | 75.10 | 55.82 | 4.44 |
| Citeseer | EM-GECC-X | 56.31 +/- 0.83 | 64.26 +/- 1.13 | 68.85 | 31.22 | 1.86 |
| Citeseer | EM-no-GM | 67.89 +/- 1.26 | 72.00 +/- 0.65 | 73.25 | 57.17 | 1.21 |
| Citeseer | EM-no-geometry | 55.62 +/- 1.45 | 63.43 +/- 1.73 | 68.01 | 30.84 | 1.84 |

构造时间是各阶段构造区间的记录值之和，CUDA 前后有同步，不包含完整数据准备和下游 Student 训练。显存记录是构造区间的最大 CUDA allocated 值，不是整个系统占用；完整 EM 的 Cora/Citeseer 峰值为 `309.6/628.6 MiB`。这组小图结果不能替代大图效率证据。

## 4. Go/No-Go 决策复核

每个数据集以 GECC 适配版和旧 Q pilot 中 F1 更高者为筛选比较对象，在相同 Student seed 上计算差值。原始门槛与分数均未改动，独立复算全部一致。

| Gate 指标 | 观测值(pp) | 预设要求 | 通过 |
|---|---:|---:|---|
| 相对较强 pilot 对照的平均 Macro-F1 增益 | -12.17 | >= 1.00 | 否 |
| 最差数据集 Macro-F1 增益 | -12.37 | >= -0.50 | 否 |
| 平均 Accuracy 增益 | -10.03 | >= 0.00 | 否 |
| 最差数据集 Accuracy 增益 | -11.07 | >= -1.00 | 否 |
| 相对 EM-no-GM 的平均 Macro-F1 增益 | -10.55 | >= 0.50 | 否 |

即使不使用旧 Q 对照，完整 EM 也低于两个数据集的 GECC 适配版，因此不需要靠正式 Q 主方法的重跑来确定当前 EM pilot 没有通过筛选。

## 5. 诊断：观察到什么、尚未证明什么

### 5.1 当前 GM 配置的负效应

去掉 GM 后，Cora/Citeseer F1 分别回升 `9.51/11.58 pp`，New F1 从 `62.48/31.22` 回升至 `77.68/57.17`。完整分支相对 no-geometry 只高 `1.17/0.70 pp`，不足以弥补退化。可据此指出当前 GM 更新配置有负效应，不能据此断言 gradient matching 普遍无效。

### 5.2 交替流程没有改变最终硬文本支持

全部 18 个 EM“数据集 × 阶段 × 分支”单元的 4 次 E-step 硬来源都相同，各轮硬切换次数为 0；最后一次 E-step 到保存时的切换也为 0。三个 EM 分支在全部 6 个“数据集 × 阶段”单元保存相同的硬文本来源。

这说明本轮后续 E-step 没有重选文本支持，分数差异不能用三个 EM 分支选择了不同文本来解释。**它不证明软 Q 一直不变**：软权重和 Q 条件的几何目标仍可能随 X 变化。

### 5.3 熵、软硬差异与 X 漂移

下面均为最终阶段新 slot 的诊断，不是整份累计 memory 的统计。

| 数据集 | 分支 | H(source given slot)归一化 | 软/硬文本 cosine | 正确归一化类中心 cosine | 相对 GECC 的 X 漂移 |
|---|---|---:|---:|---:|---:|
| Cora | EM-GECC-X | 0.9892 | 0.5706 | 0.6669 | 61.68% |
| Cora | EM-no-GM | 0.9963 | 0.5469 | 0.6864 | 51.26% |
| Cora | EM-no-geometry | 0.9891 | 0.5707 | 0.6651 | 61.77% |
| Citeseer | EM-GECC-X | 0.9844 | 0.5489 | 0.5382 | 50.47% |
| Citeseer | EM-no-GM | 0.9928 | 0.5221 | 0.6748 | 53.69% |
| Citeseer | EM-no-geometry | 0.9843 | 0.5492 | 0.5365 | 50.68% |

- 原始 `q_entropy_normalized` 是每个 slot 对同类来源的条件熵，除以同类来源数的对数；不是“raw 节点分配到 slot 的熵”。
- `slot_text_coverage` 和不同文本字符串比例均为 100%。前者由 Hungarian 的不重复选取约束保证，不是独立的性能收益证据。完整 EM 的硬文本两两平均 cosine 为 Cora `0.1963`、Citeseer `0.1718`。
- 软硬 cosine 约 `0.55` 只能说明软均值与被选文本不同，不能解释本轮的训练/保存失配；实际 M-step 与保存的硬来源一致。
- 原始 `class_center_cosine_mean` 未归一化，是中心点积，不能当成 cosine；上表使用重新归一化后计算的值。
- no-GM 同样有很大的 X 漂移，所以漂移与退化的因果关系尚未得到证明。

### 5.4 新旧样本权重不一致：已核对的风险，不是已证明的原因

GM 真实侧将旧 synthetic 行和当前 raw 行合并，合成侧将旧 synthetic 行和新 synthetic 行合并。全局损失没有分别归一化旧/新组的权重，最终阶段新行占比如下：

| 数据集 | 真实侧新行占比 | 合成侧新行占比 |
|---|---:|---:|
| Cora | 92.04% | 36.17% |
| Citeseer | 89.29% | 29.73% |

因此两侧并不具有相同的新旧行权重，可能使梯度匹配目标偏离希望保留的持续学习行为。这是合理的失败假设，尚未做受控修复验证，不应写成确定原因。

其它限制包括：构造使用固定随机 128 维 proxy，而非训练中的下游 Student；新 X 无显式位移约束；只运行 20 个 M-step；没有测试旧 memory 的有界更新、时序 teacher/student logits 对齐或独立确认。`mass` 元数据在 GM 损失和下游 Student 训练中均未使用，不能将其差异当成性能解释。

## 6. 审计与可复核文件

全部 30 份 memory 通过哈希、预算、有限值、空边、旧前缀张量/字符串/来源、共同 Student 预算及 F1/Acc 复算检查。服务器端还核对了原始合法池与 report-dev mask 哈希、累计 floor-5% 调度，以及每个新增文本来源属于当前 delta 和对应类别。

审计记录绑定 35 个原始文件的 SHA256：30 个 memory、4 个 protocol/results JSON 和原始 decision。输出写在 `run_v2` 之外，没有修改 raw 分数、原始决策或 sealed 表。

审计没有重新运行文本编码器，也没有通过运行时访问跟踪独立证明历史 raw 读取次数为 0。通过文件检查不等于完成 official reproduction。

审计、汇总和 EM 构造器的自动化测试共 25 项，全部通过；相关脚本通过 `py_compile`。测试覆盖冻结前缀与可选来源字段、分数复算、硬来源切换统计、原始文件指纹，以及禁止向 raw 运行目录写入派生汇总的规则。

- 不可变原始运行：`results/em_gecc_x_pilot_20261013/run_v2/`，早期未完成的 `run/` 保留不动。
- 完整审计：`results/em_gecc_x_pilot_20261013/artifact_audit.json`。
- 独立复算决策：`results/em_gecc_x_pilot_20261013/decision.json`。
- 英文结果与诊断：`results/em_gecc_x_pilot_20261013/pilot_summary.md`。
- 数值表：`results/em_gecc_x_pilot_20261013/final_metrics.csv`、`final_diagnostics.csv`。
- F1/Acc 可视化：`results/em_gecc_x_pilot_20261013/final_metrics.png`。

## 7. 后续决定

按用户批准的有限尝试原则，**本轮不启动大图 EM 扩展，不继续从 report-dev 负结果中搜索更有利配置，也不把 EM-no-GM 提升为新的主方法**。保留此负结果，继续围绕冻结的 Unified-Q-EF 撰写论文；正式结论仍需引用相应版本、共同协议和独立确认的已有结果。

如果未来重新研究老师的方向，应视作另一个预先定义的实验，首先验证新旧权重与 proxy 目标，而不是把本轮 post-hoc 假设直接宣称为方法贡献。本次只更新实验审计和报告，没有修改论文正文或推送 Overleaf。
