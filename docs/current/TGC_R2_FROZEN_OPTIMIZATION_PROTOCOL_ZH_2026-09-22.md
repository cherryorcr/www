# TGC-R2 最小机制优化实验

日期：2026-09-22。状态：执行前冻结。研发诊断，不是独立泛化确认。

## 边界

历史完整 raw graph 不可回读；只用旧 synthetic memory 与当前合法 raw delta。
Round 1 仅修改旧 synthetic X，固定 rows、新增 C0 block、文本、边、student、batch、steps。
Teacher 仍为 128 steps / seed 9000；report seeds 为 100--105，不把 teacher 传给 student。
Student 标签不参与训练。类标签用于阶段和 C0 的类内构造；不能声称整个算法无标签。
报告既有的 all-prompt cumulative Macro-F1 和 seen-prompt old/new Macro-F1，不混称为同一口径。

## 既有证据

Phase A 完成 648 个 fits / 6 cells，原审计通过。TGC-R final 相对 C0 平均 -0.88 pp。
该结果不因本轮调参被替换；新实验单独快照与目录保存。
旧 report-dev 和 class orders 已被反复查看，本轮只作开发集分析。

## Round 1 容量与可修改集合

Cora、Citeseer × canonical、order19101 × 3 stages × 6 report seeds。
所有结果包括 C0；baseline/no-preserve/shuffled/naive/C0-M 取同协议 Phase A。

| 名称 | protected | quota | radius | 作用 |
|---|---:|---:|---:|---|
| baseline | .80 | .10 | .10 | 原 TGC-R |
| radius002 | .80 | .10 | .02 | 小位移 |
| radius005 | .80 | .10 | .05 | 小位移 |
| radius020 | .80 | .10 | .20 | 大位移 |
| protect060 | .60 | .10 | .10 | 只放松候选集合 |
| quota025 | .60 | .25 | .10 | 在相同集合下增加 quota |
| protect040 | .40 | .10 | .10 | 只放松候选集合 |
| quota050 | .40 | .50 | .10 | 在相同集合下增加 quota |

实际 quota = min(非保护行数, floor(quota × old_rows))。
保护 80% 时不可能实际修改 50%，故不能忽略保护比例与 quota 的耦合。
报告每阶段实际尝试数、接受数、位移、teacher gain、old/new F1、构造成本和真实执行 fits。
只有 support、steps、seed、数据/报告掩码、训练实现哈希完全一致时复用 C0/首阶段 checkpoint。
复用 checkpoint 不计作本轮重新训练；成本沿用原始实测记录并注明来源。

开发门槛：四 cell 平均 final 增益 >= +0.5 pp；至少 3/4 不低于 -1 pp；
平均 new 增益 >=0；所有 cell 构造成本 <=3x C0。此门槛不是论文主方法晋升标准。

## 后续条件路线

Round 2：独立 selection seeds 700/701，32/64/128-step response curve，有限 2/4 候选，
固定无 report 标签的合法评价支持。计算 short/canonical Spearman、选择 regret、回退与成本。
必须验证复用前缀不改变训练/RNG。候选自己 memory 上的训练 loss 不能单独作为可比质量指标。
完整 canonical 排名只用于事后诊断，不进入候选选择。均值相关 <.3 则不晋升 response selector。
即使 Round 1 未达性能门槛，也允许执行这一小型离线诊断来解释失败，但不宣称新主方法。

Round 3：通过前述门槛再做 X-only、X+text、joint consistency、text-only、shuffled text。
直接更改 text embedding 会使其不再等于冻结编码器对保存文本的输出，必须单列 continuous-text
实验协议，不能混入保持可读文本/重编码一致的主表。优先保留文本、加入 joint consistency。
移动惩罚在零位移点梯度为零，不能只算一次起点梯度就声称其产生正则作用。

Round 4：前面存在可靠正收益再测 X-only、topology-only、X+topology、random/shuffled edges。
cross degree 1/2/4 必须匹配边数/端点度对照，记录归一化旧块漂移。

本轮不无限网格调参、不用 report-dev 挑每个 cell 最优后拼表、不以 teacher loss 代替 final F1。
未触发的后续轮次标为“按门槛未启动”，不标记为已经完成实验。
