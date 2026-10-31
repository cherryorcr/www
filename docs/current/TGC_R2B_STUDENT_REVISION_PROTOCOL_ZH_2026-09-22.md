# TGC-R2-B：可微 student 历史 X 修订最小实验

日期：2026-09-22。状态：结果前冻结；先做小图单增量因果实验。

## 问题与边界

检验穿过 student Adam 更新的外层梯度，是否比固定 teacher 的输入梯度更适合修改历史 synthetic X。
不是首次使用元梯度：P2-B 已用其优化新文本 attention；本轮更换的是优化变量和旧端对照。
不访问历史完整 raw graph；模块只接收固定旧 synthetic memory、当前合法 raw delta、C0 compact memory。
类标签仅用于既有阶段和 C0 构造，不进入元优化、候选选择或 gate。报告标签只在构造完成后读取。

## 固定条件

- Cora/Citeseer，各 canonical/order19101，两个增量阶段，共 8 个共享 C0 父 memory 的单阶段 cells。
- 5% growing memory；C0 新 block、文本、所有边、mass/depth/stage/rows/order 固定。
- protected fraction .8，quota .1，relative radius .1；不按本轮报告 F1 调整。
- 每个 cell 仅拟合一次原 TGC teacher：128 steps、batch 64、seed 9000。所有臂共享其排名得到的可修改行。
- student 与 canonical 相同：两层 GCN、冻结 MiniLM、双向 InfoNCE、Adam .001、batch 16。
- 正式 seeds 100--105、完整步数 max(50,min(2048,16*rows))；临时 student 参数不传给正式模型。

## 可微优化

inner steps = 1/4/8，分别形成候选；outer steps = 16，outer Adam lr=.01。
优化相对位移 u，X'=X+norm(X)*u，每步逐行投影到半径 .1。
每次外层更新从头初始化 student，训练 seeds 7500/7501 等权；不继承上一次临时参数。
外层目标为 new InfoNCE + max(0, old InfoNCE - paired C0 old InfoNCE - .01) + .01*mean(||u_i||^2)。
old/new 各 4 个 batch，batch 16；图传播和 InfoNCE 运算顺序与 canonical 相同。

probe 是修改前旧 synthetic memory 与当前 raw delta 构成的固定合法 support。
旧行与当前行分别固定拆分 80% optimization / 20% validation；每部分至少两个节点。
validation seeds 7510/7511，与优化、报告 seeds 分离。
当前全 delta 已用于 C0/teacher 构造；probe 节点共享合法当前图上下文，因此这个拆分不是未见原始数据泛化集。
不保存历史 raw probe 到下一阶段，不维护额外 C0 历史分支作为正式算法必需 memory。

在 outer steps 4/8/12/16 的非零候选中，仅按 validation 外层目标选择最低者。
另做接受/回退：两个 validation seeds 均 new loss 至少改善 1e-4，且 old loss 不恶化超过 .01 才接受；否则回退该 cell 的 C0。
所有选择在读取本轮正式 F1 前落盘；1/4/8 三种 horizon 全部报告，不逐 cell 按 F1 挑最优。

## 配对臂

1. C0：不改 X。
2. teacher：原 teacher-input-gradient + 原 gate，使用同一个父 memory。
3. student_k1/k4/k8：穿过训练更新的非零候选，尚未做外层接受回退。
4. stop_k1/k4/k8：阻断 student 更新到 X 的路径，其余优化与正则相同；零初始化下应精确等于 C0。
5. random_k1/k4/k8：逐行随机方向，位移范数与对应 student 候选相同，不使用 report F1。
6. guarded_k1/k4/k8：上述无标签 validation gate，只能选择对应 student 或 C0，不新增 fit。

每 cell 14 个记录臂，8 cells x 14 x 6 = 672 条正式记录槽位。
相同 memory/seed/steps 的 C0、stop、guard 可复用；预期最多 336 个新增 fits。
复用须核对 memory、协议/源数据、学生代码及 checkpoint 哈希；计数和成本不把复用当作新训练。

## 验收与后续触发

先过：可微 Adam 与普通 Adam 的 CPU/GPU 对照；方向有限差分；非零 hypergradient；stop-gradient 零变化；半径、行、文本/拓扑与输入不可变测试。
报告：两阶段每 cell 的 final/old/new F1 配对差、seed 区间、teacher/probe gain、实际位移、gate 回退、构造/临时训练/正式训练成本。
独立重新加载正式 checkpoint 复算报告指标，核对所有决策、memory 和冻结字段。

后续递归 rollout 的研究触发：同一个 guarded horizon 在四个 s2 cells 平均 >=+.5 pp、至少3/4 >=-1 pp、平均 new >=0；s1 平均不负；s2 平均优于对应随机方向且不低于 teacher。
成本完整列账，不因高成本否认机制信号，也不能据此宣称高效；成本晋升需另做等算力和摊销评估。
若有多 horizon 通过，按 s2 均值选择一次，明确这是开发集选型；再冻结做真实递归与未见顺序/seed确认。
没有通过则结束本批诊断，不自动追加文本/拓扑，也不把单阶段结果拼成递归持续性能。

旧 R2-C/D 仍为条件后续。本文件及运行 manifest 保留变更记录，不以事后阈值改写结果。

## 执行审计补记

2026-09-22，正式矩阵仍运行中、汇总尚未计算时检查 runner：实现只复用了 C0，stop 和 guarded 仍各自重新训练。完整矩阵实际应为 624 个新增 fit、48 个引用 C0 记录，而不是上方预期的最多 336 个新增 fit。不覆盖原始记录，不将重复训练计作机制干预。

按上述原定“guard 不新增 fit”的配对定义，汇总同时保存两种口径：

- raw_executed：所有实际执行记录原样报告，包括相同 memory 的重复训练。
- paired_policy：stop 引用同 seed 的 C0；guard 依预先落盘的 accepted 决策引用对应 student 或 C0。必须验证 memory hash 相同，不能按 F1 决定引用来源。

晋升条件逐项在两种口径下计算，不隐藏原始分数；原定配对策略以 paired_policy 为准，重复训练差异单列数值稳定性诊断。阈值、候选和数据集不变。624 个 fit 的实际成本全部计入实验账；原运行未单独记录 teacher 耗时，因此不伪造完整构造成本或总加速比。
