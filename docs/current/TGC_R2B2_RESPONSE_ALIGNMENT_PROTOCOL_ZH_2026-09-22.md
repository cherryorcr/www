# TGC-R2B2：固定 student 训练前缀的响应对齐

日期：2026-09-22。状态：本轮结果前冻结；原 R2B 代码与结果不覆盖。

## 目标与范围

只改变历史 X 元优化发生的训练阶段，回答早期 1/4 步响应是否因离正式训练太远而失真。本轮不加入混合 probe、新保护损失、文本或拓扑联合修订；它们是不同的后续干预，不捆绑调参。

数据仍为 Cora/Citeseer × canonical/order19101 × s1/s2；每阶段共享 C0 父 memory，只做单阶段因果诊断，不拼成递归 continual 成绩。沿用 5% growing memory、C0 block、文本、拓扑、protected/editable rows、半径 .1、outer16/lr.01、原 loss/gate/probe split 和全部 seeds。

报告 seeds 100--105；元优化 seeds 7500/7501，验证 seeds 7510/7511。已有条件均为暴露开发集，不作独立泛化声明。

## 唯一干预

warm-up w=0/64/256，suffix K=1/4，形成 3×2 的矩阵。

共同前缀在未修改的合法 C0 compact memory 上，按 canonical student、Adam、lr=.001、batch16、seed*7+1 的 minibatch generator 正常训练 w 步。保存参数、Adam 一二阶矩、步数和 generator state。每次候选更新都从同一前缀开始，仅后续 K 步的 memory 使用可修改 X。

外层目标仍为 new InfoNCE + old-loss hinge(.01 tolerance) + .01 位移正则。前缀断梯度，是 truncated surrogate，不是对修改后 memory 完整从头训练的精确梯度。最终报告 student 仍从头训练到 canonical 步数，不继承前缀参数。

同一 w 的四个临时 seeds 的前缀在 K1/K4、student/stop 和 C0 配对参考间复用。其构造时间单独列账，不能把复用后的边际时间当部署总成本。前缀仅为临时构造状态，不增加持久 memory。

## 控制与计数

每阶段记录 C0、teacher，以及每个(w,K)的 student、等逐行位移 random、stop、guarded，共 26 臂。

w=0 的 K1/K4 直接引用上一轮已审计的固定候选和正式 fits，不借机重选。teacher/editable rows 也固定引用，避免独立 teacher 的数值变化造成混淆。

stop 在新代码中确实执行断梯度优化并验证 X 完全不变，再引用 C0 fit；guarded 只按既有无标签接受规则引用对应 student 或 C0。相同 memory/seed/steps 结果核对完整 memory、checkpoint、共同代码/数据协议后复用，不重复训练。

8 个阶段条件 × 26 臂 × 6 seeds = 1248 个报告记录槽位；预计新增最多 384 个 fits：4 个新(w,K)× student/random × 8 条件 × 6 seeds。其余为可追溯引用，不计为新增训练。

候选、固定验证选择与接受决策必须全部落盘，再读取正式报告 fits。既有报告分数虽已知，不进入本轮程序的构造、优化或 gate。

## 前置数值门禁

1. CPU/GPU 上 w=0/64/256、K=1/4 的前缀+续训与不间断 Adam 等价，参数 atol2e-5、rtol2e-4。
2. w=0 续训实现与上一轮 unroll 逐位一致。
3. 双精度后缀 hypergradient 对齐两尺度有限差分，atol1e-6、rtol.001。
4. prefix params/moments/RNG、输入 memory/probe 不可变；stop 精确等于 C0；位移和固定字段契约通过。
5. 正式任务每个阶段先对真实 memory 做 Adam 续训对照，再构造新候选。

前置未通过不得启动正式 fits，不因报告分数好看放宽数值条件。

## 报告与后续

每(w,K)完整报告 s1/s2 的 cumulative/old/new F1、与 C0/匹配 random/同 K 的 w0 差值、配对六-seed区间、gate误收/误拒诊断、保护项激活、全部构造和正式训练成本。

复核所有引用、前缀、memory、不变字段和完整正式 checkpoint 的报告预测。汇总不能按 cell 挑最优后拼表。

沿用 R2B 的 guarded 研究触发条件：同一(w,K)在四个 s2 条件平均至少 +.5 pp；至少3/4 >=-1 pp；平均新类不下降；s1平均不下降；s2优于匹配随机且不低于teacher。另外必须展示相对同K的w0差异，否则不能把旧候选效果归因于warm-up。

通过也只是开发触发，不直接晋升论文主方法；需要之后冻结做递归rollout和独立确认。本轮结束后依据证据安排混合旧新probe/保护尺度实验，不自动上大图，不事后调整本轮门槛。
