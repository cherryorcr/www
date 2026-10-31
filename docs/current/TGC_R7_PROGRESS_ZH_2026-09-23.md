# TGC-R7 执行进展

状态：R7 四批实验全部完成；本文件保留中途进展，最终结论见 [R7结项报告](TGC_R7_COMPLETION_AND_NEXT_DECISION_ZH_2026-09-23.md)。

## 已完成的构造稳定性诊断

四个开发条件：Cora/Citeseer × order19341/19342，s2固定R5封存s1父memory，6个共同student seeds。

| 方案 | 最终平均Macro-F1 | 说明 |
|---|---:|---|
| C0封存 | 54.4267 | 重新训练精确复现 |
| R5递归封存 | 55.6312 | 重新训练精确复现，+1.2045 pp仍有效 |
| 三次同seed普通重构均值 | 54.3828 | 并非三种独立算法；不能挑最好一次 |
| 三次仅开确定性开关重构均值 | 54.3478 | 开关未消除构造非确定性 |
| 稠密数值构造 | 54.7239 | 三次teacher/H/P/memory全部一致；共同student未更换 |

原稀疏路径在初始化/minibatch/邻接相同时，teacher第1次更新即出现不同；仅开PyTorch确定性开关仍有此问题。CPU归一化、GPU稠密teacher与稠密P目标路径消除了本批重复差异。该实验定位到构造数值路径，未单独证明具体CUDA内核责任，也未证明所有机器完全确定。

稳定性首批240个结果槽位、240次实际student训练；稠密校准192槽位、144次实际训练（两个完全相同稠密重复按memory哈希复用），全部封存与checkpoint重载审计通过。两批封存C0/R5共96对历史student分数比较，最大差0。

## 已完成的目标校准

同一父memory、统一共享分配，alpha=0/0.25/0.5/1四个候选；没有部署选择器。按每个条件内部四候选排序检验诊断指标，四个条件的Spearman均值和top1命中如下：

| 指标（方向预定） | 平均Spearman | 最优候选命中 |
|---|---:|---:|
| 实际teacher重构误差越低越好 | +0.20 | 2/4 |
| 图文关系KL越低越好 | -0.20 | 1/4 |
| 来源/合成memory梯度余弦越高越好 | 0.00 | 1/4 |
| 一次fresh Adam后合法数据InfoNCE改善越高越好 | -0.50 | 1/4 |

每格只有四个候选，相关系数仅用于本轮校准，不做显著性或一般性推断。结果不足以支持加入响应损失或恢复短训selector。一次更新是固定teacher检查点处重置Adam状态的一步，不是恢复teacher优化器或完整student轨迹。

固定父memory的alpha=0/.25/.5/1平均F1分别53.7046、54.0520、53.4339、54.7239；alpha=.5没有局部正收益。仍按评分前预定值做自身历史递归对照，不改选最好的alpha。

## 当前后续

递归开发已比较C0、原稀疏R5、稠密R5、alpha=.5候选；有界候选最终 +0.5603 pp，但阶段平均 -0.2132 pp。独立确认随后在order19605/19606/19608上完成：原始、稠密、有界TGC-R最终分别为 +1.3201/+1.4265/+1.0553 pp，但new F1分别为 -1.1404/-0.8243/-2.0593 pp。结论是保留TGC-R研究主线，下一步转向C0-anchored new-class plasticity，不继续调response proxy。

原协议：[TGC_R7_EXECUTION_PROTOCOL_ZH_2026-09-23.md](TGC_R7_EXECUTION_PROTOCOL_ZH_2026-09-23.md)。已完成产物：[稳定性summary](../../results/tgc_r7_stability_20260923/summary.json)、[校准summary](../../results/tgc_r7_calibration_20260923/summary.json)。
