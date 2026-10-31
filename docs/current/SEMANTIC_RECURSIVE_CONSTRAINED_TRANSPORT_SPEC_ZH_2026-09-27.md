# Semantic-Recursive 受约束传输：实现一致性规范

日期：2026-09-27。状态：依据当前代码建立的形式化规范；不代表新增算法或新增训练结果。

## 1. 冻结边界

保留现有 Semantic-Recursive 的 rows、来源分组、split/merge、Mean-X、文本、边、seed 和 Student。形式化的作用是准确描述已有方法，而不是用新的优化问题偷偷替换已测实现。C0 是内部构造控制与拓扑模板，不是公开强基线；主效果评价面向同权限、同预算的外部方法。

## 2. Payload、slot 与来源

一个实际被搬移的 payload 包含 `x`、`text_emb`、`texts`、`mass`、`depth`、`stage`；来源 ID 与类别记录随同搬移，保存在审计记录中。类别是构造分块信息，不是 Student 的监督分类损失。当前方法不应称为完全 label-free condensation。

slot 是新 block 中的一个固定行位置，对应既有拓扑端点及原始来源分组。slot 的语义锚来自当前 delta 的原分组均值；拓扑 degree 并不进入默认 semantic cost。不能将 role-aware 或 message-aware 代价写成当前主方法的组成部分。

## 3. 局部重组与两次匹配

`run_tgc_two_prototype_smoke.py::build_two_prototype_block` 对每个可行类别至多进行一次局部重组：

1. 类别至少有三个非空来源组，且至少一组有四个来源节点。
2. 在可拆组中选择联合表示平均平方离散度最高者；并列按原 slot 顺序处理。
3. 将其确定性聚类为两组；从其余组中选择联合中心距离最近的一对合并。这里没有额外的低离散度筛选，也没有保证合并代价全局最小。
4. 两个 split piece 加一个 merged piece，替换原三个组，保持新增 rows 不变。局部 3×3 assignment 先把三个 proposal 对齐到原三个 slot。
5. 由同一 piece 生成 Mean-X 与中心代表文本，拼接后重新编码。其它 payload 直接复制。

第二次匹配来自 `run_tgc_role_aware_assignment_20260925.py::complete_mapping`。它可能把这三个已修改 payload 分配到该类的其它 slot，随后把未修改 payload 补齐到剩余位置。因此“只改三个 payload 的内容”不等于“最终最多移动三个 payload”。

## 4. 与代码一致的语义代价

设 A_c 为类别 c 中被 split/merge 修改的 payload 下标，S_c 为该类全部新 slot。对 i∈A_c，使用其 source piece 的原始 X 均值与原始文本 embedding 均值；对 j∈S_c，使用原 slot 来源组的同类均值。把两个均值拼接后整体 L2 归一化，计算平方欧氏距离。

关键区别：assignment 使用的是来源文本 embedding 的均值，不是最终拼接文本的 re-encoded embedding；最终被搬移的 payload 仍包含后者。实现中的 joint source clustering 采用的归一化也应与这个匹配代价分开描述。

代价矩阵随后作全矩阵标准化：减去均值，除以 `std(unbiased=False).clamp_min(1e-6)`。在必须选择相同数量配对的前提下，这是正仿射变换，不改变精确算术下的最优解集合；不能由此保证不同数值实现对并列代价返回同一排列。

## 5. 实际优化问题与保证范围

对每类求解从 A_c 到 S_c 的最小代价单射，保证每个已修改 payload 恰好选择一个 slot，目标 slot 不重复。实现调用 SciPy 的 `linear_sum_assignment`。随后：

1. 对剩余来源和剩余目标的交集保留 identity；
2. 按既有顺序将残余来源与残余目标逐一配对；
3. 验证完整映射是一对一、同类且覆盖所有新行。

可陈述的最优性仅为：给定固定代价矩阵，对已修改 payload 的矩形 assignment 子问题求得最优解。残余补全没有优化其语义代价，因此不能声称整个类别的全部 payload assignment 达到全局最小；更不能推导 Student loss 或 F1 最优。Hungarian/线性 assignment 是已有工具，不单独计为算法创新。

## 6. 递归与不变量

- Budget：split/merge 与 assignment 均不改变该阶段新增 rows；累计预算仍随 fixed-rate 增长。
- Payload：以完整行及来源记录为单位，assignment 前后 multiset 守恒。此守恒只针对 placement 前后的 proposal，不是声称 split/merge 不改变原始 C0 内容。
- Class：所有搬移限于同类新 slot。
- Prefix：旧 payload 与上一阶段 Semantic-Recursive 输出逐值相同。
- Topology：当前实现复制同阶段 C0 `edge_index`；旧旧边应另外与 parent 核验。增加跨块边仍可能改变旧节点的归一化传播权重，不能把旧边不变写成旧传播算子不变。
- Access：构造数学依赖仅允许当前 delta 与持久 synthetic 状态。runner 会加载完整数据容器用于切片和评价，声明 `historical_raw_reads=0` 本身不是进程级访问隔离证明；须区分合法输入依赖审计与物理 I/O 隔离。
- Auxiliary state：若生产构造需要保留 C0 的影子 synthetic/topology 状态，也应计入存储和访问清单，不能只统计主 memory。

## 7. 保输出一致的封装验收

封装应只抽出已存在的同类映射补全及 payload 搬移；不改变代价计算、求解器、dtype、遍历顺序、RNG 或重组规则。

验收分层：

1. 单元级：空改动、矩形匹配、同代价、被占用 identity、多个类别；检查同类双射与原实现相同。
2. 已存产物回放：复用原 `mapping`，从保存的 memory 还原 proposal，比较完整 payload、来源、旧 prefix 和 topology 的内容 hash。该测试只能证明 placement 回放等价，不能替代重新构造 mapping 的端到端等价测试。
3. 构造级：只有原输入、依赖与编码器均可用时，比较同一 delta/parent 下的 mapping、payload 和 memory 内容。不得把缺失输入的测试标为通过。

哈希应基于 tensor dtype、shape、连续内容及文本等字段，而不是直接依赖 `torch.save` 容器字节；后者会受序列化元数据影响。原结果与训练产物保持不可覆盖。

## 8. 当前不做的算法替换

不把 full-class assignment、gain-based split/merge、attention、response selector、动态预算或旧 memory revision 混入形式化版本。它们是独立候选，需要另立对照和结果目录。正式是否保留 Semantic-Recursive，不以每个顺序都超过 C0 为门槛；同时仍完整报告相对外部基线的均值、方差、最差 cell 和失败条件。
