# 受限跨边传输：第三顺序确认协议

日期：2026-09-28。冻结对象：现有 `message_balanced`，不新增参数搜索。

## 1. 比较对象与范围

本轮比较 **Semantic-Recursive（SR）** 与 **SR + message_balanced 跨边替换**。
旧脚本的 `c0` arm 实际读取 `semantic_perm_memory.pt`，不能将其解释为原始 C0。
原始 C0 仍是 SR 的构造来源，但不是这张拓扑组件表中的直接比较对象。

第三顺序使用 `order19613`：Cora 为 `[2,5] -> [4,6] -> [0,1,3]`，
Citeseer 为 `[4,5] -> [1,2] -> [0,3]`。在查看该顺序结果前，核对仓库中
已使用顺序的类别分组，没有完整三阶段分组重合。它是新类别顺序确认，
不是新数据集、独立数据划分或新的 Student seed 确认。

## 2. 冻结条件

- 三阶段、5% 累计 fixed-rate；按实际 pool 大小计算每阶段 rows。
- C0 来源仍调用 R10 使用的 `build_c0`，不运行无关的 teacher/容量调整分支。
- 先用已有顺序逐值回放 C0，确认文本、X、mass 和 edge_index 一致。
- SR 构造保持 two-prototype、同类 semantic assignment 和递归 SR parent。
- support 分数仍为 0.75 一跳消息 cosine + 0.25 直接 X cosine。
- support cross-degree=1；旧 support 容量上限为 `ceil(N_delta / K_old)`。
- 初始 P 保持 top-r=8；X、文本、mass、rows、P 不更新。
- hard materializer 保持旧旧边、新新边，以及跨边两侧端点的度数。
- Student 为原两层 GCN、冻结 MiniLM、双向 InfoNCE；seeds=150..155。
- 构造仅依赖当前 delta 和旧 synthetic；完整容器加载不等于进程级访问隔离。
- 不依据本轮 report-dev F1 选择参数、方法、顺序或检查点。

## 3. 必须保留的实验边界

本轮仍为 **共享 SR parent 的逐阶段组件实验**。stage 2 不继承 stage 1
的 message_balanced 拓扑；不能声称已经完成递归跨边传输长链验证。
通过后仍须另做候选自身 parent 的递归实验。

这里的 message score 没有训练 Teacher，也没有对 X 或 P 求梯度。
它是给后续梯度联合更新选择稳定传播基础的拓扑实验，不能替代旧记忆更新实验。

## 4. 指标和验收

主指标是两个数据集各自 **stage 2 的累计 Macro-F1**，相对 SR 的差值。
补充 Final Accuracy、三阶段累计 F1/Accuracy 的真实过程平均、old/new F1、
最差阶段、4 个增量 stage-cell 的正收益比例、六个种子的配对差与条件置信区间。

不能把 stage 1 和 2 的累计分数均值称为 Final；也不能把同一检查点下的
`per_stage` 类别组平均称为训练过程的 stage-average。旧表中的标准差是
cell 差值的离散程度，不是 Student seed 标准差。

保存每个 Student 的检查点，并重载核对所有已报告指标。两方法 stage 0
完全相同，复用六个相同结果；总计 60 次唯一训练，不能虚报为 72 次独立训练。

冻结后续条件：两个数据集最终 F1 均正、增量 cell 最差差值不低于 -2 pp，
再进行 `message_balanced + boundary-old-X` / `message_balanced + new-only-X`
最小组合。未通过时保留全部分数，分析失败环节，不按本轮 F1 改参数重新称为确认。
此条件只决定该拓扑组件能否作为下一轮共同基底，不代表放弃梯度或旧记忆更新。

## 5. 产物

- 入口：`code/v2_llm_fusion/run_tgr_cross_edge_confirmation.py`
- 启动器：`code/v2_llm_fusion/launch_tgr_cross_edge_confirmation_20260928.sh`
- 远端输出：`results/tgr_cross_edge_confirmation_20260928/{cora,citeseer}/`
- 每数据集含 preregister、构造 seal、两阶段结构审计、三阶段评估和 reload audit。

结果待运行完成后另写报告；本协议不是实验通过声明。
